"""Optional write-back of the discrepancies found by validate_resumes (``run --apply-to-neon``).

The resume is treated as the source of truth: for every field the validator reported as a mismatch, the stored value
is replaced with what the resume says. There is no review step. What stays in place:

* only the mismatched fields of that one profile are touched (first_name, last_name, specialty, city, state_code,
  zip_code), through one UPDATE per profile_id with an exact row-count check;
* lost-update guard: a column is only changed while it still holds the value that was read, so a change made by
  someone else in the meantime is never overwritten;
* a specialty is written only as a label of the shared specialty vocabulary (free text from a resume would pollute
  the column); names and cities pass the same shape checks the NULL-specialty pipeline uses;
* every intended change is appended (and fsynced) to ``resume_validation_neon_updates.csv`` with its old value
  *before* the UPDATE runs, so ``revert-updates`` can put the old values back.
"""

from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from profile_enrichment import neon_db, validation
from resume_collab import compare as cmp

AUDIT_FILE = "resume_validation_neon_updates.csv"
AUDIT_COLUMNS = ["profile_id", "field", "old_value", "new_value", "result", "detail", "updated_at"]
WRITABLE = ("first_name", "last_name", "specialty", "city", "state_code", "zip_code")
LIMITS = {"first_name": 100, "last_name": 100, "specialty": 100, "city": 120, "state_code": 2, "zip_code": 10}

Change = Tuple[Optional[str], str]  # (value read from the database, value the resume says)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _tidy_name(value: str) -> str:
    value = " ".join(value.split())
    return value.title() if value.isupper() or value.islower() else value


def build_corrections(
    profile: Dict[str, Any], results: Dict[str, "cmp.FieldResult"], first_name: str, last_name: str,
    address: Dict[str, str], specialties: Sequence[str], profession: str, lexicon,
) -> Tuple[Dict[str, Change], List[Tuple[str, str, str]]]:
    """(changes to write, [(field, resume value, reason it was not written)]) for the mismatched fields."""
    changes: Dict[str, Change] = {}
    skipped: List[Tuple[str, str, str]] = []

    def propose(column: str, new: str, shape_problem: Optional[str] = None) -> None:
        new = (new or "").strip()
        old = profile.get(column)
        if not new:
            skipped.append((column, "", "resume value empty"))
        elif shape_problem:
            skipped.append((column, new, shape_problem))
        elif len(new) > LIMITS[column]:
            skipped.append((column, new, "exceeds_column_length"))
        elif (old or "") != new:
            changes[column] = (old, new)

    for column, resume_value in (("first_name", first_name), ("last_name", last_name)):
        if results[column].status == cmp.MISMATCH:
            name = _tidy_name(resume_value or "")
            propose(column, name, validation._check_name(name) if name else None)

    if results["specialty"].status == cmp.MISMATCH:
        label = None
        for candidate in [*specialties, profession]:
            mapped = lexicon.normalize_label(candidate) if (lexicon and candidate) else None
            if mapped:
                label = mapped[0]
                break
        if label:
            propose("specialty", label)
        else:
            skipped.append(("specialty", "; ".join(specialties) or profession or "",
                            "resume specialty is not a label of the specialty vocabulary"))

    if results["address"].status == cmp.MISMATCH:
        stored_city, resume_city = cmp.fold(profile.get("city")), cmp.fold(address.get("city"))
        if stored_city and resume_city and stored_city != resume_city:
            city = " ".join((address.get("city") or "").split())
            propose("city", city.title() if city.isupper() or city.islower() else city, validation._check_city(city))
        stored_state, resume_state = cmp.state_code(profile.get("state_code")), cmp.state_code(address.get("state"))
        if stored_state and resume_state and stored_state != resume_state:
            propose("state_code", resume_state)
        stored_zip, resume_zip = cmp.zip5(profile.get("zip_code")), cmp.zip5(address.get("zip"))
        if stored_zip and resume_zip and stored_zip != resume_zip:
            propose("zip_code", resume_zip)
    return changes, skipped


class UpdateAudit:
    """Append-only, fsynced record of every intended and attempted change."""

    def __init__(self, out_dir: Path | str):
        self.path = Path(out_dir) / AUDIT_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
                csv.writer(handle).writerow(AUDIT_COLUMNS)

    def add(self, profile_id: str, field: str, old: Any, new: Any, result: str, detail: str = "") -> None:
        row = [profile_id, field, "" if old is None else old, new, result, detail, _now()]
        with self._lock, self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass


def apply_corrections(profile_id: str, changes: Dict[str, Change], audit: UpdateAudit) -> Dict[str, str]:
    """Write `changes` to one profile. Returns {field: 'applied' | 'skipped_changed_since_read' | 'skipped_row_missing'}."""
    if not changes:
        return {}
    unknown = [name for name in changes if name not in WRITABLE]
    if unknown:
        raise ValueError(f"refusing to write non-profile columns: {sorted(unknown)}")
    for name, (old, new) in changes.items():
        audit.add(profile_id, name, old, new, "intended")  # on disk before the database is touched

    assignments, params = [], []
    for name, (old, new) in changes.items():
        column = f'"{name}"'
        assignments.append(f"{column} = CASE WHEN {column} IS NOT DISTINCT FROM %s THEN %s ELSE {column} END")
        params.extend([old, new])
    returning = ", ".join(f'"{name}"' for name in changes)
    sql = (f'UPDATE {neon_db._table_sql()} SET {", ".join(assignments)} WHERE "profile_id" = %s '
           f"RETURNING {returning}")
    params.append(profile_id)

    def run() -> Optional[Dict[str, Any]]:
        with neon_db.transaction() as conn, conn.cursor() as cursor:
            cursor.execute(sql, params)
            if cursor.rowcount == 0:
                return None
            if cursor.rowcount != 1:
                raise RuntimeError(f"UPDATE for profile_id={profile_id!r} touched {cursor.rowcount} rows (expected 1)")
            return neon_db._rows(cursor)[0]

    try:
        after = neon_db._retry_connection_errors(run)
    except Exception as exc:  # noqa: BLE001 - one failed profile must not stop the run
        for name, (old, new) in changes.items():
            audit.add(profile_id, name, old, new, "failed", f"{type(exc).__name__}: {exc}"[:300])
        return {name: "failed" for name in changes}

    outcome: Dict[str, str] = {}
    for name, (old, new) in changes.items():
        if after is None:
            outcome[name] = "skipped_row_missing"
        elif after.get(name) == new:
            outcome[name] = "applied"
        else:
            outcome[name] = "skipped_changed_since_read"
        audit.add(profile_id, name, old, new, outcome[name],
                  "" if outcome[name] == "applied" else f"database now holds {None if after is None else after.get(name)!r}")
    return outcome


def revert_updates(audit_path: Path | str, apply: bool) -> Tuple[int, int, int]:
    """Put the old values back where the column still holds exactly what we wrote. Returns (reverted, skipped, total)."""
    reverted = skipped = total = 0
    seen = set()
    with Path(audit_path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["result"] == "intended"]
    for row in rows:
        key = (row["profile_id"], row["field"], row["new_value"])
        if key in seen or row["field"] not in WRITABLE:
            continue
        seen.add(key)
        total += 1
        if not apply:
            print(f"WOULD REVERT {row['profile_id']}.{row['field']}: {row['new_value']!r} -> {row['old_value']!r}")
            continue
        column = f'"{row["field"]}"'
        sql = (f'UPDATE {neon_db._table_sql()} SET {column} = %s WHERE "profile_id" = %s '
               f"AND {column} IS NOT DISTINCT FROM %s")
        old = row["old_value"] or None

        def run(sql=sql, old=old, row=row) -> bool:
            with neon_db.transaction() as conn, conn.cursor() as cursor:
                cursor.execute(sql, (old, row["profile_id"], row["new_value"]))
                return cursor.rowcount == 1

        if neon_db._retry_connection_errors(run):
            reverted += 1
        else:
            skipped += 1
    return reverted, skipped, total
