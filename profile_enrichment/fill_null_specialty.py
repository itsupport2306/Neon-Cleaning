"""Fill in every profile whose `specialty` is NULL, with no date window and no row limit.

Run:
    python -m profile_enrichment.fill_null_specialty

The full list of matching `profile_id`s is read once up front and becomes the
manifest for the run. Each ID ends in exactly one terminal status:

* ``resolved``   - specialty is no longer NULL (other verified columns may have changed too)
* ``unresolved`` - the resume was processed but no verified specialty could be found; reason logged
* ``failed``     - download / OCR / database error; reason logged

``resolved + unresolved + failed`` always equals the number of IDs identified, and the
run exits non-zero if they do not. Per-ID status is saved to a run-state file, so an
interrupted run resumes without redoing finished IDs.

Each profile goes through the same Qwen pipeline as ``processor --reprocess-all``
(download resume -> native text/OCR -> Qwen -> validation -> compare with the stored
row): a stored value the resume confirms is never overwritten, and a value that is
not in the resume is never written.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from profile_enrichment import processor
from profile_enrichment.config import CONFIG
from profile_enrichment.extractors import consume_ai_call_failure, run_ai_extraction
from profile_enrichment.neon_db import PROFILE_COLUMNS, fetch_profile_by_id, get_connection
from profile_enrichment.state_manager import StateManager

logger = processor.logger

RESOLVED = "resolved"
UNRESOLVED = "unresolved"
FAILED = "failed"
TERMINAL_STATUSES = (RESOLVED, UNRESOLVED, FAILED)

_PROFILE_ID_IN_MESSAGE = re.compile(r"profile_id=(\S+)")


class _LastProblemCapture(logging.Handler):
    """Remember the latest WARNING/ERROR message per profile so a skip or failure can say why."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self._lock = threading.Lock()
        self._messages: Dict[str, str] = {}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = _PROFILE_ID_IN_MESSAGE.search(message)
        if match:
            with self._lock:
                self._messages[match.group(1)] = message

    def pop(self, profile_id: str) -> str:
        with self._lock:
            return self._messages.pop(profile_id, "")


class RunLedger:
    """Thread-safe per-ID status, persisted so a rerun skips IDs that already finished."""

    def __init__(self, path: Path, fresh: bool):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        if not fresh and path.exists():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8")).get("entries", {})
            except (json.JSONDecodeError, OSError):
                logger.warning("NULL_SPECIALTY_LEDGER_UNREADABLE | path=%s | starting_fresh=true", path)
        self._since_save = 0

    def status_of(self, profile_id: str) -> str | None:
        with self._lock:
            return self._entries.get(profile_id, {}).get("status")

    def record(self, profile_id: str, status: str, reason: str = "", updated_fields: List[str] | None = None) -> None:
        with self._lock:
            self._entries[profile_id] = {
                "status": status,
                "reason": reason,
                "updated_fields": updated_fields or [],
                "at": datetime.now(timezone.utc).isoformat(),
            }
            self._since_save += 1
            if self._since_save >= CONFIG["state_save_every"]:
                self._save_locked()

    def entries_for(self, profile_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {pid: self._entries[pid] for pid in profile_ids if pid in self._entries}

    def flush(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        self.path.write_text(json.dumps({"entries": self._entries}, indent=2, sort_keys=True), encoding="utf-8")
        self._since_save = 0


def fetch_null_specialty_ids() -> List[str | None]:
    """`SELECT * FROM profiles WHERE specialty IS NULL`, reduced to the key column."""
    table = CONFIG["neon_table_name"]
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT profile_id FROM {table} WHERE specialty IS NULL ORDER BY profile_id;")
            return [processor._normalize_candidate(row[0]) or None for row in cursor.fetchall()]


def process_one(
    profile_id: str,
    state: StateManager,
    ledger: RunLedger,
    capture: _LastProblemCapture,
    disable_ocr: bool,
) -> str:
    """Process one ID and always record a terminal status, whatever happens."""
    try:
        before = fetch_profile_by_id(profile_id)
        if not before:
            ledger.record(profile_id, FAILED, "profile row no longer exists")
            return FAILED

        outcome = processor.process_profile_record(
            before,
            state,
            existing=before,
            comprehensive_reprocess=True,
            disable_ocr=disable_ocr,
        )
        problem = capture.pop(profile_id)
        qwen_error = consume_ai_call_failure()
        if qwen_error:
            # Qwen never answered, so "no specialty found" would be a false conclusion.
            # Mark it failed so a rerun retries it instead of treating it as finished.
            reason = f"Qwen call failed: {qwen_error}"
            ledger.record(profile_id, FAILED, reason)
            logger.error("NULL_SPECIALTY_QWEN_FAILED | profile_id=%s | reason=%s", profile_id, reason)
            return FAILED
        if outcome in ("failed", "invalid"):
            ledger.record(profile_id, FAILED, problem or f"processor returned {outcome}")
            return FAILED

        after = fetch_profile_by_id(profile_id) or {}
        changed = [
            field
            for field in PROFILE_COLUMNS
            if processor._normalize_candidate(before.get(field)) != processor._normalize_candidate(after.get(field))
        ]
        if processor._normalize_candidate(after.get("specialty")):
            ledger.record(profile_id, RESOLVED, updated_fields=changed)
            logger.info("NULL_SPECIALTY_RESOLVED | profile_id=%s | specialty=%s | updated_fields=%s", profile_id, after.get("specialty"), changed)
            return RESOLVED

        # `problem` is the last unrelated warning (e.g. a rejected state), so it is not the reason.
        reason = "Qwen and regex found no specialty that is verifiably present in the resume"
        ledger.record(profile_id, UNRESOLVED, reason, changed)
        logger.warning("NULL_SPECIALTY_UNRESOLVED | profile_id=%s | reason=%s | updated_fields=%s", profile_id, reason, changed)
        return UNRESOLVED
    except Exception as exc:  # pragma: no cover - runtime safety net so no ID is ever dropped
        logger.exception("NULL_SPECIALTY_FAILED | profile_id=%s | error=%s", profile_id, exc)
        ledger.record(profile_id, FAILED, f"{type(exc).__name__}: {exc}")
        return FAILED


def build_report(ids: List[str | None], ledger: RunLedger, missing_id_rows: int) -> Dict[str, Any]:
    real_ids = [pid for pid in ids if pid]
    entries = ledger.entries_for(real_ids)
    counts = {status: 0 for status in TERMINAL_STATUSES}
    for entry in entries.values():
        counts[entry["status"]] += 1
    counts[FAILED] += missing_id_rows
    not_attempted = [pid for pid in real_ids if pid not in entries]

    return {
        "total_identified": len(ids),
        "processed": counts[RESOLVED] + counts[UNRESOLVED],
        "resolved_specialty_filled": counts[RESOLVED],
        "rows_with_any_column_updated": sum(1 for e in entries.values() if e["updated_fields"]),
        "unresolved": counts[UNRESOLVED],
        "failed": counts[FAILED],
        "rows_without_profile_id": missing_id_rows,
        "not_attempted": len(not_attempted),
        "accounted_for": counts[RESOLVED] + counts[UNRESOLVED] + counts[FAILED] == len(ids),
        "not_attempted_ids": not_attempted,
        "unresolved_and_failed": {
            pid: {"status": e["status"], "reason": e["reason"]}
            for pid, e in entries.items()
            if e["status"] in (UNRESOLVED, FAILED)
        },
    }


def write_report_files(report: Dict[str, Any], log_dir: Path) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = log_dir / f"null_specialty_report_{stamp}.json"
    csv_path = log_dir / f"null_specialty_unresolved_{stamp}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile_id", "status", "reason"])
        for pid, item in report["unresolved_and_failed"].items():
            writer.writerow([pid, item["status"], item["reason"]])
    return json_path, csv_path


def print_report(report: Dict[str, Any], json_path: Path, csv_path: Path) -> None:
    print("\n=== NULL specialty run report ===")
    print(f"Total records identified : {report['total_identified']}")
    print(f"Processed                : {report['processed']}")
    print(f"  specialty filled       : {report['resolved_specialty_filled']}")
    print(f"  any column updated     : {report['rows_with_any_column_updated']}")
    print(f"Unresolved (logged)      : {report['unresolved']}")
    print(f"Failed                   : {report['failed']}")
    print(f"Not attempted            : {report['not_attempted']}")
    print(f"All IDs accounted for    : {report['accounted_for']}")
    print(f"Report JSON              : {json_path}")
    print(f"Unresolved/failed CSV    : {csv_path}")


def check_qwen_reachable() -> None:
    """Stop before touching 100k+ rows if the Qwen endpoint is down."""
    if not CONFIG["ai_extractor_enabled"]:
        raise SystemExit("AI_EXTRACTOR_ENABLED is false; this run needs Qwen.")
    run_ai_extraction(None, "Jane Smith, MD. Cardiology. Dallas, TX.")
    error = consume_ai_call_failure()
    if error:
        raise SystemExit(f"Qwen is not reachable at {CONFIG['local_model_base_url']}: {error}")


def run(workers: int, disable_ocr: bool, fresh: bool, retry_failed: bool) -> Dict[str, Any]:
    check_qwen_reachable()
    ids = fetch_null_specialty_ids()
    missing_id_rows = sum(1 for pid in ids if pid is None)
    real_ids = [pid for pid in ids if pid]
    logger.info("NULL_SPECIALTY_CANDIDATES_FOUND | count=%s | rows_without_profile_id=%s", len(ids), missing_id_rows)

    # Every ID is re-evaluated through Qwen; the processed-state skip must not hide any of them.
    CONFIG["skip_already_processed"] = False

    root = Path(CONFIG["state_file"]).parent
    ledger = RunLedger(root / "null_specialty_run.json", fresh=fresh)
    state = StateManager(str(root / "null_specialty_state.json"))
    capture = _LastProblemCapture()
    logger.addHandler(capture)

    def already_done(pid: str) -> bool:
        status = ledger.status_of(pid)
        return status in (RESOLVED, UNRESOLVED) or (status == FAILED and not retry_failed)

    todo = [pid for pid in real_ids if not already_done(pid)]
    if len(todo) != len(real_ids):
        logger.info("NULL_SPECIALTY_RESUME | already_finished=%s | remaining=%s", len(real_ids) - len(todo), len(todo))

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(process_one, pid, state, ledger, capture, disable_ocr): pid for pid in todo}
            for future in as_completed(futures):
                done += 1
                future.result()
                if done % 25 == 0 or done == len(todo):
                    logger.info("NULL_SPECIALTY_PROGRESS | done=%s | total_this_run=%s", done, len(todo))
    finally:
        logger.removeHandler(capture)
        ledger.flush()
        state.flush()

    return build_report(ids, ledger, missing_id_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill every profile with specialty IS NULL using Qwen.")
    parser.add_argument("--workers", type=int, default=CONFIG["worker_count"], help="Concurrent workers (a single local Qwen server usually serves 1-2 well).")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR; use native text and Qwen only.")
    parser.add_argument("--fresh", action="store_true", help="Ignore the saved run ledger and process every ID again.")
    parser.add_argument("--no-retry-failed", action="store_true", help="On resume, do not retry IDs that previously failed.")
    args = parser.parse_args()

    report = run(
        workers=args.workers,
        disable_ocr=args.no_ocr,
        fresh=args.fresh,
        retry_failed=not args.no_retry_failed,
    )
    json_path, csv_path = write_report_files(report, Path(CONFIG["log_dir"]))
    print_report(report, json_path, csv_path)
    return 0 if report["accounted_for"] and not report["not_attempted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
