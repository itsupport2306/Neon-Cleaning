"""Incremental, resumable CSV outputs. One row is appended, flushed and fsynced per profile."""

from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set

DISCREPANCY_FILE = "resume_validation_discrepancies.csv"
NO_DISCREPANCY_FILE = "resume_validation_no_discrepancies.csv"
ERROR_FILE = "resume_validation_errors.csv"

DISCREPANCY_COLUMNS = [
    "profile_id", "resume_url", "first_name_db", "first_name_resume", "last_name_db", "last_name_resume",
    "specialty_db", "specialty_resume", "address_db", "address_resume", "mismatched_fields",
    "discrepancy_reason", "model_confidence", "review_status", "validation_status", "field_statuses",
    "validated_at",
]
NO_DISCREPANCY_COLUMNS = [
    "profile_id", "resume_url", "validation_status", "review_reason", "field_statuses", "model_confidence",
    "validated_at",
]
ERROR_COLUMNS = [
    "profile_id", "resume_url", "validation_status", "error_category", "error_detail", "retryable",
    "validated_at",
]

COLUMNS = {DISCREPANCY_FILE: DISCREPANCY_COLUMNS, NO_DISCREPANCY_FILE: NO_DISCREPANCY_COLUMNS, ERROR_FILE: ERROR_COLUMNS}


def read_ids(path: Path) -> List[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [row["profile_id"] for row in csv.DictReader(handle) if row.get("profile_id")]


class ResultStore:
    """The three CSV files plus the set of profile_ids already recorded in any of them.

    `record_*` returns False (and writes nothing) for an id that is already recorded, so a resumed run, or a
    crash between two writes, can never create a duplicate row.
    """

    def __init__(self, out_dir: Path | str, retry_errors: bool = False, redo_discrepancies: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if retry_errors:
            self._drop_error_rows()
        if redo_discrepancies:
            self._archive(DISCREPANCY_FILE, "discrepancies")
        self._done: Set[str] = set()
        for name in COLUMNS:
            self._done.update(read_ids(self.dir / name))
        for name, columns in COLUMNS.items():
            self._ensure_header(self.dir / name, columns)

    @staticmethod
    def _ensure_header(path: Path, columns: List[str]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8-sig") as handle:  # BOM: Excel opens it as UTF-8
                csv.writer(handle).writerow(columns)

    def _drop_error_rows(self) -> None:
        """--retry-errors: archive the error file and start a new one, so those profiles are validated again."""
        self._archive(ERROR_FILE, "errors")

    def _archive(self, filename: str, label: str) -> None:
        path = self.dir / filename
        if not path.exists() or path.stat().st_size == 0:
            return
        os.replace(path, path.with_name(f"resume_validation_{label}.{int(time.time())}.bak.csv"))

    @property
    def done(self) -> Set[str]:
        return self._done

    def is_done(self, profile_id: str) -> bool:
        with self._lock:
            return profile_id in self._done

    def _append(self, filename: str, row: Dict[str, object]) -> bool:
        profile_id = str(row["profile_id"])
        with self._lock:
            if profile_id in self._done:
                return False
            columns = COLUMNS[filename]
            with (self.dir / filename).open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore").writerow(
                    {c: ("" if row.get(c) is None else row.get(c)) for c in columns})
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:  # some network mounts do not support fsync; flush() above is still done
                    pass
            self._done.add(profile_id)
            return True

    def record_discrepancy(self, row: Dict[str, object]) -> bool:
        return self._append(DISCREPANCY_FILE, row)

    def record_no_discrepancy(self, row: Dict[str, object]) -> bool:
        return self._append(NO_DISCREPANCY_FILE, row)

    def record_error(self, row: Dict[str, object]) -> bool:
        return self._append(ERROR_FILE, row)

    def counts(self) -> Dict[str, int]:
        return {name: len(read_ids(self.dir / name)) for name in COLUMNS}


def duplicates(paths: Iterable[Path]) -> Set[str]:
    seen: Set[str] = set()
    dup: Set[str] = set()
    for path in paths:
        for pid in read_ids(path):
            (dup if pid in seen else seen).add(pid)
    return dup
