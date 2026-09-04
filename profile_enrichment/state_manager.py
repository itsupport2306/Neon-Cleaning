import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from profile_enrichment.config import CONFIG


class StateManager:
    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path or CONFIG["state_file"])
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load()
        self._save_every = CONFIG["state_save_every"]
        self._dirty_changes = 0
        self._processed_lookup = set(self.processed)
        self._skipped_lookup = set(self.skipped)
        self._failed_lookup = set(self.failed)

    def _load(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"processed": [], "skipped": [], "failed": [], "last_run": None}

        try:
            with self.state_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError):
            pass

        return {"processed": [], "skipped": [], "failed": [], "last_run": None}

    def save(self, force: bool = False) -> None:
        with self._lock:
            if not force and self._dirty_changes < self._save_every:
                return
            with self.state_path.open("w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, sort_keys=True)
            self._dirty_changes = 0

    @property
    def processed(self) -> list[str]:
        return self._state.setdefault("processed", [])

    @property
    def skipped(self) -> list[str]:
        return self._state.setdefault("skipped", [])

    @property
    def failed(self) -> list[str]:
        return self._state.setdefault("failed", [])

    def mark_processed(self, profile_id: str) -> None:
        with self._lock:
            if profile_id not in self._processed_lookup:
                self.processed.append(profile_id)
                self._processed_lookup.add(profile_id)
                self._dirty_changes += 1
            self.save()

    def mark_skipped(self, profile_id: str) -> None:
        with self._lock:
            if profile_id not in self._skipped_lookup:
                self.skipped.append(profile_id)
                self._skipped_lookup.add(profile_id)
                self._dirty_changes += 1
            self.save()

    def mark_failed(self, profile_id: str) -> None:
        with self._lock:
            if profile_id not in self._failed_lookup:
                self.failed.append(profile_id)
                self._failed_lookup.add(profile_id)
                self._dirty_changes += 1
            self.save()

    def is_processed(self, profile_id: str) -> bool:
        with self._lock:
            return profile_id in self._processed_lookup

    def set_last_run(self, timestamp: str) -> None:
        with self._lock:
            self._state["last_run"] = timestamp
            self._dirty_changes += 1
            self.save()

    def flush(self) -> None:
        self.save(force=True)
