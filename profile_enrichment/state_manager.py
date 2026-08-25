import json
from pathlib import Path
from typing import Any, Dict, Optional

from profile_enrichment.config import CONFIG


class StateManager:
    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path or CONFIG["state_file"])
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

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

    def save(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, sort_keys=True)

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
        if profile_id not in self.processed:
            self.processed.append(profile_id)
        self.save()

    def mark_skipped(self, profile_id: str) -> None:
        if profile_id not in self.skipped:
            self.skipped.append(profile_id)
        self.save()

    def mark_failed(self, profile_id: str) -> None:
        if profile_id not in self.failed:
            self.failed.append(profile_id)
        self.save()

    def is_processed(self, profile_id: str) -> bool:
        return profile_id in self.processed

    def set_last_run(self, timestamp: str) -> None:
        self._state["last_run"] = timestamp
        self.save()
