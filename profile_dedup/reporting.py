from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from profile_dedup.config import DEDUP_CONFIG


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class DedupReport:
    """Accumulates one row per (primary, duplicate) pair plus run-level
    counters, then writes both a JSON and a flat CSV report."""

    def __init__(self, run_id: str, apply_mode: bool):
        self.run_id = run_id
        self.apply_mode = apply_mode
        self.started_at = datetime.utcnow()
        self.pairs: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, Any]] = []
        self.summary = {
            "certain": 0,
            "high": 0,
            "medium": 0,
            "merged_soft": 0,
            "deleted": 0,
        }

    def add_pair(
        self,
        *,
        primary_profile_id: str,
        duplicate_profile_id: str,
        confidence: str,
        signal: str,
        decisions: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        action: str,
        ai_opinion: Dict[str, Any] | None = None,
    ) -> None:
        self.pairs.append(
            {
                "primary_profile_id": primary_profile_id,
                "duplicate_profile_id": duplicate_profile_id,
                "match_confidence": confidence,
                "match_signal": signal,
                "field_decisions": decisions,
                "conflicts": conflicts,
                "action": action,
                "ai_opinion": ai_opinion,
            }
        )
        self.summary[confidence.lower()] = self.summary.get(confidence.lower(), 0) + 1
        if action == "soft_merge":
            self.summary["merged_soft"] += 1
        elif action == "deleted":
            self.summary["deleted"] += 1

    def add_error(self, context: Dict[str, Any]) -> None:
        self.errors.append(context)

    def add_skipped(self, context: Dict[str, Any]) -> None:
        self.skipped.append(context)

    def write(self) -> Dict[str, str]:
        report_dir = Path(DEDUP_CONFIG["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        mode = "apply" if self.apply_mode else "dryrun"
        json_path = report_dir / f"dedup_{mode}_{self.run_id}_{stamp}.json"
        csv_path = report_dir / f"dedup_{mode}_{self.run_id}_{stamp}.csv"

        payload = {
            "run_id": self.run_id,
            "apply_mode": self.apply_mode,
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "summary": self.summary,
            "pair_count": len(self.pairs),
            "error_count": len(self.errors),
            "skipped_count": len(self.skipped),
            "pairs": self.pairs,
            "errors": self.errors,
            "skipped": self.skipped,
        }
        json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "primary_profile_id",
                    "duplicate_profile_id",
                    "match_confidence",
                    "match_signal",
                    "action",
                    "conflict_count",
                    "conflict_fields",
                ]
            )
            for pair in self.pairs:
                conflict_fields = ";".join(c["field"] for c in pair["conflicts"])
                writer.writerow(
                    [
                        pair["primary_profile_id"],
                        pair["duplicate_profile_id"],
                        pair["match_confidence"],
                        pair["match_signal"],
                        pair["action"],
                        len(pair["conflicts"]),
                        conflict_fields,
                    ]
                )

        return {"json": str(json_path), "csv": str(csv_path)}
