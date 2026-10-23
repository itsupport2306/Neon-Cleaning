"""Primary-profile selection: which record in a cluster keeps its profile_id."""
from __future__ import annotations

from typing import Any, Dict, List

from profile_dedup.normalize import is_empty_value


def score_profile(row: Dict[str, Any]) -> float:
    score = 0.0
    if not is_empty_value(row.get("npi_number"), kind="text"):
        score += 1000
    if not is_empty_value(row.get("email"), kind="text") and (
        row.get("contact_updated_by_user_id") or row.get("contact_updated_by_email")
    ):
        score += 500
    if not is_empty_value(row.get("phone"), kind="text") and row.get("contact_updated_at"):
        score += 300
    if not is_empty_value(row.get("resume_url"), kind="text"):
        score += 200
    if not is_empty_value(row.get("license_number"), kind="text"):
        score += 150
    if not is_empty_value(row.get("profile_photo_url"), kind="text"):
        score += 50
    completion = row.get("completion_score")
    if completion is not None:
        score += float(completion)
    if row.get("is_listable"):
        score += 10
    return score


def select_primary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the survivor. Ties broken by most-recently-updated, then oldest
    account (first created), then profile_id for full determinism."""

    def sort_key(row: Dict[str, Any]):
        return (
            -score_profile(row),
            -(row.get("updated_at").timestamp() if row.get("updated_at") else 0),
            (row.get("created_at").timestamp() if row.get("created_at") else 0),
            row.get("profile_id") or "",
        )

    return sorted(rows, key=sort_key)[0]
