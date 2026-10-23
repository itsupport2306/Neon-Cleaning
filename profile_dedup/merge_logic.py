"""Field-by-field merge rules.

Each profile column gets a rule describing: how "empty" is detected for that
column, and which source-priority strategy picks the surviving value when
both the primary and a duplicate have a non-empty value that differs.

Columns that are system/bookkeeping fields (profile_id, merged_into,
created_at, ...) are intentionally left out of MERGE_RULES - they are never
touched by a merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from profile_dedup.normalize import is_empty_value

# Strategy identifiers
PREFER_NON_EMPTY = "prefer_non_empty"               # longer non-empty value wins
PREFER_VERIFIED_CONTACT = "prefer_verified_contact"  # contact_updated_at-backed value wins
PREFER_MOST_RECENT = "prefer_most_recent"           # value from the more recently updated row
PREFER_MAX = "prefer_max"                           # numeric max wins
PREFER_NON_EMPTY_JSON = "prefer_non_empty_json"     # longer non-empty array/object wins


@dataclass
class FieldRule:
    name: str
    kind: str  # passed to is_empty_value()
    strategy: str


MERGE_RULES: Tuple[FieldRule, ...] = (
    FieldRule("first_name", "text", PREFER_NON_EMPTY),
    FieldRule("last_name", "text", PREFER_NON_EMPTY),
    FieldRule("name_prefix", "text", PREFER_NON_EMPTY),
    FieldRule("credential", "text", PREFER_NON_EMPTY),
    FieldRule("headline", "text", PREFER_NON_EMPTY),
    FieldRule("bio", "text", PREFER_NON_EMPTY),
    FieldRule("specialty", "text", PREFER_NON_EMPTY),
    FieldRule("profession_type", "text", PREFER_NON_EMPTY),
    FieldRule("provider_category", "text", PREFER_NON_EMPTY),
    FieldRule("american_board", "text", PREFER_NON_EMPTY),
    FieldRule("board", "text", PREFER_NON_EMPTY),
    FieldRule("license_number", "text", PREFER_NON_EMPTY),
    FieldRule("organization_name", "text", PREFER_NON_EMPTY),
    FieldRule("job_category", "text", PREFER_NON_EMPTY),
    FieldRule("work_authorization", "text", PREFER_NON_EMPTY),
    FieldRule("years_experience", "zeroable_numeric", PREFER_MAX),
    FieldRule("city", "text", PREFER_NON_EMPTY),
    FieldRule("state_code", "text", PREFER_NON_EMPTY),
    FieldRule("practice_state", "text", PREFER_NON_EMPTY),
    FieldRule("zip_code", "text", PREFER_NON_EMPTY),
    FieldRule("address", "text", PREFER_NON_EMPTY),
    FieldRule("lat", "numeric", PREFER_NON_EMPTY),
    FieldRule("lng", "numeric", PREFER_NON_EMPTY),
    FieldRule("npi_number", "text", PREFER_NON_EMPTY),
    FieldRule("resume_url", "text", PREFER_NON_EMPTY),
    FieldRule("profile_photo_url", "text", PREFER_NON_EMPTY),
    FieldRule("resume_sections", "json", PREFER_NON_EMPTY_JSON),
    FieldRule("education", "json", PREFER_NON_EMPTY_JSON),
    FieldRule("search_text", "text", PREFER_NON_EMPTY),
    FieldRule("phone", "text", PREFER_VERIFIED_CONTACT),
    FieldRule("email", "text", PREFER_VERIFIED_CONTACT),
    FieldRule("open_to_work", "bool", PREFER_MOST_RECENT),
    FieldRule("job_type_prefs", "json", PREFER_MOST_RECENT),
    FieldRule("pay_min_hourly", "zeroable_numeric", PREFER_MOST_RECENT),
    FieldRule("available_date", "text", PREFER_MOST_RECENT),
    FieldRule("completion_score", "numeric", PREFER_MAX),
)

_RULES_BY_NAME: Dict[str, FieldRule] = {r.name: r for r in MERGE_RULES}


def _json_len(value: Any) -> int:
    if isinstance(value, (list, dict)):
        return len(value)
    return 0 if value is None else len(str(value))


def _row_updated_at(row: Dict[str, Any]) -> Any:
    return row.get("updated_at")


def _row_is_contact_verified(row: Dict[str, Any], field_name: str) -> bool:
    # A phone/email is treated as "verified" when a user explicitly
    # confirmed it (contact_updated_by_user_id/email set), matching the
    # existing conservative write policy in profile_enrichment/neon_db.py.
    return bool(row.get("contact_updated_by_user_id") or row.get("contact_updated_by_email"))


@dataclass
class FieldDecision:
    field: str
    primary_value: Any
    duplicate_value: Any
    duplicate_profile_id: str
    chosen_value: Any
    chosen_source: str  # "primary" or duplicate_profile_id
    is_conflict: bool
    note: str = ""


def decide_field(rule: FieldRule, primary: Dict[str, Any], duplicate: Dict[str, Any]) -> FieldDecision:
    p_val = primary.get(rule.name)
    d_val = duplicate.get(rule.name)
    p_empty = is_empty_value(p_val, kind=rule.kind)
    d_empty = is_empty_value(d_val, kind=rule.kind)
    dup_id = duplicate.get("profile_id")

    def keep_primary(note: str = "") -> FieldDecision:
        return FieldDecision(rule.name, p_val, d_val, dup_id, p_val, "primary", False, note)

    def take_duplicate(note: str = "") -> FieldDecision:
        return FieldDecision(rule.name, p_val, d_val, dup_id, d_val, dup_id, False, note)

    # Universal fast paths: fill blanks, never discard the only known value.
    if p_empty and d_empty:
        return keep_primary()
    if p_empty and not d_empty:
        return take_duplicate("primary was empty")
    if not p_empty and d_empty:
        return keep_primary()

    # Both non-empty from here on.
    if rule.strategy == PREFER_NON_EMPTY:
        if str(p_val).strip() == str(d_val).strip():
            return keep_primary()
        if len(str(d_val)) > len(str(p_val)) * 1.2:
            return take_duplicate("duplicate value materially more complete")
        return FieldDecision(
            rule.name, p_val, d_val, dup_id, p_val, "primary", True,
            "both non-empty and differ; kept primary, flagged for review",
        )

    if rule.strategy == PREFER_NON_EMPTY_JSON:
        if _json_len(d_val) > _json_len(p_val):
            return take_duplicate("duplicate array/object has more entries")
        if _json_len(p_val) > 0 and _json_len(d_val) > 0 and p_val != d_val:
            return FieldDecision(
                rule.name, p_val, d_val, dup_id, p_val, "primary", True,
                "both non-empty and differ; kept primary (longer-or-equal), flagged for review",
            )
        return keep_primary()

    if rule.strategy == PREFER_VERIFIED_CONTACT:
        p_verified = _row_is_contact_verified(primary, rule.name)
        d_verified = _row_is_contact_verified(duplicate, rule.name)
        if str(p_val).strip().lower() == str(d_val).strip().lower():
            return keep_primary()
        if p_verified and not d_verified:
            return keep_primary("primary contact is user-verified")
        if d_verified and not p_verified:
            return take_duplicate("duplicate contact is user-verified")
        # Neither (or both) verified and they disagree: never silently
        # overwrite a contact detail on a guess - flag for manual review.
        return FieldDecision(
            rule.name, p_val, d_val, dup_id, p_val, "primary", True,
            "contact values differ and verification status is equal/unknown; kept primary, flagged for review",
        )

    if rule.strategy == PREFER_MOST_RECENT:
        p_time = _row_updated_at(primary)
        d_time = _row_updated_at(duplicate)
        if p_val == d_val:
            return keep_primary()
        if d_time and (not p_time or d_time > p_time):
            return take_duplicate("duplicate row updated more recently")
        return keep_primary()

    if rule.strategy == PREFER_MAX:
        try:
            p_num = float(p_val)
            d_num = float(d_val)
        except (TypeError, ValueError):
            return keep_primary()
        if d_num > p_num:
            return take_duplicate("duplicate value is larger")
        return keep_primary()

    return keep_primary()


def build_merge_plan(primary: Dict[str, Any], duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold every duplicate's data into `primary`, field by field, in order.

    Each duplicate is applied against the *running* merged state so that
    information from several duplicates can combine (e.g. one has a resume,
    another has an NPI number) without ever overwriting a reliable value
    with an inferior one.
    """
    merged_values: Dict[str, Any] = {rule.name: primary.get(rule.name) for rule in MERGE_RULES}
    field_sources: Dict[str, str] = {rule.name: "primary" for rule in MERGE_RULES}
    conflicts: List[Dict[str, Any]] = []
    decisions_by_duplicate: Dict[str, List[Dict[str, Any]]] = {}

    running_primary = dict(primary)
    running_primary.update(merged_values)

    for duplicate in duplicates:
        dup_id = duplicate.get("profile_id")
        decisions_by_duplicate[dup_id] = []
        for rule in MERGE_RULES:
            decision = decide_field(rule, running_primary, duplicate)
            decisions_by_duplicate[dup_id].append(
                {
                    "field": decision.field,
                    "primary_value": decision.primary_value,
                    "duplicate_value": decision.duplicate_value,
                    "chosen_value": decision.chosen_value,
                    "chosen_source": decision.chosen_source,
                    "is_conflict": decision.is_conflict,
                    "note": decision.note,
                }
            )
            if decision.chosen_source != "primary":
                merged_values[rule.name] = decision.chosen_value
                field_sources[rule.name] = decision.chosen_source
                running_primary[rule.name] = decision.chosen_value
            if decision.is_conflict:
                conflicts.append(
                    {
                        "field": rule.name,
                        "duplicate_profile_id": dup_id,
                        "primary_value": decision.primary_value,
                        "duplicate_value": decision.duplicate_value,
                        "note": decision.note,
                    }
                )

    return {
        "merged_values": merged_values,
        "field_sources": field_sources,
        "conflicts": conflicts,
        "decisions_by_duplicate": decisions_by_duplicate,
    }
