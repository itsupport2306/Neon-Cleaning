"""Pure, DB-free helpers: value normalization and emptiness rules.

Kept dependency-free (stdlib only) so they're trivial to unit test and reuse
from both the SQL-generation side and the Python-side clustering/merge logic.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_NON_DIGIT_RE = re.compile(r"\D+")
_NON_ALPHA_RE = re.compile(r"[^a-z0-9]+")

# JSON/array "empty" shapes that are semantically unset, not real data.
_EMPTY_JSON_LITERALS = {"[]", "{}", "null", "", "none"}


def normalize_phone(raw: Any) -> str | None:
    """Digits-only, last 10 digits (US), or None if too short to be real."""
    if not raw:
        return None
    digits = _NON_DIGIT_RE.sub("", str(raw))
    if len(digits) < 7:
        return None
    digits = digits[-10:]
    # All-same-digit or sequential placeholder numbers are never real.
    if len(set(digits)) <= 1:
        return None
    return digits


def normalize_name(first: Any, last: Any) -> str:
    first = (first or "").strip().lower()
    last = (last or "").strip().lower()
    combined = f"{first} {last}"
    return _NON_ALPHA_RE.sub(" ", combined).strip()


def name_similarity(name_a: str, name_b: str) -> float:
    if not name_a or not name_b:
        return 0.0
    return SequenceMatcher(None, name_a, name_b).ratio()


def is_empty_value(value: Any, *, kind: str = "text") -> bool:
    """Emptiness is field-meaning dependent, per the task spec.

    kind:
      "text"    -> None/""/whitespace-only
      "numeric" -> None only (0 is real data, e.g. completion_score)
      "zeroable_numeric" -> None or 0 (e.g. years_experience, pay_min_hourly,
                              where 0 is used as a placeholder, not a fact)
      "bool"    -> never considered empty (False is a real answer)
      "json"    -> None, [], {}, or their string/serialized equivalents
    """
    if kind == "bool":
        return False
    if value is None:
        return True
    if kind == "numeric":
        return False
    if kind == "zeroable_numeric":
        try:
            return float(value) == 0
        except (TypeError, ValueError):
            return False
    if kind == "json":
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return str(value).strip().lower() in _EMPTY_JSON_LITERALS
    # text
    return str(value).strip() == ""
