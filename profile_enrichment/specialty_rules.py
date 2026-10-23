"""Single source of truth for what counts as a "suspicious" specialty value.

Both the SQL fetch (so the database does the filtering instead of pulling the
whole table into Python) and the Python-side double-check use these same
patterns and name list, so the two can never drift out of sync.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

JUNK_VALUES: tuple[str, ...] = ("web", "n/a", "na", "none", "unknown", "test", "-", ".")

# Reference list of specialty strings already confirmed, by manual review, to be
# invalid/misspelled/non-standard (`unmatched_specialities_for_review.csv`, kept at the
# repo root and never modified by this code - it is read-only reference data). Any
# stored `specialty` value that exactly matches one of these (case/whitespace
# insensitive) is corrected the same way as the other suspicious patterns below: by
# reprocessing the resume, never by guessing a replacement from the CSV itself.
DEFAULT_REFERENCE_CSV_PATH = Path(__file__).resolve().parent.parent / "unmatched_specialities_for_review.csv"
_REFERENCE_CSV_COLUMN_CANDIDATES = ("speciality", "specialty")


@lru_cache(maxsize=4)
def _load_reference_unmatched_values(path: str) -> frozenset[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        return frozenset()

    values: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        column = next(
            (c for c in _REFERENCE_CSV_COLUMN_CANDIDATES if reader.fieldnames and c in reader.fieldnames),
            None,
        )
        if column is None:
            return frozenset()
        for row in reader:
            cleaned = " ".join(str(row.get(column) or "").split()).lower()
            if cleaned:
                values.add(cleaned)
    return frozenset(values)


def get_reference_unmatched_values(path: Optional[Path | str] = None) -> frozenset[str]:
    """Known-bad specialty strings from the review CSV, lowercased/whitespace-normalized."""
    return _load_reference_unmatched_values(str(path or DEFAULT_REFERENCE_CSV_PATH))

# A small set of common first names used only to tell "John, Cardiology" (a name
# prefix to strip) apart from a legitimate specialty that happens to contain a
# comma or period, like "Pediatrics, General".
COMMON_FIRST_NAMES: tuple[str, ...] = (
    "james", "john", "robert", "michael", "william", "david", "richard", "joseph",
    "thomas", "charles", "christopher", "daniel", "matthew", "anthony", "mark",
    "donald", "steven", "paul", "andrew", "joshua", "kenneth", "kevin", "brian",
    "george", "timothy", "ronald", "edward", "jason", "jeffrey", "ryan", "jacob",
    "gary", "nicholas", "eric", "jonathan", "stephen", "larry", "justin", "scott",
    "brandon", "benjamin", "samuel", "frank", "raymond", "patrick", "alexander",
    "jack", "dennis", "jerry", "mary", "patricia", "jennifer", "linda", "elizabeth",
    "barbara", "susan", "jessica", "sarah", "karen", "nancy", "lisa", "margaret",
    "betty", "sandra", "ashley", "kimberly", "emily", "donna", "michelle", "carol",
    "amanda", "melissa", "deborah", "stephanie", "rebecca", "laura", "sharon",
    "cynthia", "kathleen", "amy", "angela", "shirley", "anna", "ruth", "brenda",
    "pamela", "nicole", "katherine", "samantha", "christine", "emma", "catherine",
    "debra", "virginia", "rachel", "carolyn", "janet", "maria", "heather", "diane",
    "julie", "joyce", "victoria", "kelly", "christina", "lauren", "joan", "evelyn",
    "olivia", "judith", "megan", "cheryl", "martha", "andrea", "frances", "hannah",
    "jacqueline", "ann", "gloria", "jean", "kathryn", "alice", "teresa", "sara",
    "janice", "doris", "madison", "julia", "grace", "denise", "amber", "marilyn",
    "beverly", "danielle", "theresa", "sophia", "marie", "diana", "brittany",
    "natalie", "isabella", "charlotte", "rose", "alexis", "kayla",
)

# US state names/codes stored as the specialty by mistake (e.g. a state field
# scraped into the wrong column). Matched only when the *entire* value is a
# state, so a legitimate specialty that happens to contain "Washington" (as in
# "Washington University") is never flagged.
US_STATE_NAMES: tuple[str, ...] = (
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "district of columbia", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin",
    "wyoming",
)
US_STATE_CODES: tuple[str, ...] = (
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi",
    "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn",
    "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh",
    "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa",
    "wv", "wi", "wy",
)
_STATE_CODE_RE = re.compile(r"^[A-Za-z]{2}$")
# A full "First Last" name with no comma/period separator, where the first
# word is a known first name - distinct from the "John, Cardiology" prefix
# case handled by _NAME_PREFIX_RE, which requires a separator.
_FULL_NAME_RE = re.compile(r"^\s*([A-Za-z]+)\s+[A-Za-z]+(?:\s+[A-Za-z]+)?\s*$")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b[\w-]+\.(com|net|org|io|co|gov|edu)\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\d[\s.\-()]*){7,}\d")
_NAME_PREFIX_RE = re.compile(r"^\s*([A-Za-z]+)\s*[.,]\s*\S")
# No real specialty starts with a digit, so this catches two distinct OCR failure
# modes seen in practice: a misread credential prefix ("00, General Surgery" /
# "1D Pulmonology", where "MD"/"DO" got OCR'd as digits) and a street address that
# was scraped into the specialty field by mistake ("19780 Castleberry Loop...").
_LEADING_DIGIT_RE = re.compile(r"^\s*\d")

# Matches a short (1-3 char) leading token that contains at least one digit -
# the garbled "MD"/"DO" credential marker - followed by a comma/period and
# whitespace, then the real specialty text (which starts with a capital letter).
# Deliberately narrower than _LEADING_DIGIT_RE: it must not match the street-address
# case, where the leading digit run is longer than 3 characters and isn't followed
# by a separator until well past the token (see repair_leading_digit_specialty).
_CREDENTIAL_PREFIX_RE = re.compile(
    r"^\s*(?=[0-9A-Za-z]{1,3}[.,]?\s)(?=[^,\s]*\d)[0-9A-Za-z]{1,3}[.,]?\s+(?=[A-Z])"
)
_UNDERSCORE_RE = re.compile(r"(?:^|\s)_+(?:\s|$)")
_VALID_SPECIALTY_RE = re.compile(r"[A-Za-z][A-Za-z\s,/&'-]*")


def repair_leading_digit_specialty(value: Any) -> Optional[str]:
    """Try to recover the real specialty from a garbled OCR credential prefix.

    Handles values like "00, General _ General Surgery" or "1D Pulmonology"
    (where "MD"/"DO" was OCR'd as digits) by stripping the bogus prefix,
    collapsing stray underscores, and removing immediate duplicate words left
    behind by the OCR pass. Returns None when the value doesn't match this
    specific pattern (e.g. a street address scraped into the specialty field),
    so the caller knows to fall back to full resume reprocessing instead.
    """
    text = "" if value is None else " ".join(str(value).split())
    match = _CREDENTIAL_PREFIX_RE.match(text)
    if not match:
        return None

    remainder = text[match.end():]
    remainder = _UNDERSCORE_RE.sub(" ", remainder)
    words = remainder.split()
    deduped: list[str] = []
    for word in words:
        if deduped and deduped[-1].lower() == word.lower():
            continue
        deduped.append(word)
    remainder = " ".join(deduped).strip(" ,.")

    if not remainder or any(ch.isdigit() for ch in remainder):
        return None
    if not _VALID_SPECIALTY_RE.fullmatch(remainder):
        return None
    return remainder


def classify_suspicious_specialty(value: Any) -> Optional[str]:
    """Return a short reason string when a stored specialty value looks wrong, else None."""
    text = "" if value is None else " ".join(str(value).split())
    if not text:
        return None

    lowered = text.lower()
    if lowered in JUNK_VALUES:
        return "junk_value"
    if _EMAIL_RE.search(text):
        return "email_address"
    if _URL_RE.search(text):
        return "url"
    if _DOMAIN_RE.search(text):
        return "domain_like_value"
    if _PHONE_RE.search(text):
        return "phone_number"
    if _LEADING_DIGIT_RE.match(text):
        return "leading_digit_junk"

    name_match = _NAME_PREFIX_RE.match(text)
    if name_match and name_match.group(1).lower() in COMMON_FIRST_NAMES:
        return "name_prefix"

    if lowered in US_STATE_NAMES or (_STATE_CODE_RE.match(text) and lowered in US_STATE_CODES):
        return "state_value"

    full_name_match = _FULL_NAME_RE.match(text)
    if full_name_match and full_name_match.group(1).lower() in COMMON_FIRST_NAMES:
        return "name_full"

    if lowered in get_reference_unmatched_values():
        return "unmatched_reference_csv"

    return None


def _sql_string_array(values: tuple[str, ...]) -> str:
    if not values:
        return "ARRAY[]::text[]"
    escaped = ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    return f"ARRAY[{escaped}]"


def build_specialty_fix_sql(table_name: str) -> str:
    """Build the default suspicious-specialty SELECT, mirroring `classify_suspicious_specialty`.

    Filtering happens in SQL so the database only hands back the rows that
    actually need a resume reprocessed, instead of every row with a non-empty
    specialty.

    This string is later executed through psycopg2 together with bound
    parameters (e.g. `... LIMIT %s`), so every literal `%` in it - such as the
    one in the email regex's character class - must be escaped as `%%` or
    psycopg2's parameter substitution miscounts placeholders and raises
    `IndexError: tuple index out of range`.
    """
    junk_array = _sql_string_array(JUNK_VALUES)
    names_array = _sql_string_array(COMMON_FIRST_NAMES)
    states_array = _sql_string_array(US_STATE_NAMES)
    state_codes_array = _sql_string_array(US_STATE_CODES)
    reference_array = _sql_string_array(tuple(sorted(get_reference_unmatched_values())))
    return (
        f"SELECT * FROM {table_name}\n"
        "WHERE specialty IS NOT NULL\n"
        "  AND specialty !~ '^\\s*$'\n"
        "  AND (\n"
        f"    lower(btrim(specialty)) = ANY ({junk_array})\n"
        "    OR specialty ~* '[a-z0-9._%%+-]+@[a-z0-9.-]+\\.[a-z]{2,}'\n"
        "    OR specialty ~* 'https?://|www\\.'\n"
        "    OR specialty ~* '\\b[a-z0-9-]+\\.(com|net|org|io|co|gov|edu)\\b'\n"
        "    OR specialty ~ '(?:\\d[\\s.\\-()]*){7,}\\d'\n"
        "    OR specialty ~ '^\\s*\\d'\n"
        f"    OR lower(substring(specialty from '^\\s*([A-Za-z]+)\\s*[.,]\\s*\\S')) = ANY ({names_array})\n"
        f"    OR lower(btrim(specialty)) = ANY ({states_array})\n"
        f"    OR (btrim(specialty) ~ '^[A-Za-z]{{2}}$' AND lower(btrim(specialty)) = ANY ({state_codes_array}))\n"
        f"    OR lower(substring(btrim(specialty) from '^([A-Za-z]+)\\s+[A-Za-z]+(?:\\s+[A-Za-z]+)?$')) = ANY ({names_array})\n"
        f"    OR lower(btrim(specialty)) = ANY ({reference_array})\n"
        "  )"
    )
