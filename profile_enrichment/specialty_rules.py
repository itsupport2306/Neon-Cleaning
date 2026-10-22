"""Single source of truth for what counts as a "suspicious" specialty value.

Both the SQL fetch (so the database does the filtering instead of pulling the
whole table into Python) and the Python-side double-check use these same
patterns and name list, so the two can never drift out of sync.
"""

from __future__ import annotations

import re
from typing import Any, Optional

JUNK_VALUES: tuple[str, ...] = ("web", "n/a", "na", "none", "unknown", "test", "-", ".")

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

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b[\w-]+\.(com|net|org|io|co|gov|edu)\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\d[\s.\-()]*){7,}\d")
_NAME_PREFIX_RE = re.compile(r"^\s*([A-Za-z]+)\s*[.,]\s*\S")


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

    name_match = _NAME_PREFIX_RE.match(text)
    if name_match and name_match.group(1).lower() in COMMON_FIRST_NAMES:
        return "name_prefix"

    return None


def _sql_string_array(values: tuple[str, ...]) -> str:
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
        f"    OR lower(substring(specialty from '^\\s*([A-Za-z]+)\\s*[.,]\\s*\\S')) = ANY ({names_array})\n"
        "  )"
    )
