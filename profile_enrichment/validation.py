"""Field-level validation for AI/regex extracted values.

The rule this module enforces is: **no evidence -> no update**. Every candidate
value must survive three gates before it is allowed near NeonDB:

1. shape validation (does it look like an email/phone/name/state at all?)
2. placeholder rejection (johndoe@example.com, 123-456-7890, John Doe, ...)
3. source grounding (does the value actually occur in the document text?)

Anything rejected is logged as ``REJECTED_AI_VALUE`` so a bad prompt, a bad
model and a bad validation rule can be told apart from the logs alone.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

VALIDATED_FIELDS = (
    "first_name",
    "last_name",
    "city",
    "state",
    "state_code",
    "specialty",
    "phone",
    "email",
)

# Values that mean "the model had nothing" regardless of the field.
_NULL_TOKENS = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "n.a.",
    "none",
    "null",
    "nil",
    "nan",
    "unknown",
    "unknown candidate",
    "candidate",
    "not provided",
    "not available",
    "not found",
    "not specified",
    "not mentioned",
    "no information",
    "no data",
    "missing",
    "tbd",
    "todo",
    "string",
    "example",
    "placeholder",
    "xxx",
    "xxxx",
}

_PLACEHOLDER_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "email.com",
    "domain.com",
    "yourdomain.com",
    "test.com",
    "sample.com",
    "abc.com",
    "xyz.com",
    "acme.com",
}

_PLACEHOLDER_EMAIL_LOCALS = {
    "johndoe",
    "john.doe",
    "john_doe",
    "john",
    "janedoe",
    "jane.doe",
    "jane",
    "firstname.lastname",
    "firstnamelastname",
    "name",
    "yourname",
    "your.name",
    "email",
    "youremail",
    "user",
    "username",
    "example",
    "test",
    "sample",
    "someone",
    "abc",
    "xyz",
    "noreply",
    "no-reply",
    "candidate",
}

_PLACEHOLDER_NAMES = {
    "doe",
    "john doe",
    "jane doe",
    "john smith",
    "first",
    "last",
    "firstname",
    "lastname",
    "first name",
    "last name",
    "full name",
    "your name",
    "name",
    "candidate",
    "applicant",
    "resume",
    "curriculum vitae",
    "cv",
}

# Digit strings that are obviously fabricated rather than transcribed.
_PLACEHOLDER_PHONE_DIGITS = {
    "1234567890",
    "11234567890",
    "0123456789",
    "5551234567",
    "15551234567",
    "5555555555",
    "1111111111",
    "0000000000",
    "9999999999",
    "1234567891",
    "9876543210",
}

_PLACEHOLDER_CITIES = {
    "city",
    "your city",
    "yourcity",
    "anytown",
    "hometown",
    "somewhere",
    "unknown city",
}

# Words that mark a "name" as really being a sentence, a heading or a job title.
_NAME_STOPWORDS = {
    "i", "im", "am", "my", "me", "the", "a", "an", "and", "of", "is", "are",
    "was", "were", "this", "that", "we", "you", "your", "he", "she", "they",
    "hello", "hi", "resume", "cv", "curriculum", "vitae", "profile", "summary",
    "objective", "experience", "education", "skills", "certified",
    "certification", "licensed", "license", "registered", "assistant", "nurse",
    "technician", "manager", "director", "specialist", "coordinator",
    "supervisor", "student", "intern", "professional", "references", "contact",
    "address", "phone", "email", "mobile", "tel", "name", "employer", "company",
    "hospital", "clinic", "university", "college", "school", "department",
    "present", "current",
}

_US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_NAME_CHARS_RE = re.compile("^[A-Za-z][A-Za-z'’.\\- ]*$")


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _is_null_token(value: str) -> bool:
    return value.strip().lower().strip(".") in _NULL_TOKENS


# ---------------------------------------------------------------------------
# source grounding
# ---------------------------------------------------------------------------

def _contains_phrase(lowered_source: str, phrase: str) -> bool:
    words = re.findall(r"[a-z0-9]+", phrase)
    if not words:
        return False
    pattern = r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(re.escape(w) for w in words) + r"(?![a-z0-9])"
    return re.search(pattern, lowered_source) is not None


def _equivalent_in_source(value: str, source_text: str) -> bool:
    """True when a reviewed alias/vocabulary phrase that maps to the same label as `value` is in the text."""
    from profile_enrichment.specialty_lexicon import SpecialtyLexicon

    lexicon = SpecialtyLexicon.load()
    mapped = lexicon.normalize_label(value)
    if not mapped:
        return False
    return any(match.label == mapped[0] for match in lexicon.find(source_text))


def _is_grounded(field: str, value: str, source_text: str) -> bool:
    """True when the value can actually be found in the document text."""
    if not source_text:
        return False

    lowered = re.sub(r"\s+", " ", source_text.lower())

    if field == "phone":
        value_digits = _digits(value)[-10:]
        return bool(value_digits) and value_digits in _digits(source_text)

    if field == "email":
        return value.lower() in lowered

    if field in {"state", "state_code"}:
        # A state code may legitimately be inferred from the spelled-out name.
        from profile_enrichment.extractors import _state_from_text  # local import avoids a cycle

        if re.search(r"\b" + re.escape(value.lower()) + r"\b", lowered):
            return True
        return _state_from_text(source_text) == value.upper()

    if field == "specialty":
        # The phrase itself must be in the text, or a reviewed equivalent of it (alias/abbreviation of a known
        # label) must be. A shared 6-letter word prefix is not evidence: "Inspect" is not a specialty because
        # "inspection" appears in a job description.
        if _contains_phrase(lowered, value.lower()):
            return True
        return _equivalent_in_source(value, source_text)

    # names and cities must appear verbatim
    return re.search(r"\b" + re.escape(value.lower()) + r"\b", lowered) is not None


# ---------------------------------------------------------------------------
# per-field shape / placeholder rules
# ---------------------------------------------------------------------------

def _check_email(value: str) -> Optional[str]:
    if not _EMAIL_RE.match(value):
        return "invalid_email_format"
    local, _, domain = value.lower().partition("@")
    if domain in _PLACEHOLDER_EMAIL_DOMAINS or domain.startswith("example."):
        return "placeholder_email"
    if local in _PLACEHOLDER_EMAIL_LOCALS:
        return "placeholder_email"
    if re.fullmatch(r"[a-z]?\d+", local):
        return "placeholder_email"
    return None


def _check_phone(value: str) -> Optional[str]:
    digits = _digits(value)
    if len(digits) < 10 or len(digits) > 15:
        return "invalid_phone_length"
    core = digits[-10:]
    if digits in _PLACEHOLDER_PHONE_DIGITS or core in _PLACEHOLDER_PHONE_DIGITS:
        return "placeholder_phone"
    if len(set(core)) <= 2:
        return "placeholder_phone"
    if core in "01234567890123456789" or core in "98765432109876543210":
        return "sequential_phone"
    if core.startswith("555"):
        return "placeholder_phone"
    return None


def _check_name(value: str) -> Optional[str]:
    lowered = value.lower()
    if lowered in _PLACEHOLDER_NAMES:
        return "placeholder_name"
    if len(value) < 2:
        return "name_too_short"
    if len(value) > 40:
        return "name_too_long"
    if any(ch.isdigit() for ch in value):
        return "name_contains_digits"
    if not _NAME_CHARS_RE.match(value):
        return "name_has_invalid_characters"
    words = re.findall(r"[a-z]+", lowered)
    if not words:
        return "name_has_no_letters"
    if len(words) > 3:
        return "name_looks_like_a_sentence"
    if any(word in _NAME_STOPWORDS for word in words):
        return "name_looks_like_a_sentence_or_title"
    return None


def _check_state(value: str) -> Optional[str]:
    if len(value) != 2 or not value.isalpha():
        return "state_not_two_letter_code"
    if value.upper() not in _US_STATE_CODES:
        return "state_not_a_us_state"
    return None


def _check_city(value: str) -> Optional[str]:
    if value.lower() in _PLACEHOLDER_CITIES:
        return "placeholder_city"
    if len(value) < 2 or len(value) > 60:
        return "city_length_out_of_range"
    if any(ch.isdigit() for ch in value):
        return "city_contains_digits"
    if not _NAME_CHARS_RE.match(value):
        return "city_has_invalid_characters"
    if len(value.split()) > 4:
        return "city_looks_like_a_sentence"
    return None


# Words that give away a "specialty" that is really a slice of prose.
_SPECIALTY_STOPWORDS = {
    "with", "years", "year", "experience", "i", "am", "my", "the", "was", "is",
    "who", "seeking", "looking", "dedicated", "responsible",
}


# Credentials, degrees, trainings and tools are not specialties, whatever the model says.
_SPECIALTY_NON_VALUES = {
    "bls", "acls", "pals", "cpr", "nrp", "tncc", "hipaa", "epic", "cerner", "meditech", "ms office",
    "microsoft office", "bsn", "msn", "dnp", "bs", "ba", "ms", "ma", "mba", "phd", "associate degree",
    "bachelor of science", "master of science", "license", "licensed", "certified", "certification",
    "board certified", "cad", "sql", "python", "java", "excel", "word", "inspect",
}


def _check_specialty(value: str) -> Optional[str]:
    if value.lower().strip(". ") in _SPECIALTY_NON_VALUES:
        return "specialty_is_credential_degree_or_skill"
    if len(value) < 3 or len(value) > 80:
        return "specialty_length_out_of_range"
    if any(ch.isdigit() for ch in value):
        return "specialty_contains_digits"
    if any(word.strip(".,") in _SPECIALTY_STOPWORDS for word in value.lower().split()):
        return "specialty_looks_like_prose"
    if len(value.split()) > 6:
        return "specialty_looks_like_a_sentence"
    if not re.search(r"[A-Za-z]{3,}", value):
        return "specialty_has_no_words"
    return None


_FIELD_CHECKS = {
    "email": _check_email,
    "phone": _check_phone,
    "first_name": _check_name,
    "last_name": _check_name,
    "city": _check_city,
    "state": _check_state,
    "state_code": _check_state,
    "specialty": _check_specialty,
}


def canonicalize(field: str, value: str) -> str:
    if field in {"state", "state_code"}:
        return value.upper()
    return value


_DB_PLACEHOLDERS = {"unknown", "candidate", "unknown candidate", "n/a", "na", "none", "null", "-", "--", "tbd"}


def is_blank_or_placeholder(field: str, value: Any) -> bool:
    """True for a stored value that carries no information (NULL, '', 'Unknown', 'Candidate', ...).

    Only unambiguous no-information tokens count. A populated-but-wrong value such as "Class Of" is NOT
    blank: replacing it is a correction and belongs in the review workflow.
    """
    text = normalize(value)
    if not text:
        return True
    if field in {"first_name", "last_name", "city", "specialty", "state_code", "email"}:
        return text.lower().strip(".") in _DB_PLACEHOLDERS
    return False


def validate_field(
    field: str,
    value: Any,
    source_text: str = "",
    require_grounding: bool = True,
) -> Tuple[bool, str, str]:
    """Validate one extracted value.

    Returns ``(is_valid, canonical_value, reason)``. ``reason`` carries the
    rejection reason when invalid and ``"ok"`` otherwise.
    """
    cleaned = normalize(value)
    if not cleaned:
        return False, "", "empty"
    if _is_null_token(cleaned):
        return False, "", "null_token"

    check = _FIELD_CHECKS.get(field)
    if check is not None:
        reason = check(cleaned)
        if reason:
            return False, "", reason

    cleaned = canonicalize(field, cleaned)

    if require_grounding and field in _FIELD_CHECKS and not _is_grounded(field, cleaned, source_text):
        return False, "", "not_found_in_source"

    return True, cleaned, "ok"


def validate_extracted(
    extracted: Dict[str, Any],
    source_text: str = "",
    profile_id: str = "",
    require_grounding: bool = True,
) -> Dict[str, str]:
    """Validate a whole extraction result, logging every rejected value."""
    validated: Dict[str, str] = {}
    for field, raw_value in (extracted or {}).items():
        if field not in VALIDATED_FIELDS:
            continue
        ok, value, reason = validate_field(field, raw_value, source_text, require_grounding)
        if ok:
            validated[field] = value
        elif reason not in {"empty", "null_token"}:
            logger.warning(
                "REJECTED_AI_VALUE | profile_id=%s | field=%s | value=%s | reason=%s",
                profile_id or "-",
                field,
                normalize(raw_value),
                reason,
            )
    return validated
