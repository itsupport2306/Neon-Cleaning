"""Deterministic comparison of stored profile values with values read from a resume.

Every comparison returns a :class:`FieldResult` whose status is one of

    match            the resume supports the stored value
    mismatch         the resume explicitly states something different (a confirmed discrepancy)
    not_verifiable   not enough evidence either way: never reported as a discrepancy
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

MATCH, MISMATCH, NOT_VERIFIABLE = "match", "mismatch", "not_verifiable"

_CREDENTIALS = frozenset(
    "rn md do np pa pac bsn msn dnp phd lpn lvn crna cna aprn fnp pharmd rph dds dmd mba cnm rrt dpt otr slp "
    "ccrn cen pmhnp agnp acnp cnp fnpc bls acls pals".split()
)
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})
_PLACEHOLDER_NAMES = frozenset({"unknown", "candidate", "n/a", "na", "none", "null", "test", "name", "firstname", "lastname"})

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "district of columbia": "DC", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}
STATE_CODES = frozenset(STATE_NAMES.values())


@dataclass
class FieldResult:
    status: str
    note: str = ""
    db_value: str = ""
    resume_value: str = ""


# ---------------------------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------------------------

def fold(value: Optional[str]) -> str:
    """Lower-case, accent-free text with every run of non-alphanumerics collapsed to one space."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def appears_in(value: Optional[str], text: str) -> bool:
    """True when `value` (compared on folded words) occurs verbatim in `text`: guards against invented values."""
    needle = fold(value)
    return bool(needle) and f" {needle} " in f" {fold(text)} "


def name_tokens(value: Optional[str]) -> List[str]:
    return [t for t in fold(value).split() if t not in _CREDENTIALS and t not in _SUFFIXES]


def state_code(value: Optional[str]) -> str:
    key = fold(value)
    if key.upper() in STATE_CODES and len(key) == 2:
        return key.upper()
    return STATE_NAMES.get(key, "")


def zip5(value: Optional[str]) -> str:
    match = re.search(r"\d{5}", str(value or ""))
    return match.group(0) if match else ""


# ---------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------

def compare_name(db_value: Optional[str], resume_value: Optional[str], label: str) -> FieldResult:
    db_text, resume_text = (db_value or "").strip(), (resume_value or "").strip()
    resume_tokens, db_tokens = name_tokens(resume_text), name_tokens(db_text)
    result = FieldResult(NOT_VERIFIABLE, db_value=db_text, resume_value=resume_text)
    if not resume_tokens:
        result.note = f"{label}: not stated in the resume"
        return result
    if not db_tokens:
        result.note = f"{label}: stored value is empty (nothing to validate)"
        return result
    if set(db_tokens) <= _PLACEHOLDER_NAMES:
        result.status = MISMATCH
        result.note = f"{label}: stored value {db_text!r} is a placeholder; resume says {resume_text!r}"
        return result
    if db_tokens == resume_tokens or set(db_tokens) <= set(resume_tokens) or set(resume_tokens) <= set(db_tokens):
        result.status = MATCH
        return result
    result.status = MISMATCH
    result.note = f"{label}: stored {db_text!r} but resume says {resume_text!r}"
    return result


def detect_swap(first_db: str, last_db: str, first_resume: str, last_resume: str) -> bool:
    fd, ld, fr, lr = (set(name_tokens(v)) for v in (first_db, last_db, first_resume, last_resume))
    return bool(fd and ld and fr and lr and fd <= lr and ld <= fr and not (fd <= fr))


# ---------------------------------------------------------------------------------------------
# address (the profiles table has city, state_code and zip_code but no street column)
# ---------------------------------------------------------------------------------------------

def format_address(parts: Dict[str, Optional[str]]) -> str:
    street, city = (parts.get("street") or "").strip(), (parts.get("city") or "").strip()
    state, zip_code = (parts.get("state") or "").strip(), (parts.get("zip") or "").strip()
    locality = ", ".join(p for p in (city, " ".join(p for p in (state, zip_code) if p)) if p)
    return ", ".join(p for p in (street, locality) if p)


def compare_address(db: Dict[str, Optional[str]], resume: Dict[str, Optional[str]]) -> FieldResult:
    """Compare city, state and ZIP when both sides have them. A street is reported but never compared."""
    result = FieldResult(NOT_VERIFIABLE, db_value=format_address(db), resume_value=format_address(resume))
    compared, differences, skipped = 0, [], []

    pairs = (
        ("city", fold(db.get("city")), fold(resume.get("city"))),
        ("state", state_code(db.get("state")), state_code(resume.get("state"))),
        ("zip", zip5(db.get("zip")), zip5(resume.get("zip"))),
    )
    for name, stored, found in pairs:
        if not found:
            skipped.append(f"{name} not in resume")
        elif not stored:
            skipped.append(f"{name} not stored")
        else:
            compared += 1
            if stored != found:
                differences.append(f"{name}: stored {stored.upper() if name != 'city' else db.get(name)!r} "
                                   f"but resume says {found if name != 'city' else resume.get(name)!r}")
    if differences:
        result.status = MISMATCH
        result.note = "address " + "; ".join(differences)
    elif compared:
        result.status = MATCH
        result.note = ("compared " + str(compared) + " component(s); " + "; ".join(skipped)) if skipped else ""
    else:
        result.note = "address: " + ("; ".join(skipped) if skipped else "nothing to compare")
    return result


# ---------------------------------------------------------------------------------------------
# specialty / profession
# ---------------------------------------------------------------------------------------------

VERDICTS = ("supported", "not_supported", "unclear")


def compare_specialty(
    db_specialty: Optional[str],
    resume_specialties: Sequence[str],
    profession: Optional[str],
    verdict: str,
    model_reason: str,
    resume_text: str,
    lexicon=None,
) -> FieldResult:
    """Specialty verdict. Equivalent names and abbreviations are resolved with the shared specialty lexicon
    (the same vocabulary/aliases the NULL-specialty pipeline uses) before the model's own verdict is trusted."""
    stored = (db_specialty or "").strip()
    found = "; ".join(s.strip() for s in resume_specialties if s and s.strip())
    summary = "; ".join(p for p in ((f"profession: {profession.strip()}" if profession and profession.strip() else ""),
                                    (f"specialties: {found}" if found else "")) if p)
    result = FieldResult(NOT_VERIFIABLE, db_value=stored, resume_value=summary)
    if not stored:
        result.note = "specialty: stored value is empty (nothing to validate)"
        return result

    stored_key = fold(stored)
    resume_keys = {fold(s) for s in resume_specialties if s}
    if lexicon is not None:
        mapped = lexicon.normalize_label(stored)
        if mapped:
            stored_key = fold(mapped[0])
            resume_keys |= {fold(m[0]) for s in resume_specialties if s for m in [lexicon.normalize_label(s)] if m}
            try:
                if any(fold(m.label) == stored_key for m in lexicon.find(resume_text)):
                    result.status, result.note = MATCH, f"stored specialty {stored!r} (or an alias) appears in the resume"
                    return result
            except Exception:  # noqa: BLE001 - the lexicon is a shortcut; the model verdict below still applies
                pass
    if stored_key in resume_keys:
        result.status = MATCH
        return result

    if verdict == "supported":
        result.status, result.note = MATCH, model_reason
    elif verdict == "not_supported" and (found or (profession or "").strip()):
        result.status = MISMATCH
        result.note = f"specialty: stored {stored!r} is not supported by the resume. {model_reason}".strip()
    else:
        result.note = f"specialty: unclear from the resume. {model_reason}".strip()
    return result


# ---------------------------------------------------------------------------------------------
# whole profile
# ---------------------------------------------------------------------------------------------

FIELDS = ("first_name", "last_name", "specialty", "address")


def mismatched(results: Dict[str, FieldResult]) -> List[str]:
    return [name for name in FIELDS if results[name].status == MISMATCH]


def overall_status(results: Dict[str, FieldResult]) -> str:
    """discrepancy | no_discrepancy | unverifiable (nothing at all could be checked)."""
    if mismatched(results):
        return "discrepancy"
    if any(r.status == MATCH for r in results.values()):
        return "no_discrepancy"
    return "unverifiable"


def field_statuses(results: Dict[str, FieldResult]) -> str:
    return "; ".join(f"{name}={results[name].status}" for name in FIELDS)


def reasons(results: Dict[str, FieldResult], only: Iterable[str]) -> str:
    return " | ".join(results[name].note for name in only if results[name].note)
