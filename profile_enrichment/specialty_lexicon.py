"""Evidence-based specialty candidates: vocabulary matching, reviewed aliases and context scoring.

`profiles.specialty` holds taxonomy-style labels ("Family Medicine", "Registered Nurse", "ICU"). A value
is only accepted when it maps to a label already used in the database (the vocabulary), through a
*reviewed* alias, or when it sits on an explicit "Specialty:" line. Every candidate carries the exact
source text it came from and where it sits in the resume (section, line), so a mention in an
Education block or an employer name cannot be mistaken for the person's specialty.
"""

from __future__ import annotations

import bisect
import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_VOCABULARY_PATH = DATA_DIR / "specialty_vocabulary.csv"
DEFAULT_ALIAS_PATH = DATA_DIR / "specialty_aliases.csv"

# Labels that exist in the vocabulary but are ordinary words or noise in a resume.
LABEL_STOPLIST = frozenset(
    {"or", "do", "bus", "java", "taxi", "sales", "other", "others", "none", "unknown", "student", "n/a",
     "na", "md/do", "md do", "test", "web", "nurse", "general", "medical", "health", "care", "support",
     "services", "management", "administration", "education", "research", "technician", "assistant",
     "specialist", "marketing", "accounting", "devops", "software", "engineering", "accountant", "finance",
     "networking", "manager", "technical", "technology", "masters", "doctorate", "developer", "consultant",
     "administrative", "recruiter", "security", "cybersecurity", "programming", "paralegal", "architect",
     "salesforce", "sharepoint", "receptionist", "telecommunications", "caregiver", "homemaker", "driver",
     "ambulance", "office manager", "physician", "pharmacy", "hospice", "home health", "clinic center"}
)
# Generic one-word areas: only trusted in a headline/labeled position, never from a passing mention.
WEAK_LABELS = frozenset(
    {"medicine", "surgery", "nursing", "pediatric", "emergency", "psychiatric", "neonatal", "geriatric",
     "plastic", "surgical", "diabetes", "neuro", "rehab", "psych", "radiology", "pathology", "dialysis"}
)
# Vocabulary entries that describe a facility/organisation (provider taxonomy), never a person.
_FACILITY_LABEL_RE = re.compile(
    r"(agency|facility|clinic/center|hospital|organization|laborator\w*|\(dme\)|equipment|supplies|"
    r"transport|contractor|provider|\(van\)|residential treatment|day training)\s*(\(.*\))?$",
    re.IGNORECASE,
)
# A label ending in one of these names a role/credential ("Registered Nurse"); anything else is a practice area.
_PROFESSION_TAILS = frozenset(
    {"nurse", "practitioner", "assistant", "therapist", "pharmacist", "dentist", "technician", "technologist",
     "physician", "counselor", "worker", "aide", "hygienist", "analyst", "psychologist", "midwife", "anesthetist",
     "dietitian", "pathologist", "chiropractor", "optometrist", "acupuncturist", "trainer", "specialist",
     "phlebotomist", "paramedic", "surgeon", "psychiatrist", "pediatrician", "anesthesiologist", "attendant"}
)
MIN_VOCAB_COUNT = 20
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|&")
_ORG_WORDS = frozenset(
    {"hospital", "hospitals", "clinic", "clinics", "university", "college", "school", "institute", "center",
     "centre", "health", "system", "systems", "foundation", "inc", "llc", "group", "associates"}
)
_LABELED_FIELD_RE = re.compile(
    r"^\s*(?:clinical\s+)?(?:specialty|specialties|speciality|specialization|practice\s+area|area\s+of\s+practice)"
    r"\s*[:\-–]\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
# A line describing a degree, school or training is education, whichever heading the parser thought it was under.
_EDUCATION_LINE_RE = re.compile(
    r"\b(master|bachelor|associate|doctorate|doctor of|diploma|degree|university|college|school of|"
    r"b\.?s\.?n?|m\.?s\.?n?|ph\.?d|dnp|residency at|fellowship at|graduat\w+)\b", re.IGNORECASE)
MIN_SCORE = 5.0       # one stray mention in a bullet (<=4.5) is not enough; a header, credential line, label or repeats are
MIN_SCORE_WEAK = 6.0  # generic one-word areas ("Surgery", "Medicine") need a headline-grade position
_HEADING_WORDS = {
    "summary": ("summary", "profile", "objective", "about", "overview"),
    "experience": ("experience", "employment", "work history", "professional history", "career"),
    "education": ("education", "academic", "training", "coursework"),
    "skills": ("skills", "competencies", "expertise", "proficiencies"),
    "credentials": ("certification", "certifications", "licensure", "licenses", "license", "credentials"),
    "references": ("references",),
}


def normalize_key(value: str) -> str:
    """Case/punctuation-insensitive key: 'Marriage & Family Therapist' == 'marriage and family therapist'."""
    words = [("and" if tok == "&" else tok.lower()) for tok in _TOKEN_RE.findall(value or "")]
    return " ".join(words)


def label_kind(label: str) -> str:
    """'profession' for role/credential labels, 'practice_area' for everything else."""
    words = normalize_key(label).split()
    return "profession" if words and (words[-1] in _PROFESSION_TAILS or label in {"RN", "LPN", "CNA", "CRNA"}) else "practice_area"


def _is_weak(label: str) -> bool:
    key = normalize_key(label)
    return len(label) <= 3 or key in LABEL_STOPLIST or key in WEAK_LABELS


def _is_abbreviation(label: str) -> bool:
    return bool(label) and label == label.upper() and any(ch.isalpha() for ch in label) and len(label) <= 6


@dataclass(frozen=True)
class SpecialtyMatch:
    label: str
    phrase: str
    start: int
    end: int
    via: str  # "vocabulary" | "alias" | "labeled_field"
    section: str = "unknown"
    line: str = ""
    line_no: int = 0


@dataclass
class SpecialtyDecision:
    status: str  # "selected" | "ambiguous" | "none"
    label: Optional[str] = None
    evidence: str = ""
    score: float = 0.0
    section: str = ""
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    reason: str = ""


_HEALTHCARE_SIGNALS = (
    "patient", "clinical", "hospital", "nursing", "nurse", "medical", "healthcare", "health care", "physician",
    "therapy", "therapist", "pharmac", "dental", "surgery", "surgical", "npi", "bls", "acls", "cpr", "hipaa",
    "licensure", "board certified", "clinic", "rehabilitation", "counseling", "behavioral", "residency",
    "caregiver", "phlebotom", "radiolog", "diagnos", "oncolog", "pediatric", "emergency", "icu", "telemetry",
)
_HEALTHCARE_CREDENTIALS_RE = re.compile(r"(?:RN|LPN|LVN|CNA|CRNA|NP|FNP|APRN|PA-C|MD|DO|DDS|DMD|PharmD|DPT|OTR|SLP|LCSW|LMFT|BCBA|RBT)")


def healthcare_signal(text: str) -> int:
    """Number of distinct healthcare indicators in the resume (words plus credential abbreviations)."""
    lowered = text.lower()
    hits = sum(1 for word in _HEALTHCARE_SIGNALS if word in lowered)
    return hits + len(set(_HEALTHCARE_CREDENTIALS_RE.findall(text)))


class ResumeLayout:
    """Lines with offsets and the section each belongs to, so a match can be weighed by where it sits."""

    def __init__(self, text: str):
        self.text = text
        self.lines: List[Tuple[int, int, str, str]] = []  # (start, end, stripped line, section)
        self.nonempty_ordinal: Dict[int, int] = {}
        section = "header"
        offset = 0
        count = 0
        for raw in text.split("\n"):
            line = raw.strip()
            heading = self._heading_section(line)
            if heading:
                section = heading
            if line:
                self.nonempty_ordinal[len(self.lines)] = count
                count += 1
            self.lines.append((offset, offset + len(raw), line, "heading" if heading else section))
            offset += len(raw) + 1
        self._starts = [start for start, *_ in self.lines]

    @staticmethod
    def _heading_section(line: str) -> Optional[str]:
        if not line or len(line) > 48 or line.endswith(".") or "@" in line or ":" in line.rstrip(":"):
            return None
        lowered = " ".join(re.sub(r"[^a-z& ]", " ", line.lower()).split())
        words = lowered.split()
        if not words or len(words) > 4:
            return None
        for section, names in _HEADING_WORDS.items():
            if any(name in lowered for name in names):
                return section
        return None

    def locate(self, position: int) -> Tuple[str, str, int]:
        """(line text, section, ordinal among non-empty lines) for a character offset."""
        index = max(0, bisect.bisect_right(self._starts, position) - 1)
        _, _, line, section = self.lines[index]
        return line, section, self.nonempty_ordinal.get(index, 10**6)


class SpecialtyLexicon:
    def __init__(
        self,
        vocabulary: Dict[str, Tuple[str, int]],
        aliases: Dict[str, Tuple[str, bool]] | None = None,
    ):
        # vocabulary: key -> (canonical label, count); aliases: key -> (label, case_sensitive)
        self.vocabulary = vocabulary
        self.aliases = aliases or {}
        self._max_words = max([len(k.split()) for k in list(vocabulary) + list(self.aliases)] or [1])

    @classmethod
    def load(cls, vocabulary_path: Path | str | None = None, alias_path: Path | str | None = None) -> "SpecialtyLexicon":
        return cls(*_load_files(str(vocabulary_path or DEFAULT_VOCABULARY_PATH), str(alias_path or DEFAULT_ALIAS_PATH)))

    # -- normalisation -------------------------------------------------------------------------
    def normalize_label(self, value: str) -> Optional[Tuple[str, str]]:
        """Map free text to a database label. Returns (label, via) or None when it is not a known label."""
        key = normalize_key(value)
        if not key:
            return None
        stripped = (value or "").strip()
        alias = self.aliases.get(key)
        if alias and (not alias[1] or stripped == stripped.upper()):
            return alias[0], "alias"
        hit = self.vocabulary.get(key)
        if hit and key not in LABEL_STOPLIST and not (_is_abbreviation(hit[0]) and stripped != stripped.upper()):
            return hit[0], "vocabulary"
        return None

    def is_known_label(self, value: str) -> bool:
        return self.normalize_label(value) is not None

    # -- scanning ------------------------------------------------------------------------------
    def find(self, text: str, layout: ResumeLayout | None = None) -> List[SpecialtyMatch]:
        """Longest-match scan of the text, plus explicit 'Specialty: X' lines."""
        layout = layout or ResumeLayout(text)
        matches: List[SpecialtyMatch] = []

        for index, (start, end, line, _) in enumerate(layout.lines):
            labeled = _LABELED_FIELD_RE.match(line)
            if labeled:
                value = labeled.group("value").strip(" .;")
                mapped = self.normalize_label(value)
                matches.append(
                    SpecialtyMatch(label=mapped[0] if mapped else value, phrase=value, start=start, end=end,
                                   via="labeled_field", section="labeled", line=line,
                                   line_no=layout.nonempty_ordinal.get(index, 10**6))
                )

        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
        i = 0
        while i < len(tokens):
            found = None
            for width in range(min(self._max_words, len(tokens) - i), 0, -1):
                window = tokens[i : i + width]
                key = " ".join(("and" if t[0] == "&" else t[0].lower()) for t in window)
                span_text = text[window[0][1] : window[-1][2]]
                alias = self.aliases.get(key)
                if alias and (not alias[1] or span_text == span_text.upper()):
                    found = (alias[0], "alias", width, span_text)
                    break
                hit = self.vocabulary.get(key)
                if hit and key not in LABEL_STOPLIST:
                    if _is_abbreviation(hit[0]) and span_text != span_text.upper():
                        continue
                    found = (hit[0], "vocabulary", width, span_text)
                    break
            if found is None:
                i += 1
                continue
            label, via, width, span_text = found
            start, end = tokens[i][1], tokens[i + width - 1][2]
            line, section, ordinal = layout.locate(start)
            matches.append(
                SpecialtyMatch(label=label, phrase=span_text, start=start, end=end, via=via,
                               section=section, line=line, line_no=ordinal)
            )
            i += width
        return matches

    def score(self, match: SpecialtyMatch, text: str) -> float:
        """How strongly the match's position says 'this is the person's specialty'."""
        if match.via == "labeled_field":
            return 10.0
        if match.section in {"references", "education", "heading"}:
            return 0.0
        looks_like_title = (len(match.line) <= 90 and not match.line.rstrip().endswith((".", ";"))
                            and len(match.line.split()) <= 12)
        score = 2.0
        if match.line_no <= 6 and match.section in {"header", "summary"} and looks_like_title:
            score = 7.0  # headline / title block, before any "Skills" or "Experience" heading
        elif match.section == "credentials":
            score = 5.0
        elif match.section in {"header", "summary"}:
            score = 4.0
        elif match.section == "experience":
            score = 4.0 if looks_like_title else 2.0
        elif match.section == "skills":
            score = 1.5
        if self._adjacent_org_word(text, match):
            score *= 0.2
        if _EDUCATION_LINE_RE.search(match.line) and match.line_no > 2:
            score *= 0.25
        if len(match.phrase.split()) >= 2:
            score += 0.5
        return score

    @staticmethod
    def _adjacent_org_word(text: str, match: SpecialtyMatch) -> bool:
        before = re.findall(r"[A-Za-z]+", text[max(0, match.start - 24) : match.start].lower())[-1:]
        after = re.findall(r"[A-Za-z]+", text[match.end : match.end + 24].lower())[:1]
        return bool((before and before[0] in _ORG_WORDS) or (after and after[0] in _ORG_WORDS))

    def decide(self, text: str, extra_candidates: Iterable[Tuple[str, str]] = ()) -> SpecialtyDecision:
        """Pick the specialty the document supports, or say precisely why it cannot.

        `extra_candidates` are (value, evidence) pairs proposed elsewhere (the model). They are weighed
        by where their evidence sits in the document, earn a corroboration bonus when they agree with a
        deterministic match, and never bypass the vocabulary gate.
        """
        layout = ResumeLayout(text)
        labeled = any(m.via == "labeled_field" for m in self.find(text, layout))
        if not labeled and healthcare_signal(text) < 2:
            return SpecialtyDecision(status="none", reason="no_healthcare_context")
        totals: Dict[str, float] = {}
        best: Dict[str, Tuple[float, SpecialtyMatch]] = {}
        occurrences: Dict[str, int] = {}
        for match in self.find(text, layout):
            score = self.score(match, text)
            if score <= 0:
                continue
            seen = occurrences.get(match.label, 0)
            occurrences[match.label] = seen + 1
            totals[match.label] = totals.get(match.label, 0.0) + (score if seen == 0 else min(score, 2.0) * 0.5)
            if match.label not in best or score > best[match.label][0]:
                best[match.label] = (score, match)

        # The model can only corroborate a label the text itself contains; it can never introduce one.
        for value, _evidence in extra_candidates:
            mapped = self.normalize_label(value)
            if mapped and mapped[0] in totals:
                totals[mapped[0]] += 3.0

        def threshold_for(label: str) -> float:
            return MIN_SCORE_WEAK if _is_weak(label) else MIN_SCORE

        ranked = sorted(
            ((label, score) for label, score in totals.items() if score >= threshold_for(label)),
            key=lambda kv: kv[1], reverse=True,
        )
        candidates = [(label, round(score, 2)) for label, score in sorted(totals.items(), key=lambda kv: -kv[1])[:5]]
        if not ranked:
            return SpecialtyDecision("none", reason="specialty_mention_too_weak", candidates=candidates)
        # A practice area ("ICU", "Oncology") is the specialty; a role ("Registered Nurse") is the fallback when the
        # resume names no clear practice area. Two close practice areas are ambiguous, but the role they share is
        # still stated, so it is used rather than guessing between the areas.
        areas = [item for item in ranked if label_kind(item[0]) == "practice_area"]
        roles = [item for item in ranked if label_kind(item[0]) == "profession"]
        reason = ""
        if areas and (len(areas) == 1 or areas[0][1] >= areas[1][1] * 1.4):
            ranked = areas
        elif roles and (len(roles) == 1 or roles[0][1] >= roles[1][1] * 1.4):
            ranked = roles
            reason = "profession_fallback" if areas else ""
        else:
            return SpecialtyDecision("ambiguous", reason="competing_specialties", candidates=candidates)
        top_label, top_score = ranked[0]
        evidence_match = best[top_label][1]
        return SpecialtyDecision(
            status="selected", label=top_label, evidence=evidence_match.line or evidence_match.phrase,
            score=round(top_score, 2), section=evidence_match.section, candidates=candidates, reason=reason,
        )


@lru_cache(maxsize=4)
def _load_files(vocabulary_path: str, alias_path: str):
    vocabulary: Dict[str, Tuple[str, int]] = {}
    path = Path(vocabulary_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = (row.get("specialty") or "").strip()
                try:
                    count = int(row.get("count") or 0)
                except ValueError:
                    count = 0
                key = normalize_key(label)
                if not key or count < MIN_VOCAB_COUNT:
                    continue
                if label.count("(") != label.count(")") or _FACILITY_LABEL_RE.search(label):
                    continue
                if key not in vocabulary or count > vocabulary[key][1]:
                    vocabulary[key] = (label, count)
    # "Otolaryngology (ENT)" is the same label as "Otolaryngology": keep only the plain form when both exist.
    for key, (label, _) in list(vocabulary.items()):
        if "(" in label:
            base = normalize_key(re.sub(r"\([^)]*\)", "", label))
            if base and base != key and base in vocabulary:
                del vocabulary[key]
    aliases: Dict[str, Tuple[str, bool]] = {}
    apath = Path(alias_path)
    if apath.exists():
        with apath.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("reviewed") or "").strip().lower() not in {"yes", "y", "true", "1"}:
                    continue
                key = normalize_key(row.get("phrase") or "")
                label = (row.get("label") or "").strip()
                if key and label:
                    aliases[key] = (label, (row.get("case_sensitive") or "").strip().lower() in {"yes", "y", "true", "1"})
    return vocabulary, aliases
