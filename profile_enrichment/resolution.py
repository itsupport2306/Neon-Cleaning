"""Decide each field's value from *all* the evidence, instead of letting whichever source ran first win.

Every source (regex over the text, the lexicon scan, Qwen) contributes candidates with a position in the
resume. A candidate is scored by field-specific context (where in the document it sits, what surrounds it,
whether it is consistent with the other fields) and by how many independent sources agree. The best
candidate wins only with a clear margin; a tie between *different* values resolves to nothing - a field left
blank is always better than a coin flip written to production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from profile_enrichment import validation
from profile_enrichment.extractors import HEADER_CHARS, _STATE_CODES, _STATES, find_emails, find_phones, normalize_text
from profile_enrichment.inference import FieldCandidate
from profile_enrichment.neon_db import PROFILE_COLUMNS
from profile_enrichment.specialty_lexicon import ResumeLayout, SpecialtyDecision, SpecialtyLexicon

REVIEW_MIN_SCORE = 4.0  # a correction to a populated field needs strong evidence before a human even sees it
DB_LIMITS = {"first_name": 100, "last_name": 100, "city": 120, "state_code": 2, "specialty": 100, "phone": 30, "email": 255}
_GENERIC_EMAIL_LOCALS = {"hr", "jobs", "careers", "info", "contact", "admin", "support", "recruiting", "recruiter",
                         "careers", "apply", "office", "hello", "mail", "webmaster", "billing", "sales"}
_PHONE_LABEL_RE = re.compile(r"(phone|mobile|cell|tel|telephone|contact|m:|c:|p:|ph:)\s*[:.\-]?\s*$", re.IGNORECASE)
_NAME_LINE_RE = re.compile(
    r"^(?P<name>[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){1,3})(?:\s*,\s*[A-Za-z.]{2,8}(?:\s*,\s*[A-Za-z.]{2,8})*)?$"
)
_CITY_STATE_RE = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z.'\- ]{1,28}?),\s*([A-Z]{2})\b(?!\w)")
_CITY_STATENAME_RE = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z.'\- ]{1,28}?),\s*([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b")


@dataclass
class Resolved:
    field: str
    value: str
    source: str  # regex | qwen | both | lexicon | lexicon+qwen
    evidence: str
    page: int
    score: float
    alternatives: List[str] = field(default_factory=list)


@dataclass
class Resolution:
    resolved: Dict[str, Resolved] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)  # field -> why nothing was resolved
    rejected: List[Dict[str, str]] = field(default_factory=list)  # candidates dropped by validation
    specialty: Optional[SpecialtyDecision] = None


class DocView:
    """The whole resume as one string plus the page each character belongs to."""

    def __init__(self, pages: Sequence[Tuple[int, str]]):
        parts, self.spans, offset = [], [], 0
        for number, text in pages:
            if not text.strip():
                continue
            if parts:
                parts.append("\n\n")
                offset += 2
            parts.append(text)
            self.spans.append((number, offset, offset + len(text)))
            offset += len(text)
        self.text = "".join(parts)
        self.layout = ResumeLayout(self.text)

    def page_at(self, position: int) -> int:
        for number, start, end in self.spans:
            if start <= position < end:
                return number
        return self.spans[0][0] if self.spans else 1

    def line_at(self, position: int) -> str:
        start = self.text.rfind("\n", 0, position) + 1
        end = self.text.find("\n", position)
        return self.text[start : end if end >= 0 else len(self.text)].strip()

    def section_at(self, position: int) -> str:
        return self.layout.locate(position)[1]

    def find_ci(self, needle: str) -> int:
        return self.text.lower().find(needle.lower())


@dataclass
class _Cand:
    value: str
    pos: int
    sources: set
    evidence: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _group(cands: List[_Cand], key=lambda v: v.casefold()) -> List[_Cand]:
    merged: Dict[str, _Cand] = {}
    for cand in cands:
        k = key(cand.value)
        if k in merged:
            merged[k].sources |= cand.sources
            merged[k].pos = min(merged[k].pos, cand.pos)
            merged[k].evidence = merged[k].evidence or cand.evidence
        else:
            merged[k] = _Cand(cand.value, cand.pos, set(cand.sources), cand.evidence, dict(cand.extra))
    return list(merged.values())


def _pick(scored: List[Tuple[float, _Cand]], margin: float = 1.0) -> Tuple[Optional[_Cand], List[str], str]:
    """Best candidate if it beats the runner-up by `margin`; otherwise (None, alternatives, 'conflict')."""
    if not scored:
        return None, [], "no_candidate"
    scored.sort(key=lambda item: (-item[0], item[1].pos))
    best_score, best = scored[0]
    others = [c.value for _, c in scored[1:]]
    if len(scored) > 1 and best_score - scored[1][0] < margin:
        return None, [best.value] + others, "conflicting_candidates"
    return best, others, ""


def _source_label(sources: set) -> str:
    if "qwen" in sources and ("regex" in sources or "lexicon" in sources):
        return "both" if "regex" in sources else "lexicon+qwen"
    return next(iter(sorted(sources)), "regex")


def _evidence_line(view: DocView, cand: _Cand) -> str:
    return cand.evidence or view.line_at(cand.pos)


def _qwen(cands: Dict[str, List[FieldCandidate]], name: str) -> List[FieldCandidate]:
    return cands.get(name, [])


# ---------------------------------------------------------------------------------------------
# per-field resolvers
# ---------------------------------------------------------------------------------------------

def _regex_names(view: DocView) -> List[_Cand]:
    out: List[_Cand] = []
    seen = 0
    for start, _end, line, section in view.layout.lines:
        if not line:
            continue
        seen += 1
        if seen > 8:
            break
        if section == "heading" or "@" in line or any(ch.isdigit() for ch in line):
            continue
        match = _NAME_LINE_RE.match(line)
        if not match:
            continue
        tokens = match.group("name").replace(".", ". ").split()
        tokens = [t.strip() for t in tokens if t.strip(". ")]
        while tokens and tokens[0].lower().strip(".") in {"dr", "mr", "mrs", "ms", "miss", "prof", "rev"}:
            tokens.pop(0)
        if len(tokens) < 2 or validation._check_name(tokens[0]) or validation._check_name(tokens[-1]):
            continue
        out.append(_Cand(f"{tokens[0]}\t{tokens[-1]}", start, {"regex"}, line,
                         {"first": tokens[0], "last": tokens[-1], "ordinal": seen}))
    return out


def resolve_name(view: DocView, qwen: Dict[str, List[FieldCandidate]], need_first: bool, need_last: bool) -> Dict[str, Resolved]:
    cands = _regex_names(view)
    firsts = {c.value.casefold(): c for c in _qwen(qwen, "first_name")}
    lasts = {c.value.casefold(): c for c in _qwen(qwen, "last_name")}
    pairs: List[_Cand] = list(cands)
    for fc in firsts.values():
        for lc in lasts.values():
            pos = view.find_ci(fc.value)
            pairs.append(_Cand(f"{fc.value}\t{lc.value}", max(0, pos), {"qwen"}, fc.evidence,
                               {"first": fc.value, "last": lc.value, "ordinal": 1 + view.text[:max(0, pos)].count("\n")}))
    if not pairs:
        return {}
    # a qwen pair like ("Saskya", "Jean Charles") and a regex pair ("Saskya", "Charles") agree on the first name
    # and overlap on the last: treat the longer, evidence-backed last name as the same person.
    merged: List[_Cand] = []
    for cand in pairs:
        for other in merged:
            if other.extra["first"].casefold() == cand.extra["first"].casefold() and (
                other.extra["last"].casefold().endswith(cand.extra["last"].casefold())
                or cand.extra["last"].casefold().endswith(other.extra["last"].casefold())
            ):
                other.sources |= cand.sources
                if "qwen" in cand.sources:
                    other.extra["last"], other.value = cand.extra["last"], cand.value
                    other.evidence = cand.evidence or other.evidence
                break
        else:
            merged.append(_Cand(cand.value, cand.pos, set(cand.sources), cand.evidence, dict(cand.extra)))
    scored = []
    for cand in merged:
        score = 1.0 + (3.0 if cand.extra["ordinal"] <= 3 else 1.5 if cand.extra["ordinal"] <= 8 else 0.0)
        score += 1.5 * (len(cand.sources) - 1)
        if validation._check_name(cand.extra["first"]) or validation._check_name(cand.extra["last"]):
            continue
        scored.append((score, cand))
    best, alternatives, why = _pick(scored)
    if best is None:
        return {}
    page = view.page_at(best.pos)
    out: Dict[str, Resolved] = {}
    for name, key, needed in (("first_name", "first", need_first), ("last_name", "last", need_last)):
        if needed:
            out[name] = Resolved(name, best.extra[key], _source_label(best.sources), _evidence_line(view, best), page,
                                 round(max(s for s, c in scored if c is best), 2), alternatives[:3])
    return out


def _name_tokens(row: Dict[str, Any], resolved: Dict[str, Resolved]) -> List[str]:
    tokens: List[str] = []
    for name in ("first_name", "last_name"):
        value = resolved[name].value if name in resolved else (row.get(name) or "")
        if not validation.is_blank_or_placeholder(name, value):
            tokens.extend(t.lower() for t in re.findall(r"[A-Za-z]{3,}", str(value)))
    return tokens


def resolve_email(view: DocView, qwen: Dict[str, List[FieldCandidate]], name_tokens: List[str]) -> Tuple[Optional[Resolved], str]:
    cands = [_Cand(v, p, {"regex"}) for v, p in find_emails(view.text)]
    for c in _qwen(qwen, "email"):
        pos = view.find_ci(c.value)
        if pos >= 0:
            cands.append(_Cand(view.text[pos : pos + len(c.value)], pos, {"qwen"}, c.evidence))
    valid = [c for c in cands if not validation._check_email(c.value)]
    scored = []
    for cand in _group(valid):
        local = cand.value.split("@")[0].lower()
        score = 1.0 + (3.0 if cand.pos < HEADER_CHARS else 0.0) + 1.5 * (len(cand.sources) - 1)
        score += 2.0 if any(t in local for t in name_tokens) else 0.0
        score -= 4.0 if re.sub(r"[^a-z]", "", local) in _GENERIC_EMAIL_LOCALS else 0.0
        score -= 2.0 if view.section_at(cand.pos) == "references" else 0.0
        scored.append((score, cand))
    best, alts, why = _pick(scored, margin=0.5)
    if best is None:
        return None, why
    score = max(s for s, c in scored if c is best)
    if score < 1.0:
        return None, "low_confidence"
    return Resolved("email", best.value, _source_label(best.sources), _evidence_line(view, best),
                    view.page_at(best.pos), round(score, 2), alts[:3]), ""


def resolve_phone(view: DocView, qwen: Dict[str, List[FieldCandidate]]) -> Tuple[Optional[Resolved], str]:
    cands = [_Cand(v, p, {"regex"}) for v, p in find_phones(view.text)]
    for c in _qwen(qwen, "phone"):
        digits = re.sub(r"\D", "", c.value)[-10:]
        for value, pos in find_phones(view.text):
            if re.sub(r"\D", "", value)[-10:] == digits:
                cands.append(_Cand(value, pos, {"qwen"}, c.evidence))
                break
    scored = []
    for cand in _group(cands, key=lambda v: re.sub(r"\D", "", v)[-10:]):
        before = view.text[max(0, cand.pos - 16) : cand.pos]
        if re.search(r"\b(fax|npi|license|lic\.?|ssn|dea)\b", before, re.IGNORECASE) or validation._check_phone(cand.value):
            continue
        score = 1.0 + (2.0 if cand.pos < HEADER_CHARS else 0.0) + (2.0 if _PHONE_LABEL_RE.search(before) else 0.0)
        score += 1.5 * (len(cand.sources) - 1)
        score -= 2.0 if view.section_at(cand.pos) == "references" else 0.0
        scored.append((score, cand))
    best, alts, why = _pick(scored, margin=0.5)
    if best is None:
        return None, why
    return Resolved("phone", best.value[:30], _source_label(best.sources), _evidence_line(view, best),
                    view.page_at(best.pos), round(max(s for s, c in scored if c is best), 2), alts[:3]), ""


def resolve_location(view: DocView, qwen: Dict[str, List[FieldCandidate]], needs: Sequence[str]) -> Dict[str, Resolved]:
    """City and state are resolved *together* from one line, so they can never come from different places."""
    cands: List[_Cand] = []
    for match in _CITY_STATE_RE.finditer(view.text[: HEADER_CHARS * 3]):
        if match.group(2) in _STATE_CODES:
            cands.append(_Cand(f"{normalize_text(match.group(1))}\t{match.group(2)}", match.start(), {"regex"}, match.group(0)))
    for match in _CITY_STATENAME_RE.finditer(view.text[: HEADER_CHARS * 3]):
        code = _STATES.get(match.group(2).lower())
        if code:
            cands.append(_Cand(f"{normalize_text(match.group(1))}\t{code}", match.start(), {"regex"}, match.group(0)))
    qwen_cities = {c.value.casefold(): c for c in _qwen(qwen, "city")}
    qwen_states = {c.value.upper(): c for c in _qwen(qwen, "state_code")}
    for cc in qwen_cities.values():
        for sc in qwen_states.values():
            cands.append(_Cand(f"{cc.value}\t{sc.value.upper()}", max(0, view.find_ci(cc.value)), {"qwen"}, cc.evidence or sc.evidence))
    out: Dict[str, Resolved] = {}
    if cands:
        scored = []
        for cand in _group(cands):
            city, _, state = cand.value.partition("\t")
            if validation._check_city(city) or validation._check_state(state):
                continue
            score = 1.0 + (3.0 if cand.pos < HEADER_CHARS else 0.0) + 1.5 * (len(cand.sources) - 1)
            score -= 2.0 if view.section_at(cand.pos) in {"experience", "education", "references"} else 0.0
            scored.append((score, cand))
        best, alts, _ = _pick(scored, margin=1.0)
        if best is not None and max(s for s, c in scored if c is best) >= 2.0:
            city, _, state = best.value.partition("\t")
            score = round(max(s for s, c in scored if c is best), 2)
            page = view.page_at(best.pos)
            if "city" in needs:
                out["city"] = Resolved("city", city, _source_label(best.sources), _evidence_line(view, best), page, score, alts[:3])
            if "state_code" in needs:
                out["state_code"] = Resolved("state_code", state, _source_label(best.sources), _evidence_line(view, best), page, score, alts[:3])
    return out


def resolve_specialty(view: DocView, qwen: Dict[str, List[FieldCandidate]], lexicon: SpecialtyLexicon) -> Tuple[Optional[Resolved], SpecialtyDecision]:
    extra = [(c.value, c.evidence) for c in _qwen(qwen, "specialty")]
    decision = lexicon.decide(view.text, extra)
    if decision.status != "selected" or not decision.label:
        return None, decision
    position = max(0, view.find_ci(decision.evidence)) if decision.evidence else 0
    sources = {"lexicon"} | ({"qwen"} if any(lexicon.normalize_label(v) and lexicon.normalize_label(v)[0] == decision.label
                                              for v, _ in extra) else set())
    return Resolved("specialty", decision.label, "+".join(sorted(sources)), decision.evidence, view.page_at(position),
                    decision.score, [label for label, _ in decision.candidates if label != decision.label][:3]), decision


# ---------------------------------------------------------------------------------------------
# orchestration + proposal
# ---------------------------------------------------------------------------------------------

def _blank(row: Dict[str, Any], name: str) -> bool:
    if name == "phone":
        return row.get(name) is None
    if name == "specialty":
        return row.get(name) is None or not normalize_text(row.get(name))
    return validation.is_blank_or_placeholder(name, row.get(name))


def needed_fields(row: Dict[str, Any]) -> List[str]:
    return [name for name in PROFILE_COLUMNS if _blank(row, name)]


def resolve_fields(
    view: DocView, row: Dict[str, Any], qwen_candidates: Sequence[FieldCandidate], lexicon: SpecialtyLexicon,
    needed: Optional[Sequence[str]] = None,
) -> Resolution:
    # Resolved for every column (cheap, deterministic); only blank ones become fills, the rest feed the review queue.
    needs = list(needed) if needed is not None else list(PROFILE_COLUMNS)
    by_field: Dict[str, List[FieldCandidate]] = {}
    for cand in qwen_candidates:
        by_field.setdefault(cand.field, []).append(cand)
    result = Resolution()

    if "specialty" in needs:
        res, decision = resolve_specialty(view, by_field, lexicon)
        result.specialty = decision
        if res:
            result.resolved["specialty"] = res
        else:
            result.notes["specialty"] = decision.reason or decision.status
    if "first_name" in needs or "last_name" in needs:
        result.resolved.update(resolve_name(view, by_field, "first_name" in needs, "last_name" in needs))
        for name in ("first_name", "last_name"):
            if name in needs and name not in result.resolved:
                result.notes[name] = "no_reliable_name"
    if "city" in needs or "state_code" in needs:
        located = resolve_location(view, by_field, needs)
        result.resolved.update(located)
        for name in ("city", "state_code"):
            if name in needs and name not in result.resolved:
                result.notes[name] = "no_reliable_location"
    if "email" in needs:
        res, why = resolve_email(view, by_field, _name_tokens(row, result.resolved))
        if res:
            result.resolved["email"] = res
        else:
            result.notes["email"] = why
    if "phone" in needs:
        res, why = resolve_phone(view, by_field)
        if res:
            result.resolved["phone"] = res
        else:
            result.notes["phone"] = why

    # shape, placeholder and length gates, then grounding in the full text (specialty has its own vocabulary gate)
    for name, res in list(result.resolved.items()):
        reason = ""
        if name == "specialty":
            if res.source.startswith("lexicon") and validation._is_null_token(res.value):
                reason = "null_token"
            elif "labeled" in (result.specialty.section if result.specialty else "") and not lexicon.is_known_label(res.value):
                reason = validation._check_specialty(res.value) or ""
        else:
            ok, canonical, why = validation.validate_field(name, res.value, view.text, require_grounding=True)
            if ok:
                res.value = canonical
            else:
                reason = why
        if not reason and len(res.value) > DB_LIMITS[name]:
            reason = "exceeds_column_length"
        if reason:
            result.rejected.append({"field": name, "value": res.value, "reason": reason})
            del result.resolved[name]
            result.notes[name] = f"validation:{reason}"
    return result


def build_proposal(row: Dict[str, Any], resolution: Resolution) -> Tuple[Dict[str, Resolved], Dict[str, Resolved]]:
    """(fills, reviews): blank fields to fill now; populated fields the resume disagrees with (never auto-applied)."""
    fills: Dict[str, Resolved] = {}
    reviews: Dict[str, Resolved] = {}
    for name, res in resolution.resolved.items():
        current = row.get(name)
        if _blank(row, name):
            fills[name] = res
        elif (name != "phone" and res.score >= REVIEW_MIN_SCORE
              and normalize_text(current).casefold() != normalize_text(res.value).casefold()):
            reviews[name] = res
    return fills, reviews
