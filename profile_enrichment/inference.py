"""Structured Qwen inference over a whole resume.

The old contract was "return a dict, and `{}` on any trouble", which made *model down*, *timeout*,
*truncated JSON*, *wrong schema* and *nothing found* look identical. Here every call returns an
:class:`ExtractionResult` whose ``status`` says which one it was:

========================  ======================================================================
``ok``                    valid output for every chunk, >=1 field backed by evidence from the text
``ok_no_fields``          valid output for every chunk, the model reported nothing
``insufficient_evidence`` valid output, but every value it proposed failed evidence verification
``model_unavailable``     could not reach / load the model (after bounded retries)
``timeout``               the server did not answer in time (after bounded retries)
``malformed_json``        output was not parseable JSON (after bounded retries)
``truncated``             generation hit the token limit / the prompt overflowed the context
``schema_invalid``        JSON parsed but did not match the requested schema
========================  ======================================================================

The first three are *successful inference*; the rest are failures and must be retried, never recorded
as "no specialty found". A resume is split into token-budgeted chunks covering every page; each value the
model returns must come with a quote that exists in the chunk it was read from.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from profile_enrichment.config import CONFIG, PROMPT_VERSION
from profile_enrichment.extractors import find_phones
from profile_enrichment.specialty_lexicon import SpecialtyLexicon, normalize_key

logger = logging.getLogger(__name__)

FIELDS = ("first_name", "last_name", "city", "state_code", "specialty", "phone", "email")
_ABBREV_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "district of columbia": "DC", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


class Status(str, Enum):
    OK = "ok"
    OK_NO_FIELDS = "ok_no_fields"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    MALFORMED_JSON = "malformed_json"
    TRUNCATED = "truncated"
    SCHEMA_INVALID = "schema_invalid"


SUCCESS_STATUSES = frozenset({Status.OK, Status.OK_NO_FIELDS, Status.INSUFFICIENT_EVIDENCE})


@dataclass
class FieldCandidate:
    field: str
    value: str
    evidence: str
    page: int
    chunk_index: int
    source: str = "qwen"


@dataclass
class ChunkOutcome:
    index: int
    pages: List[int]
    status: str
    attempts: int = 1
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class ExtractionResult:
    status: Status
    candidates: List[FieldCandidate] = field(default_factory=list)
    rejected: List[Dict[str, str]] = field(default_factory=list)
    chunks: List[ChunkOutcome] = field(default_factory=list)
    chunks_total: int = 0
    error: str = ""
    model_fingerprint: str = ""
    prompt_version: str = PROMPT_VERSION
    seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status in SUCCESS_STATUSES

    @property
    def coverage_complete(self) -> bool:
        return self.chunks_total > 0 and len(self.chunks) == self.chunks_total and all(
            c.status in {"ok"} for c in self.chunks
        )

    @property
    def retryable(self) -> bool:
        return not self.succeeded

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        return cls(
            status=Status(data["status"]),
            candidates=[FieldCandidate(**c) for c in data.get("candidates", [])],
            rejected=list(data.get("rejected", [])),
            chunks=[ChunkOutcome(**c) for c in data.get("chunks", [])],
            chunks_total=int(data.get("chunks_total", 0)),
            error=data.get("error", ""),
            model_fingerprint=data.get("model_fingerprint", ""),
            prompt_version=data.get("prompt_version", ""),
            seconds=float(data.get("seconds", 0.0)),
        )


# ---------------------------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------------------------

@dataclass
class Completion:
    text: str
    finish_reason: str  # "stop" | "length" | other
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BackendError(Exception):
    """kind: 'unavailable' | 'timeout'. `permanent` errors (model missing, bad credentials) are never retried."""

    def __init__(self, kind: str, message: str, permanent: bool = False):
        super().__init__(message)
        self.kind = kind
        self.permanent = permanent


class Backend:
    name = "base"

    def generate(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int,
                 temperature: float, seed: int) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    def fingerprint(self) -> str:
        return f"{self.name}:{CONFIG['local_model_name']}:ctx{CONFIG['ai_context_tokens']}"


def _get_openai_client(base_url: str, api_key: str, timeout: float):
    from openai import OpenAI

    key = (base_url, api_key, timeout)
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = OpenAI(base_url=base_url.rstrip("/") + "/", api_key=api_key, timeout=timeout, max_retries=0)
            _CLIENTS[key] = client
        return client


_CLIENTS: Dict[Tuple[str, str, float], Any] = {}
_CLIENT_LOCK = threading.Lock()


class OpenAICompatBackend(Backend):
    """Any OpenAI-wire-compatible local server. "OpenAI" is only the protocol: nothing leaves the machine."""

    name = "openai"

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        self.base_url = base_url or CONFIG["local_model_base_url"]
        self.model = model or CONFIG["local_model_name"]
        self.api_key = api_key or CONFIG["local_model_api_key"]
        self.timeout = timeout or CONFIG["ai_request_timeout"]
        self._schema_supported = True

    def generate(self, system, user, schema, max_tokens, temperature, seed) -> Completion:
        import openai

        client = _get_openai_client(self.base_url, self.api_key, self.timeout)
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens, seed=seed,
        )
        if self._schema_supported:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "resume_fields", "schema": schema, "strict": True},
            }
        try:
            response = client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise BackendError("timeout", f"timeout after {self.timeout}s: {exc}") from exc
        except openai.NotFoundError as exc:
            raise BackendError("unavailable", f"model/endpoint not found: {exc}", permanent=True) from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise BackendError("unavailable", f"not authorised: {exc}", permanent=True) from exc
        except openai.BadRequestError as exc:
            if self._schema_supported and "response_format" in str(exc).lower():
                logger.warning("MODEL_SCHEMA_OUTPUT_UNSUPPORTED | model=%s | falling back to json_object", self.model)
                self._schema_supported = False
                kwargs["response_format"] = {"type": "json_object"}
                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as exc2:  # noqa: BLE001
                    raise BackendError("unavailable", f"{type(exc2).__name__}: {exc2}") from exc2
            else:
                raise BackendError("unavailable", f"bad request: {exc}", permanent=True) from exc
        except (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError) as exc:
            raise BackendError("unavailable", f"{type(exc).__name__}: {exc}") from exc
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class OllamaBackend(Backend):
    """Ollama's native /api/chat: grammar-constrained `format`, explicit `num_ctx`, and `done_reason`."""

    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float | None = None):
        raw = (base_url or CONFIG["local_model_base_url"] or "http://127.0.0.1:11434").rstrip("/")
        self.base_url = re.sub(r"/v1$", "", raw)
        self.model = model or CONFIG["local_model_name"]
        self.timeout = timeout or CONFIG["ai_request_timeout"]

    def generate(self, system, user, schema, max_tokens, temperature, seed) -> Completion:
        import requests

        payload = {
            "model": self.model, "stream": False, "format": schema, "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed,
                        "num_ctx": CONFIG["ai_context_tokens"]},
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise BackendError("timeout", f"timeout after {self.timeout}s") from exc
        except requests.exceptions.RequestException as exc:
            raise BackendError("unavailable", f"{type(exc).__name__}: {exc}") from exc
        if response.status_code == 404:
            raise BackendError("unavailable", f"model not found: {response.text[:200]}", permanent=True)
        if response.status_code >= 400:
            raise BackendError("unavailable", f"HTTP {response.status_code}: {response.text[:200]}",
                               permanent=response.status_code in (400, 401, 403))
        try:
            body = response.json()
        except ValueError as exc:
            raise BackendError("unavailable", f"non-JSON response: {response.text[:200]}") from exc
        return Completion(
            text=(body.get("message") or {}).get("content", ""),
            finish_reason=body.get("done_reason") or ("stop" if body.get("done") else ""),
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            completion_tokens=int(body.get("eval_count") or 0),
        )


class LlamaCppBackend(Backend):
    """Embedded llama-cpp-python. One shared model, so generation is serialised behind a lock."""

    name = "llama_cpp"
    _lock = threading.Lock()
    _model = None
    _load_error: Optional[Tuple[float, str]] = None

    def fingerprint(self) -> str:
        source = CONFIG["local_model_path"] or CONFIG["local_model_filename"]
        return f"llama_cpp:{os.path.basename(str(source))}:ctx{CONFIG['ai_context_tokens']}:gpu{CONFIG['local_model_n_gpu_layers']}"

    @classmethod
    def _load(cls):
        with cls._lock:
            if cls._model is not None:
                return cls._model
            if cls._load_error and time.time() - cls._load_error[0] < 60:
                raise BackendError("unavailable", cls._load_error[1], permanent=True)
            try:
                from llama_cpp import Llama

                kwargs: Dict[str, Any] = dict(
                    n_ctx=CONFIG["ai_context_tokens"], verbose=False,
                    n_gpu_layers=CONFIG["local_model_n_gpu_layers"],
                )
                if CONFIG["local_model_n_threads"]:
                    kwargs["n_threads"] = CONFIG["local_model_n_threads"]
                path = CONFIG["local_model_path"]
                if path and os.path.exists(path):
                    cls._model = Llama(model_path=path, **kwargs)
                elif CONFIG["local_model_enabled"]:
                    cls._model = Llama.from_pretrained(
                        repo_id=CONFIG["local_model_repo_id"], filename=CONFIG["local_model_filename"],
                        local_dir=CONFIG["local_model_dir"], **kwargs,
                    )
                else:
                    raise ValueError("LOCAL_MODEL_ENABLED is false and LOCAL_MODEL_PATH does not exist")
            except Exception as exc:  # noqa: BLE001 - any load problem means "unavailable", with the reason kept
                cls._load_error = (time.time(), f"failed to load local model: {type(exc).__name__}: {exc}")
                raise BackendError("unavailable", cls._load_error[1], permanent=True) from exc
            return cls._model

    def generate(self, system, user, schema, max_tokens, temperature, seed) -> Completion:
        model = self._load()
        try:
            with self._lock:  # llama.cpp contexts are not safe for concurrent calls
                response = model.create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=temperature, max_tokens=max_tokens, seed=seed,
                    response_format={"type": "json_object", "schema": schema},
                )
        except Exception as exc:  # noqa: BLE001
            raise BackendError("unavailable", f"{type(exc).__name__}: {exc}") from exc
        choice = response["choices"][0]
        usage = response.get("usage") or {}
        return Completion(
            text=(choice.get("message") or {}).get("content") or "",
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


def get_backend() -> Backend:
    """The configured text-model backend. LOCAL_MODEL_BASE_URL set => a server; unset => embedded llama.cpp."""
    kind = CONFIG["local_model_backend"]
    if not CONFIG["local_model_base_url"] and kind != "llama_cpp":
        kind = "llama_cpp"
    if kind == "ollama":
        return OllamaBackend()
    if kind == "llama_cpp":
        return LlamaCppBackend()
    return OpenAICompatBackend()


# ---------------------------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / max(0.5, CONFIG["ai_chars_per_token"])))


@dataclass
class Chunk:
    index: int
    text: str
    pages: List[int]
    segments: List[Tuple[int, int, int]]  # (page, start offset, end offset) into `text`

    def page_at(self, offset: int) -> int:
        for page, start, end in self.segments:
            if start <= offset < end:
                return page
        return self.pages[0] if self.pages else 1


def build_chunks(pages: Sequence[Tuple[int, str]], budget_tokens: int, overlap_tokens: int = 0) -> List[Chunk]:
    """Pack every page, in order, into chunks of at most `budget_tokens` (estimated).

    A line is never cut unless it alone exceeds the budget. Consecutive chunks share up to
    `overlap_tokens` of trailing lines so a fact straddling a boundary is seen whole.
    """
    max_chars = max(200, int(budget_tokens * CONFIG["ai_chars_per_token"]))
    overlap_chars = int(overlap_tokens * CONFIG["ai_chars_per_token"])
    units: List[Tuple[int, str]] = []
    for page, text in pages:
        for raw in (text or "").splitlines():
            line = raw.strip()
            while len(line) > max_chars:  # pathological line (no newlines): hard split on a space
                cut = line.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                units.append((page, line[:cut].strip()))
                line = line[cut:].strip()
            if line:
                units.append((page, line))

    chunks: List[Chunk] = []
    current: List[Tuple[int, str]] = []
    size = 0
    fresh = 0  # lines added since the last flush (carried-over overlap does not count)

    def flush() -> None:
        nonlocal current, size, fresh
        parts: List[str] = []
        segments: List[Tuple[int, int, int]] = []
        offset = 0
        last_page = None
        for page, line in current:
            if page != last_page:
                marker = f"[Page {page}]\n"
                parts.append(marker)
                offset += len(marker)
                last_page = page
            start = offset
            parts.append(line + "\n")
            offset += len(line) + 1
            if segments and segments[-1][0] == page:
                segments[-1] = (page, segments[-1][1], offset)
            else:
                segments.append((page, start, offset))
        chunks.append(Chunk(len(chunks), "".join(parts), sorted({p for p, _ in current}), segments))
        carry: List[Tuple[int, str]] = []
        carried = 0
        for unit in reversed(current):
            if carried + len(unit[1]) + 1 > overlap_chars:
                break
            carry.insert(0, unit)
            carried += len(unit[1]) + 1
        current, size, fresh = carry, carried, 0

    for unit in units:
        cost = len(unit[1]) + 1
        if fresh and size + cost > max_chars:
            flush()
        current.append(unit)
        size += cost
        fresh += 1
    if fresh:
        flush()
    return chunks


# ---------------------------------------------------------------------------------------------
# prompt / schema
# ---------------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You extract facts from resume text. Reply with JSON only, matching the schema. "
    "Copy each value exactly as it is written in the text. Use null whenever a value is not explicitly "
    "written there; never guess and never invent values. Never answer with a template or sample value "
    "(a made-up name, address, phone number or email): if the text does not contain the real value, use null."
)

FIELD_GUIDE = {
    "first_name": (
        "the candidate's own first name, as written in the name line at the top of the resume (not a heading, "
        "job title, employer, school, reference or supervisor)"
    ),
    "last_name": "the candidate's own last name, from the same name line (without credentials such as RN, MD, BSN)",
    "city": "the city of the candidate's own address or location line (not an employer's or school's city)",
    "state_code": "the 2-letter US state code of the candidate's own location",
    "specialty": (
        "the candidate's clinical specialty or practice area, worded as in the resume (the unit, department or "
        "discipline they practise in, preferring their current or most recent role). Not a bare credential "
        "abbreviation, employer, school, degree, software or skill"
    ),
    "phone": "the candidate's own phone number (not a reference's or an employer's)",
    "email": "the candidate's own email address (not an employer's or a reference's)",
}

# Vocabulary labels found in a chunk are offered to the model as a shortlist for the specialty. Offering them
# changes nothing about what is accepted (verify_candidate still requires a taxonomy label found in the text);
# it only stops the model from paraphrasing a label the resume already uses into one that fails the gate.
MAX_SPECIALTY_OPTIONS = 15
_SPECIALTY_OPTIONS_TOKENS = 200


def build_schema(fields: Sequence[str]) -> Dict[str, Any]:
    """Flat `{field: string|null}`. Evidence (snippet + page) is derived from the source text during verification:
    a model's own "quote" proved unreliable and costs a third of the output tokens on CPU."""
    return {
        "type": "object",
        "properties": {name: {"type": ["string", "null"]} for name in fields},
        "required": list(fields),
        "additionalProperties": False,
    }


def specialty_options(text: str, lexicon: SpecialtyLexicon | None) -> List[str]:
    """Distinct vocabulary labels that occur in `text`, in order of first appearance."""
    if lexicon is None:
        return []
    labels: List[str] = []
    room = int(_SPECIALTY_OPTIONS_TOKENS * CONFIG["ai_chars_per_token"]) - 200  # leave room for the instruction
    for match in sorted(lexicon.find(text), key=lambda m: m.start):
        if match.label in labels:
            continue
        room -= len(match.label) + 2
        if room < 0 or len(labels) >= MAX_SPECIALTY_OPTIONS:
            break
        labels.append(match.label)
    return labels


def build_user_prompt(fields: Sequence[str], chunk: Chunk, total_chunks: int, options: Sequence[str] = ()) -> str:
    guide = "\n".join(f'- "{name}": {FIELD_GUIDE[name]}' for name in fields)
    shortlist = ""
    if options and "specialty" in fields:
        shortlist = (
            "\n\nSpecialty labels that appear in this text: " + "; ".join(options) + ". If one of them is the "
            "candidate's own specialty, return it spelled exactly as listed; otherwise return null."
        )
    return (
        f"Extract these fields from part {chunk.index + 1} of {total_chunks} of a resume "
        f"(pages {chunk.pages[0]}-{chunk.pages[-1]}):\n{guide}{shortlist}\n\n"
        "If this part does not state a field, return null for it.\n\nRESUME TEXT:\n" + chunk.text
    )


def output_budget(fields: Sequence[str]) -> int:
    return max(128, 48 + 32 * len(fields))


def input_budget(fields: Sequence[str]) -> int:
    """Largest chunk (in estimated tokens) that still fits the context together with prompt and answer."""
    overhead = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(build_user_prompt(fields, Chunk(0, "", [1], []), 1))
    if "specialty" in fields:
        overhead += _SPECIALTY_OPTIONS_TOKENS
    room =CONFIG["ai_context_tokens"] - output_budget(fields) - overhead - 96
    return max(200, min(CONFIG["ai_chunk_tokens"], room))


# ---------------------------------------------------------------------------------------------
# parsing and verification
# ---------------------------------------------------------------------------------------------

def parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        data = None
        for index, char in enumerate(text):
            if char == "{":
                try:
                    data, _ = decoder.raw_decode(text[index:])
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            raise ValueError("no JSON object in response")
    if not isinstance(data, dict):
        raise ValueError("top-level JSON is not an object")
    return data


def validate_schema(data: Dict[str, Any], fields: Sequence[str]) -> Optional[str]:
    """None when `data` has every requested field as a string or null (and nothing it was not asked for)."""
    for name in fields:
        if name not in data:
            return f"field {name!r} missing"
        if not (data[name] is None or isinstance(data[name], str)):
            return f"field {name!r} is not a string or null"
    extra = [k for k in data if k not in fields]
    if extra:
        return f"unexpected keys {extra[:3]}"
    return None


_WS = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("\u00a0", " ").replace("\u200b", "")).strip().lower()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _squash(text))


def _locate_tokens(text: str, phrase: str) -> int:
    """Offset where `phrase`'s words occur contiguously in `text` (case/punctuation-insensitive), else -1."""
    words = re.findall(r"[A-Za-z0-9]+", phrase)
    if not words:
        return -1
    match = re.search(r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]+".join(re.escape(w) for w in words) + r"(?![A-Za-z0-9])",
                      text, re.IGNORECASE)
    return match.start() if match else -1


def _line_at(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    line = text[start : end if end >= 0 else len(text)].strip()
    return line[:200]


def verify_candidate(
    name: str, value: str, chunk: Chunk, lexicon: SpecialtyLexicon | None
) -> Tuple[Optional[FieldCandidate], str]:
    """Check one model-proposed value against the chunk it came from.

    The value must be findable in the text; the evidence snippet and page are taken from where it was found,
    not from anything the model claims. Returns (candidate, "") or (None, reason).
    """
    value = (value or "").strip()
    if not value:
        return None, "empty"
    text = chunk.text
    position = -1

    if name == "email":
        match = re.search(re.escape(value), text, re.IGNORECASE)
        if not match:
            return None, "value_not_in_source"
        position, value = match.start(), match.group(0)  # keep the capitalisation the resume uses
    elif name == "phone":
        digits = re.sub(r"\D", "", value)[-10:]
        if len(digits) < 10:
            return None, "invalid_phone_length"
        for found, offset in find_phones(text):
            if re.sub(r"\D", "", found)[-10:] == digits:
                position, value = offset, found
                break
        if position < 0:
            return None, "value_not_in_source"
    elif name in {"first_name", "last_name", "city"}:
        position = _locate_tokens(text, value)
        if position < 0:
            return None, "value_not_in_source"
    elif name == "state_code":
        code = value.upper()
        if len(code) != 2:
            code = _ABBREV_STATES.get(_squash(value), "")
            if not code:
                return None, "state_not_two_letter_code"
        hit = re.search(r"(?<![A-Za-z])" + code + r"(?![A-Za-z])", text)
        if hit:
            position = hit.start()
        else:
            name_hits = [(_locate_tokens(text, n), c) for n, c in _ABBREV_STATES.items() if c == code]
            position = next((p for p, _ in name_hits if p >= 0), -1)
        if position < 0:
            return None, "value_not_in_source"
        value = code
    elif name == "specialty":
        mapped = lexicon.normalize_label(value) if lexicon else None
        if not mapped:
            return None, "specialty_not_in_taxonomy"
        label = mapped[0]
        hits = [m for m in lexicon.find(text) if m.label == label]
        if not hits:
            return None, "value_not_in_source"
        position, value = min(hits, key=lambda m: m.start).start, label
    return FieldCandidate(name, value, _line_at(text, position), chunk.page_at(position), chunk.index), ""


def _squash_phone(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------------

_sleep: Callable[[float], None] = time.sleep


def _call_chunk(
    backend: Backend, chunk: Chunk, total: int, fields: Sequence[str], schema: Dict[str, Any],
    options: Sequence[str] = (),
) -> Tuple[ChunkOutcome, Optional[Dict[str, Any]]]:
    outcome = ChunkOutcome(index=chunk.index, pages=chunk.pages, status="")
    max_attempts = CONFIG["ai_max_attempts"]
    max_tokens = output_budget(fields)
    prompt = build_user_prompt(fields, chunk, total, options)
    started = time.time()
    data: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        temperature = 0.0 if attempt == 1 else 0.2
        seed = 1234 + attempt
        try:
            completion = backend.generate(SYSTEM_PROMPT, prompt, schema, max_tokens, temperature, seed)
        except BackendError as exc:
            outcome.status = Status.TIMEOUT.value if exc.kind == "timeout" else Status.MODEL_UNAVAILABLE.value
            outcome.error = str(exc)
            if exc.permanent or attempt == max_attempts:
                break
            _sleep(CONFIG["ai_retry_backoff"] * (2 ** (attempt - 1)) * (0.75 + random.random() / 2))
            continue

        outcome.finish_reason = completion.finish_reason
        outcome.prompt_tokens, outcome.completion_tokens = completion.prompt_tokens, completion.completion_tokens
        overflow = completion.prompt_tokens and completion.prompt_tokens + 8 >= CONFIG["ai_context_tokens"]
        if completion.finish_reason not in ("stop", "") or overflow:
            outcome.status = Status.TRUNCATED.value
            outcome.error = (
                f"finish_reason={completion.finish_reason!r} prompt_tokens={completion.prompt_tokens} "
                f"completion_tokens={completion.completion_tokens} ctx={CONFIG['ai_context_tokens']}"
            )
            max_tokens = min(1024, int(max_tokens * 1.5))
            continue
        try:
            parsed = parse_json_object(completion.text)
        except ValueError as exc:
            outcome.status, outcome.error = Status.MALFORMED_JSON.value, f"{exc}: {completion.text[:120]!r}"
            continue
        problem = validate_schema(parsed, fields)
        if problem:
            outcome.status, outcome.error = Status.SCHEMA_INVALID.value, problem
            continue
        outcome.status, outcome.error, data = "ok", "", parsed
        break

    outcome.seconds = round(time.time() - started, 3)
    return outcome, data


def extract_fields(
    pages: Sequence[Tuple[int, str]],
    fields: Sequence[str] = FIELDS,
    backend: Backend | None = None,
    lexicon: SpecialtyLexicon | None = None,
    max_chunks: int | None = None,
) -> ExtractionResult:
    """Run the model over every chunk of the resume and return a structured, evidence-checked result."""
    started = time.time()
    backend = backend or get_backend()
    lexicon = lexicon or SpecialtyLexicon.load()
    fields = tuple(f for f in fields if f in FIELD_GUIDE)
    result = ExtractionResult(status=Status.OK_NO_FIELDS, model_fingerprint=model_fingerprint(backend))
    chunks = build_chunks(pages, input_budget(fields), CONFIG["ai_chunk_overlap_tokens"])
    limit = max_chunks or CONFIG["ai_max_chunks"]
    result.chunks_total = len(chunks)
    if not chunks:
        result.error = "no text to extract from"
        result.status = Status.OK_NO_FIELDS
        return result
    schema = build_schema(fields)

    proposed = 0
    failure: Optional[ChunkOutcome] = None
    for chunk in chunks[:limit]:
        options = specialty_options(chunk.text, lexicon) if "specialty" in fields else []
        outcome, data = _call_chunk(backend, chunk, len(chunks), fields, schema, options)
        result.chunks.append(outcome)
        if data is None:
            failure = failure or outcome
            if outcome.status == Status.MODEL_UNAVAILABLE.value:
                break  # no point asking a model that is not there for the remaining chunks
            continue
        for name in fields:
            raw = data[name]
            if not raw or not str(raw).strip():
                continue
            proposed += 1
            candidate, reason = verify_candidate(name, str(raw), chunk, lexicon)
            if candidate:
                result.candidates.append(candidate)
            else:
                result.rejected.append({"field": name, "value": str(raw)[:120], "reason": reason, "chunk": str(chunk.index)})

    result.seconds = round(time.time() - started, 3)
    if failure is not None:
        result.status = Status(failure.status)
        result.error = failure.error
    elif result.candidates:
        result.status = Status.OK
    elif proposed:
        result.status = Status.INSUFFICIENT_EVIDENCE
    else:
        result.status = Status.OK_NO_FIELDS
    if not failure and len(chunks) > limit:
        result.error = f"coverage partial: {limit}/{len(chunks)} chunks processed"
    return result


def model_fingerprint(backend: Backend | None = None) -> str:
    backend = backend or get_backend()
    raw = f"{backend.fingerprint()}|max_chunks={CONFIG['ai_max_chunks']}|chunk={CONFIG['ai_chunk_tokens']}"
    return raw


def cache_key(doc_sha256: str, fields: Sequence[str], backend: Backend | None = None) -> str:
    material = "|".join([doc_sha256, model_fingerprint(backend), PROMPT_VERSION, ",".join(sorted(fields))])
    return hashlib.sha256(material.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# readiness check
# ---------------------------------------------------------------------------------------------

CANARY_TEXT = (
    "Priya Ramanathan, RN\n"
    "Columbus, OH\n"
    "Phone: (614) 555-0182 | priya.ramanathan@mailbox.org\n"
    "Specialty: Cardiology\n"
    "Experience\nStaff nurse, telemetry unit, 2016-2024\n"
)


@dataclass
class ReadinessReport:
    ok: bool
    detail: str
    seconds: float = 0.0
    status: str = ""
    fingerprint: str = ""
    deployment: Dict[str, Any] = field(default_factory=dict)


def check_model_ready(backend: Backend | None = None) -> ReadinessReport:
    """Prove the configured model returns a *valid, evidence-backed extraction*, not merely that it answers.

    A known canary resume goes through the same schema/retry/verification path as real work; the model must
    return the canary's name, email and specialty with verifiable quotes.
    """
    started = time.time()
    try:
        backend = backend or get_backend()
        result = extract_fields([(1, CANARY_TEXT)], backend=backend, max_chunks=1)
    except Exception as exc:  # noqa: BLE001
        return ReadinessReport(False, f"{type(exc).__name__}: {exc}", time.time() - started)
    elapsed = time.time() - started
    deployment = describe_deployment(backend)
    if not result.succeeded:
        return ReadinessReport(False, f"{result.status.value}: {result.error}", elapsed, result.status.value,
                               result.model_fingerprint, deployment)
    found = {c.field: c.value for c in result.candidates}
    expected = {"first_name": "Priya", "last_name": "Ramanathan", "email": "priya.ramanathan@mailbox.org",
                "specialty": "Cardiology"}
    missing = {k: (found.get(k), v) for k, v in expected.items() if _squash(found.get(k, "")) != _squash(v)}
    if missing:
        return ReadinessReport(False, f"valid JSON but wrong/missing canary values: {missing}", elapsed,
                               result.status.value, result.model_fingerprint, deployment)
    return ReadinessReport(True, f"canary extraction verified ({len(found)} fields)", elapsed, result.status.value,
                           result.model_fingerprint, deployment)


def describe_deployment(backend: Backend | None = None) -> Dict[str, Any]:
    """What is *actually* serving the model (Ollama reports loaded context and GPU use), not the .env defaults."""
    info: Dict[str, Any] = {
        "configured_backend": CONFIG["local_model_backend"], "configured_model": CONFIG["local_model_name"],
        "base_url": CONFIG["local_model_base_url"], "configured_context": CONFIG["ai_context_tokens"],
    }
    if not CONFIG["local_model_base_url"] or "11434" not in CONFIG["local_model_base_url"]:
        return info
    try:
        import requests

        root = re.sub(r"/v1/?$", "", CONFIG["local_model_base_url"].rstrip("/"))
        loaded = requests.get(f"{root}/api/ps", timeout=5).json().get("models", [])
        for entry in loaded:
            if entry.get("name", "").split(":")[0] == CONFIG["local_model_name"].split(":")[0]:
                info.update(loaded_model=entry.get("name"), loaded_context=entry.get("context_length"),
                            size_bytes=entry.get("size"), size_vram_bytes=entry.get("size_vram"))
        info["ollama_version"] = requests.get(f"{root}/api/version", timeout=5).json().get("version")
    except Exception as exc:  # noqa: BLE001
        info["describe_error"] = f"{type(exc).__name__}: {exc}"
    return info
