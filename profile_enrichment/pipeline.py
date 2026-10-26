"""The per-profile stages for the NULL-specialty job, split so each can run on its own bounded worker pool.

    fetch_stage   : exact-id row read -> resume reference -> download (byte cache) -> text/OCR (content-hash cache)
    infer_stage   : deterministic candidates -> Qwen only where needed (validated-result cache) -> resolution
    write_stage   : guarded UPDATE ... RETURNING -> independent read-back -> outcome

Each stage sets ``work.outcome`` when the task is finished early (for example ``no_resume``); the runner
persists every outcome through the job store, so nothing is ever silently dropped.

Outcome vocabulary (see job_store): the outcome describes the *objective* - filling ``specialty``. A resume
that gave a verified name but no specialty is ``insufficient_evidence`` (its extra field writes are listed in
``fields_written``), so it stays visible to ``--retry-unresolved`` instead of looking finished.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from profile_enrichment import inference, job_store as js
from profile_enrichment.config import CONFIG, PROMPT_VERSION
from profile_enrichment.downloaders import DownloadError, fetch_bytes
from profile_enrichment.extractors import (
    EXTRACTOR_VERSION, DocumentContent, PageText, consume_ocr_seconds, extract_document, sniff_extension,
)
from profile_enrichment.neon_db import FieldChange, PROFILE_COLUMNS, fetch_profile_by_id, guarded_update
from profile_enrichment.resolution import DocView, Resolution, build_proposal, needed_fields, resolve_fields
from profile_enrichment.specialty_lexicon import SpecialtyLexicon, healthcare_signal

logger = logging.getLogger("profile_enrichment")

AI_SCREEN_REASON = "AI"
EVIDENCE_REASONS = {"value_not_in_source", "value_not_in_evidence", "specialty_not_in_taxonomy", "empty", "no_evidence_quote"}
# Specialty chosen deterministically with at least this score needs no model call in `auto` mode.
AUTO_SKIP_MODEL_SCORE = 7.0


@dataclass
class Work:
    run_id: str
    profile_id: str
    resume_ref: Optional[str]
    owner: str
    invocation: str
    row: Optional[Dict[str, Any]] = None
    doc: Optional[DocumentContent] = None
    doc_sha: str = ""
    extraction: Optional[inference.ExtractionResult] = None
    resolution: Optional[Resolution] = None
    proposal: Optional[Dict[str, Any]] = None
    outcome: Optional[str] = None
    reason: str = ""
    fields_written: List[str] = field(default_factory=list)
    screen_reason_set: bool = False
    timings: Dict[str, float] = field(default_factory=dict)
    qwen_status: str = ""
    cache_hits: List[str] = field(default_factory=list)

    def finish_early(self, outcome: str, reason: str) -> "Work":
        self.outcome, self.reason = outcome, reason
        return self


class DocumentCache:
    """Downloaded bytes on disk keyed by content hash, with a size cap (oldest files go first)."""

    def __init__(self, store: js.JobStore, directory: str | Path | None = None, max_bytes: int | None = None):
        self.store = store
        self.dir = Path(directory or Path(CONFIG["cache_dir"]) / "docs")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = CONFIG["doc_cache_max_bytes"] if max_bytes is None else max_bytes
        self._puts = 0
        self._lock = threading.Lock()

    def get(self, url: str) -> Optional[bytes]:
        row = self.store.get_url(url)
        if not row:
            return None
        path = self.dir / f"{row['sha256']}.bin"
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            return None
        return data

    def put(self, url: str, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        if self.max_bytes:
            path = self.dir / f"{sha}.bin"
            if not path.exists():
                tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            self.store.put_url(url, sha, len(data))
            with self._lock:
                self._puts += 1
                if self._puts % 200 == 0:
                    self._evict()
        return sha

    def _evict(self) -> None:
        files = [(f.stat().st_mtime, f.stat().st_size, f) for f in self.dir.glob("*.bin")]
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files):
            if total <= self.max_bytes * 0.9:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                pass


def _doc_to_pages(doc: DocumentContent) -> List[Dict[str, Any]]:
    return [{"n": p.number, "t": p.text, "s": p.source} for p in doc.pages]


def _pages_to_doc(ext: str, sha: str, pages: List[Dict[str, Any]]) -> DocumentContent:
    return DocumentContent(pages=[PageText(number=p["n"], text=p["t"], source=p["s"]) for p in pages], extension=ext, sha256=sha)


class Pipeline:
    def __init__(self, store: js.JobStore, *, llm_mode: str = "auto", ocr_enabled: bool = True,
                 refresh_documents: bool = False, backend: inference.Backend | None = None,
                 lexicon: SpecialtyLexicon | None = None, model_gate: threading.Semaphore | None = None,
                 max_chunks: int | None = None):
        assert llm_mode in {"auto", "always", "never"}
        self.store = store
        self.llm_mode = llm_mode
        self.ocr_enabled = ocr_enabled
        self.refresh_documents = refresh_documents
        self.backend = backend
        self._lexicon = lexicon
        self.model_gate = model_gate or threading.Semaphore(1)
        self.max_chunks = max_chunks
        self.cache = DocumentCache(store)

    # -- identity of "the model + prompt" used to decide whether an old verdict is stale --------------
    @property
    def lexicon(self) -> SpecialtyLexicon:
        if self._lexicon is None:
            self._lexicon = SpecialtyLexicon.load()
        return self._lexicon

    def fingerprint(self) -> str:
        backend = self.backend or inference.get_backend()
        base = inference.model_fingerprint(backend)
        return f"{base}|llm={self.llm_mode}|extractor={EXTRACTOR_VERSION}"

    # -- timing helper -------------------------------------------------------------------------
    def _time(self, work: Work, stage: str, started: float) -> None:
        seconds = time.monotonic() - started
        work.timings[stage] = work.timings.get(stage, 0.0) + seconds
        try:
            self.store.record_timing(work.run_id, work.profile_id, stage, seconds)
        except Exception:  # noqa: BLE001 - timing must never fail a task
            pass

    # -- stage 1 -------------------------------------------------------------------------------
    def fetch_stage(self, work: Work) -> Work:
        started = time.monotonic()
        try:
            row = fetch_profile_by_id(work.profile_id)
        except Exception as exc:  # noqa: BLE001
            return work.finish_early(js.UPDATE_FAILED, f"db_read_failed: {type(exc).__name__}: {exc}")
        finally:
            self._time(work, "db_read", started)
        if row is None:
            return work.finish_early(js.CONCURRENT_CHANGE, "row_missing: profile no longer exists")
        if row.get("specialty") is not None and str(row.get("specialty")).strip():
            return work.finish_early(js.CONCURRENT_CHANGE, f"specialty_already_filled: {row['specialty']!r}")
        work.row = row

        ref = (row.get("resume_url") or work.resume_ref or "").strip()
        if not ref:
            return work.finish_early(js.NO_RESUME, "resume_url is empty")

        started = time.monotonic()
        data = None if self.refresh_documents else self.cache.get(ref)
        if data is not None:
            work.cache_hits.append("bytes")
        else:
            try:
                data = fetch_bytes(ref)
            except DownloadError as exc:
                self._time(work, "download", started)
                if exc.kind == "not_found":
                    return work.finish_early(js.NO_RESUME, f"document_not_found: {exc}")
                return work.finish_early(js.DOWNLOAD_FAILED, f"{exc.kind}: {exc}")
            except Exception as exc:  # noqa: BLE001
                self._time(work, "download", started)
                return work.finish_early(js.DOWNLOAD_FAILED, f"{type(exc).__name__}: {exc}")
            self.cache.put(ref, data)
        self._time(work, "download", started)
        work.doc_sha = hashlib.sha256(data).hexdigest()

        ext = sniff_extension(data, ref)
        started = time.monotonic()
        cached = None if self.refresh_documents else self.store.get_doc(work.doc_sha, EXTRACTOR_VERSION)
        if cached:
            work.doc = _pages_to_doc(cached[0], work.doc_sha, cached[1])
            work.cache_hits.append("text")
        else:
            consume_ocr_seconds()
            try:
                work.doc = extract_document(data, ext, ocr_enabled=self.ocr_enabled)
            except ValueError as exc:
                self._time(work, "extract", started)
                return work.finish_early(js.EXTRACTION_FAILED, f"unsupported_document_type: {exc}")
            except Exception as exc:  # noqa: BLE001
                self._time(work, "extract", started)
                return work.finish_early(js.EXTRACTION_FAILED, f"{type(exc).__name__}: {exc}")
            ocr_seconds = consume_ocr_seconds()
            if ocr_seconds:
                work.timings["ocr"] = ocr_seconds
                self.store.record_timing(work.run_id, work.profile_id, "ocr", ocr_seconds)
            work.doc.sha256 = work.doc_sha
        self._time(work, "extract", started)

        problem = work.doc.failure_reason()
        if problem:
            # Retryable by design: an unreadable or only partly OCR'd resume says nothing about the specialty.
            return work.finish_early(js.EXTRACTION_FAILED, problem[:400])
        if not cached:
            self.store.put_doc(work.doc_sha, EXTRACTOR_VERSION, ext, _doc_to_pages(work.doc))
        if not work.doc.text.strip():
            return work.finish_early(js.INSUFFICIENT_EVIDENCE, "empty_document: no text on any page (OCR complete)")
        return work

    # -- stage 2 -------------------------------------------------------------------------------
    def _model_fields(self, row: Dict[str, Any]) -> List[str]:
        return [f for f in needed_fields(row) if f in inference.FIELDS]

    def _model_needed(self, work: Work, view: DocView) -> bool:
        if self.llm_mode == "never":
            return False
        if self.llm_mode == "always":
            return True
        if healthcare_signal(view.text) < 2 and "specialty: " not in view.text.lower():
            # Not a clinical resume: the specialty vocabulary cannot match, so the model has nothing to add for the
            # specialty. Other blank fields are filled from the regex/lexicon pass; no model time is spent here.
            return False
        decision = self.lexicon.decide(view.text)
        if decision.status != "selected" or decision.score < AUTO_SKIP_MODEL_SCORE:
            return True
        # specialty is settled deterministically; ask the model only if other blank fields have no regex answer
        deterministic = resolve_fields(view, work.row, [], self.lexicon).resolved
        return any(f not in deterministic for f in self._model_fields(work.row) if f != "specialty")

    def infer_stage(self, work: Work) -> Work:
        view = DocView(work.doc.page_texts)
        fields = self._model_fields(work.row)
        candidates: List[inference.FieldCandidate] = []

        started = time.monotonic()
        use_model = self._model_needed(work, view)
        self._time(work, "deterministic", started)

        if use_model:
            backend = self.backend or inference.get_backend()
            key = inference.cache_key(work.doc_sha, fields + [f"mode={self.llm_mode}"], backend)
            cached = self.store.get_inference(key)
            started = time.monotonic()
            if cached:
                result = inference.ExtractionResult.from_dict(cached)
                work.cache_hits.append("inference")
            else:
                with self.model_gate:
                    result = inference.extract_fields(work.doc.page_texts, fields, backend=backend,
                                                      lexicon=self.lexicon, max_chunks=self.max_chunks)
                if result.succeeded:  # only validated successes are reusable; failures must be retried
                    self.store.put_inference(key, result.to_dict())
            self._time(work, "qwen", started)
            work.extraction, work.qwen_status = result, result.status.value
            if not result.succeeded:
                return work.finish_early(js.INFERENCE_FAILED, f"{result.status.value}: {result.error}"[:400])
            candidates = result.candidates

        started = time.monotonic()
        work.resolution = resolve_fields(view, work.row, candidates, self.lexicon)
        fills, reviews = build_proposal(work.row, work.resolution)
        self._time(work, "validate", started)
        work.proposal = self._proposal(work, fills, reviews, use_model)
        self._classify(work, fills)
        return work

    def _proposal(self, work: Work, fills, reviews, used_model: bool) -> Dict[str, Any]:
        def encode(items) -> Dict[str, Any]:
            return {
                name: {"value": r.value, "old": work.row.get(name), "evidence": r.evidence[:300], "page": r.page,
                       "source": r.source, "score": r.score, "alternatives": r.alternatives}
                for name, r in items.items()
            }

        extraction = work.extraction
        ai_success = bool(
            used_model and extraction is not None and extraction.succeeded and extraction.coverage_complete
            and extraction.status in (inference.Status.OK, inference.Status.OK_NO_FIELDS, inference.Status.INSUFFICIENT_EVIDENCE)
        )
        return {
            "fields": encode(fills),
            "reviews": encode(reviews),
            "ai_success": ai_success,
            "qwen_status": work.qwen_status or ("not_called" if not used_model else ""),
            "baseline_screen_reason": work.row.get("screen_reason"),
            "notes": dict(work.resolution.notes),
            "rejected": list(work.resolution.rejected) + (extraction.rejected if extraction else []),
            "specialty_candidates": (work.resolution.specialty.candidates if work.resolution.specialty else []),
            "coverage": ("complete" if not extraction or extraction.coverage_complete else extraction.error),
            "doc_sha256": work.doc_sha,
        }

    def _classify(self, work: Work, fills) -> None:
        """Record why the specialty stays NULL (when it does). The verdict is kept in the proposal so that other
        verified fields can still be written, while the task outcome remains about the specialty."""
        proposal = work.proposal
        if "specialty" not in fills:
            all_rejected = [r for r in proposal["rejected"] if r.get("field") == "specialty"]
            # Not literally in the resume / not a known label = the resume does not establish it (insufficient
            # evidence). Only a candidate that failed a *validation rule* (credential, degree, prose, placeholder)
            # counts as validation_failed.
            rejected_specialty = [r for r in all_rejected if r.get("reason") not in EVIDENCE_REASONS]
            note = proposal["notes"].get("specialty", "no_specialty_found")
            taxonomy_misses = [r["value"] for r in rejected_specialty if r.get("reason") == "specialty_not_in_taxonomy"]
            if proposal["coverage"] != "complete":
                verdict = (js.INSUFFICIENT_EVIDENCE, f"{proposal['coverage']}; {note}")
            elif str(note).startswith("validation:"):
                verdict = (js.VALIDATION_FAILED, str(note))
            elif taxonomy_misses:
                verdict = (js.INSUFFICIENT_EVIDENCE, f"specialty_not_in_taxonomy: {taxonomy_misses[:3]}")
            elif rejected_specialty:
                verdict = (js.VALIDATION_FAILED, f"rejected: {[(r['value'], r['reason']) for r in rejected_specialty][:3]}")
            else:
                cands = proposal.get("specialty_candidates") or []
                verdict = (js.INSUFFICIENT_EVIDENCE, f"{note}" + (f" {cands[:3]}" if cands else ""))
            proposal["unresolved"] = {"outcome": verdict[0], "reason": verdict[1][:400]}
            if not proposal["fields"]:
                work.outcome, work.reason = verdict[0], verdict[1][:400]

    # -- stage 3 -------------------------------------------------------------------------------
    def write_stage(self, work: Work, apply: bool) -> Work:
        proposal = work.proposal or {}
        fields = proposal.get("fields", {})
        if not fields:
            return work
        if not apply:
            return work  # runner stores the proposal as 'prepared'
        return self.apply_proposal(work)

    def apply_proposal(self, work: Work) -> Work:
        proposal = work.proposal
        fields = proposal["fields"]
        changes = {name: FieldChange(old=item["old"], new=item["value"]) for name, item in fields.items()}
        screen_reason = AI_SCREEN_REASON if (
            proposal.get("ai_success") and str(proposal.get("baseline_screen_reason") or "").upper() != AI_SCREEN_REASON
        ) else None
        started = time.monotonic()
        try:
            result = guarded_update(work.profile_id, changes, screen_reason=screen_reason)
        except Exception as exc:  # noqa: BLE001
            self._time(work, "db_write", started)
            work.outcome, work.reason = js.UPDATE_FAILED, f"{type(exc).__name__}: {exc}"[:400]
            return work
        self._time(work, "db_write", started)

        work.fields_written = list(result.applied)
        work.screen_reason_set = result.screen_reason_set
        for name in result.applied:
            item = fields[name]
            self.store.record_write(work.run_id, work.profile_id, name, item["old"], item["value"], item["evidence"],
                                    item["page"], item["source"])
        if result.email_conflict:
            proposal.setdefault("notes", {})["email"] = "email_conflict: address belongs to another profile"

        if not result.row_found:
            return work.finish_early(js.CONCURRENT_CHANGE, "row_missing_at_write")
        if result.mismatches:
            return work.finish_early(js.UPDATE_FAILED, f"read_back_mismatch: {result.mismatches}"[:400])
        if "specialty" in fields:
            if "specialty" in result.applied:
                work.outcome, work.reason = js.VERIFIED_UPDATED, "specialty written and verified by read-back"
            elif "specialty" in result.already_present:
                work.outcome, work.reason = js.VERIFIED_UPDATED, "specialty already held the same value (verified by read-back)"
            elif "specialty" in result.concurrent_change:
                work.outcome, work.reason = js.CONCURRENT_CHANGE, (
                    f"specialty changed by another writer: {result.concurrent_change['specialty']!r}")
            else:
                work.outcome, work.reason = js.UPDATE_FAILED, "specialty not persisted"
        else:
            verdict = proposal.get("unresolved") or {"outcome": js.INSUFFICIENT_EVIDENCE, "reason": "no_specialty_found"}
            work.outcome, work.reason = verdict["outcome"], verdict["reason"]
        return work
