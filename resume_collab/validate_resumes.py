"""Validate stored profile data (first name, last name, specialty, address) against the resume. READ-ONLY.

    SELECT ... FROM profiles WHERE specialty IS NOT NULL AND resume_url IS NOT NULL

Nothing is ever written to the database: findings go to three CSV files, one row per profile_id, appended as
each profile finishes (see resume_collab/README.md):

    resume_validation_discrepancies.csv     a field is explicitly contradicted by the resume
    resume_validation_no_discrepancies.csv  readable resume, nothing contradicted: for manual review
    resume_validation_errors.csv            could not be validated (download, unreadable, model, nothing verifiable)

Commands (python -m resume_collab.validate_resumes <command>): count, preflight, run (default), status.
It reuses profile_enrichment for download, text/OCR extraction, the Qwen backend, config and the DB pool.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from profile_enrichment import inference, neon_db
from profile_enrichment.config import CONFIG, ROOT_DIR
from profile_enrichment.downloaders import DownloadError, fetch_bytes
from profile_enrichment.extractors import extract_document, sniff_extension
from profile_enrichment.inference import BackendError, build_chunks, estimate_tokens, parse_json_object
from profile_enrichment.logging_utils import setup_logger
from profile_enrichment.specialty_lexicon import SpecialtyLexicon

from resume_collab import compare as cmp
from resume_collab.csv_store import (DISCREPANCY_FILE, ERROR_FILE, NO_DISCREPANCY_FILE, ResultStore,
                                     duplicates)

logger = setup_logger()

EXIT_OK, EXIT_INCOMPLETE, EXIT_PREFLIGHT = 0, 2, 3
PROMPT_VERSION = "resume-validation.2026-10-29.v1"
OUTPUT_TOKENS = 600
CONFIDENCE_ORDER = ("low", "medium", "high")
DEFAULT_OUT_DIR = ROOT_DIR / "resume_collab" / "output"

PROFILE_COLUMNS = ("profile_id", "resume_url", "first_name", "last_name", "specialty", "profession_type",
                   "city", "state_code", "zip_code")
WHERE_ELIGIBLE = "specialty IS NOT NULL AND resume_url IS NOT NULL"


def default_out_dir() -> Path:
    return Path(os.getenv("OUTPUT_DIR") or DEFAULT_OUT_DIR)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------------------------
# database (SELECT only, inside READ ONLY transactions)
# ---------------------------------------------------------------------------------------------

def _select_sql() -> str:
    cols = ", ".join(f'"{c}"' for c in PROFILE_COLUMNS)
    return f"SELECT {cols} FROM {neon_db._table_sql()} WHERE {WHERE_ELIGIBLE}"


def count_eligible() -> int:
    def run() -> int:
        with neon_db.transaction(read_only=True, timeout_ms=600_000) as conn, conn.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {neon_db._table_sql()} WHERE {WHERE_ELIGIBLE}")
            return int(cursor.fetchone()[0])

    return neon_db._retry_connection_errors(run)


def iter_eligible(page_size: int = 2000, profile_ids: Optional[Sequence[str]] = None) -> Iterator[List[Dict[str, Any]]]:
    """Eligible profiles in `profile_id` order (keyset paging: an interrupted scan restarts cheaply)."""
    if profile_ids is not None:
        ids = list(profile_ids)
        for offset in range(0, len(ids), 1000):
            def run(chunk=ids[offset: offset + 1000]) -> List[Dict[str, Any]]:
                with neon_db.transaction(read_only=True) as conn, conn.cursor() as cursor:
                    cursor.execute(f'{_select_sql()} AND "profile_id" = ANY(%s) ORDER BY "profile_id"', (chunk,))
                    return neon_db._rows(cursor)

            page = neon_db._retry_connection_errors(run)
            if page:
                yield page
        return

    last: Optional[str] = None
    while True:
        def run(after=last) -> List[Dict[str, Any]]:
            with neon_db.transaction(read_only=True, timeout_ms=120_000) as conn, conn.cursor() as cursor:
                if after is None:
                    cursor.execute(f'{_select_sql()} ORDER BY "profile_id" LIMIT %s', (page_size,))
                else:
                    cursor.execute(f'{_select_sql()} AND "profile_id" > %s ORDER BY "profile_id" LIMIT %s',
                                   (after, page_size))
                return neon_db._rows(cursor)

        page = neon_db._retry_connection_errors(run)
        if not page:
            return
        last = str(page[-1]["profile_id"])
        yield page
        if len(page) < page_size:
            return


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You read resume text and report what it explicitly says. Reply with JSON only, matching the schema. "
    "Copy names, addresses and specialties exactly as written. Use null (or an empty list) whenever the text "
    "does not state something; never guess and never invent values. Never answer with a template or sample "
    "value (a made-up name or address)."
)

VERDICT_GUIDE = (
    '"specialty_verdict": is the STORED specialty below supported by this resume? "supported" = the resume shows '
    "the candidate practises, works in or trained in it - accept equivalent names and common abbreviations "
    '(e.g. ICU = Intensive Care, OB = Obstetrics, ER = Emergency); "not_supported" = the resume clearly shows a '
    'different or unrelated profession/specialty; "unclear" = this text does not say enough to decide.'
)


def build_schema() -> Dict[str, Any]:
    text = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "first_name": text, "last_name": text, "street": text, "city": text, "state": text, "zip": text,
            "profession": text,
            "specialties": {"type": "array", "items": {"type": "string"}},
            "specialty_verdict": {"type": "string", "enum": list(cmp.VERDICTS)},
            "specialty_reason": {"type": "string"},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_ORDER)},
        },
        "required": ["first_name", "last_name", "street", "city", "state", "zip", "profession", "specialties",
                     "specialty_verdict", "specialty_reason", "confidence"],
        "additionalProperties": False,
    }


def build_prompt(profile: Dict[str, Any], index: int, total: int, pages: Sequence[int], text: str) -> str:
    stored = (profile.get("specialty") or "").strip()
    kind = (profile.get("profession_type") or "").strip()
    return (
        f"Part {index + 1} of {total} of a resume (pages {pages[0]}-{pages[-1]}). Fill these fields:\n"
        '- "first_name", "last_name": the candidate\'s own name from the name line at the top (not a reference, '
        "supervisor, employer or school; without credentials such as RN, MD)\n"
        '- "street", "city", "state", "zip": the candidate\'s own home/mailing address as written (not an '
        "employer's or school's); null for any part that is not written\n"
        '- "profession": the candidate\'s profession as the resume shows it (e.g. Registered Nurse, Physician, '
        "Physical Therapist)\n"
        '- "specialties": the clinical specialties / practice areas the resume supports, worded as in the resume\n'
        f"- {VERDICT_GUIDE}\n"
        '- "specialty_reason": one short sentence explaining the verdict\n'
        '- "confidence": "high", "medium" or "low" for your answers on this part\n\n'
        f"STORED specialty: {stored!r}" + (f" (stored profession type: {kind!r})" if kind else "") +
        "\nIf this part does not state a field, return null (or an empty list).\n\nRESUME TEXT:\n" + text
    )


def _chunk_budget() -> int:
    overhead = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(
        build_prompt({"specialty": "x" * 40}, 0, 1, [1], "")) + 60
    room = CONFIG["ai_context_tokens"] - OUTPUT_TOKENS - overhead - 96
    return max(200, min(CONFIG["ai_chunk_tokens"], room))


class ModelFailure(Exception):
    """category: model_error (unreachable / timeout) | invalid_model_output (not JSON, truncated, wrong schema)."""

    def __init__(self, category: str, detail: str):
        super().__init__(detail)
        self.category, self.detail = category, detail


def check_output(data: Dict[str, Any]) -> Optional[str]:
    for name in ("first_name", "last_name", "street", "city", "state", "zip", "profession"):
        if name not in data or not (data[name] is None or isinstance(data[name], str)):
            return f"field {name!r} missing or not a string/null"
    if not isinstance(data.get("specialties"), list) or not all(isinstance(s, str) for s in data["specialties"]):
        return "field 'specialties' is not a list of strings"
    if data.get("specialty_verdict") not in cmp.VERDICTS:
        return f"specialty_verdict {data.get('specialty_verdict')!r} is not one of {cmp.VERDICTS}"
    if not isinstance(data.get("specialty_reason"), str):
        return "field 'specialty_reason' is not a string"
    if data.get("confidence") not in CONFIDENCE_ORDER:
        return f"confidence {data.get('confidence')!r} is not one of {CONFIDENCE_ORDER}"
    return None


def call_model(backend: inference.Backend, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Same retry conventions as profile_enrichment.inference: temperature 0 first, seeded retries with backoff."""
    attempts, max_tokens = CONFIG["ai_max_attempts"], OUTPUT_TOKENS
    failure = ModelFailure("model_error", "no attempt was made")
    for attempt in range(1, attempts + 1):
        try:
            completion = backend.generate(SYSTEM_PROMPT, prompt, schema, max_tokens,
                                          0.0 if attempt == 1 else 0.2, 1234 + attempt)
        except BackendError as exc:
            failure = ModelFailure("model_error", f"{exc.kind}: {exc}")
            if exc.permanent or attempt == attempts:
                raise failure
            time.sleep(CONFIG["ai_retry_backoff"] * 2 ** (attempt - 1))
            continue
        overflow = completion.prompt_tokens and completion.prompt_tokens + 8 >= CONFIG["ai_context_tokens"]
        if completion.finish_reason not in ("stop", "") or overflow:
            failure = ModelFailure("invalid_model_output", f"truncated: finish_reason={completion.finish_reason!r} "
                                   f"prompt_tokens={completion.prompt_tokens} ctx={CONFIG['ai_context_tokens']}")
            max_tokens = min(1500, int(max_tokens * 1.5))
            continue
        try:
            data = parse_json_object(completion.text)
        except ValueError as exc:
            failure = ModelFailure("invalid_model_output", f"{exc}: {completion.text[:120]!r}")
            continue
        problem = check_output(data)
        if problem:
            failure = ModelFailure("invalid_model_output", problem)
            continue
        return data
    raise failure


@dataclass
class ResumeReading:
    """What the resume says, merged over its chunks, with every quoted value checked against the resume text."""
    first_name: str = ""
    last_name: str = ""
    address: Dict[str, str] = field(default_factory=dict)
    profession: str = ""
    specialties: List[str] = field(default_factory=list)
    verdict: str = "unclear"
    reason: str = ""
    confidence: str = "high"
    chunks_read: int = 0
    chunks_total: int = 0


def read_resume(profile: Dict[str, Any], pages: Sequence[Tuple[int, str]], backend: inference.Backend) -> ResumeReading:
    full_text = "\n".join(t for _, t in pages)
    chunks = build_chunks(pages, _chunk_budget(), CONFIG["ai_chunk_overlap_tokens"])
    reading = ResumeReading(chunks_total=len(chunks))
    schema = build_schema()
    verdicts: List[str] = []
    reasons: List[str] = []
    for chunk in chunks[:CONFIG["ai_max_chunks"]]:
        data = call_model(backend, build_prompt(profile, chunk.index, len(chunks), chunk.pages, chunk.text), schema)
        reading.chunks_read += 1
        if not reading.first_name and cmp.appears_in(data["first_name"], full_text):
            reading.first_name = data["first_name"].strip()
        if not reading.last_name and cmp.appears_in(data["last_name"], full_text):
            reading.last_name = data["last_name"].strip()
        for key in ("street", "city", "state", "zip"):
            if key not in reading.address and cmp.appears_in(data[key], full_text):
                reading.address[key] = data[key].strip()
        if not reading.profession and cmp.appears_in(data["profession"], full_text):
            reading.profession = data["profession"].strip()
        for item in data["specialties"]:
            if cmp.appears_in(item, full_text) and item.strip() not in reading.specialties:
                reading.specialties.append(item.strip())
        verdicts.append(data["specialty_verdict"])
        if data["specialty_reason"].strip():
            reasons.append(data["specialty_reason"].strip())
        if CONFIDENCE_ORDER.index(data["confidence"]) < CONFIDENCE_ORDER.index(reading.confidence):
            reading.confidence = data["confidence"]
    reading.verdict = ("supported" if "supported" in verdicts
                       else "unclear" if "unclear" in verdicts else "not_supported")
    reading.reason = " ".join(dict.fromkeys(reasons))[:400]
    return reading


# ---------------------------------------------------------------------------------------------
# one profile
# ---------------------------------------------------------------------------------------------

@dataclass
class Outcome:
    kind: str  # discrepancy | no_discrepancy | error
    row: Dict[str, Any]
    category: str = ""


def _error(profile: Dict[str, Any], category: str, detail: str, retryable: bool,
           status: str = "failed") -> Outcome:
    row = {"profile_id": profile["profile_id"], "resume_url": profile.get("resume_url"), "validation_status": status,
           "error_category": category, "error_detail": detail[:500], "retryable": "yes" if retryable else "no",
           "validated_at": now()}
    return Outcome("error", row, category)


def evaluate(profile: Dict[str, Any], reading: ResumeReading, text: str, lexicon: Optional[SpecialtyLexicon]) -> Outcome:
    first = cmp.compare_name(profile.get("first_name"), reading.first_name, "first_name")
    last = cmp.compare_name(profile.get("last_name"), reading.last_name, "last_name")
    if first.status == last.status == cmp.MISMATCH and cmp.detect_swap(
            profile.get("first_name") or "", profile.get("last_name") or "", reading.first_name, reading.last_name):
        first.note = last.note = (f"names look swapped: stored {first.db_value!r} {last.db_value!r}, "
                                  f"resume says {first.resume_value!r} {last.resume_value!r}")

    verdict, reason = reading.verdict, reading.reason
    if verdict == "not_supported" and (reading.confidence == "low" or reading.chunks_read < reading.chunks_total):
        verdict = "unclear"
        reason += " (not reported: low model confidence or only part of the resume was read)"
    specialty = cmp.compare_specialty(profile.get("specialty"), reading.specialties, reading.profession,
                                      verdict, reason, text, lexicon)
    address = cmp.compare_address(
        {"city": profile.get("city"), "state": profile.get("state_code"), "zip": profile.get("zip_code")},
        reading.address)
    results = {"first_name": first, "last_name": last, "specialty": specialty, "address": address}

    base = {"profile_id": profile["profile_id"], "resume_url": profile.get("resume_url"),
            "model_confidence": reading.confidence, "field_statuses": cmp.field_statuses(results),
            "validated_at": now()}
    status = cmp.overall_status(results)
    unverified = [n for n in cmp.FIELDS if results[n].status == cmp.NOT_VERIFIABLE]
    if status == "discrepancy":
        bad = cmp.mismatched(results)
        row = {**base, "first_name_db": first.db_value, "first_name_resume": first.resume_value,
               "last_name_db": last.db_value, "last_name_resume": last.resume_value,
               "specialty_db": specialty.db_value, "specialty_resume": specialty.resume_value,
               "address_db": address.db_value, "address_resume": address.resume_value,
               "mismatched_fields": ";".join(bad), "discrepancy_reason": cmp.reasons(results, bad),
               "review_status": "pending_review", "validation_status": "discrepancy_found"}
        return Outcome("discrepancy", row)
    if status == "no_discrepancy":
        why = "Resume readable and no field contradicted; profile left unchanged for manual review."
        if unverified:
            why += " Not verifiable from the resume: " + ", ".join(unverified) + "."
        return Outcome("no_discrepancy", {**base, "validation_status": "no_discrepancy_found", "review_reason": why})
    error = _error(profile, "unverifiable", "no field could be verified from the resume. " +
                   cmp.reasons(results, cmp.FIELDS), retryable=False, status="unverifiable")
    error.row["model_confidence"] = reading.confidence
    return error


def validate_profile(profile: Dict[str, Any], backend: inference.Backend, lexicon: Optional[SpecialtyLexicon],
                     model_gate: threading.Semaphore, ocr_enabled: bool = True) -> Outcome:
    ref = (profile.get("resume_url") or "").strip()
    if not ref:
        return _error(profile, "invalid_resume_url", "resume_url is empty", retryable=False)
    try:
        data = fetch_bytes(ref)
    except DownloadError as exc:
        if exc.kind == "not_found":
            return _error(profile, "resume_not_found", str(exc), retryable=False)
        if str(exc).startswith("resume reference is not a downloadable URL"):
            return _error(profile, "invalid_resume_url", str(exc), retryable=False)
        return _error(profile, "download_failed", f"{exc.kind}: {exc}", retryable=exc.kind == "transient")
    except Exception as exc:  # noqa: BLE001
        return _error(profile, "download_failed", f"{type(exc).__name__}: {exc}", retryable=True)

    try:
        doc = extract_document(data, sniff_extension(data, ref), ocr_enabled=ocr_enabled)
    except ValueError as exc:
        return _error(profile, "unsupported_document", str(exc), retryable=False)
    except Exception as exc:  # noqa: BLE001
        return _error(profile, "extraction_failed", f"{type(exc).__name__}: {exc}", retryable=True)
    problem = doc.failure_reason()
    if problem:
        return _error(profile, "extraction_failed", problem, retryable=True)
    if not doc.text.strip():
        return _error(profile, "empty_resume", "no text on any page (OCR complete)", retryable=False)

    try:
        with model_gate:
            reading = read_resume(profile, doc.page_texts, backend)
    except ModelFailure as exc:
        return _error(profile, exc.category, exc.detail, retryable=True)
    return evaluate(profile, reading, doc.text, lexicon)


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------

@dataclass
class Progress:
    total: Optional[int]
    started: float = field(default_factory=time.monotonic)
    counts: Dict[str, int] = field(default_factory=lambda: {"discrepancy": 0, "no_discrepancy": 0, "error": 0})
    skipped: int = 0
    last_print: float = 0.0

    @property
    def processed(self) -> int:
        return sum(self.counts.values())

    def line(self, prior_done: int) -> str:
        elapsed = max(1e-6, time.monotonic() - self.started)
        rate = self.processed / elapsed
        text = (f"processed {self.processed} this run | discrepancy={self.counts['discrepancy']} "
                f"no_discrepancy={self.counts['no_discrepancy']} error={self.counts['error']} | "
                f"{rate * 60:.1f}/min")
        if self.total:
            remaining = max(0, self.total - prior_done - self.processed)
            text += f" | remaining ~{remaining}" + (f" (~{remaining / rate / 3600:.1f} h)" if rate else "")
        return text


def load_ids(args) -> Optional[List[str]]:
    ids: List[str] = []
    if args.profile_ids:
        ids += [p.strip() for p in args.profile_ids.split(",") if p.strip()]
    if args.profile_ids_file:
        ids += [ln.strip() for ln in Path(args.profile_ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return list(dict.fromkeys(ids)) or None


def cmd_run(args) -> int:
    out_dir = Path(args.out_dir)
    if not args.skip_preflight:
        report = preflight(check_count=False)
        if not report["ok"]:
            print_preflight(report)
            return EXIT_PREFLIGHT
    store = ResultStore(out_dir, retry_errors=args.retry_errors)
    prior = len(store.done)
    total: Optional[int] = None
    if not args.skip_count and not args.limit and load_ids(args) is None:
        total = count_eligible()
    print(f"output folder: {out_dir.resolve()}\nalready recorded: {prior} profile(s)"
          + (f" | eligible in database: {total}" if total is not None else ""), flush=True)

    backend, lexicon = inference.get_backend(), SpecialtyLexicon.load()
    gate = threading.Semaphore(max(1, args.model_concurrency))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: (print("\nstop requested: finishing in-flight profiles ...", flush=True), stop.set()))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    progress = Progress(total)
    pending: Dict[Future, Dict[str, Any]] = {}
    submitted = 0

    def harvest(block: bool) -> None:
        done, _ = wait(list(pending), timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for future in done:
            profile = pending.pop(future)
            try:
                outcome = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad resume must never stop the run
                logger.exception("UNEXPECTED_ERROR | profile_id=%s", profile["profile_id"])
                outcome = _error(profile, "unexpected_error", f"{type(exc).__name__}: {exc}", retryable=True)
            record = {"discrepancy": store.record_discrepancy, "no_discrepancy": store.record_no_discrepancy,
                      "error": store.record_error}[outcome.kind]
            if record(outcome.row):
                progress.counts[outcome.kind] += 1
            logger.info("VALIDATED | profile_id=%s | result=%s | %s", profile["profile_id"], outcome.kind,
                        outcome.category or outcome.row.get("mismatched_fields", ""))
        if time.monotonic() - progress.last_print > 30 or (done and progress.processed % 10 == 0):
            progress.last_print = time.monotonic()
            print(progress.line(prior), flush=True)

    limit_reached = False
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for page in iter_eligible(args.page_size, load_ids(args)):
                for profile in page:
                    if stop.is_set() or (args.limit and submitted >= args.limit):
                        limit_reached = bool(args.limit and submitted >= args.limit)
                        break
                    if store.is_done(str(profile["profile_id"])):
                        progress.skipped += 1
                        continue
                    profile["profile_id"] = str(profile["profile_id"])
                    pending[pool.submit(validate_profile, profile, backend, lexicon, gate, not args.no_ocr)] = profile
                    submitted += 1
                    while len(pending) >= args.workers * 2:
                        harvest(block=True)
                else:
                    continue
                break
            while pending:
                harvest(block=True)
    finally:
        neon_db.close_pool()

    print(progress.line(prior), flush=True)
    dup = duplicates(store.dir / n for n in (DISCREPANCY_FILE, NO_DISCREPANCY_FILE, ERROR_FILE))
    print(f"skipped (already recorded): {progress.skipped} | duplicate ids across files: {len(dup)}")
    if stop.is_set():
        print("stopped early: run the same command again to continue where it left off.")
        return EXIT_INCOMPLETE
    print("limit reached." if limit_reached else "all eligible profiles have been processed.")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# count / preflight / status
# ---------------------------------------------------------------------------------------------

def preflight(check_count: bool = True, require_ocr: bool = False) -> Dict[str, Any]:
    from profile_enrichment import extractors

    report: Dict[str, Any] = {"ok": True, "problems": [], "warnings": [], "selection": f"{_select_sql()}"}
    try:
        if check_count:
            report["eligible_profiles"] = count_eligible()
        else:
            with neon_db.transaction(read_only=True) as conn, conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            report["database"] = "reachable"
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"database: {type(exc).__name__}: {exc}")
    ready = inference.check_model_ready()
    report["text_model"] = {"ok": ready.ok, "detail": ready.detail, "seconds": round(ready.seconds, 1),
                            "fingerprint": ready.fingerprint, "deployment": ready.deployment}
    if not ready.ok:
        report["problems"].append(f"text model not ready: {ready.detail}")
    vision_ok, vision_detail = extractors.check_vision_ready()
    tess_ok, tess_detail = extractors.check_tesseract_ready()
    report["ocr"] = {"vision": {"ok": vision_ok, "detail": vision_detail}, "tesseract": {"ok": tess_ok, "detail": tess_detail}}
    if not (vision_ok or tess_ok):
        (report["problems"] if require_ocr else report["warnings"]).append(
            "no working OCR engine: scanned resumes will be logged as extraction_failed")
    report["ok"] = not report["problems"]
    return report


def print_preflight(report: Dict[str, Any]) -> None:
    print("\n=== preflight ===")
    print(json.dumps({k: v for k, v in report.items() if k not in {"problems", "warnings"}}, indent=2, default=str))
    for item in report["warnings"]:
        print(f"WARNING: {item}")
    for item in report["problems"]:
        print(f"PROBLEM: {item}")
    print("preflight:", "OK" if report["ok"] else "FAILED")


def cmd_count(_args) -> int:
    print(f"{_select_sql()}\n-> {count_eligible()} eligible profile(s)")
    neon_db.close_pool()
    return EXIT_OK


def cmd_preflight(args) -> int:
    report = preflight(check_count=not args.skip_count, require_ocr=args.require_ocr)
    print_preflight(report)
    neon_db.close_pool()
    return EXIT_OK if report["ok"] else EXIT_PREFLIGHT


def cmd_status(args) -> int:
    store = ResultStore(args.out_dir)
    counts = store.counts()
    print(f"folder: {Path(args.out_dir).resolve()}")
    for name, value in counts.items():
        print(f"  {name}: {value}")
    print(f"  total profiles recorded: {sum(counts.values())}")
    dup = duplicates(store.dir / n for n in counts)
    print(f"  duplicate ids across files: {len(dup)}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Validate eligible profiles and append results to the CSV files (read-only).")
    run.add_argument("--limit", type=int, default=None, help="Stop after this many NEW profiles (test batch).")
    run.add_argument("--profile-ids", help="Comma-separated profile_ids (must still be eligible).")
    run.add_argument("--profile-ids-file", help="File with one profile_id per line.")
    run.add_argument("--out-dir", default=str(default_out_dir()), help="Folder for the CSV files (env OUTPUT_DIR).")
    run.add_argument("--workers", type=int, default=CONFIG["worker_count"], help="Download/extract/OCR concurrency.")
    run.add_argument("--model-concurrency", type=int, default=2, help="Concurrent model requests.")
    run.add_argument("--page-size", type=int, default=2000, help="Profiles fetched per database page.")
    run.add_argument("--retry-errors", action="store_true",
                     help="Archive the errors CSV and validate those profiles again.")
    run.add_argument("--no-ocr", action="store_true", help="Disable OCR (scanned resumes are logged as errors).")
    run.add_argument("--skip-count", action="store_true", help="Do not count eligible rows first (faster start).")
    run.add_argument("--skip-preflight", action="store_true", help="Skip readiness checks (not recommended).")
    pre = sub.add_parser("preflight", help="Check the database, the Qwen text model and OCR.")
    pre.add_argument("--skip-count", action="store_true"); pre.add_argument("--require-ocr", action="store_true")
    sub.add_parser("count", help="Count profiles with specialty IS NOT NULL AND resume_url IS NOT NULL.")
    st = sub.add_parser("status", help="Row counts of the three CSV files.")
    st.add_argument("--out-dir", default=str(default_out_dir()))

    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0].startswith("-"):
        raw = ["run"] + raw
    args = parser.parse_args(raw)
    return {"run": cmd_run, "preflight": cmd_preflight, "count": cmd_count, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
