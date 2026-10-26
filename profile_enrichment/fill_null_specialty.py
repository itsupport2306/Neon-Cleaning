"""Fill `profiles.specialty` (and other blank, evidence-backed fields) for every row where specialty IS NULL.

    SELECT * FROM public.profiles WHERE specialty IS NULL;      -- no date window, no hidden filter

Safe by default: ``run`` only *prepares* proposals (download -> text/OCR -> Qwen -> validation) and writes
nothing; add ``--apply`` to write. Every selected profile_id is stored in a durable manifest (SQLite) and ends
in exactly one outcome:

    verified_updated      specialty written and confirmed by an independent read-back
    no_resume             no resume reference, or the file no longer exists (HTTP 404/410)
    download_failed       network / server error fetching the resume            (retryable)
    extraction_failed     unreadable document, or OCR could not be completed    (retryable)
    inference_failed      Qwen unavailable / timeout / malformed / truncated     (retryable)
    validation_failed     a specialty candidate existed but failed validation
    insufficient_evidence the resume does not establish a specialty (or only a non-taxonomy phrase)
    concurrent_change     the row changed after it was read; nothing was overwritten (retryable)
    update_failed         database error or read-back mismatch                   (retryable)
    pending               not attempted yet (or prepared and waiting for --apply)

Commands (python -m profile_enrichment.fill_null_specialty <command>): run (default), resume, apply, status,
preflight, export-review, apply-review, export-unresolved, rollback, refresh-vocabulary.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from profile_enrichment import inference, job_store as js
from profile_enrichment.config import CONFIG, PROMPT_VERSION
from profile_enrichment.logging_utils import setup_logger
from profile_enrichment.pipeline import Pipeline, Work

logger = setup_logger()

EXIT_OK, EXIT_FAILURES, EXIT_INCOMPLETE, EXIT_PREFLIGHT = 0, 1, 2, 3
_SENTINEL = object()


# ---------------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------------

def build_manifest(store: js.JobStore, run_id: str, limit: Optional[int], profile_ids: Optional[Sequence[str]]) -> int:
    """Persist the selected ids once. The manifest - not the live NULL count - defines the run from then on.

    Idempotent (INSERT OR IGNORE): an interrupted build is simply started again.
    """
    from profile_enrichment import neon_db

    sample = limit if (limit is not None and not profile_ids) else None
    for page in neon_db.iter_null_specialty_pages(batch_size=5000, profile_ids=profile_ids, sample_size=sample):
        if limit is not None:
            page = page[: max(0, limit - store.total_tasks(run_id))]
        if page:
            store.add_tasks(run_id, [(r["profile_id"], r["resume_url"]) for r in page])
            logger.info("MANIFEST_PROGRESS | run=%s | ids=%s", run_id, store.total_tasks(run_id))
        if limit is not None and store.total_tasks(run_id) >= limit:
            break
    store.finish_manifest(run_id)
    return store.total_tasks(run_id)


# ---------------------------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------------------------

def preflight(llm_mode: str, require_ocr: bool, check_count: bool = True) -> Dict[str, Any]:
    from profile_enrichment import extractors, neon_db

    report: Dict[str, Any] = {"ok": True, "problems": [], "warnings": []}
    try:
        report["profile_id_unique_index"] = neon_db.verify_profile_id_unique()
        if not report["profile_id_unique_index"]:
            report["problems"].append("profiles.profile_id has no single-column unique index: one-row updates are not safe")
        if check_count:
            report["db_null_specialty_count"] = neon_db.count_null_specialty()
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"database: {type(exc).__name__}: {exc}")

    if llm_mode != "never":
        ready = inference.check_model_ready()
        report["text_model"] = {"ok": ready.ok, "detail": ready.detail, "seconds": round(ready.seconds, 1),
                                "fingerprint": ready.fingerprint, "deployment": ready.deployment}
        if not ready.ok:
            report["problems"].append(f"text model not ready: {ready.detail}")
        loaded = ready.deployment.get("loaded_context")
        if loaded and loaded < CONFIG["ai_context_tokens"]:
            report["warnings"].append(
                f"server loaded context is {loaded} tokens but AI_CONTEXT_TOKENS={CONFIG['ai_context_tokens']}: "
                "long chunks would be silently cut by the server")
        if ready.deployment.get("size_vram_bytes") == 0:
            report["warnings"].append("model is running on CPU (size_vram=0): expect minutes, not seconds, per resume")
    else:
        report["text_model"] = {"ok": True, "detail": "skipped (llm-mode never)"}

    vision_ok, vision_detail = extractors.check_vision_ready()
    tess_ok, tess_detail = extractors.check_tesseract_ready()
    report["ocr"] = {"vision": {"ok": vision_ok, "detail": vision_detail}, "tesseract": {"ok": tess_ok, "detail": tess_detail}}
    if not (vision_ok or tess_ok):
        message = "no working OCR engine: scanned resumes will end as extraction_failed (retryable)"
        (report["problems"] if require_ocr else report["warnings"]).append(message)
    elif not vision_ok:
        report["warnings"].append(f"vision OCR unavailable ({vision_detail}); Tesseract will be used")
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


# ---------------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------------

class Runner:
    """Three bounded stages (fetch/extract, model, database) joined by bounded queues.

    Backpressure: a fetch worker blocks on `put` when the model queue is full, so it stops *claiming* tasks;
    nothing piles up in memory and no lease is taken for work that cannot start soon.
    """

    def __init__(self, store: js.JobStore, pipeline: Pipeline, run_id: str, *, apply: bool, fetch_workers: int,
                 model_workers: int, write_workers: int, limit: Optional[int], prepared_only: bool = False,
                 profile_ids: Optional[Sequence[str]] = None):
        self.store, self.pipeline, self.run_id, self.apply = store, pipeline, run_id, apply
        self.fetch_workers, self.model_workers, self.write_workers = fetch_workers, model_workers, write_workers
        self.limit, self.prepared_only, self.profile_ids = limit, prepared_only, profile_ids
        self.invocation = uuid.uuid4().hex[:10]
        self.owner = f"{os.getpid()}-{self.invocation}"
        self.stop = threading.Event()
        self.q_model: "queue.Queue[Any]" = queue.Queue(maxsize=max(2, model_workers * 2))
        self.q_write: "queue.Queue[Any]" = queue.Queue(maxsize=max(4, write_workers * 4))
        self._budget = limit
        self._lock = threading.Lock()
        self.processed = 0
        self.fingerprint = pipeline.fingerprint()
        self.started = time.time()

    # -- helpers -------------------------------------------------------------------------------
    def _take_budget(self) -> bool:
        with self._lock:
            if self._budget is None:
                return True
            if self._budget <= 0:
                return False
            self._budget -= 1
            return True

    def _put(self, q: "queue.Queue[Any]", item: Any) -> bool:
        while not self.stop.is_set():
            try:
                q.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _beat(self, work: Work, stage: str) -> bool:
        try:
            self.store.heartbeat(work.run_id, work.profile_id, work.owner, CONFIG["lease_seconds"], stage)
            return True
        except js.LeaseLost:
            logger.warning("LEASE_LOST | profile_id=%s | stage=%s | result discarded", work.profile_id, stage)
            return False

    def _finish(self, work: Work) -> None:
        proposal = work.proposal
        try:
            self.store.finish(
                work.run_id, work.profile_id, work.owner, work.outcome or js.PENDING, work.reason,
                invocation=self.invocation, doc_sha256=work.doc_sha or None, model_fp=self.fingerprint,
                prompt_version=PROMPT_VERSION, qwen_status=work.qwen_status or None, fields_written=work.fields_written,
                screen_reason_set=work.screen_reason_set, proposal=proposal,
            )
            if proposal:
                for name, item in (proposal.get("reviews") or {}).items():
                    self.store.add_proposal(work.run_id, work.profile_id, name, item["old"], item["value"],
                                            item["evidence"], item["page"], item["source"])
        except js.LeaseLost:
            logger.warning("LEASE_LOST | profile_id=%s | outcome=%s discarded", work.profile_id, work.outcome)
            return
        with self._lock:
            self.processed += 1
            done = self.processed
        logger.info("TASK_DONE | profile_id=%s | outcome=%s | reason=%s | fields_written=%s | qwen=%s | timings=%s",
                    work.profile_id, work.outcome, (work.reason or "")[:120], work.fields_written,
                    work.qwen_status or "-", {k: round(v, 1) for k, v in work.timings.items()})
        if done % 25 == 0:
            logger.info("PROGRESS | run=%s | done_this_invocation=%s | counts=%s", work.run_id, done,
                        {k: v for k, v in self.store.outcome_counts(work.run_id).items() if v})

    def _prepared(self, work: Work) -> None:
        try:
            self.store.mark_prepared(work.run_id, work.profile_id, work.owner, work.proposal, invocation=self.invocation,
                                     doc_sha256=work.doc_sha or None, model_fp=self.fingerprint,
                                     prompt_version=PROMPT_VERSION, qwen_status=work.qwen_status or None)
            for name, item in (work.proposal.get("reviews") or {}).items():
                self.store.add_proposal(work.run_id, work.profile_id, name, item["old"], item["value"],
                                        item["evidence"], item["page"], item["source"])
        except js.LeaseLost:
            logger.warning("LEASE_LOST | profile_id=%s | prepared proposal discarded", work.profile_id)
            return
        with self._lock:
            self.processed += 1
        spec = (work.proposal["fields"].get("specialty") or {}).get("value")
        logger.info("TASK_PREPARED | profile_id=%s | specialty=%s | other_fields=%s | qwen=%s | timings=%s",
                    work.profile_id, spec, [f for f in work.proposal["fields"] if f != "specialty"],
                    work.qwen_status or "-", {k: round(v, 1) for k, v in work.timings.items()})

    # -- workers -------------------------------------------------------------------------------
    def _fetch_worker(self) -> None:
        states = (js.STATE_PREPARED,) if self.prepared_only else (
            (js.STATE_PENDING, js.STATE_PREPARED) if self.apply else (js.STATE_PENDING,))
        while not self.stop.is_set():
            if not self._take_budget():
                return
            tasks = self.store.claim(self.run_id, self.owner, CONFIG["lease_seconds"], 1, states, self.profile_ids)
            if not tasks:
                with self._lock:  # nothing left to claim: give the unused budget back and stop
                    if self._budget is not None:
                        self._budget += 1
                return
            task = tasks[0]
            work = Work(self.run_id, task.profile_id, task.resume_ref, self.owner, self.invocation)
            try:
                if task.proposal is not None and self.apply:
                    work.proposal, work.doc_sha = task.proposal, task.doc_sha256 or ""
                    work.qwen_status = task.proposal.get("qwen_status", "")
                    if not self._put(self.q_write, work):
                        return
                    continue
                if not self._beat(work, "fetch"):
                    continue
                self.pipeline.fetch_stage(work)
            except Exception as exc:  # noqa: BLE001 - one bad resume must never kill the run
                logger.exception("FETCH_STAGE_ERROR | profile_id=%s", task.profile_id)
                work.finish_early(js.EXTRACTION_FAILED, f"unexpected: {type(exc).__name__}: {exc}"[:400])
            if work.outcome is not None:
                self._finish(work)
            elif not self._put(self.q_model, work):
                self.store.release(self.run_id, work.profile_id, self.owner)
                return

    def _model_worker(self) -> None:
        while True:
            work = self.q_model.get()
            if work is _SENTINEL:
                return
            try:
                if self._beat(work, "infer"):
                    self.pipeline.infer_stage(work)
                else:
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("INFER_STAGE_ERROR | profile_id=%s", work.profile_id)
                work.finish_early(js.INFERENCE_FAILED, f"unexpected: {type(exc).__name__}: {exc}"[:400])
            if work.outcome is not None and not (work.proposal or {}).get("fields"):
                self._finish(work)
            elif not self._put(self.q_write, work):
                self.store.release(self.run_id, work.profile_id, self.owner)

    def _write_worker(self) -> None:
        while True:
            work = self.q_write.get()
            if work is _SENTINEL:
                return
            try:
                if not self._beat(work, "write"):
                    continue
                if self.apply:
                    self.pipeline.apply_proposal(work)
                    self._finish(work)
                else:
                    self._prepared(work)
            except Exception as exc:  # noqa: BLE001
                logger.exception("WRITE_STAGE_ERROR | profile_id=%s", work.profile_id)
                work.finish_early(js.UPDATE_FAILED, f"unexpected: {type(exc).__name__}: {exc}"[:400])
                self._finish(work)

    # -- run -----------------------------------------------------------------------------------
    def run(self) -> None:
        def on_signal(signum, _frame):
            if self.stop.is_set():
                raise KeyboardInterrupt
            logger.warning("SHUTDOWN_REQUESTED | finishing in-flight work, releasing unclaimed leases (Ctrl-C again to abort)")
            self.stop.set()

        previous = signal.signal(signal.SIGINT, on_signal) if threading.current_thread() is threading.main_thread() else None
        fetchers = [threading.Thread(target=self._fetch_worker, name=f"fetch-{i}", daemon=True) for i in range(self.fetch_workers)]
        modelers = [threading.Thread(target=self._model_worker, name=f"model-{i}", daemon=True) for i in range(self.model_workers)]
        writers = [threading.Thread(target=self._write_worker, name=f"write-{i}", daemon=True) for i in range(self.write_workers)]
        try:
            for thread in fetchers + modelers + writers:
                thread.start()
            for thread in fetchers:
                thread.join()
            for _ in modelers:
                self.q_model.put(_SENTINEL)
            for thread in modelers:
                thread.join()
            for _ in writers:
                self.q_write.put(_SENTINEL)
            for thread in writers:
                thread.join()
        finally:
            self.stop.set()
            released = self.store.release_owner(self.run_id, self.owner)
            if released:
                logger.warning("LEASES_RELEASED | count=%s", released)
            if previous is not None:
                signal.signal(signal.SIGINT, previous)


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------

def exit_status(store: js.JobStore, run_id: str) -> int:
    counts = store.outcome_counts(run_id)
    run = store.get_run(run_id)
    dry = run["mode"] == "dry-run"
    failures = sum(counts[o] for o in js.FAILURE_OUTCOMES)
    # a row that simply was filled by someone else in the meantime is not a failure
    benign = sum(1 for t in store.tasks_with_outcome(run_id, [js.CONCURRENT_CHANGE]) if t["reason"].startswith("specialty_already_filled"))
    failures -= benign
    unfinished = counts[js.PENDING] + counts["claimed"] + (0 if dry else counts["prepared"])
    if not run["manifest_complete"] or unfinished:
        return EXIT_INCOMPLETE
    return EXIT_FAILURES if failures else EXIT_OK


def build_report(store: js.JobStore, run_id: str, invocation: Optional[str] = None) -> Dict[str, Any]:
    run = store.get_run(run_id)
    original = store.outcome_counts(run_id)
    report: Dict[str, Any] = {
        "run_id": run_id, "mode": run["mode"], "selection": run["selection_sql"],
        "db_null_count_when_run_started": run["db_count_at_start"], "manifest_size": store.total_tasks(run_id),
        "manifest_complete": bool(run["manifest_complete"]),
        "original_run_totals": {k: v for k, v in original.items()},
        "terminal_outcomes": sum(v for k, v in original.items() if k not in (js.PENDING, "claimed", "prepared")),
        "not_attempted_pending": original[js.PENDING] + original["claimed"],
        "specialty_filled_verified": original[js.VERIFIED_UPDATED],
        "fields_written_by_name": store.field_write_counts(run_id),
        "prepared_not_applied": original["prepared"], "prepared_fields": store.prepared_field_counts(run_id),
        "unresolved_and_failure_reasons": [{"outcome": o, "reason": r, "count": n} for o, r, n in store.reason_counts(run_id)[:40]],
        "stage_timings": store.timing_summary(run_id),
        "throughput_all_invocations": store.throughput(run_id),
        "exit_status": exit_status(store, run_id),
    }
    if invocation:
        report["this_invocation"] = {"id": invocation, "totals": {k: v for k, v in store.outcome_counts(run_id, invocation).items() if v},
                                     "throughput": store.throughput(run_id, invocation)}
    remaining = original[js.PENDING] + original["claimed"] + (original["prepared"] if run["mode"] != "dry-run" else 0)
    rate = (report.get("this_invocation") or {}).get("throughput", {}).get("per_minute") or report["throughput_all_invocations"]["per_minute"]
    if remaining and rate:
        hours = remaining / rate / 60
        report["estimated_remaining"] = {"tasks": remaining, "at_per_minute": rate, "hours": round(hours, 1), "days": round(hours / 24, 1)}
    return report


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== run {report['run_id']} ({report['mode']}) ===")
    print(f"selection                 : {report['selection']}")
    print(f"DB NULL count at start    : {report['db_null_count_when_run_started']}")
    print(f"manifest (original run)   : {report['manifest_size']}  complete={report['manifest_complete']}")
    print("-- original-run totals (all invocations) --")
    for name, value in report["original_run_totals"].items():
        if value:
            print(f"  {name:24s}: {value}")
    print(f"  specialty verified      : {report['specialty_filled_verified']}   "
          f"(terminal outcomes {report['terminal_outcomes']}, not attempted {report['not_attempted_pending']})")
    print(f"  fields written by name  : {report['fields_written_by_name']}")
    if report["prepared_not_applied"]:
        print(f"  prepared (not applied)  : {report['prepared_not_applied']}  fields={report['prepared_fields']}")
    if "this_invocation" in report:
        inv = report["this_invocation"]
        print(f"-- this invocation ({inv['id']}) --\n  {inv['totals']}\n  throughput: {inv['throughput']}")
    print("-- reasons (non-updated) --")
    for item in report["unresolved_and_failure_reasons"][:15]:
        print(f"  {item['count']:6d}  {item['outcome']:22s} {item['reason']}")
    print("-- stage timings (seconds) --")
    for stage, stats in report["stage_timings"].items():
        print(f"  {stage:14s} n={int(stats['count']):6d} mean={stats['mean_s']:8.2f} p50={stats['p50_s']:8.2f} p95={stats['p95_s']:8.2f} total={stats['total_s']}")
    if "estimated_remaining" in report:
        print(f"-- estimate -- {report['estimated_remaining']}")
    print(f"exit status: {report['exit_status']} (0 ok, 1 failures, 2 incomplete, 3 preflight failed)")


def write_report_files(store: js.JobStore, run_id: str, report: Dict[str, Any]) -> List[Path]:
    log_dir = Path(CONFIG["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = log_dir / f"null_specialty_report_{run_id}_{stamp}.json"
    csv_path = log_dir / f"null_specialty_not_updated_{run_id}_{stamp}.csv"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile_id", "outcome", "reason", "attempts", "qwen_status", "fields_written"])
        for row in store.tasks_with_outcome(run_id, [o for o in js.OUTCOMES if o not in (js.VERIFIED_UPDATED, js.PENDING)]):
            writer.writerow([row["profile_id"], row["outcome"], row["reason"], row["attempts"], row["qwen_status"], row["fields_written"]])
    return [json_path, csv_path]


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------

def _parse_ids(args) -> Optional[List[str]]:
    ids: List[str] = []
    if getattr(args, "profile_ids", None):
        ids += [x.strip() for x in args.profile_ids.split(",") if x.strip()]
    if getattr(args, "profile_ids_file", None):
        ids += [line.strip() for line in Path(args.profile_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids or None


def _unfinished_run(store: js.JobStore) -> Optional[Any]:
    run = store.latest_run()
    if not run:
        return None
    counts = store.outcome_counts(run["run_id"])
    if not run["manifest_complete"] or counts[js.PENDING] or counts["claimed"] or counts["prepared"]:
        return run
    return None


def cmd_run(args, resume_only: bool = False) -> int:
    from profile_enrichment import neon_db

    store = js.JobStore()
    profile_ids = _parse_ids(args)
    apply = bool(args.apply)

    pipeline = Pipeline(store, llm_mode=args.llm_mode, ocr_enabled=not args.no_ocr,
                        refresh_documents=args.refresh_documents, max_chunks=args.max_chunks)

    run = store.get_run(args.run_id) if args.run_id else (None if args.new_run else _unfinished_run(store))
    if resume_only and run is None:
        print("Nothing to resume: no unfinished run found (use `status` to list runs).")
        return EXIT_PREFLIGHT
    if args.run_id and run is None:
        print(f"Unknown run id {args.run_id}")
        return EXIT_PREFLIGHT

    if not args.skip_preflight:
        full_manifest = run is None and args.limit is None and not profile_ids
        report = preflight(args.llm_mode, args.require_ocr, check_count=full_manifest)
        print_preflight(report)
        if not report["ok"]:
            return EXIT_PREFLIGHT
        db_count = report.get("db_null_specialty_count")
    else:
        db_count = None

    if run is None:
        params = {"limit": args.limit, "profile_ids": profile_ids, "llm_mode": args.llm_mode, "no_ocr": args.no_ocr}
        run_id = store.create_run(neon_db.selection_description(), params, "apply" if apply else "dry-run",
                                  db_count, pipeline.fingerprint(), PROMPT_VERSION)
        print(f"Created run {run_id}")
    else:
        run_id = run["run_id"]
        print(f"Resuming run {run_id} (original manifest kept; not rebuilt from the current NULL rows)")
        if apply and run["mode"] == "dry-run":
            with store._tx() as conn:
                conn.execute("UPDATE runs SET mode='apply' WHERE run_id=?", (run_id,))

    run = store.get_run(run_id)
    if not run["manifest_complete"]:
        limit = json.loads(run["params_json"]).get("limit")
        ids = json.loads(run["params_json"]).get("profile_ids")
        total = build_manifest(store, run_id, limit, ids)
        print(f"Manifest: {total} profile_ids persisted")

    reclaimed = store.reclaim_expired(run_id)
    requeued = {}
    if args.retry_failed:
        requeued["failed"] = store.requeue(run_id, js.FAILURE_OUTCOMES)
    if args.retry_unresolved:
        requeued["unresolved"] = store.requeue(run_id, js.UNRESOLVED_OUTCOMES)
    if args.retry_stale:
        requeued["stale"] = store.requeue(run_id, js.FAILURE_OUTCOMES + js.UNRESOLVED_OUTCOMES, stale_only=True,
                                          model_fp=pipeline.fingerprint(), prompt_version=PROMPT_VERSION)
    if reclaimed or requeued:
        print(f"Re-queued: expired_leases={reclaimed} {requeued}")

    runner = Runner(store, pipeline, run_id, apply=apply, fetch_workers=args.fetch_workers,
                    model_workers=args.model_concurrency, write_workers=args.write_workers, limit=args.limit,
                    prepared_only=False, profile_ids=profile_ids)
    print(f"Mode: {'APPLY (writes to the database)' if apply else 'DRY RUN (no database writes; use --apply to write)'}"
          f" | fetch={args.fetch_workers} model={args.model_concurrency} write={args.write_workers} | llm={args.llm_mode}")
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Aborted; unfinished leases will expire and be reclaimed on the next run.")

    report = build_report(store, run_id, runner.invocation)
    print_report(report)
    paths = write_report_files(store, run_id, report)
    print("Report files:", *paths, sep="\n  ")
    return report["exit_status"]


def cmd_apply(args) -> int:
    store = js.JobStore()
    run = store.get_run(args.run_id) if args.run_id else store.latest_run()
    if not run:
        print("No run found.")
        return EXIT_PREFLIGHT
    run_id = run["run_id"]
    if not args.yes:
        counts = store.outcome_counts(run_id)
        print(f"Run {run_id}: {counts['prepared']} prepared proposals would be written "
              f"(fields: {store.prepared_field_counts(run_id)}). Re-run with --apply to write them.")
        return EXIT_OK
    pipeline = Pipeline(store, llm_mode="never", ocr_enabled=False)
    with store._tx() as conn:
        conn.execute("UPDATE runs SET mode='apply' WHERE run_id=?", (run_id,))
    store.reclaim_expired(run_id)
    runner = Runner(store, pipeline, run_id, apply=True, fetch_workers=1, model_workers=1,
                    write_workers=args.write_workers, limit=args.limit, prepared_only=True)
    runner.run()
    report = build_report(store, run_id, runner.invocation)
    print_report(report)
    write_report_files(store, run_id, report)
    return report["exit_status"]


def cmd_status(args) -> int:
    store = js.JobStore()
    run = store.get_run(args.run_id) if args.run_id else store.latest_run()
    if not run:
        print("No runs recorded yet.")
        return EXIT_OK
    report = build_report(store, run["run_id"])
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
        others = [r["run_id"] for r in store.list_runs() if r["run_id"] != run["run_id"]]
        if others:
            print("other runs:", ", ".join(others[:10]))
    return report["exit_status"]


def cmd_export_review(args) -> int:
    store = js.JobStore()
    run = store.get_run(args.run_id) if args.run_id else store.latest_run()
    rows = store.proposals(run["run_id"], status="open")
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile_id", "field", "current_value", "proposed_value", "evidence", "page", "source", "decision"])
        for row in rows:
            writer.writerow([row["profile_id"], row["field"], row["current_value"], row["proposed_value"], row["evidence"],
                             row["page"], row["source"], ""])
    print(f"Wrote {len(rows)} correction proposals to {args.out}. Put 'approve' in the decision column, then run apply-review.")
    return EXIT_OK


def cmd_apply_review(args) -> int:
    from profile_enrichment import neon_db
    from profile_enrichment.neon_db import FieldChange

    store = js.JobStore()
    applied = skipped = 0
    with open(args.file, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("decision") or "").strip().lower() != "approve":
                continue
            pid, name, new = row["profile_id"], row["field"], row["proposed_value"]
            current = neon_db.fetch_profile_by_id(pid)
            if current is None or name not in neon_db.PROFILE_COLUMNS or name in {"phone", "specialty"}:
                skipped += 1
                continue
            if str(current.get(name) or "") != (row.get("current_value") or ""):
                print(f"SKIP {pid}.{name}: value changed since the proposal was made")
                skipped += 1
                continue
            if not args.apply:
                print(f"WOULD SET {pid}.{name}: {current.get(name)!r} -> {new!r}")
                continue
            result = neon_db.guarded_update(pid, {name: FieldChange(current.get(name), new)})
            if result.verified:
                store.record_write("review", pid, name, current.get(name), new, row.get("evidence") or "", int(row.get("page") or 0), "review")
                applied += 1
            else:
                skipped += 1
    print(f"applied={applied} skipped={skipped}" + ("" if args.apply else " (dry run; add --apply)"))
    return EXIT_OK


def cmd_export_unresolved(args) -> int:
    store = js.JobStore()
    run = store.get_run(args.run_id) if args.run_id else store.latest_run()
    rows = store.tasks_with_outcome(run["run_id"], [o for o in js.OUTCOMES if o not in (js.VERIFIED_UPDATED, js.PENDING)])
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile_id", "outcome", "reason", "attempts", "qwen_status"])
        for row in rows:
            writer.writerow([row["profile_id"], row["outcome"], row["reason"], row["attempts"], row["qwen_status"]])
    print(f"Wrote {len(rows)} rows to {args.out}")
    return EXIT_OK


def cmd_rollback(args) -> int:
    """Undo this run's writes where the column still holds the value we wrote (never touches later changes)."""
    from profile_enrichment import neon_db

    store = js.JobStore()
    writes = store.writes_for_run(args.run_id)
    reverted = skipped = 0
    for w in writes:
        if not args.apply:
            print(f"WOULD REVERT {w['profile_id']}.{w['field']}: {w['new_value']!r} -> {w['old_value']!r}")
            continue
        if neon_db.revert_field(w["profile_id"], w["field"], w["new_value"], w["old_value"]):
            reverted += 1
        else:
            skipped += 1
    print(f"{len(writes)} writes recorded; reverted={reverted} skipped(already changed)={skipped}"
          + ("" if args.apply else " (dry run; add --apply)"))
    return EXIT_OK


def cmd_refresh_vocabulary(args) -> int:
    from profile_enrichment import neon_db
    from profile_enrichment.specialty_lexicon import DEFAULT_VOCABULARY_PATH

    with neon_db.transaction(read_only=True) as conn, conn.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = 300000")
        cursor.execute("SELECT specialty, count(*) FROM " + neon_db._table_sql() +
                       " WHERE specialty IS NOT NULL GROUP BY 1 HAVING count(*) >= 20 ORDER BY 2 DESC")
        rows = cursor.fetchall()
    with open(DEFAULT_VOCABULARY_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["specialty", "count"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labels to {DEFAULT_VOCABULARY_PATH}")
    return EXIT_OK


def cmd_preflight(args) -> int:
    report = preflight(args.llm_mode, args.require_ocr, check_count=not args.skip_count)
    print_preflight(report)
    return EXIT_OK if report["ok"] else EXIT_PREFLIGHT


# ---------------------------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------------------------

def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--apply", action="store_true", help="Write to the database. Without it nothing is written (dry run).")
    p.add_argument("--limit", type=int, default=None, help="Only this many profile_ids (manifest and this invocation).")
    p.add_argument("--profile-ids", help="Comma-separated profile_ids to process (must still match specialty IS NULL).")
    p.add_argument("--profile-ids-file", help="File with one profile_id per line.")
    p.add_argument("--run-id", help="Resume a specific run instead of the latest unfinished one.")
    p.add_argument("--new-run", action="store_true", help="Start a new run even if an unfinished one exists.")
    p.add_argument("--retry-failed", action="store_true", help="Re-queue download/extraction/inference/update failures and concurrent changes.")
    p.add_argument("--retry-unresolved", action="store_true", help="Re-queue insufficient_evidence, validation_failed and no_resume.")
    p.add_argument("--retry-stale", action="store_true", help="Re-queue unresolved/failed tasks decided under a different model, prompt or mode.")
    p.add_argument("--refresh-documents", action="store_true", help="Ignore cached bytes/text (re-download; detects changed resumes).")
    p.add_argument("--llm-mode", choices=["auto", "always", "never"], default="auto",
                   help="auto: Qwen only where deterministic evidence is not conclusive; always: every resume; never: rules only.")
    p.add_argument("--max-chunks", type=int, default=None, help="Cap model chunks per resume (default AI_MAX_CHUNKS).")
    p.add_argument("--fetch-workers", type=int, default=CONFIG["worker_count"], help="Download/extract/OCR concurrency.")
    p.add_argument("--model-concurrency", type=int, default=1, help="Concurrent model requests (keep 1 for a single CPU/GPU server).")
    p.add_argument("--write-workers", type=int, default=2, help="Database write concurrency (<= DB_POOL_MAX).")
    p.add_argument("--no-ocr", action="store_true", help="Disable OCR (scanned pages then end as extraction_failed).")
    p.add_argument("--require-ocr", action="store_true", help="Fail preflight if no OCR engine works.")
    p.add_argument("--skip-preflight", action="store_true", help="Skip readiness checks (not recommended).")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    _add_run_options(sub.add_parser("run", help="Select, prepare and (with --apply) write. Resumes an unfinished run."))
    _add_run_options(sub.add_parser("resume", help="Resume the latest unfinished run (fails if there is none)."))
    p = sub.add_parser("apply", help="Write proposals prepared by an earlier dry run (no model calls).")
    p.add_argument("--run-id"); p.add_argument("--yes", action="store_true", help="Actually write (otherwise just summarises).")
    p.add_argument("--limit", type=int); p.add_argument("--write-workers", type=int, default=2)
    p = sub.add_parser("status", help="Original-run totals, this run's reasons, timings and a completion estimate.")
    p.add_argument("--run-id"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("preflight", help="Check database, text model, vision/Tesseract OCR.")
    p.add_argument("--llm-mode", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--require-ocr", action="store_true"); p.add_argument("--skip-count", action="store_true")
    p = sub.add_parser("export-review", help="Write corrections to populated fields for human review.")
    p.add_argument("--run-id"); p.add_argument("--out", required=True)
    p = sub.add_parser("apply-review", help="Apply rows marked 'approve' in an exported review CSV.")
    p.add_argument("--file", required=True); p.add_argument("--apply", action="store_true")
    p = sub.add_parser("export-unresolved", help="CSV of every task that did not end verified_updated.")
    p.add_argument("--run-id"); p.add_argument("--out", required=True)
    p = sub.add_parser("rollback", help="Revert a run's writes (only where the value is still what we wrote).")
    p.add_argument("--run-id", required=True); p.add_argument("--apply", action="store_true")
    sub.add_parser("refresh-vocabulary", help="Rebuild data/specialty_vocabulary.csv from the database.")

    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0].startswith("-"):
        raw = ["run"] + raw  # `python -m ... --limit 100` means `run --limit 100`
    args = parser.parse_args(raw)

    handlers = {
        "run": cmd_run, "resume": lambda a: cmd_run(a, resume_only=True), "apply": cmd_apply, "status": cmd_status,
        "preflight": cmd_preflight, "export-review": cmd_export_review, "apply-review": cmd_apply_review,
        "export-unresolved": cmd_export_unresolved, "rollback": cmd_rollback, "refresh-vocabulary": cmd_refresh_vocabulary,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
