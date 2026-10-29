"""End-to-end: selection manifest -> download -> text -> model -> validation -> guarded write -> read-back."""

import csv
import io
import json
import threading
import time
from unittest import mock

import pytest
from docx import Document

from profile_enrichment import downloaders, fill_null_specialty as cli, inference, job_store as js, neon_db, pipeline
from profile_enrichment.inference import BackendError, Completion
from profile_enrichment.specialty_lexicon import SpecialtyLexicon
from tests import fakepg

CLINICAL = [
    "Jordan Lee, RN",
    "Austin, TX",
    "Phone: (512) 887-2043 | Jordan.Lee@Hospital.org",
    "ICU Registered Nurse",
    "Experience",
    "Registered Nurse, Intensive Care Unit, Seton Medical Center",
    "Patient care, clinical assessment and hospital telemetry monitoring.",
]
NON_CLINICAL = ["Sam Porter", "Dayton, OH", "Phone: (937) 555-0147", "Warehouse Associate", "Forklift operation and inventory."]


def docx_bytes(lines):
    document = Document()
    for line in lines:
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class FakeModel(inference.Backend):
    name = "fake"

    def __init__(self):
        self.calls = 0
        self.mode = "ok"
        self.lock = threading.Lock()
        self.delay = 0.0
        self.in_flight = 0
        self.peak = 0

    def fingerprint(self):
        return f"fake:{self.mode_tag}"

    mode_tag = "v1"

    def generate(self, system, user, schema, max_tokens, temperature, seed):
        with self.lock:
            self.calls += 1
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.mode == "down":
                raise BackendError("unavailable", "connection refused")
            if self.mode == "garbage":
                return Completion("{not json", "stop")
            answer = {}
            for name in schema["properties"]:
                answer[name] = {"specialty": "ICU", "first_name": "Jordan", "last_name": "Lee", "city": "Austin",
                                "state_code": "TX", "phone": "(512) 887-2043",
                                "email": "jordan.lee@hospital.org"}.get(name) if "Jordan" in user else None
            return Completion(json.dumps(answer), "stop", 400, 40)
        finally:
            with self.lock:
                self.in_flight -= 1


class World:
    def __init__(self, tmp_path, monkeypatch):
        self.path = str(tmp_path / "pg.sqlite")
        fakepg.create_table(self.path)
        self.pool = fakepg.FakePool(self.path)
        monkeypatch.setattr(neon_db, "get_connection", lambda: self.pool.make_lease())
        monkeypatch.setitem(neon_db.CONFIG, "neon_table_name", "profiles")
        monkeypatch.setitem(pipeline.CONFIG, "cache_dir", str(tmp_path / "cache"))
        self.store = js.JobStore(tmp_path / "jobs.sqlite3")
        self.files = {}
        self.downloads = 0
        self.model = FakeModel()
        self.lexicon = SpecialtyLexicon.load()
        monkeypatch.setattr(pipeline, "fetch_bytes", self._fetch)
        monkeypatch.setitem(pipeline.CONFIG, "ai_max_attempts", 1)
        monkeypatch.setattr(inference, "_sleep", lambda _s: None)
        monkeypatch.setattr(cli, "logger", cli.logger)

    def _fetch(self, ref):
        self.downloads += 1
        item = self.files.get(ref)
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise downloaders.DownloadError("not_found", "HTTP 404", 404)
        return item

    def add(self, pid, lines=None, *, resume=True, **row):
        url = f"/files/resumes/{pid}.docx" if resume else None
        if lines is not None:
            # unique content per profile: the caches are keyed by content hash, identical files would be shared
            self.files[url] = docx_bytes(list(lines) + [f"Reference number {pid}"])
        fakepg.insert(self.path, profile_id=pid, resume_url=url, **row)

    def pipeline(self, mode="always", **kw):
        p = pipeline.Pipeline(self.store, llm_mode=mode, ocr_enabled=False, backend=self.model, lexicon=self.lexicon, **kw)
        p.cache = pipeline.DocumentCache(self.store, self.path + ".docs")
        return p

    def new_run(self, ids, mode="apply"):
        run_id = self.store.create_run("SELECT * FROM profiles WHERE specialty IS NULL", {}, mode, len(ids), "fp", "pv")
        self.store.add_tasks(run_id, [(i, None) for i in ids])
        self.store.finish_manifest(run_id)
        return run_id

    def run(self, run_id, mode="always", apply=True, **kw):
        runner = cli.Runner(self.store, self.pipeline(mode), run_id, apply=apply, fetch_workers=kw.pop("fetch", 2),
                            model_workers=kw.pop("model", 1), write_workers=1, limit=kw.pop("limit", None), **kw)
        runner.run()
        return runner

    def outcome(self, run_id, pid):
        row = self.store._conn().execute("SELECT * FROM tasks WHERE run_id=? AND profile_id=?", (run_id, pid)).fetchone()
        return row["outcome"], row["reason"], row["state"]

    def row(self, pid):
        return fakepg.get(self.path, pid)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------------------------
# happy path, dry run, read-back
# ---------------------------------------------------------------------------------------------

def test_dry_run_writes_nothing_then_apply_writes_and_verifies(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"], mode="dry-run")

    world.run(run_id, apply=False)
    assert world.row("p1")["specialty"] is None  # nothing written
    outcome, reason, state = world.outcome(run_id, "p1")
    assert (outcome, state) == ("pending", "prepared")
    proposal = json.loads(world.store._conn().execute("SELECT proposal_json FROM tasks").fetchone()[0])
    assert proposal["fields"]["specialty"]["value"] == "ICU" and proposal["fields"]["specialty"]["page"] == 1
    assert proposal["fields"]["specialty"]["evidence"]  # the exact source line travels with the value

    out = world.store.path.parent / "prepared.csv"
    with mock.patch.object(cli.js, "JobStore", return_value=world.store):
        assert cli.main(["export-prepared", "--run-id", run_id, "--out", str(out)]) == cli.EXIT_OK
    with open(out, newline="", encoding="utf-8") as handle:
        exported = {r["field"]: r for r in csv.DictReader(handle)}
    assert exported["specialty"]["proposed_value"] == "ICU" and exported["specialty"]["evidence"]
    assert exported["specialty"]["current_value"] == ""  # NULL today
    calls_before = world.model.calls

    world.run(run_id, apply=True)  # applying prepared proposals needs no new model call
    assert world.model.calls == calls_before
    outcome, reason, _ = world.outcome(run_id, "p1")
    assert outcome == "verified_updated" and "read-back" in reason
    row = world.row("p1")
    assert (row["specialty"], row["city"], row["state_code"], row["screen_reason"]) == ("ICU", "Austin", "TX", "AI")
    assert row["email"] == "Jordan.Lee@Hospital.org"  # capitalisation as written in the resume
    assert row["phone"] == "(512) 887-2043" and (row["first_name"], row["last_name"]) == ("Jordan", "Lee")
    writes = {w["field"]: w for w in world.store.writes_for_run(run_id)}
    assert writes["specialty"]["new_value"] == "ICU" and writes["specialty"]["old_value"] is None


def test_existing_populated_values_are_never_overwritten_only_proposed_for_review(world):
    world.add("p1", CLINICAL, first_name="Pat", last_name="Smith", city="Dallas", state_code="TX", phone="5550000000",
              email="Existing@Case.COM")
    run_id = world.new_run(["p1"])
    world.run(run_id)
    row = world.row("p1")
    assert row["specialty"] == "ICU"
    assert (row["first_name"], row["last_name"], row["city"]) == ("Pat", "Smith", "Dallas")
    assert row["phone"] == "5550000000" and row["email"] == "Existing@Case.COM"  # phone untouched, email case preserved
    review = {r["field"]: r for r in world.store.proposals(run_id)}
    assert {"first_name", "last_name", "city"} <= set(review) and "phone" not in review and "email" in review
    assert world.store.field_write_counts(run_id) == {"specialty": 1}


# ---------------------------------------------------------------------------------------------
# outcome accounting: every failure stage has its own outcome
# ---------------------------------------------------------------------------------------------

def test_no_resume_and_missing_file_and_download_error(world):
    world.add("no_url", resume=False)
    world.add("gone", None)  # url set, file missing -> 404
    world.add("flaky", None)
    world.files["/files/resumes/flaky.docx"] = downloaders.DownloadError("transient", "HTTP 503", 503)
    run_id = world.new_run(["no_url", "gone", "flaky"])
    world.run(run_id)
    assert world.outcome(run_id, "no_url")[0] == "no_resume"
    assert world.outcome(run_id, "gone")[0] == "no_resume" and "404" in world.outcome(run_id, "gone")[1]
    assert world.outcome(run_id, "flaky")[0] == "download_failed"


def test_model_failures_are_inference_failed_never_a_completed_empty_result(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    for mode, expect in (("down", "model_unavailable"), ("garbage", "malformed_json")):
        world.model.mode = mode
        world.store.requeue(run_id, [js.INFERENCE_FAILED])
        world.run(run_id)
        outcome, reason, _ = world.outcome(run_id, "p1")
        assert outcome == "inference_failed" and expect in reason
        row = world.row("p1")
        assert row["specialty"] is None and row["screen_reason"] is None  # no "AI" label on a failed response


def test_failed_inference_is_retryable_and_then_succeeds(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    world.model.mode = "down"
    world.run(run_id)
    assert world.store.requeue(run_id, js.FAILURE_OUTCOMES) == 1
    world.model.mode = "ok"
    world.run(run_id)
    assert world.outcome(run_id, "p1")[0] == "verified_updated" and world.row("p1")["screen_reason"] == "AI"


def test_incomplete_ocr_is_extraction_failed_not_no_specialty_found(world, monkeypatch):
    from profile_enrichment import extractors
    from tests.test_documents import make_pdf

    world.files["/files/resumes/scan.pdf"] = make_pdf([None])
    fakepg.insert(world.path, profile_id="scan", resume_url="/files/resumes/scan.pdf")
    monkeypatch.setattr(extractors, "ocr_image", lambda image: extractors.OcrOutcome("", False, "tesseract", "engine down"))
    run_id = world.new_run(["scan"])
    p = world.pipeline()
    p.ocr_enabled = True
    cli.Runner(world.store, p, run_id, apply=True, fetch_workers=1, model_workers=1, write_workers=1, limit=None).run()
    outcome, reason, _ = world.outcome(run_id, "scan")
    assert outcome == "extraction_failed" and "ocr_incomplete" in reason
    assert world.model.calls == 0


def test_non_clinical_resume_is_insufficient_evidence_and_nothing_is_guessed(world):
    world.add("p1", NON_CLINICAL)
    run_id = world.new_run(["p1"])
    world.run(run_id)
    outcome, reason, _ = world.outcome(run_id, "p1")
    assert outcome == "insufficient_evidence"
    row = world.row("p1")
    assert row["specialty"] is None
    assert row["phone"] == "(937) 555-0147" and row["city"] == "Dayton"  # independent verified fields still land
    assert row["screen_reason"] == "AI"  # a successful inference that wrote something
    assert json.loads(world.store._conn().execute("SELECT proposal_json FROM tasks").fetchone()[0])["unresolved"]["outcome"] == outcome


def test_screen_reason_not_set_when_deterministic_path_skips_the_model(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    world.run(run_id, mode="never")
    assert world.model.calls == 0
    assert world.row("p1")["specialty"] == "ICU" and world.row("p1")["screen_reason"] is None


def test_auto_mode_skips_the_model_for_non_clinical_resumes(world):
    world.add("p1", NON_CLINICAL)
    run_id = world.new_run(["p1"])
    world.run(run_id, mode="auto")
    assert world.model.calls == 0 and world.outcome(run_id, "p1")[0] == "insufficient_evidence"


def test_empty_document_is_insufficient_evidence_with_a_reason(world):
    world.files["/files/resumes/p1.docx"] = docx_bytes([])
    fakepg.insert(world.path, profile_id="p1", resume_url="/files/resumes/p1.docx")
    run_id = world.new_run(["p1"])
    world.run(run_id)
    outcome, reason, _ = world.outcome(run_id, "p1")
    assert outcome == "insufficient_evidence" and "empty_document" in reason


# ---------------------------------------------------------------------------------------------
# concurrent changes and conflicts at write time
# ---------------------------------------------------------------------------------------------

def test_row_changed_between_prepare_and_apply_is_not_overwritten(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"], mode="dry-run")
    world.run(run_id, apply=False)
    fakepg.update(world.path, "p1", specialty="Cardiology")  # someone else fills it meanwhile
    world.run(run_id, apply=True)
    assert world.outcome(run_id, "p1")[0] == "concurrent_change"
    assert world.row("p1")["specialty"] == "Cardiology"


def test_row_already_filled_elsewhere_before_we_start(world):
    world.add("p1", CLINICAL, specialty="ICU")
    run_id = world.new_run(["p1"])
    world.run(run_id)
    outcome, reason, _ = world.outcome(run_id, "p1")
    assert outcome == "concurrent_change" and reason.startswith("specialty_already_filled")
    assert cli.exit_status(world.store, run_id) == cli.EXIT_OK  # benign: not counted as a failure


def test_deleted_row_is_accounted_for(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"], mode="dry-run")
    world.run(run_id, apply=False)
    db = fakepg.sqlite3.connect(world.path, isolation_level=None)
    db.execute("DELETE FROM profiles WHERE profile_id='p1'")
    db.close()
    world.run(run_id, apply=True)
    assert world.outcome(run_id, "p1")[0] == "concurrent_change"


def test_email_owned_by_another_profile_does_not_block_the_specialty(world):
    world.add("owner", None, resume=False, email="jordan.lee@hospital.org", specialty="ICU")
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    world.run(run_id)
    row = world.row("p1")
    assert world.outcome(run_id, "p1")[0] == "verified_updated"
    assert row["specialty"] == "ICU" and row["email"] is None and row["city"] == "Austin"


def test_update_error_is_update_failed_and_retryable(world, monkeypatch):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    real = pipeline.guarded_update
    monkeypatch.setattr(pipeline, "guarded_update", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    world.run(run_id)
    assert world.outcome(run_id, "p1")[0] == "update_failed"
    monkeypatch.setattr(pipeline, "guarded_update", real)
    world.store.requeue(run_id, js.FAILURE_OUTCOMES)
    world.run(run_id)
    assert world.outcome(run_id, "p1")[0] == "verified_updated"


# ---------------------------------------------------------------------------------------------
# resumability, retries, leases, backpressure
# ---------------------------------------------------------------------------------------------

def test_unresolved_is_retried_after_the_model_or_prompt_changes(world):
    world.add("p1", NON_CLINICAL)
    run_id = world.new_run(["p1"])
    world.run(run_id)
    assert world.outcome(run_id, "p1")[0] == "insufficient_evidence"
    current = world.pipeline().fingerprint()

    # same model + prompt: an old "no specialty" verdict is final unless explicitly retried
    assert world.store.requeue(run_id, list(js.FAILURE_OUTCOMES + js.UNRESOLVED_OUTCOMES), stale_only=True,
                               model_fp=current, prompt_version=pipeline.PROMPT_VERSION) == 0
    world.model.mode_tag = "v2-bigger-model"
    changed = world.pipeline().fingerprint()
    assert changed != current
    assert world.store.requeue(run_id, list(js.FAILURE_OUTCOMES + js.UNRESOLVED_OUTCOMES), stale_only=True,
                               model_fp=changed, prompt_version=pipeline.PROMPT_VERSION) == 1
    assert world.outcome(run_id, "p1")[2] == "pending"
    # and a changed prompt version alone is enough
    world.run(run_id)
    assert world.store.requeue(run_id, list(js.UNRESOLVED_OUTCOMES), stale_only=True, model_fp=changed,
                               prompt_version="a-new-prompt") == 1


def test_verified_updates_are_never_requeued(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    world.run(run_id)
    assert world.outcome(run_id, "p1")[0] == "verified_updated"
    for flag in (js.FAILURE_OUTCOMES, js.UNRESOLVED_OUTCOMES):
        assert js.VERIFIED_UPDATED not in flag and js.PENDING not in flag
    retry_everything = list(js.FAILURE_OUTCOMES + js.UNRESOLVED_OUTCOMES)
    assert world.store.requeue(run_id, retry_everything) == 0
    assert world.store.requeue(run_id, retry_everything, stale_only=True, model_fp="other", prompt_version="other") == 0


def test_interruption_and_restart_resume_without_redoing_or_losing_work(world):
    ids = [f"p{i:02d}" for i in range(6)]
    for pid in ids:
        world.add(pid, CLINICAL)
    run_id = world.new_run(ids)

    runner = cli.Runner(world.store, world.pipeline(), run_id, apply=True, fetch_workers=1, model_workers=1,
                        write_workers=1, limit=2)
    runner.run()  # "interrupted" after two tasks
    done = world.store.outcome_counts(run_id)
    assert done["verified_updated"] == 2 and done[js.PENDING] == 4

    # a crashed worker: claimed a task and vanished; its lease has expired
    crashed = world.store.claim(run_id, "crashed-worker", lease_seconds=-1, limit=1)
    assert crashed
    world.store.reclaim_expired(run_id)

    world.run(run_id)  # restart: original manifest, only unfinished tasks
    counts = world.store.outcome_counts(run_id)
    assert counts["verified_updated"] == 6 and counts[js.PENDING] == 0 and counts["claimed"] == 0
    attempts = {r["profile_id"]: r["attempts"] for r in world.store._conn().execute("SELECT profile_id, attempts FROM tasks")}
    assert max(attempts.values()) <= 2 and sum(1 for v in attempts.values() if v == 2) == 1  # only the crashed one retried
    assert world.model.calls == 6  # nothing was redone: 2 + 4 tasks, one model call each


def test_stop_request_releases_unprocessed_leases(world):
    ids = [f"p{i:02d}" for i in range(8)]
    for pid in ids:
        world.add(pid, CLINICAL)
    run_id = world.new_run(ids)
    world.model.delay = 0.05
    runner = cli.Runner(world.store, world.pipeline(), run_id, apply=True, fetch_workers=3, model_workers=1,
                        write_workers=1, limit=None)
    threading.Timer(0.15, runner.stop.set).start()
    runner.run()
    counts = world.store.outcome_counts(run_id)
    assert counts["claimed"] == 0  # nothing left leased to a dead worker
    assert counts["verified_updated"] + counts[js.PENDING] == 8


def test_two_workers_never_process_the_same_task(world):
    ids = [f"p{i:02d}" for i in range(24)]
    for pid in ids:
        world.add(pid, CLINICAL)
    run_id = world.new_run(ids)
    runners = [cli.Runner(world.store, world.pipeline(), run_id, apply=True, fetch_workers=3, model_workers=1,
                          write_workers=1, limit=None) for _ in range(2)]
    threads = [threading.Thread(target=r.run) for r in runners]
    [t.start() for t in threads]
    [t.join() for t in threads]
    rows = world.store._conn().execute("SELECT attempts, outcome FROM tasks").fetchall()
    assert all(r["attempts"] == 1 and r["outcome"] == "verified_updated" for r in rows)
    assert sum(r.processed for r in runners) == 24
    assert world.model.calls == 24


def test_lease_fencing_discards_a_stale_workers_result(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    first = world.store.claim(run_id, "slow", lease_seconds=-1)[0]
    second = world.store.claim(run_id, "fast", lease_seconds=60)[0]
    assert first.profile_id == second.profile_id
    with pytest.raises(js.LeaseLost):
        world.store.finish(run_id, "p1", "slow", js.VERIFIED_UPDATED)
    world.store.finish(run_id, "p1", "fast", js.INSUFFICIENT_EVIDENCE, "x")


def test_backpressure_bounds_in_flight_work(world):
    ids = [f"p{i:02d}" for i in range(20)]
    for pid in ids:
        world.add(pid, CLINICAL)
    run_id = world.new_run(ids)
    world.model.delay = 0.03
    runner = cli.Runner(world.store, world.pipeline(), run_id, apply=True, fetch_workers=4, model_workers=1,
                        write_workers=1, limit=None)
    peak = {"claimed": 0}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak["claimed"] = max(peak["claimed"], world.store.outcome_counts(run_id)["claimed"])
            time.sleep(0.005)

    watcher = threading.Thread(target=watch)
    watcher.start()
    runner.run()
    stop.set()
    watcher.join()
    bound = 4 + runner.q_model.maxsize + 1 + runner.q_write.maxsize + 1 + 2
    assert peak["claimed"] <= bound < 20  # leases are bounded by the pipeline depth, not by the manifest size
    assert world.model.peak == 1  # one model request at a time


def test_original_manifest_is_not_rebuilt_from_the_remaining_null_rows(world):
    ids = ["p1", "p2", "p3"]
    for pid in ids:
        world.add(pid, CLINICAL)
    run_id = world.new_run(ids)
    world.run(run_id, limit=1)
    fakepg.update(world.path, "p3", specialty="Filled elsewhere")  # the live NULL set shrinks
    assert world.store.total_tasks(run_id) == 3  # the manifest still has every original id
    world.run(run_id)
    report = cli.build_report(world.store, run_id)
    assert report["manifest_size"] == 3 and sum(report["original_run_totals"][o] for o in js.OUTCOMES) == 3
    assert world.outcome(run_id, "p3")[0] == "concurrent_change"


def test_report_separates_original_run_from_this_invocation(world):
    ids = ["p1", "p2", "p3"]
    for pid in ids:
        world.add(pid, CLINICAL)
    run_id = world.new_run(ids)
    first = world.run(run_id, limit=2)
    second = world.run(run_id)
    report = cli.build_report(world.store, run_id, second.invocation)
    assert report["original_run_totals"]["verified_updated"] == 3
    assert report["this_invocation"]["totals"] == {"verified_updated": 1}
    assert world.store.outcome_counts(run_id, first.invocation)["verified_updated"] == 2


def test_exit_status_exposes_failures_even_when_every_id_is_accounted_for(world):
    world.add("ok", CLINICAL)
    world.add("bad", CLINICAL)
    world.files["/files/resumes/bad.docx"] = downloaders.DownloadError("transient", "HTTP 503", 503)
    run_id = world.new_run(["ok", "bad"])
    assert cli.exit_status(world.store, run_id) == cli.EXIT_INCOMPLETE  # pending work remains
    world.run(run_id)
    counts = world.store.outcome_counts(run_id)
    assert counts[js.PENDING] == 0 and counts["verified_updated"] == 1 and counts[js.DOWNLOAD_FAILED] == 1
    assert cli.exit_status(world.store, run_id) == cli.EXIT_FAILURES
    world.files["/files/resumes/bad.docx"] = docx_bytes(CLINICAL)
    world.store.requeue(run_id, js.FAILURE_OUTCOMES)
    world.run(run_id)
    assert cli.exit_status(world.store, run_id) == cli.EXIT_OK


def test_dry_run_with_all_tasks_prepared_is_accounted_but_apply_run_is_not(world):
    world.add("p1", CLINICAL)
    dry = world.new_run(["p1"], mode="dry-run")
    world.run(dry, apply=False)
    assert cli.exit_status(world.store, dry) == cli.EXIT_OK
    world.store._conn().execute("UPDATE runs SET mode='apply' WHERE run_id=?", (dry,))
    assert cli.exit_status(world.store, dry) == cli.EXIT_INCOMPLETE  # prepared but not written yet


# ---------------------------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------------------------

def test_documents_text_and_model_results_are_cached_by_content_hash(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"], mode="dry-run")
    world.run(run_id, apply=False)
    assert (world.downloads, world.model.calls) == (1, 1)

    world.store.requeue_prepared(run_id)
    world.run(run_id, apply=False)
    assert (world.downloads, world.model.calls) == (1, 1)  # bytes, text and validated model output all reused

    world.model.mode_tag = "v2"  # a different model must not reuse the old model's answers
    world.store.requeue_prepared(run_id)
    world.run(run_id, apply=False)
    assert world.model.calls == 2 and world.downloads == 1

    world.files["/files/resumes/p1.docx"] = docx_bytes(CLINICAL + ["Additional certification line"])
    world.store.requeue_prepared(run_id)
    p = world.pipeline(refresh_documents=True)
    cli.Runner(world.store, p, run_id, apply=False, fetch_workers=1, model_workers=1, write_workers=1, limit=None).run()
    assert world.downloads == 2 and world.model.calls == 3  # a changed document is re-read and re-inferred


def test_failed_model_results_are_never_cached(world):
    world.add("p1", CLINICAL)
    run_id = world.new_run(["p1"])
    world.model.mode = "garbage"
    world.run(run_id)
    assert world.store._conn().execute("SELECT count(*) FROM inference_cache").fetchone()[0] == 0


def test_incomplete_documents_are_not_cached(world):
    from tests.test_documents import make_pdf

    world.files["/files/resumes/scan.pdf"] = make_pdf([None])
    fakepg.insert(world.path, profile_id="scan", resume_url="/files/resumes/scan.pdf")
    run_id = world.new_run(["scan"])
    world.run(run_id)  # OCR disabled in the world pipeline: scanned page stays incomplete
    assert world.outcome(run_id, "scan")[0] == "extraction_failed"
    assert world.store._conn().execute("SELECT count(*) FROM doc_cache").fetchone()[0] == 0
