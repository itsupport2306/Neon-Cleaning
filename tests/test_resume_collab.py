import inspect
import json
import threading

import pytest

from profile_enrichment.downloaders import DownloadError
from profile_enrichment.inference import BackendError, Completion
from profile_enrichment.specialty_lexicon import SpecialtyLexicon
from resume_collab import compare as cmp
from resume_collab import validate_resumes as vr
from resume_collab.csv_store import ResultStore, read_ids

RESUME = (
    "Priya Ramanathan, RN\n12 Oak Street, Columbus, OH 43004\nPhone: (614) 555-0182\n"
    "Experience\nStaff nurse, intensive care unit (ICU), Mercy Hospital, 2016-2024\n"
)
PROFILE = {"profile_id": "p1", "resume_url": "https://x/r.txt", "first_name": "Priya", "last_name": "Ramanathan",
           "specialty": "ICU", "profession_type": None, "city": "Columbus", "state_code": "OH", "zip_code": "43004"}


class FakeBackend(vr.inference.Backend):
    name = "fake"

    def __init__(self, answer=None, error=None, text=None):
        self.answer, self.error, self.text, self.calls = answer or {}, error, text, 0

    def generate(self, system, user, schema, max_tokens, temperature, seed):
        self.calls += 1
        if self.error:
            raise self.error
        base = {"first_name": "Priya", "last_name": "Ramanathan", "street": "12 Oak Street", "city": "Columbus",
                "state": "OH", "zip": "43004", "profession": "Registered Nurse", "specialties": ["intensive care unit"],
                "specialty_verdict": "supported", "specialty_reason": "ICU role", "confidence": "high"}
        return Completion(self.text if self.text is not None else json.dumps({**base, **self.answer}), "stop")

    def fingerprint(self):
        return "fake"


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(vr, "fetch_bytes", lambda ref: RESUME.encode())

    def run(profile=None, backend=None, resume=None):
        if resume is not None:
            monkeypatch.setattr(vr, "fetch_bytes", lambda ref: resume.encode())
        return vr.validate_profile({**PROFILE, **(profile or {})}, backend or FakeBackend(),
                                   SpecialtyLexicon.load(), threading.Semaphore(1), ocr_enabled=False)
    return run


# --- comparison rules ------------------------------------------------------------------------

def test_names_ignore_case_accents_credentials():
    assert cmp.compare_name("JOSE", "José, RN", "first_name").status == cmp.MATCH
    assert cmp.compare_name("Mary", "Mary Ann", "first_name").status == cmp.MATCH
    assert cmp.compare_name("O'Brien", "OBrien", "last_name").status == cmp.MATCH


def test_name_mismatch_placeholder_and_missing():
    assert cmp.compare_name("Jon", "John", "first_name").status == cmp.MISMATCH
    assert cmp.compare_name("Unknown", "Priya", "first_name").status == cmp.MISMATCH
    assert cmp.compare_name("Priya", "", "first_name").status == cmp.NOT_VERIFIABLE
    assert cmp.compare_name("", "Priya", "first_name").status == cmp.NOT_VERIFIABLE


def test_swapped_names_detected():
    assert cmp.detect_swap("Ramanathan", "Priya", "Priya", "Ramanathan")
    assert not cmp.detect_swap("Priya", "Ramanathan", "Priya", "Ramanathan")


def test_address_components():
    stored = {"city": "Columbus", "state": "OH", "zip": "43004"}
    assert cmp.compare_address(stored, {"city": "columbus", "state": "Ohio", "zip": "43004-1234"}).status == cmp.MATCH
    wrong = cmp.compare_address(stored, {"city": "Dayton", "state": "OH"})
    assert wrong.status == cmp.MISMATCH and "city" in wrong.note
    assert cmp.compare_address(stored, {}).status == cmp.NOT_VERIFIABLE
    assert cmp.compare_address({}, {"city": "Dayton"}).status == cmp.NOT_VERIFIABLE  # nothing stored to contradict


def test_specialty_alias_and_verdict_rules():
    lexicon = SpecialtyLexicon.load()
    assert cmp.compare_specialty("ICU", [], None, "unclear", "", RESUME, lexicon).status == cmp.MATCH
    unsupported = cmp.compare_specialty("Cardiology", ["Dermatology"], "Physician", "not_supported", "derm only",
                                        "Dermatology clinic", None)
    assert unsupported.status == cmp.MISMATCH
    # the resume says nothing usable: never a confirmed discrepancy
    assert cmp.compare_specialty("Cardiology", [], None, "not_supported", "", "text", None).status == cmp.NOT_VERIFIABLE
    assert cmp.compare_specialty("Cardiology", [], None, "unclear", "", "text", None).status == cmp.NOT_VERIFIABLE


# --- CSV store -------------------------------------------------------------------------------

def test_store_is_incremental_and_deduplicated(tmp_path):
    store = ResultStore(tmp_path)
    assert store.record_no_discrepancy({"profile_id": "a", "validation_status": "no_discrepancy_found"})
    assert not store.record_error({"profile_id": "a"})  # same id, other file: ignored
    assert not store.record_no_discrepancy({"profile_id": "a"})
    assert store.record_error({"profile_id": "b", "error_category": "download_failed"})
    resumed = ResultStore(tmp_path)  # a new process sees what was already written
    assert resumed.is_done("a") and resumed.is_done("b") and not resumed.is_done("c")
    assert read_ids(tmp_path / "resume_validation_no_discrepancies.csv") == ["a"]


def test_retry_errors_archives_error_file(tmp_path):
    ResultStore(tmp_path).record_error({"profile_id": "b", "error_category": "download_failed"})
    store = ResultStore(tmp_path, retry_errors=True)
    assert not store.is_done("b")
    assert list(tmp_path.glob("resume_validation_errors.*.bak.csv"))


# --- end to end with a fake model ------------------------------------------------------------

def test_matching_profile_goes_to_manual_review(pipeline):
    outcome = pipeline()
    assert outcome.kind == "no_discrepancy"
    assert outcome.row["validation_status"] == "no_discrepancy_found"


def test_wrong_name_and_city_reported_with_both_values(pipeline):
    outcome = pipeline({"first_name": "Maria", "city": "Dayton"})
    assert outcome.kind == "discrepancy"
    assert outcome.row["mismatched_fields"] == "first_name;address"
    assert outcome.row["first_name_db"] == "Maria" and outcome.row["first_name_resume"] == "Priya"
    assert "Dayton" in outcome.row["address_db"] and "Columbus" in outcome.row["address_resume"]


def test_invented_resume_values_are_ignored(pipeline):
    # the model "reads" a name that is not in the resume: it must not become a discrepancy
    outcome = pipeline(backend=FakeBackend({"first_name": "Zelda", "last_name": "Quux"}))
    assert outcome.kind == "no_discrepancy"
    assert "first_name=not_verifiable" in outcome.row["field_statuses"]


def test_unsupported_specialty_is_a_discrepancy(pipeline):
    backend = FakeBackend({"specialty_verdict": "not_supported", "specialties": ["intensive care unit"]})
    outcome = pipeline({"specialty": "Dermatology"}, backend)
    assert outcome.kind == "discrepancy" and outcome.row["mismatched_fields"] == "specialty"


def test_low_confidence_negative_verdict_is_not_reported(pipeline):
    backend = FakeBackend({"specialty_verdict": "not_supported", "confidence": "low"})
    outcome = pipeline({"specialty": "Dermatology"}, backend)
    assert outcome.kind == "no_discrepancy" and "specialty=not_verifiable" in outcome.row["field_statuses"]


def test_nothing_verifiable_goes_to_errors(pipeline):
    backend = FakeBackend({"first_name": None, "last_name": None, "street": None, "city": None, "state": None,
                           "zip": None, "specialties": [], "profession": None, "specialty_verdict": "unclear"})
    outcome = pipeline({"specialty": "Dermatology"}, backend, resume="Curriculum vitae\nReferences available\n")
    assert outcome.kind == "error" and outcome.category == "unverifiable"


@pytest.mark.parametrize("error,category,retryable", [
    (DownloadError("not_found", "HTTP 404", 404), "resume_not_found", "no"),
    (DownloadError("transient", "HTTP 503", 503), "download_failed", "yes"),
    (DownloadError("permanent", "resume reference is not a downloadable URL: 'x'"), "invalid_resume_url", "no"),
])
def test_download_failures_are_logged_not_flagged(monkeypatch, pipeline, error, category, retryable):
    def boom(ref):
        raise error

    monkeypatch.setattr(vr, "fetch_bytes", boom)
    outcome = pipeline()
    assert (outcome.kind, outcome.category, outcome.row["retryable"]) == ("error", category, retryable)


def test_model_errors_and_invalid_output_are_logged(pipeline):
    down = pipeline(backend=FakeBackend(error=BackendError("unavailable", "connection refused")))
    assert (down.kind, down.category) == ("error", "model_error")
    garbage = pipeline(backend=FakeBackend(text="not json"))
    assert (garbage.kind, garbage.category) == ("error", "invalid_model_output")
    wrong = pipeline(backend=FakeBackend(text=json.dumps({"first_name": "x"})))
    assert wrong.category == "invalid_model_output"


def test_empty_url_is_logged(pipeline):
    assert pipeline({"resume_url": "  "}).category == "invalid_resume_url"


# --- safety ----------------------------------------------------------------------------------

def test_module_issues_only_select_statements():
    source = inspect.getsource(vr)
    for word in ("UPDATE ", "INSERT ", "DELETE ", "DROP ", "TRUNCATE ", "guarded_update", "revert_field"):
        assert word not in source, f"{word!r} must not appear in the read-only validator"
    assert source.count("read_only=True") == source.count("neon_db.transaction(")
    assert "specialty IS NOT NULL AND resume_url IS NOT NULL" in vr.WHERE_ELIGIBLE
