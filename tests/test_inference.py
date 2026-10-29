import json

import pytest

from profile_enrichment import inference
from profile_enrichment.inference import (
    BackendError, Completion, Status, build_chunks, check_model_ready, extract_fields,
)
from profile_enrichment.specialty_lexicon import SpecialtyLexicon

TEXT = "Priya Ramanathan, RN\nColumbus, OH\nPhone: (614) 555-0182 | Priya.Ramanathan@Mailbox.org\nSpecialty: Cardiology\n"


class ScriptedBackend(inference.Backend):
    """Replays a list of outcomes: Completion objects are returned, exceptions are raised."""

    name = "scripted"

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def generate(self, system, user, schema, max_tokens, temperature, seed):
        self.calls.append({"user": user, "schema": schema, "max_tokens": max_tokens, "temperature": temperature})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Reply):  # answer only the fields that were asked for, like a schema-constrained model
            wanted = schema["properties"]
            return Completion(json.dumps({k: item.values.get(k) for k in wanted}), "stop", 300, 60)
        return item

    def fingerprint(self):
        return "scripted:test"


class Reply:
    def __init__(self, values):
        self.values = values


def ok(**values):
    return Reply(values)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(inference, "_sleep", lambda _s: None)
    monkeypatch.setitem(inference.CONFIG, "ai_max_attempts", 3)


LEX = SpecialtyLexicon.load()


def run(backend, text=TEXT, fields=inference.FIELDS):
    return extract_fields([(1, text)], fields, backend=backend, lexicon=LEX)


def test_success_returns_verified_values_with_evidence_and_page():
    result = run(ScriptedBackend(ok(first_name="Priya", email="priya.ramanathan@mailbox.org", specialty="Cardiology")))
    assert result.status is Status.OK and result.succeeded and result.coverage_complete
    by_field = {c.field: c for c in result.candidates}
    assert by_field["email"].value == "Priya.Ramanathan@Mailbox.org"  # capitalisation comes from the resume, not the model
    assert by_field["specialty"].evidence == "Specialty: Cardiology" and by_field["specialty"].page == 1


def test_nothing_found_is_a_success_with_no_fields_not_an_error():
    result = run(ScriptedBackend(ok()))
    assert result.status is Status.OK_NO_FIELDS and result.succeeded and result.candidates == []


def test_every_value_failing_evidence_is_insufficient_evidence():
    result = run(ScriptedBackend(ok(specialty="Neonatology", city="Dayton", phone="(212) 555-0100")))
    assert result.status is Status.INSUFFICIENT_EVIDENCE and result.succeeded
    assert {r["field"] for r in result.rejected} == {"specialty", "city", "phone"}


def test_hallucinated_and_non_taxonomy_values_are_rejected_with_reasons():
    result = run(ScriptedBackend(ok(specialty="Inspect", city="Dayton")))
    reasons = {r["field"]: r["reason"] for r in result.rejected}
    assert reasons == {"specialty": "specialty_not_in_taxonomy", "city": "value_not_in_source"}


def test_malformed_json_is_retried_then_reported():
    backend = ScriptedBackend(Completion('{"first_name": "Pri', "stop", 10, 5))
    result = run(backend)
    assert result.status is Status.MALFORMED_JSON and not result.succeeded and result.retryable
    assert len(backend.calls) == 3  # bounded: exactly max attempts
    assert [c["temperature"] for c in backend.calls] == [0.0, 0.2, 0.2]  # a retry is not a replay of the same sample


def test_malformed_then_valid_recovers():
    result = run(ScriptedBackend(Completion("not json at all", "stop"), ok(first_name="Priya")))
    assert result.status is Status.OK and result.chunks[0].attempts == 2


def test_truncated_generation_is_detected_from_finish_reason():
    backend = ScriptedBackend(Completion('{"first_name": "Priya", "last_na', "length", 300, 128))
    result = run(backend)
    assert result.status is Status.TRUNCATED and not result.succeeded
    assert backend.calls[1]["max_tokens"] > backend.calls[0]["max_tokens"]  # retried with more room


def test_prompt_overflowing_the_context_counts_as_truncation(monkeypatch):
    monkeypatch.setitem(inference.CONFIG, "ai_context_tokens", 4096)
    result = run(ScriptedBackend(Completion(json.dumps({k: None for k in inference.FIELDS}), "stop", 4090, 20)))
    assert result.status is Status.TRUNCATED


def test_wrong_schema_is_schema_invalid_not_empty():
    # the real qwen:4b answered {'name': 'CAD', 'description': ...} to an extraction request
    result = run(ScriptedBackend(Completion(json.dumps({"name": "CAD", "description": "Computer-aided design"}), "stop")))
    assert result.status is Status.SCHEMA_INVALID and not result.succeeded


def test_timeout_is_retried_with_backoff_then_reported():
    sleeps = []
    inference._sleep = sleeps.append
    backend = ScriptedBackend(BackendError("timeout", "timed out"))
    result = run(backend)
    assert result.status is Status.TIMEOUT and len(backend.calls) == 3 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0] or sleeps == [0, 0]


def test_model_unavailable_transient_then_recovers():
    result = run(ScriptedBackend(BackendError("unavailable", "connection refused"), ok(first_name="Priya")))
    assert result.status is Status.OK


def test_permanent_unavailability_is_not_retried():
    backend = ScriptedBackend(BackendError("unavailable", "model 'x' not found", permanent=True))
    result = run(backend)
    assert result.status is Status.MODEL_UNAVAILABLE and len(backend.calls) == 1


def test_local_model_load_failure_is_model_unavailable(monkeypatch):
    from profile_enrichment.inference import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_model", None)
    monkeypatch.setattr(LlamaCppBackend, "_load_error", None)
    monkeypatch.setitem(inference.CONFIG, "local_model_path", "does-not-exist.gguf")
    monkeypatch.setitem(inference.CONFIG, "local_model_enabled", False)  # no download fallback
    result = run(LlamaCppBackend())
    assert result.status is Status.MODEL_UNAVAILABLE and "failed to load local model" in result.error
    LlamaCppBackend._load_error = None


def test_a_failed_chunk_makes_the_whole_result_a_failure_even_if_others_succeeded(monkeypatch):
    monkeypatch.setitem(inference.CONFIG, "ai_chunk_tokens", 200)
    monkeypatch.setitem(inference.CONFIG, "ai_max_attempts", 1)
    text = "\n".join(f"Line number {i} describing duties on the unit" for i in range(80))
    backend = ScriptedBackend(ok(), Completion("garbage", "stop"), ok())
    result = run(backend, text)
    assert result.chunks_total > 2 and result.status is Status.MALFORMED_JSON and not result.coverage_complete


# ---- long resumes -----------------------------------------------------------------------------

def test_long_resume_is_split_into_chunks_that_cover_every_line_and_page(monkeypatch):
    monkeypatch.setitem(inference.CONFIG, "ai_chars_per_token", 3.0)
    pages = [(n, "\n".join(f"page{n} line{i} routine duties and responsibilities" for i in range(60))) for n in (1, 2, 3)]
    chunks = build_chunks(pages, budget_tokens=300, overlap_tokens=30)
    assert len(chunks) > 3
    joined = "\n".join(c.text for c in chunks)
    for n in (1, 2, 3):
        for i in range(60):
            assert f"page{n} line{i} " in joined  # nothing silently dropped (the old code kept only the first 12k chars)
    assert all(len(c.text) <= int(300 * 3.0) + 200 for c in chunks)  # budget respected (marker slack)
    assert {p for c in chunks for p in c.pages} == {1, 2, 3}


def test_fact_on_the_last_page_of_a_long_resume_is_found_with_its_page(monkeypatch):
    monkeypatch.setitem(inference.CONFIG, "ai_chunk_tokens", 300)
    monkeypatch.setitem(inference.CONFIG, "ai_max_chunks", 200)
    filler = "\n".join(f"Responsible for routine unit duties item {i}" for i in range(400))
    pages = [(1, filler), (2, filler), (3, "Specialty: Cardiology\n")]
    seen = []

    class Backend(ScriptedBackend):
        def generate(self, system, user, schema, max_tokens, temperature, seed):
            seen.append(len(user))
            return Completion(json.dumps({"specialty": "Cardiology" if "Cardiology" in user else None}), "stop", 300, 20)

    result = extract_fields(pages, ["specialty"], backend=Backend(ok()), lexicon=LEX)
    assert result.status is Status.OK and result.coverage_complete
    cardiology = [c for c in result.candidates if c.field == "specialty"]
    assert cardiology and cardiology[0].page == 3
    assert max(seen) < 4096 * 3  # no prompt larger than the context window


def test_chunk_cap_is_reported_as_partial_coverage_not_silently_dropped(monkeypatch):
    monkeypatch.setitem(inference.CONFIG, "ai_chunk_tokens", 200)
    text = "\n".join(f"Line {i} of an extremely long resume with many repeated duties" for i in range(500))
    result = extract_fields([(1, text)], ["city"], backend=ScriptedBackend(ok()), lexicon=LEX, max_chunks=2)
    assert result.chunks_total > 2 and len(result.chunks) == 2
    assert not result.coverage_complete and "coverage partial" in result.error


def test_schema_is_requested_for_exactly_the_needed_fields():
    backend = ScriptedBackend(ok(specialty="Cardiology"))
    extract_fields([(1, TEXT)], ["specialty", "email"], backend=backend, lexicon=LEX)
    schema = backend.calls[0]["schema"]
    assert list(schema["properties"]) == ["specialty", "email"] and schema["additionalProperties"] is False
    assert "first_name" not in backend.calls[0]["user"]


# ---- specialty shortlist ------------------------------------------------------------------------

def test_prompt_lists_only_specialty_labels_found_in_that_chunk():
    backend = ScriptedBackend(ok(specialty="Cardiology"))
    run(backend)
    prompt = backend.calls[0]["user"]
    assert "Specialty labels that appear in this text:" in prompt and "Cardiology" in prompt
    assert "Pharmacist" not in prompt  # a vocabulary label the resume does not contain is never offered

    no_specialty = ScriptedBackend(ok(email="priya.ramanathan@mailbox.org"))
    extract_fields([(1, TEXT)], ["email"], backend=no_specialty, lexicon=LEX)
    assert "Specialty labels" not in no_specialty.calls[0]["user"]


def test_shortlist_does_not_relax_the_evidence_gate():
    # the model names a real vocabulary label that is absent from the resume: still rejected
    result = run(ScriptedBackend(ok(specialty="Pharmacist")))
    assert not [c for c in result.candidates if c.field == "specialty"]
    assert any(r["field"] == "specialty" and r["reason"] == "value_not_in_source" for r in result.rejected)


# ---- readiness --------------------------------------------------------------------------------

def test_readiness_check_demands_a_valid_extraction_not_just_an_answer():
    good = ScriptedBackend(ok(first_name="Priya", last_name="Ramanathan", email="priya.ramanathan@mailbox.org",
                              specialty="Cardiology", city="Columbus", state_code="OH"))
    assert check_model_ready(good).ok

    assert not check_model_ready(ScriptedBackend(Completion("hello!", "stop"))).ok  # answers, but not with JSON
    assert not check_model_ready(ScriptedBackend(ok())).ok  # valid JSON, canary facts missing
    wrong = ScriptedBackend(ok(first_name="John", last_name="Doe", email="johndoe@example.com", specialty="Cardiology"))
    report = check_model_ready(wrong)
    assert not report.ok and "wrong/missing" in report.detail
    assert not check_model_ready(ScriptedBackend(BackendError("unavailable", "down", permanent=True))).ok


def test_result_round_trips_through_the_cache():
    result = run(ScriptedBackend(ok(specialty="Cardiology")))
    clone = inference.ExtractionResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert clone.status is result.status and clone.candidates == result.candidates
