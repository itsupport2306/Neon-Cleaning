"""End-to-end guard: a hallucinating model must not be able to change NeonDB."""

from profile_enrichment import processor
from profile_enrichment.state_manager import StateManager

RESUME_TEXT = """
Patrick Vrtis
I'm Patrick, a dedicated professional with 10 years of experience.
Registered Nurse
"""


MARIA_TEXT = """
Maria Armelin
Dedicated professional with 10 years of experience.
Registered Nurse
"""


def _run(monkeypatch, tmp_path, ai_result, existing, text=RESUME_TEXT):
    writes = {}

    monkeypatch.setattr(processor, "download_file_bytes", lambda ref: b"fake-bytes")
    monkeypatch.setattr(
        processor,
        "extract_text_layers_from_bytes",
        lambda data, ext, ocr_enabled=None: {
            "native_text": text,
            "ocr_text": "",
            "combined_text": text,
        },
    )
    monkeypatch.setattr(processor, "run_ai_extraction", lambda path, text: dict(ai_result))
    monkeypatch.setattr(processor, "update_profile_fields", lambda pid, updates: writes.update(updates))
    monkeypatch.setattr(processor, "fetch_profile_by_id", lambda pid: dict(existing))

    state = StateManager(str(tmp_path / "state.json"))
    processor.process_profile_record(
        {"profile_id": "p-1", "resume_url": "https://example.invalid/resume.pdf"},
        state,
        existing=dict(existing),
        comprehensive_reprocess=True,
    )
    return writes


def test_hallucinated_payload_is_discarded(monkeypatch, tmp_path):
    """The exact failure from the logs: every fabricated value must be dropped."""
    ai_result = {
        "first_name": "John",
        "last_name": "Doe",
        "specialty": "Cardiology",
        "state_code": "NY",
        "email": "johndoe@example.com",
        "phone": "123-456-7890",
        "city": "New York",
    }
    existing = {
        "profile_id": "p-1",
        "first_name": "Maria",
        "last_name": "Armelin",
        "specialty": None,
        "state_code": None,
        "email": None,
        "phone": None,
        "city": None,
    }
    assert _run(monkeypatch, tmp_path, ai_result, existing, text=MARIA_TEXT) == {"screen_reason": "AI"}


def test_sentence_fragment_does_not_replace_a_real_name(monkeypatch, tmp_path):
    ai_result = {"first_name": "Patrick", "last_name": "I'm Patrick"}
    existing = {
        "profile_id": "p-1",
        "first_name": "Patrick",
        "last_name": "Vrtis",
        "specialty": None,
        "state_code": None,
        "email": None,
        "phone": None,
        "city": None,
    }
    assert _run(monkeypatch, tmp_path, ai_result, existing) == {"screen_reason": "AI"}


def test_values_present_in_the_document_still_get_written(monkeypatch, tmp_path):
    text = RESUME_TEXT + "\nAustin, TX\npatrick.vrtis@stdavids.org\n(512) 887-2043\n"
    ai_result = {
        "first_name": "Patrick",
        "last_name": "Vrtis",
        "city": "Austin",
        "state_code": "TX",
        "email": "patrick.vrtis@stdavids.org",
        "phone": "(512) 887-2043",
    }
    existing = {
        "profile_id": "p-1",
        "first_name": "Patrick",
        "last_name": "Vrtis",
        "specialty": None,
        "state_code": None,
        "email": None,
        "phone": None,
        "city": None,
    }

    writes = _run(monkeypatch, tmp_path, ai_result, existing, text=text)
    assert writes["city"] == "Austin"
    assert writes["state_code"] == "TX"
    assert writes["email"] == "patrick.vrtis@stdavids.org"
    assert "phone" in writes
