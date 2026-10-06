from pathlib import Path

from profile_enrichment import neon_db, processor
from profile_enrichment.extractors import extract_profile_fields_from_text
from profile_enrichment.neon_db import build_update_map, update_profile_fields
from profile_enrichment.processor import _is_reliable_value, _verify_and_merge_fields, sanitize_extracted
from profile_enrichment.state_manager import StateManager


def test_reliable_value_filter_rejects_unknown():
    assert _is_reliable_value("Unknown") is False
    assert _is_reliable_value("Candidate") is False
    assert _is_reliable_value("John Smith") is True


def test_sanitize_extracted_keeps_real_values_only():
    data = {
        "first_name": "John",
        "last_name": "Smith",
        "email": "j@company.com",
        "phone": "5551234567",
        "city": "Unknown",
        "state_code": "tx",
        "specialty": "Cardiology",
    }
    cleaned = sanitize_extracted(data)
    assert cleaned["email"] == "j@company.com"
    assert cleaned["state_code"] == "TX"
    assert "city" not in cleaned


def test_build_update_map_only_fills_blank_columns():
    existing = {
        "first_name": None,
        "last_name": "",
        "city": "",
        "state_code": None,
        "specialty": None,
        "phone": None,
        "email": "existing@example.com",
    }
    extracted = {
        "first_name": "John",
        "last_name": "Smith",
        "city": "Dallas",
        "state_code": "TX",
        "specialty": "Cardiology",
        "phone": "5551234567",
        "email": "new@example.com",
    }
    assert build_update_map(existing, extracted) == {
        "first_name": "John",
        "last_name": "Smith",
        "city": "Dallas",
        "state_code": "TX",
        "specialty": "Cardiology",
        "phone": "5551234567",
    }


def test_build_update_map_only_fills_phone_when_null():
    existing = {
        "first_name": None,
        "last_name": None,
        "city": None,
        "state_code": None,
        "specialty": None,
        "phone": "",
        "email": None,
    }
    extracted = {"phone": "5551234567"}

    assert "phone" not in build_update_map(existing, extracted)


def test_comprehensive_merge_does_not_overwrite_non_null_phone(monkeypatch):
    monkeypatch.setitem(processor.CONFIG, "allow_overwrite_existing", True)
    existing = {"phone": "5550000000"}
    extracted = {"phone": "5551234567"}

    updates, changes = _verify_and_merge_fields(
        "abc-123",
        existing,
        extracted,
        source_text="Phone: 555-123-4567",
    )

    assert "phone" not in updates
    assert "phone" not in changes


def test_comprehensive_merge_fills_null_phone_from_resume(monkeypatch):
    monkeypatch.setitem(processor.CONFIG, "allow_overwrite_existing", True)
    existing = {"phone": None}
    extracted = {"phone": "5551234567"}

    updates, changes = _verify_and_merge_fields(
        "abc-123",
        existing,
        extracted,
        source_text="Phone: 555-123-4567",
    )

    assert updates["phone"] == "5551234567"
    assert changes["phone"] == "<empty> -> 5551234567"


def test_update_profile_fields_guards_phone_only_update_at_sql_level(monkeypatch):
    executed = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, sql, params):
            executed["sql"] = sql
            executed["params"] = params

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def cursor(self):
            return FakeCursor()

        def commit(self):
            executed["committed"] = True

    monkeypatch.setitem(neon_db.CONFIG, "neon_table_name", "profiles")
    monkeypatch.setattr(neon_db, "get_connection", lambda: FakeConnection())

    update_profile_fields("abc-123", {"phone": "5551234567"})

    assert '"phone" = CASE WHEN "phone" IS NULL THEN %s ELSE "phone" END' in executed["sql"]
    assert 'WHERE profile_id = %s AND "phone" IS NULL;' in executed["sql"]
    assert executed["params"] == ("5551234567", "abc-123")
    assert executed["committed"] is True


def test_extract_profile_fields_parses_location_state_code_and_contact_fields():
    text = """
    Jane Doe
    Location: Austin, TX
    Specialty: ICU RN
    Phone: (555) 123-4567
    Email: jane@example.com
    """
    extracted = extract_profile_fields_from_text(text)
    assert extracted["first_name"] == "Jane"
    assert extracted["last_name"] == "Doe"
    assert extracted["city"] == "Austin"
    assert extracted["state_code"] == "TX"
    assert extracted["specialty"] == "ICU RN"
    assert extracted["email"] == "jane@example.com"


def test_process_profile_record_marks_screen_reason_ai_when_qwen_processes(monkeypatch, tmp_path: Path):
    state = StateManager(str(tmp_path / "state.json"))
    existing_row = {
        "profile_id": "abc-123",
        "first_name": "Jane",
        "last_name": "Doe",
        "city": "Austin",
        "state_code": "TX",
        "specialty": "ICU RN",
        "phone": "5551234567",
        "email": "jane@example.com",
        "screen_reason": None,
    }
    saved_updates = {}

    monkeypatch.setitem(processor.CONFIG, "skip_already_processed", False)
    monkeypatch.setattr(processor, "download_file_bytes", lambda _: b"%PDF-1.4")
    monkeypatch.setattr(
        processor,
        "extract_text_layers_from_bytes",
        lambda *args, **kwargs: {
            "native_text": "Jane Doe resume text",
            "ocr_text": "",
            "combined_text": "Jane Doe resume text",
        },
    )
    monkeypatch.setattr(processor, "extract_profile_fields_from_text", lambda _: {})
    monkeypatch.setattr(processor, "run_ai_extraction", lambda *_: {})
    monkeypatch.setattr(processor, "sanitize_extracted", lambda extracted, **kwargs: extracted)
    monkeypatch.setattr(processor, "fetch_profile_by_id", lambda _: existing_row)
    monkeypatch.setattr(processor, "email_belongs_to_another_profile", lambda *args: False)

    def _capture_update(profile_id, updates):
        saved_updates["profile_id"] = profile_id
        saved_updates["updates"] = updates

    monkeypatch.setattr(processor, "update_profile_fields", _capture_update)

    outcome = processor.process_profile_record(
        {"profile_id": "abc-123", "resume_url": "resume.pdf"},
        state,
    )

    assert outcome == "processed"
    assert saved_updates == {
        "profile_id": "abc-123",
        "updates": {"screen_reason": "AI"},
    }


def test_process_profile_record_skips_ai_when_missing_fields_are_not_actionable(monkeypatch, tmp_path: Path):
    state = StateManager(str(tmp_path / "state.json"))
    existing_row = {
        "profile_id": "abc-123",
        "first_name": "Jane",
        "last_name": "Doe",
        "city": "Austin",
        "state_code": "TX",
        "specialty": None,
        "phone": "5551234567",
        "email": "jane@example.com",
        "screen_reason": None,
    }
    saved_updates = {}

    monkeypatch.setitem(processor.CONFIG, "skip_already_processed", False)
    monkeypatch.setattr(processor, "download_file_bytes", lambda _: b"%PDF-1.4")
    monkeypatch.setattr(
        processor,
        "extract_text_layers_from_bytes",
        lambda *args, **kwargs: {
            "native_text": "Profession: ICU RN",
            "ocr_text": "",
            "combined_text": "Profession: ICU RN",
        },
    )
    monkeypatch.setattr(processor, "extract_profile_fields_from_text", lambda _: {"specialty": "ICU RN"})
    monkeypatch.setattr(
        processor,
        "run_ai_extraction",
        lambda *_: (_ for _ in ()).throw(AssertionError("Qwen should not run when only already-filled fields are missing")),
    )
    monkeypatch.setattr(processor, "sanitize_extracted", lambda extracted, **kwargs: extracted)
    monkeypatch.setattr(processor, "email_belongs_to_another_profile", lambda *args: False)

    def _capture_update(profile_id, updates):
        saved_updates["profile_id"] = profile_id
        saved_updates["updates"] = updates

    monkeypatch.setattr(processor, "update_profile_fields", _capture_update)

    outcome = processor.process_profile_record(
        {"profile_id": "abc-123", "resume_url": "resume.pdf"},
        state,
        existing=existing_row,
    )

    assert outcome == "processed"
    assert saved_updates == {
        "profile_id": "abc-123",
        "updates": {"specialty": "ICU RN"},
    }
