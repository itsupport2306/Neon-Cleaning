from profile_enrichment.extractors import extract_profile_fields_from_text
from profile_enrichment.neon_db import build_update_map
from profile_enrichment.processor import _is_reliable_value, sanitize_extracted


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
