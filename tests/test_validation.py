import logging

import pytest

from profile_enrichment import validation
from profile_enrichment.processor import _verify_and_merge_fields

RESUME = """
Patrick Vrtis
I'm Patrick, a dedicated professional.
Registered Nurse - Cardiology
Austin, TX
patrick.vrtis@hospital.org
(512) 887-2043
"""


@pytest.mark.parametrize(
    "value",
    [
        "johndoe@example.com",
        "john@example.com",
        "jane.doe@example.org",
        "test@test.com",
        "user@domain.com",
    ],
)
def test_placeholder_emails_are_rejected(value):
    ok, _, reason = validation.validate_field("email", value, RESUME)
    assert not ok
    assert reason in {"placeholder_email", "not_found_in_source"}


@pytest.mark.parametrize("value", ["123-456-7890", "(555) 123-4567", "000-000-0000", "9876543210"])
def test_placeholder_phones_are_rejected(value):
    ok, _, reason = validation.validate_field("phone", value, RESUME)
    assert not ok
    assert reason != "ok"


@pytest.mark.parametrize(
    "value",
    ["John Doe", "I'm Patrick", "Certified Assistant", "Registered Nurse", "Resume", "Contact"],
)
def test_bad_names_are_rejected(value):
    ok, _, reason = validation.validate_field("first_name", value, RESUME)
    assert not ok, f"{value} should not be accepted (reason={reason})"


def test_real_values_from_the_source_are_accepted():
    assert validation.validate_field("first_name", "Patrick", RESUME)[0]
    assert validation.validate_field("last_name", "Vrtis", RESUME)[0]
    assert validation.validate_field("email", "patrick.vrtis@hospital.org", RESUME)[0]
    assert validation.validate_field("phone", "(512) 887-2043", RESUME)[0]
    assert validation.validate_field("city", "Austin", RESUME)[0]
    assert validation.validate_field("state_code", "tx", RESUME)[1] == "TX"
    assert validation.validate_field("specialty", "Cardiology", RESUME)[0]


def test_values_absent_from_the_source_are_rejected():
    for field, value in [
        ("city", "New York"),
        ("specialty", "Dermatology"),
        ("state_code", "NY"),
        ("email", "someone.else@hospital.org"),
    ]:
        ok, _, reason = validation.validate_field(field, value, RESUME)
        assert not ok and reason == "not_found_in_source", f"{field}={value} slipped through"


def test_rejected_values_are_logged():
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    validation.logger.addHandler(handler)
    try:
        cleaned = validation.validate_extracted(
            {"email": "johndoe@example.com", "first_name": "Patrick"},
            source_text=RESUME,
            profile_id="p-1",
        )
    finally:
        validation.logger.removeHandler(handler)

    assert cleaned == {"first_name": "Patrick"}
    assert records == [
        "REJECTED_AI_VALUE | profile_id=p-1 | field=email | "
        "value=johndoe@example.com | reason=placeholder_email"
    ]


def test_null_tokens_never_reach_the_database():
    assert validation.validate_extracted(
        {"city": "N/A", "specialty": "unknown", "phone": "none", "email": ""},
        source_text=RESUME,
    ) == {}


def test_existing_name_confirmed_by_source_is_not_overwritten():
    existing = {"first_name": "Patrick", "last_name": "Vrtis"}
    updates, changes = _verify_and_merge_fields(
        "p-1", existing, {"first_name": "Patrick", "last_name": "Patrick"}, source_text=RESUME
    )
    assert updates == {}
    assert changes == {}


def test_blank_columns_are_filled_from_verified_values():
    existing = {"first_name": "Patrick", "last_name": "Vrtis", "city": None, "specialty": ""}
    updates, _ = _verify_and_merge_fields(
        "p-1", existing, {"city": "Austin", "specialty": "Cardiology"}, source_text=RESUME
    )
    assert updates == {"city": "Austin", "specialty": "Cardiology"}
