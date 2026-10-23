"""Pure-logic tests for profile_dedup - no DB connection required."""
from datetime import datetime

from profile_dedup.matching import HIGH, MEDIUM, cluster_phone_group
from profile_dedup.merge_logic import build_merge_plan
from profile_dedup.normalize import is_empty_value, normalize_name, normalize_phone
from profile_dedup.primary import select_primary


def test_normalize_phone_strips_formatting():
    assert normalize_phone("(555) 123-4567") == "5551234567"
    assert normalize_phone("+1 555-123-4567") == "5551234567"


def test_normalize_phone_rejects_placeholders():
    assert normalize_phone("0000000000") is None
    assert normalize_phone("123") is None
    assert normalize_phone(None) is None


def test_normalize_name():
    assert normalize_name("Jo-Ann", "O'Brien") == "jo ann o brien"
    assert normalize_name(" Bob ", "Smith") == normalize_name("bob", "SMITH")


def test_is_empty_value_rules():
    assert is_empty_value("", kind="text")
    assert is_empty_value("  ", kind="text")
    assert not is_empty_value("x", kind="text")
    assert not is_empty_value(0, kind="numeric")  # completion_score=0 is real
    assert is_empty_value(0, kind="zeroable_numeric")  # years_experience=0 is a placeholder
    assert is_empty_value([], kind="json")
    assert is_empty_value("[]", kind="json")
    assert not is_empty_value([{"a": 1}], kind="json")
    assert not is_empty_value(False, kind="bool")  # False is a real answer


def _row(**overrides):
    base = dict(
        profile_id="p1",
        first_name="John",
        last_name="Doe",
        headline=None,
        bio=None,
        specialty=None,
        profession_type=None,
        provider_category=None,
        american_board=None,
        board=None,
        license_number=None,
        organization_name=None,
        job_category=None,
        work_authorization=None,
        years_experience=0,
        city=None,
        state_code=None,
        practice_state=None,
        zip_code=None,
        address=None,
        lat=None,
        lng=None,
        npi_number=None,
        resume_url=None,
        profile_photo_url=None,
        resume_sections=None,
        education=None,
        search_text=None,
        phone="5551234567",
        email=None,
        open_to_work=False,
        job_type_prefs=[],
        pay_min_hourly=None,
        available_date=None,
        completion_score=10,
        contact_updated_by_user_id=None,
        contact_updated_by_email=None,
        contact_updated_at=None,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        is_listable=True,
        merged_into=None,
        merged_at=None,
        source="manual",
        screen_reason=None,
        user_id=None,
        name_prefix=None,
        credential=None,
    )
    base.update(overrides)
    return base


def test_merge_fills_blanks_without_overwriting():
    primary = _row(profile_id="primary", resume_url=None, completion_score=20)
    duplicate = _row(profile_id="dup", resume_url="https://example.com/resume.pdf", npi_number="1234567890")

    plan = build_merge_plan(primary, [duplicate])

    assert plan["merged_values"]["resume_url"] == "https://example.com/resume.pdf"
    assert plan["merged_values"]["npi_number"] == "1234567890"
    assert plan["field_sources"]["resume_url"] == "dup"
    assert plan["conflicts"] == []


def test_merge_never_overwrites_verified_contact_with_unverified():
    primary = _row(
        profile_id="primary",
        phone="5551234567",
        contact_updated_by_user_id="user-1",
        contact_updated_at=datetime(2026, 2, 1),
    )
    duplicate = _row(profile_id="dup", phone="5559999999")

    plan = build_merge_plan(primary, [duplicate])

    assert plan["merged_values"]["phone"] == "5551234567"
    assert plan["field_sources"]["phone"] == "primary"


def test_merge_flags_unresolvable_conflict_without_overwriting():
    primary = _row(profile_id="primary", specialty="Cardiology")
    duplicate = _row(profile_id="dup", specialty="Neurology")

    plan = build_merge_plan(primary, [duplicate])

    assert plan["merged_values"]["specialty"] == "Cardiology"
    assert any(c["field"] == "specialty" for c in plan["conflicts"])


def test_merge_combines_info_from_multiple_duplicates():
    primary = _row(profile_id="primary")
    dup_with_resume = _row(profile_id="dup1", resume_url="https://example.com/r.pdf")
    dup_with_npi_and_address = _row(profile_id="dup2", npi_number="1234567890", address="123 Main St")
    dup_with_experience = _row(profile_id="dup3", years_experience=12)

    plan = build_merge_plan(primary, [dup_with_resume, dup_with_npi_and_address, dup_with_experience])

    assert plan["merged_values"]["resume_url"] == "https://example.com/r.pdf"
    assert plan["merged_values"]["npi_number"] == "1234567890"
    assert plan["merged_values"]["address"] == "123 Main St"
    assert plan["merged_values"]["years_experience"] == 12


def test_select_primary_prefers_npi_and_completion():
    bare = _row(profile_id="bare", completion_score=90)
    verified = _row(profile_id="verified", npi_number="1234567890", completion_score=10)

    assert select_primary([bare, verified])["profile_id"] == "verified"


def test_cluster_phone_group_splits_exact_and_fuzzy_names():
    exact_a = _row(profile_id="a1", first_name="John", last_name="Doe")
    exact_b = _row(profile_id="a2", first_name="John", last_name="Doe")
    fuzzy_a = _row(profile_id="b1", first_name="Katherine", last_name="Lee")
    fuzzy_b = _row(profile_id="b2", first_name="Katherin", last_name="Lee")

    clusters = cluster_phone_group([exact_a, exact_b, fuzzy_a, fuzzy_b])

    confidences = {c.confidence for c in clusters}
    assert HIGH in confidences
    assert MEDIUM in confidences


def test_cluster_phone_group_never_merges_when_npi_numbers_disagree():
    org_a = _row(profile_id="org1", first_name="HEALTH INSURANCE ORG.", last_name="HEALTH INSURANCE ORG.", npi_number="1124331434")
    org_b = _row(profile_id="org2", first_name="HEALTH INSURANCE ORG.", last_name="HEALTH INSURANCE ORG.", npi_number="1063726438")

    clusters = cluster_phone_group([org_a, org_b])

    assert clusters == []


def test_cluster_phone_group_ignores_unrelated_names():
    unrelated_a = _row(profile_id="x1", first_name="Alice", last_name="Smith")
    unrelated_b = _row(profile_id="x2", first_name="Bob", last_name="Jones")

    clusters = cluster_phone_group([unrelated_a, unrelated_b])

    assert clusters == []
