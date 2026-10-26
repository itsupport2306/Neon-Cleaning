import csv

from profile_enrichment import validation
from profile_enrichment.inference import FieldCandidate
from profile_enrichment.resolution import DocView, build_proposal, resolve_fields
from profile_enrichment.specialty_lexicon import DEFAULT_ALIAS_PATH, DEFAULT_VOCABULARY_PATH, SpecialtyLexicon

LEX = SpecialtyLexicon.load()


def decide(text, extra=()):
    return LEX.decide(text, extra)


def test_every_reviewed_alias_maps_to_a_label_that_exists_in_the_vocabulary():
    labels = {row["specialty"] for row in csv.DictReader(open(DEFAULT_VOCABULARY_PATH, encoding="utf-8"))}
    for row in csv.DictReader(open(DEFAULT_ALIAS_PATH, encoding="utf-8")):
        if row["reviewed"] == "yes":
            assert row["label"] in labels, row


def test_bare_pa_is_not_an_alias_for_physician_assistant():
    assert LEX.normalize_label("PA") is None
    assert LEX.normalize_label("PA-C") == ("Physician Assistant", "alias")
    assert decide("Jane Roe\nAustin, PA\nPatient clinical hospital notes\n").status != "selected"


def test_explicit_specialty_line_wins():
    d = decide("Pat Lee\nSpecialty: Family Medicine\nExperience\nOffice manager")
    assert d.status == "selected" and d.label == "Family Medicine" and d.evidence == "Specialty: Family Medicine"


def test_practice_area_beats_the_profession_when_both_are_present():
    d = decide("Jane Doe, RN\nICU Registered Nurse\nSummary\nCompassionate bedside nursing care of patients in hospital.\n")
    assert (d.status, d.label) == ("selected", "ICU")


def test_profession_is_the_fallback_when_no_practice_area_is_stated():
    d = decide("Jane Doe\nRegistered Nurse\nSummary\nCompassionate nursing care of patients in the hospital.\n")
    assert (d.status, d.label) == ("selected", "Registered Nurse")


def test_degree_lines_are_education_not_specialty():
    text = ("Arun Agrawal, MD\nEducation\n3/1975 to 3/1978 Master of Surgery, Delhi University, India\n"
            "Patient care, clinical practice at the hospital.\n")
    d = decide(text)
    assert d.status != "selected" or d.label != "Surgery"


def test_single_skills_bullet_is_not_enough():
    text = ("Cheryl Hernandez\nProfile\nExperienced healthcare worker.\nSkills\n"
            "Critical Care Experience Medication Administration, Including Critical Drips\nPatient hospital clinical\n")
    assert decide(text).status != "selected"


def test_employer_names_do_not_count():
    text = "Lee Park\nRN\nExperience\nCardiology Associates Medical Group\nPatient care at hospital\n"
    d = decide(text)
    assert d.label != "Cardiology"


def test_non_clinical_resume_has_no_specialty_even_if_a_vocabulary_word_appears():
    d = decide("Sam Porter\nNetwork & Security Engineer\nSkills\nJava, CAD, Security, Sales\nWarehouse operations\n")
    assert d.status == "none" and d.reason == "no_healthcare_context"


def test_a_mention_is_scored_by_position_not_counted_blindly():
    header = decide("Casey Roe\nOncology Nurse Practitioner\nPatient clinical hospital\n")
    body = decide("Casey Roe\nWork history\nHelped patients at the clinic.\nSome oncology exposure during rotations.\n")
    assert header.status == "selected" and header.label == "Oncology"
    assert body.status != "selected"


def test_model_can_corroborate_but_never_introduce_a_specialty():
    text = "Jane Doe\nRegistered Nurse\nICU and telemetry\nPatient clinical hospital\n"
    assert decide(text, [("Cardiology", "Jane Doe")]).label != "Cardiology"
    boosted = decide(text, [("ICU", "ICU and telemetry")])
    assert boosted.label == "ICU" and boosted.score > decide(text).score


def row(**kw):
    base = {"profile_id": "p", "first_name": "Unknown", "last_name": "Candidate", "city": None, "state_code": None,
            "specialty": None, "phone": None, "email": None}
    base.update(kw)
    return base


def resolve(text, db_row, candidates=()):
    return resolve_fields(DocView([(1, text)]), db_row, list(candidates), LEX)


RESUME = ("Jordan Lee, RN\nAustin, TX\nPhone: (512) 887-2043 | Jordan.Lee@Hospital.org\nICU Registered Nurse\n"
          "Experience\nPatient care in the hospital.\nReferences\nDana Fox (210) 555-0188 dana.fox@clinic.org\n")


def test_regex_does_not_win_automatically_the_better_supported_candidate_does():
    # regex finds the reference's email/phone first in a naive scan of the body; the header candidates score higher
    res = resolve(RESUME, row()).resolved
    assert res["email"].value == "Jordan.Lee@Hospital.org" and res["phone"].value == "(512) 887-2043"
    qwen = [FieldCandidate("email", "dana.fox@clinic.org", "x", 1, 0), FieldCandidate("phone", "(210) 555-0188", "x", 1, 0)]
    res = resolve(RESUME, row(), qwen).resolved  # a model pick from the references section does not override the header
    assert res["email"].value == "Jordan.Lee@Hospital.org" and res["phone"].value == "(512) 887-2043"


def test_agreement_between_sources_raises_confidence():
    base = resolve(RESUME, row()).resolved["email"].score
    agree = resolve(RESUME, row(), [FieldCandidate("email", "jordan.lee@hospital.org", "x", 1, 0)]).resolved["email"]
    assert agree.score > base and agree.source == "both" and agree.value == "Jordan.Lee@Hospital.org"


def test_generic_mailboxes_lose_to_personal_ones():
    text = "Pat Quinn\nhr@company.org | pat.quinn@mail.com\nRN\nPatient care hospital\n"
    assert resolve(text, row()).resolved["email"].value == "pat.quinn@mail.com"


def test_conflicting_equal_candidates_resolve_to_nothing():
    text = "Pat Quinn\nRN patient hospital\n(512) 887-2043 | (737) 555-0111\n"
    res = resolve(text, row())
    assert "phone" not in res.resolved and res.notes["phone"] == "conflicting_candidates"


def test_names_are_resolved_as_a_pair_and_multi_part_last_names_prefer_the_model_evidence():
    text = "Saskya Jean Charles\nWebster, MA\nPhlebotomist\nPatient clinical\n"
    res = resolve(text, row(), [FieldCandidate("first_name", "Saskya", "Saskya Jean Charles", 1, 0),
                                FieldCandidate("last_name", "Jean Charles", "Saskya Jean Charles", 1, 0)]).resolved
    assert (res["first_name"].value, res["last_name"].value) == ("Saskya", "Jean Charles")


def test_city_and_state_come_from_the_same_line():
    text = "Pat Quinn\nAustin, TX 78701\nRN patient hospital\n"
    res = resolve(text, row()).resolved
    assert (res["city"].value, res["state_code"].value) == ("Austin", "TX")
    assert res["city"].evidence == res["state_code"].evidence


def test_two_equally_placed_locations_are_ambiguous_so_neither_field_is_guessed():
    text = "Pat Quinn\n123 Main St, Dayton, OH\nAustin, TX 78701\nRN patient hospital\n"
    res = resolve(text, row())
    assert "city" not in res.resolved and "state_code" not in res.resolved


def test_a_state_name_deep_in_the_body_is_not_the_candidates_state():
    text = "Pat Quinn\nRN\n" + "Patient care.\n" * 80 + "Worked at a hospital in Texas for two years.\n"
    assert "state_code" not in resolve(text, row()).resolved


def test_proposal_only_fills_blank_and_placeholder_fields_and_queues_the_rest_for_review():
    existing = row(first_name="Pat", last_name="Quinn", city="Dallas", phone="5550000000", email="Keep@Case.com")
    fills, reviews = build_proposal(existing, resolve(RESUME, existing))
    assert set(fills) == {"specialty", "state_code"}
    assert "phone" not in fills and "phone" not in reviews  # an existing phone is never even proposed for change
    assert "email" in reviews and "city" in reviews
    assert "Keep@Case.com" != reviews["email"].value


def test_same_email_differing_only_in_case_is_not_a_correction():
    existing = row(email="jordan.lee@HOSPITAL.org")
    fills, reviews = build_proposal(existing, resolve(RESUME, existing))
    assert "email" not in fills and "email" not in reviews


def test_placeholder_values_count_as_blank_but_wrong_looking_values_do_not():
    assert validation.is_blank_or_placeholder("first_name", "Unknown")
    assert validation.is_blank_or_placeholder("last_name", "candidate")
    assert validation.is_blank_or_placeholder("city", None)
    assert not validation.is_blank_or_placeholder("first_name", "Class")  # a correction, not a blank


# ---- validation -------------------------------------------------------------------------------

def test_weak_prefix_grounding_no_longer_accepts_inspect_as_a_specialty():
    source = "Dana Jenkins\nComputer Clerk\nUniversal Leaf\nInspection of leaf shipments\n"
    ok, _, reason = validation.validate_field("specialty", "Inspect", source)
    assert not ok


def test_specialty_grounded_by_a_reviewed_equivalent_phrase():
    source = "Jane Doe, FNP-C\nPatient clinical care\n"
    ok, value, _ = validation.validate_field("specialty", "Family Nurse Practitioner", source)
    assert ok and value == "Family Nurse Practitioner"  # the label is not literally in the text; FNP-C is its reviewed alias


def test_credentials_degrees_and_skills_are_not_specialties():
    for bad in ("BLS", "ACLS", "BSN", "Epic", "Python"):
        assert not validation.validate_field("specialty", bad, f"{bad} certified", require_grounding=False)[0]


def test_email_capitalisation_is_preserved_by_validation():
    ok, value, _ = validation.validate_field("email", "Jordan.Lee@Hospital.org", "x Jordan.Lee@Hospital.org")
    assert ok and value == "Jordan.Lee@Hospital.org"
