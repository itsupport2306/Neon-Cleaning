from profile_enrichment import specialty_rules as sr


def test_classify_junk_value():
    assert sr.classify_suspicious_specialty("web") == "junk_value"


def test_classify_valid_specialty_is_not_suspicious():
    assert sr.classify_suspicious_specialty("Cardiology") is None


def test_classify_leading_digit_junk():
    # OCR-garbled credential prefixes ("MD"/"DO" misread as digits).
    assert sr.classify_suspicious_specialty("00, General _ General Surgery") == "leading_digit_junk"
    assert sr.classify_suspicious_specialty("0O, General General Surgery") == "leading_digit_junk"
    assert sr.classify_suspicious_specialty("1D Pulmonology") == "leading_digit_junk"
    # A street address scraped into the specialty field by mistake.
    assert sr.classify_suspicious_specialty("19780 Castleberry Loop Oregon City, OR") == "leading_digit_junk"


def test_classify_state_name_and_code():
    assert sr.classify_suspicious_specialty("Texas") == "state_value"
    assert sr.classify_suspicious_specialty("  california ") == "state_value"
    assert sr.classify_suspicious_specialty("NY") == "state_value"
    # Not flagged: a real specialty that happens to contain a state-like word.
    assert sr.classify_suspicious_specialty("Washington University Fellow") is None


def test_classify_full_name_without_separator():
    assert sr.classify_suspicious_specialty("John Smith") == "name_full"
    assert sr.classify_suspicious_specialty("Mary Jane Watson") == "name_full"
    assert sr.classify_suspicious_specialty("Cardiology") is None
    assert sr.classify_suspicious_specialty("Family Medicine") is None


def test_classify_matches_reference_csv(tmp_path):
    csv_path = tmp_path / "unmatched.csv"
    csv_path.write_text("speciality,row_count\nSome Bad Value,12\n", encoding="utf-8")
    sr._load_reference_unmatched_values.cache_clear()

    assert sr.get_reference_unmatched_values(csv_path) == frozenset({"some bad value"})
    assert sr.classify_suspicious_specialty("Some Bad Value") is None  # default path, not tmp_path

    sr._load_reference_unmatched_values.cache_clear()


def test_build_specialty_fix_sql_includes_reference_values(tmp_path, monkeypatch):
    csv_path = tmp_path / "unmatched.csv"
    csv_path.write_text("speciality,row_count\nSome Bad Value,12\n", encoding="utf-8")
    sr._load_reference_unmatched_values.cache_clear()
    monkeypatch.setattr(sr, "DEFAULT_REFERENCE_CSV_PATH", csv_path)

    sql = sr.build_specialty_fix_sql("profiles")

    assert "some bad value" in sql
    sr._load_reference_unmatched_values.cache_clear()


def test_missing_reference_csv_returns_empty_set(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    sr._load_reference_unmatched_values.cache_clear()

    assert sr.get_reference_unmatched_values(missing) == frozenset()

    sr._load_reference_unmatched_values.cache_clear()
