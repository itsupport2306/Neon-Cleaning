"""guarded_update against a real SQL engine: zero-row updates, lost-update protection, email conflicts,
phone/email-case preservation, idempotence and independent read-back."""

import psycopg2.errors
import pytest

from profile_enrichment import neon_db
from profile_enrichment.neon_db import FieldChange, guarded_update
from tests import fakepg


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "pg.sqlite")
    fakepg.create_table(path)
    pool = fakepg.FakePool(path)
    monkeypatch.setattr(neon_db, "get_connection", lambda: pool.make_lease())
    monkeypatch.setitem(neon_db.CONFIG, "neon_table_name", "profiles")
    return path, pool


def test_writes_only_the_expected_row_and_reads_it_back(db):
    path, pool = db
    fakepg.insert(path, profile_id="p1")
    fakepg.insert(path, profile_id="p2")

    result = guarded_update("p1", {"specialty": FieldChange(None, "Cardiology"), "city": FieldChange(None, "Austin")},
                            screen_reason="AI")

    assert result.row_found and result.verified and not result.mismatches
    assert result.applied == {"specialty": "Cardiology", "city": "Austin"}
    assert result.screen_reason_set
    row = fakepg.get(path, "p1")
    assert (row["specialty"], row["city"], row["screen_reason"]) == ("Cardiology", "Austin", "AI")
    assert fakepg.get(path, "p2")["specialty"] is None  # the neighbouring row is untouched
    assert pool.leases >= 2  # write transaction and the independent read-back used different connections


def test_zero_row_update_is_reported_not_swallowed(db):
    path, _ = db
    result = guarded_update("does-not-exist", {"specialty": FieldChange(None, "Cardiology")})
    assert result.row_found is False and result.applied == {} and not result.verified


def test_rowcount_other_than_one_aborts_the_transaction(db, monkeypatch):
    path, pool = db
    fakepg.insert(path, profile_id="p1")
    real = fakepg.FakeCursor.execute

    def lying_execute(self, sql, params=()):
        real(self, sql, params)
        if sql.strip().upper().startswith("UPDATE"):
            self.rowcount = 2

    monkeypatch.setattr(fakepg.FakeCursor, "execute", lying_execute)
    with pytest.raises(RuntimeError, match="expected 1"):
        guarded_update("p1", {"specialty": FieldChange(None, "Cardiology")})
    assert fakepg.get(path, "p1")["specialty"] is None  # rolled back


def test_value_changed_after_the_read_is_not_overwritten(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1", city="Dallas")
    # we read city=None / specialty=None earlier; meanwhile someone else wrote city and specialty
    fakepg.update(path, "p1", city="Houston", specialty="ICU")

    result = guarded_update("p1", {"city": FieldChange(None, "Austin"), "specialty": FieldChange(None, "Cardiology")})

    assert result.applied == {}
    assert result.concurrent_change == {"city": "Houston", "specialty": "ICU"}
    row = fakepg.get(path, "p1")
    assert (row["city"], row["specialty"]) == ("Houston", "ICU")


def test_one_concurrent_field_does_not_block_the_others(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1")
    fakepg.update(path, "p1", city="Houston")

    result = guarded_update("p1", {"city": FieldChange(None, "Austin"), "specialty": FieldChange(None, "Cardiology")})

    assert result.applied == {"specialty": "Cardiology"}
    assert result.concurrent_change == {"city": "Houston"}
    assert fakepg.get(path, "p1")["city"] == "Houston"


def test_email_conflict_drops_only_the_email(db):
    path, _ = db
    fakepg.insert(path, profile_id="owner", email="Jane.Doe@Mail.com")
    fakepg.insert(path, profile_id="p1")

    result = guarded_update(
        "p1",
        {"email": FieldChange(None, "jane.doe@mail.com"), "specialty": FieldChange(None, "Cardiology"),
         "city": FieldChange(None, "Austin")},
        screen_reason="AI",
    )

    assert result.email_conflict
    assert result.applied == {"specialty": "Cardiology", "city": "Austin"}
    row = fakepg.get(path, "p1")
    assert row["email"] is None and row["specialty"] == "Cardiology" and row["screen_reason"] == "AI"
    assert fakepg.get(path, "owner")["email"] == "Jane.Doe@Mail.com"  # the other profile keeps its capitalisation


def test_email_race_between_check_and_write_is_handled_by_the_database(db):
    """Even when the uniqueness check would have said 'free', the unique index decides - and we recover."""
    path, pool = db
    fakepg.insert(path, profile_id="p1")
    fakepg.insert(path, profile_id="p2")
    pool.fail_next.append(("email", psycopg2.errors.UniqueViolation("duplicate key value violates ux_profiles_email_ci")))

    result = guarded_update("p1", {"email": FieldChange(None, "a@b.org"), "specialty": FieldChange(None, "Cardiology")})

    assert result.email_conflict and result.applied == {"specialty": "Cardiology"}


def test_email_capitalisation_from_the_resume_is_stored_as_written(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1")
    result = guarded_update("p1", {"email": FieldChange(None, "Priya.Raman@Mailbox.org")})
    assert result.verified
    assert fakepg.get(path, "p1")["email"] == "Priya.Raman@Mailbox.org"


def test_existing_phone_is_never_replaced(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1", phone="5550000000")
    with pytest.raises(ValueError, match="phone"):
        guarded_update("p1", {"phone": FieldChange("5550000000", "6145550182")})
    # and even a stale "NULL" expectation cannot clobber a phone that appeared in the meantime
    result = guarded_update("p1", {"phone": FieldChange(None, "6145550182")})
    assert result.concurrent_change == {"phone": "5550000000"}
    assert fakepg.get(path, "p1")["phone"] == "5550000000"


def test_empty_values_are_refused(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1", city="Austin")
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            guarded_update("p1", {"city": FieldChange("Austin", bad)})
    assert fakepg.get(path, "p1")["city"] == "Austin"


def test_unknown_columns_are_refused(db):
    with pytest.raises(ValueError, match="non-profile"):
        guarded_update("p1", {"state": FieldChange(None, "TX")})


def test_writes_are_idempotent(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1")
    first = guarded_update("p1", {"specialty": FieldChange(None, "Cardiology")}, screen_reason="AI")
    second = guarded_update("p1", {"specialty": FieldChange(None, "Cardiology")}, screen_reason="AI")
    assert first.applied == {"specialty": "Cardiology"}
    assert second.applied == {} and second.already_present == ["specialty"] and second.verified
    assert fakepg.get(path, "p1")["specialty"] == "Cardiology"


def test_screen_reason_is_not_set_when_nothing_was_written(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1", specialty="ICU")
    result = guarded_update("p1", {"specialty": FieldChange(None, "Cardiology")}, screen_reason="AI")
    assert not result.screen_reason_set
    assert fakepg.get(path, "p1")["screen_reason"] is None


def test_placeholder_name_can_be_replaced_when_it_is_still_the_placeholder(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1", first_name="Unknown", last_name="Candidate")
    result = guarded_update("p1", {"first_name": FieldChange("Unknown", "Priya"), "last_name": FieldChange("Candidate", "Raman")})
    assert result.applied == {"first_name": "Priya", "last_name": "Raman"}


def test_revert_only_undoes_our_own_value(db):
    path, _ = db
    fakepg.insert(path, profile_id="p1", specialty="Cardiology")
    assert neon_db.revert_field("p1", "specialty", "Cardiology", None) is True
    assert fakepg.get(path, "p1")["specialty"] is None
    fakepg.update(path, "p1", specialty="ICU")
    assert neon_db.revert_field("p1", "specialty", "Cardiology", None) is False
    assert fakepg.get(path, "p1")["specialty"] == "ICU"


def test_table_identifier_is_validated():
    neon_db.CONFIG["neon_table_name"], saved = "profiles; DROP TABLE x", neon_db.CONFIG["neon_table_name"]
    try:
        with pytest.raises(ValueError):
            neon_db.table_parts()
    finally:
        neon_db.CONFIG["neon_table_name"] = saved
    assert neon_db._table_sql() == '"profiles"' or neon_db._table_sql().endswith('"profiles"')


def test_pool_is_bounded_and_shared(monkeypatch):
    monkeypatch.setitem(neon_db.CONFIG, "db_pool_max", 3)
    monkeypatch.setitem(neon_db.CONFIG, "neon_db_url", "postgresql://u:p@localhost/db")
    created = []

    class FakePoolClass:
        def __init__(self, minconn, maxconn, dsn, **kwargs):
            created.append((minconn, maxconn, dsn, kwargs))

    import psycopg2.pool

    monkeypatch.setattr(psycopg2.pool, "ThreadedConnectionPool", FakePoolClass)
    first = neon_db._get_pool()
    second = neon_db._get_pool()
    assert first is second and len(created) == 1
    assert created[0][1] == 3 and created[0][3]["connect_timeout"] == neon_db.CONFIG["db_connect_timeout"]


def test_selection_filter_adds_screen_reason_only_when_configured(monkeypatch):
    from profile_enrichment import neon_db
    from profile_enrichment.config import CONFIG

    monkeypatch.setitem(CONFIG, "null_specialty_screen_reason", "")
    assert neon_db._selection_filter() == ("specialty IS NULL", ())
    monkeypatch.setitem(CONFIG, "null_specialty_screen_reason", "AI")
    where, params = neon_db._selection_filter()
    assert "screen_reason" in where and params == ("AI",)
    assert neon_db.selection_description().endswith("AND screen_reason = 'AI'")
