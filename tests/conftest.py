"""Unit tests must never reach the production database, a model server, or the real log/state files."""

import os
import tempfile

_SCRATCH = tempfile.mkdtemp(prefix="enrichment-tests-")
# Set before profile_enrichment.config is imported: .env is loaded with override=False, so these win.
os.environ["LOG_DIR"] = os.path.join(_SCRATCH, "logs")
os.environ["STATE_FILE"] = os.path.join(_SCRATCH, "state", "enrichment_state.json")
os.environ["JOB_STORE_PATH"] = os.path.join(_SCRATCH, "state", "jobs.sqlite3")
os.environ["CACHE_DIR"] = os.path.join(_SCRATCH, "cache")
os.environ["AI_RETRY_BACKOFF"] = "0"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_database_or_network(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return

    def refuse(*_args, **_kwargs):
        raise RuntimeError("unit tests must not open a database connection (mark with @pytest.mark.live to allow)")

    import psycopg2
    import psycopg2.pool

    from profile_enrichment import neon_db

    monkeypatch.setattr(neon_db, "get_connection", refuse)
    monkeypatch.setattr(psycopg2, "connect", refuse)
    monkeypatch.setattr(psycopg2.pool, "ThreadedConnectionPool", refuse)
    monkeypatch.setattr(neon_db, "_pool", None)


def pytest_configure(config):
    config.addinivalue_line("markers", "live: test talks to the real database/model server")
