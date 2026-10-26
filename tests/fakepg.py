"""A tiny in-process stand-in for the Postgres `profiles` table, backed by SQLite.

It executes the *real* SQL strings built by profile_enrichment.neon_db (CASE ... IS NOT DISTINCT FROM,
RETURNING, SAVEPOINT, a case-insensitive unique index on email) after a minimal dialect translation, so the
guards are exercised for real. Each `connect()` call is a separate connection, so "read back on a different
connection after COMMIT" is genuinely tested.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional

import psycopg2.errors

COLUMNS = ["profile_id", "first_name", "last_name", "city", "state_code", "specialty", "phone", "email",
           "screen_reason", "resume_url", "created_at", "updated_at"]


class FakeCursor:
    def __init__(self, conn: "FakeConnection"):
        self._conn = conn
        self._cur = conn._db.cursor()
        self.rowcount = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()

    @property
    def description(self):
        return self._cur.description

    def execute(self, sql: str, params=()):
        text = sql.strip()
        if re.match(r"^(SET LOCAL|SET TRANSACTION)", text, re.I):
            return
        if text.upper().startswith(("SAVEPOINT", "RELEASE SAVEPOINT", "ROLLBACK TO SAVEPOINT")):
            self._conn._db.execute(text)
            return
        text = text.replace("%s", "?")
        text = re.sub(r"\s+FOR UPDATE\b", "", text, flags=re.I)
        text = re.sub(r"IS NOT DISTINCT FROM", "IS", text, flags=re.I)
        text = re.sub(r'"public"\."profiles"|"profiles"', "profiles", text)
        self._conn.statements.append(text)
        for index, (needle, exc) in enumerate(self._conn.fail_next):
            if text.upper().startswith("UPDATE") and needle in text:
                del self._conn.fail_next[index]
                raise exc
        try:
            self._cur.execute(text, list(params))
        except sqlite3.IntegrityError as exc:
            raise psycopg2.errors.UniqueViolation(str(exc)) from exc
        self._buffer = None
        if "RETURNING" in text.upper():
            # psycopg2 reports the affected row count immediately; sqlite3 only after the rows are consumed
            self._buffer = self._cur.fetchall()
            self.rowcount = len(self._buffer)
        else:
            self.rowcount = self._cur.rowcount

    def fetchall(self):
        if getattr(self, "_buffer", None) is not None:
            rows, self._buffer = self._buffer, None
            return rows
        return self._cur.fetchall()

    def fetchone(self):
        rows = self.fetchall() if getattr(self, "_buffer", None) is not None else None
        return (rows[0] if rows else None) if rows is not None else self._cur.fetchone()


class FakeConnection:
    def __init__(self, path: str, fail_next=None):
        self._db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self._db.execute("PRAGMA foreign_keys=OFF")
        self.statements: List[str] = []
        self.fail_next: List[Any] = fail_next if fail_next is not None else []
        self.closed = 0
        self._in_tx = False

    def cursor(self):
        if not self._in_tx:
            self._db.execute("BEGIN IMMEDIATE")
            self._in_tx = True
        return FakeCursor(self)

    def commit(self):
        if self._in_tx:
            self._db.execute("COMMIT")
            self._in_tx = False

    def rollback(self):
        if self._in_tx:
            self._db.execute("ROLLBACK")
            self._in_tx = False


class FakePool:
    """Stands in for neon_db's pool; `lease()` hands out a fresh connection each time."""

    def __init__(self, path: str):
        self.path = path
        self.statements: List[str] = []
        self.fail_next: List[Any] = []
        self.leases = 0

    def make_lease(self):
        pool = self

        class Lease:
            def __init__(self):
                pool.leases += 1
                self.conn = FakeConnection(pool.path, pool.fail_next)
                self.conn.statements = pool.statements

            def __enter__(self):
                return self.conn

            def __exit__(self, exc_type, exc, tb):
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()

        return Lease()


def create_table(path: str) -> None:
    db = sqlite3.connect(path, isolation_level=None)
    db.executescript(
        """
        CREATE TABLE profiles(
          profile_id TEXT PRIMARY KEY, first_name TEXT NOT NULL, last_name TEXT NOT NULL, city TEXT, state_code TEXT,
          specialty TEXT, phone TEXT, email TEXT, screen_reason TEXT, resume_url TEXT, created_at TEXT, updated_at TEXT);
        CREATE UNIQUE INDEX ux_profiles_email_ci ON profiles(lower(email)) WHERE email IS NOT NULL AND email <> '';
        """
    )
    db.close()


def insert(path: str, **row: Any) -> None:
    values: Dict[str, Any] = {c: None for c in COLUMNS}
    values.update(first_name="Unknown", last_name="Candidate")
    values.update(row)
    db = sqlite3.connect(path, isolation_level=None)
    db.execute(f"INSERT INTO profiles({','.join(COLUMNS)}) VALUES({','.join('?' * len(COLUMNS))})", [values[c] for c in COLUMNS])
    db.close()


def get(path: str, profile_id: str) -> Optional[Dict[str, Any]]:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT * FROM profiles WHERE profile_id=?", (profile_id,)).fetchone()
    db.close()
    return dict(row) if row else None


def update(path: str, profile_id: str, **values: Any) -> None:
    db = sqlite3.connect(path, isolation_level=None)
    db.execute(f"UPDATE profiles SET {', '.join(k + '=?' for k in values)} WHERE profile_id=?", [*values.values(), profile_id])
    db.close()
