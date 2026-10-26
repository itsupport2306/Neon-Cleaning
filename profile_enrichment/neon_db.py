"""NeonDB access: a bounded connection pool, exact-`profile_id` reads and guarded, verified writes.

Write rules (all enforced in SQL, not just in Python):

* every UPDATE targets one ``profile_id`` and is checked with ``RETURNING`` + an exact row-count assertion;
* each column is assigned with ``CASE WHEN col IS NOT DISTINCT FROM <value we read> THEN <new> ELSE col END``,
  so a value changed by anyone after we read the row is never overwritten (lost-update protection);
* ``phone`` is only ever written while it is still NULL;
* a unique-index conflict on ``email`` is caught with a SAVEPOINT and only the email is dropped - the other
  verified fields in the same statement still land;
* after COMMIT the row is read back on a *different* connection and compared with what was meant to be written.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from profile_enrichment.config import CONFIG

logger = logging.getLogger(__name__)

PROFILE_COLUMNS = (
    "first_name",
    "last_name",
    "city",
    "state_code",
    "specialty",
    "phone",
    "email",
)

WRITABLE_COLUMNS = PROFILE_COLUMNS + ("screen_reason",)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------------------------

def table_parts() -> Tuple[str, ...]:
    """The configured table as validated identifier parts, e.g. ('public', 'profiles')."""
    parts = tuple(CONFIG["neon_table_name"].strip().split("."))
    if not 1 <= len(parts) <= 2 or not all(_IDENT_RE.match(p) for p in parts):
        raise ValueError(f"NEON_TABLE_NAME is not a plain [schema.]table identifier: {CONFIG['neon_table_name']!r}")
    return parts


def _table_sql() -> str:
    """Quoted, validated table identifier for plain-string SQL."""
    return ".".join(f'"{p}"' for p in table_parts())


def _column_sql(name: str) -> str:
    if name not in WRITABLE_COLUMNS and name not in {"profile_id", "resume_url", "created_at", "updated_at"}:
        raise ValueError(f"Refusing to reference non-profile column: {name!r}")
    return f'"{name}"'


# ---------------------------------------------------------------------------------------------
# connection pool
# ---------------------------------------------------------------------------------------------

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if not CONFIG["neon_db_url"]:
        raise ValueError("NEON_DB_URL or DATABASE_URL is not configured")
    with _pool_lock:
        if _pool is None:
            from psycopg2.pool import ThreadedConnectionPool

            _pool = ThreadedConnectionPool(
                1, CONFIG["db_pool_max"], CONFIG["neon_db_url"], connect_timeout=CONFIG["db_connect_timeout"]
            )
        return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


class _Lease:
    """`with get_connection() as conn:` - borrow from the pool, commit/rollback, give back (never leaked)."""

    def __init__(self) -> None:
        self._pool = _get_pool()
        self.conn = self._pool.getconn()
        if self.conn.closed:  # Neon drops idle connections
            self._pool.putconn(self.conn, close=True)
            self.conn = self._pool.getconn()

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:
        broken = bool(self.conn.closed)
        try:
            if not broken:
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
        except Exception:  # noqa: BLE001
            broken = True
        self._pool.putconn(self.conn, close=broken)


def get_connection() -> _Lease:
    return _Lease()


@contextlib.contextmanager
def transaction(read_only: bool = False, timeout_ms: Optional[int] = None) -> Iterator[Any]:
    """One short transaction with a server-side statement timeout (SET LOCAL survives transaction pooling)."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            timeout = int(CONFIG["db_statement_timeout_ms"] if timeout_ms is None else timeout_ms)
            # one round trip: the link to Neon is ~0.3 s per statement
            cursor.execute(f"SET LOCAL statement_timeout = {timeout}" + ("; SET TRANSACTION READ ONLY" if read_only else ""))
        yield conn


def _retry_connection_errors(func, attempts: int = 3):
    """Re-run an idempotent DB operation when Neon dropped the connection (suspended compute, pooler restart)."""
    import time

    import psycopg2

    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return func()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            last = exc
            logger.warning("NEON_CONNECTION_RETRY | attempt=%s | error=%s", attempt + 1, exc)
            time.sleep(min(5.0, 0.5 * 2**attempt))
    raise last  # type: ignore[misc]


def _rows(cursor) -> List[Dict[str, Any]]:
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------------

SELECTION_SQL = "SELECT * FROM {table} WHERE specialty IS NULL"


def selection_description() -> str:
    return SELECTION_SQL.format(table=".".join(table_parts()))


def verify_profile_id_unique() -> bool:
    """True when `profile_id` is guarded by a plain unique/primary-key index, so one-row reads/updates are sound."""
    sql = (
        "SELECT 1 FROM pg_index i "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
        "WHERE i.indrelid = %s::regclass AND a.attname = 'profile_id' AND i.indnatts = 1 "
        "AND i.indisunique AND i.indisvalid AND i.indpred IS NULL LIMIT 1"
    )

    def run() -> bool:
        with transaction(read_only=True) as conn, conn.cursor() as cursor:
            cursor.execute(sql, (".".join(table_parts()),))
            return cursor.fetchone() is not None

    return _retry_connection_errors(run)


def count_null_specialty() -> int:
    """Exact count (a full scan of a ~12M-row table: allow minutes, not the default 30 s)."""
    def run() -> int:
        with transaction(read_only=True, timeout_ms=600_000) as conn, conn.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {_table_sql()} WHERE specialty IS NULL")
            return int(cursor.fetchone()[0])

    return _retry_connection_errors(run)


def iter_null_specialty_pages(
    batch_size: int = 5000, profile_ids: Optional[Sequence[str]] = None, sample_size: Optional[int] = None
) -> Iterator[List[Dict[str, Any]]]:
    """Pages of ``(profile_id, resume_url)`` for rows with ``specialty IS NULL`` - no date window.

    * ``profile_ids``: exact lookups (still filtered by ``specialty IS NULL``).
    * ``sample_size``: a random block sample (TABLESAMPLE), for pilots; fast on a table this large.
    * otherwise: one unordered, streaming scan on a server-side cursor. ``ORDER BY profile_id`` / keyset
      paging makes Postgres walk the primary key and filter, which times out on ~12M rows; the unordered scan
      reads the table once (~40 s). The caller stores ids with INSERT OR IGNORE, so a re-scan is harmless.
    """
    if profile_ids is not None:
        ids = list(profile_ids)
        for offset in range(0, len(ids), 2000):
            def run(chunk=ids[offset : offset + 2000]) -> List[Dict[str, Any]]:
                with transaction(read_only=True) as conn, conn.cursor() as cursor:
                    cursor.execute(f'SELECT "profile_id", "resume_url" FROM {_table_sql()} '
                                   f'WHERE specialty IS NULL AND "profile_id" = ANY(%s)', (chunk,))
                    return _rows(cursor)

            page = _retry_connection_errors(run)
            if page:
                yield page
        return

    if sample_size is not None:
        percent = 1.0
        while True:
            def run(percent=percent) -> List[Dict[str, Any]]:
                with transaction(read_only=True, timeout_ms=120_000) as conn, conn.cursor() as cursor:
                    cursor.execute(f'SELECT "profile_id", "resume_url" FROM {_table_sql()} TABLESAMPLE SYSTEM (%s) '
                                   f"WHERE specialty IS NULL LIMIT %s", (percent, sample_size))
                    return _rows(cursor)

            page = _retry_connection_errors(run)
            if len(page) >= sample_size or percent >= 100:
                yield page[:sample_size]
                return
            percent = min(100.0, percent * 4)
        return

    import psycopg2

    for attempt in range(3):
        try:
            with transaction(read_only=True, timeout_ms=0) as conn:
                cursor = conn.cursor(name="null_specialty_manifest")
                cursor.itersize = batch_size
                cursor.execute(f'SELECT "profile_id", "resume_url" FROM {_table_sql()} WHERE specialty IS NULL')
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    yield [{"profile_id": r[0], "resume_url": r[1]} for r in rows]
                cursor.close()
            return
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            logger.warning("MANIFEST_SCAN_RETRY | attempt=%s | error=%s", attempt + 1, exc)
    raise RuntimeError("manifest scan failed 3 times")


def fetch_profile_by_id(profile_id: str) -> Optional[Dict[str, Any]]:
    """Exact-id read. `profile_id` is unique (see verify_profile_id_unique); more than one row is an error."""
    sql = f'SELECT * FROM {_table_sql()} WHERE "profile_id" = %s LIMIT 2'

    def run() -> Optional[Dict[str, Any]]:
        with transaction(read_only=True) as conn, conn.cursor() as cursor:
            cursor.execute(sql, (profile_id,))
            rows = _rows(cursor)
            if len(rows) > 1:
                raise RuntimeError(f"profile_id {profile_id!r} is not unique")
            return rows[0] if rows else None

    return _retry_connection_errors(run)


# ---------------------------------------------------------------------------------------------
# guarded write + read-back
# ---------------------------------------------------------------------------------------------

@dataclass
class FieldChange:
    old: Any  # the value we read (None for NULL): the write only happens while the column still holds it
    new: Any


@dataclass
class UpdateResult:
    row_found: bool = True
    applied: Dict[str, Any] = field(default_factory=dict)  # field -> value now in the row
    already_present: List[str] = field(default_factory=list)  # row already held exactly the intended value
    concurrent_change: Dict[str, Any] = field(default_factory=dict)  # field -> value someone else put there
    email_conflict: bool = False
    screen_reason_set: bool = False
    verified: bool = False
    read_back: Dict[str, Any] = field(default_factory=dict)
    mismatches: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)  # field -> (expected, found) after commit


def _is_unique_violation(exc: Exception) -> bool:
    import psycopg2.errors

    return getattr(exc, "pgcode", None) == "23505" or isinstance(exc, psycopg2.errors.UniqueViolation)


_ALL_COLUMNS = PROFILE_COLUMNS + ("screen_reason",)


def guarded_update(
    profile_id: str, changes: Dict[str, FieldChange], screen_reason: Optional[str] = None,
) -> UpdateResult:
    """Apply `changes` to exactly one row (see module docstring), then verify the commit on a second connection.

    The row is locked (SELECT ... FOR UPDATE) so the pre-image we compare against cannot move under us; any
    column that no longer holds the value we originally read is left alone and reported as a concurrent change.
    `screen_reason` is set only when at least one field was actually written by this call.
    """
    unknown = [name for name in changes if name not in PROFILE_COLUMNS]
    if unknown:
        raise ValueError(f"Refusing to update non-profile columns: {sorted(unknown)}")
    if not changes:
        return UpdateResult(verified=True)
    for name, change in changes.items():
        if change.new is None or str(change.new).strip() == "":
            raise ValueError(f"Refusing to write an empty value to {name}")
        if name == "phone" and change.old is not None:
            raise ValueError("phone may only be written while it is NULL")

    columns_sql = ", ".join(_column_sql(c) for c in _ALL_COLUMNS)

    def write() -> UpdateResult:
        import psycopg2

        result = UpdateResult()
        with transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f'SELECT {columns_sql} FROM {_table_sql()} WHERE "profile_id" = %s FOR UPDATE', (profile_id,))
                rows = _rows(cursor)
            if not rows:
                result.row_found = False
                return result
            if len(rows) != 1:
                raise RuntimeError(f"profile_id={profile_id!r} matched {len(rows)} rows (expected 1)")
            before = rows[0]

            pending: Dict[str, FieldChange] = {}
            for name, change in changes.items():
                if before.get(name) == change.new:
                    result.already_present.append(name)
                elif before.get(name) != change.old:
                    result.concurrent_change[name] = before.get(name)
                else:
                    pending[name] = change

            while pending:
                assignments, params = [], []
                for name, change in pending.items():
                    column = _column_sql(name)
                    assignments.append(f"{column} = CASE WHEN {column} IS NOT DISTINCT FROM %s THEN %s ELSE {column} END")
                    params.extend([change.old, change.new])
                sql = (f'UPDATE {_table_sql()} SET {", ".join(assignments)} WHERE "profile_id" = %s '
                       f"RETURNING {columns_sql}")
                params.append(profile_id)
                with conn.cursor() as cursor:
                    cursor.execute("SAVEPOINT guarded_write")
                    try:
                        cursor.execute(sql, params)
                        if cursor.rowcount != 1:
                            raise RuntimeError(
                                f"UPDATE for profile_id={profile_id!r} touched {cursor.rowcount} rows (expected 1)")
                        after = _rows(cursor)[0]
                    except psycopg2.Error as exc:
                        cursor.execute("ROLLBACK TO SAVEPOINT guarded_write")
                        if _is_unique_violation(exc) and "email" in pending:
                            # Another profile owns this address (or took it a moment ago): keep the rest.
                            logger.warning("EMAIL_CONFLICT | profile_id=%s | email=%s", profile_id, pending["email"].new)
                            pending.pop("email")
                            result.email_conflict = True
                            continue
                        raise
                    cursor.execute("RELEASE SAVEPOINT guarded_write")
                for name in pending:
                    if after.get(name) == pending[name].new:
                        result.applied[name] = pending[name].new
                    else:
                        result.concurrent_change[name] = after.get(name)
                break

            if screen_reason and result.applied:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f'UPDATE {_table_sql()} SET "screen_reason" = %s WHERE "profile_id" = %s '
                        f'AND "screen_reason" IS NOT DISTINCT FROM %s',
                        (screen_reason, profile_id, before.get("screen_reason")),
                    )
                    result.screen_reason_set = cursor.rowcount == 1
        return result

    result = _retry_connection_errors(write)
    if not result.row_found:
        return result

    # independent read-back, on a different transaction/connection than the writer
    wanted = {**{n: changes[n].new for n in result.already_present}, **result.applied}
    if result.screen_reason_set:
        wanted["screen_reason"] = screen_reason
    fresh = fetch_profile_by_id(profile_id)
    if fresh is None:
        result.mismatches = {"__row__": ("present", "missing")}
        return result
    result.read_back = {k: fresh.get(k) for k in _ALL_COLUMNS}
    result.mismatches = {k: (v, fresh.get(k)) for k, v in wanted.items() if fresh.get(k) != v}
    result.verified = bool(wanted) and not result.mismatches
    return result


def revert_field(profile_id: str, name: str, expected_current: Any, restore: Any) -> bool:
    """Undo one of our writes: only if the column still holds exactly what we wrote."""
    if name not in PROFILE_COLUMNS:
        raise ValueError(f"Refusing to revert non-profile column: {name!r}")
    sql = (f'UPDATE {_table_sql()} SET {_column_sql(name)} = %s WHERE "profile_id" = %s '
           f"AND {_column_sql(name)} IS NOT DISTINCT FROM %s")

    def run() -> bool:
        with transaction() as conn, conn.cursor() as cursor:
            cursor.execute(sql, (restore, profile_id, expected_current))
            return cursor.rowcount == 1

    return _retry_connection_errors(run)


# ---------------------------------------------------------------------------------------------
# legacy helpers still used by processor.py's --reprocess-all / --fix-specialty paths
# ---------------------------------------------------------------------------------------------

def fetch_profiles_for_reprocessing(limit: int | None = None) -> List[Dict[str, Any]]:
    sql = CONFIG["neon_reprocess_query"].strip().rstrip(";")
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql = f"{sql} LIMIT %s"
        params = (limit,)
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(sql, params)
        return _rows(cursor)


def fetch_reprocessing_created_at_bounds(start_created_at: datetime) -> tuple[Optional[datetime], Optional[datetime]]:
    sql = (
        "SELECT MIN(created_at), MAX(created_at) "
        f"FROM ({CONFIG['neon_reprocess_query'].strip().rstrip(';')}) AS candidate_rows "
        "WHERE created_at >= %s;"
    )
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(sql, (start_created_at,))
        row = cursor.fetchone()
        return (row[0], row[1]) if row else (None, None)


def fetch_profiles_for_reprocessing_range(
    start_created_at: datetime,
    end_created_at: datetime,
    batch_size: int,
    last_created_at: datetime | None = None,
    last_profile_id: str | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    sql = [
        f"SELECT * FROM ({CONFIG['neon_reprocess_query'].strip().rstrip(';')}) AS candidate_rows",
        "WHERE profile_id IS NOT NULL",
        "AND created_at >= %s",
        "AND created_at < %s",
    ]
    params: List[Any] = [start_created_at, end_created_at]
    if last_created_at is not None and last_profile_id is not None:
        sql.append("AND (created_at > %s OR (created_at = %s AND profile_id > %s))")
        params.extend([last_created_at, last_created_at, last_profile_id])
    sql.append("ORDER BY created_at ASC, profile_id ASC")
    sql.append("LIMIT %s")
    params.append(min(batch_size, limit) if limit is not None else batch_size)
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute("\n".join(sql), tuple(params))
        return _rows(cursor)


def fetch_profiles_with_candidate_specialty(limit: int | None = None) -> List[Dict[str, Any]]:
    """One-shot fetch of rows `NEON_SPECIALTY_FIX_QUERY` flags as suspicious."""
    sql = CONFIG["neon_specialty_fix_query"].strip().rstrip(";")
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql = f"{sql} LIMIT %s"
        params = (limit,)
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(sql, params)
        return _rows(cursor)


def count_profiles_with_candidate_specialty() -> int:
    sql = CONFIG["neon_specialty_fix_query"].strip().rstrip(";")
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(f"SELECT count(*) FROM ({sql}) AS candidate_rows;")
        return cursor.fetchone()[0]


def fetch_profiles_with_candidate_specialty_page(
    batch_size: int, last_profile_id: str | None = None, limit: int | None = None
) -> List[Dict[str, Any]]:
    """Keyset-paginated fetch so a large number of flagged rows is never held in memory at once."""
    sql = [
        f"SELECT * FROM ({CONFIG['neon_specialty_fix_query'].strip().rstrip(';')}) AS candidate_rows",
        "WHERE profile_id IS NOT NULL",
    ]
    params: List[Any] = []
    if last_profile_id is not None:
        sql.append("AND profile_id > %s")
        params.append(last_profile_id)
    sql.append("ORDER BY profile_id ASC")
    sql.append("LIMIT %s")
    params.append(min(batch_size, limit) if limit is not None else batch_size)
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute("\n".join(sql), tuple(params))
        return _rows(cursor)


def update_specialty_only(profile_id: str, specialty: str) -> None:
    """Write a corrected specialty value without touching any other column."""
    sql = f'UPDATE {_table_sql()} SET "specialty" = %s WHERE profile_id = %s;'
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (specialty, profile_id))
        conn.commit()


def email_belongs_to_another_profile(email: str, profile_id: str) -> bool:
    """Advisory pre-check (guarded_update handles the race itself via the unique index)."""
    if not email:
        return False
    sql = f"SELECT 1 FROM {_table_sql()} WHERE lower(email) = lower(%s) AND profile_id <> %s LIMIT 1;"
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(sql, (email, profile_id))
        return cursor.fetchone() is not None


def update_profile_fields(profile_id: str, updates: Dict[str, Any]) -> None:
    if not updates:
        return
    unknown = [key for key in updates if key not in WRITABLE_COLUMNS]
    if unknown:
        raise ValueError(f"Refusing to update non-profile columns: {sorted(unknown)}")

    assignments = []
    parameters = []
    for key, value in updates.items():
        if key == "phone":
            assignments.append('"phone" = CASE WHEN "phone" IS NULL THEN %s ELSE "phone" END')
        else:
            assignments.append(f'"{key}" = %s')
        parameters.append(value)
    parameters.append(profile_id)

    where_clause = "profile_id = %s"
    if set(updates) == {"phone"}:
        where_clause += ' AND "phone" IS NULL'

    sql = f"UPDATE {_table_sql()} SET {', '.join(assignments)} WHERE {where_clause};"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, tuple(parameters))
        conn.commit()


def build_update_map(existing: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    update_map: Dict[str, Any] = {}
    field_aliases = {
        "first_name": ("first_name",),
        "last_name": ("last_name",),
        "city": ("city",),
        "state_code": ("state_code", "state"),
        "specialty": ("specialty",),
        "phone": ("phone",),
        "email": ("email",),
    }
    for db_field in PROFILE_COLUMNS:
        current_value = existing.get(db_field)
        if db_field == "phone":
            if current_value is not None:
                continue
        elif current_value not in (None, "", " "):
            continue
        for candidate_field in field_aliases[db_field]:
            candidate = extracted.get(candidate_field)
            if candidate not in (None, "", " "):
                update_map[db_field] = candidate
                break
    return update_map
