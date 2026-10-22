from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from profile_enrichment.config import CONFIG

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


def get_connection():
    if not CONFIG["neon_db_url"]:
        raise ValueError("NEON_DB_URL or DATABASE_URL is not configured")
    import psycopg2

    return psycopg2.connect(CONFIG["neon_db_url"])


def fetch_profiles_for_reprocessing(limit: int | None = None) -> List[Dict[str, Any]]:
    sql = CONFIG["neon_reprocess_query"].strip().rstrip(";")
    if limit is not None:
        sql = f"{sql} LIMIT %s"
        params: tuple[Any, ...] = (limit,)
    else:
        params = ()

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]


def fetch_reprocessing_created_at_bounds(start_created_at: datetime) -> tuple[Optional[datetime], Optional[datetime]]:
    sql = (
        "SELECT MIN(created_at), MAX(created_at) "
        f"FROM ({CONFIG['neon_reprocess_query'].strip().rstrip(';')}) AS candidate_rows "
        "WHERE created_at >= %s;"
    )
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (start_created_at,))
            row = cursor.fetchone()
            if not row:
                return None, None
            return row[0], row[1]


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

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("\n".join(sql), tuple(params))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]


def fetch_profiles_with_candidate_specialty(limit: int | None = None) -> List[Dict[str, Any]]:
    """Broad fetch of rows with a non-empty specialty; callers filter for suspicious values themselves."""
    sql = CONFIG["neon_specialty_fix_query"].strip().rstrip(";")
    if limit is not None:
        sql = f"{sql} LIMIT %s"
        params: tuple[Any, ...] = (limit,)
    else:
        params = ()

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]


def update_specialty_only(profile_id: str, specialty: str) -> None:
    """Write a corrected specialty value without touching any other column."""
    table_name = CONFIG["neon_table_name"]
    sql = f'UPDATE {table_name} SET "specialty" = %s WHERE profile_id = %s;'
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (specialty, profile_id))
        conn.commit()


def fetch_profile_by_id(profile_id: str) -> Optional[Dict[str, Any]]:
    table_name = CONFIG["neon_table_name"]
    sql = f"SELECT * FROM {table_name} WHERE profile_id = %s LIMIT 1;"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (profile_id,))
            row = cursor.fetchone()
            if not row:
                return None
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))


def email_belongs_to_another_profile(email: str, profile_id: str) -> bool:
    """True when this email is already stored on a different row.

    `profiles` carries a case-insensitive unique index on email
    (`ux_profiles_email_ci`), so writing a duplicate raises a UniqueViolation and
    aborts the whole update. Checking first lets the other verified fields land.
    """
    if not email:
        return False

    table_name = CONFIG["neon_table_name"]
    sql = f"SELECT 1 FROM {table_name} WHERE lower(email) = lower(%s) AND profile_id <> %s LIMIT 1;"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (email, profile_id))
            return cursor.fetchone() is not None


def update_profile_fields(profile_id: str, updates: Dict[str, Any]) -> None:
    if not updates:
        return

    # Last line of defence: an alias that is not a real column (e.g. "state",
    # "location") would otherwise abort the entire UPDATE statement.
    unknown = [key for key in updates if key not in WRITABLE_COLUMNS]
    if unknown:
        raise ValueError(f"Refusing to update non-profile columns: {sorted(unknown)}")

    table_name = CONFIG["neon_table_name"]
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

    sql = f"UPDATE {table_name} SET {', '.join(assignments)} WHERE {where_clause};"
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
