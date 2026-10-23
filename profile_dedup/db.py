from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from profile_dedup.config import DEDUP_CONFIG
from profile_dedup.merge_logic import MERGE_RULES

# Columns pulled for every candidate row: merge-rule columns plus the
# bookkeeping columns matching/primary-selection/audit need.
_EXTRA_COLUMNS = (
    "profile_id",
    "contact_updated_by_user_id",
    "contact_updated_by_email",
    "contact_updated_at",
    "created_at",
    "updated_at",
    "is_listable",
    "merged_into",
    "merged_at",
    "source",
    "screen_reason",
    "user_id",
)
_SELECT_COLUMNS = tuple(dict.fromkeys([r.name for r in MERGE_RULES] + list(_EXTRA_COLUMNS)))

# Tables with a foreign key into profiles.profile_id, and what happens to
# their rows if the referenced profile is deleted. Hard-coded from the live
# schema (see README "Schema notes") rather than re-introspected on every
# run, since this list only changes with a migration.
FK_CHILD_TABLES: Tuple[Tuple[str, str], ...] = (
    ("certifications", "profile_id"),
    ("connections", "recipient_profile_id"),
    ("connections", "requester_profile_id"),
    ("licenses", "profile_id"),
    ("posts", "author_profile_id"),
    ("profile_skills", "profile_id"),
    ("work_history", "profile_id"),
    ("applications", "profile_id"),
    ("pay_packages", "profile_id"),
    ("post_comments", "author_profile_id"),
    ("post_likes", "profile_id"),
    ("saved_jobs", "profile_id"),
    ("interviews", "profile_id"),
    ("match_results", "profile_id"),
    ("offers", "profile_id"),
    ("talent_pool_members", "profile_id"),
    ("outreach_messages", "profile_id"),
    ("submissions", "profile_id"),
    ("profile_contact_backfills", "profile_id"),
)


def get_connection():
    if not DEDUP_CONFIG["neon_db_url"]:
        raise ValueError("NEON_DB_URL or DATABASE_URL is not configured")
    import psycopg2

    return psycopg2.connect(DEDUP_CONFIG["neon_db_url"])


def _rows_as_dicts(cursor) -> List[Dict[str, Any]]:
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def ensure_audit_table(conn) -> None:
    table = DEDUP_CONFIG["audit_table_name"]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                audit_id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL,
                primary_profile_id VARCHAR NOT NULL,
                duplicate_profile_id VARCHAR NOT NULL,
                match_confidence VARCHAR NOT NULL,
                match_signal VARCHAR NOT NULL,
                original_primary_data JSONB NOT NULL,
                original_duplicate_data JSONB NOT NULL,
                merged_field_values JSONB NOT NULL,
                field_sources JSONB NOT NULL,
                conflicts JSONB NOT NULL,
                action VARCHAR NOT NULL,
                merged_at TIMESTAMP NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS ix_{table}_run_id ON {table}(run_id);
            CREATE INDEX IF NOT EXISTS ix_{table}_primary ON {table}(primary_profile_id);
            CREATE INDEX IF NOT EXISTS ix_{table}_duplicate ON {table}(duplicate_profile_id);
            """
        )
    conn.commit()


def fetch_npi_duplicate_groups(limit_groups: int | None = None) -> List[List[Dict[str, Any]]]:
    table = DEDUP_CONFIG["table_name"]
    cols = ", ".join(f'"{c}"' for c in _SELECT_COLUMNS)
    sql = f"""
        SELECT {cols} FROM {table}
        WHERE merged_into IS NULL AND npi_number IS NOT NULL AND npi_number <> ''
          AND npi_number IN (
              SELECT npi_number FROM {table}
              WHERE merged_into IS NULL AND npi_number IS NOT NULL AND npi_number <> ''
              GROUP BY npi_number HAVING count(*) > 1
          )
        ORDER BY npi_number, profile_id
    """
    return _fetch_and_group(sql, (), group_key="npi_number", limit_groups=limit_groups)


def fetch_email_duplicate_groups(limit_groups: int | None = None) -> List[List[Dict[str, Any]]]:
    table = DEDUP_CONFIG["table_name"]
    cols = ", ".join(f'"{c}"' for c in _SELECT_COLUMNS)
    sql = f"""
        SELECT {cols}, lower(email) AS email_key FROM {table}
        WHERE merged_into IS NULL AND email IS NOT NULL AND email <> ''
          AND lower(email) IN (
              SELECT lower(email) FROM {table}
              WHERE merged_into IS NULL AND email IS NOT NULL AND email <> ''
              GROUP BY lower(email) HAVING count(*) > 1
          )
        ORDER BY email_key, profile_id
    """
    return _fetch_and_group(sql, (), group_key="email_key", limit_groups=limit_groups)


def fetch_phone_duplicate_groups(
    limit_groups: int | None = None,
    after_phone10: str | None = None,
) -> Tuple[List[List[Dict[str, Any]]], Optional[str]]:
    """Returns (groups, last_phone10_seen) for keyset pagination."""
    table = DEDUP_CONFIG["table_name"]
    max_size = DEDUP_CONFIG["max_phone_group_size"]
    min_digits = DEDUP_CONFIG["min_phone_digits"]
    batch = limit_groups or DEDUP_CONFIG["groups_per_batch"]
    cols = ", ".join(f'"{c}"' for c in _SELECT_COLUMNS)

    sql = f"""
        WITH normed AS (
            SELECT {cols},
                   right(regexp_replace(phone, '\\D', '', 'g'), 10) AS phone10
            FROM {table}
            WHERE merged_into IS NULL AND phone IS NOT NULL AND phone <> ''
              AND length(regexp_replace(phone, '\\D', '', 'g')) >= %(min_digits)s
        ),
        groups AS (
            SELECT phone10, count(*) AS grp_size
            FROM normed
            GROUP BY phone10
            HAVING count(*) BETWEEN 2 AND %(max_size)s
              {"AND phone10 > %(after)s" if after_phone10 else ""}
            ORDER BY phone10
            LIMIT %(batch)s
        )
        SELECT n.* FROM normed n
        JOIN groups g ON n.phone10 = g.phone10
        ORDER BY n.phone10, n.profile_id
    """
    params: Dict[str, Any] = {
        "min_digits": min_digits,
        "max_size": max_size,
        "batch": batch,
    }
    if after_phone10:
        params["after"] = after_phone10

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = _rows_as_dicts(cur)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["phone10"], []).append(row)

    last_key = max(groups.keys()) if groups else None
    return list(groups.values()), last_key


def _fetch_and_group(
    sql: str, params: Tuple[Any, ...], group_key: str, limit_groups: int | None
) -> List[List[Dict[str, Any]]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = _rows_as_dicts(cur)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    group_list = list(groups.values())
    if limit_groups is not None:
        group_list = group_list[:limit_groups]
    return group_list


def resolve_to_current_primary(conn, profile_id: str, max_depth: int = 10) -> str:
    """Follow merged_into chains so a cluster never operates on a profile_id
    that a previous run already folded into something else."""
    current = profile_id
    with conn.cursor() as cur:
        for _ in range(max_depth):
            cur.execute(
                f"SELECT merged_into FROM {DEDUP_CONFIG['table_name']} WHERE profile_id = %s",
                (current,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return current
            current = row[0]
    return current


def lock_profiles_for_update(conn, profile_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    table = DEDUP_CONFIG["table_name"]
    cols = ", ".join(f'"{c}"' for c in _SELECT_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {cols} FROM {table} WHERE profile_id = ANY(%s) FOR UPDATE",
            (profile_ids,),
        )
        rows = _rows_as_dicts(cur)
    return {row["profile_id"]: row for row in rows}


def count_fk_references(conn, profile_id: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with conn.cursor() as cur:
        for table, column in FK_CHILD_TABLES:
            cur.execute(f"SELECT count(*) FROM {table} WHERE {column} = %s", (profile_id,))
            counts[f"{table}.{column}"] = cur.fetchone()[0]
    return counts


def apply_merged_values(conn, primary_profile_id: str, merged_values: Dict[str, Any]) -> None:
    table = DEDUP_CONFIG["table_name"]
    assignments = []
    params: List[Any] = []
    for col, value in merged_values.items():
        if col in ("job_type_prefs", "resume_sections", "education") and isinstance(value, (list, dict)):
            assignments.append(f'"{col}" = %s::json')
            params.append(json.dumps(value))
        else:
            assignments.append(f'"{col}" = %s')
            params.append(value)
    params.append(primary_profile_id)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET {', '.join(assignments)} WHERE profile_id = %s",
            params,
        )


def soft_merge_duplicate(conn, duplicate_profile_id: str, primary_profile_id: str) -> None:
    table = DEDUP_CONFIG["table_name"]
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET merged_into = %s, merged_at = now() WHERE profile_id = %s",
            (primary_profile_id, duplicate_profile_id),
        )


def delete_duplicate(conn, duplicate_profile_id: str) -> None:
    table = DEDUP_CONFIG["table_name"]
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE profile_id = %s", (duplicate_profile_id,))


def insert_audit_record(conn, run_id: str, record: Dict[str, Any]) -> None:
    table = DEDUP_CONFIG["audit_table_name"]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {table} (
                run_id, primary_profile_id, duplicate_profile_id, match_confidence,
                match_signal, original_primary_data, original_duplicate_data,
                merged_field_values, field_sources, conflicts, action
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
            """,
            (
                run_id,
                record["primary_profile_id"],
                record["duplicate_profile_id"],
                record["match_confidence"],
                record["match_signal"],
                json.dumps(record["original_primary_data"], default=_json_default),
                json.dumps(record["original_duplicate_data"], default=_json_default),
                json.dumps(record["merged_field_values"], default=_json_default),
                json.dumps(record["field_sources"], default=_json_default),
                json.dumps(record["conflicts"], default=_json_default),
                record["action"],
            ),
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
