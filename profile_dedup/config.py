import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DEDUP_CONFIG = {
    "neon_db_url": os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL"),
    "table_name": os.getenv("NEON_TABLE_NAME", "profiles"),
    "audit_table_name": os.getenv("PROFILE_DEDUP_AUDIT_TABLE", "profile_merge_audit"),
    # Phone numbers shared by more groups than this are treated as shared
    # lines (front desk / agency / call center) rather than a personal
    # number, and are excluded from matching entirely.
    "max_phone_group_size": int(os.getenv("PROFILE_DEDUP_MAX_PHONE_GROUP_SIZE", "5") or 5),
    "min_phone_digits": int(os.getenv("PROFILE_DEDUP_MIN_PHONE_DIGITS", "10") or 10),
    # Ratio threshold (difflib SequenceMatcher) above which two non-identical
    # names sharing a phone number are treated as a MEDIUM confidence match.
    "name_fuzzy_ratio": float(os.getenv("PROFILE_DEDUP_NAME_FUZZY_RATIO", "0.82") or 0.82),
    "groups_per_batch": int(os.getenv("PROFILE_DEDUP_GROUPS_PER_BATCH", "500") or 500),
    "report_dir": os.getenv("PROFILE_DEDUP_REPORT_DIR", str(Path(__file__).resolve().parent / "reports")),
    # Optional AI second opinion on MEDIUM confidence clusters only. Never
    # used to drive automatic merges/deletes - purely an annotation for the
    # human reviewing the dry-run report.
    "use_qwen_review": _get_bool("PROFILE_DEDUP_USE_QWEN", True),
    "qwen_timeout": float(os.getenv("PROFILE_DEDUP_QWEN_TIMEOUT", "30") or 30),
}
