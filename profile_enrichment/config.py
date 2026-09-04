import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)


def get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


neon_table_name = os.getenv("NEON_TABLE_NAME", "profiles")

CONFIG = {
    "profile_source": os.getenv("PROFILE_SOURCE", "neon").strip().lower(),
    "csv_path": os.getenv("CSV_PATH", str(ROOT_DIR / "flat-glade-56456784_production_neondb_2026-08-19_20-25-26.csv")),
    "neon_db_url": os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL"),
    "neon_table_name": neon_table_name,
    "neon_reprocess_query": os.getenv(
        "NEON_REPROCESS_QUERY",
        f"SELECT * FROM {neon_table_name} WHERE specialty IS NULL",
    ),
    "cloudflare_public_base_url": os.getenv("CLOUDFLARE_PUBLIC_BASE_URL"),
    "cloudflare_access_key": os.getenv("CLOUDFLARE_ACCESS_KEY"),
    "cloudflare_secret_key": os.getenv("CLOUDFLARE_SECRET_KEY"),
    "cloudflare_bucket": os.getenv("CLOUDFLARE_BUCKET"),
    "ai_provider": os.getenv("AI_PROVIDER", "local_qwen"),
    "ai_model": os.getenv("AI_MODEL", "gpt-4o-mini"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "ai_extractor_enabled": get_bool("AI_EXTRACTOR_ENABLED", True),
    "process_limit": int(os.getenv("PROCESS_LIMIT", "0") or 0),
    "worker_count": max(1, int(os.getenv("WORKER_COUNT", "3") or 3)),
    "created_at_start": os.getenv("CREATED_AT_START", "").strip(),
    "created_at_end": os.getenv("CREATED_AT_END", "").strip(),
    "created_at_partition_days": max(1, int(os.getenv("CREATED_AT_PARTITION_DAYS", "1") or 1)),
    "db_fetch_batch_size": max(1, int(os.getenv("DB_FETCH_BATCH_SIZE", "250") or 250)),
    "state_save_every": max(1, int(os.getenv("STATE_SAVE_EVERY", "50") or 50)),
    "state_file": os.getenv("STATE_FILE", str(ROOT_DIR / "state" / "enrichment_state.json")),
    "log_dir": os.getenv("LOG_DIR", str(ROOT_DIR / "logs")),
    "download_dir": os.getenv("DOWNLOAD_DIR", str(ROOT_DIR / "downloads")),
    "skip_already_processed": get_bool("SKIP_ALREADY_PROCESSED", True),
    # When false, a populated NeonDB column is never overwritten, only blanks are filled.
    "allow_overwrite_existing": get_bool("ALLOW_OVERWRITE_EXISTING", True),
    "tesseract_cmd": os.getenv("TESSERACT_CMD"),
    "ocr_enabled": get_bool("OCR_ENABLED", True),
    "ocr_min_text_chars": int(os.getenv("OCR_MIN_TEXT_CHARS", "120") or 120),
    # OCR runs Qwen vision first and falls back to Tesseract.
    "vision_ocr_enabled": get_bool("VISION_OCR_ENABLED", True),
    "vision_ocr_base_url": os.getenv("VISION_OCR_BASE_URL"),
    "vision_ocr_api_key": os.getenv("VISION_OCR_API_KEY"),
    "vision_ocr_model": os.getenv("VISION_OCR_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "vision_ocr_max_tokens": int(os.getenv("VISION_OCR_MAX_TOKENS", "2048") or 2048),
    "vision_ocr_max_image_side": int(os.getenv("VISION_OCR_MAX_IMAGE_SIDE", "1600") or 1600),
    "vision_ocr_timeout": float(os.getenv("VISION_OCR_TIMEOUT", "120") or 120),
    "ocr_skip_when_native_text": get_bool("OCR_SKIP_WHEN_NATIVE_TEXT", True),
    "local_model_enabled": get_bool("LOCAL_MODEL_ENABLED", True),
    "local_model_base_url": os.getenv("LOCAL_MODEL_BASE_URL"),
    "local_model_api_key": os.getenv("LOCAL_MODEL_API_KEY", "local-qwen"),
    "local_model_name": os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
    "local_model_repo_id": os.getenv("LOCAL_MODEL_REPO_ID", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
    "local_model_filename": os.getenv("LOCAL_MODEL_FILENAME", "*Q4_K_M.gguf"),
    "local_model_path": os.getenv("LOCAL_MODEL_PATH"),
    "local_model_dir": os.getenv("LOCAL_MODEL_DIR", str(ROOT_DIR / "models")),
    "local_model_n_ctx": int(os.getenv("LOCAL_MODEL_N_CTX", "8192") or 8192),
    "local_model_max_tokens": int(os.getenv("LOCAL_MODEL_MAX_TOKENS", "512") or 512),
    "ai_prompt_max_chars": max(1000, int(os.getenv("AI_PROMPT_MAX_CHARS", "12000") or 12000)),
}
