import os
from pathlib import Path

from dotenv import load_dotenv

from profile_enrichment.specialty_rules import build_specialty_fix_sql

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
    # Filters at the database level for specialty values that look like a name
    # prefix, phone number, email/domain, URL, or junk text - see
    # profile_enrichment.specialty_rules for the exact patterns. This keeps the
    # fix-specialty run from pulling every row with a non-empty specialty just
    # to discard most of them in Python.
    "neon_specialty_fix_query": os.getenv(
        "NEON_SPECIALTY_FIX_QUERY",
        build_specialty_fix_sql(neon_table_name),
    ),
    # fill_null_specialty only selects rows whose screen_reason equals this (empty = no screen_reason filter).
    "null_specialty_screen_reason": os.getenv("NULL_SPECIALTY_SCREEN_REASON", "").strip(),
    "cloudflare_public_base_url": os.getenv("CLOUDFLARE_PUBLIC_BASE_URL"),
    "cloudflare_access_key": os.getenv("CLOUDFLARE_ACCESS_KEY"),
    "cloudflare_secret_key": os.getenv("CLOUDFLARE_SECRET_KEY"),
    "cloudflare_bucket": os.getenv("CLOUDFLARE_BUCKET"),
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
    # A 4B model on CPU can need well over a minute per resume (and longer when workers queue up).
    "ai_request_timeout": float(os.getenv("AI_REQUEST_TIMEOUT", "300") or 300),
    # Deprecated: the resume is no longer truncated to a character budget. Kept so old .env files load.
    "ai_prompt_max_chars": max(1000, int(os.getenv("AI_PROMPT_MAX_CHARS", "12000") or 12000)),
    # --- inference (text model). The vision OCR model above is configured and checked separately. ---
    # openai = any OpenAI-compatible server using response_format json_schema (vLLM, llama-server, LM Studio,
    # Ollama's /v1); ollama = native /api/chat (lets us set num_ctx and read done_reason); llama_cpp = embedded.
    "local_model_backend": os.getenv("LOCAL_MODEL_BACKEND", "ollama" if "11434" in (os.getenv("LOCAL_MODEL_BASE_URL") or "") else "openai").strip().lower(),
    # Must equal the context the *server* actually loaded (Ollama defaults to 4096 whatever the model supports).
    "ai_context_tokens": int(os.getenv("AI_CONTEXT_TOKENS") or os.getenv("LOCAL_MODEL_N_CTX") or 4096),
    "ai_chunk_tokens": int(os.getenv("AI_CHUNK_TOKENS", "1500") or 1500),
    "ai_chunk_overlap_tokens": int(os.getenv("AI_CHUNK_OVERLAP_TOKENS", "100") or 100),
    # Deliberately pessimistic (English text is ~4); a wrong guess is caught by the server-reported prompt_tokens.
    "ai_chars_per_token": float(os.getenv("AI_CHARS_PER_TOKEN", "3.0") or 3.0),
    "ai_max_chunks": int(os.getenv("AI_MAX_CHUNKS", "8") or 8),
    "ai_max_attempts": max(1, int(os.getenv("AI_MAX_ATTEMPTS", "3") or 3)),
    "ai_retry_backoff": float(os.getenv("AI_RETRY_BACKOFF", "2.0") or 2.0),
    "local_model_n_gpu_layers": int(os.getenv("LOCAL_MODEL_N_GPU_LAYERS", "0") or 0),
    "local_model_n_threads": int(os.getenv("LOCAL_MODEL_N_THREADS", "0") or 0),
    # --- vision OCR circuit breaker: open after N consecutive failures, retry after the cooldown ---
    "vision_ocr_failure_threshold": max(1, int(os.getenv("VISION_OCR_FAILURE_THRESHOLD", "3") or 3)),
    "vision_ocr_cooldown": float(os.getenv("VISION_OCR_COOLDOWN", "60") or 60),
    "ocr_min_page_chars": int(os.getenv("OCR_MIN_PAGE_CHARS", "40") or 40),
    # --- database ---
    "db_pool_max": max(1, int(os.getenv("DB_POOL_MAX", "4") or 4)),
    "db_connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "20") or 20),
    "db_statement_timeout_ms": int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "30000") or 30000),
    # --- durable job store, caches, download ---
    "job_store_path": os.getenv("JOB_STORE_PATH", str(ROOT_DIR / "state" / "null_specialty_jobs.sqlite3")),
    "cache_dir": os.getenv("CACHE_DIR", str(ROOT_DIR / "cache")),
    "doc_cache_max_bytes": int(os.getenv("DOC_CACHE_MAX_BYTES", str(2 * 1024**3)) or 0),
    "download_timeout": float(os.getenv("DOWNLOAD_TIMEOUT", "30") or 30),
    "download_attempts": max(1, int(os.getenv("DOWNLOAD_ATTEMPTS", "3") or 3)),
    "lease_seconds": int(os.getenv("LEASE_SECONDS", "900") or 900),
}

# Bump when the prompt, schema or the evidence rules change: cached results and "unresolved" verdicts
# recorded under an older version become eligible for retry.
PROMPT_VERSION = "2026-10-26.v2"
