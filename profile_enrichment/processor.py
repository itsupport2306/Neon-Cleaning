from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from profile_enrichment.config import CONFIG
from profile_enrichment.downloaders import download_file
from profile_enrichment.extractors import (
    combine_extracted_values,
    extract_profile_fields_from_text,
    extract_text_layers_from_file,
    run_ai_extraction,
)
from profile_enrichment.logging_utils import setup_logger
from profile_enrichment.neon_db import (
    build_update_map,
    fetch_profile_by_id,
    fetch_profiles_for_reprocessing,
    update_profile_fields,
)
from profile_enrichment.state_manager import StateManager

logger = setup_logger()
TARGET_FIELDS = ("specialty", "state_code", "email", "phone")


def _normalize_candidate(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _is_reliable_value(value: Any) -> bool:
    text = _normalize_candidate(value)
    if not text:
        return False
    normalized = text.lower()
    invalid = {
        "unknown",
        "candidate",
        "n/a",
        "na",
        "none",
        "null",
        "not provided",
        "unknown candidate",
        "no information",
        "-",
    }
    if normalized in invalid:
        return False
    if normalized.startswith("unknown") or normalized.startswith("candidate"):
        return False
    if len(text) < 2:
        return False
    return True


def sanitize_extracted(extracted: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in extracted.items():
        normalized = _normalize_candidate(value)
        if not normalized:
            continue

        if key in {"phone"} and len("".join(ch for ch in normalized if ch.isdigit())) < 7:
            continue
        if key == "email" and "@" not in normalized:
            continue
        if key in {"state", "state_code"}:
            if len(normalized) != 2 or not normalized.isalpha():
                continue
            normalized = normalized.upper()

        if not _is_reliable_value(normalized):
            continue

        cleaned[key] = normalized
    return cleaned


def _log_no_useful_information(profile_id: str, reason: str) -> None:
    logger.info("NO_USEFUL_INFORMATION | profile_id=%s | reason=%s", profile_id, reason)


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _missing_target_fields(extracted: Dict[str, Any]) -> Tuple[str, ...]:
    missing = []
    for field in TARGET_FIELDS:
        candidate = extracted.get(field)
        if field == "state_code" and not candidate:
            candidate = extracted.get("state")
        if not _normalize_candidate(candidate):
            missing.append(field)
    return tuple(missing)


def process_profile_record(row: Dict[str, Any], state: StateManager, existing: Dict[str, Any] | None = None) -> None:
    profile_id = _normalize_candidate(row.get("profile_id"))
    if not profile_id:
        logger.warning("MISSING_PROFILE_ID | row=%s", row)
        return

    if CONFIG["skip_already_processed"] and state.is_processed(profile_id):
        state.mark_skipped(profile_id)
        logger.info("SKIPPED_ALREADY_PROCESSED | profile_id=%s", profile_id)
        return

    file_ref = (
        row.get("resume_url")
        or row.get("resume_path")
        or row.get("cloudflare_file")
        or row.get("file_path")
        or row.get("document_path")
    )
    if not file_ref:
        _log_no_useful_information(profile_id, "missing file reference on source record")
        state.mark_failed(profile_id)
        return

    logger.info("START_PROCESSING | profile_id=%s | file_ref=%s", profile_id, file_ref)

    try:
        downloaded_path = download_file(file_ref)
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("CLOUDFLARE_DOWNLOAD_FAILURE | profile_id=%s | file_ref=%s | error=%s", profile_id, file_ref, exc)
        state.mark_failed(profile_id)
        return

    if not downloaded_path:
        logger.error("CLOUDFLARE_DOWNLOAD_FAILURE | profile_id=%s | file_ref=%s | reason=no file downloaded", profile_id, file_ref)
        state.mark_failed(profile_id)
        return

    try:
        text_layers = extract_text_layers_from_file(downloaded_path)
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("OCR_EXTRACTION_FAILURE | profile_id=%s | path=%s | error=%s", profile_id, downloaded_path, exc)
        state.mark_failed(profile_id)
        return

    combined_text = text_layers["combined_text"]
    if not combined_text.strip():
        _log_no_useful_information(profile_id, "document was downloaded but no text was extracted")
        state.mark_failed(profile_id)
        return

    extracted = sanitize_extracted(extract_profile_fields_from_text(combined_text))
    missing_before_ai = _missing_target_fields(extracted)

    try:
        ai_extracted = run_ai_extraction(downloaded_path, combined_text) if missing_before_ai else {}
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("AI_EXTRACTION_FAILURE | profile_id=%s | path=%s | error=%s", profile_id, downloaded_path, exc)
        ai_extracted = {}

    final_extracted = sanitize_extracted(combine_extracted_values(extracted, ai_extracted))
    if not final_extracted:
        _log_no_useful_information(profile_id, "no reliable personal/profile data extracted from document")
        state.mark_skipped(profile_id)
        return

    try:
        existing_row = existing or fetch_profile_by_id(profile_id)
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("NEONDB_QUERY_FAILURE | profile_id=%s | error=%s", profile_id, exc)
        state.mark_failed(profile_id)
        return

    if not existing_row:
        logger.error("PROFILE_NOT_FOUND | profile_id=%s", profile_id)
        state.mark_failed(profile_id)
        return

    update_map = build_update_map(existing_row, final_extracted)
    if not update_map:
        logger.info(
            "NO_UPDATES_REQUIRED | profile_id=%s | missing_before_ai=%s | native_chars=%s | ocr_chars=%s",
            profile_id,
            list(missing_before_ai),
            len(text_layers["native_text"]),
            len(text_layers["ocr_text"]),
        )
        state.mark_processed(profile_id)
        return

    try:
        update_profile_fields(profile_id, update_map)
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("NEONDB_UPDATE_FAILURE | profile_id=%s | updates=%s | error=%s", profile_id, update_map, exc)
        state.mark_failed(profile_id)
        return

    logger.info(
        "SUCCESSFUL_UPDATE | profile_id=%s | updated_fields=%s | native_chars=%s | ocr_chars=%s",
        profile_id,
        sorted(update_map.keys()),
        len(text_layers["native_text"]),
        len(text_layers["ocr_text"]),
    )
    state.mark_processed(profile_id)


def run_enrichment(
    csv_path: str | None = None,
    max_rows: int | None = None,
    source: str | None = None,
) -> Dict[str, Any]:
    source_name = (source or CONFIG["profile_source"]).strip().lower()
    state = StateManager(CONFIG["state_file"])
    state.set_last_run(datetime.now(timezone.utc).isoformat())

    if source_name == "csv":
        rows: List[Dict[str, Any]] = _read_csv_rows(csv_path or CONFIG["csv_path"])
    elif source_name == "neon":
        rows = fetch_profiles_for_reprocessing(limit=max_rows)
    else:
        raise ValueError(f"Unsupported source: {source_name}")

    total_rows = 0
    processed = 0
    skipped = 0

    for row in rows:
        if source_name == "csv" and max_rows is not None and total_rows >= max_rows:
            break
        total_rows += 1

        profile_id = _normalize_candidate(row.get("profile_id"))
        if not profile_id:
            continue

        if CONFIG["skip_already_processed"] and state.is_processed(profile_id):
            skipped += 1
            continue

        process_profile_record(row, state, existing=(row if source_name == "neon" else None))
        if state.is_processed(profile_id):
            processed += 1
        elif profile_id in state.failed:
            continue
        elif profile_id in state.skipped:
            skipped += 1

    return {
        "source": source_name,
        "csv_path": (csv_path or CONFIG["csv_path"]) if source_name == "csv" else None,
        "total_rows": total_rows,
        "processed": processed,
        "skipped": skipped,
        "state_file": str(state.state_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess Cloudflare resumes and update missing NeonDB profile fields.")
    parser.add_argument("--source", default=CONFIG["profile_source"], choices=["neon", "csv"], help="Where input rows come from.")
    parser.add_argument("--csv-path", default=CONFIG["csv_path"], help="Path to the exported NeonDB CSV to process when --source csv.")
    parser.add_argument("--limit", type=int, default=CONFIG["process_limit"], help="Maximum number of rows to process.")
    parser.add_argument("--no-skip-processed", action="store_true", help="Process rows even if they were already marked as processed.")
    args = parser.parse_args()

    if args.no_skip_processed:
        CONFIG["skip_already_processed"] = False

    result = run_enrichment(
        csv_path=args.csv_path,
        max_rows=(None if args.limit in (0, None) else args.limit),
        source=args.source,
    )
    print(result)
