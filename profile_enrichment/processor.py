from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from profile_enrichment import validation
from profile_enrichment.config import CONFIG
from profile_enrichment.downloaders import download_file_bytes
from profile_enrichment.extractors import (
    combine_extracted_values,
    extract_profile_fields_from_text,
    extract_text_layers_from_bytes,
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


def _extract_file_extension(file_ref: str) -> str:
    """Extract file extension from file reference (URL or path)."""
    url_path = file_ref.split("?")[0] if "?" in file_ref else file_ref
    _, ext = os.path.splitext(url_path.lower())
    return ext or ".bin"


MERGE_FIELDS = (
    "first_name",
    "last_name",
    "specialty",
    "state_code",
    "email",
    "phone",
    "city",
    "state",
)


def _existing_value_is_in_source(value: str, source_text: str) -> bool:
    """True when the value already stored in NeonDB is visible in the document."""
    if not value or not source_text:
        return False
    lowered = source_text.lower()
    if any(ch.isdigit() for ch in value):
        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) >= 7:
            return digits[-10:] in "".join(ch for ch in source_text if ch.isdigit())
    return value.lower() in " ".join(lowered.split())


def _verify_and_merge_fields(
    profile_id: str,
    existing: Dict[str, Any],
    extracted: Dict[str, Any],
    source_text: str = "",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Compare *already validated* values with the stored row.

    Every value reaching here has passed :mod:`profile_enrichment.validation`,
    so it is well formed and present in the source document. This step decides
    only whether the database should change at all:

    * empty column + verified value -> fill it in
    * populated column that the document does NOT confirm -> overwrite
    * populated column that the document DOES confirm -> leave it alone

    That last rule is what keeps `Patrick Vrtis` from becoming `I'm Patrick`.
    """
    updates: Dict[str, Any] = {}
    changes: Dict[str, str] = {}

    for field in MERGE_FIELDS:
        existing_clean = _normalize_candidate(existing.get(field))
        extracted_clean = _normalize_candidate(extracted.get(field))

        if not extracted_clean:
            continue

        if not existing_clean:
            updates[field] = extracted_clean
            changes[field] = f"<empty> -> {extracted_clean}"
            logger.info(
                "FIELD_ADDED | profile_id=%s | field=%s | value=%s",
                profile_id,
                field,
                extracted_clean,
            )
            continue

        if existing_clean.lower() == extracted_clean.lower():
            logger.info(
                "FIELD_VERIFIED | profile_id=%s | field=%s | value=%s",
                profile_id,
                field,
                existing_clean,
            )
            continue

        if not CONFIG["allow_overwrite_existing"]:
            logger.info(
                "FIELD_KEPT | profile_id=%s | field=%s | value=%s | reason=overwrites_disabled",
                profile_id,
                field,
                existing_clean,
            )
            continue

        if _existing_value_is_in_source(existing_clean, source_text):
            logger.info(
                "FIELD_KEPT | profile_id=%s | field=%s | value=%s | candidate=%s | "
                "reason=existing_value_confirmed_by_source",
                profile_id,
                field,
                existing_clean,
                extracted_clean,
            )
            continue

        updates[field] = extracted_clean
        changes[field] = f"{existing_clean} -> {extracted_clean}"
        logger.info(
            "FIELD_UPDATED | profile_id=%s | field=%s | old=%s | new=%s",
            profile_id,
            field,
            existing_clean,
            extracted_clean,
        )

    return updates, changes


def _is_reliable_value(value: Any) -> bool:
    """Back-compat helper: a value is reliable when it is not a null/placeholder token."""
    text = _normalize_candidate(value)
    if len(text) < 2:
        return False
    return not validation._is_null_token(text)


def sanitize_extracted(
    extracted: Dict[str, Any],
    source_text: str = "",
    profile_id: str = "",
    require_grounding: bool = False,
) -> Dict[str, Any]:
    """Validate an extraction result, dropping (and logging) everything unsafe."""
    return validation.validate_extracted(
        extracted,
        source_text=source_text,
        profile_id=profile_id,
        require_grounding=require_grounding,
    )


def _format_changes(existing: Dict[str, Any], update_map: Dict[str, Any]) -> str:
    """Render the applied updates as `field: old -> new` for the log line."""
    parts = []
    for field in sorted(update_map):
        old = _normalize_candidate(existing.get(field)) or "<empty>"
        parts.append(f"{field}: {old} -> {update_map[field]}")
    return "; ".join(parts)


_no_useful_information_count = 0


def _log_no_useful_information(profile_id: str, reason: str) -> None:
    """Logged to the file only; the terminal gets the run-end count instead."""
    global _no_useful_information_count
    _no_useful_information_count += 1
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


def process_profile_record(row: Dict[str, Any], state: StateManager, existing: Dict[str, Any] | None = None, comprehensive_reprocess: bool = False, disable_ocr: bool = False) -> None:
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
        file_data = download_file_bytes(file_ref)
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("CLOUDFLARE_DOWNLOAD_FAILURE | profile_id=%s | file_ref=%s | error=%s", profile_id, file_ref, exc)
        state.mark_failed(profile_id)
        return

    if not file_data:
        logger.error("CLOUDFLARE_DOWNLOAD_FAILURE | profile_id=%s | file_ref=%s | reason=no file downloaded", profile_id, file_ref)
        state.mark_failed(profile_id)
        return

    file_ext = _extract_file_extension(file_ref)
    try:
        text_layers = extract_text_layers_from_bytes(file_data, file_ext, ocr_enabled=not disable_ocr)
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("OCR_EXTRACTION_FAILURE | profile_id=%s | file_ref=%s | error=%s", profile_id, file_ref, exc)
        state.mark_failed(profile_id)
        return

    combined_text = text_layers["combined_text"]
    if not combined_text.strip():
        _log_no_useful_information(profile_id, "document was downloaded but no text was extracted")
        state.mark_failed(profile_id)
        return

    # Stage 1: extract from the document itself (regex over native text + OCR).
    extracted = sanitize_extracted(
        extract_profile_fields_from_text(combined_text),
        source_text=combined_text,
        profile_id=profile_id,
        require_grounding=True,
    )
    missing_before_ai = _missing_target_fields(extracted)

    try:
        if comprehensive_reprocess:
            ai_extracted = run_ai_extraction(None, combined_text) if combined_text.strip() else {}
        else:
            should_run_ai = missing_before_ai or not combined_text.strip()
            ai_extracted = run_ai_extraction(None, combined_text) if should_run_ai else {}
    except Exception as exc:  # pragma: no cover - exercised in runtime only
        logger.exception("AI_EXTRACTION_FAILURE | profile_id=%s | file_ref=%s | error=%s", profile_id, file_ref, exc)
        ai_extracted = {}

    # Stage 2: validate the AI output on its own, against the same source text, so a
    # hallucinated value is rejected and logged instead of merged in silently.
    ai_validated = sanitize_extracted(
        ai_extracted,
        source_text=combined_text,
        profile_id=profile_id,
        require_grounding=True,
    )

    final_extracted = combine_extracted_values(extracted, ai_validated)
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

    if comprehensive_reprocess:
        update_map, changes = _verify_and_merge_fields(
            profile_id, existing_row, final_extracted, source_text=combined_text
        )
        if update_map:
            # The applied result is logged as SUCCESSFUL_UPDATE once the write lands.
            logger.debug(
                "COMPREHENSIVE_REPROCESS_SUMMARY | profile_id=%s | changes=%s | native_chars=%s | ocr_chars=%s",
                profile_id,
                changes,
                len(text_layers["native_text"]),
                len(text_layers["ocr_text"]),
            )
    else:
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
        "SUCCESSFUL_UPDATE | profile_id=%s | changes=%s | native_chars=%s | ocr_chars=%s | source=%s",
        profile_id,
        _format_changes(existing_row, update_map),
        len(text_layers["native_text"]),
        len(text_layers["ocr_text"]),
        "ocr" if text_layers["ocr_text"].strip() else "native",
    )
    state.mark_processed(profile_id)


def run_enrichment(
    csv_path: str | None = None,
    max_rows: int | None = None,
    source: str | None = None,
    comprehensive_reprocess: bool = False,
    disable_ocr: bool = False,
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

    global _no_useful_information_count
    _no_useful_information_count = 0

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

        process_profile_record(
            row,
            state,
            existing=(row if source_name == "neon" else None),
            comprehensive_reprocess=comprehensive_reprocess,
            disable_ocr=disable_ocr,
        )
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
        "no_text_extracted": _no_useful_information_count,
        "state_file": str(state.state_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess Cloudflare resumes and update missing NeonDB profile fields.")
    parser.add_argument("--source", default=CONFIG["profile_source"], choices=["neon", "csv"], help="Where input rows come from.")
    parser.add_argument("--csv-path", default=CONFIG["csv_path"], help="Path to the exported NeonDB CSV to process when --source csv.")
    parser.add_argument("--limit", type=int, default=CONFIG["process_limit"], help="Maximum number of rows to process.")
    parser.add_argument("--no-skip-processed", action="store_true", help="Process rows even if they were already marked as processed.")
    parser.add_argument("--comprehensive", action="store_true", help="Reprocess entire row: extract all fields and cross-verify with existing data.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR (Tesseract) and use only native text and AI extraction.")
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help=(
            "Reprocess every row from its original document, ignoring the processed-state file "
            "and any values produced by a previous run. Implies --comprehensive and --no-skip-processed."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Only fill blank NeonDB columns; never change a column that already has a value.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-profile diagnostic lines that are normally file-only.")
    args = parser.parse_args()

    if args.verbose:
        from profile_enrichment.logging_utils import ConsoleNoiseFilter

        for handler in logger.handlers:
            for log_filter in list(handler.filters):
                if isinstance(log_filter, ConsoleNoiseFilter):
                    handler.removeFilter(log_filter)

    if args.no_skip_processed or args.reprocess_all:
        CONFIG["skip_already_processed"] = False

    if args.reprocess_all:
        # A full reprocess reads every row from its original document, so it must not
        # inherit the default "only rows still missing a specialty" filter.
        CONFIG["neon_reprocess_query"] = f"SELECT * FROM {CONFIG['neon_table_name']}"

    if args.no_overwrite:
        CONFIG["allow_overwrite_existing"] = False

    result = run_enrichment(
        csv_path=args.csv_path,
        max_rows=(None if args.limit in (0, None) else args.limit),
        source=args.source,
        comprehensive_reprocess=args.comprehensive or args.reprocess_all,
        disable_ocr=args.no_ocr,
    )
    print(result)
