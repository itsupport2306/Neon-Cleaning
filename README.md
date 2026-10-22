# Phase 2 Profile Enrichment and Correction

This project reprocesses live NeonDB profile rows, downloads the matching Cloudflare resume for each profile, extracts the most reliable personal/profile data from that source, and updates the existing NeonDB row using `profile_id` as the unique key.

The goal is not to create a new extraction pipeline from scratch. It is a safe Phase 2 enrichment workflow that checks the document evidence against the current NeonDB record and only writes trustworthy updates.

## Overview

The workflow is:

1. Query NeonDB for candidate rows starting at `CREATED_AT_START`, partitioned by `created_at` time windows.
2. For each row, read the Cloudflare reference such as `/files/resumes/import/...` or a direct public URL.
3. Download the file from Cloudflare.
4. Extract native text from the file where possible.
5. Reprocess the document with Tesseract OCR.
6. Use a lightweight extraction pass and a local Qwen GGUF fallback when the target fields are still missing.
7. Validate candidates and discard unreliable placeholders like `Unknown`, `Candidate`, or empty strings.
8. Match by `profile_id` to the existing NeonDB record.
9. Update only missing fields with strong evidence.
10. Log every outcome and keep resumable progress in a JSON state file.
11. Keep exactly three concurrent workers by default, with bounded in-flight tasks and batched state flushes for higher throughput.

## Project structure

```text
.
├── .env.example
├── README.md
├── requirements.txt
├── flat-glade-56456784_production_neondb_2026-08-19_20-25-26.csv
├── logs/
├── downloads/
├── state/
├── profile_enrichment/
│   ├── __init__.py
│   ├── config.py
│   ├── downloaders.py
│   ├── extractors.py
│   ├── logging_utils.py
│   ├── neon_db.py
│   ├── processor.py
│   └── state_manager.py
└── tests/
    └── test_processor.py
```

## Installation

1. Open a terminal in the project root.
2. Create a virtual environment if desired.
3. Install the project dependencies:

```bash
pip install -r requirements.txt
```

OCR support (scanned/image-only resumes):

OCR runs the **Qwen vision model first** and falls back to Tesseract.

- Qwen: serve a Qwen *vision* model (e.g. `Qwen/Qwen2.5-VL-7B-Instruct` via vLLM/llama.cpp/Ollama)
  behind an OpenAI-compatible endpoint and set `VISION_OCR_BASE_URL` (and `VISION_OCR_MODEL`).
  If unset, it reuses `LOCAL_MODEL_BASE_URL` — note the default `Qwen2.5-1.5B-Instruct` there is
  text-only and cannot do OCR, so point `VISION_OCR_BASE_URL` at a VL server.
- Tesseract: install it and set `TESSERACT_CMD` if it is not on `PATH`.

If an engine is unusable, it is disabled once with a single warning and the run continues.
OCR is skipped entirely when a document already yields enough native text
(`OCR_SKIP_WHEN_NATIVE_TEXT=false` to always OCR).

## Environment variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Key values:

- `PROFILE_SOURCE`: `neon` by default. Set `csv` only if you want to process the old export file path.
- `CSV_PATH`: optional path to the NeonDB-exported CSV when `PROFILE_SOURCE=csv`.
- `CLOUDFLARE_PUBLIC_BASE_URL`: public base URL for the Cloudflare bucket.
- `CLOUDFLARE_ACCESS_KEY`: optional storage access key.
- `CLOUDFLARE_SECRET_KEY`: optional storage secret key.
- `CLOUDFLARE_BUCKET`: Cloudflare bucket name.
- `NEON_DB_URL`: direct Postgres connection string for NeonDB.
- `NEON_TABLE_NAME`: table holding the profile rows.
- `NEON_REPROCESS_QUERY`: query used to fetch candidate rows for reprocessing.
- `CREATED_AT_START`: first `created_at` timestamp to process, for example `2026-08-09 19:24:59.563398`.
- `CREATED_AT_END`: last `created_at` boundary for the run, for example `2026-08-10 00:00:00`. If omitted, the run continues to the newest matching row.
- `CREATED_AT_PARTITION_DAYS`: size of each internal `created_at` partition window in days. Default `1`.
- `DB_FETCH_BATCH_SIZE`: number of rows fetched per Neon round-trip inside each time window. Default `250`.
- `AI_EXTRACTOR_ENABLED`: set to `true` to enable the Qwen model fallback.
- `WORKER_COUNT`: number of concurrent resume-processing workers. Default `3`.
- `STATE_FILE`: JSON file used for resumability.
- `LOG_DIR`: directory for processing logs.
- `DOWNLOAD_DIR`: local directory for downloaded files.
- `SKIP_ALREADY_PROCESSED`: set to `true` to avoid reprocessing already completed rows.
- `WORKER_COUNT`: concurrent workers. Default `3`.
- `STATE_SAVE_EVERY`: save the resumable state file every N changes instead of every profile. Default `50`.
- `OCR_ENABLED`: enables Tesseract OCR for PDFs, images, and DOCX embedded images.
- `TESSERACT_CMD`: path to the Tesseract executable if it is not on `PATH`.
- `LOCAL_MODEL_REPO_ID`: default Hugging Face repo for the GGUF model.
- `LOCAL_MODEL_FILENAME`: GGUF filename or glob pattern, such as `*Q4_K_M.gguf`.
- `LOCAL_MODEL_PATH`: local path to a pre-downloaded GGUF file. If omitted, the app can download from Hugging Face through `llama-cpp-python`.
- `AI_PROMPT_MAX_CHARS`: max resume text sent to the extraction model per profile. Default `12000`.

## Cloudflare configuration

The CSV contains references like:

```text
/files/resumes/import/51/51cb8d5d47a9e35c5d2aeb5262a3dbae7239706b5f6f1e65fbed21cd2f0ae39c.pdf
```

The downloader resolves these in two ways:

- If the value is already a full HTTP URL, it uses it directly.
- Otherwise, it joins the value to `CLOUDFLARE_PUBLIC_BASE_URL` and downloads the public object.

This works well when your Cloudflare bucket or R2 endpoint is serving public files.

## NeonDB configuration

The application expects the row key to be `profile_id` and updates the table directly with SQL.

Required settings:

```env
NEON_DB_URL=postgresql://user:password@host:5432/dbname?sslmode=require
NEON_TABLE_NAME=profiles
```

The default selection query is:

```sql
SELECT *
FROM profiles
WHERE specialty IS NULL
  AND state_code IS NULL
  AND email IS NULL
  AND phone IS NULL;
```

Then it updates only missing fields with useful extracted data.

## AI model configuration

By default, Qwen only runs when OCR plus regex extraction still misses one or more of the target fields: `specialty`, `state_code`, `email`, or `phone`. With `--reprocess-all` / `--comprehensive`, Qwen runs on every row's full document text regardless of what is already stored - including rows already marked `screen_reason = "AI"` - because a previous AI pass is not treated as proof the row is clean.

The default local fallback is `Qwen/Qwen2.5-1.5B-Instruct-GGUF`. The most reliable Windows setup is to run a local `llama.cpp` server and point the app at it with `LOCAL_MODEL_BASE_URL` and `LOCAL_MODEL_NAME`.

Example local server command:

```bash
llama serve -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
```

If you already have a local GGUF runtime embedded in Python, you can still use `LOCAL_MODEL_PATH` or the Hugging Face repo settings.

Qwen is asked for a small JSON payload containing:

- `first_name`
- `last_name`
- `city`
- `state_code`
- `specialty`
- `phone`
- `email`

If the model call fails, the process keeps the OCR/text result and continues instead of stopping the full batch.

There is no OpenAI (or any other cloud LLM) provider in this pipeline - Qwen, served locally or via `LOCAL_MODEL_BASE_URL`, is the only extraction model.

## Source format

The primary input source is the live NeonDB query. If you use `PROFILE_SOURCE=csv`, the keys used by the code include:

- `profile_id`
- `resume_url`
- `first_name`
- `last_name`
- `city`
- `state_code`
- `specialty`
- `phone`
- `email`

The most important field is `profile_id`, because it determines the exact NeonDB row to update.

## Extraction process

The extraction pipeline is intentionally conservative:

1. Download the file from Cloudflare.
2. Parse native PDF or DOCX text where available.
3. Run OCR (Qwen vision first, Tesseract as fallback) across PDF page renders, image files, and DOCX embedded media.
4. Combine native text and OCR text - this combined text is the only source of truth for the rest of the run.
5. Extract candidate fields using regex patterns.
6. Ask the configured model for strict JSON extraction.
7. **Validate** every candidate value independently (see below).
8. **Compare** the surviving values with the stored NeonDB row.
9. Update only the fields that passed every gate.

Extraction, validation, comparison and updating are separate stages. An
extraction result is never treated as an update on its own: **no evidence, no
update.**

### Validation gates

`profile_enrichment/validation.py` puts every candidate value through three gates:

1. **Shape** - does it look like an email / phone / name / 2-letter state at all?
2. **Placeholder rejection** - values such as `johndoe@example.com`, `123-456-7890`,
   `John Doe`, `Anytown`, `N/A` and `Unknown` are refused outright. A name that is
   really a sentence, heading or job title (`I'm Patrick`, `Certified Assistant`) is
   refused too, as is a specialty that is really a slice of prose.
3. **Source grounding** - the value must actually occur in the document text. A city,
   specialty, email or phone number that is nowhere in the resume is never written,
   no matter how confidently the model returned it.

Every rejected value is logged so a bad model, a bad prompt and a bad rule can be
told apart:

```
REJECTED_AI_VALUE | profile_id=... | field=email | value=johndoe@example.com | reason=placeholder_email
```

The extraction prompt itself tells the model to return `null` rather than guess,
infer, complete partial data or emit example values.

### Existing values

A column that already holds a value is only changed when the document does **not**
confirm what is stored and the new value is fully verified. If the resume shows
`Patrick Vrtis`, that name stays - a competing extraction cannot displace it. Pass
`--no-overwrite` (or set `ALLOW_OVERWRITE_EXISTING=false`) to fill blanks only and
never touch a populated column.

`phone` is stricter: it is only written when the existing database value is `NULL`
and the resume contains a verified phone number. Existing non-`NULL` phone values
are never replaced.

An empty field is an acceptable outcome. A fabricated field is not.

## `profile_id` matching and update logic

The update logic uses the live NeonDB row as the input reference, then matches it by `profile_id` against the same table for the update statement.

The update rules are:

- Do not update if there is no `profile_id`.
- Do not overwrite a non-empty existing field with empty or placeholder data.
- Update only the fields that are currently blank and have valid extracted values.

The actual update function is intentionally narrow:

```python
build_update_map(existing, extracted)
```

This checks each field and only includes it if it passes the validation rules and is not clearly worse than the existing value.

## Logging and error handling

The app writes logs to the console and to `logs/enrichment.log`. Every row that is
skipped or fails prints its reason to the terminal, not only to the log file - no
row is ever silently dropped.

It tracks:

- Successfully processed records
- Successfully updated records
- Records with no useful information
- Cloudflare download failures
- AI/OCR extraction failures
- NeonDB update failures

Examples:

- `START_PROCESSING | profile_id=...`
- `SUCCESSFUL_UPDATE | profile_id=... | updated_fields=[...]`
- `SKIPPED_ROW | profile_id=... | reason=...`
- `CLOUDFLARE_DOWNLOAD_FAILURE | profile_id=...`
- `NEONDB_UPDATE_FAILURE | profile_id=...`

## Site-safe resumable processing

A JSON state file stores processed and failed IDs so the job can resume safely.

State file path defaults to:

```text
./state/enrichment_state.json
```

The state tracks:

- `processed`
- `skipped`
- `failed`
- `last_run`

This means that if the script is interrupted, you can rerun it and it will skip previously completed rows.

To rerun from the latest point:

```bash
python -m profile_enrichment.processor --source neon
```

To run with 3 workers explicitly:

```bash
python -m profile_enrichment.processor --source neon --workers 3
```

To force reprocessing all rows:

```bash
python -m profile_enrichment.processor --source neon --no-skip-processed
```

## Running the enrichment job

For Neon processing, the runtime now reads `CREATED_AT_START` and optional `CREATED_AT_END` from `.env` and walks the table in ordered `created_at` windows. Within each window it uses batched fetches and keeps only `WORKER_COUNT` active tasks in flight, so Qwen stays busy without loading the entire candidate set into memory.

```bash
python -m profile_enrichment.processor --source neon --limit 100
```

To reprocess the whole dataset from the original documents, ignoring the
processed-state file and every value a previous run produced:

```bash
python -m profile_enrichment.processor --source neon --reprocess-all
```

`--reprocess-all` implies `--comprehensive` and `--no-skip-processed`, and reads
every row in the table rather than only the rows still missing a specialty. Each
row is rebuilt from its source document: original file -> native text/OCR -> AI
extraction -> validation -> comparison with the stored row -> update of the
verified fields only.

The script prints a summary result like:

```python
{
  'source': 'neon',
  'csv_path': None,
  'total_rows': 100,
  'processed': 42,
  'skipped': 12,
  'state_file': './state/enrichment_state.json'
}
```

## Fixing invalid `specialty` values

Some rows already have a non-empty `specialty` that is actually a name prefix
(`John, Cardiology`), a phone number, an email/domain (`yahoo.com`), a URL, or
junk text like `web`. `--fix-specialty` finds only those rows, reprocesses
their resume through the same extraction/validation pipeline, and writes a
corrected value to `specialty` alone - every other column is left untouched.

```bash
python -m profile_enrichment.processor --fix-specialty
```

It uses the same `--workers`, `--no-ocr`, `--limit`, and `--verbose` flags as
the normal run. Detection logic lives in
`profile_enrichment.processor.classify_suspicious_specialty`; adjust
`NEON_SPECIALTY_FIX_QUERY` in `.env` to narrow the initial candidate fetch.
Rows it fixes are logged as `SPECIALTY_FIX_SUCCESS`; rows it could not resolve
are logged as `SKIPPED_ROW | reason=could not resolve a verified specialty from the resume`.

## Testing

The project includes a small pytest file to validate the sanitization logic and update safety rules.

```bash
pytest -q
```

## Safety rules

This process is intentionally conservative:

- It does not overwrite with blank values.
- It does not overwrite with placeholder strings.
- It never invents a value to fill an empty column; every value must be present in the source document.
- It does not replace a stored value that the source document confirms.
- It does not process rows without a valid `profile_id`.
- It will keep the original record if the source document is missing or unusable.
- It logs the reason when there is no reliable information to use.

## Notes

This is not a full replacement for the original extraction pipeline. It is a targeted Phase 2 update process designed to verify and improve data already exported from NeonDB using the source documents stored in Cloudflare.

The corrected data is written directly to NeonDB, not back to the CSV.
