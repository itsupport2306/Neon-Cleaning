# Resume validation on Google Colab (GPU) - read-only

Notebook: [resume_validation_colab.ipynb](resume_validation_colab.ipynb). Runtime helper: [colab_runtime.py](colab_runtime.py).
Code: [validate_resumes.py](validate_resumes.py) (CLI + model), [compare.py](compare.py) (comparison rules), [csv_store.py](csv_store.py) (CSV files).

For every profile with `specialty IS NOT NULL AND resume_url IS NOT NULL`, the validator downloads the resume,
extracts its text (OCR for scans), asks Qwen what the resume says, and compares that with the stored
`first_name`, `last_name`, `specialty` and address. **It never writes to the database.** It opens read-only
transactions and issues `SELECT` only (a unit test enforces this). Results go to CSV files for a developer to review.

It is built on the same pieces as [../colab/README.md](../colab/README.md) and changes nothing in `colab/` or `profile_enrichment/`:

| Piece | Setting (identical to `colab/`) |
|---|---|
| Text model | `qwen2.5:7b-instruct` on Ollama (native API, JSON-schema constrained), 16k context, temperature 0, seeded retries |
| Scanned resumes | `qwen2.5vl:7b` on the same Ollama server, Tesseract as fallback |
| Download / text / OCR | `profile_enrichment.downloaders`, `profile_enrichment.extractors` |
| Database | `profile_enrichment.neon_db` pool; `NEON_DB_URL`, table `profiles` |
| Specialty equivalents | `profile_enrichment/data/specialty_vocabulary.csv` + `specialty_aliases.csv` |
| Secrets | Colab Secrets: `NEON_DB_URL`, `CLOUDFLARE_PUBLIC_BASE_URL`, `GITHUB_TOKEN` |

## What is checked

| Field | Compared with | Result |
|---|---|---|
| `first_name`, `last_name` | the name line of the resume (credentials such as RN/MD and Jr/Sr ignored; case, accents, apostrophes ignored; "Mary" matches "Mary Ann") | `mismatch` for a different, placeholder (`Unknown`) or swapped name |
| `specialty` | the profession and specialties the resume shows. Aliases and abbreviations (ICU = Intensive Care) are resolved with the shared vocabulary first, then the model's verdict | `mismatch` only when the resume clearly shows something else |
| address | `city`, `state_code`, `zip_code` against the city/state/ZIP written in the resume (the `profiles` table has no street column, so a street is shown but not compared) | `mismatch` if a component present on both sides differs |

Each field ends as `match`, `mismatch` or `not_verifiable`. **Missing information is never a discrepancy:** a field the resume
does not state, a stored value that is empty, a low-confidence "not supported" verdict, or a resume that was only partly read all
give `not_verifiable`. Every value the model quotes from the resume must literally occur in the resume text, otherwise it is ignored.

A profile gets exactly one status and one row in exactly one file:

| `validation_status` | File | Meaning |
|---|---|---|
| `discrepancy_found` | `resume_validation_discrepancies.csv` | at least one field is `mismatch` |
| `no_discrepancy_found` | `resume_validation_no_discrepancies.csv` | resume readable, nothing contradicted, at least one field `match`. For manual review; the profile is unchanged |
| `unverifiable` | `resume_validation_errors.csv` | resume read, but no field could be verified |
| `failed` | `resume_validation_errors.csv` | no validation possible (see `error_category`) |

Fields that were `not_verifiable` inside a discrepancy or no-discrepancy row are listed in its `field_statuses` column.

## One-time setup

1. Push this repo, including `resume_collab/` and `tests/test_resume_collab.py`, to GitHub. The notebook clones `itsupport2306/Neon-Cleaning`
   (branch `main`; change `BRANCH` in step 2 if needed). Re-running step 2 later runs `git pull`.
2. Upload `resume_collab/resume_validation_colab.ipynb` to Colab (*File > Upload notebook*) or open it from GitHub.
3. *Runtime > Change runtime type*: **T4 GPU** (L4 / A100 on Colab Pro is faster).
4. *Secrets* (key icon): add `NEON_DB_URL` (`sslmode=require` is added if missing), `CLOUDFLARE_PUBLIC_BASE_URL` and `GITHUB_TOKEN`
   (a fine-grained token with read access to the repo). Turn on *Notebook access* for each.

## Notebook steps (run 1-4 on every new runtime)

| Step | Command | Does |
|---|---|---|
| 1 | `drive.mount(...)`, `DRIVE_DIR = '/content/drive/MyDrive/resume_validation'` | GPU check, Drive for the CSV files |
| 2 | clone / `git pull` | gets the latest code |
| 3 | `apt-get install tesseract-ocr zstd ...`, Ollama installer, `pip install -r requirements.txt` | dependencies (~3 min) |
| 4 | `rc.configure(DRIVE_DIR)`, `rc.start_ollama()`, `rc.pull_models()`, `rc.check_gpu()` | secrets, same Qwen 7B models, GPU check |
| 5 | `!python -m pytest -q tests/test_resume_collab.py` | offline tests |
| 6 | `rc.validate('preflight')`, `rc.validate('count')` | DB reachable, model canary, OCR, **eligible count** |
| 7 | `rc.validate('run', '--limit', '25')` | **test batch**, then inspect the CSV files |
| 8 | `rc.validate('run', '--workers', '4', '--model-concurrency', '2')` | **full run** (also the resume command) |
| 9 | `rc.validate('status')`, review the three CSV files | results |

## Running outside Colab

Needs a reachable Ollama with the two models, `NEON_DB_URL` (and `CLOUDFLARE_PUBLIC_BASE_URL` for relative `resume_url`s) in `.env` or the environment, and
`LOCAL_MODEL_BACKEND=ollama LOCAL_MODEL_BASE_URL=http://127.0.0.1:11434 LOCAL_MODEL_NAME=qwen2.5:7b-instruct AI_CONTEXT_TOKENS=16384`
(see `pipeline_env()` in [../colab/colab_runtime.py](../colab/colab_runtime.py) for the full list).

```bash
pip install -r requirements.txt
python -m resume_collab.validate_resumes count
python -m resume_collab.validate_resumes preflight
python -m resume_collab.validate_resumes run --limit 25
python -m resume_collab.validate_resumes run
python -m resume_collab.validate_resumes status
```

## Commands and options

`python -m resume_collab.validate_resumes <command>` (in a notebook: `rc.validate('<command>', ...)`):

| Command | Purpose |
|---|---|
| `count` | prints the `SELECT` used and the number of eligible profiles |
| `preflight [--skip-count] [--require-ocr]` | database, text-model canary extraction, vision/Tesseract OCR |
| `run [options]` (default) | validate and append to the CSV files |
| `status [--out-dir DIR]` | row count per CSV file and duplicate check |

`run` options: `--limit N` (stop after N *new* profiles), `--profile-ids a,b,c` / `--profile-ids-file FILE` (specific eligible profiles),
`--out-dir DIR`, `--workers N` (download/extract/OCR concurrency, default `WORKER_COUNT`), `--model-concurrency N` (default 2),
`--page-size N`, `--retry-errors`, `--no-ocr`, `--skip-count`, `--skip-preflight`.

Environment variables: the ones in [../colab/README.md](../colab/README.md) / `profile_enrichment/config.py`, plus `OUTPUT_DIR` (CSV folder;
default `<DRIVE_DIR>/reports` on Colab, `resume_collab/output/` elsewhere - git-ignored because it holds candidate data).

## Test batch, then the full run

1. Step 6: `preflight` must print `preflight: OK`; `count` prints the number of eligible profiles.
2. Step 7: `rc.validate('run', '--limit', '25')`. The notebook shows the head of each CSV. For every discrepancy compare `*_db` with `*_resume`
   and `discrepancy_reason`; open a few of the resumes. Look at `no_discrepancies` and `errors` too.
3. If the batch is wrong, delete the three CSV files (they are the only state) and re-run it. If it is right, keep them: step 8 continues after those 25.
4. Step 8 processes everything else. Progress lines show profiles per minute and an estimate of the time remaining.

## Resuming an interrupted run

The CSV files are the only state. Each row is written, flushed and fsynced as soon as its profile finishes, and on start the validator reads the
`profile_id`s in all three files and skips them, so **re-running the same command is the resume**, and no profile can appear twice.

- Stop button: in-flight profiles finish and are recorded, then the cell exits (exit code 2). Run step 8 again.
- Disconnect: run steps 1-4, then step 8. Only profiles that were in flight (a few) are validated again.
- Profiles that failed with a retryable problem stay in the errors file. `rc.validate('run', '--retry-errors')` moves that file to
  `resume_validation_errors.<timestamp>.bak.csv` and validates those profiles again. Do this after fixing the cause (model down, network).

## CSV files

All files are UTF-8 (with BOM, so Excel opens them correctly), one row per `profile_id`, in `OUTPUT_DIR`.

**`resume_validation_discrepancies.csv`**: `profile_id, resume_url, first_name_db, first_name_resume, last_name_db, last_name_resume, specialty_db,
specialty_resume, address_db, address_resume, mismatched_fields, discrepancy_reason, model_confidence, review_status, validation_status, field_statuses, validated_at`

```
profile_id,...,first_name_db,first_name_resume,...,specialty_db,specialty_resume,address_db,address_resume,mismatched_fields,discrepancy_reason,model_confidence,review_status,...
3f2a...,https://.../r.pdf,Maria,Priya,...,Cardiology,"profession: Registered Nurse; specialties: intensive care unit","Dayton, OH 45402","12 Oak Street, Columbus, OH 43004",first_name;specialty;address,"first_name: stored 'Maria' but resume says 'Priya' | specialty: stored 'Cardiology' is not supported by the resume. Resume shows ICU nursing only. | address city: stored 'Dayton' but resume says 'Columbus'",high,pending_review,...
```

`mismatched_fields` is a `;` list of `first_name`, `last_name`, `specialty`, `address`. `review_status` starts as `pending_review`; a developer can
edit it in the CSV. Stored values are never changed by this workflow.

**`resume_validation_no_discrepancies.csv`**: `profile_id, resume_url, validation_status, review_reason, field_statuses, model_confidence, validated_at`

```
7c19...,https://.../r.pdf,no_discrepancy_found,"Resume readable and no field contradicted; profile left unchanged for manual review. Not verifiable from the resume: address.",first_name=match; last_name=match; specialty=match; address=not_verifiable,high,2026-10-29 10:15:02
```

**`resume_validation_errors.csv`**: `profile_id, resume_url, validation_status, error_category, error_detail, retryable, validated_at`

| `error_category` | Cause | Retryable |
|---|---|---|
| `invalid_resume_url` | empty or not a downloadable URL / path | no |
| `resume_not_found` | HTTP 404 / 410 | no |
| `download_failed` | timeout, connection error, 5xx/429, other 4xx, file too large | transient only |
| `unsupported_document` | file type the extractor cannot read | no |
| `extraction_failed` | corrupt/encrypted file, or OCR could not finish a page | yes |
| `empty_resume` | no text on any page after OCR | no |
| `model_error` | Qwen/Ollama unreachable or timed out after retries | yes |
| `invalid_model_output` | not JSON, truncated, or wrong schema after retries | yes |
| `unverifiable` | resume read, but no field could be checked (`validation_status=unverifiable`) | no |
| `unexpected_error` | a bug or unforeseen exception for this one profile (details in `LOG_DIR/enrichment.log`) | yes |

## Troubleshooting

| Symptom | What to do |
|---|---|
| `Add these in Colab > Secrets and enable notebook access` | add the named secret and switch on *Notebook access* |
| `database: OperationalError ...` in preflight | check `NEON_DB_URL` (needs `sslmode=require`), that the Neon project is awake, and the password. Connections are retried 3 times automatically |
| `count` is slow | it scans the table; allow a few minutes (10 min timeout). Use `--skip-count` on `run` to skip it |
| `no GPU` / `partly on CPU` | *Runtime > Change runtime type > GPU*; use L4/A100 if the T4 cannot hold both models |
| `text model not ready: ...` | step 4 did not finish: re-run `rc.start_ollama(); rc.pull_models(); rc.check_gpu()`. See `/content/pipeline_local/logs/ollama.log` |
| many `model_error` | Ollama died (runtime restarted). Run steps 1-4, then `run --retry-errors` |
| many `invalid_model_output` | usually a truncated reply from an oversized resume; check `AI_CONTEXT_TOKENS=16384` matches the loaded context in `preflight` |
| many `download_failed` / `invalid_resume_url` | `CLOUDFLARE_PUBLIC_BASE_URL` missing or wrong (relative `resume_url`s like `/files/resumes/...` need it); check one URL in a browser |
| many `extraction_failed` | scanned PDFs with OCR down; check the `ocr` block of `preflight`, then `run --retry-errors` |
| a resume is reported wrongly | look at `field_statuses` and `discrepancy_reason`; add missing specialty aliases in `profile_enrichment/data/specialty_aliases.csv` and re-validate that profile (delete its row from the CSV, then `run --profile-ids <id>`) |
| CSV looks duplicated | run `status`; it reports ids that appear in more than one file (should be 0) |

## Limits

- Name matching is exact after normalisation: nicknames ("Bob"/"Robert") and spelling variants are reported as discrepancies for a human to judge.
- Only `AI_MAX_CHUNKS` (4) chunks of a very long resume are read; a negative specialty verdict is then not reported (it becomes `not_verifiable`).
- Address validation covers city, state and ZIP only; a street found in the resume is shown in `address_resume` but never compared.
- Speed has not been measured on a GPU for this workflow; `run` prints profiles per minute, so use the test batch to estimate the full run.
  Free Colab sessions end after about 12 h or when idle; the CSV files on Drive survive that.
