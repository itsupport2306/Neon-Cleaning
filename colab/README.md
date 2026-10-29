# Running the NULL-specialty pipeline on Google Colab (GPU)

Notebook: [resume_pipeline_colab.ipynb](resume_pipeline_colab.ipynb). Runtime helper: [colab_runtime.py](colab_runtime.py).

The notebook runs the existing `profile_enrichment.fill_null_specialty` pipeline. Nothing about the selection,
validation or write rules changes on Colab. Only the place the model runs changes: the Colab GPU instead of a CPU.

| Piece | Colab setting |
|---|---|
| Text extraction | `qwen2.5:7b-instruct` on Ollama (native API, JSON-schema constrained), 16k context |
| Scanned resumes | `qwen2.5vl:7b` on the same Ollama server (`/v1`), Tesseract as fallback |
| GPU memory | flash attention + q8 KV cache: both 7B models and a 16k context fit a 15 GB T4 |
| Job store | SQLite on the VM disk (`/content/pipeline_local`), snapshot to Drive every 2 min and on stop |
| Secrets | Colab Secrets: `NEON_DB_URL`, `CLOUDFLARE_PUBLIC_BASE_URL`, `GITHUB_TOKEN`. No `.env` on Colab |

## One-time setup

1. Push this branch to GitHub. The notebook clones `itsupport2306/Neon-Cleaning` (branch `main` by default; change
   `BRANCH` in step 2 if needed).
2. Upload `colab/resume_pipeline_colab.ipynb` to Colab (*File > Upload notebook*) or open it from GitHub.
3. *Runtime > Change runtime type*: **T4 GPU** (or L4 / A100 on Colab Pro, about 2-4x faster).
4. *Secrets* (key icon): add `NEON_DB_URL` (with `sslmode=require`; it is added if missing),
   `CLOUDFLARE_PUBLIC_BASE_URL` and `GITHUB_TOKEN` (a fine-grained token with read access to the repo). Turn on
   *Notebook access* for each one.

## Every session

Run steps 1-4: GPU and Drive, clone, install, configure. `cr.check_gpu()` stops the notebook if either model is not
fully in GPU memory, because a silent CPU fallback is about 20x slower.

## First use: pilot, then the full run

| Step | What it does | Writes to Neon? |
|---|---|---|
| 5 | offline unit tests | no |
| 6 | preflight: DB reachable, `profile_id` unique index, model canary extraction, OCR engines | no |
| 7 | dry run on 25 random NULL-specialty profiles | no |
| 7a | `export-prepared` (every value that would be written + its evidence line), `export-unresolved` | no |
| 8 | `apply --yes`: writes exactly the reviewed pilot proposals, verified by read-back | yes |
| 9a | full run with `--apply` | yes |
| 10 | one `--retry-failed` pass, final status, unresolved and review exports | yes |

To undo a run: `cr.pipeline('rollback', '--run-id', '<run_id>', '--apply')`. It only reverts a column that still holds
exactly the value the run wrote.

## After a disconnect

Run steps 1-4 again, then **9b** (`cr.resume_until_done('--apply', ...)`). `restore_state()` copies the last Drive
snapshot back and returns the tasks that were in flight to `pending`. Do **not** run 9a again: `--new-run` starts a
new manifest.

The snapshot is at most 2 minutes old. A resume that was written to Neon in those last 2 minutes is processed again on
resume. The pipeline then sees that `specialty` is already filled and records `concurrent_change /
specialty_already_filled` without writing anything. Its audit row in the `writes` table, which `rollback` uses, is lost.
Lower the risk by passing `snapshot_every=30` to `cr.pipeline(...)`.

## Data-integrity rules (enforced by the pipeline, unchanged)

- A value is written only when it passes shape and placeholder checks and is found in the resume text. A specialty must
  also be a label from `data/specialty_vocabulary.csv` (labels already used in Neon) or an alias of one.
- Only empty columns are filled. A different value for a populated column goes to `export-review` for a human.
- `phone` is written only while it is NULL. An email already owned by another profile is skipped, and the other fields
  are still written.
- Each UPDATE targets one `profile_id` with a lost-update guard, and is read back on a second connection.
- `screen_reason` is set to `AI` only when the model's run succeeded and at least one field was written.

## Notes

- **Address:** the `profiles` table has `city` and `state_code` and no street-address column, so those are the
  address fields that get filled.
- **Non-clinical resumes** (warehouse, IT, logistics) get no specialty, because the vocabulary is healthcare
  specialties already used in Neon. They end as `insufficient_evidence` with reason `no_healthcare_context` or
  `specialty_not_in_taxonomy: [...]`. If a clinical label is missing, add it to `data/specialty_aliases.csv` and re-run
  with `--retry-unresolved`.
- **Speed:** the last CPU run averaged about 45 s of model time per resume. GPU speed has not been measured yet.
  After the pilot, `status` prints p50/p95 per stage (`qwen`, `ocr`, `download`); use those numbers to estimate the
  full run. Free Colab sessions end after about 12 h or when idle. Progress survives that, but keep the tab active.
