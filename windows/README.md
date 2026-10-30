# Running with Qwen 2.5 3B on a Windows PC (no Colab)

Works for both `profile_enrichment` (fills NULL specialties) and `resume_collab` (read-only validation).
Same code and same safety rules; only the model and its limits change:

| | Colab (7B) | Windows (3B) |
|---|---|---|
| Text model | `qwen2.5:7b-instruct`, 16k context | `qwen2.5:3b-instruct`, 8k context |
| Scanned resumes | Qwen vision 7B + Tesseract | Tesseract (vision 3B optional) |
| Model requests at once | 2 | 1 |
| Settings | `colab/colab_runtime.py` | [env_3b.bat](env_3b.bat) |

A 3B model is less accurate than 7B. Validation is unchanged (a value must literally occur in the resume), so a weaker
model produces more "not found" results, not more wrong writes. Always run the 25-profile pilot first.

## One-time setup (Command Prompt)

```bat
cd C:\Users\91630\Downloads\resumes_correction
.venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen2.5:3b-instruct
winget install UB-Mannheim.TesseractOCR
```

Close and reopen the Command Prompt after installing Tesseract. Optional vision model (slow on a CPU):
`ollama pull qwen2.5vl:3b`, then `set VISION_OCR_ENABLED=true` after loading the settings.

## Every session

**Window 1: the model server.** Quit the Ollama tray app first (taskbar > right-click the Ollama icon > Quit), then:

```bat
cd C:\Users\91630\Downloads\resumes_correction
windows\start_ollama_3b.bat
```

Leave it open.

**Window 2: the pipeline.**

```bat
cd C:\Users\91630\Downloads\resumes_correction
.venv\Scripts\activate
call windows\env_3b.bat
```

`NEON_DB_URL` is read from `.env`. Check the model is up: `ollama list` should show `qwen2.5:3b-instruct`.

## A. Fill NULL specialties (writes to Neon only with `--apply` / `--yes`)

```bat
python -m profile_enrichment.fill_null_specialty preflight --skip-count

rem pilot: dry run on 25 profiles, writes nothing
python -m profile_enrichment.fill_null_specialty run --new-run --limit 25 --llm-mode always --fetch-workers 2 --model-concurrency 1

rem review what it would write, with the evidence line for each value
python -m profile_enrichment.fill_null_specialty export-prepared --out pilot_prepared.csv
python -m profile_enrichment.fill_null_specialty export-unresolved --out pilot_unresolved.csv

rem only if the pilot is correct: write those reviewed values
python -m profile_enrichment.fill_null_specialty apply --yes

rem full run (writes to Neon)
python -m profile_enrichment.fill_null_specialty run --new-run --apply --llm-mode always --fetch-workers 2 --model-concurrency 1

rem if it stops or the PC restarts: resume the same run (never --new-run again)
python -m profile_enrichment.fill_null_specialty resume --apply --llm-mode always --fetch-workers 2 --model-concurrency 1

rem progress, retries, undo
python -m profile_enrichment.fill_null_specialty status
python -m profile_enrichment.fill_null_specialty resume --apply --retry-failed --llm-mode always --fetch-workers 2 --model-concurrency 1
python -m profile_enrichment.fill_null_specialty rollback --run-id RUN_ID --apply
```

Progress is saved in `state\null_specialty_jobs.sqlite3` after every resume. Ctrl+C once stops cleanly.

## B. Validate stored data against resumes (read-only, never writes to Neon)

```bat
python -m resume_collab.validate_resumes count
python -m resume_collab.validate_resumes preflight
python -m resume_collab.validate_resumes run --limit 25 --workers 2 --model-concurrency 1
python -m resume_collab.validate_resumes run --workers 2 --model-concurrency 1
python -m resume_collab.validate_resumes status
```

Results go to `resume_collab\output\*.csv`. Running the same `run` command again is the resume.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `text model not ready` / connection refused | Window 1 is not running, or the tray app still owns port 11434 (quit it, start `start_ollama_3b.bat`) |
| `model not found` | `ollama pull qwen2.5:3b-instruct` |
| very slow (minutes per resume) | normal on a CPU-only PC: 3B is roughly 20-60 s per resume. Check the pilot's `qwen` timing in `status` before the full run |
| many `timeout` / `truncated` | lower `AI_CHUNK_TOKENS` (e.g. `set AI_CHUNK_TOKENS=3500`) and rerun with `--retry-failed` |
| out of memory | close other apps, or `set AI_CONTEXT_TOKENS=4096` and `set AI_CHUNK_TOKENS=2500` |
| scanned resumes end `extraction_failed` | Tesseract not installed or not found: `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `'python' is not recognized` | run `.venv\Scripts\activate` first |
