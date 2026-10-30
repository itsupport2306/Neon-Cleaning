# Running on a Windows server with Qwen 2.5 3B (PowerShell)

Works for `profile_enrichment` (fills NULL specialties, `--fix-specialty`) and `resume_collab` (validation, with an
optional write-back of every discrepancy to Neon). `.bat` versions of the scripts exist for Command Prompt; in PowerShell
use the `.ps1` ones, because `call env_3b.bat` cannot set variables in a PowerShell session.

## 1. One-time setup on the server

```powershell
cd C:\Users\admin\Downloads\Worker-clean\Neon-Cleaning
git pull                                  # get these scripts
python -m venv .venv
.\.venv\Scripts\Activate.ps1              # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```

**Install Ollama** (the `'ollama' is not recognized` error means it is not installed on that server):

```powershell
winget install -e --id Ollama.Ollama
```

No winget? Download `OllamaSetup.exe` from https://ollama.com/download/windows and run it (no admin rights needed).
Then **close and reopen PowerShell** so `ollama` is on the PATH, and check `ollama --version`.

Optional OCR for scanned resumes: `winget install -e --id UB-Mannheim.TesseractOCR`.

Create `.env` in the repo folder (it is not in git) with at least:

```
NEON_DB_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require
CLOUDFLARE_PUBLIC_BASE_URL=https://pub-xxxx.r2.dev
```

## 2. Every session: two PowerShell windows

**Window 1: the model server.** Leave it open.

```powershell
cd C:\Users\admin\Downloads\Worker-clean\Neon-Cleaning
.\windows\start_ollama_3b.ps1
```

**Window 2: the work.** The first time, download the model:

```powershell
cd C:\Users\admin\Downloads\Worker-clean\Neon-Cleaning
.\.venv\Scripts\Activate.ps1
. .\windows\env_3b.ps1                    # note the leading dot
ollama pull qwen2.5:3b-instruct           # first time only (about 2 GB)
ollama list
```

## 3. Commands (all write straight to Neon; nothing is held for review)

### A. Fill NULL specialties
```powershell
python -m profile_enrichment.fill_null_specialty preflight --skip-count
python -m profile_enrichment.fill_null_specialty run --new-run --apply --llm-mode always --fetch-workers 2 --model-concurrency 1
# stopped / server rebooted: continue the same run (never --new-run again)
python -m profile_enrichment.fill_null_specialty resume --apply --llm-mode always --fetch-workers 2 --model-concurrency 1
python -m profile_enrichment.fill_null_specialty status
```
For a first look without writing, drop `--apply` and add `--limit 25`.

### B. Fix wrong specialties (values like a phone number, email or "web")
```powershell
python -m profile_enrichment.processor --fix-specialty --workers 2
```

### C. Validate against resumes and write every discrepancy to Neon
```powershell
python -m resume_collab.validate_resumes preflight
python -m resume_collab.validate_resumes run --apply-to-neon --workers 2 --model-concurrency 1
# continue after a stop: run the same command again
# apply discrepancies that an earlier read-only run (e.g. on Colab) already recorded:
python -m resume_collab.validate_resumes run --apply-to-neon --redo-discrepancies --workers 2 --model-concurrency 1
python -m resume_collab.validate_resumes status
```
Without `--apply-to-neon` the validator stays read-only.

What `--apply-to-neon` writes: for each mismatched field of a profile, the resume's value replaces the stored one
(`first_name`, `last_name`, `specialty`, `city`, `state_code`, `zip_code`). Safeguards that remain:

- a column is changed only while it still holds the value that was read (a concurrent edit is never overwritten);
- a specialty is written only if it maps to a label of `profile_enrichment/data/specialty_vocabulary.csv` / aliases;
  names and cities must pass the usual shape checks (no digits, sentences, placeholders); everything skipped is listed;
- every old and new value is logged first in `resume_collab\output\resume_validation_neon_updates.csv`.

Undo (only restores columns that still hold what was written):
```powershell
python -m resume_collab.validate_resumes revert-updates              # dry run
python -m resume_collab.validate_resumes revert-updates --apply
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ollama` is not recognized | Ollama not installed, or PowerShell was not reopened after installing |
| `text model not ready` / connection refused | Window 1 is not running, or the tray app owns port 11434 (quit it, rerun `start_ollama_3b.ps1`) |
| `model not found` | `ollama pull qwen2.5:3b-instruct` |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope Process Bypass` in that window |
| settings not applied | you ran `.\windows\env_3b.ps1` without the leading dot |
| slow (tens of seconds per resume) | normal without a GPU; check timings with `fill_null_specialty status` |
| many `timeout` / `truncated` | `$env:AI_CHUNK_TOKENS = "3500"`, then rerun (`--retry-failed` for fill_null_specialty, `--retry-errors` for the validator) |
| scanned resumes fail | install Tesseract, reopen PowerShell, re-run `. .\windows\env_3b.ps1` |
