@echo off
rem Settings for Qwen 2.5 3B on a Windows machine (works for profile_enrichment AND resume_collab).
rem Usage, in every new Command Prompt, from the repo folder:   call windows\env_3b.bat
rem These are process environment variables; they override the same names in .env (NEON_DB_URL stays in .env).

set LOCAL_MODEL_BACKEND=ollama
set LOCAL_MODEL_BASE_URL=http://127.0.0.1:11434
set LOCAL_MODEL_NAME=qwen2.5:3b-instruct

rem 8k context (the Colab 7B setup used 16k): a 3B model on a PC runs out of memory/time far sooner.
set AI_CONTEXT_TOKENS=8192
set AI_CHUNK_TOKENS=5500
set AI_CHUNK_OVERLAP_TOKENS=150
set AI_MAX_CHUNKS=4
set AI_REQUEST_TIMEOUT=300
set AI_MAX_ATTEMPTS=3

rem Scanned resumes: Tesseract only by default (the 3B vision model is very slow on a CPU).
rem To use the vision model as well:  set VISION_OCR_ENABLED=true   (after: ollama pull qwen2.5vl:3b)
set OCR_ENABLED=true
set VISION_OCR_ENABLED=false
set VISION_OCR_BASE_URL=http://127.0.0.1:11434/v1
set VISION_OCR_API_KEY=ollama
set VISION_OCR_MODEL=qwen2.5vl:3b
set VISION_OCR_TIMEOUT=300
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

rem One model request at a time is right for a single PC; downloads/OCR can run in parallel.
set WORKER_COUNT=2
set DB_POOL_MAX=4
set DOWNLOAD_TIMEOUT=45
set DOWNLOAD_ATTEMPTS=4
set ALLOW_OVERWRITE_EXISTING=false

echo 3B settings loaded: model=%LOCAL_MODEL_NAME% ctx=%AI_CONTEXT_TOKENS% vision=%VISION_OCR_ENABLED%
