# Qwen 2.5 3B settings for PowerShell (profile_enrichment AND resume_collab).
# Load them into the CURRENT session with a leading dot:   . .\windows\env_3b.ps1
# (running it without the dot would set the variables in a child scope that disappears immediately)
# NEON_DB_URL / CLOUDFLARE_PUBLIC_BASE_URL come from .env or from $env:NEON_DB_URL = "..." typed in the session.

$env:LOCAL_MODEL_BACKEND  = "ollama"
$env:LOCAL_MODEL_BASE_URL = "http://127.0.0.1:11434"
$env:LOCAL_MODEL_NAME     = "qwen2.5:3b-instruct"

# 8k context: the 7B/16k Colab setup does not fit a small server.
$env:AI_CONTEXT_TOKENS       = "8192"
$env:AI_CHUNK_TOKENS         = "5500"
$env:AI_CHUNK_OVERLAP_TOKENS = "150"
$env:AI_MAX_CHUNKS           = "4"
$env:AI_REQUEST_TIMEOUT      = "300"
$env:AI_MAX_ATTEMPTS         = "3"

# Scanned resumes: Tesseract only (the vision model is very slow without a GPU).
# To use the vision model too:  $env:VISION_OCR_ENABLED = "true"   (after: ollama pull qwen2.5vl:3b)
$env:OCR_ENABLED          = "true"
$env:VISION_OCR_ENABLED   = "false"
$env:VISION_OCR_BASE_URL  = "http://127.0.0.1:11434/v1"
$env:VISION_OCR_API_KEY   = "ollama"
$env:VISION_OCR_MODEL     = "qwen2.5vl:3b"
$env:VISION_OCR_TIMEOUT   = "300"
$tesseract = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tesseract) { $env:TESSERACT_CMD = $tesseract }

$env:WORKER_COUNT            = "2"
$env:DB_POOL_MAX             = "4"
$env:DOWNLOAD_TIMEOUT        = "45"
$env:DOWNLOAD_ATTEMPTS       = "4"
$env:ALLOW_OVERWRITE_EXISTING = "false"

Write-Host "3B settings loaded: model=$env:LOCAL_MODEL_NAME ctx=$env:AI_CONTEXT_TOKENS vision=$env:VISION_OCR_ENABLED"
