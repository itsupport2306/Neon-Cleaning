# Starts the Ollama server with settings suited to a server. Leave this PowerShell window open.
# Run from the repo folder:   .\windows\start_ollama_3b.ps1
# If the Ollama tray app is running, quit it first (it already owns port 11434 with other settings).

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
if (-not (Test-Path $ollama)) { throw "Ollama is not installed. Install it first (see windows\README.md, 'Install Ollama')." }

$env:OLLAMA_HOST              = "127.0.0.1:11434"
$env:OLLAMA_NUM_PARALLEL      = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE        = "-1"
$env:OLLAMA_CONTEXT_LENGTH    = "8192"
$env:OLLAMA_FLASH_ATTENTION   = "1"

& $ollama serve
