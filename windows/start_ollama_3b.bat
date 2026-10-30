@echo off
rem Starts the Ollama server with settings suited to a PC. Leave this window open while the pipeline runs.
rem First quit the Ollama tray app (right-click its icon in the taskbar > Quit), otherwise port 11434 is taken.

set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_KEEP_ALIVE=-1
set OLLAMA_CONTEXT_LENGTH=8192
set OLLAMA_FLASH_ATTENTION=1

ollama serve
