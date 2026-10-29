"""Google Colab runtime for the NULL-specialty pipeline (profile_enrichment.fill_null_specialty).

Colab specifics this module takes care of, so the pipeline itself stays unchanged:

* Ollama on the Colab GPU serves both Qwen 7B models (text extraction + vision OCR) on 127.0.0.1:11434;
* secrets come from Colab's Secrets panel and only ever live in this process's environment (no .env file);
* the job store (SQLite, WAL) lives on the VM's local disk - WAL does not work on the Google Drive FUSE
  mount - and a consistent snapshot is copied to Drive every few minutes and when a command ends, so a
  disconnected runtime resumes from the last snapshot instead of from zero.

Use from a notebook cell::

    import colab_runtime as cr
    cr.configure(drive_dir="/content/drive/MyDrive/resume_pipeline")
    cr.start_ollama(); cr.pull_models(); cr.check_gpu()
    cr.restore_state()
    cr.pipeline("preflight", "--require-ocr")
    cr.pipeline("run", "--new-run", "--limit", "25")      # dry run on a small sample

See colab/README.md for the full step-by-step procedure.
"""

from __future__ import annotations

import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import requests

REPO_DIR = Path(__file__).resolve().parent.parent
LOCAL_DIR = Path("/content/pipeline_local")  # VM disk: fast, supports SQLite WAL, lost on disconnect
OLLAMA_URL = "http://127.0.0.1:11434"

TEXT_MODEL = "qwen2.5:7b-instruct"
VISION_MODEL = "qwen2.5vl:7b"
CONTEXT_TOKENS = 16384  # a whole resume fits one chunk, so the name header and the specialty are seen together

SECRET_NAMES = ("NEON_DB_URL", "CLOUDFLARE_PUBLIC_BASE_URL")

_drive_dir: Optional[Path] = None
_ollama: Optional[subprocess.Popen] = None


def _log(message: str) -> None:
    print(f"[colab] {message}", flush=True)


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

def pipeline_env() -> Dict[str, str]:
    """Settings for a single-GPU Colab runtime. Every value can still be overridden before `configure`."""
    return {
        # text model: Ollama native API (grammar-constrained JSON, explicit num_ctx, done_reason)
        "LOCAL_MODEL_BACKEND": "ollama",
        "LOCAL_MODEL_BASE_URL": OLLAMA_URL,
        "LOCAL_MODEL_NAME": TEXT_MODEL,
        "AI_CONTEXT_TOKENS": str(CONTEXT_TOKENS),
        "AI_CHUNK_TOKENS": "12000",
        "AI_CHUNK_OVERLAP_TOKENS": "200",
        "AI_MAX_CHUNKS": "4",
        "AI_REQUEST_TIMEOUT": "180",
        "AI_MAX_ATTEMPTS": "3",
        "AI_RETRY_BACKOFF": "2.0",
        # vision OCR for scanned resumes: same Ollama server, OpenAI-compatible endpoint; Tesseract is the fallback
        "VISION_OCR_ENABLED": "true",
        "VISION_OCR_BASE_URL": f"{OLLAMA_URL}/v1",
        "VISION_OCR_API_KEY": "ollama",
        "VISION_OCR_MODEL": VISION_MODEL,
        "VISION_OCR_TIMEOUT": "180",
        "VISION_OCR_MAX_IMAGE_SIDE": "1600",
        "OCR_ENABLED": "true",
        "TESSERACT_CMD": shutil.which("tesseract") or "/usr/bin/tesseract",
        # local state (snapshotted to Drive)
        "JOB_STORE_PATH": str(LOCAL_DIR / "state" / "null_specialty_jobs.sqlite3"),
        "CACHE_DIR": str(LOCAL_DIR / "cache"),
        "LOG_DIR": str(LOCAL_DIR / "logs"),
        "DOC_CACHE_MAX_BYTES": str(5 * 1024**3),
        # database / network
        "DB_POOL_MAX": "4",
        "DB_CONNECT_TIMEOUT": "30",
        "DOWNLOAD_TIMEOUT": "45",
        "DOWNLOAD_ATTEMPTS": "4",
        "WORKER_COUNT": "4",
        # data integrity: never replace a populated column; corrections go to export-review instead
        "ALLOW_OVERWRITE_EXISTING": "false",
    }


def load_secrets() -> None:
    """Copy required secrets from Colab's Secrets panel (key icon) into os.environ. Values are never printed."""
    from google.colab import userdata  # type: ignore

    missing = []
    for name in SECRET_NAMES:
        try:
            value = userdata.get(name)
        except Exception:  # noqa: BLE001 - SecretNotFoundError / NotebookAccessError
            value = None
        if value:
            os.environ[name] = value.strip()
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Add these in Colab > Secrets and enable notebook access: {', '.join(missing)}")
    url = os.environ["NEON_DB_URL"]
    if "sslmode=" not in url:
        os.environ["NEON_DB_URL"] = url + ("&" if "?" in url else "?") + "sslmode=require"
    elif "sslmode=disable" in url:
        raise RuntimeError("NEON_DB_URL has sslmode=disable; Neon needs sslmode=require")
    _log("secrets loaded: " + ", ".join(SECRET_NAMES))


def configure(drive_dir: str, overrides: Optional[Dict[str, str]] = None) -> None:
    """Load secrets, apply the Colab settings and create the local + Drive folders."""
    global _drive_dir
    _drive_dir = Path(drive_dir)
    (_drive_dir / "state").mkdir(parents=True, exist_ok=True)
    (_drive_dir / "reports").mkdir(parents=True, exist_ok=True)
    for sub in ("state", "cache", "logs", "reports"):
        (LOCAL_DIR / sub).mkdir(parents=True, exist_ok=True)
    load_secrets()
    for key, value in pipeline_env().items():
        os.environ.setdefault(key, value)
    os.environ.update(overrides or {})
    _log(f"local state: {LOCAL_DIR} | Drive backups: {_drive_dir}")


# ---------------------------------------------------------------------------------------------
# Ollama on the GPU
# ---------------------------------------------------------------------------------------------

def _ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def start_ollama() -> None:
    """Start `ollama serve` in the background (idempotent). Logs go to <local>/logs/ollama.log."""
    global _ollama
    if _ollama_up():
        _log("ollama already running")
        return
    if not shutil.which("ollama"):
        raise RuntimeError("ollama is not installed: run the install cell first")
    env = {
        **os.environ,
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_MAX_LOADED_MODELS": "2",  # keep text + vision resident: no reload between a scan and its extraction
        "OLLAMA_NUM_PARALLEL": "2",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",  # halves KV memory so both 7B models + 16k context fit a 15 GB T4
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_CONTEXT_LENGTH": "8192",  # default for the vision endpoint; text requests send num_ctx explicitly
    }
    log = open(LOCAL_DIR / "logs" / "ollama.log", "ab")
    _ollama = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        if _ollama_up():
            _log("ollama started")
            return
        time.sleep(1)
    raise RuntimeError(f"ollama did not start; see {LOCAL_DIR / 'logs' / 'ollama.log'}")


def pull_models() -> None:
    for model in (TEXT_MODEL, VISION_MODEL):
        _log(f"pulling {model} ...")
        subprocess.run(["ollama", "pull", model], check=True)


def check_gpu() -> None:
    """Fail loudly unless both models are loaded fully into GPU memory (a CPU fallback is ~20x slower)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("no GPU: Runtime > Change runtime type > GPU (T4 / L4 / A100)")
    _log(f"GPU: {out.stdout.strip()}")
    # load each model with the context it will be used with (a different num_ctx would force a reload later)
    for model, options in ((TEXT_MODEL, {"num_ctx": CONTEXT_TOKENS}), (VISION_MODEL, {})):
        response = requests.post(f"{OLLAMA_URL}/api/generate", timeout=600, json={
            "model": model, "prompt": "ok", "stream": False, "keep_alive": -1, "options": {**options, "num_predict": 1},
        })
        response.raise_for_status()
    loaded = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    for item in loaded:
        size, vram = item.get("size", 0), item.get("size_vram", 0)
        share = vram / size if size else 0
        _log(f"{item['name']}: {vram / 1e9:.1f} GB of {size / 1e9:.1f} GB on GPU ({share:.0%})")
        if share < 0.99:
            raise RuntimeError(f"{item['name']} is partly on CPU; use a larger GPU or VISION_OCR_ENABLED=false")
    if len(loaded) < 2:
        raise RuntimeError(f"expected both models resident, ollama reports {[m['name'] for m in loaded]}")


# ---------------------------------------------------------------------------------------------
# state on Drive
# ---------------------------------------------------------------------------------------------

def _store_path() -> Path:
    return Path(os.environ["JOB_STORE_PATH"])


def _snapshot_path() -> Path:
    if _drive_dir is None:
        raise RuntimeError("call configure() first")
    return _drive_dir / "state" / "null_specialty_jobs.sqlite3"


def snapshot_state(quiet: bool = False) -> None:
    """Copy a transactionally consistent snapshot of the job store and the logs/reports to Drive."""
    source = _store_path()
    if not source.exists():
        return
    local_copy = source.with_name("snapshot.sqlite3")
    src, dst = sqlite3.connect(str(source), timeout=60), sqlite3.connect(str(local_copy))
    try:
        src.backup(dst)  # consistent even while the pipeline is writing
        dst.execute("PRAGMA journal_mode=DELETE")  # fold any -wal content in: the copy must be one self-contained file
    finally:
        dst.close()
        src.close()
    target = _snapshot_path()
    staging = target.with_name(target.name + ".tmp")
    shutil.copyfile(local_copy, staging)
    os.replace(staging, target)  # Drive never holds a half-written snapshot
    for sub in ("logs", "reports"):
        for path in (LOCAL_DIR / sub).glob("*"):
            if path.is_file() and path.suffix in {".log", ".csv", ".json"}:
                shutil.copyfile(path, _drive_dir / "reports" / path.name)
    if not quiet:
        _log(f"state snapshot saved to {target}")


def restore_state() -> None:
    """On a fresh VM, bring back the last Drive snapshot. Tasks claimed by the dead runtime go back to pending."""
    local, snapshot = _store_path(), _snapshot_path()
    if local.exists():
        _log("local job store present (same runtime): nothing to restore")
        return
    if not snapshot.exists():
        _log("no snapshot on Drive: starting fresh")
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(snapshot, local)
    conn = sqlite3.connect(str(local))
    try:
        with conn:
            released = conn.execute(
                "UPDATE tasks SET state='pending', lease_owner=NULL, lease_expires=NULL WHERE state='claimed'"
            ).rowcount
    finally:
        conn.close()
    _log(f"restored job store from Drive; {released} in-flight tasks returned to pending")


# ---------------------------------------------------------------------------------------------
# running the pipeline
# ---------------------------------------------------------------------------------------------

def pipeline(*args: str, snapshot_every: float = 120.0) -> int:
    """Run `python -m profile_enrichment.fill_null_specialty <args>`, streaming its output.

    The job store is snapshotted to Drive every `snapshot_every` seconds and once more at the end. Interrupting
    the cell (stop button) sends SIGINT: the runner finishes its in-flight tasks, releases its leases and exits.
    """
    command = [sys.executable, "-m", "profile_enrichment.fill_null_specialty", *args]
    _log("$ " + " ".join(command[1:]))
    # own session: the notebook's interrupt reaches the child only once, as the SIGINT forwarded below
    process = subprocess.Popen(command, cwd=str(REPO_DIR), env=dict(os.environ), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    stop = threading.Event()

    def backups() -> None:
        while not stop.wait(snapshot_every):
            try:
                snapshot_state(quiet=True)
            except Exception as exc:  # noqa: BLE001 - a failed backup must not kill the run
                _log(f"snapshot failed (will retry): {exc}")

    thread = threading.Thread(target=backups, daemon=True)
    thread.start()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        _log("interrupt: asking the pipeline to stop cleanly ...")
        process.send_signal(signal.SIGINT)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
    finally:
        stop.set()
        thread.join(timeout=5)
        snapshot_state()
    _log(f"exit code {code}")
    return code


def export(command: str, filename: str) -> Path:
    """Run an export command (export-prepared / export-review / export-unresolved) into reports/ and back it up."""
    out = LOCAL_DIR / "reports" / filename
    pipeline(command, "--out", str(out))
    return out


def resume_until_done(*args: str, max_passes: int = 20) -> int:
    """`resume` the latest unfinished run until no task is pending (exit code 2 = incomplete).

    Uses `resume`, never `run`: `run` with no unfinished run would start a *new* run over every NULL row.
    Stops on 0 (done), 1 (done with retryable failures: use `--retry-failed`) and 3 (preflight / nothing to resume).
    """
    code = 2
    for attempt in range(1, max_passes + 1):
        code = pipeline("resume", *args)
        if code != 2:
            break
        _log(f"pass {attempt} incomplete; resuming in 30 s")
        time.sleep(30)
    return code
