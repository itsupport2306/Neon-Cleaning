"""Google Colab runtime for the read-only resume validation (resume_collab.validate_resumes).

Everything about the GPU, Ollama, the two Qwen 7B models and the Colab secrets is the existing
``colab/colab_runtime.py`` and is reused as is: this module only adds where the CSV files go and how the
validator is launched.

Use from a notebook cell::

    import sys; sys.path.insert(0, f"{REPO}/resume_collab")
    import colab_runtime as rc
    rc.configure("/content/drive/MyDrive/resume_validation")
    rc.start_ollama(); rc.pull_models(); rc.check_gpu()
    rc.validate("count")
    rc.validate("run", "--limit", "25")

The CSV files are written straight to Google Drive (``<drive_dir>/reports``), one row at a time, so a
disconnected runtime loses nothing that was already recorded. See resume_collab/README.md.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_DIR = Path(__file__).resolve().parent.parent

# Load the existing runtime under a private name: this file is also called colab_runtime.
_spec = importlib.util.spec_from_file_location("_colab_base_runtime", REPO_DIR / "colab" / "colab_runtime.py")
_base = importlib.util.module_from_spec(_spec)
sys.modules["_colab_base_runtime"] = _base
_spec.loader.exec_module(_base)  # type: ignore[union-attr]

start_ollama, pull_models, check_gpu = _base.start_ollama, _base.pull_models, _base.check_gpu
TEXT_MODEL, VISION_MODEL = _base.TEXT_MODEL, _base.VISION_MODEL

_log = _base._log
_output_dir: Optional[Path] = None


def configure(drive_dir: str, overrides: Optional[Dict[str, str]] = None) -> Path:
    """Load the Colab secrets, apply the same Qwen/Ollama settings as `colab`, and choose the CSV folder."""
    global _output_dir
    _base.configure(drive_dir, overrides={"NEON_TABLE_NAME": "profiles", **(overrides or {})})
    _output_dir = Path(os.environ.setdefault("OUTPUT_DIR", str(Path(drive_dir) / "reports")))
    _output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"validation CSV files: {_output_dir}")
    return _output_dir


def validate(*args: str) -> int:
    """Run `python -m resume_collab.validate_resumes <args>`, streaming its output.

    Interrupting the cell (stop button) forwards one SIGINT: the validator finishes the profiles it has started,
    records them and exits. Run the same command again to continue.
    """
    command = [sys.executable, "-m", "resume_collab.validate_resumes", *args]
    _log("$ " + " ".join(command[1:]))
    process = subprocess.Popen(command, cwd=str(REPO_DIR), env=dict(os.environ), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        _log("interrupt: asking the validator to stop cleanly ...")
        process.send_signal(signal.SIGINT)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
    _log(f"exit code {code}")
    return code


def output_dir() -> Path:
    if _output_dir is None:
        raise RuntimeError("call configure() first")
    return _output_dir
