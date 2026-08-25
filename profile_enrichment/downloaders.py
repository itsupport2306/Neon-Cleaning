from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin

import requests

from profile_enrichment.config import CONFIG


def resolve_cloudflare_url(file_ref: str | None) -> str | None:
    if not file_ref:
        return None

    if file_ref.startswith("http://") or file_ref.startswith("https://"):
        return file_ref

    base_url = (CONFIG["cloudflare_public_base_url"] or "").rstrip("/")
    return urljoin(f"{base_url}/", file_ref.lstrip("/")) if base_url else file_ref


def download_file(file_ref: str | None, target_dir: str | None = None) -> str | None:
    url = resolve_cloudflare_url(file_ref)
    if not url:
        return None

    target_path = Path(target_dir or CONFIG["download_dir"])
    target_path.mkdir(parents=True, exist_ok=True)

    file_name = os.path.basename(url)
    if not file_name:
        file_name = "download.bin"

    target_file = target_path / file_name
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with target_file.open("wb") as fh:
        fh.write(response.content)

    return str(target_file)
