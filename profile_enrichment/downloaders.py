from __future__ import annotations

from urllib.parse import urljoin

import requests

from profile_enrichment.config import CONFIG


def resolve_cloudflare_url(file_ref: str | None) -> str | None:
    if not file_ref:
        return None

    if file_ref.startswith("http://") or file_ref.startswith("https://"):
        return file_ref

    cleaned_ref = file_ref.lstrip("/")
    cleaned_ref = cleaned_ref.replace("files/", "", 1) if cleaned_ref.startswith("files/") else cleaned_ref

    base_url = (CONFIG["cloudflare_public_base_url"] or "").rstrip("/")
    return urljoin(f"{base_url}/", cleaned_ref) if base_url else cleaned_ref


def download_file_bytes(file_ref: str | None) -> bytes | None:
    """Download file and return as bytes without saving to disk."""
    url = resolve_cloudflare_url(file_ref)
    if not url:
        return None

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content
