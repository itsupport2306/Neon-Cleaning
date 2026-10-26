from __future__ import annotations

import threading
import time
from urllib.parse import urljoin

import requests

from profile_enrichment.config import CONFIG

MAX_DOCUMENT_BYTES = 40 * 1024 * 1024
_local = threading.local()


class DownloadError(Exception):
    """kind: 'not_found' (404/410 - the file is gone), 'transient' (retry later), 'permanent' (other 4xx / too large)."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


def resolve_cloudflare_url(file_ref: str | None) -> str | None:
    if not file_ref:
        return None

    if file_ref.startswith("http://") or file_ref.startswith("https://"):
        return file_ref

    cleaned_ref = file_ref.lstrip("/")
    cleaned_ref = cleaned_ref.replace("files/", "", 1) if cleaned_ref.startswith("files/") else cleaned_ref

    base_url = (CONFIG["cloudflare_public_base_url"] or "").rstrip("/")
    return urljoin(f"{base_url}/", cleaned_ref) if base_url else cleaned_ref


def _session() -> requests.Session:
    session = getattr(_local, "session", None)
    if session is None:
        session = _local.session = requests.Session()
    return session


def fetch_bytes(file_ref: str | None) -> bytes:
    """Download with bounded retries. Raises DownloadError; never returns empty bytes silently."""
    url = resolve_cloudflare_url(file_ref)
    if not url or not url.startswith(("http://", "https://")):
        raise DownloadError("permanent", f"resume reference is not a downloadable URL: {file_ref!r}")

    attempts = CONFIG["download_attempts"]
    last: DownloadError | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = _session().get(url, timeout=CONFIG["download_timeout"], stream=True)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last = DownloadError("transient", f"{type(exc).__name__}: {exc}")
        except requests.exceptions.RequestException as exc:
            raise DownloadError("permanent", f"{type(exc).__name__}: {exc}") from exc
        else:
            status = response.status_code
            if status in (404, 410):
                raise DownloadError("not_found", f"HTTP {status}", status)
            if status >= 500 or status in (408, 425, 429):
                last = DownloadError("transient", f"HTTP {status}", status)
            elif status >= 400:
                raise DownloadError("permanent", f"HTTP {status}", status)
            else:
                length = int(response.headers.get("Content-Length") or 0)
                if length > MAX_DOCUMENT_BYTES:
                    raise DownloadError("permanent", f"document too large ({length} bytes)", status)
                body = response.content
                if len(body) > MAX_DOCUMENT_BYTES:
                    raise DownloadError("permanent", f"document too large ({len(body)} bytes)", status)
                if not body:
                    last = DownloadError("transient", "empty response body", status)
                else:
                    return body
        if attempt < attempts:
            time.sleep(min(10.0, 1.0 * 2 ** (attempt - 1)))
    assert last is not None
    raise last


def download_file_bytes(file_ref: str | None) -> bytes | None:
    """Legacy entry point: None when there is no reference, otherwise the bytes (raises on failure)."""
    if not file_ref:
        return None
    return fetch_bytes(file_ref)
