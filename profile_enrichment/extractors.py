from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional

from PIL import Image

from profile_enrichment.config import CONFIG

logger = logging.getLogger(__name__)


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


_TESSERACT_UNAVAILABLE_REASON: Optional[str] = None
_VISION_UNAVAILABLE_REASON: Optional[str] = None


def _configure_tesseract() -> None:
    import pytesseract

    if CONFIG["tesseract_cmd"]:
        pytesseract.pytesseract.tesseract_cmd = CONFIG["tesseract_cmd"]


def _disable_tesseract(reason: str) -> None:
    """Remember that Tesseract is unusable so we stop retrying it for every page."""
    global _TESSERACT_UNAVAILABLE_REASON
    if _TESSERACT_UNAVAILABLE_REASON is None:
        _TESSERACT_UNAVAILABLE_REASON = reason
        logger.warning("TESSERACT_DISABLED | reason=%s", reason)


def _disable_vision_ocr(reason: str) -> None:
    """Remember that the Qwen vision endpoint is unusable for the rest of the run."""
    global _VISION_UNAVAILABLE_REASON
    if _VISION_UNAVAILABLE_REASON is None:
        _VISION_UNAVAILABLE_REASON = reason
        logger.warning("VISION_OCR_DISABLED | reason=%s", reason)


def _vision_ocr_base_url() -> Optional[str]:
    return CONFIG["vision_ocr_base_url"] or CONFIG["local_model_base_url"]


def _image_to_data_uri(image: Image.Image) -> str:
    rgb_image = image.convert("RGB")
    max_side = CONFIG["vision_ocr_max_image_side"]
    if max_side and max(rgb_image.size) > max_side:
        ratio = max_side / max(rgb_image.size)
        new_size = (max(1, int(rgb_image.width * ratio)), max(1, int(rgb_image.height * ratio)))
        rgb_image = rgb_image.resize(new_size)

    buffer = io.BytesIO()
    rgb_image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _qwen_vision_ocr(image: Image.Image) -> str:
    """Transcribe an image with the Qwen vision model. Returns "" when unavailable."""
    if not CONFIG["vision_ocr_enabled"] or _VISION_UNAVAILABLE_REASON is not None:
        return ""

    base_url = _vision_ocr_base_url()
    if not base_url:
        _disable_vision_ocr("no VISION_OCR_BASE_URL / LOCAL_MODEL_BASE_URL configured")
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url.rstrip("/") + "/",
            api_key=CONFIG["vision_ocr_api_key"] or CONFIG["local_model_api_key"] or "local-qwen",
            timeout=CONFIG["vision_ocr_timeout"],
        )
        response = client.chat.completions.create(
            model=CONFIG["vision_ocr_model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR engine. Transcribe the page verbatim as plain text. No commentary.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe every piece of text in this resume page, preserving reading order. "
                                "Include names, phone numbers, emails, addresses and headings. "
                                "If the page has no text, reply with an empty response."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=CONFIG["vision_ocr_max_tokens"],
        )
    except Exception as exc:
        _disable_vision_ocr(f"{type(exc).__name__}: {exc}")
        return ""

    return response.choices[0].message.content or ""


def _tesseract_ocr(image: Image.Image) -> str:
    if not CONFIG["ocr_enabled"] or _TESSERACT_UNAVAILABLE_REASON is not None:
        return ""

    try:
        import pytesseract

        _configure_tesseract()
        return pytesseract.image_to_string(image.convert("RGB"))
    except (PermissionError, FileNotFoundError, OSError) as exc:
        _disable_tesseract(f"tesseract not runnable ({exc})")
        return ""
    except Exception as exc:
        if "TesseractNotFound" in type(exc).__name__:
            _disable_tesseract("tesseract binary not found")
        else:
            logger.debug("TESSERACT_PAGE_SKIPPED | %s: %s", type(exc).__name__, exc)
        return ""


def _needs_ocr(native_text: str) -> bool:
    """Skip OCR when the document already yielded enough native text."""
    if not CONFIG["ocr_skip_when_native_text"]:
        return True
    return not has_meaningful_text(native_text)


def _ocr_is_available() -> bool:
    vision_possible = (
        CONFIG["vision_ocr_enabled"] and _VISION_UNAVAILABLE_REASON is None and bool(_vision_ocr_base_url())
    )
    tesseract_possible = CONFIG["ocr_enabled"] and _TESSERACT_UNAVAILABLE_REASON is None
    return vision_possible or tesseract_possible


def _safe_ocr(func, *args) -> str:
    """Run an OCR helper, degrading to empty text instead of raising."""
    if not _ocr_is_available():
        return ""
    try:
        return func(*args)
    except Exception as exc:
        logger.debug("OCR_SKIPPED | %s: %s", type(exc).__name__, exc)
        return ""


def has_meaningful_text(text: Optional[str], minimum_chars: Optional[int] = None) -> bool:
    content = normalize_text(text)
    alnum_chars = sum(1 for ch in content if ch.isalnum())
    return alnum_chars >= (minimum_chars or CONFIG["ocr_min_text_chars"])


def parse_pdf_text(path: str) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception as e:
                logger.debug(f"Failed to extract text from PDF page: {e}")
        return "\n".join(pages)
    except Exception as e:
        logger.debug(f"Failed to read PDF file: {e}")
        return ""


def parse_pdf_bytes(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception as e:
                logger.debug(f"Failed to extract text from PDF page: {e}")
        return "\n".join(pages)
    except Exception as e:
        logger.debug(f"Failed to read PDF file: {e}")
        return ""


def parse_docx_text(path: str) -> str:
    from docx import Document

    try:
        doc = Document(path)
        chunks: List[str] = []
        chunks.extend(p.text for p in doc.paragraphs if p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if values:
                    chunks.append(" | ".join(values))
        return "\n".join(chunks)
    except Exception as e:
        logger.debug(f"Failed to read DOCX file: {e}")
        return ""


def parse_docx_bytes(data: bytes) -> str:
    from docx import Document

    try:
        doc = Document(io.BytesIO(data))
        chunks: List[str] = []
        chunks.extend(p.text for p in doc.paragraphs if p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if values:
                    chunks.append(" | ".join(values))
        return "\n".join(chunks)
    except Exception as e:
        logger.debug(f"Failed to read DOCX file: {e}")
        return ""


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _ocr_pil_image(image: Image.Image) -> str:
    """OCR one page image: Qwen vision model first, Tesseract as the fallback."""
    text = _qwen_vision_ocr(image)
    if normalize_text(text):
        return text
    return _tesseract_ocr(image)


def _ocr_image_file(path: str) -> str:
    with Image.open(path) as image:
        return _ocr_pil_image(image)


def _ocr_pdf_pages(path: str) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    segments: List[str] = []
    try:
        for page in pdf:
            bitmap = page.render(scale=2)
            image = bitmap.to_pil()
            text = _ocr_pil_image(image)
            if text.strip():
                segments.append(text)
    finally:
        pdf.close()
    return "\n".join(segments)


def _ocr_pdf_pages_bytes(data: bytes) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(io.BytesIO(data))
    segments: List[str] = []
    try:
        for page in pdf:
            bitmap = page.render(scale=2)
            image = bitmap.to_pil()
            text = _ocr_pil_image(image)
            if text.strip():
                segments.append(text)
    finally:
        pdf.close()
    return "\n".join(segments)


def _ocr_docx_embedded_images(path: str) -> str:
    from PIL import Image

    segments: List[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith("word/media/"):
                continue
            with archive.open(name) as handle:
                raw = handle.read()
            with Image.open(io.BytesIO(raw)) as image:
                text = _ocr_pil_image(image)
                if text.strip():
                    segments.append(text)
    return "\n".join(segments)


def _ocr_docx_embedded_images_bytes(data: bytes) -> str:
    segments: List[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if not name.startswith("word/media/"):
                continue
            with archive.open(name) as handle:
                raw = handle.read()
            with Image.open(io.BytesIO(raw)) as image:
                text = _ocr_pil_image(image)
                if text.strip():
                    segments.append(text)
    return "\n".join(segments)


def extract_text_layers_from_file(path: str) -> Dict[str, str]:
    ext = os.path.splitext(path.lower())[1]
    native_text = ""
    ocr_text = ""

    if ext == ".pdf":
        native_text = parse_pdf_text(path)
        if CONFIG["ocr_enabled"] and _needs_ocr(native_text):
            ocr_text = _safe_ocr(_ocr_pdf_pages, path)
    elif ext == ".docx":
        native_text = parse_docx_text(path)
        if CONFIG["ocr_enabled"] and _needs_ocr(native_text):
            ocr_text = _safe_ocr(_ocr_docx_embedded_images, path)
    elif ext in {".txt", ".rtf"}:
        native_text = _read_text_file(path)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        if CONFIG["ocr_enabled"]:
            ocr_text = _safe_ocr(_ocr_image_file, path)
    else:
        raise ValueError(f"Unsupported document type: {ext}")

    combined_parts = [segment for segment in (native_text, ocr_text) if normalize_text(segment)]
    return {
        "native_text": native_text,
        "ocr_text": ocr_text,
        "combined_text": "\n\n".join(combined_parts),
    }


def extract_text_layers_from_bytes(data: bytes, file_extension: str, ocr_enabled: bool | None = None) -> Dict[str, str]:
    """Extract text layers from file bytes without saving to disk."""
    ext = file_extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"

    ocr_enabled = ocr_enabled if ocr_enabled is not None else CONFIG["ocr_enabled"]
    native_text = ""
    ocr_text = ""

    if ext == ".pdf":
        native_text = parse_pdf_bytes(data)
        if ocr_enabled and _needs_ocr(native_text):
            ocr_text = _safe_ocr(_ocr_pdf_pages_bytes, data)
    elif ext == ".docx":
        native_text = parse_docx_bytes(data)
        if ocr_enabled and _needs_ocr(native_text):
            ocr_text = _safe_ocr(_ocr_docx_embedded_images_bytes, data)
    elif ext in {".txt", ".rtf"}:
        native_text = data.decode("utf-8", errors="ignore")
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        if ocr_enabled:
            ocr_text = _safe_ocr(lambda: _ocr_pil_image(Image.open(io.BytesIO(data))))
    else:
        raise ValueError(f"Unsupported document type: {ext}")

    combined_parts = [segment for segment in (native_text, ocr_text) if normalize_text(segment)]
    return {
        "native_text": native_text,
        "ocr_text": ocr_text,
        "combined_text": "\n\n".join(combined_parts),
    }


def _state_from_text(text: str) -> Optional[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return None

    match = re.search(r",\s*([A-Z]{2})(?:\b|$)", cleaned)
    if match:
        return match.group(1).upper()

    states = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "district of columbia": "DC",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
    }
    lowered = cleaned.lower()
    for state_name, code in states.items():
        if state_name in lowered:
            return code
    return None


def _first_json_object(payload: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(payload):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(payload[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError("No JSON object found", payload, 0)


def _extract_phone(text: str) -> Optional[str]:
    """Extract phone number, filtering out year ranges like 2020-2024."""
    patterns = [
        r"\+1?\s*\(?([0-9]{3})\)?\s*[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})",
        r"\+1?\s*([0-9]{3})\s+([0-9]{3})\s+([0-9]{4})",
        r"(?<!\d)\(?([0-9]{3})\)?\s*[-.]?\s*([0-9]{3})[-.]([0-9]{4})(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            phone = "".join(re.findall(r"\d", match.group(0)))
            if len(phone) >= 10:
                return normalize_text(match.group(0))
    return None


def extract_profile_fields_from_text(text: str) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    cleaned = text.replace("\r", "\n")

    def safe_get(pattern: str, flags: int = 0, group: int = 1) -> Optional[str]:
        match = re.search(pattern, cleaned, flags=flags)
        if not match:
            return None
        return normalize_text(match.group(group))

    email = safe_get(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE, group=0)
    if email:
        extracted["email"] = email

    phone = _extract_phone(cleaned)
    if phone:
        extracted["phone"] = phone

    name_match = re.search(r"(?im)^\s*([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,3})\s*$", cleaned)
    if name_match:
        full_name = name_match.group(1).strip()
        name_parts = full_name.split()
        if len(name_parts) >= 2:
            extracted["first_name"] = name_parts[0]
            extracted["last_name"] = name_parts[-1]

    location = safe_get(r"\b(?:Location|Address|Based in)\s*[:\-]?\s*([A-Za-z0-9 .,'#-]+(?:,\s*[A-Z]{2})?)", re.IGNORECASE)
    if location:
        extracted["location"] = location
        if "," in location:
            extracted["city"] = normalize_text(location.split(",", 1)[0])

    city = safe_get(r"\b(?:City)\s*[:\-]?\s*([A-Za-z .'-]+(?:,\s*[A-Z]{2})?)", re.IGNORECASE)
    if city:
        extracted["city"] = city.split(",", 1)[0].strip()
        location = extracted.get("location") or city
        extracted["location"] = location

    state_code = _state_from_text(extracted.get("location") or city or cleaned[:5000])
    if state_code:
        extracted["state_code"] = state_code
        extracted["state"] = state_code

    specialty = safe_get(
        r"\b(?:Specialty|Specialties|Focus|Practice|Profession Type|Profession)\s*[:\-]?\s*([A-Za-z0-9 /&.,()-]+)",
        re.IGNORECASE,
    )
    if specialty:
        extracted["specialty"] = specialty

    return extracted


def _run_openai_extraction(text: str) -> Dict[str, Any]:
    from openai import OpenAI

    api_key = CONFIG["openai_api_key"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key)
    prompt = (
        "Extract a structured JSON object from this resume text. "
        "Return only JSON with keys: first_name, last_name, city, state_code, specialty, phone, email. "
        "Use null for missing values and do not invent data.\n\n"
        f"TEXT:\n{text[:16000]}"
    )

    response = client.chat.completions.create(
        model=CONFIG["ai_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800,
    )

    content = response.choices[0].message.content or "{}"
    data = _first_json_object(content)
    return {k: v for k, v in data.items() if v not in (None, "", " ")}


def _run_openai_compatible_extraction(base_url: str, api_key: str, model: str, text: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url.rstrip("/") + "/", api_key=api_key, timeout=60.0)
    prompt = (
        "Extract all profile fields from this resume text. "
        "Return ONLY a JSON object with these keys: first_name, last_name, city, state_code, specialty, phone, email. "
        "SPECIALTY: Medical/professional specialty (e.g., Cardiology, Emergency Medicine, Pediatrics, etc.). "
        "STATE_CODE: 2-letter state abbreviation if mentioned. "
        "PHONE: Contact phone number. "
        "EMAIL: Contact email. "
        "Set any missing field to null. Do not invent data or add markdown formatting.\n\n"
        f"RESUME TEXT:\n{text[:16000]}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract data from resumes. Return only valid JSON with no markdown formatting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=CONFIG["local_model_max_tokens"],
        )
    except Exception as exc:
        logger.warning(f"OpenAI-compatible API error: {exc}")
        return {}

    content = response.choices[0].message.content or "{}"
    try:
        data = _first_json_object(content)
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.items() if v not in (None, "", " ")}


def _load_local_qwen():
    from llama_cpp import Llama

    model_path = CONFIG["local_model_path"]
    if model_path and os.path.exists(model_path):
        return Llama(
            model_path=model_path,
            n_ctx=CONFIG["local_model_n_ctx"],
            verbose=False,
        )

    if not CONFIG["local_model_enabled"]:
        raise ValueError("LOCAL_MODEL_ENABLED is false and LOCAL_MODEL_PATH is unavailable")

    return Llama.from_pretrained(
        repo_id=CONFIG["local_model_repo_id"],
        filename=CONFIG["local_model_filename"],
        local_dir=CONFIG["local_model_dir"],
        n_ctx=CONFIG["local_model_n_ctx"],
        verbose=False,
    )


def _run_local_qwen_extraction(text: str) -> Dict[str, Any]:
    if CONFIG["local_model_base_url"]:
        return _run_openai_compatible_extraction(
            base_url=CONFIG["local_model_base_url"],
            api_key=CONFIG["local_model_api_key"],
            model=CONFIG["local_model_name"],
            text=text,
        )

    try:
        llm = _load_local_qwen()
    except Exception as exc:
        logger.warning(f"Failed to load local Qwen model: {exc}")
        return {}

    prompt = (
        "Extract all profile fields from this resume text. "
        "Return ONLY a JSON object with these keys: first_name, last_name, city, state_code, specialty, phone, email. "
        "SPECIALTY: Medical/professional specialty (e.g., Cardiology, Emergency Medicine, Pediatrics, etc.). "
        "STATE_CODE: 2-letter state abbreviation if mentioned. "
        "PHONE: Contact phone number. "
        "EMAIL: Contact email. "
        "Set any missing field to null. Do not invent data or add markdown formatting.\n\n"
        f"RESUME TEXT:\n{text[:16000]}"
    )

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Extract data from resumes. Return only valid JSON with no markdown formatting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=CONFIG["local_model_max_tokens"],
        )
    except Exception as exc:
        logger.warning(f"Local Qwen extraction error: {exc}")
        return {}

    content = response.get("choices", [{}])[0].get("message", {}).get("content") or "{}"
    try:
        data = _first_json_object(content)
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.items() if v not in (None, "", " ")}


def run_ai_extraction(document_path: str, text: str) -> Dict[str, Any]:
    del document_path
    if not CONFIG["ai_extractor_enabled"] or not text.strip():
        return {}

    provider = (CONFIG["ai_provider"] or "").strip().lower()
    if provider == "openai":
        return _run_openai_extraction(text)
    if provider in {"local_qwen", "qwen", "llama_cpp"}:
        return _run_local_qwen_extraction(text)
    raise ValueError(f"Unsupported AI_PROVIDER: {CONFIG['ai_provider']}")


def combine_extracted_values(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if value in (None, "", " "):
            continue
        if key not in merged or not merged[key] or not str(merged[key]).strip():
            merged[key] = value
    return merged
