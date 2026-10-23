"""Optional local-Qwen second opinion on MEDIUM confidence clusters.

This never drives a merge or delete decision - MEDIUM confidence clusters
are always written to the dry-run report for manual review regardless of
what Qwen says. It only adds an annotation to help the human reviewer,
reusing the same local OpenAI-compatible Qwen server the rest of this repo
talks to (profile_enrichment/extractors.py).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from profile_dedup.config import DEDUP_CONFIG
from profile_enrichment.config import CONFIG as ENRICHMENT_CONFIG

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You help a human reviewer decide whether two healthcare-provider profile "
    "records describe the same real person. You never have the final say - you "
    "only give an opinion and a short reason. Respond with strict JSON only: "
    '{"same_person": true|false, "confidence": "low"|"medium"|"high", "reason": "..."}'
)


def _profile_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": f"{row.get('first_name', '')} {row.get('last_name', '')}".strip(),
        "phone": row.get("phone"),
        "city": row.get("city"),
        "state_code": row.get("state_code"),
        "specialty": row.get("specialty"),
        "organization_name": row.get("organization_name"),
    }


def get_ai_opinion(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not DEDUP_CONFIG["use_qwen_review"]:
        return None
    if not ENRICHMENT_CONFIG.get("local_model_enabled"):
        return None

    try:
        from profile_enrichment.extractors import _get_openai_client  # reuse existing client builder
    except Exception as exc:  # pragma: no cover - import-time environment issue
        logger.warning(f"Qwen review unavailable (import failed): {exc}")
        return None

    base_url = ENRICHMENT_CONFIG.get("local_model_base_url")
    if not base_url:
        return None

    try:
        client = _get_openai_client(
            base_url=base_url,
            api_key=ENRICHMENT_CONFIG.get("local_model_api_key") or "local-qwen",
            timeout=DEDUP_CONFIG["qwen_timeout"],
        )
        prompt = (
            "Profile A: " + json.dumps(_profile_summary(row_a)) + "\n"
            "Profile B: " + json.dumps(_profile_summary(row_b)) + "\n"
            "They share a phone number but their names are not identical. "
            "Are they the same real person?"
        )
        response = client.chat.completions.create(
            model=ENRICHMENT_CONFIG.get("local_model_name"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content or "{}"
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(content[start : end + 1])
    except Exception as exc:
        logger.warning(f"Qwen review call failed, continuing without it: {exc}")
        return None
