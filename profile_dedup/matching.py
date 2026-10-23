"""Cluster candidate rows (already pre-filtered to a shared-phone group or a
shared NPI/email group) into duplicate clusters with a confidence level.

Confidence levels
------------------
CERTAIN - matching NPI number or matching email (both are DB-unique when
          non-empty, so this only fires if that constraint is ever relaxed,
          e.g. during a bulk import). Eligible for --apply.
HIGH    - same normalized phone number (already filtered to small,
          non-shared-line groups) AND exactly matching normalized name.
          Eligible for --apply.
MEDIUM  - same normalized phone number AND a *similar but not identical*
          name (typo, nickname, missing middle name, etc). Never
          auto-applied - always flagged for manual review, matching the
          requirement to never merge on name alone and to flag uncertain
          matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from profile_dedup.config import DEDUP_CONFIG
from profile_dedup.normalize import name_similarity, normalize_name

CERTAIN = "CERTAIN"
HIGH = "HIGH"
MEDIUM = "MEDIUM"


@dataclass
class Cluster:
    confidence: str
    signal: str  # "npi", "email", "phone+exact_name", "phone+similar_name"
    profile_ids: List[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    name_similarity_score: float | None = None


def cluster_identifier_group(rows: List[Dict[str, Any]], signal: str) -> List[Cluster]:
    """NPI / email groups: every row with the same non-empty value is one
    cluster. Identifiers are unique identity signals - no further
    corroboration is required, unlike phone."""
    if len(rows) < 2:
        return []
    return [Cluster(CERTAIN, signal, [r["profile_id"] for r in rows], rows)]


def _distinct_npis_conflict(rows: List[Dict[str, Any]]) -> bool:
    """True when two or more rows carry a different non-empty NPI number.

    NPI is a verified, government-unique identifier - if it disagrees across
    rows that otherwise look alike (same phone, same name), they are
    provably different entities (e.g. two facilities sharing a switchboard
    number and a generic org name), not a duplicate.
    """
    npis = {row.get("npi_number") for row in rows if row.get("npi_number")}
    return len(npis) > 1


def cluster_phone_group(rows: List[Dict[str, Any]]) -> List[Cluster]:
    """Split one shared-phone group into sub-clusters by name agreement."""
    ratio_threshold = DEDUP_CONFIG["name_fuzzy_ratio"]
    remaining = list(rows)
    clusters: List[Cluster] = []

    # Exact-name sub-clusters first (HIGH confidence).
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for row in remaining:
        key = normalize_name(row.get("first_name"), row.get("last_name"))
        by_name.setdefault(key, []).append(row)

    exact_matched_ids = set()
    for name_key, group in by_name.items():
        if not name_key or len(group) < 2:
            continue
        if _distinct_npis_conflict(group):
            # Provably different entities despite matching name+phone -
            # never cluster, regardless of confidence level.
            continue
        clusters.append(Cluster(HIGH, "phone+exact_name", [r["profile_id"] for r in group], group))
        exact_matched_ids.update(r["profile_id"] for r in group)

    leftover = [r for r in remaining if r["profile_id"] not in exact_matched_ids]

    # Pairwise fuzzy-name matches among what's left (MEDIUM confidence,
    # manual review only). Greedy pairing - groups are tiny (<= max_phone_group_size).
    used = set()
    for i, row_a in enumerate(leftover):
        if row_a["profile_id"] in used:
            continue
        name_a = normalize_name(row_a.get("first_name"), row_a.get("last_name"))
        if not name_a:
            continue
        for row_b in leftover[i + 1 :]:
            if row_b["profile_id"] in used:
                continue
            name_b = normalize_name(row_b.get("first_name"), row_b.get("last_name"))
            if not name_b:
                continue
            ratio = name_similarity(name_a, name_b)
            if ratio >= ratio_threshold and not _distinct_npis_conflict([row_a, row_b]):
                clusters.append(
                    Cluster(
                        MEDIUM,
                        "phone+similar_name",
                        [row_a["profile_id"], row_b["profile_id"]],
                        [row_a, row_b],
                        name_similarity_score=ratio,
                    )
                )
                used.add(row_a["profile_id"])
                used.add(row_b["profile_id"])
                break

    return clusters
