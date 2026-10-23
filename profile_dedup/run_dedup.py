"""CLI entry point for profile deduplication.

Dry run (default, makes no DB changes):
    python -m profile_dedup.run_dedup

Apply (writes merges/soft-deletes inside transactions, after review):
    python -m profile_dedup.run_dedup --apply

See profile_dedup/README.md for the full design and safety notes.
"""
from __future__ import annotations

import argparse
import logging
import uuid
from typing import Any, Dict, List

from profile_dedup import db
from profile_dedup.config import DEDUP_CONFIG
from profile_dedup.matching import Cluster, cluster_identifier_group, cluster_phone_group
from profile_dedup.merge_logic import build_merge_plan
from profile_dedup.primary import select_primary
from profile_dedup.qwen_review import get_ai_opinion
from profile_dedup.reporting import DedupReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("profile_dedup")


def _process_cluster(
    conn,
    cluster: Cluster,
    report: DedupReport,
    apply_mode: bool,
    apply_medium: bool,
) -> None:
    # Re-resolve every member to its current primary in case an earlier
    # cluster in this same run (or a previous run) already merged one of
    # them elsewhere. Idempotency guard.
    resolved_ids = {db.resolve_to_current_primary(conn, pid) for pid in cluster.profile_ids}
    if len(resolved_ids) < 2:
        report.add_skipped(
            {"profile_ids": cluster.profile_ids, "reason": "already resolved to a single profile"}
        )
        return

    if cluster.confidence == "MEDIUM" and not apply_medium:
        # Always report MEDIUM for manual review; never auto-act on it.
        rows = cluster.rows
        ai_opinion = None
        if len(rows) == 2 and DEDUP_CONFIG["use_qwen_review"]:
            ai_opinion = get_ai_opinion(rows[0], rows[1])
        primary = select_primary(rows)
        others = [r for r in rows if r["profile_id"] != primary["profile_id"]]
        plan = build_merge_plan(primary, others)
        for dup in others:
            report.add_pair(
                primary_profile_id=primary["profile_id"],
                duplicate_profile_id=dup["profile_id"],
                confidence=cluster.confidence,
                signal=cluster.signal,
                decisions=plan["decisions_by_duplicate"][dup["profile_id"]],
                conflicts=[c for c in plan["conflicts"] if c["duplicate_profile_id"] == dup["profile_id"]],
                action="manual_review_only",
                ai_opinion=ai_opinion,
            )
        return

    if apply_mode:
        with conn:
            fresh = db.lock_profiles_for_update(conn, list(resolved_ids))
            live_ids = [pid for pid in resolved_ids if fresh.get(pid) and not fresh[pid].get("merged_into")]
            if len(live_ids) < 2:
                report.add_skipped({"profile_ids": list(resolved_ids), "reason": "race: already merged"})
                return

            rows = [fresh[pid] for pid in live_ids]
            primary = select_primary(rows)
            duplicates = [r for r in rows if r["profile_id"] != primary["profile_id"]]
            plan = build_merge_plan(primary, duplicates)

            db.apply_merged_values(conn, primary["profile_id"], plan["merged_values"])

            for dup in duplicates:
                fk_counts = db.count_fk_references(conn, dup["profile_id"])
                has_references = any(count > 0 for count in fk_counts.values())
                action = "soft_merge"
                if not has_references:
                    db.delete_duplicate(conn, dup["profile_id"])
                    action = "deleted"
                else:
                    db.soft_merge_duplicate(conn, dup["profile_id"], primary["profile_id"])

                db.insert_audit_record(
                    conn,
                    report.run_id,
                    {
                        "primary_profile_id": primary["profile_id"],
                        "duplicate_profile_id": dup["profile_id"],
                        "match_confidence": cluster.confidence,
                        "match_signal": cluster.signal,
                        "original_primary_data": primary,
                        "original_duplicate_data": dup,
                        "merged_field_values": plan["merged_values"],
                        "field_sources": plan["field_sources"],
                        "conflicts": [c for c in plan["conflicts"] if c["duplicate_profile_id"] == dup["profile_id"]],
                        "action": action,
                    },
                )
                report.add_pair(
                    primary_profile_id=primary["profile_id"],
                    duplicate_profile_id=dup["profile_id"],
                    confidence=cluster.confidence,
                    signal=cluster.signal,
                    decisions=plan["decisions_by_duplicate"][dup["profile_id"]],
                    conflicts=[c for c in plan["conflicts"] if c["duplicate_profile_id"] == dup["profile_id"]],
                    action=action,
                )
        # `with conn:` commits on clean exit, rolls back on exception.
    else:
        rows = cluster.rows
        primary = select_primary(rows)
        duplicates = [r for r in rows if r["profile_id"] != primary["profile_id"]]
        plan = build_merge_plan(primary, duplicates)
        for dup in duplicates:
            report.add_pair(
                primary_profile_id=primary["profile_id"],
                duplicate_profile_id=dup["profile_id"],
                confidence=cluster.confidence,
                signal=cluster.signal,
                decisions=plan["decisions_by_duplicate"][dup["profile_id"]],
                conflicts=[c for c in plan["conflicts"] if c["duplicate_profile_id"] == dup["profile_id"]],
                action="would_soft_merge_or_delete",
            )


def run(apply_mode: bool, apply_medium: bool, limit_groups: int | None) -> Dict[str, str]:
    run_id = str(uuid.uuid4())
    report = DedupReport(run_id, apply_mode)
    logger.info(f"Starting profile dedup run {run_id} (apply={apply_mode})")

    with db.get_connection() as conn:
        if apply_mode:
            db.ensure_audit_table(conn)

        clusters: List[Cluster] = []

        for group in db.fetch_npi_duplicate_groups(limit_groups=limit_groups):
            clusters.extend(cluster_identifier_group(group, "npi"))
        for group in db.fetch_email_duplicate_groups(limit_groups=limit_groups):
            clusters.extend(cluster_identifier_group(group, "email"))

        phone_groups, _ = db.fetch_phone_duplicate_groups(limit_groups=limit_groups)
        for group in phone_groups:
            clusters.extend(cluster_phone_group(group))

        logger.info(f"Found {len(clusters)} candidate clusters")

        for cluster in clusters:
            try:
                _process_cluster(conn, cluster, report, apply_mode, apply_medium)
            except Exception as exc:
                logger.exception(f"Cluster failed, rolled back: {cluster.profile_ids}")
                report.add_error({"profile_ids": cluster.profile_ids, "error": str(exc)})

    paths = report.write()
    logger.info(f"Run complete. Summary: {report.summary}")
    logger.info(f"Reports written: {paths}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe profile deduplication for the NeonDB profiles table")
    parser.add_argument("--apply", action="store_true", help="Execute merges/deletes. Default is dry-run.")
    parser.add_argument(
        "--apply-medium",
        action="store_true",
        help="Also apply MEDIUM confidence (similar-but-not-identical name) clusters. "
        "Strongly discouraged - review the dry-run report for these by hand instead.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Cap on candidate groups fetched per identifier type, for testing on a sample.",
    )
    args = parser.parse_args()

    if args.apply:
        confirm = input(
            "This will modify the live profiles table. Type 'APPLY' to continue: "
        ).strip()
        if confirm != "APPLY":
            logger.info("Aborted - confirmation text did not match.")
            return

    run(apply_mode=args.apply, apply_medium=args.apply_medium, limit_groups=args.limit_groups)


if __name__ == "__main__":
    main()
