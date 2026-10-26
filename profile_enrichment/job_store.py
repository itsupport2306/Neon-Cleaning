"""Durable run manifest, task queue, caches and audit trail in SQLite (WAL).

Why SQLite: the run is a single-machine job (one model server, one laptop/workstation). It gives atomic
claims, crash-safe incremental commits (no more rewriting a 150k-entry JSON file), and a queryable ledger.
Several worker *processes* on the same machine can share one file safely. Running workers on several
*servers* needs a shared database job table instead (same claim/lease protocol, different driver) - this
module does not pretend to solve that over a network filesystem.

Task life cycle::

    pending --claim--> claimed --prepare--> prepared --apply--> done(outcome)
                         |                                    ^
                         +----------------finish--------------+
                         +--lease expires--> claimed by someone else (attempts + 1)

Every finish/prepare is fenced by ``lease_owner``: a worker that lost its lease (slow model, suspended
laptop) cannot overwrite the result of the worker that re-claimed the task.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

PENDING = "pending"
VERIFIED_UPDATED = "verified_updated"
NO_RESUME = "no_resume"
DOWNLOAD_FAILED = "download_failed"
EXTRACTION_FAILED = "extraction_failed"
INFERENCE_FAILED = "inference_failed"
VALIDATION_FAILED = "validation_failed"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CONCURRENT_CHANGE = "concurrent_change"
UPDATE_FAILED = "update_failed"

OUTCOMES = (
    VERIFIED_UPDATED, NO_RESUME, DOWNLOAD_FAILED, EXTRACTION_FAILED, INFERENCE_FAILED,
    VALIDATION_FAILED, INSUFFICIENT_EVIDENCE, CONCURRENT_CHANGE, UPDATE_FAILED, PENDING,
)
# Failures count against the exit status; unresolved ("we looked and the resume does not establish it") do not.
FAILURE_OUTCOMES = (DOWNLOAD_FAILED, EXTRACTION_FAILED, INFERENCE_FAILED, UPDATE_FAILED, CONCURRENT_CHANGE)
UNRESOLVED_OUTCOMES = (INSUFFICIENT_EVIDENCE, VALIDATION_FAILED, NO_RESUME)

STATE_PENDING, STATE_CLAIMED, STATE_PREPARED, STATE_DONE = "pending", "claimed", "prepared", "done"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, created_at REAL, selection_sql TEXT, params_json TEXT, mode TEXT,
  db_count_at_start INTEGER, manifest_size INTEGER DEFAULT 0, manifest_complete INTEGER DEFAULT 0,
  last_manifest_id TEXT DEFAULT '', model_fp TEXT, prompt_version TEXT, notes TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS tasks(
  run_id TEXT NOT NULL, profile_id TEXT NOT NULL, resume_ref TEXT,
  state TEXT NOT NULL DEFAULT 'pending', outcome TEXT NOT NULL DEFAULT 'pending',
  reason TEXT DEFAULT '', stage TEXT DEFAULT '', attempts INTEGER DEFAULT 0,
  lease_owner TEXT, lease_expires REAL, invocation TEXT,
  doc_sha256 TEXT, model_fp TEXT, prompt_version TEXT, qwen_status TEXT,
  proposal_json TEXT, fields_written TEXT DEFAULT '', screen_reason_set INTEGER DEFAULT 0,
  created_at REAL, updated_at REAL, completed_at REAL,
  PRIMARY KEY(run_id, profile_id));
CREATE INDEX IF NOT EXISTS tasks_claim ON tasks(run_id, state, lease_expires);
CREATE INDEX IF NOT EXISTS tasks_outcome ON tasks(run_id, outcome);
CREATE TABLE IF NOT EXISTS stage_timings(
  run_id TEXT, profile_id TEXT, stage TEXT, seconds REAL, at REAL);
CREATE INDEX IF NOT EXISTS timings_stage ON stage_timings(run_id, stage);
CREATE TABLE IF NOT EXISTS doc_cache(
  sha256 TEXT, extractor_version TEXT, ext TEXT, pages_json TEXT, created_at REAL,
  PRIMARY KEY(sha256, extractor_version));
CREATE TABLE IF NOT EXISTS url_cache(url TEXT PRIMARY KEY, sha256 TEXT, size INTEGER, fetched_at REAL);
CREATE TABLE IF NOT EXISTS inference_cache(key TEXT PRIMARY KEY, result_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS writes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, profile_id TEXT, at REAL, field TEXT,
  old_value TEXT, new_value TEXT, evidence TEXT, page INTEGER, source TEXT);
CREATE TABLE IF NOT EXISTS proposals(
  run_id TEXT, profile_id TEXT, field TEXT, current_value TEXT, proposed_value TEXT,
  evidence TEXT, page INTEGER, source TEXT, status TEXT DEFAULT 'open',
  PRIMARY KEY(run_id, profile_id, field));
"""


@dataclass
class Task:
    run_id: str
    profile_id: str
    resume_ref: Optional[str]
    state: str
    outcome: str
    reason: str
    attempts: int
    lease_owner: Optional[str]
    proposal: Optional[Dict[str, Any]]
    doc_sha256: Optional[str]


class LeaseLost(Exception):
    """The task was re-claimed by another worker; this worker's result must be discarded."""


class JobStore:
    def __init__(self, path: str | Path | None = None):
        from profile_enrichment.config import CONFIG

        self.path = Path(path or CONFIG["job_store_path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)

    # -- plumbing ------------------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- runs ----------------------------------------------------------------------------------
    def create_run(self, selection_sql: str, params: Dict[str, Any], mode: str, db_count: Optional[int],
                   model_fp: str, prompt_version: str, run_id: str | None = None) -> str:
        run_id = run_id or time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs(run_id, created_at, selection_sql, params_json, mode, db_count_at_start, model_fp, prompt_version)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (run_id, time.time(), selection_sql, json.dumps(params, sort_keys=True), mode, db_count, model_fp, prompt_version),
            )
        return run_id

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        return self._conn().execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def latest_run(self) -> Optional[sqlite3.Row]:
        return self._conn().execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()

    def list_runs(self) -> List[sqlite3.Row]:
        return self._conn().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()

    def add_tasks(self, run_id: str, rows: Sequence[Tuple[str, Optional[str]]], last_id: str | None = None) -> int:
        """Insert manifest rows (idempotent) and remember how far the manifest got, so an interrupted
        manifest build resumes from there instead of starting over."""
        now = time.time()
        with self._tx() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO tasks(run_id, profile_id, resume_ref, created_at, updated_at) VALUES(?,?,?,?,?)",
                [(run_id, pid, ref, now, now) for pid, ref in rows],
            )
            inserted = conn.total_changes - before
            conn.execute(
                "UPDATE runs SET manifest_size=(SELECT count(*) FROM tasks WHERE run_id=?), last_manifest_id=COALESCE(?, last_manifest_id) WHERE run_id=?",
                (run_id, last_id, run_id),
            )
        return inserted

    def finish_manifest(self, run_id: str) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE runs SET manifest_complete=1 WHERE run_id=?", (run_id,))

    # -- claiming ------------------------------------------------------------------------------
    def claim(self, run_id: str, owner: str, lease_seconds: float, limit: int = 1,
              states: Sequence[str] = (STATE_PENDING,), profile_ids: Sequence[str] | None = None) -> List[Task]:
        """Atomically lease up to `limit` tasks: pending ones, plus ones whose previous lease expired."""
        now = time.time()
        marks = ",".join("?" * len(states))
        id_filter = ""
        params: List[Any] = [run_id, *states, now]
        if profile_ids:
            id_filter = f" AND profile_id IN ({','.join('?' * len(profile_ids))})"
            params.extend(profile_ids)
        params.append(limit)
        with self._tx() as conn:
            rows = conn.execute(
                f"SELECT profile_id FROM tasks WHERE run_id=? AND (state IN ({marks}) OR (state='claimed' AND lease_expires<?)){id_filter} "
                "ORDER BY profile_id LIMIT ?", params,
            ).fetchall()
            ids = [r["profile_id"] for r in rows]
            if not ids:
                return []
            conn.execute(
                f"UPDATE tasks SET state='claimed', lease_owner=?, lease_expires=?, attempts=attempts+1, updated_at=? "
                f"WHERE run_id=? AND profile_id IN ({','.join('?' * len(ids))})",
                [owner, now + lease_seconds, now, run_id, *ids],
            )
            claimed = conn.execute(
                f"SELECT * FROM tasks WHERE run_id=? AND profile_id IN ({','.join('?' * len(ids))}) ORDER BY profile_id",
                [run_id, *ids],
            ).fetchall()
        return [self._task(r) for r in claimed]

    @staticmethod
    def _task(row: sqlite3.Row) -> Task:
        return Task(
            run_id=row["run_id"], profile_id=row["profile_id"], resume_ref=row["resume_ref"], state=row["state"],
            outcome=row["outcome"], reason=row["reason"] or "", attempts=row["attempts"], lease_owner=row["lease_owner"],
            proposal=json.loads(row["proposal_json"]) if row["proposal_json"] else None, doc_sha256=row["doc_sha256"],
        )

    def heartbeat(self, run_id: str, profile_id: str, owner: str, lease_seconds: float, stage: str = "") -> None:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE tasks SET lease_expires=?, stage=COALESCE(NULLIF(?, ''), stage), updated_at=? "
                "WHERE run_id=? AND profile_id=? AND lease_owner=? AND state='claimed'",
                (time.time() + lease_seconds, stage, time.time(), run_id, profile_id, owner),
            )
            if cur.rowcount != 1:
                raise LeaseLost(profile_id)

    def release(self, run_id: str, profile_id: str, owner: str) -> None:
        """Give a claimed-but-unstarted task back (graceful shutdown); the attempt is not counted."""
        with self._tx() as conn:
            conn.execute(
                "UPDATE tasks SET state='pending', lease_owner=NULL, lease_expires=NULL, attempts=MAX(0, attempts-1) "
                "WHERE run_id=? AND profile_id=? AND lease_owner=? AND state='claimed'",
                (run_id, profile_id, owner),
            )

    def release_owner(self, run_id: str, owner: str) -> int:
        """Return every task this worker still holds (interrupt / shutdown); the attempts are not counted."""
        with self._tx() as conn:
            return conn.execute(
                "UPDATE tasks SET state='pending', lease_owner=NULL, lease_expires=NULL, attempts=MAX(0, attempts-1) "
                "WHERE run_id=? AND lease_owner=? AND state='claimed'", (run_id, owner)).rowcount

    def _fenced(self, conn, run_id: str, profile_id: str, owner: str, assignments: str, params: Sequence[Any]) -> None:
        cur = conn.execute(
            f"UPDATE tasks SET {assignments}, updated_at=? WHERE run_id=? AND profile_id=? AND lease_owner=? AND state='claimed'",
            [*params, time.time(), run_id, profile_id, owner],
        )
        if cur.rowcount != 1:
            raise LeaseLost(profile_id)

    def finish(self, run_id: str, profile_id: str, owner: str, outcome: str, reason: str = "", *, invocation: str = "",
               doc_sha256: str | None = None, model_fp: str | None = None, prompt_version: str | None = None,
               qwen_status: str | None = None, fields_written: Sequence[str] = (), screen_reason_set: bool = False,
               proposal: Dict[str, Any] | None = None) -> None:
        assert outcome in OUTCOMES, outcome
        with self._tx() as conn:
            self._fenced(
                conn, run_id, profile_id, owner,
                "state='done', outcome=?, reason=?, invocation=?, doc_sha256=COALESCE(?, doc_sha256), model_fp=COALESCE(?, model_fp), "
                "prompt_version=COALESCE(?, prompt_version), qwen_status=COALESCE(?, qwen_status), fields_written=?, "
                "screen_reason_set=?, proposal_json=COALESCE(?, proposal_json), lease_owner=NULL, lease_expires=NULL, completed_at=?",
                [outcome, reason[:500], invocation, doc_sha256, model_fp, prompt_version, qwen_status,
                 ",".join(fields_written), int(screen_reason_set), json.dumps(proposal) if proposal is not None else None, time.time()],
            )

    def mark_prepared(self, run_id: str, profile_id: str, owner: str, proposal: Dict[str, Any], *, invocation: str = "",
                      doc_sha256: str | None = None, model_fp: str | None = None, prompt_version: str | None = None,
                      qwen_status: str | None = None) -> None:
        """Proposal computed and validated but not written (dry run, or waiting for --apply)."""
        with self._tx() as conn:
            self._fenced(
                conn, run_id, profile_id, owner,
                "state='prepared', outcome='pending', reason='prepared', invocation=?, proposal_json=?, doc_sha256=COALESCE(?, doc_sha256), "
                "model_fp=COALESCE(?, model_fp), prompt_version=COALESCE(?, prompt_version), qwen_status=COALESCE(?, qwen_status), "
                "lease_owner=NULL, lease_expires=NULL",
                [invocation, json.dumps(proposal), doc_sha256, model_fp, prompt_version, qwen_status],
            )

    # -- retries -------------------------------------------------------------------------------
    def requeue(self, run_id: str, outcomes: Sequence[str], *, stale_only: bool = False,
                model_fp: str | None = None, prompt_version: str | None = None) -> int:
        """Return finished tasks to `pending`. With `stale_only`, only those decided under another model/prompt."""
        if not outcomes:
            return 0
        marks = ",".join("?" * len(outcomes))
        sql = (f"UPDATE tasks SET state='pending', outcome='pending', reason='requeued', lease_owner=NULL, lease_expires=NULL, "
               f"proposal_json=NULL, updated_at=? WHERE run_id=? AND state='done' AND outcome IN ({marks})")
        params: List[Any] = [time.time(), run_id, *outcomes]
        if stale_only:
            sql += " AND (COALESCE(model_fp,'')<>? OR COALESCE(prompt_version,'')<>?)"
            params.extend([model_fp or "", prompt_version or ""])
        with self._tx() as conn:
            return conn.execute(sql, params).rowcount

    def requeue_prepared(self, run_id: str) -> int:
        with self._tx() as conn:
            return conn.execute(
                "UPDATE tasks SET state='pending', outcome='pending', proposal_json=NULL WHERE run_id=? AND state='prepared'",
                (run_id,),
            ).rowcount

    def reclaim_expired(self, run_id: str) -> int:
        with self._tx() as conn:
            return conn.execute(
                "UPDATE tasks SET state='pending', lease_owner=NULL, lease_expires=NULL WHERE run_id=? AND state='claimed' AND lease_expires<?",
                (run_id, time.time()),
            ).rowcount

    # -- reporting -----------------------------------------------------------------------------
    def outcome_counts(self, run_id: str, invocation: str | None = None) -> Dict[str, int]:
        sql = "SELECT outcome, state, count(*) n FROM tasks WHERE run_id=?"
        params: List[Any] = [run_id]
        if invocation is not None:
            sql += " AND invocation=?"
            params.append(invocation)
        sql += " GROUP BY outcome, state"
        counts = {o: 0 for o in OUTCOMES}
        counts["prepared"] = 0
        counts["claimed"] = 0
        for row in self._conn().execute(sql, params):
            if row["state"] == STATE_PREPARED:
                counts["prepared"] += row["n"]
            elif row["state"] == STATE_CLAIMED:
                counts["claimed"] += row["n"]
            else:
                counts[row["outcome"]] += row["n"]
        return counts

    def total_tasks(self, run_id: str) -> int:
        return self._conn().execute("SELECT count(*) FROM tasks WHERE run_id=?", (run_id,)).fetchone()[0]

    def tasks_with_outcome(self, run_id: str, outcomes: Sequence[str]) -> List[sqlite3.Row]:
        marks = ",".join("?" * len(outcomes))
        return self._conn().execute(
            f"SELECT * FROM tasks WHERE run_id=? AND state='done' AND outcome IN ({marks}) ORDER BY profile_id", [run_id, *outcomes]
        ).fetchall()

    def reason_counts(self, run_id: str) -> List[Tuple[str, str, int]]:
        rows = self._conn().execute(
            "SELECT outcome, substr(reason,1,80) r, count(*) n FROM tasks WHERE run_id=? AND state='done' AND outcome<>'verified_updated' "
            "GROUP BY outcome, r ORDER BY n DESC", (run_id,)).fetchall()
        return [(r["outcome"], r["r"], r["n"]) for r in rows]

    def field_write_counts(self, run_id: str) -> Dict[str, int]:
        rows = self._conn().execute("SELECT field, count(*) n FROM writes WHERE run_id=? GROUP BY field", (run_id,)).fetchall()
        return {r["field"]: r["n"] for r in rows}

    def prepared_field_counts(self, run_id: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self._conn().execute("SELECT proposal_json FROM tasks WHERE run_id=? AND state='prepared'", (run_id,)):
            for name in (json.loads(row["proposal_json"]).get("fields") or {}):
                counts[name] = counts.get(name, 0) + 1
        return counts

    def record_timing(self, run_id: str, profile_id: str, stage: str, seconds: float) -> None:
        self._conn().execute("INSERT INTO stage_timings VALUES(?,?,?,?,?)", (run_id, profile_id, stage, seconds, time.time()))

    def timing_summary(self, run_id: str) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        stages = [r[0] for r in self._conn().execute("SELECT DISTINCT stage FROM stage_timings WHERE run_id=?", (run_id,))]
        for stage in stages:
            values = sorted(r[0] for r in self._conn().execute(
                "SELECT seconds FROM stage_timings WHERE run_id=? AND stage=?", (run_id, stage)))
            n = len(values)
            summary[stage] = {"count": n, "total_s": round(sum(values), 2), "mean_s": round(sum(values) / n, 3),
                              "p50_s": round(values[n // 2], 3), "p95_s": round(values[min(n - 1, int(n * 0.95))], 3)}
        return summary

    def throughput(self, run_id: str, invocation: str | None = None) -> Dict[str, float]:
        sql = "SELECT count(*) n, min(completed_at) a, max(completed_at) b FROM tasks WHERE run_id=? AND completed_at IS NOT NULL"
        params: List[Any] = [run_id]
        if invocation is not None:
            sql += " AND invocation=?"
            params.append(invocation)
        row = self._conn().execute(sql, params).fetchone()
        n, a, b = row["n"], row["a"], row["b"]
        span = (b - a) if a and b and b > a else 0.0
        return {"tasks": n, "span_s": round(span, 1), "per_minute": round(n / span * 60, 2) if span else 0.0}

    # -- caches --------------------------------------------------------------------------------
    def get_doc(self, sha256: str, extractor_version: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        row = self._conn().execute("SELECT ext, pages_json FROM doc_cache WHERE sha256=? AND extractor_version=?",
                                   (sha256, extractor_version)).fetchone()
        return (row["ext"], json.loads(row["pages_json"])) if row else None

    def put_doc(self, sha256: str, extractor_version: str, ext: str, pages: List[Dict[str, Any]]) -> None:
        self._conn().execute("INSERT OR REPLACE INTO doc_cache VALUES(?,?,?,?,?)",
                             (sha256, extractor_version, ext, json.dumps(pages), time.time()))

    def get_inference(self, key: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute("SELECT result_json FROM inference_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def put_inference(self, key: str, result: Dict[str, Any]) -> None:
        self._conn().execute("INSERT OR REPLACE INTO inference_cache VALUES(?,?,?)", (key, json.dumps(result), time.time()))

    def get_url(self, url: str) -> Optional[sqlite3.Row]:
        return self._conn().execute("SELECT * FROM url_cache WHERE url=?", (url,)).fetchone()

    def put_url(self, url: str, sha256: str, size: int) -> None:
        self._conn().execute("INSERT OR REPLACE INTO url_cache VALUES(?,?,?,?)", (url, sha256, size, time.time()))

    # -- audit + review queue ------------------------------------------------------------------
    def record_write(self, run_id: str, profile_id: str, field: str, old: Any, new: Any, evidence: str, page: int, source: str) -> None:
        self._conn().execute(
            "INSERT INTO writes(run_id, profile_id, at, field, old_value, new_value, evidence, page, source) VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, profile_id, time.time(), field, None if old is None else str(old), str(new), evidence[:300], page, source))

    def writes_for_run(self, run_id: str) -> List[sqlite3.Row]:
        return self._conn().execute("SELECT * FROM writes WHERE run_id=? ORDER BY id", (run_id,)).fetchall()

    def add_proposal(self, run_id: str, profile_id: str, field: str, current: Any, proposed: str, evidence: str,
                     page: int, source: str) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO proposals(run_id, profile_id, field, current_value, proposed_value, evidence, page, source, status) "
            "VALUES(?,?,?,?,?,?,?,?, COALESCE((SELECT status FROM proposals WHERE run_id=? AND profile_id=? AND field=?), 'open'))",
            (run_id, profile_id, field, None if current is None else str(current), proposed, evidence[:300], page, source,
             run_id, profile_id, field))

    def proposals(self, run_id: str, status: str | None = None) -> List[sqlite3.Row]:
        sql, params = "SELECT * FROM proposals WHERE run_id=?", [run_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        return self._conn().execute(sql + " ORDER BY profile_id, field", params).fetchall()

    def set_proposal_status(self, run_id: str, profile_id: str, field: str, status: str) -> None:
        self._conn().execute("UPDATE proposals SET status=? WHERE run_id=? AND profile_id=? AND field=?",
                             (status, run_id, profile_id, field))
