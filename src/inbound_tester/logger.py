"""Persist test runs, raw WebSocket events, and evaluation results."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationLogger:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS test_runs (
                    id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    conversation_id TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    turn_count INTEGER DEFAULT 0,
                    transcript_json TEXT,
                    evaluation_json TEXT,
                    duration_sec REAL,
                    avg_latency_ms REAL,
                    avg_tester_latency_ms REAL,
                    avg_outbound_latency_ms REAL
                );

                CREATE TABLE IF NOT EXISTS ws_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    event_type TEXT,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_ws_events_run ON ws_events(test_run_id);
                """
            )
            # Backwards-compatible migration for existing DBs
            for col, coltype in [
                ("duration_sec", "REAL"),
                ("avg_latency_ms", "REAL"),
                ("avg_tester_latency_ms", "REAL"),
                ("avg_outbound_latency_ms", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE test_runs ADD COLUMN {col} {coltype}")
                except Exception:
                    pass  # column already exists

    def start_run(self, scenario_id: str, agent_id: str) -> str:
        run_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO test_runs (id, scenario_id, agent_id, status, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, scenario_id, agent_id, "running", _utc_now()),
            )
        return run_id

    def log_ws_event(self, test_run_id: str, event: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ws_events (test_run_id, received_at, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    test_run_id,
                    _utc_now(),
                    event.get("type"),
                    json.dumps(event),
                ),
            )

    def finish_run(
        self,
        test_run_id: str,
        *,
        status: str,
        conversation_id: str | None,
        turn_count: int,
        transcript: list[dict[str, str]],
        evaluation: dict[str, Any] | None = None,
        duration_sec: float | None = None,
        avg_latency_ms: float | None = None,
        avg_tester_latency_ms: float | None = None,
        avg_outbound_latency_ms: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE test_runs
                SET status = ?, ended_at = ?, conversation_id = ?,
                    turn_count = ?, transcript_json = ?, evaluation_json = ?,
                    duration_sec = ?, avg_latency_ms = ?,
                    avg_tester_latency_ms = ?, avg_outbound_latency_ms = ?
                WHERE id = ?
                """,
                (
                    status,
                    _utc_now(),
                    conversation_id,
                    turn_count,
                    json.dumps(transcript),
                    json.dumps(evaluation) if evaluation else None,
                    duration_sec,
                    avg_latency_ms,
                    avg_tester_latency_ms,
                    avg_outbound_latency_ms,
                    test_run_id,
                ),
            )

    def get_run(self, test_run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM test_runs WHERE id = ?",
                (test_run_id,),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get("transcript_json"):
                result["transcript"] = json.loads(result["transcript_json"])
            if result.get("evaluation_json"):
                result["evaluation"] = json.loads(result["evaluation_json"])
            return result

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, scenario_id, agent_id, conversation_id, status,
                       started_at, ended_at, turn_count, evaluation_json,
                       duration_sec, avg_latency_ms,
                       avg_tester_latency_ms, avg_outbound_latency_ms
                FROM test_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                if item.get("evaluation_json"):
                    item["evaluation"] = json.loads(item["evaluation_json"])
                del item["evaluation_json"]
                results.append(item)
            return results

    def get_run_by_prefix(self, prefix: str) -> dict[str, Any] | None:
        """Find a run whose ID starts with the given prefix."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM test_runs WHERE id LIKE ? ORDER BY started_at DESC LIMIT 1",
                (f"{prefix}%",),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get("transcript_json"):
                result["transcript"] = json.loads(result["transcript_json"])
            if result.get("evaluation_json"):
                result["evaluation"] = json.loads(result["evaluation_json"])
            return result
 