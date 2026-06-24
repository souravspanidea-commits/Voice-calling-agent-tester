"""Audit & Deduplication System for the Voice Agent Testing Platform.

Provides scenario fingerprinting, duplicate detection, execution logging,
coverage tracking, and full audit trail capabilities.  Designed for
10,000+ test scenarios across 14 testing dimensions in a hospital /
medical procurement context.

Dependencies: Python standard library only (hashlib, sqlite3, json, datetime).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

logger = logging.getLogger(__name__)

# The 14 testing dimensions — order matters for hash stability.
DIMENSION_KEYS: list[str] = [
    "dimension_1_personas",
    "dimension_2_scenarios",
    "dimension_3_languages",
    "dimension_4_speech_styles",
    "dimension_5_audio_environments",
    "dimension_6_interruptions",
    "dimension_7_questions",
    "dimension_8_compliance",
    "dimension_9_telephony",
    "dimension_10_data_availability",
    "dimension_11_emotional_states",
    "dimension_12_stt_difficulty",
    "dimension_13_conversation_states",
    "dimension_14_hospital_information",
]

# Short column names for the dimension columns in test_scenarios table.
_DIM_COL_MAP: dict[str, str] = {
    "dimension_1_personas":             "dim_persona",
    "dimension_2_scenarios":            "dim_scenario",
    "dimension_3_languages":            "dim_language",
    "dimension_4_speech_styles":        "dim_speech_style",
    "dimension_5_audio_environments":   "dim_audio_environment",
    "dimension_6_interruptions":        "dim_interruption",
    "dimension_7_questions":            "dim_question",
    "dimension_8_compliance":           "dim_compliance",
    "dimension_9_telephony":            "dim_telephony",
    "dimension_10_data_availability":   "dim_data_availability",
    "dimension_11_emotional_states":    "dim_emotional_state",
    "dimension_12_stt_difficulty":      "dim_stt_difficulty",
    "dimension_13_conversation_states": "dim_conversation_state",
    "dimension_14_hospital_information":"dim_hospital_info",
}


def _utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AuditSystem:
    """Complete audit, deduplication, and coverage tracking system.

    Usage::

        audit = AuditSystem("data/audit.db")

        # Register a scenario (returns existing if duplicate)
        result = audit.register_scenario(scenario_name, dimension_combo)

        # Log an execution
        audit.log_test_execution(scenario_id, status="passed", ...)

        # Check for duplicates before running
        dup = audit.check_duplicate(dimension_combo)

        # Generate reports
        coverage = audit.get_coverage_report()
        trail = audit.get_audit_trail(scenario_id)
    """

    def __init__(
        self,
        db_path: str | Path,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize the audit system and create tables if needed.

        Args:
            db_path: Path to the SQLite database file.  Will be created
                     if it does not exist.
            config_path: Path to ``test_configurations.json``.  If not
                provided, the system searches for it in the same
                directory as the database file, then in ``data/``
                relative to the project root.  This file is the
                **source of truth** for all dimension values, totals,
                and gap analysis.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load test_configurations.json — the source of truth
        self._config_path: Path | None = None
        self._config_data: dict[str, Any] = {}
        self._dimensions: dict[str, list[dict[str, Any]]] = {}
        self._dimension_totals: dict[str, int] = {}
        self._dimension_all_values: dict[str, list[str]] = {}
        self._load_test_configurations(config_path)

        self._init_schema()
        logger.info(
            "AuditSystem initialized: db=%s, config=%s, dimensions=%d",
            self.db_path,
            self._config_path or "NOT FOUND",
            len(self._dimensions),
        )

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def _load_test_configurations(
        self, config_path: str | Path | None = None
    ) -> None:
        """Load dimension data from ``test_configurations.json``.

        Searches for the config file in the following locations:
        1. Explicit ``config_path`` argument
        2. Same directory as the database file (e.g. ``data/``)
        3. ``data/`` under the project root (3 levels up from this module)

        Populates:
        - ``self._dimensions``:  raw dimension lists from the JSON
        - ``self._dimension_totals``:  {dim_key: count} for coverage
        - ``self._dimension_all_values``:  {dim_key: [name, ...]} for gap analysis
        """
        search_paths: list[Path] = []
        if config_path:
            search_paths.append(Path(config_path))
        search_paths.extend([
            self.db_path.parent / "test_configurations.json",
            Path(__file__).resolve().parent.parent.parent / "data" / "test_configurations.json",
        ])

        for candidate in search_paths:
            if candidate.exists():
                self._config_path = candidate
                break

        if self._config_path is None:
            logger.warning(
                "test_configurations.json not found in any search path. "
                "Coverage totals will be estimated from registered scenarios."
            )
            return

        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config_data = json.load(f)

        self._dimensions = self._config_data.get("dimensions", {})

        for dim_key, dim_list in self._dimensions.items():
            names = [item.get("name", "Unknown") for item in dim_list]
            self._dimension_totals[dim_key] = len(names)
            self._dimension_all_values[dim_key] = names

        logger.info(
            "Loaded %d dimensions from %s (total combinations: %s)",
            len(self._dimensions),
            self._config_path,
            " x ".join(str(v) for v in self._dimension_totals.values()),
        )

    def get_dimension_totals(self) -> dict[str, int]:
        """Return the total number of values per dimension from test_configurations.json.

        Returns:
            Dict mapping dimension keys to their value counts.
            Example: {"dimension_1_personas": 14, "dimension_2_scenarios": 14, ...}
        """
        return dict(self._dimension_totals)

    def get_all_dimension_values(self) -> dict[str, list[str]]:
        """Return all known dimension value names from test_configurations.json.

        Returns:
            Dict mapping dimension keys to lists of value names.
            Example: {"dimension_1_personas": ["Cooperative", "Angry", ...], ...}
        """
        return {k: list(v) for k, v in self._dimension_all_values.items()}

    def get_test_configurations_summary(self) -> dict[str, Any]:
        """Return a summary of the loaded test_configurations.json.

        Returns:
            Dict with metadata, dimension counts, total combinations, etc.
        """
        total_combinations = 1
        for count in self._dimension_totals.values():
            total_combinations *= count

        return {
            "config_path": str(self._config_path) if self._config_path else None,
            "metadata": self._config_data.get("metadata", {}),
            "total_dimensions": len(self._dimensions),
            "dimension_counts": self._dimension_totals,
            "total_possible_combinations": total_combinations,
            "dimensions": {
                dim_key: {
                    "count": len(values),
                    "values": values,
                }
                for dim_key, values in self._dimension_all_values.items()
            },
        }

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections with auto-commit."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create all tables and indexes from the SQL schema file.

        Searches for ``audit_schema.sql`` in the following locations
        (in order):
        1. Same directory as the database file (e.g. ``data/``)
        2. ``data/`` under the project root (3 levels up from this file)
        3. Next to this Python module (legacy location)

        If none are found, the schema is created inline.
        """
        search_paths = [
            self.db_path.parent / "audit_schema.sql",
            Path(__file__).resolve().parent.parent.parent / "data" / "audit_schema.sql",
            Path(__file__).resolve().parent / "audit_schema.sql",
        ]

        schema_sql: str | None = None
        for candidate in search_paths:
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                logger.debug("Loaded audit schema from %s", candidate)
                break

        if schema_sql is None:
            logger.warning(
                "audit_schema.sql not found in any search path; "
                "creating schema inline."
            )
            schema_sql = self._inline_schema()

        with self._connect() as conn:
            conn.executescript(schema_sql)
        logger.debug("Audit schema initialized successfully")

    @staticmethod
    def _inline_schema() -> str:
        """Return the full CREATE TABLE schema as a string (fallback)."""
        return """
        CREATE TABLE IF NOT EXISTS test_scenarios (
            id TEXT PRIMARY KEY, scenario_hash TEXT NOT NULL UNIQUE,
            scenario_name TEXT NOT NULL, description TEXT DEFAULT '',
            dim_persona TEXT, dim_scenario TEXT, dim_language TEXT,
            dim_speech_style TEXT, dim_audio_environment TEXT,
            dim_interruption TEXT, dim_question TEXT, dim_compliance TEXT,
            dim_telephony TEXT, dim_data_availability TEXT,
            dim_emotional_state TEXT, dim_stt_difficulty TEXT,
            dim_conversation_state TEXT, dim_hospital_info TEXT,
            full_config_json TEXT NOT NULL,
            difficulty TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'active',
            execution_count INTEGER DEFAULT 0,
            last_executed_at TEXT, created_by TEXT DEFAULT 'system',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_scenarios_hash ON test_scenarios(scenario_hash);
        CREATE INDEX IF NOT EXISTS idx_scenarios_persona ON test_scenarios(dim_persona);
        CREATE INDEX IF NOT EXISTS idx_scenarios_scenario ON test_scenarios(dim_scenario);
        CREATE INDEX IF NOT EXISTS idx_scenarios_status ON test_scenarios(status);

        CREATE TABLE IF NOT EXISTS test_execution_history (
            id TEXT PRIMARY KEY, scenario_id TEXT NOT NULL,
            scenario_hash TEXT NOT NULL, test_run_id TEXT,
            conversation_id TEXT,
            status TEXT NOT NULL, was_deduplicated INTEGER DEFAULT 0,
            dedup_record_id TEXT,
            started_at TEXT NOT NULL, ended_at TEXT, duration_sec REAL,
            turn_count INTEGER DEFAULT 0, avg_latency_ms REAL,
            avg_tester_latency_ms REAL, avg_outbound_latency_ms REAL,
            transcript_json TEXT, evaluation_json TEXT,
            eval_score REAL, eval_passed INTEGER,
            environment_json TEXT, triggered_by TEXT DEFAULT 'cli',
            execution_mode TEXT DEFAULT 'manual', created_at TEXT NOT NULL,
            FOREIGN KEY (scenario_id) REFERENCES test_scenarios(id)
        );
        CREATE INDEX IF NOT EXISTS idx_exec_scenario ON test_execution_history(scenario_id);
        CREATE INDEX IF NOT EXISTS idx_exec_status ON test_execution_history(status);
        CREATE INDEX IF NOT EXISTS idx_exec_started ON test_execution_history(started_at);

        CREATE TABLE IF NOT EXISTS scenario_deduplication (
            id TEXT PRIMARY KEY, scenario_hash TEXT NOT NULL,
            original_scenario_id TEXT NOT NULL,
            duplicate_scenario_id TEXT,
            action_taken TEXT NOT NULL,
            reason TEXT DEFAULT '', similarity_score REAL DEFAULT 1.0,
            submitted_config_json TEXT, detected_by TEXT DEFAULT 'system',
            detected_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dedup_hash ON scenario_deduplication(scenario_hash);

        CREATE TABLE IF NOT EXISTS coverage_tracking (
            id TEXT PRIMARY KEY, snapshot_date TEXT NOT NULL,
            dimension_name TEXT NOT NULL,
            total_values INTEGER NOT NULL, covered_values INTEGER NOT NULL,
            coverage_pct REAL NOT NULL,
            covered_list_json TEXT, gap_list_json TEXT,
            total_executions INTEGER DEFAULT 0,
            passed_executions INTEGER DEFAULT 0,
            pass_rate_pct REAL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            UNIQUE(snapshot_date, dimension_name)
        );
        CREATE INDEX IF NOT EXISTS idx_coverage_date ON coverage_tracking(snapshot_date);
        """

    # ------------------------------------------------------------------
    # 1. generate_scenario_hash()
    # ------------------------------------------------------------------

    @staticmethod
    def generate_scenario_hash(dimension_combo: dict[str, Any]) -> str:
        """Create a unique SHA-256 fingerprint for a scenario configuration.

        The hash is computed from a *canonical* JSON representation of
        the dimension values.  Only the ``name`` field of each dimension
        is used so that cosmetic config changes (descriptions, ordering)
        do not alter the hash.

        Args:
            dimension_combo: Dictionary mapping dimension keys to their
                chosen value dicts.  Each value dict should contain at
                least a ``"name"`` key.

                Example::

                    {
                        "dimension_1_personas": {"name": "Angry", ...},
                        "dimension_2_scenarios": {"name": "Identity Verification", ...},
                        ...
                    }

        Returns:
            A 64-character lowercase hex SHA-256 hash string.

        Raises:
            ValueError: If ``dimension_combo`` is empty.
        """
        if not dimension_combo:
            raise ValueError("dimension_combo must not be empty")

        # Build a canonical, ordered representation using only dimension names
        canonical: dict[str, str] = {}
        for key in DIMENSION_KEYS:
            value = dimension_combo.get(key, {})
            canonical[key] = value.get("name", "Unknown") if isinstance(value, dict) else str(value)

        # Serialize with sorted keys and no whitespace for determinism
        canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        hash_value = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        logger.debug("Generated hash %s for combo: %s", hash_value[:12], canonical)
        return hash_value

    # ------------------------------------------------------------------
    # 2. check_duplicate()
    # ------------------------------------------------------------------

    def check_duplicate(
        self,
        dimension_combo: dict[str, Any],
    ) -> dict[str, Any]:
        """Check if a scenario with the same configuration already exists.

        Performs an O(1) hash-based lookup against the ``test_scenarios``
        table.

        Args:
            dimension_combo: The dimension combination to check.

        Returns:
            A dictionary with the following structure::

                {
                    "is_duplicate": bool,
                    "scenario_hash": str,
                    "existing_scenario": {    # None if not a duplicate
                        "id": str,
                        "scenario_name": str,
                        "execution_count": int,
                        "last_executed_at": str | None,
                        "status": str,
                    },
                    "recommendation": str,    # "skip" | "run" | "rerun_stale"
                    "checked_at": str,
                }
        """
        scenario_hash = self.generate_scenario_hash(dimension_combo)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, scenario_name, execution_count, last_executed_at,
                       status, created_at
                FROM test_scenarios
                WHERE scenario_hash = ?
                """,
                (scenario_hash,),
            ).fetchone()

        result: dict[str, Any] = {
            "is_duplicate": False,
            "scenario_hash": scenario_hash,
            "existing_scenario": None,
            "recommendation": "run",
            "checked_at": _utc_now(),
        }

        if row:
            existing = dict(row)
            result["is_duplicate"] = True
            result["existing_scenario"] = existing

            # Decide recommendation based on staleness
            if existing["execution_count"] == 0:
                result["recommendation"] = "run"  # registered but never executed
            elif existing["status"] == "deprecated":
                result["recommendation"] = "skip"
            else:
                # Check if last execution was more than 7 days ago
                if existing["last_executed_at"]:
                    last_exec = datetime.fromisoformat(existing["last_executed_at"])
                    age_days = (datetime.now(timezone.utc) - last_exec).days
                    if age_days > 7:
                        result["recommendation"] = "rerun_stale"
                    else:
                        result["recommendation"] = "skip"
                else:
                    result["recommendation"] = "run"

            logger.info(
                "Duplicate check: hash=%s → DUPLICATE (original=%s, recommendation=%s)",
                scenario_hash[:12],
                existing["id"][:8],
                result["recommendation"],
            )
        else:
            logger.info("Duplicate check: hash=%s → NEW scenario", scenario_hash[:12])

        return result

    # ------------------------------------------------------------------
    # 3. register_scenario()
    # ------------------------------------------------------------------

    def register_scenario(
        self,
        scenario_name: str,
        dimension_combo: dict[str, Any],
        *,
        description: str = "",
        difficulty: str = "medium",
        created_by: str = "system",
        force: bool = False,
    ) -> dict[str, Any]:
        """Register a new scenario in the database.

        If a scenario with the same hash already exists, returns the
        existing record unless ``force=True`` (which creates a
        deduplication record and returns the original).

        Args:
            scenario_name: Human-readable name for the scenario.
            dimension_combo: Full dimension combination dict.
            description: Optional description.
            difficulty: One of 'easy', 'medium', 'hard', 'critical'.
            created_by: Identifier of who registered this scenario.
            force: If True, log a forced re-registration in the
                   deduplication table.

        Returns:
            A dictionary with::

                {
                    "scenario_id": str,
                    "scenario_hash": str,
                    "is_new": bool,
                    "action": str,        # "registered" | "existing" | "forced_rerun"
                    "registered_at": str,
                }
        """
        scenario_hash = self.generate_scenario_hash(dimension_combo)
        now = _utc_now()

        # Check for existing
        dup_check = self.check_duplicate(dimension_combo)

        if dup_check["is_duplicate"] and not force:
            existing = dup_check["existing_scenario"]

            # Log the deduplication event
            dedup_id = str(uuid4())
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO scenario_deduplication
                        (id, scenario_hash, original_scenario_id, action_taken,
                         reason, similarity_score, submitted_config_json,
                         detected_by, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dedup_id,
                        scenario_hash,
                        existing["id"],
                        "skipped",
                        f"Exact duplicate of scenario {existing['id']}",
                        1.0,
                        json.dumps(dimension_combo, default=str),
                        created_by,
                        now,
                    ),
                )

            logger.info(
                "Scenario already registered: hash=%s, id=%s",
                scenario_hash[:12],
                existing["id"][:8],
            )
            return {
                "scenario_id": existing["id"],
                "scenario_hash": scenario_hash,
                "is_new": False,
                "action": "existing",
                "dedup_record_id": dedup_id,
                "registered_at": existing["created_at"],
            }

        if dup_check["is_duplicate"] and force:
            existing = dup_check["existing_scenario"]
            dedup_id = str(uuid4())
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO scenario_deduplication
                        (id, scenario_hash, original_scenario_id, action_taken,
                         reason, similarity_score, submitted_config_json,
                         detected_by, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dedup_id,
                        scenario_hash,
                        existing["id"],
                        "forced_rerun",
                        "Force re-registration requested",
                        1.0,
                        json.dumps(dimension_combo, default=str),
                        created_by,
                        now,
                    ),
                )

            logger.info(
                "Force re-registration: hash=%s, original=%s",
                scenario_hash[:12],
                existing["id"][:8],
            )
            return {
                "scenario_id": existing["id"],
                "scenario_hash": scenario_hash,
                "is_new": False,
                "action": "forced_rerun",
                "dedup_record_id": dedup_id,
                "registered_at": now,
            }

        # New scenario — insert it
        scenario_id = str(uuid4())

        # Extract individual dimension names for indexed columns
        dim_values: dict[str, str] = {}
        for dim_key, col_name in _DIM_COL_MAP.items():
            val = dimension_combo.get(dim_key, {})
            dim_values[col_name] = val.get("name", "Unknown") if isinstance(val, dict) else str(val)

        full_config_json = json.dumps(dimension_combo, default=str)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO test_scenarios
                    (id, scenario_hash, scenario_name, description,
                     dim_persona, dim_scenario, dim_language, dim_speech_style,
                     dim_audio_environment, dim_interruption, dim_question,
                     dim_compliance, dim_telephony, dim_data_availability,
                     dim_emotional_state, dim_stt_difficulty,
                     dim_conversation_state, dim_hospital_info,
                     full_config_json, difficulty, status, execution_count,
                     created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, 'active', 0, ?, ?, ?)
                """,
                (
                    scenario_id,
                    scenario_hash,
                    scenario_name,
                    description,
                    dim_values.get("dim_persona"),
                    dim_values.get("dim_scenario"),
                    dim_values.get("dim_language"),
                    dim_values.get("dim_speech_style"),
                    dim_values.get("dim_audio_environment"),
                    dim_values.get("dim_interruption"),
                    dim_values.get("dim_question"),
                    dim_values.get("dim_compliance"),
                    dim_values.get("dim_telephony"),
                    dim_values.get("dim_data_availability"),
                    dim_values.get("dim_emotional_state"),
                    dim_values.get("dim_stt_difficulty"),
                    dim_values.get("dim_conversation_state"),
                    dim_values.get("dim_hospital_info"),
                    full_config_json,
                    difficulty,
                    created_by,
                    now,
                    now,
                ),
            )

        logger.info(
            "Registered new scenario: id=%s, hash=%s, name=%s",
            scenario_id[:8],
            scenario_hash[:12],
            scenario_name,
        )
        return {
            "scenario_id": scenario_id,
            "scenario_hash": scenario_hash,
            "is_new": True,
            "action": "registered",
            "registered_at": now,
        }

    # ------------------------------------------------------------------
    # 4. log_test_execution()
    # ------------------------------------------------------------------

    def log_test_execution(
        self,
        scenario_id: str,
        *,
        status: str,
        test_run_id: str | None = None,
        conversation_id: str | None = None,
        was_deduplicated: bool = False,
        dedup_record_id: str | None = None,
        duration_sec: float | None = None,
        turn_count: int = 0,
        avg_latency_ms: float | None = None,
        avg_tester_latency_ms: float | None = None,
        avg_outbound_latency_ms: float | None = None,
        transcript: list[dict[str, str]] | None = None,
        evaluation: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        triggered_by: str = "cli",
        execution_mode: str = "manual",
    ) -> dict[str, Any]:
        """Log a test execution with full audit trail.

        Args:
            scenario_id: FK to test_scenarios.id.
            status: One of 'passed', 'failed', 'error', 'timeout',
                    'skipped_duplicate', 'skipped_manual'.
            test_run_id: Optional link to the existing test_runs table.
            conversation_id: ElevenLabs conversation ID.
            was_deduplicated: Whether this was skipped due to dedup.
            dedup_record_id: FK to scenario_deduplication.id.
            duration_sec: How long the test took.
            turn_count: Number of conversation turns.
            avg_latency_ms: Average response latency.
            transcript: Full conversation transcript.
            evaluation: Evaluation results dict.
            environment: Execution environment metadata.
            triggered_by: Who triggered the run ('cli', 'ci', 'api').
            execution_mode: How scenarios were selected.

        Returns:
            Dict with execution_id, scenario_id, status, logged_at.
        """
        execution_id = str(uuid4())
        now = _utc_now()

        # Extract evaluation quick-access fields
        eval_score: float | None = None
        eval_passed: int | None = None
        if evaluation:
            eval_score = evaluation.get("llm_score") or evaluation.get("score")
            passed = evaluation.get("passed")
            eval_passed = 1 if passed else (0 if passed is False else None)

        # Look up the scenario hash
        scenario_hash = ""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT scenario_hash FROM test_scenarios WHERE id = ?",
                (scenario_id,),
            ).fetchone()
            if row:
                scenario_hash = row["scenario_hash"]

            conn.execute(
                """
                INSERT INTO test_execution_history
                    (id, scenario_id, scenario_hash, test_run_id, conversation_id,
                     status, was_deduplicated, dedup_record_id,
                     started_at, ended_at, duration_sec,
                     turn_count, avg_latency_ms,
                     avg_tester_latency_ms, avg_outbound_latency_ms,
                     transcript_json, evaluation_json, eval_score, eval_passed,
                     environment_json, triggered_by, execution_mode, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    scenario_id,
                    scenario_hash,
                    test_run_id,
                    conversation_id,
                    status,
                    1 if was_deduplicated else 0,
                    dedup_record_id,
                    now,
                    now if duration_sec is not None else None,
                    duration_sec,
                    turn_count,
                    avg_latency_ms,
                    avg_tester_latency_ms,
                    avg_outbound_latency_ms,
                    json.dumps(transcript) if transcript else None,
                    json.dumps(evaluation) if evaluation else None,
                    eval_score,
                    eval_passed,
                    json.dumps(environment) if environment else None,
                    triggered_by,
                    execution_mode,
                    now,
                ),
            )

            # Update execution count and last_executed_at on the scenario
            if status not in ("skipped_duplicate", "skipped_manual"):
                conn.execute(
                    """
                    UPDATE test_scenarios
                    SET execution_count = execution_count + 1,
                        last_executed_at = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, scenario_id),
                )

        logger.info(
            "Logged execution: id=%s, scenario=%s, status=%s",
            execution_id[:8],
            scenario_id[:8],
            status,
        )
        return {
            "execution_id": execution_id,
            "scenario_id": scenario_id,
            "status": status,
            "logged_at": now,
        }

    # ------------------------------------------------------------------
    # 5. get_coverage_report()
    # ------------------------------------------------------------------

    def get_coverage_report(
        self,
        dimension_totals: dict[str, int] | None = None,
        save_snapshot: bool = True,
    ) -> dict[str, Any]:
        """Generate a comprehensive coverage report across all dimensions.

        Uses ``test_configurations.json`` as the source of truth for
        the total number of values per dimension and for computing
        coverage gaps (which values have NOT been tested yet).

        Args:
            dimension_totals: Optional override dict mapping dimension
                keys to the total number of possible values.  If not
                provided, values are loaded from test_configurations.json.
                If that is also unavailable, counts are derived from
                the registered scenarios.
            save_snapshot: If True, save the results to ``coverage_tracking``.

        Returns:
            A comprehensive JSON-serializable coverage report.
        """
        # Merge: explicit overrides > loaded config > fallback to DB counts
        effective_totals = dict(self._dimension_totals)  # from test_configurations.json
        if dimension_totals:
            effective_totals.update(dimension_totals)
        now = _utc_now()
        today = _today()
        report: dict[str, Any] = {
            "report_date": today,
            "generated_at": now,
            "dimensions": {},
            "overall_coverage_pct": 0.0,
            "total_scenarios_registered": 0,
            "total_executions": 0,
            "total_passed": 0,
            "total_failed": 0,
            "dedup_stats": {},
        }

        with self._connect() as conn:
            # Overall counts
            row = conn.execute("SELECT COUNT(*) AS cnt FROM test_scenarios WHERE status = 'active'").fetchone()
            report["total_scenarios_registered"] = row["cnt"]

            row = conn.execute("SELECT COUNT(*) AS cnt FROM test_execution_history").fetchone()
            report["total_executions"] = row["cnt"]

            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM test_execution_history WHERE eval_passed = 1"
            ).fetchone()
            report["total_passed"] = row["cnt"]

            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM test_execution_history WHERE eval_passed = 0"
            ).fetchone()
            report["total_failed"] = row["cnt"]

            # Deduplication stats
            row = conn.execute("SELECT COUNT(*) AS cnt FROM scenario_deduplication").fetchone()
            dedup_total = row["cnt"]
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM scenario_deduplication WHERE action_taken = 'skipped'"
            ).fetchone()
            dedup_skipped = row["cnt"]
            report["dedup_stats"] = {
                "total_duplicates_detected": dedup_total,
                "duplicates_skipped": dedup_skipped,
                "duplicates_forced_rerun": dedup_total - dedup_skipped,
                "time_saved_estimate_min": round(dedup_skipped * 2.5, 1),  # ~2.5 min per test
            }

            # Per-dimension coverage
            coverage_pcts: list[float] = []
            for dim_key, col_name in _DIM_COL_MAP.items():
                # Get distinct values that have been tested
                rows = conn.execute(
                    f"SELECT DISTINCT {col_name} FROM test_scenarios WHERE status = 'active' AND {col_name} IS NOT NULL"
                ).fetchall()
                covered_values = [r[col_name] for r in rows]
                covered_count = len(covered_values)

                # Get total possible values (from test_configurations.json)
                total = effective_totals.get(dim_key, covered_count)
                if total == 0:
                    total = max(covered_count, 1)  # avoid division by zero

                coverage_pct = round((covered_count / total) * 100, 1)
                coverage_pcts.append(coverage_pct)

                # Get pass/fail counts for this dimension
                exec_count = 0
                passed_count = 0
                if covered_values:
                    placeholders = ", ".join("?" * len(covered_values))
                    row = conn.execute(
                        f"""
                        SELECT COUNT(*) AS cnt FROM test_execution_history eh
                        JOIN test_scenarios ts ON eh.scenario_id = ts.id
                        WHERE ts.{col_name} IN ({placeholders})
                        """,
                        covered_values,
                    ).fetchone()
                    exec_count = row["cnt"]

                    row = conn.execute(
                        f"""
                        SELECT COUNT(*) AS cnt FROM test_execution_history eh
                        JOIN test_scenarios ts ON eh.scenario_id = ts.id
                        WHERE ts.{col_name} IN ({placeholders}) AND eh.eval_passed = 1
                        """,
                        covered_values,
                    ).fetchone()
                    passed_count = row["cnt"]

                # Determine gaps from test_configurations.json
                # (which values exist in the config but have NOT been tested)
                all_known_values = self._dimension_all_values.get(dim_key, [])
                covered_set = set(covered_values)
                gap_values = [v for v in all_known_values if v not in covered_set]
                gap_count = len(gap_values) if all_known_values else total - covered_count

                dim_report = {
                    "dimension_key": dim_key,
                    "column": col_name,
                    "total_values": total,
                    "covered_values": covered_count,
                    "coverage_pct": coverage_pct,
                    "gap_count": gap_count,
                    "covered_list": covered_values,
                    "gap_list": gap_values,  # actual missing value names
                    "all_known_values": all_known_values,  # from test_configurations.json
                    "total_executions": exec_count,
                    "passed_executions": passed_count,
                    "pass_rate_pct": round((passed_count / exec_count) * 100, 1) if exec_count > 0 else 0.0,
                }
                report["dimensions"][dim_key] = dim_report

                # Save snapshot if requested
                if save_snapshot:
                    try:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO coverage_tracking
                                (id, snapshot_date, dimension_name, total_values,
                                 covered_values, coverage_pct, covered_list_json,
                                 gap_list_json, total_executions, passed_executions,
                                 pass_rate_pct, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(uuid4()),
                                today,
                                dim_key,
                                total,
                                covered_count,
                                coverage_pct,
                                json.dumps(covered_values),
                                json.dumps(gap_values),
                                exec_count,
                                passed_count,
                                round((passed_count / exec_count) * 100, 1) if exec_count > 0 else 0.0,
                                now,
                            ),
                        )
                    except Exception as e:
                        logger.warning("Failed to save coverage snapshot for %s: %s", dim_key, e)

            # Overall coverage
            report["overall_coverage_pct"] = (
                round(sum(coverage_pcts) / len(coverage_pcts), 1) if coverage_pcts else 0.0
            )

        logger.info("Coverage report generated: overall=%.1f%%", report["overall_coverage_pct"])
        return report

    # ------------------------------------------------------------------
    # 6. get_audit_trail()
    # ------------------------------------------------------------------

    def get_audit_trail(
        self,
        scenario_id: str | None = None,
        scenario_hash: str | None = None,
        *,
        limit: int = 100,
        include_transcript: bool = False,
    ) -> dict[str, Any]:
        """Get the complete audit history for a scenario.

        Args:
            scenario_id: The scenario's UUID.  Either this or
                ``scenario_hash`` must be provided.
            scenario_hash: The scenario's hash.  Used if ``scenario_id``
                is not given.
            limit: Max number of execution records to return.
            include_transcript: If True, include full transcript JSON
                in each execution record.

        Returns:
            Comprehensive audit trail including scenario details, all
            executions, and deduplication events.
        """
        if not scenario_id and not scenario_hash:
            raise ValueError("Must provide either scenario_id or scenario_hash")

        with self._connect() as conn:
            # Find the scenario
            if scenario_id:
                row = conn.execute(
                    "SELECT * FROM test_scenarios WHERE id = ?",
                    (scenario_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM test_scenarios WHERE scenario_hash = ?",
                    (scenario_hash,),
                ).fetchone()

            if not row:
                return {
                    "found": False,
                    "error": "Scenario not found",
                    "query": {"scenario_id": scenario_id, "scenario_hash": scenario_hash},
                }

            scenario = dict(row)
            scenario_id = scenario["id"]
            the_hash = scenario["scenario_hash"]

            # Parse full_config_json for readability
            if scenario.get("full_config_json"):
                try:
                    scenario["full_config"] = json.loads(scenario["full_config_json"])
                except (json.JSONDecodeError, TypeError):
                    scenario["full_config"] = None

            # Get execution history
            transcript_col = ", transcript_json" if include_transcript else ""
            rows = conn.execute(
                f"""
                SELECT id, scenario_id, scenario_hash, test_run_id,
                       conversation_id, status, was_deduplicated,
                       started_at, ended_at, duration_sec,
                       turn_count, avg_latency_ms,
                       eval_score, eval_passed,
                       triggered_by, execution_mode, created_at
                       {transcript_col}
                FROM test_execution_history
                WHERE scenario_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (scenario_id, limit),
            ).fetchall()
            executions = [dict(r) for r in rows]

            # Get deduplication events
            dedup_rows = conn.execute(
                """
                SELECT id, scenario_hash, original_scenario_id,
                       duplicate_scenario_id, action_taken, reason,
                       similarity_score, detected_by, detected_at
                FROM scenario_deduplication
                WHERE scenario_hash = ?
                ORDER BY detected_at DESC
                """,
                (the_hash,),
            ).fetchall()
            dedup_events = [dict(r) for r in dedup_rows]

        # Compute summary statistics
        total_execs = len(executions)
        passed = sum(1 for e in executions if e.get("eval_passed") == 1)
        failed = sum(1 for e in executions if e.get("eval_passed") == 0)
        avg_duration = (
            round(sum(e["duration_sec"] for e in executions if e.get("duration_sec")) / total_execs, 2)
            if total_execs > 0
            else 0
        )

        trail: dict[str, Any] = {
            "found": True,
            "scenario": scenario,
            "summary": {
                "total_executions": total_execs,
                "passed": passed,
                "failed": failed,
                "errors": total_execs - passed - failed,
                "pass_rate_pct": round((passed / total_execs) * 100, 1) if total_execs > 0 else 0.0,
                "avg_duration_sec": avg_duration,
                "duplicate_detections": len(dedup_events),
            },
            "executions": executions,
            "deduplication_events": dedup_events,
            "generated_at": _utc_now(),
        }

        logger.info(
            "Audit trail for scenario %s: %d executions, %d dedup events",
            scenario_id[:8],
            total_execs,
            len(dedup_events),
        )
        return trail

    # ------------------------------------------------------------------
    # Additional utility methods
    # ------------------------------------------------------------------

    def get_efficiency_report(self) -> dict[str, Any]:
        """Generate a report on testing efficiency and deduplication savings.

        Returns:
            Dict with metrics on duplicates skipped, time saved,
            execution modes breakdown, and pass rates by dimension.
        """
        with self._connect() as conn:
            # Total scenarios vs unique
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM test_scenarios WHERE status = 'active'"
            ).fetchone()
            total_active = row["cnt"]

            # Total executions
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM test_execution_history"
            ).fetchone()
            total_execs = row["cnt"]

            # Deduplicated executions
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM test_execution_history WHERE was_deduplicated = 1"
            ).fetchone()
            deduped = row["cnt"]

            # Execution mode breakdown
            mode_rows = conn.execute(
                """
                SELECT execution_mode, COUNT(*) AS cnt,
                       SUM(CASE WHEN eval_passed = 1 THEN 1 ELSE 0 END) AS passed
                FROM test_execution_history
                GROUP BY execution_mode
                """
            ).fetchall()

            modes = {}
            for r in mode_rows:
                r = dict(r)
                modes[r["execution_mode"]] = {
                    "total": r["cnt"],
                    "passed": r["passed"],
                    "pass_rate_pct": round((r["passed"] / r["cnt"]) * 100, 1) if r["cnt"] > 0 else 0,
                }

            # Top failure dimensions
            failure_dims: list[dict[str, Any]] = []
            for dim_key, col_name in _DIM_COL_MAP.items():
                rows = conn.execute(
                    f"""
                    SELECT ts.{col_name} AS dim_value,
                           COUNT(*) AS total,
                           SUM(CASE WHEN eh.eval_passed = 0 THEN 1 ELSE 0 END) AS failures
                    FROM test_execution_history eh
                    JOIN test_scenarios ts ON eh.scenario_id = ts.id
                    WHERE ts.{col_name} IS NOT NULL
                    GROUP BY ts.{col_name}
                    HAVING failures > 0
                    ORDER BY failures DESC
                    LIMIT 5
                    """
                ).fetchall()
                for r in rows:
                    r = dict(r)
                    failure_dims.append({
                        "dimension": dim_key,
                        "value": r["dim_value"],
                        "total_executions": r["total"],
                        "failures": r["failures"],
                        "failure_rate_pct": round((r["failures"] / r["total"]) * 100, 1),
                    })

            failure_dims.sort(key=lambda x: x["failures"], reverse=True)

        return {
            "total_active_scenarios": total_active,
            "total_executions": total_execs,
            "deduplicated_runs": deduped,
            "unique_runs": total_execs - deduped,
            "dedup_rate_pct": round((deduped / total_execs) * 100, 1) if total_execs > 0 else 0,
            "estimated_time_saved_min": round(deduped * 2.5, 1),
            "execution_modes": modes,
            "top_failure_dimensions": failure_dims[:10],
            "generated_at": _utc_now(),
        }

    def get_stale_scenarios(self, days_threshold: int = 7) -> list[dict[str, Any]]:
        """Find scenarios that haven't been executed recently.

        Args:
            days_threshold: Number of days after which a scenario is
                considered stale.

        Returns:
            List of stale scenario records.
        """
        cutoff = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, scenario_name, scenario_hash, dim_persona,
                       dim_scenario, difficulty, execution_count,
                       last_executed_at, created_at
                FROM test_scenarios
                WHERE status = 'active'
                  AND (last_executed_at IS NULL
                       OR julianday('now') - julianday(last_executed_at) > ?)
                ORDER BY last_executed_at ASC NULLS FIRST
                """,
                (days_threshold,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_scenario_by_hash(self, scenario_hash: str) -> dict[str, Any] | None:
        """Look up a scenario by its hash.

        Args:
            scenario_hash: The SHA-256 hash to look up.

        Returns:
            The scenario record as a dict, or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM test_scenarios WHERE scenario_hash = ?",
                (scenario_hash,),
            ).fetchone()
        return dict(row) if row else None

    def get_dashboard_stats(self) -> dict[str, Any]:
        """Get quick dashboard statistics for monitoring.

        Returns:
            Dict with key metrics for the testing dashboard.
        """
        with self._connect() as conn:
            stats: dict[str, Any] = {}

            # Scenario counts
            row = conn.execute("SELECT COUNT(*) AS cnt FROM test_scenarios").fetchone()
            stats["total_scenarios"] = row["cnt"]

            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM test_scenarios WHERE status = 'active'"
            ).fetchone()
            stats["active_scenarios"] = row["cnt"]

            # Execution counts (today)
            today = _today()
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM test_execution_history WHERE started_at >= ?",
                (today,),
            ).fetchone()
            stats["executions_today"] = row["cnt"]

            # Pass rate (today)
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN eval_passed = 1 THEN 1 ELSE 0 END) AS passed
                FROM test_execution_history
                WHERE started_at >= ?
                """,
                (today,),
            ).fetchone()
            total = row["total"]
            passed = row["passed"] or 0
            stats["today_pass_rate_pct"] = round((passed / total) * 100, 1) if total > 0 else 0

            # Duplicates blocked today
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM scenario_deduplication WHERE detected_at >= ?",
                (today,),
            ).fetchone()
            stats["duplicates_blocked_today"] = row["cnt"]

            # Average execution time (last 50 runs)
            row = conn.execute(
                """
                SELECT AVG(duration_sec) AS avg_dur
                FROM (
                    SELECT duration_sec FROM test_execution_history
                    WHERE duration_sec IS NOT NULL
                    ORDER BY started_at DESC
                    LIMIT 50
                )
                """
            ).fetchone()
            stats["avg_execution_sec"] = round(row["avg_dur"], 1) if row["avg_dur"] else 0

        stats["generated_at"] = _utc_now()
        return stats
