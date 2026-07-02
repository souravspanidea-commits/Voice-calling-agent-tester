"""Scenario configuration schema and loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# File prefix that marks a shared/base file — not a runnable scenario.
_BASE_PREFIX = "_"


class RuleCheck(BaseModel):
    id: str
    description: str
    type: str = "keyword_any"
    patterns: list[str] = Field(default_factory=list)
    role: str = "agent"
    required: bool = True


class ScenarioConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    persona_prompt: str
    goals: list[str] = Field(default_factory=list)
    opening_behavior: str = "wait_for_agent"
    max_turns: int | None = None
    dynamic_variables: dict[str, str | int | float | bool] = Field(default_factory=dict)
    end_signals: list[str] = Field(
        default_factory=lambda: ["[END]", "[HANGUP]", "[SCENARIO_COMPLETE]"]
    )
    rule_checks: list[RuleCheck] = Field(default_factory=list)
    llm_evaluation_prompt: str = ""
    language_instructions: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    active_configuration: str = ""


# ---------------------------------------------------------------------------
# Base file loading & inheritance
# ---------------------------------------------------------------------------

def _load_base(base_path: Path) -> dict[str, Any]:
    """Load and cache a _base_shared.yaml file."""
    with base_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_inheritance(
    data: dict[str, Any],
    scenario_dir: Path,
) -> dict[str, Any]:
    """Merge a scenario's data with its inherited base file.

    Inheritance rules:
    - ``inherits`` names a base file (relative to the scenario directory).
    - ``dynamic_variables``: base values are used as defaults; scenario values
      override per-key.
    - ``max_turns``: scenario value wins if set; otherwise base value.
    - ``shared_rule_checks`` from base are **prepended** to the scenario's
      ``rule_checks`` so every scenario gets the shared checks automatically.
      Duplicate check IDs are skipped (scenario takes priority).
    - ``shared_llm_evaluation_dimensions`` and ``shared_pass_criteria`` are
      injected into the scenario's ``llm_evaluation_prompt`` by replacing
      the ``[INHERIT ...]`` markers.
    - Fields not in the base are left untouched.
    """
    inherits = data.pop("inherits", None)
    if not inherits:
        return data

    base_path = scenario_dir / inherits
    if not base_path.exists():
        logger.warning("Base file %s not found — skipping inheritance", base_path)
        return data

    base = _load_base(base_path)

    # --- dynamic_variables: base defaults, scenario overrides ---
    base_vars = dict(base.get("dynamic_variables", {}))
    scenario_vars = dict(data.get("dynamic_variables", {}))
    merged_vars = {**base_vars, **scenario_vars}
    if merged_vars:
        data["dynamic_variables"] = merged_vars

    # --- max_turns: scenario wins, else base ---
    if data.get("max_turns") is None and base.get("max_turns") is not None:
        data["max_turns"] = base["max_turns"]

    # --- rule_checks: shared checks prepended, deduped by id ---
    shared_checks: list[dict[str, Any]] = base.get("shared_rule_checks", [])
    scenario_checks: list[dict[str, Any]] = data.get("rule_checks", [])
    scenario_check_ids = {c["id"] for c in scenario_checks}
    merged_checks = [c for c in shared_checks if c["id"] not in scenario_check_ids]
    merged_checks.extend(scenario_checks)
    data["rule_checks"] = merged_checks

    # --- llm_evaluation_prompt: inject shared dimensions & pass criteria ---
    prompt = data.get("llm_evaluation_prompt", "")
    shared_dims = base.get("shared_llm_evaluation_dimensions", "")
    shared_pass = base.get("shared_pass_criteria", "")

    if shared_dims:
        prompt = prompt.replace("[INHERIT shared_llm_evaluation_dimensions]", shared_dims)
    if shared_pass:
        prompt = prompt.replace("[INHERIT shared_pass_criteria]", shared_pass)

    data["llm_evaluation_prompt"] = prompt

    return data


# ---------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------- 

def load_scenario(path: Path | str) -> ScenarioConfig:
    """Load a single scenario YAML and resolve inheritance."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    data = _resolve_inheritance(data, scenario_dir=path.parent)
    return ScenarioConfig.model_validate(data)


def load_scenarios(directory: Path | str) -> list[ScenarioConfig]:
    """Load all runnable scenarios in *directory* (skips ``_``-prefixed files)."""
    directory = Path(directory)
    scenarios: list[ScenarioConfig] = []
    for path in sorted(directory.glob("*.yaml")):
        # Skip base/shared files (e.g. _base_shared.yaml)
        if path.name.startswith(_BASE_PREFIX):
            logger.debug("Skipping base file: %s", path.name)
            continue
        try:
            scenarios.append(load_scenario(path))
        except Exception as e:
            logger.warning("Skipping file %s which is not a valid ScenarioConfig: %s", path.name, e)
    return scenarios
