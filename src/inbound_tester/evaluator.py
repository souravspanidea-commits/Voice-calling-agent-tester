"""Rule-based and LLM-as-judge evaluation of test conversations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from inbound_tester.scenarios import RuleCheck, ScenarioConfig

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    id: str
    description: str
    passed: bool
    detail: str = ""


@dataclass
class EvaluationResult:
    passed: bool
    rule_results: list[CheckResult] = field(default_factory=list)
    llm_score: float | None = None
    llm_summary: str = ""
    llm_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rule_results": [
                {
                    "id": r.id,
                    "description": r.description,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in self.rule_results
            ],
            "llm_score": self.llm_score,
            "llm_summary": self.llm_summary,
            "llm_details": self.llm_details,
        }


def _texts_for_role(transcript: list[dict[str, str]], role: str) -> list[str]:
    return [e["text"] for e in transcript if e.get("role") == role]


def _run_rule_check(check: RuleCheck, transcript: list[dict[str, str]]) -> CheckResult:
    texts = _texts_for_role(transcript, check.role)
    combined = " ".join(texts).lower()

    if check.type == "keyword_any":
        matched = [p for p in check.patterns if p.lower() in combined]
        passed = bool(matched) if check.required else True
        detail = f"Matched: {matched}" if matched else f"None of {check.patterns} found in {check.role} messages"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "keyword_none":
        # Passes when NONE of the patterns are found (negative check).
        matched = [p for p in check.patterns if p.lower() in combined]
        passed = not bool(matched)
        detail = f"Unwanted matches found: {matched}" if matched else "No unwanted patterns found"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "keyword_all":
        missing = [p for p in check.patterns if p.lower() not in combined]
        passed = not missing if check.required else True
        detail = "All patterns found" if not missing else f"Missing: {missing}"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "regex_any":
        matched = [p for p in check.patterns if re.search(p, combined, re.IGNORECASE)]
        passed = bool(matched) if check.required else True
        detail = f"Matched regex: {matched}" if matched else f"No regex match among {check.patterns}"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "max_agent_turns":
        agent_turns = len(_texts_for_role(transcript, "agent"))
        limit = int(check.patterns[0]) if check.patterns else 20
        passed = agent_turns <= limit
        detail = f"Agent turns: {agent_turns}, limit: {limit}"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    return CheckResult(
        id=check.id,
        description=check.description,
        passed=False,
        detail=f"Unknown check type: {check.type}",
    )


class Evaluator:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model

    def evaluate(
        self,
        scenario: ScenarioConfig,
        transcript: list[dict[str, str]],
        *,
        use_llm: bool = True,
    ) -> EvaluationResult:
        rule_results = [_run_rule_check(check, transcript) for check in scenario.rule_checks]
        rules_passed = all(r.passed for r in rule_results if r.passed is not None)

        llm_score: float | None = None
        llm_summary = ""
        llm_details: dict[str, Any] = {}

        if use_llm and self.client and scenario.llm_evaluation_prompt.strip():
            llm_score, llm_summary, llm_details = self._llm_judge(scenario, transcript)

        passed = rules_passed
        if llm_score is not None and llm_score < 0.6:
            passed = False

        return EvaluationResult(
            passed=passed,
            rule_results=rule_results,
            llm_score=llm_score,
            llm_summary=llm_summary,
            llm_details=llm_details,
        )

    def _llm_judge(
        self,
        scenario: ScenarioConfig,
        transcript: list[dict[str, str]],
    ) -> tuple[float | None, str, dict[str, Any]]:
        transcript_text = "\n".join(
            f"{e['role']}: {e['text']}" for e in transcript if e.get("text")
        )

        prompt = f"""Evaluate how well the outbound agent handled this test call.

Scenario: {scenario.name}
{scenario.description}

Evaluation criteria:
{scenario.llm_evaluation_prompt}

Transcript:
{transcript_text}

Respond with JSON only:
{{
  "score": 0.0 to 1.0,
  "passed": true/false,
  "summary": "one paragraph",
  "strengths": ["..."],
  "issues": ["..."]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a QA evaluator for outbound voice agents. Respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            return (
                float(data.get("score", 0)),
                str(data.get("summary", "")),
                data,
            )
        except Exception as exc:
            logger.exception("LLM evaluation failed")
            return None, f"LLM evaluation failed: {exc}", {}
