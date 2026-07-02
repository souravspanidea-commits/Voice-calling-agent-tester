"""Rule-based and LLM-as-judge evaluation of test conversations."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
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
    audio_score: float | None = None
    audio_summary: str = ""
    audio_details: dict[str, Any] = field(default_factory=dict)

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
            "audio_score": self.audio_score,
            "audio_summary": self.audio_summary,
            "audio_details": self.audio_details,
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
        audio_path: Path | str | None = None,
    ) -> EvaluationResult:
        rule_results = [_run_rule_check(check, transcript) for check in scenario.rule_checks]
        rules_passed = all(r.passed for r in rule_results if r.passed is not None)

        llm_score: float | None = None
        llm_summary = ""
        llm_details: dict[str, Any] = {}

        audio_score: float | None = None
        audio_summary = ""
        audio_details: dict[str, Any] = {}

        if use_llm and self.client and scenario.llm_evaluation_prompt.strip():
            llm_score, llm_summary, llm_details = self._llm_judge(scenario, transcript)
            
            if audio_path:
                audio_score, audio_summary, audio_details = self._llm_audio_judge(scenario, audio_path)

        passed = rules_passed
        if llm_score is not None and llm_score < 0.6:
            passed = False
        if audio_score is not None and audio_score < 0.6:
            passed = False

        return EvaluationResult(
            passed=passed,
            rule_results=rule_results,
            llm_score=llm_score,
            llm_summary=llm_summary,
            llm_details=llm_details,
            audio_score=audio_score,
            audio_summary=audio_summary,
            audio_details=audio_details,
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

    def _llm_audio_judge(
        self,
        scenario: ScenarioConfig,
        audio_path: Path | str,
    ) -> tuple[float | None, str, dict[str, Any]]:
        path = Path(audio_path)
        if not path.exists():
            logger.error("Audio file not found for evaluation: %s", path)
            return None, "Audio file not found", {}
            
        try:
            with open(path, "rb") as f:
                b64_audio = base64.b64encode(f.read()).decode("utf-8")
        except Exception as exc:
            logger.exception("Failed to read audio file")
            return None, f"Failed to read audio file: {exc}", {}
            
        prompt = f"""Evaluate the acoustic and audio quality of the outbound agent in this recording.

Scenario: {scenario.name}
{scenario.description}

Evaluate the following acoustic parameters:
1. Pronunciation: Did the agent pronounce Hindi and English words accurately?
2. Pacing & Fluency: Was the cadence natural, or robotic/rushed?
3. Tone & Empathy: Did the agent sound polite and appropriately empathetic?

Provide a summary paragraph and a score from 0.0 to 1.0.

Respond with JSON only:
{{
  "score": 0.0 to 1.0,
  "passed": true/false,
  "summary": "one paragraph focusing purely on audio quality",
  "strengths": ["..."],
  "issues": ["..."]
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-audio-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": b64_audio,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
            )
            
            raw = response.choices[0].message.content or "{}"
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
                
            data = json.loads(raw)
            return (
                float(data.get("score", 0)),
                str(data.get("summary", "")),
                data,
            )
        except Exception as exc:
            logger.exception("LLM audio evaluation failed")
            return None, f"LLM audio evaluation failed: {exc}", {}
 