"""Rule-based and LLM-as-judge evaluation of test conversations."""

from __future__ import annotations

import base64
import json
import logging
import re
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from inbound_tester.scenarios import RuleCheck, ScenarioConfig

logger = logging.getLogger(__name__)

# -- Audio model ---------------------------------------------------------------
_AUDIO_MODEL = "gpt-audio-mini"

# -- Auto-cut scenarios: calls that correctly terminate in <=2 agent turns ------
_AUTO_CUT_SCENARIOS = {"Voicemail", "Wrong Hospital"}
_AUTO_CUT_MAX_TURNS = 2

# -- Default pass threshold (overridable via scenario.metadata) ----------------
_DEFAULT_PASS_THRESHOLD = 0.6

# -- Semantic parameter names (fixed set — matches final eval spec) ------------
_SEMANTIC_PARAMS = [
    "goal_alignment",
    "expected_outcome_match",
    "success_criteria_adherence",
    "failure_mode_avoidance",
    "required_data_points_collection",
    "readback_confirmation_accuracy",
    "scope_handling",
    "graceful_exit",
]

# -- Audio parameter names (fixed set — matches final eval spec) ---------------
_AUDIO_PARAMS = [
    "pronunciation_accuracy",
    "code_switching_fluency",
    "pacing_and_fluency",
    "tone_and_empathy",
    "audio_overlap_interruption_handling",
]

# -- Auto-cut reduced rubric parameters ---------------------------------------
_AUTO_CUT_SEMANTIC_PARAMS = [
    "goal_alignment",           # did it correctly identify and terminate?
    "expected_outcome_match",   # outcome == "terminated_correctly"?
]


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

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

    # Semantic judge
    llm_score: float | None = None
    llm_summary: str = ""
    llm_details: dict[str, Any] = field(default_factory=dict)
    semantic_scores: dict[str, float] = field(default_factory=dict)

    # Audio judge
    audio_score: float | None = None
    audio_summary: str = ""
    audio_details: dict[str, Any] = field(default_factory=dict)
    acoustic_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        rule_gates: dict[str, bool] = {}
        flags: dict[str, bool] = {}
        for r in self.rule_results:
            if r.id.endswith("_flag"):
                flags[r.id] = r.passed
            else:
                rule_gates[r.id] = r.passed

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
            "semantic_scores":  self.semantic_scores,
            "acoustic_scores":  self.acoustic_scores,
            "rule_gates":       rule_gates,
            "flags":            flags,
            "llm_score":        self.llm_score,
            "llm_summary":      self.llm_summary,
            "llm_details":      self.llm_details,
            "audio_score":      self.audio_score,
            "audio_summary":    self.audio_summary,
            "audio_details":    self.audio_details,
        }


# -----------------------------------------------------------------------------
# Rule check helpers
# -----------------------------------------------------------------------------

def _texts_for_role(transcript: list[dict[str, str]], role: str) -> list[str]:
    return [e["text"] for e in transcript if e.get("role") == role and e.get("text")]


def _run_rule_check(check: RuleCheck, transcript: list[dict[str, str]], outbound_latencies_ms: list[float] | None = None) -> CheckResult:
    texts = _texts_for_role(transcript, check.role)
    combined = " ".join(texts).lower()

    if check.type == "keyword_any":
        matched = [p for p in check.patterns if p.lower() in combined]
        passed = bool(matched) if check.required else True
        detail = f"Matched: {matched}" if matched else f"None of {check.patterns} found"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "keyword_none":
        matched = [p for p in check.patterns if p.lower() in combined]
        passed = not bool(matched)
        detail = f"Unwanted matches: {matched}" if matched else "No unwanted patterns found"
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
        limit = int(check.patterns[0]) if check.patterns else 30
        passed = agent_turns <= limit
        detail = f"Agent turns: {agent_turns}, ceiling: {limit}"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "min_agent_turns":
        agent_turns = len(_texts_for_role(transcript, "agent"))
        floor = int(check.patterns[0]) if check.patterns else 3
        passed = agent_turns >= floor
        detail = f"Agent turns: {agent_turns}, floor: {floor}"
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "repeated_utterance":
        threshold = int(check.patterns[0]) if check.patterns else 3
        agent_texts = _texts_for_role(transcript, "agent")
        max_consecutive = 1
        current_run = 1
        for i in range(1, len(agent_texts)):
            if agent_texts[i].strip() == agent_texts[i - 1].strip():
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 1
        passed = max_consecutive < threshold
        detail = (
            f"Max consecutive identical utterances: {max_consecutive} (threshold: {threshold})"
        )
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    if check.type == "dead_air":
        has_silence = "[silence]" in combined
        high_latency = False
        if outbound_latencies_ms:
            high_latency = any(lat > 5000 for lat in outbound_latencies_ms)
        passed = not (has_silence or high_latency)
        detail = "No dead air detected."
        if has_silence: detail = "Transcript contained explicit [SILENCE] timeout."
        elif high_latency: detail = "Agent response took longer than 5 seconds mid-call."
        return CheckResult(id=check.id, description=check.description, passed=passed, detail=detail)

    return CheckResult(
        id=check.id,
        description=check.description,
        passed=False,
        detail=f"Unknown check type: {check.type}",
    )


# -----------------------------------------------------------------------------
# JSON fence stripping
# -----------------------------------------------------------------------------

def _strip_json_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?\s*", "", raw).strip()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------

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
        outbound_latencies_ms: list[float] | None = None,
    ) -> EvaluationResult:
        # -- Rule-based gates -------------------------------------------------
        rule_results = [_run_rule_check(check, transcript, outbound_latencies_ms) for check in scenario.rule_checks]
        
        # We manually inject the dead_air and no_ai flags here if they aren't in YAML to preserve your setup
        # But if you've added them to YAML, we don't duplicate. We'll just enforce the rules that are returned.
        rules_passed = all(r.passed for r in rule_results if not r.id.endswith("_flag"))

        pass_threshold: float = float(
            scenario.metadata.get("llm_pass_threshold", _DEFAULT_PASS_THRESHOLD)
        )

        agent_turn_count = len(_texts_for_role(transcript, "agent"))
        is_auto_cut = (
            scenario.name in _AUTO_CUT_SCENARIOS
            and agent_turn_count <= _AUTO_CUT_MAX_TURNS
        )

        llm_score: float | None = None
        llm_summary = ""
        llm_details: dict[str, Any] = {}
        semantic_scores: dict[str, float] = {}

        audio_score: float | None = None
        audio_summary = ""
        audio_details: dict[str, Any] = {}
        acoustic_scores: dict[str, float] = {}

        if use_llm and self.client and scenario.llm_evaluation_prompt.strip():
            if is_auto_cut:
                llm_score, llm_summary, llm_details, semantic_scores = (
                    self._llm_judge_auto_cut(scenario, transcript)
                )
            else:
                llm_score, llm_summary, llm_details, semantic_scores = (
                    self._llm_judge(scenario, transcript)
                )

            if audio_path:
                audio_score, audio_summary, audio_details, acoustic_scores = (
                    self._llm_audio_judge(scenario, audio_path, is_auto_cut=is_auto_cut)
                )

        passed = rules_passed
        if llm_score is not None and llm_score < pass_threshold:
            passed = False
        if audio_score is not None and audio_score < pass_threshold:
            passed = False

        return EvaluationResult(
            passed=passed,
            rule_results=rule_results,
            llm_score=llm_score,
            llm_summary=llm_summary,
            llm_details=llm_details,
            semantic_scores=semantic_scores,
            audio_score=audio_score,
            audio_summary=audio_summary,
            audio_details=audio_details,
            acoustic_scores=acoustic_scores,
        )

    def _llm_judge(
        self,
        scenario: ScenarioConfig,
        transcript: list[dict[str, str]],
    ) -> tuple[float | None, str, dict[str, Any], dict[str, float]]:
        transcript_text = "\n".join(
            f"{e['role']}: {e['text']}" for e in transcript if e.get("text")
        )

        param_list = "\n".join(
            f'  "{p}": <0.0 to 1.0>' for p in _SEMANTIC_PARAMS
        )

        prompt = f"""Evaluate how well the outbound agent handled this test call.

Scenario: {scenario.name}
{scenario.description}

Evaluation criteria:
{scenario.llm_evaluation_prompt}

Transcript:
{transcript_text}

Score EACH of the following parameters individually from 0.0 to 1.0:
  goal_alignment                  — Did the agent steer toward its objective without being derailed?
  expected_outcome_match          — Did the call end as expected (completed/partial/abandoned/escalated)?
  success_criteria_adherence      — Did the agent follow positive behavioral rules (readbacks, SSML tags, etc.)?
  failure_mode_avoidance          — Did the agent avoid critical errors listed in failure_modes?
  required_data_points_collection — Were all required data fields successfully extracted?
  readback_confirmation_accuracy  — Did the agent read back collected data correctly before closing?
  scope_handling                  — Did the agent stay on task when the caller tried to derail it?
  graceful_exit                   — Did the agent close the call professionally with a summary?

Respond with JSON only:
{{
  "score": <overall average 0.0 to 1.0>,
  "passed": <true if score >= 0.6>,
  "summary": "<one paragraph>",
  "parameter_scores": {{
{param_list}
  }},
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

            param_scores: dict[str, float] = {}
            raw_params = data.get("parameter_scores", {})
            for p in _SEMANTIC_PARAMS:
                try:
                    param_scores[p] = float(raw_params.get(p, 0.0))
                except (TypeError, ValueError):
                    param_scores[p] = 0.0

            overall = (
                sum(param_scores.values()) / len(param_scores)
                if param_scores else float(data.get("score", 0.0))
            )

            return overall, str(data.get("summary", "")), data, param_scores

        except Exception as exc:
            logger.exception("LLM semantic evaluation failed")
            return None, f"LLM evaluation failed: {exc}", {}, {}

    def _llm_judge_auto_cut(
        self,
        scenario: ScenarioConfig,
        transcript: list[dict[str, str]],
    ) -> tuple[float | None, str, dict[str, Any], dict[str, float]]:
        transcript_text = "\n".join(
            f"{e['role']}: {e['text']}" for e in transcript if e.get("text")
        )

        prompt = f"""Evaluate this outbound agent call that ended with an auto-cut (voicemail or wrong number).

Scenario: {scenario.name}
{scenario.description}

The agent is expected to detect this is a voicemail or wrong number and terminate the call
within 1-2 turns WITHOUT attempting data collection.

Score EACH parameter from 0.0 to 1.0:
  goal_alignment         — Did the agent correctly identify this as voicemail/wrong number and terminate?
  expected_outcome_match — Did the call end with the correct outcome (terminated_correctly)?

Respond with JSON only:
{{
  "score": <average of both scores>,
  "passed": <true if score >= 0.6>,
  "summary": "<one paragraph>",
  "parameter_scores": {{
    "goal_alignment": <0.0 to 1.0>,
    "expected_outcome_match": <0.0 to 1.0>
  }},
  "strengths": ["..."],
  "issues": ["..."]
}}

Transcript:
{transcript_text}
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

            raw_params = data.get("parameter_scores", {})
            param_scores: dict[str, float] = {}
            for p in _AUTO_CUT_SEMANTIC_PARAMS:
                try:
                    param_scores[p] = float(raw_params.get(p, 0.0))
                except (TypeError, ValueError):
                    param_scores[p] = 0.0

            overall = (
                sum(param_scores.values()) / len(param_scores)
                if param_scores else float(data.get("score", 0.0))
            )

            return overall, str(data.get("summary", "")), data, param_scores

        except Exception as exc:
            logger.exception("LLM auto-cut evaluation failed")
            return None, f"LLM evaluation failed: {exc}", {}, {}

    def _llm_audio_judge(
        self,
        scenario: ScenarioConfig,
        audio_path: Path | str,
        *,
        is_auto_cut: bool = False,
    ) -> tuple[float | None, str, dict[str, Any], dict[str, float]]:
        if is_auto_cut:
            logger.debug("Skipping audio eval for auto-cut scenario: %s", scenario.name)
            return None, "Audio evaluation skipped for auto-cut scenario", {}, {}

        path = Path(audio_path)
        if not path.exists():
            logger.error("Audio file not found for evaluation: %s", path)
            return None, "Audio file not found", {}, {}

        try:
            with open(path, "rb") as f:
                b64_audio = base64.b64encode(f.read()).decode("utf-8")
        except Exception as exc:
            logger.exception("Failed to read audio file")
            return None, f"Failed to read audio file: {exc}", {}, {}

        param_list = "\n".join(
            f'  "{p}": <0.0 to 1.0>' for p in _AUDIO_PARAMS
        )

        prompt = f"""Evaluate the acoustic and audio quality of the outbound agent in this recording.

Scenario: {scenario.name}
{scenario.description}

Score EACH of the following audio parameters individually from 0.0 to 1.0:

  pronunciation_accuracy
    — Were all Hindi and English words (including proper nouns and addresses)
      pronounced correctly? Check accuracy of Hinglish words, Indian names,
      and alphanumeric strings (phone numbers, pin codes).

  code_switching_fluency
    — Did the agent switch naturally between Hindi and English at semantically
      appropriate moments? A fluent switch happens at clause or phrase
      boundaries — not mid-word or mid-phrase. Score separately from
      pronunciation: an agent can pronounce each word correctly but switch
      at the wrong moment.

  pacing_and_fluency
    — Was the speaking cadence natural and human-like? Penalise: robotic
      monotone, rushed delivery during data readbacks, unnatural mid-sentence
      pauses longer than 1 second, or staccato word-by-word delivery.

  tone_and_empathy
    — Did the agent maintain warmth and professionalism throughout? Did the
      agent respond calmly to a frustrated or hostile caller without matching
      their negative tone or becoming cold and mechanical?

  audio_overlap_interruption_handling
    — Did the agent wait for the caller to finish speaking before responding?
      Penalise any audible overlap where the agent begins speaking while the
      caller is still mid-utterance. This is an acoustic signal — listen for
      two voices simultaneously.

Respond with JSON only (no markdown fences):
{{
  "score": <overall average 0.0 to 1.0>,
  "passed": <true if score >= 0.6>,
  "summary": "<one paragraph focusing purely on audio quality>",
  "parameter_scores": {{
{param_list}
  }},
  "strengths": ["..."],
  "issues": ["..."]
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=_AUDIO_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": b64_audio,
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
            )

            raw = response.choices[0].message.content or "{}"
            raw = _strip_json_fences(raw)
            data = json.loads(raw)

            param_scores: dict[str, float] = {}
            raw_params = data.get("parameter_scores", {})
            for p in _AUDIO_PARAMS:
                try:
                    param_scores[p] = float(raw_params.get(p, 0.0))
                except (TypeError, ValueError):
                    param_scores[p] = 0.0

            overall = (
                sum(param_scores.values()) / len(param_scores)
                if param_scores else float(data.get("score", 0.0))
            )

            return overall, str(data.get("summary", "")), data, param_scores

        except Exception as exc:
            logger.exception("LLM audio evaluation failed")
            return None, f"LLM audio evaluation failed: {exc}", {}, {}

# -----------------------------------------------------------------------------
# Markdown Report Generator
# -----------------------------------------------------------------------------

def generate_markdown_report(res: dict) -> str:
    """Generates a highly attractive Markdown report for a test run."""
    scenario_id = res.get("scenario_id", "Unknown")
    ts = res.get("started_at", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    eval_data = res.get("evaluation") or {}
    is_pass = eval_data.get("passed")
    label = "? PASS" if is_pass else ("? FAIL" if is_pass is False else "? N/A")
    status = res.get('status', 'Unknown')
    
    md = [f"# ?? Call Report: {scenario_id}"]
    md.append(f"**Date:** `{ts}`")
    md.append(f"**Status:** {label} `{status}`")
    md.append(f"**Turns:** `{res.get('turn_count', 0)}`")
    
    if res.get("duration_sec"):
        md.append(f"**Duration:** `{res['duration_sec']:.1f}s`")
    if res.get("avg_latency_ms"):
        md.append(f"**Avg Latency:** `{res['avg_latency_ms']:.0f}ms`")
    
    md.append("\n---\n")
    
    if eval_data:
        md.append("## ?? Evaluation Metrics")
        
        # Semantic
        md.append("\n### ?? Semantic Evaluation (LLM)")
        if eval_data.get("llm_score") is not None:
            md.append(f"**Score:** `{eval_data['llm_score']}/1.0`")
            md.append(f"**Summary:** {eval_data.get('llm_summary', '')}\n")
            
            semantic_scores = eval_data.get('semantic_scores', {})
            if semantic_scores:
                md.append("**Detailed Parameters:**")
                for k, v in semantic_scores.items():
                    md.append(f"- {k.replace('_', ' ').title()}: `{v}`")
                md.append("")
            
            llm_details = eval_data.get('llm_details', {})
            if llm_details.get('strengths'):
                md.append("**Strengths:**")
                for s in llm_details['strengths']: md.append(f"- ? {s}")
                md.append("")
            
            if llm_details.get('issues'):
                md.append("**Issues:**")
                for i in llm_details['issues']: md.append(f"- ?? {i}")
                md.append("")
        else:
            md.append("*No Semantic Evaluation Available.*\n")
            
        # Acoustic
        md.append("\n### ??? Acoustic Evaluation (Audio)")
        if eval_data.get("audio_score") is not None:
            md.append(f"**Score:** `{eval_data['audio_score']}/1.0`")
            md.append(f"**Summary:** {eval_data.get('audio_summary', '')}\n")
            
            acoustic_scores = eval_data.get('acoustic_scores', {})
            if acoustic_scores:
                md.append("**Detailed Parameters:**")
                for k, v in acoustic_scores.items():
                    md.append(f"- {k.replace('_', ' ').title()}: `{v}`")
                md.append("")
                
            audio_details = eval_data.get('audio_details', {})
            if audio_details.get('strengths'):
                md.append("**Strengths:**")
                for s in audio_details['strengths']: md.append(f"- ? {s}")
                md.append("")
                
            if audio_details.get('issues'):
                md.append("**Issues:**")
                for i in audio_details['issues']: md.append(f"- ?? {i}")
                md.append("")
        else:
            md.append("*No Acoustic Evaluation Available.*\n")
            
        # Rules (Gates and Flags)
        rules = eval_data.get("rule_results", [])
        if rules:
            md.append("\n### ?? Objective Rule Checks")
            for rule in rules:
                rmark = "?" if rule.get("passed") else "?"
                md.append(f"- {rmark} **{rule.get('id')}**: {rule.get('detail')}")
        
    md.append("\n---\n")
    md.append("## ?? Transcript")
    
    transcript = res.get("transcript", [])
    if transcript:
        for msg in transcript:
            role = msg.get('role', '')
            text = msg.get('text', '')
            timing = ""
            
            if role.lower() == 'user':
                t_ms = msg.get('tester_response_ms')
                if t_ms is not None:
                    llm = msg.get('llm_ms')
                    tts = msg.get('tts_ms')
                    if llm is not None and tts is not None and tts > 0:
                        timing = f" `[{t_ms:.0f}ms] (TTFT: {llm:.0f}ms, TTFA: {tts:.0f}ms)`"
                    else:
                        timing = f" `[{t_ms:.0f}ms]`"
                md.append(f"\n**?? Persona:** {text}{timing}")
            elif role.lower() == 'agent':
                o_ms = msg.get('outbound_response_ms')
                if o_ms is not None:
                    timing = f" `[{o_ms:.0f}ms]`"
                md.append(f"\n**?? Agent:** {text}{timing}")
            else:
                md.append(f"\n**{role.title()}:** {text}")
    else:
        md.append("\n*No transcript available.*")
        
    return "\n".join(md)

