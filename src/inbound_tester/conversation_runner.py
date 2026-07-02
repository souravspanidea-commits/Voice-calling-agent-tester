"""Orchestrates a full text-based test conversation."""

from __future__ import annotations

import asyncio
import logging
import time
import websockets
from dataclasses import dataclass, field
from typing import Any

from inbound_tester.config import Settings
from inbound_tester.elevenlabs_ws import ElevenLabsConvAIClient
from inbound_tester.evaluator import EvaluationResult, Evaluator
from inbound_tester.logger import ConversationLogger
from inbound_tester.persona_engine import PersonaEngine
from inbound_tester.scenarios import ScenarioConfig

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    test_run_id: str
    scenario_id: str
    conversation_id: str | None
    status: str
    turn_count: int
    transcript: list[dict[str, str]]
    evaluation: EvaluationResult | None = None
    turn_latencies_ms: list[float] = field(default_factory=list)
    tester_latencies_ms: list[float] = field(default_factory=list)
    outbound_latencies_ms: list[float] = field(default_factory=list)
    duration_sec: float = 0.0 

    def to_dict(self) -> dict[str, Any]:
        avg_latency = (
            sum(self.turn_latencies_ms) / len(self.turn_latencies_ms)
            if self.turn_latencies_ms
            else None
        )
        avg_tester = (
            sum(self.tester_latencies_ms) / len(self.tester_latencies_ms)
            if self.tester_latencies_ms
            else None
        )
        avg_outbound = (
            sum(self.outbound_latencies_ms) / len(self.outbound_latencies_ms)
            if self.outbound_latencies_ms
            else None
        )
        return {
            "test_run_id": self.test_run_id,
            "scenario_id": self.scenario_id,
            "conversation_id": self.conversation_id,
            "status": self.status,
            "turn_count": self.turn_count,
            "transcript": self.transcript,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
            "avg_tester_latency_ms": round(avg_tester, 1) if avg_tester else None,
            "avg_outbound_latency_ms": round(avg_outbound, 1) if avg_outbound else None,
            "duration_sec": round(self.duration_sec, 2),
        }


class ConversationRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persona = PersonaEngine(settings.openai_api_key, settings.openai_model)
        self.evaluator = Evaluator(settings.openai_api_key, settings.openai_model)
        db_path = settings.sqlite_path
        if db_path is None:
            raise ValueError(
                f"Unsupported DATABASE_URL: {settings.database_url!r} — only sqlite:/// is supported"
            )
        self.logger = ConversationLogger(db_path)

    async def run_scenario(
        self,
        scenario: ScenarioConfig,
        *,
        skip_evaluation: bool = False,
    ) -> RunResult:
        if not self.settings.elevenlabs_api_key or not self.settings.elevenlabs_agent_id:
            raise ValueError("ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID are required")

        test_run_id = self.logger.start_run(scenario.id, self.settings.elevenlabs_agent_id)
        max_turns = scenario.max_turns or self.settings.max_turns
        turn_count = 0
        consecutive_timeouts = 0
        turn_latencies: list[float] = []
        tester_latencies: list[float] = []
        outbound_latencies: list[float] = []
        status = "completed"
        start_time = time.monotonic()

        def on_event(event: dict[str, Any]) -> None:
            self.logger.log_ws_event(test_run_id, event)

        client = ElevenLabsConvAIClient(
            self.settings.ws_url,
            self.settings.elevenlabs_api_key,
            on_event=on_event,
        )

        try:
            conversation_id = await client.connect(
                dynamic_variables=scenario.dynamic_variables or None,
            )
            logger.info("Connected — conversation_id=%s", conversation_id)

            opening = self.persona.generate_opening(scenario)
            if opening and not opening.is_silent and opening.text:
                await client.send_user_message(opening.text)
                turn_count += 1

            first_agent = await client.wait_for_agent_turn(self.settings.agent_turn_timeout_sec)
            if first_agent:
                logger.info("Agent: %s", first_agent.text[:120])

            while turn_count < max_turns:
                last_agent = _last_agent_text(client.session.transcript)

                # --- Tester latency: persona LLM decision + message send ---
                t_tester_start = time.monotonic()
                decision = await self.persona.decide_next_reply(
                    scenario,
                    client.session.transcript,
                    agent_last_message=last_agent,
                )

                if decision.should_end:
                    status = "completed_persona_end"
                    break

                t0 = time.monotonic()
                turn_count += 1

                if decision.is_silent:
                    logger.info("Persona [%d]: [SILENT]", turn_count)
                    await asyncio.sleep(5.0) # simulate 5s silence for text mode
                else:
                    logger.info("Persona [%d]: %s", turn_count, decision.text[:120])
                    await client.send_user_message(decision.text)

                tester_ms = (time.monotonic() - t_tester_start) * 1000
                tester_latencies.append(tester_ms)

                # Enrich the last user transcript entry with tester timing
                if client.session.transcript and client.session.transcript[-1].get("role") == "user":
                    client.session.transcript[-1]["tester_response_ms"] = round(tester_ms, 1)

                # --- Outbound agent latency: time until agent responds ---
                t_outbound_start = time.monotonic()
                agent_turn = await client.wait_for_agent_turn(self.settings.agent_turn_timeout_sec)
                outbound_ms = (time.monotonic() - t_outbound_start) * 1000
                latency_ms = (time.monotonic() - t0) * 1000
                turn_latencies.append(latency_ms)

                if not agent_turn:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 3:
                        status = "timeout_waiting_for_agent"
                        logger.warning("Agent silent for %d seconds. Ending test.", consecutive_timeouts * int(self.settings.agent_turn_timeout_sec))
                        break
                    else:
                        logger.warning("Agent silent for %d seconds. Triggering proactive persona reply.", int(self.settings.agent_turn_timeout_sec))
                        client.session.transcript.append({"role": "agent", "text": f"[SILENCE] (The agent has not spoken for {int(self.settings.agent_turn_timeout_sec)} seconds)"})
                        continue

                outbound_latencies.append(outbound_ms)
                consecutive_timeouts = 0
                logger.info("Agent [tester=%.0fms outbound=%.0fms total=%.0fms]: %s", tester_ms, outbound_ms, latency_ms, agent_turn.text[:120])

                # Enrich the last agent transcript entry with outbound timing
                if client.session.transcript and client.session.transcript[-1].get("role") == "agent":
                    client.session.transcript[-1]["outbound_response_ms"] = round(outbound_ms, 1)

            else:
                status = "max_turns_reached"

        except websockets.ConnectionClosedOK:
            status = "agent_ended_call"
            logger.info("Agent ended the call gracefully (1000 OK).")
        except Exception as exc:
            status = "error"
            logger.exception("Conversation failed: %s", exc)
            raise
        finally:
            await client.close()

        duration_sec = time.monotonic() - start_time

        # Run evaluator in a thread to avoid blocking the event loop (Bug 4)
        evaluation: EvaluationResult | None = None
        if not skip_evaluation:
            evaluation = await asyncio.to_thread(
                self.evaluator.evaluate, 
                scenario, 
                client.session.transcript,
                outbound_latencies_ms=outbound_latencies
            )

        avg_latency = (
            sum(turn_latencies) / len(turn_latencies) if turn_latencies else None
        )
        avg_tester = (
            sum(tester_latencies) / len(tester_latencies) if tester_latencies else None
        )
        avg_outbound = (
            sum(outbound_latencies) / len(outbound_latencies) if outbound_latencies else None
        )

        self.logger.finish_run(
            test_run_id,
            status=status,
            conversation_id=client.session.conversation_id,
            turn_count=turn_count,
            transcript=client.session.transcript,
            evaluation=evaluation.to_dict() if evaluation else None,
            duration_sec=duration_sec,
            avg_latency_ms=avg_latency,
            avg_tester_latency_ms=avg_tester,
            avg_outbound_latency_ms=avg_outbound,
        )

        res = RunResult(
            test_run_id=test_run_id,
            scenario_id=scenario.id,
            conversation_id=client.session.conversation_id,
            status=status,
            turn_count=turn_count,
            transcript=client.session.transcript,
            evaluation=evaluation,
            turn_latencies_ms=turn_latencies,
            tester_latencies_ms=tester_latencies,
            outbound_latencies_ms=outbound_latencies,
            duration_sec=duration_sec,
        )
        
        self._save_single_call_report(res.to_dict())
        return res

    def _save_single_call_report(self, res: dict) -> None:
        from datetime import datetime
        from pathlib import Path
        from inbound_tester.evaluator import generate_markdown_report
        
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_id = res.get("scenario_id", "Unknown").replace(" ", "_")[:30]
        filepath = recordings_dir / f"{scenario_id}_{ts}_report.md"
        
        md_content = generate_markdown_report(res)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
                        
            f.write("\n## Transcript\n\n")
            for entry in res.get("transcript") or []:
                role = entry.get('role', 'unknown').capitalize()
                text = entry.get('text', '')
                f.write(f"**{role}:** {text}\n\n")


def _last_agent_text(transcript: list[dict[str, str]]) -> str | None:
    for entry in reversed(transcript):
        if entry.get("role") == "agent":
            return entry.get("text")
    return None


def run_scenario_sync(scenario: ScenarioConfig, settings: Settings | None = None, **kwargs: Any) -> RunResult:
    settings = settings or Settings()
    runner = ConversationRunner(settings)
    return asyncio.run(runner.run_scenario(scenario, **kwargs))

