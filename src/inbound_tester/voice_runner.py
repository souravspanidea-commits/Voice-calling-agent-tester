"""Orchestrates a full audio-based test conversation (standalone, no LiveKit)."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import websockets
from typing import Any

from inbound_tester.audio_bridge import AudioBridge
from inbound_tester.audio_recorder import AudioRecorder
from inbound_tester.config import Settings
from inbound_tester.conversation_runner import RunResult
from inbound_tester.elevenlabs_ws import ElevenLabsConvAIClient
from inbound_tester.evaluator import EvaluationResult, Evaluator
from inbound_tester.logger import ConversationLogger
from inbound_tester.persona_engine import PersonaEngine
from inbound_tester.scenarios import ScenarioConfig
from inbound_tester.tts_client import ElevenLabsTTSClient

logger = logging.getLogger(__name__)


def _last_agent_text(transcript: list[dict[str, str]]) -> str | None:
    for entry in reversed(transcript):
        if entry.get("role") == "agent":
            return entry.get("text")
    return None


async def _timed_drain(bridge, *, quiet_ms: float = 800, timeout_ms: float = 5000):
    """Run audio drain and return (completion_time, drain_ms)."""
    drain_ms = await bridge.wait_for_audio_drain(quiet_ms=quiet_ms, timeout_ms=timeout_ms)
    return time.monotonic(), drain_ms


class VoiceRunner:
    """Runs a scenario in audio mode using direct ElevenLabs TTS REST + WS.

    Turn loop:
    1. Wait for ``agent_response_complete`` (via ``wait_for_agent_turn``).
    2. Persona engine generates a text reply (OpenAI).
    3. TTS client converts text → PCM 16 kHz 16-bit mono → µ-law 8 kHz.
    4. Audio bridge sends µ-law as base64 ``user_audio_chunk`` messages.
    5. Audio recorder saves everything to WAV.
    6. Repeat until exit condition.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persona = PersonaEngine(settings.openai_api_key, settings.openai_model)
        self.evaluator = Evaluator(settings.openai_api_key, settings.openai_model)

        db_path = settings.sqlite_path
        if db_path is None:
            raise ValueError("Unsupported DATABASE_URL: only sqlite:/// is supported")
        self.logger = ConversationLogger(db_path)

        # TTS client is created after connect() so we can detect the agent's
        # expected audio format from conversation_initiation_metadata.
        self.tts: ElevenLabsTTSClient | None = None

        # Audio recorder for saving the conversation
        self.recorder = AudioRecorder()

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

        bridge: AudioBridge | None = None
        recorder = self.recorder

        t0 = time.monotonic()
        t_agent_done = time.monotonic()
        first_agent_chunk_flag = True

        def on_event(event: dict[str, Any]) -> None:
            self.logger.log_ws_event(test_run_id, event)

        def on_audio(b64_audio: str) -> None:
            nonlocal first_agent_chunk_flag
            if bridge:
                bridge.on_agent_audio(b64_audio)
            # Record agent audio
            try:
                raw = base64.b64decode(b64_audio)
                if first_agent_chunk_flag:
                    latency = max(0, (time.monotonic() - t0) * 1000)
                    recorder.add_silence(int(latency))
                    first_agent_chunk_flag = False
                recorder.add_agent_audio(raw)
            except Exception:
                pass

        client = ElevenLabsConvAIClient(
            self.settings.ws_url,
            self.settings.elevenlabs_api_key,
            on_event=on_event,
            on_audio=on_audio,
        )
        bridge = AudioBridge(client)

        try:
            conversation_id = await client.connect(
                dynamic_variables=scenario.dynamic_variables or None,
            )
            logger.info("Connected (Audio Mode) — conversation_id=%s", conversation_id)

            # Start recording
            recorder.start(scenario.id, conversation_id)

            # Detect the agent's expected audio format from init metadata
            agent_input_fmt = (
                client.session.agent_output_audio_format or "ulaw_8000"
            )
            user_input_fmt = agent_input_fmt
            for ev in client.session.raw_events:
                meta = ev.get("conversation_initiation_metadata_event", {})
                if meta.get("user_input_audio_format"):
                    user_input_fmt = meta["user_input_audio_format"]
                    break

            logger.info(
                "Agent audio formats — output: %s, expects input: %s",
                agent_input_fmt,
                user_input_fmt,
            )

            # Create TTS client matching the agent's expected input format
            self.tts = ElevenLabsTTSClient(
                api_key=self.settings.elevenlabs_api_key,
                voice_id=self.settings.persona_voice_id,
                model_id=self.settings.persona_tts_model,
                base_url=self.settings.elevenlabs_tts_base,
                output_format=user_input_fmt,
            )

            # Handle opening behavior
            t_agent_done = time.monotonic()
            opening = self.persona.generate_opening(scenario)
            if opening and not opening.is_silent and opening.text:
                latency = max(0, (time.monotonic() - t_agent_done) * 1000)
                recorder.add_silence(int(latency))
                logger.info("Persona generating opening audio for: %s", opening.text)
                await self._synthesize_and_send(bridge, opening.text)
                client.session.transcript.append({"role": "user", "text": opening.text})
                turn_count += 1

            # Wait for the agent's first response
            t0 = time.monotonic()
            first_agent_chunk_flag = True
            first_agent = await client.wait_for_agent_turn(self.settings.agent_turn_timeout_sec)
            # Start drain as background task — LLM will run concurrently
            drain_task: asyncio.Task | None = asyncio.create_task(
                _timed_drain(bridge, quiet_ms=800, timeout_ms=5000)
            )
            if first_agent:
                logger.info("Agent: %s", first_agent.text[:120])

            # Main conversation loop
            while turn_count < max_turns:
                last_agent = _last_agent_text(client.session.transcript)

                # --- Tester latency: LLM decision + TTS + audio send ---
                t_tester_start = time.monotonic()
                stream_gen = self.persona.decide_next_reply_stream(
                    scenario,
                    client.session.transcript,
                    agent_last_message=last_agent,
                )

                full_text = ""
                is_silent = False
                should_end = False
                first_token_time = None
                first_audio_time = None

                # Eagerly start LLM so it runs concurrently with
                # TTS WebSocket connect and agent audio drain
                llm_queue: asyncio.Queue = asyncio.Queue()

                async def _pump_llm(gen=stream_gen):
                    try:
                        async for chunk in gen:
                            await llm_queue.put(chunk)
                    finally:
                        await llm_queue.put(None)

                llm_task = asyncio.create_task(_pump_llm())

                async def text_interceptor():
                    nonlocal full_text, is_silent, should_end, first_token_time
                    while True:
                        chunk = await llm_queue.get()
                        if chunk is None:
                            break
                        if first_token_time is None and (chunk.text or chunk.is_silent or chunk.should_end):
                            first_token_time = time.monotonic()
                        
                        full_text += chunk.text
                        if chunk.is_silent:
                            is_silent = True
                        if chunk.should_end:
                            should_end = True
                        
                        if chunk.text:
                            yield chunk.text

                # Connect to ElevenLabs TTS WS concurrently with LLM + drain
                t_tts_start = time.monotonic()
                audio_stream = self.tts.synthesize_stream(text_interceptor())

                ttft_ms = 0.0
                ttfa_ms = 0.0

                async for audio_chunk in audio_stream:
                    if first_audio_time is None:
                        # Ensure agent audio has fully drained before persona speaks
                        if drain_task is not None:
                            drain_end_time, _ = await drain_task
                            drain_task = None
                            t_agent_done = drain_end_time
                        first_audio_time = time.monotonic()
                        ttfa_ms = (first_audio_time - t_tester_start) * 1000
                        latency = max(0, (first_audio_time - t_agent_done) * 1000)
                        recorder.add_silence(int(latency))

                    recorder.add_persona_audio(audio_chunk)
                    await bridge.send_persona_audio(audio_chunk, audio_format=self.tts.output_format)

                # Ensure LLM task is cleaned up
                await llm_task

                # Handle drain if no audio was produced (silent/end-only response)
                if drain_task is not None:
                    drain_end_time, _ = await drain_task
                    drain_task = None
                    t_agent_done = drain_end_time

                if first_token_time:
                    ttft_ms = (first_token_time - t_tester_start) * 1000

                if should_end and not full_text.strip():
                    status = "completed_persona_end"
                    break

                turn_count += 1

                if is_silent:
                    logger.info("Persona [%d]: [SILENT]", turn_count)
                    recorder.add_silence(5000)
                    await bridge.send_silence(duration_ms=5000, audio_format=self.tts.output_format)
                else:
                    await bridge.send_silence(duration_ms=3000, audio_format=self.tts.output_format)
                    logger.info("Persona [%d]: %s", turn_count, full_text[:120])
                    client.session.transcript.append({"role": "user", "text": full_text})

                tts_ms = (time.monotonic() - t_tts_start) * 1000
                tester_ms = (time.monotonic() - t_tester_start) * 1000
                tester_latencies.append(tester_ms)

                # Enrich the last user transcript entry with tester timing
                if client.session.transcript and client.session.transcript[-1].get("role") == "user":
                    client.session.transcript[-1]["tester_response_ms"] = round(tester_ms, 1)
                    client.session.transcript[-1]["llm_ms"] = round(ttft_ms, 1)
                    client.session.transcript[-1]["tts_ms"] = round(ttfa_ms, 1)  # Storing TTFA here for visibility

                # --- Outbound agent latency: time until agent responds ---
                t0 = time.monotonic()
                first_agent_chunk_flag = True
                t_outbound_start = time.monotonic()
                agent_turn = await client.wait_for_agent_turn(self.settings.agent_turn_timeout_sec)
                outbound_ms = (time.monotonic() - t_outbound_start) * 1000
                # Start drain as background task — next LLM call runs concurrently
                drain_task = asyncio.create_task(
                    _timed_drain(bridge, quiet_ms=800, timeout_ms=5000)
                )
                logger.debug("Agent audio drain started (non-blocking)")
                latency_ms = (time.monotonic() - t0) * 1000
                turn_latencies.append(latency_ms)

                if not agent_turn:
                    # Await drain since we won't be sending audio next
                    if drain_task is not None:
                        drain_end_time, _ = await drain_task
                        drain_task = None
                        t_agent_done = drain_end_time
                    recorder.add_silence(int(latency_ms))
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
                logger.info("Agent [tester=%.0fms(ttft=%.0f,ttfa=%.0f) outbound=%.0fms]: %s", tester_ms, ttft_ms, ttfa_ms, outbound_ms, agent_turn.text[:120])

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
            if bridge:
                bridge.close()
            await client.close()
            if self.tts:
                await self.tts.close()

            # Save the recorded audio
            wav_path = recorder.save()
            if wav_path:
                logger.info("🔊 Conversation audio saved: %s", wav_path)

        duration_sec = time.monotonic() - start_time

        # Evaluate in a thread to avoid blocking the event loop
        evaluation: EvaluationResult | None = None
        if not skip_evaluation:
            evaluation = await asyncio.to_thread(
                self.evaluator.evaluate, scenario, client.session.transcript, audio_path=wav_path
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
        
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_id = res.get("scenario_id", "Unknown").replace(" ", "_")[:30]
        filepath = recordings_dir / f"{scenario_id}_{ts}_report.md"
        
        eval_data = res.get("evaluation") or {}
        is_pass = eval_data.get("passed")
        label = "PASS" if is_pass else ("FAIL" if is_pass is False else "N/A")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Call Report: {scenario_id}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status:** [{label}] {res.get('status')}\n")
            f.write(f"**Turns:** {res.get('turn_count')}\n")
            if res.get("duration_sec"):
                f.write(f"**Duration:** {res['duration_sec']:.1f}s\n")
            if res.get("avg_latency_ms"):
                f.write(f"**Avg Latency:** {res['avg_latency_ms']:.0f}ms\n")
                
            if eval_data:
                f.write("\n## Evaluation\n\n")
                if eval_data.get("llm_score") is not None:
                    f.write(f"- **Semantic Score:** {eval_data['llm_score']}\n")
                if eval_data.get("llm_summary"):
                    f.write(f"- **Semantic Summary:** {eval_data['llm_summary']}\n")
                if eval_data.get("audio_score") is not None:
                    f.write(f"- **Audio Score:** {eval_data['audio_score']}\n")
                if eval_data.get("audio_summary"):
                    f.write(f"- **Audio Summary:** {eval_data['audio_summary']}\n")
                    
                rules = eval_data.get("rule_results", [])
                if rules:
                    f.write("\n### Rule Checks\n")
                    for rule in rules:
                        rmark = "✅" if rule.get("passed") else "❌"
                        f.write(f"- {rmark} **{rule.get('id')}**: {rule.get('detail')}\n")
                        
            f.write("\n## Transcript\n\n")
            for entry in res.get("transcript") or []:
                role = entry.get('role', 'unknown').capitalize()
                text = entry.get('text', '')
                f.write(f"**{role}:** {text}\n\n")

    async def _synthesize_and_send(self, bridge: AudioBridge, text: str) -> None:
        """Synthesize *text* via ElevenLabs TTS and send as user_audio_chunk."""
        audio_bytes = await self.tts.synthesize(text)

        # Record persona audio before sending
        self.recorder.add_persona_audio(audio_bytes)

        await bridge.send_persona_audio(audio_bytes, audio_format=self.tts.output_format)
        # Send silence so the agent's VAD sees speech → silence transition
        await bridge.send_silence(
            duration_ms=3000,
            audio_format=self.tts.output_format,
        )
 