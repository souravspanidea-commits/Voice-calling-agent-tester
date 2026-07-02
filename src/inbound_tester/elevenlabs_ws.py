"""ElevenLabs ConvAI WebSocket client for text-based test conversations."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    """One outbound agent reply (text), optionally corrected.

    ``text`` is updated in-place as additional ``agent_response`` chunks
    arrive so that callers always read the latest accumulated text.
    """

    text: str
    event_id: int | None = None
    corrected_from: str | None = None


@dataclass
class ConversationSession:
    conversation_id: str | None = None
    agent_output_audio_format: str | None = None
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)

    def add_event(self, event: dict[str, Any]) -> None:
        self.raw_events.append(event)
        event_type = event.get("type")

        if event_type == "conversation_initiation_metadata":
            meta = event.get("conversation_initiation_metadata_event", {})
            self.conversation_id = meta.get("conversation_id")
            self.agent_output_audio_format = meta.get("agent_output_audio_format")

        elif event_type == "user_transcript":
            text = event.get("user_transcription_event", {}).get("user_transcript", "")
            if text:
                self.transcript.append({"role": "user_heard", "text": text})

        # NOTE: agent_response and agent_response_correction are NOT added
        # to the transcript here. They are accumulated in the receive loop
        # and a single complete entry is appended when agent_response_complete
        # fires — see ElevenLabsConvAIClient._receive_loop().

    def add_complete_agent_turn(self, text: str, *, corrected: bool = False) -> dict[str, str]:
        """Add a fully assembled agent reply to the transcript.

        Returns the dict entry so callers can update it in-place without
        relying on a fragile integer index.
        """
        entry: dict[str, str] = {"role": "agent", "text": text}
        if corrected:
            entry["corrected"] = "true"
        self.transcript.append(entry)
        return entry


class ElevenLabsConvAIClient:
    """Connects to the outbound ElevenLabs agent over WebSocket."""

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_audio: Callable[[str], None] | None = None,
    ) -> None:
        self.ws_url = ws_url
        self.api_key = api_key
        self.on_event = on_event
        self.on_audio = on_audio
        self._ws: WebSocketClientProtocol | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._pending_agent_turns: asyncio.Queue[AgentTurn] = asyncio.Queue()
        self._session = ConversationSession()
        self._closed = asyncio.Event()

    @property
    def session(self) -> ConversationSession:
        return self._session

    @property
    def is_closed(self) -> bool:
        return self._closed.is_set()

    async def connect(
        self,
        *,
        dynamic_variables: dict[str, str | int | float | bool] | None = None,
        conversation_config_override: dict[str, Any] | None = None,
    ) -> str:
        headers = {"xi-api-key": self.api_key}
        self._ws = await websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=None,
        )
        self._receive_task = asyncio.create_task(self._receive_loop())

        init_payload: dict[str, Any] = {"type": "conversation_initiation_client_data"}
        if dynamic_variables:
            init_payload["dynamic_variables"] = dynamic_variables
        if conversation_config_override:
            init_payload["conversation_config_override"] = conversation_config_override

        await self._send(init_payload)
        conversation_id = await self._wait_for_init(timeout=30.0)
        return conversation_id

    async def send_user_message(self, text: str) -> None:
        await self._send({"type": "user_message", "text": text})
        self._session.transcript.append({"role": "user", "text": text})

    async def send_audio_chunk(self, audio_base64: str) -> None:
        """Send base64 encoded audio to the outbound agent."""
        await self._send({"user_audio_chunk": audio_base64})

    async def wait_for_agent_turn(self, timeout: float) -> AgentTurn | None:
        try:
            return await asyncio.wait_for(self._pending_agent_turns.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._receive_task:
            try:
                await asyncio.wait_for(self._receive_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            self._receive_task = None
        self._closed.set()

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket is not connected")
        await self._ws.send(json.dumps(payload))

    async def _wait_for_init(self, timeout: float) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._session.conversation_id:
                return self._session.conversation_id
            await asyncio.sleep(0.05)
        raise TimeoutError("Timed out waiting for conversation_initiation_metadata")

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        current_response_parts: list[str] = []
        current_event_id: int | None = None
        current_was_corrected: bool = False
        current_agent_turn: AgentTurn | None = None
        # Direct reference to the transcript dict for the active agent turn.
        # Using a dict reference (not an integer index) so that appends of
        # user_transcript events mid-stream don't silently invalidate the index.
        current_agent_transcript_entry: dict[str, str] | None = None

        try:
            async for raw in self._ws:
                event = json.loads(raw)
                self._session.add_event(event)
                if self.on_event:
                    self.on_event(event)

                event_type = event.get("type")
                logger.debug("WS event: %s", event_type)

                if event_type == "ping":
                    ping_event = event.get("ping_event", {})
                    await self._send(
                        {
                            "type": "pong",
                            "event_id": ping_event.get("event_id"),
                        }
                    )

                elif event_type == "audio":
                    audio_b64 = event.get("audio_event", {}).get("audio_base_64")
                    if audio_b64 and self.on_audio:
                        self.on_audio(audio_b64)

                elif event_type == "agent_response":
                    text = event.get("agent_response_event", {}).get("agent_response", "")
                    eid = event.get("agent_response_event", {}).get("event_id")
                    if text:
                        if current_event_id != eid:
                            # Check if the last actual turn (ignoring hallucinatory 'user_heard' noise) was this agent turn
                            is_continuous = False
                            if current_agent_turn is not None and current_agent_transcript_entry is not None:
                                for entry in reversed(self._session.transcript):
                                    if entry.get("role") == "user_heard":
                                        continue
                                    if entry is current_agent_transcript_entry:
                                        is_continuous = True
                                    break

                            if is_continuous:
                                # Continuous speech split across multiple event_ids.
                                # Append to the existing turn instead of queuing a new one.
                                current_event_id = eid
                                prefix = " " if current_response_parts and not current_response_parts[-1].endswith((" ", "\n")) else ""
                                current_response_parts.append(prefix + text)
                                full_text = "".join(current_response_parts).strip()
                                if current_agent_turn is not None:
                                    current_agent_turn.text = full_text
                                if current_agent_transcript_entry is not None:
                                    current_agent_transcript_entry["text"] = full_text
                            else:
                                # Start of a genuinely new turn
                                current_event_id = eid
                                current_response_parts = [text]
                                current_was_corrected = False
                                current_agent_transcript_entry = self._session.add_complete_agent_turn(
                                    text, corrected=False
                                )
                                current_agent_turn = AgentTurn(text=text, event_id=eid)
                                await self._pending_agent_turns.put(current_agent_turn)
                        else:
                            # Continuation of the current turn — update the
                            # already-queued AgentTurn object and transcript entry
                            # in-place so voice_runner always sees the latest text.
                            current_response_parts.append(text)
                            full_text = "".join(current_response_parts).strip()
                            if current_agent_turn is not None:
                                current_agent_turn.text = full_text
                            if current_agent_transcript_entry is not None:
                                current_agent_transcript_entry["text"] = full_text

                elif event_type == "agent_response_correction":
                    correction = event.get("agent_response_correction_event", {})
                    corrected = correction.get("corrected_agent_response", "")
                    if corrected:
                        current_text = "".join(current_response_parts)
                        # ElevenLabs sends a "correction" when the agent is interrupted
                        # mid-speech: the corrected text is SHORTER than what was
                        # generated and ends with "..." to mark the cut-off point.
                        # We skip these interruption corrections and keep the longer,
                        # more complete text that the agent intended to say.
                        is_interruption = (
                            corrected.rstrip().endswith("...")
                            and len(corrected) <= len(current_text)
                        )
                        if is_interruption:
                            logger.debug(
                                "Skipping interruption correction (kept %d chars over %d)",
                                len(current_text), len(corrected),
                            )
                        else:
                            # Genuine LLM correction — apply it
                            current_response_parts = [corrected]
                            current_was_corrected = True
                            if current_agent_turn is not None:
                                current_agent_turn.text = corrected
                            if current_agent_transcript_entry is not None:
                                current_agent_transcript_entry["text"] = corrected
                                current_agent_transcript_entry["corrected"] = "true"

                elif event_type == "agent_response_complete":
                    complete = event.get("agent_response_complete_event", {})
                    eid = complete.get("event_id")
                    full_text = "".join(current_response_parts).strip()
                    # Final authoritative update to both the live AgentTurn and transcript entry
                    if full_text:
                        if current_agent_turn is not None:
                            current_agent_turn.text = full_text
                        if current_agent_transcript_entry is not None:
                            current_agent_transcript_entry["text"] = full_text
                            if current_was_corrected:
                                current_agent_transcript_entry["corrected"] = "true"

                    current_response_parts = []
                    current_event_id = None
                    current_was_corrected = False
                    current_agent_turn = None
                    current_agent_transcript_entry = None

                elif event_type == "client_tool_call":
                    tool = event.get("client_tool_call", {})
                    await self._send(
                        {
                            "type": "client_tool_result",
                            "tool_call_id": tool.get("tool_call_id"),
                            "result": "Tool not implemented in test agent",
                            "is_error": True,
                        }
                    )

        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in WebSocket receive loop")
            raise
        finally:
            # Unblock any pending waits when the connection dies
            self._closed.set()
            await self._pending_agent_turns.put(AgentTurn(text="[WEBSOCKET_CLOSED]", event_id=None))
