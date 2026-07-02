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
    """One outbound agent reply (text), optionally corrected."""

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

    def add_complete_agent_turn(self, text: str, *, corrected: bool = False) -> None:
        """Add a fully assembled agent reply to the transcript."""
        entry: dict[str, str] = {"role": "agent", "text": text}
        if corrected:
            entry["corrected"] = "true"
        self.transcript.append(entry)


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
        last_agent_transcript_index: int | None = None

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
                            # Start of a new turn
                            current_event_id = eid
                            current_response_parts = [text]
                            current_was_corrected = False
                            self._session.add_complete_agent_turn(text, corrected=False)
                            last_agent_transcript_index = len(self._session.transcript) - 1
                            await self._pending_agent_turns.put(
                                AgentTurn(text=text, event_id=eid)
                            )
                        else:
                            # Continuation of the current turn
                            current_response_parts.append(text)
                            if last_agent_transcript_index is not None:
                                full_text = "".join(current_response_parts).strip()
                                self._session.transcript[last_agent_transcript_index]["text"] = full_text

                elif event_type == "agent_response_correction":
                    correction = event.get("agent_response_correction_event", {})
                    corrected = correction.get("corrected_agent_response", "")
                    if corrected:
                        current_response_parts = [corrected]
                        current_was_corrected = True
                        if last_agent_transcript_index is not None:
                            self._session.transcript[last_agent_transcript_index]["text"] = corrected
                            self._session.transcript[last_agent_transcript_index]["corrected"] = "true"

                elif event_type == "agent_response_complete":
                    complete = event.get("agent_response_complete_event", {})
                    eid = complete.get("event_id")
                    full_text = "".join(current_response_parts).strip()
                    if full_text and last_agent_transcript_index is not None:
                        self._session.transcript[last_agent_transcript_index]["text"] = full_text
                        if current_was_corrected:
                            self._session.transcript[last_agent_transcript_index]["corrected"] = "true"

                    current_response_parts = []
                    current_event_id = None
                    current_was_corrected = False
                    last_agent_transcript_index = None

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
