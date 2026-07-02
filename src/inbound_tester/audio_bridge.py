"""Bridges audio between ElevenLabs WebSocket and persona TTS (standalone, no LiveKit)."""

from __future__ import annotations

import asyncio
import base64
import logging
import time

from inbound_tester.elevenlabs_ws import ElevenLabsConvAIClient

logger = logging.getLogger(__name__)


class AudioBridge:
    """Buffers incoming agent audio and sends persona audio as user_audio_chunk.

    Works entirely with raw bytes (16-bit PCM, 16 kHz, mono) and base64
    strings — no external audio framework required.
    """

    def __init__(self, ws_client: ElevenLabsConvAIClient) -> None:
        self.ws_client = ws_client
        self._incoming_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._closed = False
        self._last_agent_audio_time: float = 0.0
        self._agent_speaking = False
        # Playback estimation: ElevenLabs delivers audio ahead of real-time;
        # we track total bytes received so we can estimate when playback ends.
        self._agent_audio_first_chunk_time: float = 0.0
        self._agent_audio_total_bytes: int = 0
        self._agent_audio_bytes_per_sec: float = 8000.0  # µ-law 8 kHz default
        
        # Open mic simulation
        self._persona_speaking = False
        self._silence_task: asyncio.Task | None = None
        self._silence_audio_format = "ulaw_8000"

    # ------------------------------------------------------------------
    # Incoming agent audio (called from the WS receive loop via on_audio)
    # ------------------------------------------------------------------

    def on_agent_audio(self, audio_base64: str) -> None:
        """Buffer an incoming base64 audio chunk from the outbound agent."""
        if self._closed:
            return
        self._last_agent_audio_time = time.monotonic()
        if self._agent_audio_first_chunk_time == 0.0:
            self._agent_audio_first_chunk_time = self._last_agent_audio_time
        self._agent_speaking = True
        try:
            raw = base64.b64decode(audio_base64)
            self._agent_audio_total_bytes += len(raw)
            self._incoming_audio_queue.put_nowait(raw)
        except Exception as exc:
            logger.error("Failed to decode incoming audio: %s", exc)

    async def drain_agent_audio(self) -> bytes:
        """Return all currently queued agent audio as a single PCM buffer."""
        chunks: list[bytes] = []
        while not self._incoming_audio_queue.empty():
            try:
                chunks.append(self._incoming_audio_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Outgoing persona audio
    # ------------------------------------------------------------------

    async def wait_for_audio_drain(
        self,
        quiet_ms: float = 500,
        timeout_ms: float = 5000,
        audio_format: str = "ulaw_8000",
    ) -> float:
        """Wait until agent audio has stopped arriving for *quiet_ms*, then
        wait for the estimated remaining playback to complete.

        ElevenLabs delivers audio ahead of real-time (TTS generates faster
        than 1x), so stopping chunk arrival does not mean the audio has
        finished playing.  We track total bytes received and the time the
        first chunk arrived to compute the playback end time, then sleep
        until that point before returning.

        Returns the actual quiet duration in ms once the drain completes.
        """
        bps = 8000.0 if audio_format == "ulaw_8000" else 32000.0
        quiet_sec = quiet_ms / 1000.0
        timeout_sec = timeout_ms / 1000.0
        deadline = time.monotonic() + timeout_sec

        try:
            while time.monotonic() < deadline:
                if self._last_agent_audio_time > 0:
                    elapsed = time.monotonic() - self._last_agent_audio_time
                    if elapsed >= quiet_sec:
                        logger.debug(
                            "Audio drain complete: %.0fms since last agent audio chunk",
                            elapsed * 1000,
                        )
                        # Wait for estimated remaining playback time.
                        # If audio was delivered faster than real-time, playback
                        # is still ongoing even though chunks have stopped arriving.
                        if self._agent_audio_first_chunk_time > 0.0 and self._agent_audio_total_bytes > 0:
                            playback_end = (
                                self._agent_audio_first_chunk_time
                                + self._agent_audio_total_bytes / bps
                            )
                            extra = max(0.0, playback_end - time.monotonic())
                            if extra > 0.05:
                                logger.debug(
                                    "Waiting %.0fms for playback of %.0f audio bytes",
                                    extra * 1000,
                                    self._agent_audio_total_bytes,
                                )
                                await asyncio.sleep(extra)
                        return elapsed * 1000
                await asyncio.sleep(0.05)  # poll every 50ms

            elapsed = (time.monotonic() - self._last_agent_audio_time) * 1000 if self._last_agent_audio_time > 0 else 0
            logger.warning(
                "Audio drain timed out after %.0fms (last chunk %.0fms ago)",
                timeout_ms,
                elapsed,
            )
            return elapsed
        finally:
            self._agent_speaking = False

    async def send_persona_audio(self, pcm_bytes: bytes, audio_format: str = "ulaw_8000") -> None:
        """Base64-encode *pcm_bytes* and send as ``user_audio_chunk`` messages."""
        if self._closed:
            return

        # Reset speaking state and playback-tracking counters before sending
        # persona audio so the NEXT drain starts fresh.
        self._agent_speaking = False
        self._agent_audio_first_chunk_time = 0.0
        self._agent_audio_total_bytes = 0
        
        self.stop_silence()
        self._persona_speaking = True

        if audio_format == "ulaw_8000":
            bytes_per_sec = 8000.0
            chunk_size = 800
        else:
            bytes_per_sec = 32000.0
            chunk_size = 3200

        for offset in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[offset : offset + chunk_size]
            encoded = base64.b64encode(chunk).decode("ascii")
            await self.ws_client.send_audio_chunk(encoded)
            # Pace the audio delivery to roughly real-time
            await asyncio.sleep(len(chunk) / bytes_per_sec)

    async def send_persona_audio_b64(self, b64_chunks: list[str]) -> None:
        """Send pre-encoded base64 chunks as ``user_audio_chunk`` messages."""
        if self._closed:
            return
        self._agent_speaking = False
        self.stop_silence()
        self._persona_speaking = True
        for chunk in b64_chunks:
            await self.ws_client.send_audio_chunk(chunk)

    def start_silence(self, audio_format: str = "ulaw_8000") -> None:
        """Start streaming continuous silence to simulate an open microphone."""
        self._persona_speaking = False
        self._silence_audio_format = audio_format
        if self._silence_task is None:
            self._silence_task = asyncio.create_task(self._silence_loop())

    def stop_silence(self) -> None:
        """Stop streaming continuous silence."""
        if self._silence_task is not None:
            self._silence_task.cancel()
            self._silence_task = None

    async def _silence_loop(self) -> None:
        """Background task that continuously pumps silence frames at real-time pacing."""
        chunk_ms = 250
        try:
            while not self._closed and not self._persona_speaking:
                if self._silence_audio_format == "ulaw_8000":
                    bytes_per_chunk = 8000 * chunk_ms // 1000
                    silence_chunk = b"\xff" * bytes_per_chunk
                else:
                    bytes_per_chunk = 32000 * chunk_ms // 1000
                    silence_chunk = b"\x00" * bytes_per_chunk

                encoded = base64.b64encode(silence_chunk).decode("ascii")
                await self.ws_client.send_audio_chunk(encoded)
                await asyncio.sleep(chunk_ms / 1000.0)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Silence loop failed: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        self.stop_silence()
 