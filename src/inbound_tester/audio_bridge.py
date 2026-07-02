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

    # ------------------------------------------------------------------
    # Incoming agent audio (called from the WS receive loop via on_audio)
    # ------------------------------------------------------------------

    def on_agent_audio(self, audio_base64: str) -> None:
        """Buffer an incoming base64 audio chunk from the outbound agent."""
        if self._closed:
            return
        self._last_agent_audio_time = time.monotonic()
        self._agent_speaking = True
        try:
            raw = base64.b64decode(audio_base64)
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
    ) -> float:
        """Wait until agent audio has stopped arriving for *quiet_ms*.

        Returns the actual quiet duration in ms once the drain completes.
        This prevents the persona from speaking while the SUT is still
        streaming audio, which causes audible overlap.
        """
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

        self._agent_speaking = False

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
        for chunk in b64_chunks:
            await self.ws_client.send_audio_chunk(chunk)

    async def send_silence(
        self,
        duration_ms: int = 500,
        audio_format: str = "ulaw_8000",
        chunk_ms: int = 250,
    ) -> None:
        """Send silence frames to signal end-of-utterance to the agent's VAD.

        The agent needs to see a speech → silence transition to know the
        user finished talking.
        """
        if self._closed:
            return

        self._agent_speaking = False

        if audio_format == "ulaw_8000":
            # µ-law silence = 0xFF (encodes linear zero), 8000 samples/sec
            bytes_per_chunk = 8000 * chunk_ms // 1000  # 2000 bytes for 250ms
            silence_chunk = b"\xff" * bytes_per_chunk
        else:
            # PCM 16kHz 16-bit mono: silence = 0x00 bytes
            bytes_per_chunk = 32000 * chunk_ms // 1000  # 8000 bytes for 250ms
            silence_chunk = b"\x00" * bytes_per_chunk

        encoded = base64.b64encode(silence_chunk).decode("ascii")
        n_chunks = max(1, duration_ms // chunk_ms)
        for _ in range(n_chunks):
            if self._agent_speaking:
                logger.debug("Aborting send_silence because agent started speaking")
                break
            await self.ws_client.send_audio_chunk(encoded)
            # Sleep less than chunk_ms to fast-forward the VAD silence clock
            await asyncio.sleep((chunk_ms / 1000.0) * 0.25)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._closed = True
 