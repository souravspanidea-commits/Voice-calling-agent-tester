"""Standalone ElevenLabs TTS client using the REST API (no LiveKit).

Supports outputting in the format the ConvAI agent actually expects
(detected from ``conversation_initiation_metadata``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import websockets
from typing import Any, AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

# µ-law 8kHz mono → 8 000 bytes per second (1 byte/sample)
# ~100ms chunk → 800 bytes
_ULAW_CHUNK_BYTES = 800

# PCM 16kHz 16-bit mono → 32 000 bytes per second
# ~100ms chunk → 3 200 bytes
_PCM16_CHUNK_BYTES = 3_200

# -----------------------------------------------------------------------
# Pure-Python PCM-16k → µ-law-8k conversion (no audioop — removed in 3.13)
# -----------------------------------------------------------------------

# µ-law encoding lookup: maps 14-bit linear magnitude → 8-bit µ-law byte.
# ITU-T G.711 standard with bias = 0x84 (132).
_ULAW_BIAS = 0x84
_ULAW_CLIP = 0x7F7B  # max magnitude before clipping

# Pre-computed segment table for µ-law encoding
_ULAW_SEG_TABLE = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                   5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                   5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                   6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                   6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                   6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                   6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]


def _linear_to_ulaw(sample: int) -> int:
    """Encode a single 16-bit signed PCM sample to µ-law (ITU-T G.711)."""
    # Get sign bit
    sign = (sample >> 8) & 0x80
    if sign:
        sample = -sample
    if sample > _ULAW_CLIP:
        sample = _ULAW_CLIP
    sample += _ULAW_BIAS

    exponent = _ULAW_SEG_TABLE[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw_byte


def _downsample_2x(pcm_data: bytes) -> bytes:
    """Downsample 16-bit PCM by 2x (16 kHz → 8 kHz) using simple averaging."""
    # Unpack all 16-bit LE signed samples
    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data[:n_samples * 2])

    # Average pairs of samples
    out = []
    for i in range(0, len(samples) - 1, 2):
        avg = (samples[i] + samples[i + 1]) // 2
        out.append(avg)

    return struct.pack(f"<{len(out)}h", *out)


def _pcm16k_to_ulaw8k(pcm16k: bytes) -> bytes:
    """Convert 16-bit PCM @ 16 kHz → µ-law @ 8 kHz (mono).

    Steps:
    1. Downsample 16 kHz → 8 kHz (average pairs)
    2. Convert each 16-bit sample to 8-bit µ-law
    """
    # Step 1: downsample 16k → 8k
    pcm8k = _downsample_2x(pcm16k)

    # Step 2: convert linear PCM to µ-law
    n_samples = len(pcm8k) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm8k[:n_samples * 2])
    ulaw_bytes = bytes(_linear_to_ulaw(s) for s in samples)
    return ulaw_bytes


class ElevenLabsTTSClient:
    """Converts text to audio via ElevenLabs TTS REST API.

    Fetches PCM 16 kHz from the API, then optionally converts to µ-law
    8 kHz to match whatever format the ConvAI agent expects.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        model_id: str = "eleven_turbo_v2_5",
        base_url: str = "https://api.elevenlabs.io",
        output_format: str = "ulaw_8000",
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.output_format = output_format  # "ulaw_8000" or "pcm_16000"
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    async def synthesize(self, text: str) -> bytes:
        """Convert *text* to audio bytes in ``self.output_format``.

        Always fetches PCM 16 kHz from the API, then converts if needed.
        """
        client = await self._ensure_client()
        url = (
            f"{self.base_url}/v1/text-to-speech/{self.voice_id}"
            f"?output_format=pcm_16000&optimize_streaming_latency=3"
        )
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        payload: dict[str, Any] = {
            "text": text,
            "model_id": self.model_id,
        }

        logger.debug("TTS request: voice=%s, text=%r", self.voice_id, text[:80])
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        pcm16k = response.content
        pcm_duration_ms = len(pcm16k) / 32.0  # 32 bytes/ms at 16kHz 16-bit mono
        logger.debug(
            "TTS response: %d bytes PCM16k (%.0f ms of audio)",
            len(pcm16k),
            pcm_duration_ms,
        )

        if self.output_format == "ulaw_8000":
            audio = _pcm16k_to_ulaw8k(pcm16k)
            logger.debug(
                "Converted to µ-law 8kHz: %d bytes (%.0f ms)",
                len(audio),
                len(audio) / 8.0,  # 8 bytes/ms at 8kHz 1-byte µ-law
            )
            return audio

        # Already pcm_16000 — return as-is
        return pcm16k

    async def synthesize_base64_chunks(
        self,
        text: str,
        chunk_size: int | None = None,
    ) -> list[str]:
        """Synthesize text and return base64 chunks for ``user_audio_chunk``.

        Default chunk size gives ~250 ms of audio per chunk.
        """
        if chunk_size is None:
            chunk_size = (
                _ULAW_CHUNK_BYTES
                if self.output_format == "ulaw_8000"
                else _PCM16_CHUNK_BYTES
            )

        audio = await self.synthesize(text)
        chunks: list[str] = []
        for offset in range(0, len(audio), chunk_size):
            chunk = audio[offset : offset + chunk_size]
            chunks.append(base64.b64encode(chunk).decode("ascii"))
        return chunks

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text into audio using the ElevenLabs WebSocket input streaming API."""
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"
            f"?model_id={self.model_id}&output_format=pcm_16000&optimize_streaming_latency=1"
        )
        
        chunk_size = _ULAW_CHUNK_BYTES if self.output_format == "ulaw_8000" else _PCM16_CHUNK_BYTES

        async with websockets.connect(url) as ws:
            # 1. Send BOS (Begin of Stream) message 
            bos = {
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": self.api_key,
            }
            await ws.send(json.dumps(bos))

            # 2. Task to consume text stream and send to WS
            async def send_text_chunks() -> None:
                try:
                    async for text_chunk in text_stream:
                        if text_chunk:
                            # Force a trailing space to tell ElevenLabs the word is finished.
                            # This prevents ElevenLabs from buffering indefinitely if a space is missing.
                            if not text_chunk.endswith(" "):
                                text_chunk += " "
                            await ws.send(json.dumps({"text": text_chunk}))
                    # Send flush message to indicate EOF
                    await ws.send(json.dumps({"text": ""}))
                except Exception as e:
                    logger.error(f"Error sending text chunks to ElevenLabs WS: {e}")

            sender_task = asyncio.create_task(send_text_chunks())

            # 3. Read audio chunks from WS and yield
            audio_buffer = bytearray()
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if "error" in data and data["error"]:
                        logger.error(f"ElevenLabs WS error: {data['error']}")
                        break

                    if data.get("audio"):
                        pcm16k = base64.b64decode(data["audio"])
                        if self.output_format == "ulaw_8000":
                            audio_buffer.extend(_pcm16k_to_ulaw8k(pcm16k))
                        else:
                            audio_buffer.extend(pcm16k)
                            
                        # Yield strictly in chunk_size limits to prevent ConvAI from dropping large payloads
                        while len(audio_buffer) >= chunk_size:
                            chunk = bytes(audio_buffer[:chunk_size])
                            del audio_buffer[:chunk_size]
                            yield chunk

                    if data.get("isFinal"):
                        if audio_buffer:
                            yield bytes(audio_buffer)
                        break
            finally:
                sender_task.cancel()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
