"""Records conversation audio (both agent and persona) to a WAV file.

Captures µ-law 8kHz audio from both sides in chronological order,
converts to 16-bit PCM, and saves as a standard WAV file that any
media player can open.
"""

from __future__ import annotations

import logging
import struct
import wave
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# µ-law decoding table: maps 8-bit µ-law byte → 16-bit signed PCM sample.
# ITU-T G.711 standard (inverse of encoding).
_ULAW_DECODE_TABLE: list[int] = []

def _build_ulaw_decode_table() -> list[int]:
    """Pre-compute µ-law → linear PCM decode table (256 entries)."""
    table: list[int] = []
    for i in range(256):
        # Complement the byte (µ-law is stored complemented)
        val = ~i & 0xFF
        sign = val & 0x80
        exponent = (val >> 4) & 0x07
        mantissa = val & 0x0F
        # Reconstruct the magnitude
        magnitude = ((mantissa << 3) + 0x84) << exponent
        magnitude -= 0x84
        sample = -magnitude if sign else magnitude
        # Clamp to 16-bit range
        sample = max(-32768, min(32767, sample))
        table.append(sample)
    return table

_ULAW_DECODE_TABLE = _build_ulaw_decode_table()


def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
    """Convert µ-law 8kHz bytes to 16-bit signed PCM 8kHz bytes."""
    samples = [_ULAW_DECODE_TABLE[b] for b in ulaw_data]
    return struct.pack(f"<{len(samples)}h", *samples)


class AudioRecorder:
    """Records conversation audio from both sides into a WAV file.

    Usage::

        recorder = AudioRecorder("recordings")
        recorder.start("cooperative", "conv_123")
        recorder.add_agent_audio(ulaw_bytes)   # from WS audio events
        recorder.add_persona_audio(ulaw_bytes)  # from TTS output
        recorder.save()  # writes WAV
    """

    def __init__(self, output_dir: str | Path = "recordings") -> None:
        self.output_dir = Path(output_dir)
        self._chunks: list[tuple[str, bytes]] = []  # (speaker, ulaw_bytes)
        self._scenario_id: str = ""
        self._conversation_id: str = ""
        self._started: bool = False

    def start(self, scenario_id: str, conversation_id: str) -> None:
        """Start recording a new conversation."""
        self._chunks = []
        self._scenario_id = scenario_id
        self._conversation_id = conversation_id
        self._started = True
        logger.info("Audio recorder started for %s / %s", scenario_id, conversation_id)

    def add_agent_audio(self, ulaw_bytes: bytes) -> None:
        """Record a chunk of agent audio (µ-law 8kHz)."""
        if self._started and ulaw_bytes:
            self._chunks.append(("agent", ulaw_bytes))

    def add_persona_audio(self, ulaw_bytes: bytes) -> None:
        """Record a chunk of persona audio (µ-law 8kHz)."""
        if self._started and ulaw_bytes:
            self._chunks.append(("persona", ulaw_bytes))

    def add_silence(self, duration_ms: int = 300) -> None:
        """Add a gap of silence between turns for natural playback."""
        if self._started:
            n_samples = 8000 * duration_ms // 1000
            silence = b"\xff" * n_samples  # µ-law silence
            self._chunks.append(("silence", silence))

    def save(self) -> Path | None:
        """Convert all collected audio to PCM and save as WAV.

        Returns the path to the saved WAV file, or None if no audio.
        """
        if not self._chunks:
            logger.warning("No audio recorded, skipping save")
            return None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_scenario = self._scenario_id.replace(" ", "_")[:30]
        filename = f"{safe_scenario}_{ts}.wav"
        filepath = self.output_dir / filename

        # Combine all µ-law chunks into one buffer
        all_ulaw = b"".join(chunk_bytes for _, chunk_bytes in self._chunks)

        # Convert µ-law → 16-bit PCM
        pcm_data = ulaw_to_pcm16(all_ulaw)

        # Write WAV file (8kHz, 16-bit, mono)
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)       # mono
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(8000)    # 8kHz (matches µ-law sample rate)
            wf.writeframes(pcm_data)

        duration_sec = len(all_ulaw) / 8000.0
        logger.info(
            "Saved conversation audio: %s (%.1fs, %d bytes)",
            filepath, duration_sec, len(pcm_data),
        )

        # Also save a transcript-synced metadata file
        self._save_metadata(filepath)

        return filepath

    def _save_metadata(self, wav_path: Path) -> None:
        """Save a text file alongside the WAV with speaker turn markers."""
        meta_path = wav_path.with_suffix(".txt")
        offset_ms = 0
        lines: list[str] = []
        lines.append(f"Scenario: {self._scenario_id}")
        lines.append(f"Conversation: {self._conversation_id}")
        lines.append(f"Recorded: {datetime.now().isoformat()}")
        lines.append("")

        for speaker, chunk_bytes in self._chunks:
            duration_ms = len(chunk_bytes) * 1000 // 8000
            if speaker != "silence":
                start_s = offset_ms / 1000.0
                end_s = (offset_ms + duration_ms) / 1000.0
                lines.append(f"[{start_s:.1f}s - {end_s:.1f}s] {speaker.upper()}")
            offset_ms += duration_ms

        total_s = offset_ms / 1000.0
        lines.append(f"\nTotal duration: {total_s:.1f}s")

        meta_path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("Saved audio metadata: %s", meta_path)
 