from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    elevenlabs_api_key: str = ""
    elevenlabs_agent_id: str = ""
    elevenlabs_ws_base: str = "wss://api.elevenlabs.io"
    elevenlabs_tts_base: str = "https://api.elevenlabs.io"

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    database_url: str = "sqlite:///./data/test_runs.db"

    max_turns: int = 30
    response_timeout_sec: float = 90.0
    agent_turn_timeout_sec: float = 15.0

    conversation_mode: str = "audio"
    persona_voice_id: str = "iWNf11sz1GrUE4ppxTOL"  # viraj voice
    persona_tts_model: str = "eleven_flash_v2_5"

    @property
    def ws_url(self) -> str:
        base = self.elevenlabs_ws_base.rstrip("/")
        return f"{base}/v1/convai/conversation?agent_id={self.elevenlabs_agent_id}"

    @property
    def sqlite_path(self) -> Path | None:
        if not self.database_url.startswith("sqlite:///"):
            return None
        raw = self.database_url.removeprefix("sqlite:///")
        return Path(raw)


def get_settings() -> Settings:
    return Settings()
