"""Runtime settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Client configuration for talking to a remote vLLM server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="not-needed", alias="VLLM_API_KEY")
    vllm_base_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        alias="VLLM_BASE_MODEL",
    )
    vllm_judge_models: str = Field(
        default="flame,butter,chaos",
        alias="VLLM_JUDGE_MODELS",
    )
    request_timeout_s: float = Field(default=120.0, alias="REQUEST_TIMEOUT_S")
    max_concurrency: int = Field(default=16, alias="MAX_CONCURRENCY")

    def judge_names(self) -> list[str]:
        """Parse the comma-separated LoRA model names exposed by vLLM."""
        return [name.strip() for name in self.vllm_judge_models.split(",") if name.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached settings instance."""
    return Settings()
