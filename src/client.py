"""OpenAI-compatible client pointed at a vLLM server."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from src.config import Settings, get_settings


def build_sync_client(settings: Settings | None = None) -> OpenAI:
    """Create a synchronous OpenAI SDK client for vLLM."""
    cfg = settings or get_settings()
    return OpenAI(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        timeout=cfg.request_timeout_s,
    )


def build_async_client(settings: Settings | None = None) -> AsyncOpenAI:
    """Create an async OpenAI SDK client for concurrent panel calls."""
    cfg = settings or get_settings()
    return AsyncOpenAI(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        timeout=cfg.request_timeout_s,
    )


def guided_json_body(schema: dict) -> dict:
    """Extra body enabling vLLM structured outputs for a JSON schema."""
    return {
        "guided_json": schema,
        "chat_template_kwargs": {"enable_thinking": False},
    }
