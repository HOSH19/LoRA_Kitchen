"""Lightweight probes against the remote vLLM OpenAI server."""

from __future__ import annotations

import httpx

from src.config import get_settings


def models_url() -> str:
    """Derive the /models endpoint from the configured base URL."""
    base = get_settings().vllm_base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def check_server() -> dict:
    """Return server model listing or raise on connection failure."""
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}
    with httpx.Client(timeout=10.0) as client:
        response = client.get(models_url(), headers=headers)
        response.raise_for_status()
        return response.json()


def listed_model_ids() -> list[str]:
    """Extract model ids currently advertised by vLLM."""
    payload = check_server()
    return [item["id"] for item in payload.get("data", [])]
