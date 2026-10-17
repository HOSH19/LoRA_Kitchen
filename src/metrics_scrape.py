"""Scrape a subset of vLLM Prometheus metrics for the demo UI."""

from __future__ import annotations

import re

import httpx

from src.config import get_settings

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
)

INTERESTING = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:num_requests_running",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
)


def metrics_url() -> str:
    """Map the OpenAI base URL to vLLM's /metrics endpoint."""
    base = get_settings().vllm_base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    return f"{root}/metrics"


def fetch_metrics_text() -> str:
    """Download raw Prometheus text exposition."""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(metrics_url())
        response.raise_for_status()
        return response.text


def parse_interesting_metrics(text: str) -> dict[str, float]:
    """Keep a small dict of demo-relevant gauge/counter values."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name in INTERESTING and name not in found:
            found[name] = float(match.group("value"))
    return found


def snapshot_metrics() -> dict[str, float]:
    """Fetch and parse metrics, returning {} if the server is unreachable."""
    try:
        return parse_interesting_metrics(fetch_metrics_text())
    except Exception:
        return {}
