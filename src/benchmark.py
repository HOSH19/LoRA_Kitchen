#!/usr/bin/env python3
"""Benchmark sequential vs concurrent multi-LoRA panel traffic."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client import build_async_client
from src.config import get_settings
from src.dishes import load_dishes
from src.judge import judge_async
from src.paths import ensure_results_dir
from src.personas import all_personas
from src.schemas import DishSubmission, Persona


def parse_args() -> argparse.Namespace:
    """CLI for wall-clock serving benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=None)
    return parser.parse_args()


async def _timed_judge(client, persona: Persona, dish: DishSubmission) -> float:
    """Return latency seconds for one judge call."""
    started = time.perf_counter()
    await judge_async(client, persona, dish)
    return time.perf_counter() - started


async def run_sequential(dishes: list[DishSubmission]) -> dict:
    """Fire every dish × persona call one after another."""
    client = build_async_client()
    latencies: list[float] = []
    started = time.perf_counter()
    for dish in dishes:
        for persona in all_personas():
            latencies.append(await _timed_judge(client, persona, dish))
    return _summary("sequential", started, latencies, len(dishes))


async def run_concurrent(dishes: list[DishSubmission], concurrency: int) -> dict:
    """Fire mixed multi-LoRA traffic with a concurrency cap."""
    client = build_async_client()
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def _one(persona: Persona, dish: DishSubmission) -> None:
        """Bounded worker for a single judge request."""
        async with semaphore:
            latency = await _timed_judge(client, persona, dish)
            latencies.append(latency)

    started = time.perf_counter()
    await asyncio.gather(
        *[_one(persona, dish) for dish in dishes for persona in all_personas()]
    )
    return _summary("concurrent_multilora", started, latencies, len(dishes))


def _summary(mode: str, started: float, latencies: list[float], n_dishes: int) -> dict:
    """Build a JSON-serializable benchmark row."""
    wall = time.perf_counter() - started
    n_calls = len(latencies)
    return {
        "mode": mode,
        "dishes": n_dishes,
        "calls": n_calls,
        "wall_seconds": wall,
        "calls_per_second": (n_calls / wall) if wall else 0.0,
        "p50_latency_s": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_s": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
        ),
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
    }


async def main_async(limit: int, concurrency: int) -> dict:
    """Run both benchmark modes and return the combined report."""
    dishes = load_dishes()[:limit]
    sequential = await run_sequential(dishes)
    concurrent = await run_concurrent(dishes, concurrency)
    return {
        "base_url": get_settings().vllm_base_url,
        "sequential": sequential,
        "concurrent": concurrent,
        "speedup_wall": (
            sequential["wall_seconds"] / concurrent["wall_seconds"]
            if concurrent["wall_seconds"]
            else None
        ),
    }


def main() -> None:
    """Entrypoint writing results/benchmark.json."""
    args = parse_args()
    concurrency = args.concurrency or get_settings().max_concurrency
    report = asyncio.run(main_async(args.limit, concurrency))
    out = ensure_results_dir() / "benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
