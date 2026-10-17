"""Fan-out a dish to every LoRA judge and gather results."""

from __future__ import annotations

import asyncio

from src.client import build_async_client, build_sync_client
from src.judge import judge_async, judge_sync
from src.personas import all_personas
from src.schemas import DishSubmission, PanelResult, Persona


def run_panel_sync(dish: DishSubmission, personas: list[Persona] | None = None) -> PanelResult:
    """Judge a dish sequentially (useful for debugging)."""
    client = build_sync_client()
    selected = personas or all_personas()
    cards = [judge_sync(client, persona, dish) for persona in selected]
    return PanelResult(dish=dish, cards=cards)


async def run_panel_async(
    dish: DishSubmission,
    personas: list[Persona] | None = None,
) -> PanelResult:
    """Judge a dish concurrently across LoRA model names."""
    client = build_async_client()
    selected = personas or all_personas()
    tasks = [judge_async(client, persona, dish) for persona in selected]
    cards = await asyncio.gather(*tasks)
    return PanelResult(dish=dish, cards=list(cards))


def run_panel(dish: DishSubmission, personas: list[Persona] | None = None) -> PanelResult:
    """Sync entrypoint that uses concurrent fan-out under the hood."""
    return asyncio.run(run_panel_async(dish, personas=personas))
