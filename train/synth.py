"""Orchestrate synthetic chat rows for LoRA SFT."""

from __future__ import annotations

import hashlib
from typing import Any

from src.prompts import build_messages
from src.schemas import DishSubmission, JudgeCard, Persona
from train.synth_dishes import make_dish
from train.synth_labels import make_card


def make_chat_row(persona: Persona, dish: DishSubmission, card: JudgeCard) -> dict[str, Any]:
    """Pack one supervised chat example for TRL/PEFT training."""
    messages = build_messages(persona, dish)
    messages.append({"role": "assistant", "content": card.model_dump_json()})
    return {
        "messages": messages,
        "persona": persona.value,
        "dish": dish.model_dump(),
        "gold": card.model_dump(),
    }


def generate_split(persona: Persona, count: int, seed_prefix: str) -> list[dict[str, Any]]:
    """Generate a list of chat rows for one persona split."""
    rows: list[dict[str, Any]] = []
    offset = int(hashlib.sha256(seed_prefix.encode("utf-8")).hexdigest()[:8], 16) % 97
    for index in range(count):
        dish = make_dish(index + offset)
        card = make_card(persona, dish, f"{seed_prefix}-{persona.value}-{index}")
        rows.append(make_chat_row(persona, dish, card))
    return rows


__all__ = ["generate_split", "make_card", "make_chat_row", "make_dish"]
