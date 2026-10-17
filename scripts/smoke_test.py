#!/usr/bin/env python3
"""CPU-only smoke checks for schemas, synth data, and parsing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse import parse_judge_card
from src.personas import all_personas
from src.schemas import DishSubmission
from train.synth import generate_split, make_card, make_dish


def test_dish_roundtrip() -> None:
    """DishSubmission should serialize and validate cleanly."""
    dish = make_dish(3)
    restored = DishSubmission.model_validate_json(dish.model_dump_json())
    assert restored.title == dish.title


def test_gold_cards_validate() -> None:
    """Synthetic gold labels must satisfy JudgeCard invariants."""
    dish = make_dish(1)
    for persona in all_personas():
        card = make_card(persona, dish, seed=f"smoke-{persona.value}")
        parse_judge_card(card.model_dump_json())


def test_generate_split_nonempty() -> None:
    """Dataset generator should emit chat rows with assistant JSON."""
    rows = generate_split(all_personas()[0], count=3, seed_prefix="smoke")
    assert len(rows) == 3
    assert rows[0]["messages"][-1]["role"] == "assistant"


def main() -> None:
    """Run smoke checks and exit non-zero on failure."""
    test_dish_roundtrip()
    test_gold_cards_validate()
    test_generate_split_nonempty()
    print("smoke_test: ok")


if __name__ == "__main__":
    main()
