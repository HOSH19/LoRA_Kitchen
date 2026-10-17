"""Load demo dish submissions from JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from src.paths import DATA_DIR
from src.schemas import DishSubmission


def load_dishes(path: Path | None = None) -> list[DishSubmission]:
    """Read dish submissions used by the UI and benchmarks."""
    target = path or (DATA_DIR / "dishes.jsonl")
    dishes: list[DishSubmission] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            dishes.append(DishSubmission.model_validate_json(line))
    return dishes
