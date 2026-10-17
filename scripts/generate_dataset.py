#!/usr/bin/env python3
"""Generate synthetic train/eval JSONL and demo dishes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import DATA_DIR
from src.personas import all_personas
from train.dataset_io import write_jsonl
from train.synth import generate_split, make_dish


def write_dishes(count: int = 24) -> Path:
    """Materialize the demo dish corpus."""
    path = DATA_DIR / "dishes.jsonl"
    rows = [make_dish(index).model_dump() for index in range(count)]
    write_jsonl(path, rows)
    return path


def write_persona_splits(train_n: int = 360, eval_n: int = 72) -> None:
    """Write train/eval JSONL for every judge persona."""
    for persona in all_personas():
        train_rows = generate_split(persona, train_n, seed_prefix=f"train-{persona.value}")
        eval_rows = generate_split(persona, eval_n, seed_prefix=f"eval-{persona.value}")
        write_jsonl(DATA_DIR / persona.value / "train.jsonl", train_rows)
        write_jsonl(DATA_DIR / persona.value / "eval.jsonl", eval_rows)


def main() -> None:
    """Entrypoint used by README setup steps."""
    write_dishes()
    write_persona_splits()
    print(f"Wrote datasets under {DATA_DIR}")


if __name__ == "__main__":
    main()
