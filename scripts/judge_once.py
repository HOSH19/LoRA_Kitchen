#!/usr/bin/env python3
"""Judge one dish through the live multi-LoRA vLLM panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dishes import load_dishes
from src.panel import run_panel
from src.schemas import DishSubmission


def parse_args() -> argparse.Namespace:
    """CLI for a one-shot panel invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dish-index", type=int, default=0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--description", default=None)
    return parser.parse_args()


def main() -> None:
    """Print a PanelResult as JSON."""
    args = parse_args()
    if args.title and args.description:
        dish = DishSubmission(title=args.title, description=args.description)
    else:
        dish = load_dishes()[args.dish_index]
    panel = run_panel(dish)
    print(panel.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
