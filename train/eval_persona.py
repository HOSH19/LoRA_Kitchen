#!/usr/bin/env python3
"""Evaluate persona fidelity for a LoRA judge via the vLLM OpenAI API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client import build_sync_client, guided_json_body
from src.parse import parse_judge_card
from src.paths import RESULTS_DIR, ensure_results_dir
from src.prompts import build_messages
from src.schemas import DishSubmission, JudgeCard, Persona
from train.dataset_io import read_jsonl
from train.metrics import catchphrase_hit, mean_abs_score_error, summarize


def parse_args() -> argparse.Namespace:
    """CLI for offline-style eval against a live vLLM endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", required=True, choices=[p.value for p in Persona])
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--model",
        default=None,
        help="vLLM model id to call (defaults to persona LoRA name)",
    )
    return parser.parse_args()


def iter_eval_rows(persona: Persona, limit: int):
    """Yield dish/gold pairs from the persona eval split."""
    path = ROOT / "data" / persona.value / "eval.jsonl"
    for row in read_jsonl(path)[:limit]:
        dish = DishSubmission.model_validate(row["dish"])
        gold = JudgeCard.model_validate(row["gold"])
        yield dish, gold


def evaluate(persona: Persona, model_name: str, limit: int) -> dict:
    """Run judge calls and compute persona fidelity metrics."""
    client = build_sync_client()
    valid_flags: list[bool] = []
    catchphrase_flags: list[bool] = []
    score_errors: list[float] = []

    for dish, gold in iter_eval_rows(persona, limit):
        try:
            pred = _judge_as_model(client, model_name, persona, dish)
            valid_flags.append(True)
            catchphrase_flags.append(catchphrase_hit(pred, persona))
            score_errors.append(mean_abs_score_error(pred, gold))
        except Exception:
            valid_flags.append(False)
            catchphrase_flags.append(False)

    scores = summarize(valid_flags, catchphrase_flags, score_errors)
    return {
        "persona": persona.value,
        "model": model_name,
        "n": scores.n,
        "schema_valid_rate": scores.schema_valid_rate,
        "catchphrase_hit_rate": scores.catchphrase_hit_rate,
        "mean_abs_score_error": scores.mean_abs_score_error,
    }


def _judge_as_model(client, model_name: str, persona: Persona, dish: DishSubmission) -> JudgeCard:
    """Call vLLM with an explicit model id while keeping persona prompts."""
    schema = JudgeCard.model_json_schema()
    response = client.chat.completions.create(
        model=model_name,
        messages=build_messages(persona, dish),
        temperature=0.7,
        max_tokens=700,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_card",
                "schema": schema,
                "strict": True,
            },
        },
        extra_body=guided_json_body(schema),
    )
    return parse_judge_card(response.choices[0].message.content or "")


def main() -> None:
    """Write one eval JSON blob under results/."""
    args = parse_args()
    persona = Persona(args.persona)
    model_name = args.model or persona.value
    report = evaluate(persona, model_name, args.limit)
    ensure_results_dir()
    out = RESULTS_DIR / f"eval_{persona.value}_{model_name.replace('/', '_')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
