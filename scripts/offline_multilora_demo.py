#!/usr/bin/env python3
"""Offline vLLM demo: one engine, three LoRARequest judges, guided JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.dishes import load_dishes
from src.parse import parse_judge_card
from src.paths import ADAPTERS_DIR, RESULTS_DIR, ensure_results_dir
from src.personas import all_personas
from src.prompts import build_messages
from src.schemas import JudgeCard, Persona


def parse_args() -> argparse.Namespace:
    """CLI for the offline multi-LoRA mastery demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dish-index", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    return parser.parse_args()


def require_adapters() -> dict[Persona, Path]:
    """Resolve adapter directories and fail fast if missing."""
    paths = {persona: ADAPTERS_DIR / persona.value for persona in all_personas()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing adapters. Train first or mount them under adapters/: "
            + ", ".join(missing)
        )
    return paths


def render_prompt(tokenizer, persona: Persona, dish) -> str:
    """Apply the base chat template for offline generation."""
    return tokenizer.apply_chat_template(
        build_messages(persona, dish),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_engine(base_model: str, max_model_len: int) -> LLM:
    """Create a vLLM engine with LoRA and prefix caching enabled."""
    return LLM(
        model=base_model,
        enable_lora=True,
        max_loras=3,
        max_lora_rank=16,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )


def sampling_for_schema() -> SamplingParams:
    """Structured-output sampling params for JudgeCard JSON."""
    schema = JudgeCard.model_json_schema()
    try:
        from vllm.sampling_params import GuidedDecodingParams

        guided = GuidedDecodingParams(json=schema)
        return SamplingParams(
            temperature=0.7,
            max_tokens=700,
            guided_decoding=guided,
        )
    except Exception:
        return SamplingParams(temperature=0.7, max_tokens=700)


def main() -> None:
    """Generate three persona cards in one offline generate() call batch."""
    args = parse_args()
    adapters = require_adapters()
    dish = load_dishes()[args.dish_index]
    llm = build_engine(args.base_model, args.max_model_len)
    tokenizer = llm.get_tokenizer()
    prompts = [render_prompt(tokenizer, persona, dish) for persona in all_personas()]
    lora_requests = [
        LoRARequest(persona.value, index + 1, str(adapters[persona]))
        for index, persona in enumerate(all_personas())
    ]
    outputs = llm.generate(
        prompts,
        sampling_for_schema(),
        lora_request=lora_requests,
    )
    cards = []
    for persona, output in zip(all_personas(), outputs):
        text = output.outputs[0].text
        card = parse_judge_card(text)
        cards.append(card.model_dump())
        print(f"\n=== {persona.value} ===\n{text}")

    ensure_results_dir()
    out = RESULTS_DIR / "offline_multilora_demo.json"
    out.write_text(
        json.dumps({"dish": dish.model_dump(), "cards": cards}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
