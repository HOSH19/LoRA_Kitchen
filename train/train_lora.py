#!/usr/bin/env python3
"""Fine-tune one judge LoRA with QLoRA + TRL SFTTrainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from train.config_load import TrainConfig, load_train_config
from train.model_factory import attach_lora, load_base_model, load_tokenizer
from train.sft_args import build_sft_config, build_sft_trainer


def parse_args() -> argparse.Namespace:
    """CLI arguments for a single-persona training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to train/configs/{flame,butter,chaos}.yaml",
    )
    return parser.parse_args()


def load_splits(cfg: TrainConfig):
    """Load train and eval JSONL files as Hugging Face datasets."""
    data_files = {
        "train": str(ROOT / cfg.train_file),
        "eval": str(ROOT / cfg.eval_file),
    }
    return load_dataset("json", data_files=data_files)


def make_formatting_func(tokenizer):
    """Build a TRL formatting function over chat messages."""

    def _format(example: dict[str, Any]) -> str:
        """Render one example to a single supervised training string."""
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    return _format


def build_trainer(cfg: TrainConfig):
    """Assemble tokenizer, model, datasets, and TRL trainer."""
    tokenizer = load_tokenizer(cfg.base_model)
    model = attach_lora(load_base_model(cfg), cfg)
    dataset = load_splits(cfg)
    resolved = cfg.model_copy(update={"output_dir": str(ROOT / cfg.output_dir)})
    sft_args = build_sft_config(resolved)
    return build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        sft_args=sft_args,
        formatting_func=make_formatting_func(tokenizer),
    )


def main() -> None:
    """Train and persist one persona adapter directory."""
    args = parse_args()
    cfg = load_train_config(args.config)
    trainer = build_trainer(cfg)
    trainer.train()
    out = ROOT / cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
    tokenizer.save_pretrained(out)
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
