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
from trl import SFTConfig, SFTTrainer

from train.config_load import TrainConfig, load_train_config
from train.model_factory import attach_lora, load_base_model, load_tokenizer


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


def build_trainer(cfg: TrainConfig) -> SFTTrainer:
    """Assemble tokenizer, model, datasets, and TRL trainer."""
    tokenizer = load_tokenizer(cfg.base_model)
    model = attach_lora(load_base_model(cfg), cfg)
    dataset = load_splits(cfg)
    sft_args = SFTConfig(
        output_dir=str(ROOT / cfg.output_dir),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        logging_steps=10,
        save_steps=cfg.max_steps,
        eval_strategy="no",
        bf16=True,
        report_to=[],
        max_seq_length=cfg.max_seq_len,
        packing=False,
    )
    return SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
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
    tokenizer = trainer.processing_class
    tokenizer.save_pretrained(out)
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
