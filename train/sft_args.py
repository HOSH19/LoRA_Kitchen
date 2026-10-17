"""Build TRL SFTConfig/SFTTrainer kwargs across TRL versions."""

from __future__ import annotations

import inspect
from typing import Any

from trl import SFTConfig, SFTTrainer

from train.config_load import TrainConfig


def build_sft_config(cfg: TrainConfig) -> SFTConfig:
    """Create an SFTConfig compatible with the installed TRL version."""
    kwargs: dict[str, Any] = {
        "output_dir": str(cfg.output_dir),
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 10,
        "save_steps": cfg.max_steps,
        "bf16": True,
        "report_to": [],
        "packing": False,
    }
    params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in params:
        kwargs["max_length"] = cfg.max_seq_len
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = cfg.max_seq_len
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"
    return SFTConfig(**kwargs)


def build_sft_trainer(model, tokenizer, train_dataset, sft_args, formatting_func) -> SFTTrainer:
    """Create an SFTTrainer with tokenizer/processing_class alias support."""
    kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
        "formatting_func": formatting_func,
    }
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)
