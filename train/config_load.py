"""YAML config loading for LoRA training jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Typed training hyperparameters for one persona adapter."""

    persona: str
    base_model: str
    output_dir: str
    train_file: str
    eval_file: str
    max_steps: int = 120
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 1536
    load_in_4bit: bool = True
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


def load_train_config(path: Path) -> TrainConfig:
    """Parse a YAML file into a TrainConfig."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig.model_validate(raw)
