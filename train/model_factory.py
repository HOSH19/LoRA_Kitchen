"""Build quantized base models and LoRA wrappers for training."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train.config_load import TrainConfig


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load a chat-capable tokenizer with a safe pad token."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: TrainConfig) -> AutoModelForCausalLM:
    """Load the base causal LM, optionally in 4-bit for QLoRA."""
    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    load_kwargs = {
        "device_map": "auto",
        "quantization_config": quant,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            dtype=torch.bfloat16,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        )
    model.config.use_cache = False
    return model


def attach_lora(model: AutoModelForCausalLM, cfg: TrainConfig):
    """Wrap a base model with a PEFT LoRA adapter."""
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )
    return get_peft_model(model, lora)
