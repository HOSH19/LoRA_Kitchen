"""Tokenization helpers for chat-format SFT."""

from __future__ import annotations

from typing import Any, Callable

from transformers import PreTrainedTokenizerBase


def build_formatter(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> Callable[[dict[str, Any]], dict[str, list[int]]]:
    """Return a map function that renders chat templates to input ids."""

    def _format(example: dict[str, Any]) -> dict[str, list[int]]:
        """Convert one messages example into model inputs."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    return _format
