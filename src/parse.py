"""Helpers for turning model text into validated JudgeCards."""

from __future__ import annotations

import json
import re

from src.schemas import JudgeCard


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("no JSON object found in model output")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("decoded JSON was not an object")
    return payload


def parse_judge_card(text: str) -> JudgeCard:
    """Validate model output against the JudgeCard contract."""
    return JudgeCard.model_validate(extract_json_object(text))
