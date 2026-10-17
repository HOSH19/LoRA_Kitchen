"""Single-judge inference against a named vLLM LoRA module."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.client import guided_json_body
from src.parse import parse_judge_card
from src.prompts import build_messages
from src.schemas import DishSubmission, JudgeCard, Persona


def _completion_kwargs(persona: Persona, dish: DishSubmission) -> dict:
    """Shared chat.completions arguments for sync and async paths."""
    schema = JudgeCard.model_json_schema()
    return {
        "model": persona.value,
        "messages": build_messages(persona, dish),
        "temperature": 0.7,
        "max_tokens": 700,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_card",
                "schema": schema,
                "strict": True,
            },
        },
        "extra_body": guided_json_body(schema),
    }


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(2))
def judge_sync(client: OpenAI, persona: Persona, dish: DishSubmission) -> JudgeCard:
    """Run one judge call on the synchronous OpenAI client."""
    response = client.chat.completions.create(**_completion_kwargs(persona, dish))
    content = response.choices[0].message.content or ""
    return parse_judge_card(content)


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(2))
async def judge_async(
    client: AsyncOpenAI,
    persona: Persona,
    dish: DishSubmission,
) -> JudgeCard:
    """Run one judge call on the async OpenAI client."""
    response = await client.chat.completions.create(**_completion_kwargs(persona, dish))
    content = response.choices[0].message.content or ""
    return parse_judge_card(content)
