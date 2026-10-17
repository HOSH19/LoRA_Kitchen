"""Chat message builders for judge inference."""

from __future__ import annotations

from src.personas import bible_for
from src.schemas import DishSubmission, JudgeCard, Persona


def judge_schema_hint() -> str:
    """Compact JSON schema reminder embedded in the user turn."""
    return JudgeCard.model_json_schema()


def build_messages(persona: Persona, dish: DishSubmission) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for one judge call."""
    bible = bible_for(persona)
    user = (
        f"Judge this dish as {bible.display_name}.\n\n"
        f"{dish.as_prompt_block()}\n\n"
        "Respond with a single JSON object matching this schema:\n"
        f"{judge_schema_hint()}\n\n"
        "signature_catchphrase must be exactly one of:\n"
        + "\n".join(f"- {phrase}" for phrase in bible.catchphrases)
    )
    return [
        {"role": "system", "content": bible.system_prompt},
        {"role": "user", "content": user},
    ]


def base_prompt_messages(persona: Persona, dish: DishSubmission) -> list[dict[str, str]]:
    """Messages used for the base-model ablation (no LoRA)."""
    return build_messages(persona, dish)
