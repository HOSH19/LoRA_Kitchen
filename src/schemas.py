"""Pydantic contracts for dish submissions and judge outputs."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Persona(str, Enum):
    """Registered LoRA judge identifiers served by vLLM."""

    FLAME = "flame"
    BUTTER = "butter"
    CHAOS = "chaos"


class DishSubmission(BaseModel):
    """User-facing dish description sent to every judge."""

    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=4000)
    ingredients: list[str] = Field(default_factory=list)
    mystery_basket: list[str] = Field(default_factory=list)

    def as_prompt_block(self) -> str:
        """Render a stable text block for chat prompts."""
        lines = [f"Title: {self.title}", f"Description: {self.description}"]
        if self.ingredients:
            lines.append("Ingredients: " + ", ".join(self.ingredients))
        if self.mystery_basket:
            lines.append("Mystery basket: " + ", ".join(self.mystery_basket))
        return "\n".join(lines)


class JudgeScores(BaseModel):
    """Four rubric axes shared by every persona."""

    taste: int = Field(ge=0, le=10)
    technique: int = Field(ge=0, le=10)
    creativity: int = Field(ge=0, le=10)
    presentation: int = Field(ge=0, le=10)

    def total(self) -> int:
        """Sum the rubric for leaderboard sorting."""
        return self.taste + self.technique + self.creativity + self.presentation


class JudgeCard(BaseModel):
    """Structured verdict emitted by a single LoRA judge."""

    persona: Literal["flame", "butter", "chaos"]
    scores: JudgeScores
    one_liner: str = Field(min_length=3, max_length=200)
    critique: str = Field(min_length=20, max_length=2000)
    would_eat_again: bool
    signature_catchphrase: str = Field(min_length=3, max_length=120)

    @field_validator("critique")
    @classmethod
    def critique_has_sentences(cls, value: str) -> str:
        """Reject one-word dumps that break the show format."""
        if value.count(".") + value.count("!") + value.count("?") < 2:
            raise ValueError("critique must contain at least two sentences")
        return value.strip()


class PanelResult(BaseModel):
    """Aggregate output from the three-judge panel."""

    dish: DishSubmission
    cards: list[JudgeCard]

    def ranked(self) -> list[JudgeCard]:
        """Order judges by total score, highest first."""
        return sorted(self.cards, key=lambda card: card.scores.total(), reverse=True)
