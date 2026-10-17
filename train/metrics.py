"""Persona fidelity metrics for ablation tables."""

from __future__ import annotations

from dataclasses import dataclass

from src.personas import bible_for
from src.schemas import JudgeCard, Persona


@dataclass(frozen=True)
class PersonaScores:
    """Aggregate quality numbers for one eval pass."""

    n: int
    schema_valid_rate: float
    catchphrase_hit_rate: float
    mean_abs_score_error: float


def catchphrase_hit(card: JudgeCard, persona: Persona) -> bool:
    """Check whether the model used an allowed catchphrase."""
    allowed = set(bible_for(persona).catchphrases)
    return card.signature_catchphrase in allowed


def mean_abs_score_error(pred: JudgeCard, gold: JudgeCard) -> float:
    """Average absolute error across the four rubric axes."""
    axes = ("taste", "technique", "creativity", "presentation")
    errors = [
        abs(getattr(pred.scores, axis) - getattr(gold.scores, axis)) for axis in axes
    ]
    return sum(errors) / len(errors)


def summarize(
    valid_flags: list[bool],
    catchphrase_flags: list[bool],
    score_errors: list[float],
) -> PersonaScores:
    """Reduce per-example flags into a PersonaScores row."""
    n = len(valid_flags)
    if n == 0:
        return PersonaScores(0, 0.0, 0.0, 0.0)
    return PersonaScores(
        n=n,
        schema_valid_rate=sum(valid_flags) / n,
        catchphrase_hit_rate=sum(catchphrase_flags) / n,
        mean_abs_score_error=(sum(score_errors) / len(score_errors)) if score_errors else 0.0,
    )
