"""Gold JudgeCard labels for supervised persona training."""

from __future__ import annotations

import random

from src.personas import bible_for
from src.schemas import DishSubmission, JudgeCard, JudgeScores, Persona
from train.synth_dishes import rng_for


def _clamp(value: int) -> int:
    """Keep rubric scores inside 0..10."""
    return max(0, min(10, value))


def score_set(persona: Persona, rng: random.Random) -> JudgeScores:
    """Sample scores with persona-specific calibration."""
    if persona is Persona.FLAME:
        base = [rng.randint(3, 7) for _ in range(4)]
    elif persona is Persona.BUTTER:
        base = [rng.randint(6, 9) for _ in range(4)]
    else:
        base = [rng.randint(2, 10) for _ in range(4)]
        base[2] = rng.randint(6, 10)
    return JudgeScores(
        taste=_clamp(base[0]),
        technique=_clamp(base[1]),
        creativity=_clamp(base[2]),
        presentation=_clamp(base[3]),
    )


def critique_for(persona: Persona, dish: DishSubmission, rng: random.Random) -> str:
    """Write a short in-character critique."""
    title = dish.title
    if persona is Persona.FLAME:
        return (
            f"{title} shows ambition that outruns execution. "
            f"The salt balance is uneven and the heat management is sloppy. "
            f"Fix the fundamentals before chasing garnish."
        )
    if persona is Persona.BUTTER:
        return (
            f"I can tell you cared about {title}, and that matters. "
            f"Next time, give the aromatics a slower start and taste as you go. "
            f"You are closer than you think, dear."
        )
    metaphor = rng.choice(
        [
            "a rollercoaster that forgot the seatbelts",
            "a jazz solo played on a smoke alarm",
            "a plot twist in a grocery aisle",
        ]
    )
    return (
        f"{title} hit me like {metaphor}. "
        f"I respect the audacity even when the textures argue. "
        f"Would I eat it in a dream sequence? Absolutely."
    )


def one_liner(persona: Persona, dish: DishSubmission, rng: random.Random) -> str:
    """Short punch line used in the show UI."""
    if persona is Persona.FLAME:
        options = [
            f"{dish.title}: competent ideas, careless finish.",
            "Precision called. It hung up.",
            "The pan remembers what you did.",
        ]
    elif persona is Persona.BUTTER:
        options = [
            f"Sweet effort on {dish.title}.",
            "Come sit, let's season together.",
            "Brave flavors deserve a gentle hand.",
        ]
    else:
        options = [
            f"{dish.title} just crashed my brainstem playlist.",
            "Flavor said boom. Texture said maybe.",
            "This plate is unhinged in HD.",
        ]
    return rng.choice(options)


def make_card(persona: Persona, dish: DishSubmission, seed: str) -> JudgeCard:
    """Create a gold JudgeCard for supervised fine-tuning."""
    rng = rng_for(seed)
    bible = bible_for(persona)
    scores = score_set(persona, rng)
    threshold = 22 if persona is Persona.FLAME else 18
    return JudgeCard(
        persona=persona.value,
        scores=scores,
        one_liner=one_liner(persona, dish, rng),
        critique=critique_for(persona, dish, rng),
        would_eat_again=scores.total() >= threshold,
        signature_catchphrase=rng.choice(list(bible.catchphrases)),
    )
