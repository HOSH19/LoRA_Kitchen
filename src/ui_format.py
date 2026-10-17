"""Presentation helpers for the Streamlit kitchen stadium."""

from __future__ import annotations

from src.personas import bible_for
from src.schemas import JudgeCard, PanelResult, Persona


def card_markdown(card: JudgeCard) -> str:
    """Render one judge card as markdown."""
    bible = bible_for(card.persona)
    scores = card.scores
    return (
        f"### {bible.display_name}\n"
        f"**{card.one_liner}**\n\n"
        f"_{card.signature_catchphrase}_\n\n"
        f"- Taste: {scores.taste}/10\n"
        f"- Technique: {scores.technique}/10\n"
        f"- Creativity: {scores.creativity}/10\n"
        f"- Presentation: {scores.presentation}/10\n"
        f"- Total: **{scores.total()}/40**\n"
        f"- Would eat again: {'yes' if card.would_eat_again else 'no'}\n\n"
        f"{card.critique}"
    )


def leaderboard_rows(panel: PanelResult) -> list[dict]:
    """Build a simple table for the scoreboard."""
    rows = []
    for card in panel.ranked():
        bible = bible_for(card.persona)
        rows.append(
            {
                "judge": bible.display_name,
                "total": card.scores.total(),
                "taste": card.scores.taste,
                "technique": card.scores.technique,
                "creativity": card.scores.creativity,
                "presentation": card.scores.presentation,
                "eat_again": card.would_eat_again,
            }
        )
    return rows


def persona_label(name: str) -> str:
    """Pretty-print a persona id."""
    return bible_for(Persona(name)).display_name
