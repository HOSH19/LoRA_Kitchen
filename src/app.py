#!/usr/bin/env python3
"""Streamlit kitchen stadium for the multi-LoRA judge panel."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import get_settings
from src.dishes import load_dishes
from src.health import listed_model_ids
from src.metrics_scrape import snapshot_metrics
from src.panel import run_panel
from src.schemas import DishSubmission
from src.ui_format import card_markdown, leaderboard_rows


def _sidebar_status() -> None:
    """Show server connectivity and a metrics snapshot."""
    settings = get_settings()
    st.sidebar.header("vLLM server")
    st.sidebar.code(settings.vllm_base_url)
    try:
        models = listed_model_ids()
        st.sidebar.success("Server reachable")
        st.sidebar.write("Models:")
        st.sidebar.json(models)
    except Exception as exc:
        st.sidebar.error(f"Server unreachable: {exc}")
    metrics = snapshot_metrics()
    if metrics:
        st.sidebar.write("Metrics")
        st.sidebar.json(metrics)


def _sample_picker() -> DishSubmission | None:
    """Optional quick-pick from the demo dish corpus."""
    dishes = load_dishes()
    labels = [f"{index}: {dish.title}" for index, dish in enumerate(dishes)]
    choice = st.selectbox("Sample dish", options=["(custom)"] + labels)
    if choice == "(custom)":
        return None
    index = int(choice.split(":", 1)[0])
    return dishes[index]


def _dish_form(sample: DishSubmission | None) -> DishSubmission:
    """Collect the dish submission from widgets."""
    title = st.text_input("Title", value=sample.title if sample else "")
    description = st.text_area(
        "Description",
        value=sample.description if sample else "",
        height=140,
    )
    ingredients_raw = st.text_input(
        "Ingredients (comma-separated)",
        value=", ".join(sample.ingredients) if sample else "",
    )
    basket_raw = st.text_input(
        "Mystery basket (comma-separated)",
        value=", ".join(sample.mystery_basket) if sample else "",
    )
    return DishSubmission(
        title=title.strip() or "Untitled plate",
        description=description.strip() or "No description provided.",
        ingredients=_split_csv(ingredients_raw),
        mystery_basket=_split_csv(basket_raw),
    )


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated widget value into clean tokens."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(page_title="LoRA Kitchen", page_icon="🍳", layout="wide")
    st.title("LoRA Kitchen")
    st.caption("One GPU. Three judge LoRAs. Parallel verdicts via vLLM.")
    _sidebar_status()

    sample = _sample_picker()
    dish = _dish_form(sample)
    if st.button("Summon the judges", type="primary"):
        with st.spinner("Judges are tasting concurrently..."):
            panel = run_panel(dish)
        st.subheader("Scoreboard")
        st.dataframe(leaderboard_rows(panel), use_container_width=True)
        columns = st.columns(len(panel.cards))
        for column, card in zip(columns, panel.cards):
            with column:
                st.markdown(card_markdown(card))


if __name__ == "__main__":
    main()
