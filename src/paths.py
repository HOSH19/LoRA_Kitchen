"""Filesystem roots for the project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ADAPTERS_DIR = ROOT / "adapters"
RESULTS_DIR = ROOT / "results"
TRAIN_CONFIG_DIR = ROOT / "train" / "configs"


def ensure_results_dir() -> Path:
    """Create the results directory if needed and return it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
