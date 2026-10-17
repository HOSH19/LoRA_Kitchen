"""Synthetic dish corpus used for training and demos."""

from __future__ import annotations

import hashlib
import random

from src.schemas import DishSubmission

TITLES = [
    "Charred lemon risotto",
    "Midnight miso carbonara",
    "Crispy duck bao sliders",
    "Burnt honey halloumi",
    "Seaweed butter popcorn chicken",
    "Smoked peach panzanella",
    "Gochujang mac pie",
    "Truffle tater tot tower",
    "Yuzu chili noodles",
    "Campfire mushroom toast",
    "Saffron waffle stack",
    "Black garlic meatballs",
    "Coconut lime ceviche",
    "Anchovy caramel brussels",
    "Matcha tres leches chaos",
]

DESCRIPTIONS = [
    "Seared hard, finished with acid, plated on a cold dark plate.",
    "Soft center, crunchy edges, sauce that almost overwhelms the starch.",
    "Street-food energy with tasting-menu plating and too much garnish.",
    "Comforting base ruined slightly by an ambitious foam.",
    "Clean flavors fighting a muddy reduction and uneven salt.",
    "Bright herbs, lazy knife work, surprisingly good aroma.",
    "One perfect bite surrounded by three confused ones.",
    "Nostalgic and greasy in the best possible way.",
]

INGREDIENT_POOL = [
    "lemon",
    "miso",
    "duck",
    "honey",
    "seaweed",
    "peach",
    "gochujang",
    "truffle oil",
    "yuzu",
    "mushroom",
    "saffron",
    "black garlic",
    "coconut",
    "anchovy",
    "matcha",
    "chili",
    "butter",
    "rice",
    "egg",
    "bread",
]

BASKETS = [
    ["canned tuna", "marshmallows", "dill"],
    ["durian", "cornflakes", "blue cheese"],
    ["instant ramen", "strawberries", "thyme"],
    ["peanut butter", "oysters", "basil"],
    ["kimchi", "white chocolate", "celery"],
]


def rng_for(seed: str) -> random.Random:
    """Build a stable RNG from a string seed."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def make_dish(index: int) -> DishSubmission:
    """Create one synthetic dish submission."""
    rng = rng_for(f"dish-{index}")
    title = TITLES[index % len(TITLES)]
    if index >= len(TITLES):
        title = f"{title} #{index}"
    return DishSubmission(
        title=title,
        description=rng.choice(DESCRIPTIONS),
        ingredients=rng.sample(INGREDIENT_POOL, k=4),
        mystery_basket=rng.choice(BASKETS) if index % 3 == 0 else [],
    )
