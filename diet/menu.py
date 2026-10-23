"""The menu of diets. Kept free of heavy imports so the web server stays light.

The names follow the talk "How to make a 14GB model lose weight":
portion control (quantization), trim the hedge (pruning) and train an
apprentice (distillation).
"""

from __future__ import annotations

DIETS = {
    "quantize": {
        "name": "Portion control",
        "technical": "Quantization",
        "tagline": "Same meal, smaller portions",
        "blurb": "Every number stays, just stored with fewer digits.",
        "picture": "img/portions.webp",
        "plans": {
            "bf16": "Half portions (16-bit)",
            "int8": "Small portions (8-bit)",
            "nf4": "Tiny portions (4-bit)",
        },
    },
    "prune": {
        "name": "Trim the hedge",
        "technical": "Pruning",
        "tagline": "Cut away the dead branches",
        "blurb": "Remove the layers that barely do any work.",
        "picture": "img/hedge.webp",
        "plans": {
            "layers": "Cut the dead layers",
        },
    },
    "distill": {
        "name": "Train an apprentice",
        "technical": "Distillation",
        "tagline": "Same suit, smaller tailor",
        "blurb": "A small new model learns by copying the big one.",
        "picture": "img/apprentice.webp",
        "plans": {"student": "Train a smaller apprentice"},
    },
}


class BadPrescription(ValueError):
    pass


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def validate(diet: str, plan: str, options: dict | None) -> dict:
    """Check a prescription and clamp its options to safe ranges."""
    options = options or {}
    if diet not in DIETS:
        raise BadPrescription(f"Unknown diet '{diet}'.")
    if plan not in DIETS[diet]["plans"]:
        raise BadPrescription(f"Unknown plan '{plan}' for {DIETS[diet]['name']}.")
    if diet == "prune":
        return {"fraction": _clamp(options.get("fraction"), 0.05, 0.5, 0.25)}
    if diet == "distill":
        return {
            "keep_fraction": _clamp(options.get("keep_fraction"), 0.3, 0.75, 0.5),
            "steps": int(_clamp(options.get("steps"), 10, 300, 80)),
        }
    return {}
