"""
archetypes.py

Purpose
-------
Defines the HUMAN archetypes as 8-number style vectors, aligned with your base style classes.

Style dimensions (fixed order)
------------------------------
Index : Style
  0   : Bohemian
  1   : Chic
  2   : Denim
  3   : Elegant
  4   : Formal
  5   : Rocker
  6   : Luxury
  7   : Sportswear

Each archetype is a "style recipe" over these 8 dimensions.
We define raw weights (intuitive proportions), then normalize them
so each archetype vector sums to 1.0.

Archetypes
----------
1. The Idealist
2. The Nomad
3. The Thinker
4. The Fighter
5. The Outlaw
6. The Wizard
7. The Romantic
8. The King
"""

from typing import Dict, List


# Fixed style axis order, shared with your model
CLASS_NAMES: List[str] = [
    "Bohemian",   # 0
    "Chic",       # 1
    "Denim",      # 2
    "Elegant",    # 3
    "Formal",     # 4
    "Rocker",     # 5
    "Luxury",     # 6
    "Sportswear", # 7
]


# -------------------------------------------------------------------
# 1) RAW archetype definitions (before normalization)
#
# These numbers are relative weights that reflect your intent.
# They do NOT need to sum to 1; we normalize them below.
#
# For each archetype, the list is:
# [Bohemian, Chic, Denim, Elegant, Formal, Rocker, Luxury, Sportswear]
# -------------------------------------------------------------------

RAW_ARCHETYPES: Dict[str, List[float]] = {
    # The Idealist
    # soft, gentle, hopeful, slightly romantic, a bit practical
    "The Idealist": [
        0.35,  # Bohemian  (natural, soft)
        0.10,  # Chic      (a little curated)
        0.05,  # Denim     (casual grounding)
        0.20,  # Elegant   (gentle refinement)
        0.05,  # Formal    (small structure)
        0.02,  # Rocker    (almost none)
        0.03,  # Luxury    (very light)
        0.20,  # Sportswear (comfort, practicality)
    ],

    # The Nomad
    # layered, wandering, textured, denim/boho + some functional ease
    "The Nomad": [
        0.30,  # Bohemian
        0.05,  # Chic
        0.30,  # Denim
        0.05,  # Elegant
        0.02,  # Formal
        0.15,  # Rocker
        0.03,  # Luxury
        0.10,  # Sportswear
    ],

    # The Thinker
    # minimal, structured, quiet, non-performative, slightly formal
    "The Thinker": [
        0.05,  # Bohemian
        0.30,  # Chic      (minimal chic)
        0.05,  # Denim
        0.25,  # Elegant   (clean lines)
        0.20,  # Formal    (structure)
        0.02,  # Rocker
        0.08,  # Luxury    (subtle quality)
        0.05,  # Sportswear
    ],

    # The Fighter
    # functional, performance-driven, tough, active
    "The Fighter": [
        0.03,  # Bohemian
        0.05,  # Chic
        0.15,  # Denim       (rugged, durable)
        0.05,  # Elegant
        0.10,  # Formal      (discipline / uniforms)
        0.10,  # Rocker      (aggression edge)
        0.02,  # Luxury
        0.50,  # Sportswear  (core)
    ],

    # The Outlaw
    # rebellious, raw, anti-polish, denim + rocker + some boho
    "The Outlaw": [
        0.25,  # Bohemian   (wildness)
        0.03,  # Chic
        0.30,  # Denim
        0.03,  # Elegant
        0.02,  # Formal
        0.35,  # Rocker     (core)
        0.02,  # Luxury
        0.10,  # Sportswear (street edge)
    ],

    # The Wizard
    # liminal, mystical, elegant but strange, slightly luxe
    "The Wizard": [
        0.20,  # Bohemian   (mythic / organic)
        0.15,  # Chic       (composed)
        0.05,  # Denim
        0.30,  # Elegant    (flowing forms)
        0.05,  # Formal
        0.05,  # Rocker
        0.15,  # Luxury     (jewel tone / richness)
        0.05,  # Sportswear
    ],

    # The Romantic
    # sensual, aesthetic, polished, emotionally expressive
    "The Romantic": [
        0.10,  # Bohemian
        0.30,  # Chic       (curated, seen)
        0.05,  # Denim
        0.25,  # Elegant    (soft refinement)
        0.05,  # Formal
        0.05,  # Rocker
        0.15,  # Luxury     (indulgence)
        0.05,  # Sportswear
    ],

    # The King
    # authority, structure, status, composed, high polish
    "The King": [
        0.02,  # Bohemian
        0.20,  # Chic
        0.03,  # Denim
        0.20,  # Elegant
        0.25,  # Formal     (core)
        0.02,  # Rocker
        0.25,  # Luxury     (core)
        0.03,  # Sportswear
    ],
}


# -------------------------------------------------------------------
# 2) Normalization helper
# -------------------------------------------------------------------

def _normalize(v: List[float]) -> List[float]:
    """
    Normalize a list of non-negative numbers so they sum to 1.0.
    If the sum is 0, return a uniform vector instead.
    """
    cleaned = [max(0.0, float(x)) for x in v]
    s = sum(cleaned)
    if s <= 0.0:
        n = len(cleaned)
        return [1.0 / n] * n if n > 0 else []
    return [x / s for x in cleaned]


# -------------------------------------------------------------------
# 3) Public archetype dictionary (normalized vectors)
# -------------------------------------------------------------------

ARCHETYPES: Dict[str, List[float]] = {
    name: _normalize(vec) for name, vec in RAW_ARCHETYPES.items()
}


# -------------------------------------------------------------------
# 4) Optional: simple sanity check / demo when run directly
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("Class order:", CLASS_NAMES)
    print("\nArchetypes (normalized):")
    for name, vec in ARCHETYPES.items():
        s = sum(vec)
        pretty = ", ".join(f"{x:.2f}" for x in vec)
        print(f"- {name}: [{pretty}]  (sum={s:.3f})")