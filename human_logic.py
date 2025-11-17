"""
human_logic.py

Purpose
-------
Implements the *human / archetype interpretation* for Style as Signal.

Given:
  - an 8D style fingerprint vector p (Bohemian..Sportswear),
  - the archetype templates from archetypes.py,

This module:
  - compares p to each archetype using cosine similarity,
  - finds the best and second-best archetypes,
  - computes a blend weight alpha between them,
  - produces a human-friendly label like:
        "The Nomad"
        "The Nomad / The Outlaw"
        "The Outlaw / The Nomad" (if you're closer to Outlaw)
  - returns fit scores and an ambiguity flag.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from archetype import ARCHETYPES  # your archetype vectors (8D, normalized)


# -------------------------
# Dataclass for result
# -------------------------

@dataclass
class HumanResult:
    label: str                        # final label string
    primary: str                      # best-match archetype name (A)
    secondary: str                    # second-best archetype name (B), or "" if none
    alpha: float                      # blend weight in [0,1]
    fit: float                        # similarity to primary (cosine)
    sims: Dict[str, float]            # all archetype similarities
    ambiguous: bool                   # True if A and B are nearly tied
    top2: Tuple[Tuple[str, float], Tuple[str, float]]  # ((A, sA), (B, sB))


# -------------------------
# Helpers
# -------------------------

def _normalize(vec: List[float]) -> List[float]:
    """Normalize a non-negative vector so it sums to 1.0. Uniform if all zeros."""
    cleaned = [max(0.0, float(x)) for x in vec]
    s = sum(cleaned)
    if s <= 0.0:
        n = len(cleaned)
        return [1.0 / n] * n if n > 0 else []
    return [x / s for x in cleaned]


def _cosine(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two equal-length vectors:
      cos(a,b) = (a·b) / (||a|| * ||b||)
    Returns 0.0 if either vector has near-zero length.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have same length for cosine similarity.")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# -------------------------
# Core interpretation
# -------------------------

def interpret_human(
    p: List[float],
    *,
    ambiguity_delta: float = 0.08,
    pure_alpha_threshold: float = 0.25,
    flip_alpha_threshold: float = 0.60,
) -> HumanResult:
    """
    Main human-layer function.

    Parameters
    ----------
    p : List[float]
        The 8D style fingerprint (any non-negative values). Will be normalized.
    ambiguity_delta : float
        If |sA - sB| < ambiguity_delta, mark as ambiguous/borderline.
    pure_alpha_threshold : float
        If alpha < this, treat as mostly primary archetype.
    flip_alpha_threshold : float
        If alpha > this, label is "B / A" (you're actually very close to B).

    Returns
    -------
    HumanResult
        Structured archetype interpretation.
    """
    if not p:
        raise ValueError("Empty style vector p passed to interpret_human().")

    # 1) Normalize p
    q = _normalize(p)

    # 2) Compute cosine similarity to each archetype
    sims: Dict[str, float] = {}
    for name, vec in ARCHETYPES.items():
        sims[name] = _cosine(q, vec)

    if not sims:
        raise ValueError("No archetypes defined in ARCHETYPES.")

    # 3) Sort archetypes by similarity descending
    sorted_pairs = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    primary_name, sA = sorted_pairs[0]
    if len(sorted_pairs) > 1:
        secondary_name, sB = sorted_pairs[1]
    else:
        secondary_name, sB = "", 0.0

    # 4) Compute alpha = how much secondary is blended into primary
    denom = sA + sB
    alpha = (sB / denom) if denom > 0.0 else 0.0

    # 5) Ambiguity flag: if A and B are nearly tied
    ambiguous = False
    if secondary_name:
        if abs(sA - sB) < ambiguity_delta:
            ambiguous = True

    # 6) Build label based on alpha
    if not secondary_name or sB <= 0.0:
        # No meaningful second archetype
        label = primary_name
    else:
        if alpha < pure_alpha_threshold:
            # Mostly primary
            label = primary_name
        elif alpha <= flip_alpha_threshold:
            # Blend: A / B
            label = f"{primary_name} / {secondary_name}"
        else:
            # Strong pull towards B: B / A
            label = f"{secondary_name} / {primary_name}"

    return HumanResult(
        label       = label,
        primary     = primary_name,
        secondary   = secondary_name,
        alpha       = float(alpha),
        fit         = float(sA),
        sims        = sims,
        ambiguous   = ambiguous,
        top2        = ((primary_name, sA), (secondary_name, sB)),
    )


# ---------------------------
# Self-test / demo (optional)
# ---------------------------
if __name__ == "__main__":
    # Example: pretend this is your live style fingerprint
    example_p = [0.30, 0.05, 0.35, 0.05, 0.02, 0.18, 0.03, 0.02]  # Denim/Boho/Rocker-ish
    result = interpret_human(example_p)
    print("Label     :", result.label)
    print("Primary   :", result.primary, f"(fit={result.fit:.3f})")
    print("Secondary :", result.secondary)
    print("Alpha     :", f"{result.alpha:.3f}")
    print("Ambiguous :", result.ambiguous)
    print("Top2 sims :", result.top2)