"""
machine_logic.py

Purpose
-------
Implements the *machine interpretation* for Style as Signal.
Given an 8-number "style fingerprint" vector p (one value per style),
this module decides whether the person is:
  - PURE(<top style>)            e.g., PURE(Denim)
  - HYBRID(<style1 + style2>)    e.g., HYBRID(Denim + Bohemian)
  - TRIAD(<style1 + style2 + style3>) [optional, when a 3rd style is strong]

It also returns:
  - score:       a blunt numeric score for the label
                 (PURE → p1; HYBRID → p1 + p2; TRIAD → p1 + p2 + p3)
  - confidence:  the top style value p1 (how strong the main signal is)
  - uncertainty: entropy H(p) — how mixed the fingerprint is
  - ranks:       indices of top styles and their values, for transparency

This is intentionally "cold" and rule-based (very machine-like).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import math


@dataclass
class MachineResult:
    """A structured result for the machine layer."""
    segment_label: str       # e.g., 'PURE(Denim)' or 'HYBRID(Denim + Bohemian)'
    score: float             # numeric score for the chosen label
    confidence: float        # top style value (p1)
    uncertainty: float       # entropy H(p) in nats (ln base)
    top_indices: Tuple[int, int, int]  # (idx of top1, top2, top3)
    top_values: Tuple[float, float, float]  # (p1, p2, p3)


def _safe_normalize(p: List[float]) -> List[float]:
    """
    Ensure p is a valid probability-like vector:
      - clamp negatives to 0
      - renormalize to sum to 1 (if sum > 0)
      - if all zeros, return a uniform distribution
    """
    cleaned = [max(0.0, float(x)) for x in p]
    s = sum(cleaned)
    if s > 0.0:
        return [x / s for x in cleaned]
    # fallback: uniform distribution to avoid divide-by-zero
    n = len(cleaned)
    return [1.0 / n] * n if n > 0 else []


def _entropy(p: List[float]) -> float:
    """
    Shannon entropy in natural units (nats).
    H(p) = - sum_i p_i * ln(p_i)
    By convention, terms with p_i == 0 contribute 0.
    """
    H = 0.0
    for x in p:
        if x > 0.0:
            H -= x * math.log(x)
    return H


def _top3(p: List[float]) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Return indices and values of the top-3 components in p, sorted descending.
    If p has fewer than 3 elements, pad with -1 / 0.0 appropriately.
    """
    indexed = list(enumerate(p))
    indexed.sort(key=lambda kv: kv[1], reverse=True)
    # gather top three (or pad)
    top_idx = [kv[0] for kv in indexed[:3]]
    top_val = [kv[1] for kv in indexed[:3]]
    while len(top_idx) < 3:
        top_idx.append(-1)
        top_val.append(0.0)
    return (top_idx[0], top_idx[1], top_idx[2]), (top_val[0], top_val[1], top_val[2])


def classify_machine(
    p: List[float],
    class_names: List[str],
    *,
    pure_threshold: float = 0.60,
    dominance_ratio: float = 1.5,
    triad_threshold: float = 0.20,
    enable_triad: bool = True
) -> MachineResult:
    """
    Core machine-layer decision function.

    Parameters
    ----------
    p : List[float]
        The 8-number style fingerprint (any non-negative values). Will be normalized internally.
    class_names : List[str]
        Names for each index in p (must have same length as p).
    pure_threshold : float
        If top value p1 >= pure_threshold AND p1/p2 >= dominance_ratio → PURE.
    dominance_ratio : float
        The required ratio p1/p2 to consider the top style truly dominant (default 1.5).
    triad_threshold : float
        If p3 >= triad_threshold AND not pure → TRIAD (if enable_triad is True).
    enable_triad : bool
        Controls whether TRIAD labels can be produced.

    Returns
    -------
    MachineResult
        A structured object with the label, score, confidence, uncertainty, and top ranks.
    """
    # 1) Clean and normalize p so it sums to 1, with no negatives.
    q = _safe_normalize(p)

    # 2) Sanity check: class_names must match length of p.
    if len(q) != len(class_names):
        raise ValueError(
            f"Length mismatch: got p of length {len(q)} but {len(class_names)} class_names."
        )

    # 3) Compute entropy = "uncertainty": mixed vs pure signal.
    uncertainty = _entropy(q)

    # 4) Extract top-3 indices and values.
    (i1, i2, i3), (p1, p2, p3) = _top3(q)

    # Edge case: if no classes (empty p)
    if i1 == -1:
        return MachineResult(
            segment_label="UNKNOWN",
            score=0.0,
            confidence=0.0,
            uncertainty=uncertainty,
            top_indices=(-1, -1, -1),
            top_values=(0.0, 0.0, 0.0),
        )

    # 5) Decide label using the "cold" rules.
    #    Rule A: PURE if top is big enough (p1 >= pure_threshold) AND clearly dominates (p1/p2 >= dominance_ratio).
    pure_ok = (p1 >= pure_threshold) and (p2 > 0.0) and ((p1 / p2) >= dominance_ratio)

    # Note: handle the corner case where p2 == 0 gracefully (already handled by p2>0 in pure_ok).
    # If p2 == 0 but p1 >= pure_threshold, we can still consider it PURE by definition.
    if p2 == 0.0 and p1 >= pure_threshold:
        pure_ok = True

    if pure_ok:
        seg = f"PURE({class_names[i1]})"
        score = p1  # PURE score is just the dominant strength
        confidence = p1
        return MachineResult(
            segment_label=seg,
            score=float(score),
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            top_indices=(i1, i2, i3),
            top_values=(p1, p2, p3),
        )

    # 6) If not PURE, consider TRIAD (optional) if p3 is also substantial.
    if enable_triad and (i2 != -1) and (i3 != -1) and (p3 >= triad_threshold):
        seg = f"TRIAD({class_names[i1]} + {class_names[i2]} + {class_names[i3]})"
        score = p1 + p2 + p3
        confidence = p1  # still use top value as "confidence"
        return MachineResult(
            segment_label=seg,
            score=float(score),
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            top_indices=(i1, i2, i3),
            top_values=(p1, p2, p3),
        )

    # 7) Otherwise HYBRID by default (top two are driving perception).
    if i2 != -1:
        seg = f"HYBRID({class_names[i1]} + {class_names[i2]})"
        score = p1 + p2
        confidence = p1
        return MachineResult(
            segment_label=seg,
            score=float(score),
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            top_indices=(i1, i2, i3),
            top_values=(p1, p2, p3),
        )

    # 8) Fallback (should rarely happen)
    seg = f"PURE({class_names[i1]})"
    score = p1
    confidence = p1
    return MachineResult(
        segment_label=seg,
        score=float(score),
        confidence=float(confidence),
        uncertainty=float(uncertainty),
        top_indices=(i1, i2, i3),
        top_values=(p1, p2, p3),
    )


# ---------------------------
# Self-test / demo (optional)
# ---------------------------
if __name__ == "__main__":
    # Example class order must match your model:
    class_names = ['Bohemian', 'Chic', 'Denim', 'Elegant', 'Formal', 'Rocker', 'Luxury', 'Sportswear']

    # Example fingerprints (not normalized on purpose; function will normalize safely)
    examples = {
        "Pure-ish Denim":      [0.05, 0.02, 0.70, 0.03, 0.02, 0.10, 0.02, 0.06],
        "Hybrid Denim/Boho":   [0.35, 0.05, 0.37, 0.04, 0.03, 0.06, 0.04, 0.06],
        "Triad D/B/Chic":      [0.28, 0.21, 0.31, 0.05, 0.02, 0.04, 0.03, 0.06],
        "Flat/uncertain":      [0.125]*8,
    }

    for name, p in examples.items():
        result = classify_machine(p, class_names)
        print(f"\n=== {name} ===")
        print(f"Label       : {result.segment_label}")
        print(f"Score       : {result.score:.3f}")
        print(f"Confidence  : {result.confidence:.3f}")
        print(f"Uncertainty : {result.uncertainty:.3f} (higher = more mixed)")
        i1, i2, i3 = result.top_indices
        v1, v2, v3 = result.top_values
        def safe(i): return class_names[i] if i >= 0 else "-"
        print(f"Top styles  : {safe(i1)}={v1:.2f}, {safe(i2)}={v2:.2f}, {safe(i3)}={v3:.2f}")