"""Metrics for computing RYE (Repair Yield per Energy).

This module defines helper functions to compute:

- ΔR (improvement) for each cycle
- E  (effort / energy cost) for each cycle
- RYE = ΔR / E (Repair Yield per Energy)
- Optional rolling RYE and efficiency trends
- Optional regression slope (better long-run trend detection)

Reparodynamics interpretation:
    The research agent is a reparodynamic system trying to raise RYE over
    time. Each TGRM cycle (Test → Detect → Repair → Verify) attempts to
    reduce defects (missing info, contradictions, TODOs) with minimal
    effort. Higher RYE means more efficient self repair.

Backwards compatibility:
    All existing calls continue working exactly as before.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import math


# ---------------------------------------------------------------------------
# ΔR (improvement) computation
# ---------------------------------------------------------------------------

def compute_delta_r(
    issues_before: int,
    issues_after: int,
    repairs_applied: int,
    contradictions_resolved: int = 0,
    hypotheses_generated: int = 0,
    sources_used: int = 0,
) -> float:
    """Compute the improvement ΔR for a cycle."""

    base = max(issues_before - issues_after, 0)

    contradiction_gain = contradictions_resolved * 0.5
    hypothesis_gain = hypotheses_generated * 0.2
    source_gain = min(max(sources_used, 0), 20) * 0.05

    delta = float(base + contradiction_gain + hypothesis_gain + source_gain)

    # Maintenance credit (avoid ΔR = 0 with real work)
    if issues_before == 0 and delta == 0 and repairs_applied > 0:
        delta = repairs_applied * 0.1

    return float(delta)


# ---------------------------------------------------------------------------
# E (effort / energy) computation
# ---------------------------------------------------------------------------

def compute_energy(
    actions_taken: List[Dict[str, str]],
    web_calls: int = 0,
    pubmed_calls: int = 0,
    semantic_calls: int = 0,
    pdf_ingestions: int = 0,
    tokens_estimate: Optional[int] = None,
) -> float:
    """Estimate the effort (E) expended during the cycle."""

    base_cost = float(len(actions_taken)) if actions_taken else 1.0

    cost_web = max(web_calls, 0) * 1.5
    cost_pubmed = max(pubmed_calls, 0) * 2.0
    cost_sem = max(semantic_calls, 0) * 2.0
    cost_pdf = max(pdf_ingestions, 0) * 2.5

    total = base_cost + cost_web + cost_pubmed + cost_sem + cost_pdf

    # Soft token-costing
    if tokens_estimate is not None and tokens_estimate > 0:
        total += float(tokens_estimate) / 1000.0

    # Safety clamp to avoid RYE explosions
    if total <= 0:
        total = 1.0
    if total < 0.05:
        total = 0.05

    return float(total)


# ---------------------------------------------------------------------------
# RYE computation
# ---------------------------------------------------------------------------

def compute_rye(delta_r: float, energy_e: float) -> float:
    """Compute RYE = ΔR / E."""
    if energy_e <= 0:
        return 0.0
    return float(delta_r) / float(energy_e)


# ---------------------------------------------------------------------------
# Rolling RYE
# ---------------------------------------------------------------------------

def rolling_rye(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """Compute a rolling average of RYE."""
    if not history:
        return None

    recent = history[-window:]
    vals: List[float] = []

    for entry in recent:
        v = entry.get("RYE")
        if isinstance(v, (int, float)):
            vals.append(float(v))

    if not vals:
        return None

    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Simple trend
# ---------------------------------------------------------------------------

def efficiency_trend(history: List[Dict[str, Any]]) -> Optional[float]:
    """Compare first half vs second half RYE averages."""
    n = len(history)
    if n < 4:
        return None

    mid = n // 2
    old = history[:mid]
    recent = history[mid:]

    def _avg(h: List[Dict[str, Any]]) -> Optional[float]:
        vals = [float(e.get("RYE")) for e in h if isinstance(e.get("RYE"), (int, float))]
        return sum(vals) / len(vals) if vals else None

    avg_old = _avg(old)
    avg_recent = _avg(recent)
    if avg_old is None or avg_recent is None:
        return None

    return avg_recent - avg_old


# ---------------------------------------------------------------------------
# NEW: Regression-based RYE slope (better for long runs)
# ---------------------------------------------------------------------------

def regression_rye_slope(history: List[Dict[str, Any]]) -> Optional[float]:
    """Compute a smoother RYE trend using linear regression.

    Returns:
        slope (float) or None
    """

    if not history or len(history) < 4:
        return None

    xs: List[float] = []
    ys: List[float] = []

    for i, entry in enumerate(history):
        v = entry.get("RYE")
        if isinstance(v, (int, float)):
            xs.append(float(i))
            ys.append(float(v))

    if len(xs) < 4:
        return None

    # Linear regression: slope = covariance / variance
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))

    if den == 0:
        return None

    return num / den


# ---------------------------------------------------------------------------
# Optional domain weighting hook
# ---------------------------------------------------------------------------

def apply_domain_weight(rye_value: float, domain: Optional[str] = None) -> float:
    """Optional RYE adjustment by domain.

    Not used yet, but ready for future presets:
        general → 1.0
        math → 1.1
        longevity → 1.15
    """

    if domain is None:
        return rye_value

    multipliers = {
        "general": 1.0,
        "math": 1.1,
        "longevity": 1.15,
    }

    return rye_value * multipliers.get(domain, 1.0)
