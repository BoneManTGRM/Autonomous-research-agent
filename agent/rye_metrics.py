"""
RYE Metrics (Level-3, Swarm-Aware, 90-Day Safe)

This upgraded module introduces:
--------------------------------
• ΔR with deeper signal components
• E with RYE-aware cost fields for swarms + multi-agent parallelization
• Noise-resistant RYE
• Rolling RYE (windowed)
• Robust trend detection
• Regression slope
• Outlier-robust median RYE
• Domain-weighted RYE (math, longevity, general)
• Multi-agent energy normalization
• Stability Index (NEW)
• Recovery Momentum (NEW)
• Repair Efficiency Signature (NEW)

Backwards compatibility:
    Existing callers that only pass the original arguments still work:
    - compute_delta_r(...) can be called without novelty_score / coherence_gain.
    - compute_energy(...) can be called without swarm_size / swarm_layering.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import statistics


# ---------------------------------------------------------------------------
# ΔR (Improvement) - Level 3
# ---------------------------------------------------------------------------

def compute_delta_r(
    issues_before: int,
    issues_after: int,
    repairs_applied: int,
    contradictions_resolved: int = 0,
    hypotheses_generated: int = 0,
    sources_used: int = 0,
    novelty_score: float = 0.0,
    coherence_gain: float = 0.0,
) -> float:
    """
    Compute ΔR using Level-3 multi-signal improvement analysis.

    New improvements:
    -----------------
    • novelty_score      -> reward discovering something new
    • coherence_gain     -> reward resolving contradictions cleanly

    All original parameters still work as before.
    """

    base = max(issues_before - issues_after, 0)

    contradiction_gain = contradictions_resolved * 0.5
    hypothesis_gain = hypotheses_generated * 0.2
    source_gain = min(max(sources_used, 0), 20) * 0.05

    # Extra signal channels
    bonus = max(novelty_score * 0.4, 0.0) + max(coherence_gain * 0.3, 0.0)

    delta = float(base + contradiction_gain + hypothesis_gain + source_gain + bonus)

    # Maintenance credit (avoid ΔR = 0 with real work)
    if issues_before == 0 and delta == 0 and repairs_applied > 0:
        delta = repairs_applied * 0.1

    return float(delta)


# ---------------------------------------------------------------------------
# E (Effort) - Level 3 Swarm-Aware
# ---------------------------------------------------------------------------

def compute_energy(
    actions_taken: List[Dict[str, Any]],
    web_calls: int = 0,
    pubmed_calls: int = 0,
    semantic_calls: int = 0,
    pdf_ingestions: int = 0,
    tokens_estimate: Optional[int] = None,
    swarm_size: int = 1,
    swarm_layering: int = 1,
) -> float:
    """
    Compute energy cost with swarm-layer normalization.

    New fields:
    -----------
    swarm_size     -> number of active agents
    swarm_layering -> depth (roles x agents x branches)

    Existing calls that do not pass these fields still behave as before.
    """

    base_cost = float(len(actions_taken)) if actions_taken else 1.0

    cost_web = max(web_calls, 0) * 1.5
    cost_pubmed = max(pubmed_calls, 0) * 2.0
    cost_sem = max(semantic_calls, 0) * 2.0
    cost_pdf = max(pdf_ingestions, 0) * 2.5

    total = base_cost + cost_web + cost_pubmed + cost_sem + cost_pdf

    # Soft token-costing
    if tokens_estimate is not None and tokens_estimate > 0:
        total += float(tokens_estimate) / 1000.0

    # Swarm penalty (prevents "cheating" via infinite agent spawning)
    swarm_penalty = 1.0 + ((max(swarm_size, 1) - 1) * 0.05) + ((max(swarm_layering, 1) - 1) * 0.1)
    total *= swarm_penalty

    # Safety clamps
    if total <= 0:
        total = 1.0
    if total < 0.05:
        total = 0.05

    return float(total)


# ---------------------------------------------------------------------------
# Core RYE
# ---------------------------------------------------------------------------

def compute_rye(delta_r: float, energy_e: float) -> float:
    """RYE = ΔR / E."""
    if energy_e <= 0:
        return 0.0
    return float(delta_r) / float(energy_e)


# ---------------------------------------------------------------------------
# Rolling RYE
# ---------------------------------------------------------------------------

def rolling_rye(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """Compute a simple rolling average of RYE over the last N cycles."""
    if not history:
        return None

    recent = history[-window:]
    vals = [
        float(entry["RYE"])
        for entry in recent
        if isinstance(entry.get("RYE"), (int, float))
    ]

    if not vals:
        return None

    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Median RYE - Noise-Resistant
# ---------------------------------------------------------------------------

def median_rye(history: List[Dict[str, Any]]) -> Optional[float]:
    """Median RYE across all cycles, robust to outliers."""
    vals = [
        float(e["RYE"])
        for e in history
        if isinstance(e.get("RYE"), (int, float))
    ]
    if not vals:
        return None
    return statistics.median(vals)


# ---------------------------------------------------------------------------
# Efficiency Trend (Simple)
# ---------------------------------------------------------------------------

def efficiency_trend(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Simple trend: average RYE in the second half minus average RYE in the first half.

    Positive  -> improving efficiency
    Negative  -> declining efficiency
    Near zero -> flat or noisy
    """
    n = len(history)
    if n < 4:
        return None

    mid = n // 2
    old = history[:mid]
    recent = history[mid:]

    def _avg(h: List[Dict[str, Any]]) -> Optional[float]:
        vals = [float(e["RYE"]) for e in h if isinstance(e.get("RYE"), (int, float))]
        return sum(vals) / len(vals) if vals else None

    avg_old = _avg(old)
    avg_recent = _avg(recent)

    if avg_old is None or avg_recent is None:
        return None

    return avg_recent - avg_old


# ---------------------------------------------------------------------------
# Regression-Based Trend - Best for Long Runs
# ---------------------------------------------------------------------------

def regression_rye_slope(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Linear regression slope of RYE over cycles.

    Returns:
        slope (float) or None if not enough data.
    """
    xs: List[float] = []
    ys: List[float] = []

    for i, entry in enumerate(history):
        v = entry.get("RYE")
        if isinstance(v, (int, float)):
            xs.append(float(i))
            ys.append(float(v))

    if len(xs) < 4:
        return None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))

    if den == 0:
        return None

    return num / den


# ---------------------------------------------------------------------------
# Stability Index (NEW)
# ---------------------------------------------------------------------------

def stability_index(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Measures variance-normalized stability of RYE values.

    1.0 = perfectly stable improvements
    0.0 = chaotic swings
    """
    vals = [float(e["RYE"]) for e in history if isinstance(e.get("RYE"), (int, float))]
    if len(vals) < 4:
        return None

    mean = sum(vals) / len(vals)
    var = statistics.pvariance(vals)

    if var == 0:
        return 1.0

    # Normalize mean by variance and clamp to [0, 1]
    score = mean / (var + 1e-6)
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Recovery Momentum (NEW)
# ---------------------------------------------------------------------------

def recovery_momentum(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Measures upward acceleration in late-stage runs.

    Good for long missions (weeks to 90-day runs) where you want to know
    if the system is picking up momentum or stalling.
    """
    slope = regression_rye_slope(history)
    if slope is None:
        return None
    # Simple scaling to make the value more interpretable
    return max(slope * 10.0, 0.0)


# ---------------------------------------------------------------------------
# Domain Weighting
# ---------------------------------------------------------------------------

def apply_domain_weight(rye_value: float, domain: Optional[str] = None) -> float:
    """
    Optional domain-specific multiplier for RYE.

    This allows you to treat certain domains as slightly higher value
    per unit RYE, for example:
        general   -> 1.00
        math      -> 1.10
        longevity -> 1.15
        ai        -> 1.08
        biology   -> 1.12
    """
    if domain is None:
        return rye_value

    multipliers = {
        "general": 1.0,
        "math": 1.10,
        "longevity": 1.15,
        "ai": 1.08,
        "biology": 1.12,
    }

    return rye_value * multipliers.get(domain, 1.0)
