"""
RYE Metrics (Level 3, Swarm Aware, 90 Day Safe)

This upgraded module introduces:
--------------------------------
• ΔR with deeper signal components
• E with RYE aware cost fields for swarms and multi agent parallelization
• Noise resistant RYE
• Rolling RYE (windowed)
• Robust trend detection
• Regression slope
• Outlier robust median RYE
• Domain weighted RYE (math, longevity, general)
• Multi agent energy normalization
• Stability Index (NEW)
• Recovery Momentum (NEW)
• Repair Efficiency Signature (NEW)
• Per cycle RYE summaries (NEW)
• Run level diagnostics bundle (NEW)

Backwards compatibility:
    Existing callers that only pass the original arguments still work:
    - compute_delta_r(...) can be called without novelty_score or coherence_gain.
    - compute_energy(...) can be called without swarm_size or swarm_layering.
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
    Compute ΔR using Level 3 multi signal improvement analysis.

    New improvements:
    - novelty_score      -> reward discovering something new
    - coherence_gain     -> reward resolving contradictions cleanly

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
# E (Effort) - Level 3 Swarm Aware
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
    Compute energy cost with swarm layer normalization.

    New fields:
    - swarm_size     -> number of active agents
    - swarm_layering -> depth (roles x agents x branches)

    Existing calls that do not pass these fields still behave as before.
    """

    base_cost = float(len(actions_taken)) if actions_taken else 1.0

    cost_web = max(web_calls, 0) * 1.5
    cost_pubmed = max(pubmed_calls, 0) * 2.0
    cost_sem = max(semantic_calls, 0) * 2.0
    cost_pdf = max(pdf_ingestions, 0) * 2.5

    total = base_cost + cost_web + cost_pubmed + cost_sem + cost_pdf

    # Soft token costing
    if tokens_estimate is not None and tokens_estimate > 0:
        total += float(tokens_estimate) / 1000.0

    # Swarm penalty (prevents cheating via infinite agent spawning)
    swarm_size_eff = max(swarm_size, 1)
    swarm_layer_eff = max(swarm_layering, 1)
    swarm_penalty = 1.0 + ((swarm_size_eff - 1) * 0.05) + ((swarm_layer_eff - 1) * 0.1)
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
    if not history or window <= 0:
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
# Median RYE - Noise Resistant
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
# Regression Based Trend - Best for Long Runs
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
    Measures variance normalized stability of RYE values.

    1.0 = perfectly stable improvements
    0.0 = chaotic swings
    """
    vals = [float(e["RYE"]) for e in history if isinstance(e.get("RYE"), (int, float))]
    if len(vals) < 4:
        return None

    mean = sum(vals) / len(vals)
    var = statistics.pvariance(vals)

    if var == 0:
        # Perfectly flat. If mean is positive, treat as fully stable.
        return 1.0 if mean >= 0 else 0.0

    score = mean / (var + 1e-6)
    # Soft scaling into [0, 1]
    score *= 0.25
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Recovery Momentum (NEW)
# ---------------------------------------------------------------------------

def recovery_momentum(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Measures upward acceleration in late stage runs.

    Good for long missions (weeks to 90 day runs) where you want to know
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
    Optional domain specific multiplier for RYE.

    This allows you to treat certain domains as slightly higher value
    per unit RYE, for example:
        general   -> 1.00
        math      -> 1.10
        longevity -> 1.15
        ai        -> 1.08
        biology   -> 1.12
    """
    if not isinstance(rye_value, (int, float)):
        return 0.0

    if domain is None:
        return float(rye_value)

    multipliers = {
        "general": 1.0,
        "math": 1.10,
        "longevity": 1.15,
        "ai": 1.08,
        "biology": 1.12,
    }

    factor = multipliers.get(domain.lower(), 1.0)
    result = float(rye_value) * factor
    return result


# ---------------------------------------------------------------------------
# Per cycle helper: full RYE summary with domain weighting
# ---------------------------------------------------------------------------

def build_cycle_rye_summary(
    *,
    delta_r: float,
    energy_e: float,
    domain: Optional[str] = None,
    role: Optional[str] = None,
    cycle_index: Optional[int] = None,
    extra_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a structured per cycle RYE summary used by TGRM and CoreAgent.

    This helper is the canonical place to:
    - compute raw RYE
    - apply domain weighting
    - attach basic metadata (domain, role, cycle_index)
    - carry forward any extra signals (novelty, coherence, biomarkers, etc)
    """
    rye_raw = compute_rye(delta_r, energy_e)
    rye_weighted = apply_domain_weight(rye_raw, domain)

    summary: Dict[str, Any] = {
        "RYE": rye_raw,
        "RYE_weighted": rye_weighted,
        "delta_r": float(delta_r),
        "energy_e": float(energy_e),
    }

    if domain is not None:
        summary["domain"] = domain
    if role is not None:
        summary["role"] = role
    if cycle_index is not None:
        summary["cycle_index"] = int(cycle_index)

    if extra_signals:
        for k, v in extra_signals.items():
            summary[k] = v

    return summary


# ---------------------------------------------------------------------------
# Swarm aggregation helper
# ---------------------------------------------------------------------------

def aggregate_swarm_round(
    role_summaries: List[Dict[str, Any]],
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregate per role summaries into a swarm round signature.

    This is useful in run_swarm_continuous, where a round consists of
    multiple logical agents writing summaries.

    Returns a dict with:
        - avg_rye
        - avg_rye_weighted
        - max_rye
        - min_rye
        - roles_count
    """
    rye_vals: List[float] = []
    rye_weighted_vals: List[float] = []

    for s in role_summaries:
        v = s.get("RYE")
        if isinstance(v, (int, float)):
            rye_vals.append(float(v))

        vw = s.get("RYE_weighted")
        if isinstance(vw, (int, float)):
            rye_weighted_vals.append(float(vw))
        elif isinstance(v, (int, float)) and domain is not None:
            rye_weighted_vals.append(apply_domain_weight(float(v), domain))

    result: Dict[str, Any] = {
        "roles_count": len(role_summaries),
    }

    if rye_vals:
        result["avg_rye"] = sum(rye_vals) / len(rye_vals)
        result["max_rye"] = max(rye_vals)
        result["min_rye"] = min(rye_vals)

    if rye_weighted_vals:
        result["avg_rye_weighted"] = sum(rye_weighted_vals) / len(rye_weighted_vals)

    return result


# ---------------------------------------------------------------------------
# Run level diagnostics bundle (Repair Efficiency Signature)
# ---------------------------------------------------------------------------

def build_run_diagnostics(
    history: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    window: int = 10,
) -> Dict[str, Any]:
    """
    Build a compact diagnostics bundle over a full run.

    This is the main Repair Efficiency Signature object for a session.
    It is safe to compute at any time, including long 24 hour or 90 day runs.

    Returns:
        {
          "count": int,
          "rye_avg": Optional[float],
          "rye_median": Optional[float],
          "rye_last": Optional[float],
          "rolling_rye": Optional[float],
          "trend_simple": Optional[float],
          "trend_slope": Optional[float],
          "stability_index": Optional[float],
          "recovery_momentum": Optional[float],
          "domain": Optional[str],
        }
    """
    count = len(history)
    rolling_val = rolling_rye(history, window=window)
    med_val = median_rye(history)
    trend_val = efficiency_trend(history)
    slope_val = regression_rye_slope(history)
    stab_val = stability_index(history)
    rec_val = recovery_momentum(history)

    rye_vals = [float(e["RYE"]) for e in history if isinstance(e.get("RYE"), (int, float))]
    rye_avg = sum(rye_vals) / len(rye_vals) if rye_vals else None
    rye_last = rye_vals[-1] if rye_vals else None

    diagnostics: Dict[str, Any] = {
        "count": count,
        "rye_avg": rye_avg,
        "rye_median": med_val,
        "rye_last": rye_last,
        "rolling_rye": rolling_val,
        "trend_simple": trend_val,
        "trend_slope": slope_val,
        "stability_index": stab_val,
        "recovery_momentum": rec_val,
    }

    if domain is not None:
        diagnostics["domain"] = domain

    if rye_avg is not None and domain is not None:
        diagnostics["rye_avg_weighted"] = apply_domain_weight(rye_avg, domain)

    return diagnostics
