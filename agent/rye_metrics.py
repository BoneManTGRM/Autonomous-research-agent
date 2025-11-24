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
• Tool RYE diagnostics for tool_events (NEW)

Backwards compatibility:
    Existing callers that only pass the original arguments still work:
    - compute_delta_r(...) can be called without novelty_score or coherence_gain.
    - compute_energy(...) can be called without swarm_size or swarm_layering.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import statistics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert to float if numeric, otherwise return None."""
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize history to only entries that contain numeric RYE.

    This makes downstream metrics more robust and predictable for long runs
    where some cycles might be missing RYE or contain partial data.
    """
    if not isinstance(history, list):
        return []

    norm: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        v = _safe_float(entry.get("RYE"))
        if v is None:
            continue
        e = dict(entry)
        e["RYE"] = v
        norm.append(e)
    return norm


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
    """RYE = ΔR / E, with a defensive zero floor for bad E."""
    e = float(energy_e)
    if e <= 0:
        return 0.0
    return float(delta_r) / e


# ---------------------------------------------------------------------------
# Rolling RYE (mean)
# ---------------------------------------------------------------------------

def rolling_rye(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """Compute a simple rolling average of RYE over the last N cycles."""
    if not history or window <= 0:
        return None

    norm = normalize_history(history)
    if not norm:
        return None

    recent = norm[-window:]
    vals = [float(entry["RYE"]) for entry in recent]

    if not vals:
        return None

    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Robust rolling RYE (median style)
# ---------------------------------------------------------------------------

def robust_rolling_rye(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """
    Median based rolling RYE.

    This is more robust to occasional spikes or outliers, and useful
    for noisy long runs with irregular repair bursts.
    """
    if not history or window <= 0:
        return None

    norm = normalize_history(history)
    if not norm:
        return None

    recent = norm[-window:]
    vals = [float(entry["RYE"]) for entry in recent]

    if not vals:
        return None

    return statistics.median(vals)


# ---------------------------------------------------------------------------
# Median RYE - Noise Resistant
# ---------------------------------------------------------------------------

def median_rye(history: List[Dict[str, Any]]) -> Optional[float]:
    """Median RYE across all cycles, robust to outliers."""
    norm = normalize_history(history)
    if not norm:
        return None

    vals = [float(e["RYE"]) for e in norm]
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
    norm = normalize_history(history)
    n = len(norm)
    if n < 4:
        return None

    mid = n // 2
    old = norm[:mid]
    recent = norm[mid:]

    def _avg(h: List[Dict[str, Any]]) -> Optional[float]:
        vals = [float(e["RYE"]) for e in h]
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
    norm = normalize_history(history)
    if len(norm) < 4:
        return None

    xs: List[float] = []
    ys: List[float] = []

    for i, entry in enumerate(norm):
        xs.append(float(i))
        ys.append(float(entry["RYE"]))

    n = len(xs)
    if n < 2:
        return None

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
    norm = normalize_history(history)
    if len(norm) < 4:
        return None

    vals = [float(e["RYE"]) for e in norm]
    mean = sum(vals) / len(vals)
    var = statistics.pvariance(vals)

    if var == 0:
        return 1.0 if mean >= 0 else 0.0

    score = mean / (var + 1e-6)
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
# Distribution Shape: Percentiles
# ---------------------------------------------------------------------------

def rye_percentiles(
    history: List[Dict[str, Any]],
    *,
    q_low: float = 0.1,
    q_mid: float = 0.5,
    q_high: float = 0.9,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute approximate low, median, and high RYE percentiles.

    This gives you a quick sense of distribution shape:
        low  -> pessimistic floor
        mid  -> central tendency (usually close to median_rye)
        high -> optimistic ceiling
    """
    norm = normalize_history(history)
    if len(norm) == 0:
        return None, None, None

    vals = sorted(float(e["RYE"]) for e in norm)
    n = len(vals)
    if n == 0:
        return None, None, None

    def _percentile(arr: List[float], q: float) -> float:
        q_clamped = min(max(q, 0.0), 1.0)
        if n == 1:
            return arr[0]
        idx = q_clamped * (n - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, n - 1)
        frac = idx - i0
        return arr[i0] * (1.0 - frac) + arr[i1] * frac

    low = _percentile(vals, q_low)
    mid = _percentile(vals, q_mid)
    high = _percentile(vals, q_high)
    return low, mid, high


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
        v = _safe_float(s.get("RYE"))
        if v is not None:
            rye_vals.append(v)

        vw = _safe_float(s.get("RYE_weighted"))
        if vw is not None:
            rye_weighted_vals.append(vw)
        elif v is not None and domain is not None:
            rye_weighted_vals.append(apply_domain_weight(v, domain))

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
          "robust_rolling_rye": Optional[float],
          "trend_simple": Optional[float],
          "trend_slope": Optional[float],
          "stability_index": Optional[float],
          "recovery_momentum": Optional[float],
          "low_percentile": Optional[float],
          "mid_percentile": Optional[float],
          "high_percentile": Optional[float],
          "domain": Optional[str],
        }
    """
    norm = normalize_history(history)
    count = len(norm)

    rolling_val = rolling_rye(norm, window=window)
    robust_rolling_val = robust_rolling_rye(norm, window=window)
    med_val = median_rye(norm)
    trend_val = efficiency_trend(norm)
    slope_val = regression_rye_slope(norm)
    stab_val = stability_index(norm)
    rec_val = recovery_momentum(norm)
    low_p, mid_p, high_p = rye_percentiles(norm)

    rye_vals = [float(e["RYE"]) for e in norm]
    rye_avg = sum(rye_vals) / len(rye_vals) if rye_vals else None
    rye_last = rye_vals[-1] if rye_vals else None

    diagnostics: Dict[str, Any] = {
        "count": count,
        "rye_avg": rye_avg,
        "rye_median": med_val,
        "rye_last": rye_last,
        "rolling_rye": rolling_val,
        "robust_rolling_rye": robust_rolling_val,
        "trend_simple": trend_val,
        "trend_slope": slope_val,
        "stability_index": stab_val,
        "recovery_momentum": rec_val,
        "low_percentile": low_p,
        "mid_percentile": mid_p,
        "high_percentile": high_p,
    }

    if domain is not None:
        diagnostics["domain"] = domain

    if rye_avg is not None and domain is not None:
        diagnostics["rye_avg_weighted"] = apply_domain_weight(rye_avg, domain)

    return diagnostics


# ---------------------------------------------------------------------------
# Tool RYE diagnostics (for MemoryStore.get_tool_stats hook)
# ---------------------------------------------------------------------------

def compute_tool_rye(
    tool_events: List[Dict[str, Any]],
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregate RYE style metrics per tool from tool_events.

    Expected tool_events format (from MemoryStore.log_tool_event):
        {
          "timestamp": str,
          "run_id": Optional[str],
          "goal": Optional[str],
          "domain": Optional[str],
          "role": Optional[str],
          "cycle_index": Optional[int],
          "tool_name": str,
          "status": str,            # "ok" or "error"
          "duration_seconds": Optional[float],
          "energy_cost": Optional[float],
          "rye_delta": Optional[float],
          "error": Optional[str],
          "extra": Optional[dict],
        }

    We compute for each tool:
        - events: total events
        - ok_events: non error events
        - error_events: events with status == "error"
        - sum_delta_r: sum of rye_delta where available
        - sum_energy: sum of energy_cost where available and positive
        - rye_avg: sum_delta_r / sum_energy if possible
        - rye_median: median per event rye_delta / energy_cost when both exist
        - rye_last: last per event ratio when both exist

    Returns:
        {
          "tools": {
             "browser": { ... },
             "sandbox": { ... },
             ...
          }
        }
    """
    if not isinstance(tool_events, list):
        tool_events = []

    per_tool: Dict[str, Dict[str, Any]] = {}

    for ev in tool_events:
        if not isinstance(ev, dict):
            continue
        if run_id is not None and ev.get("run_id") != run_id:
            continue

        name = ev.get("tool_name")
        if not name:
            continue

        t = per_tool.get(name) or {
            "events": 0,
            "ok_events": 0,
            "error_events": 0,
            "sum_delta_r": 0.0,
            "sum_energy": 0.0,
            "ratios": [],       # per event delta_r / energy_e
            "last_timestamp": None,
        }

        t["events"] = int(t.get("events", 0)) + 1

        status = ev.get("status", "ok")
        if status == "error":
            t["error_events"] = int(t.get("error_events", 0)) + 1
        else:
            t["ok_events"] = int(t.get("ok_events", 0)) + 1

        delta_r = _safe_float(ev.get("rye_delta"))
        energy_e = _safe_float(ev.get("energy_cost"))

        if delta_r is not None:
            t["sum_delta_r"] = float(t.get("sum_delta_r", 0.0)) + delta_r
        if energy_e is not None and energy_e > 0:
            t["sum_energy"] = float(t.get("sum_energy", 0.0)) + energy_e

        if delta_r is not None and energy_e is not None and energy_e > 0:
            t["ratios"].append(delta_r / energy_e)

        ts = ev.get("timestamp")
        if isinstance(ts, str):
            prev_ts = t.get("last_timestamp")
            if prev_ts is None or str(ts) > str(prev_ts):
                t["last_timestamp"] = ts

        per_tool[name] = t

    tools_out: Dict[str, Any] = {}

    for name, t in per_tool.items():
        ratios_raw = t.get("ratios", [])
        ratios: List[float] = [
            float(r) for r in ratios_raw if isinstance(r, (int, float))
        ]

        sum_energy = float(t.get("sum_energy", 0.0))
        sum_delta_r = float(t.get("sum_delta_r", 0.0))

        rye_avg: Optional[float] = None
        rye_median_val: Optional[float] = None
        rye_last: Optional[float] = None

        if sum_energy > 0 and sum_delta_r != 0:
            rye_avg = sum_delta_r / sum_energy

        if ratios:
            rye_median_val = statistics.median(ratios)
            rye_last = ratios[-1]

        events = int(t.get("events", 0))
        ok_events = int(t.get("ok_events", 0))
        error_events = int(t.get("error_events", 0))
        last_timestamp = t.get("last_timestamp")

        success_rate: Optional[float] = None
        if events > 0:
            success_rate = ok_events / float(events)

        tools_out[name] = {
            "events": events,
            "ok_events": ok_events,
            "error_events": error_events,
            "success_rate": success_rate,
            "sum_delta_r": sum_delta_r,
            "sum_energy": sum_energy,
            "rye_avg": rye_avg,
            "rye_median": rye_median_val,
            "rye_last": rye_last,
            "last_timestamp": last_timestamp,
        }

    return {"tools": tools_out}


# ======================================================================
#                         OPTION C EXTRAS
# ======================================================================

# RYE Volatility Signature (local self-diagnosis)

def rye_volatility_signature(
    history: List[Dict[str, Any]],
    *,
    window: int = 20,
) -> Dict[str, Optional[float]]:
    """
    Local volatility signature of RYE over a recent window.

    Returns:
        {
          "std": float or None,
          "range": float or None,
          "cv": float or None,
          "volatility_score": float or None in [0, 1],
        }

    volatility_score:
        ~1.0 -> very stable
        ~0.0 -> highly volatile
    """
    norm = normalize_history(history)
    if len(norm) == 0 or window <= 1:
        return {
            "std": None,
            "range": None,
            "cv": None,
            "volatility_score": None,
        }

    recent = norm[-window:]
    vals = [float(e["RYE"]) for e in recent]
    if len(vals) < 2:
        return {
            "std": None,
            "range": None,
            "cv": None,
            "volatility_score": None,
        }

    mean = sum(vals) / len(vals)
    std = statistics.pstdev(vals)
    vmin = min(vals)
    vmax = max(vals)
    r = vmax - vmin

    if mean == 0:
        cv = None
    else:
        cv = abs(std / mean)

    if cv is None:
        volatility_score = None
    else:
        raw = 1.0 / (1.0 + cv)
        volatility_score = max(0.0, min(raw, 1.0))

    return {
        "std": std,
        "range": r,
        "cv": cv,
        "volatility_score": volatility_score,
    }


# RYE Equilibrium Detector

def detect_rye_equilibrium(
    history: List[Dict[str, Any]],
    *,
    window: int = 30,
    slope_tolerance: float = 0.01,
) -> Dict[str, Optional[Any]]:
    """
    Detects whether the system appears to be in a RYE equilibrium zone.

    Conditions (heuristic):
        - enough recent points
        - regression slope magnitude below slope_tolerance
        - volatility is not excessive
    """
    norm = normalize_history(history)
    if len(norm) < window:
        return {
            "in_equilibrium": False,
            "reason": "not_enough_data",
            "window_size": len(norm),
            "local_slope": None,
            "local_volatility_score": None,
        }

    recent = norm[-window:]
    slope = regression_rye_slope(recent)
    vol_sig = rye_volatility_signature(recent, window=window)

    if slope is None:
        return {
            "in_equilibrium": False,
            "reason": "no_slope",
            "window_size": len(recent),
            "local_slope": None,
            "local_volatility_score": vol_sig.get("volatility_score"),
        }

    vol_score = vol_sig.get("volatility_score")
    if vol_score is None:
        in_eq = abs(slope) <= slope_tolerance
    else:
        in_eq = (abs(slope) <= slope_tolerance) and (vol_score >= 0.4)

    reason = "equilibrium" if in_eq else "non_equilibrium"

    return {
        "in_equilibrium": in_eq,
        "reason": reason,
        "window_size": len(recent),
        "local_slope": slope,
        "local_volatility_score": vol_score,
    }


# TGRM Harmonic Index (proxy)

def tgrm_harmonic_index(
    history: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Proxy for TGRM harmonic resonance.

    Approximated via:
        - stability_index
        - recovery_momentum

    Returns:
        float in [0, 1] or None
    """
    stab = stability_index(history)
    rec = recovery_momentum(history)

    if stab is None and rec is None:
        return None

    stab_norm = stab if stab is not None else 0.0
    rec_norm = min(max((rec or 0.0) / 5.0, 0.0), 1.0)

    idx = 0.6 * stab_norm + 0.4 * rec_norm
    return max(0.0, min(idx, 1.0))


# Breakthrough Probability Estimator (per run snapshot)

def estimate_breakthrough_probability(
    diagnostics: Dict[str, Any],
    *,
    domain: Optional[str] = None,
    horizon_hours: Optional[int] = None,
) -> Dict[str, Optional[Any]]:
    """
    Estimate the probability that this run configuration could yield a
    meaningful breakthrough within a given horizon.

    Inputs (from diagnostics):
        - trend_slope
        - recovery_momentum
        - stability_index
        - high_percentile
        - rye_avg
    """
    trend = diagnostics.get("trend_slope") or 0.0
    momentum = diagnostics.get("recovery_momentum") or 0.0
    stab = diagnostics.get("stability_index") or 0.0
    high_p = diagnostics.get("high_percentile") or 0.0
    rye_avg = diagnostics.get("rye_avg") or 0.0

    trend_component = max(trend, 0.0) * 8.0
    trend_component = min(trend_component, 0.30)

    momentum_component = min(momentum / 5.0, 0.25)
    stability_component = stab * 0.25

    high_component = 0.0
    if high_p > 1.0:
        high_component += 0.10
    if high_p > 2.0:
        high_component += 0.05
    if rye_avg > 0.5:
        high_component += 0.05

    domain_factor = 0.0
    if domain:
        dl = domain.lower()
        if dl in {"longevity", "biology", "bioai"}:
            domain_factor = 0.05
        elif dl in {"math", "ai"}:
            domain_factor = 0.03

    base_prob = trend_component + momentum_component + stability_component + high_component + domain_factor

    if horizon_hours is not None and horizon_hours > 0:
        if horizon_hours <= 24:
            horizon_scale = 0.8
        elif horizon_hours <= 7 * 24:
            horizon_scale = 1.0
        elif horizon_hours <= 90 * 24:
            horizon_scale = 1.2
        else:
            horizon_scale = 1.3
    else:
        horizon_scale = 1.0

    prob = base_prob * horizon_scale
    prob = max(0.0, min(prob, 0.95))

    return {
        "probability": prob,
        "trend_component": trend_component,
        "momentum_component": momentum_component,
        "stability_component": stability_component,
        "high_component": high_component,
        "domain_boost": domain_factor,
        "horizon_hours": horizon_hours,
    }


# Autonomy–Stability Safety Envelope

def autonomy_safety_envelope(
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify the current run's safety envelope qualitatively.

    Returns:
        {
          "state": "stable" | "oscillatory" | "collapsing" | "explosive" | "neutral" | "unknown",
          "details": { ... },
        }
    """
    stab = diagnostics.get("stability_index")
    slope = diagnostics.get("trend_slope")
    low_p = diagnostics.get("low_percentile")
    high_p = diagnostics.get("high_percentile")

    if stab is None or slope is None or low_p is None or high_p is None:
        return {
            "state": "unknown",
            "details": {
                "reason": "insufficient_metrics",
            },
        }

    spread = high_p - low_p

    if stab >= 0.7 and abs(slope) < 0.02:
        state = "stable"
        reason = "high_stability_flat_trend"
    elif stab >= 0.5 and slope > 0:
        state = "healthy_growth"
        reason = "stable_and_improving"
    elif stab >= 0.4 and spread > 1.5:
        state = "oscillatory"
        reason = "moderate_stability_high_spread"
    elif stab < 0.3 and slope < 0:
        state = "collapsing"
        reason = "low_stability_negative_trend"
    elif spread > 3.0:
        state = "explosive"
        reason = "very_high_spread"
    else:
        state = "neutral"
        reason = "mixed_signals"

    return {
        "state": state,
        "details": {
            "stability_index": stab,
            "trend_slope": slope,
            "low_percentile": low_p,
            "high_percentile": high_p,
            "spread": spread,
            "reason": reason,
        },
    }


# Run Tier Classifier (Tier 0 / 1 / 2 / 3)

def classify_run_tier(
    diagnostics: Dict[str, Any],
    *,
    breakthrough_prob: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Classify a run into tiers:

        Tier 0: unstable / chaotic
        Tier 1: normal working agent (positive average RYE)
        Tier 2: self-repairing, long-run stable
        Tier 3: "Major Breakthrough Zone" candidate
    """
    count = diagnostics.get("count") or 0
    stab = diagnostics.get("stability_index") or 0.0
    avg = diagnostics.get("rye_avg") or 0.0
    trend = diagnostics.get("trend_slope") or 0.0
    mid_p = diagnostics.get("mid_percentile") or 0.0

    if breakthrough_prob is None:
        bp = estimate_breakthrough_probability(diagnostics).get("probability") or 0.0
    else:
        bp = breakthrough_prob

    tier = "Tier 0"
    reason = "insufficient_data"

    if count < 4:
        tier = "Tier 0"
        reason = "too_few_cycles"
    elif stab < 0.2 and avg <= 0.0:
        tier = "Tier 0"
        reason = "unstable_negative_or_zero_rye"
    elif avg > 0.0 and stab >= 0.2:
        tier = "Tier 1"
        reason = "positive_rye_basic_stability"
    if avg > 0.2 and stab >= 0.4 and trend >= 0.0 and mid_p > 0.0:
        tier = "Tier 2"
        reason = "self_repairing_long_run_candidate"
    if avg > 0.4 and stab >= 0.6 and trend > 0.0 and mid_p > 0.2 and bp >= 0.30:
        tier = "Tier 3"
        reason = "high_stability_positive_trend_breakthrough_zone"

    return {
        "tier": tier,
        "reason": reason,
        "metrics": {
            "count": count,
            "stability_index": stab,
            "rye_avg": avg,
            "trend_slope": trend,
            "mid_percentile": mid_p,
            "breakthrough_probability": bp,
        },
    }


# Critical-Failure Early Warning Score

def early_failure_warning_score(
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Estimate how close the system may be to a failure state.

    High scores mean higher risk and may warrant human attention.
    """
    stab = diagnostics.get("stability_index")
    trend = diagnostics.get("trend_slope")
    avg = diagnostics.get("rye_avg")
    low_p = diagnostics.get("low_percentile")
    last = diagnostics.get("rye_last")

    if stab is None or trend is None or avg is None or low_p is None or last is None:
        return {
            "score": None,
            "factors": {
                "reason": "insufficient_metrics",
            },
        }

    inv_stability = 1.0 - max(0.0, min(stab, 1.0))

    downward = max(-trend * 10.0, 0.0)
    downward = min(downward, 1.0)

    avg_risk = 0.0
    if avg < 0:
        avg_risk = min(abs(avg), 1.0) * 0.7
    elif low_p < 0:
        avg_risk = 0.3

    last_risk = 0.0
    if last < 0:
        last_risk = min(abs(last), 1.0)

    score = (
        0.35 * inv_stability +
        0.25 * downward +
        0.20 * avg_risk +
        0.20 * last_risk
    )
    score = max(0.0, min(score, 1.0))

    return {
        "score": score,
        "factors": {
            "inv_stability": inv_stability,
            "downward": downward,
            "avg_risk": avg_risk,
            "last_risk": last_risk,
        },
    }


# 90-Day Breakthrough Likelihood Score

def breakthrough_likelihood_90d(
    diagnostics: Dict[str, Any],
    *,
    domain: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: estimate breakthrough likelihood specifically for
    a 90-day horizon, optionally conditioned on how long the system has
    already been running.
    """
    horizon_hours = 90 * 24
    base = estimate_breakthrough_probability(
        diagnostics,
        domain=domain,
        horizon_hours=horizon_hours,
    )
    prob = base.get("probability") or 0.0

    if hours_run_so_far is None or hours_run_so_far <= 0:
        progress_factor = 0.7
    else:
        progress_factor = min(hours_run_so_far / float(horizon_hours), 1.0)
        progress_factor = max(progress_factor, 0.5)

    adjusted_prob = max(0.0, min(prob * progress_factor, 0.99))

    return {
        "probability": adjusted_prob,
        "base_probability": prob,
        "horizon_hours": horizon_hours,
        "hours_run_so_far": hours_run_so_far,
    }


# Option C master bundle for a run

def build_option_c_signature(
    history: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
    window: int = 10,
) -> Dict[str, Any]:
    """
    High-level Option C self-diagnosis snapshot.

    Combines:
        - run diagnostics
        - volatility signature
        - equilibrium detection
        - TGRM harmonic index
        - breakthrough probability
        - 90-day breakthrough likelihood
        - autonomy safety envelope
        - early failure warning
        - run tier classification
    """
    diag = build_run_diagnostics(history, domain=domain, window=window)
    vol = rye_volatility_signature(history)
    eq = detect_rye_equilibrium(history)
    harm = tgrm_harmonic_index(history)
    bp = estimate_breakthrough_probability(diag, domain=domain, horizon_hours=None)
    bp90 = breakthrough_likelihood_90d(diag, domain=domain, hours_run_so_far=hours_run_so_far)
    env = autonomy_safety_envelope(diag)
    fail = early_failure_warning_score(diag)
    tier_info = classify_run_tier(diag, breakthrough_prob=bp.get("probability"))

    return {
        "diagnostics": diag,
        "volatility": vol,
        "equilibrium": eq,
        "tgrm_harmonic_index": harm,
        "breakthrough_probability": bp,
        "breakthrough_likelihood_90d": bp90,
        "autonomy_safety_envelope": env,
        "early_failure_warning": fail,
        "run_tier": tier_info,
    }
