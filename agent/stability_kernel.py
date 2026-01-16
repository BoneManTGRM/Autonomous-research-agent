# agent/stability_kernel.py

"""
Stability Kernel for Reparodynamics and RYE based agents.

This module provides analytic helpers to understand the stability,
volatility, equilibrium windows, and recovery behavior of long runs.

It is designed to be:

- Purely analytic (no side effects, no IO)
- Safe to ignore (all functions are optional helpers)
- Backwards compatible with simple callers that only need a few metrics
- Rich enough for advanced engines (learning bursts, MSIL, Option C)

Key concepts:

    history:
        List of cycle dicts written by the TGRM loop. Typical fields:
            - "cycle" (int)
            - "RYE" (float)
            - "delta_R" (float)
            - "energy_E" (float)
            - "timestamp" (isoformat string)

    stability_index:
        Rough measure of how stable the RYE process is over a window.
        Higher is more stable. Normalized in [0, 1].

    recovery_momentum:
        How strongly the system recovers from negative shocks. Positive
        values indicate that RYE tends to climb after dips.

    volatility_signature:
        Statistical signature of RYE oscillation: std, range, regime
        classification, and a compact volatility score.

    equilibrium_window:
        Detection of plateau periods where RYE stabilizes in a band.

    critical_transitions:
        Detection of big structural changes in RYE, such as sudden
        increases, collapses, or regime switches.

The kernel accepts either raw history or precomputed RYE sequences.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import statistics
import math


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """
    Convert any numeric-like value into a finite float.  

    This helper attempts to coerce integers, floats and numeric strings into
    Python ``float`` objects.  If the input is ``None``, empty, not numeric
    or results in a NaN/inf value, the function returns ``None`` instead.

    Examples:

        >>> _safe_float("42")
        42.0
        >>> _safe_float("\t")
        None
        >>> _safe_float(float('nan'))
        None

    Parameters
    ----------
    value:
        A candidate value to convert. It may be a number, a numeric string
        or any other Python object. Unsupported types are ignored quietly.

    Returns
    -------
    Optional[float]
        The converted finite float or ``None`` if conversion is impossible
        or the resulting value is not finite.
    """
    try:
        # Fast path for native numeric types.
        if isinstance(value, (int, float)):
            result = float(value)
        elif isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            result = float(v)
        else:
            # Unknown types cannot be converted
            return None
        # Guard against NaN and infinity which can poison downstream metrics
        if math.isfinite(result):
            return result
        else:
            return None
    except Exception:
        # Conversion failed; return None to skip value
        return None


def _finite_series(series: List[float]) -> List[float]:
    """
    Return only finite numeric values from a sequence.

    This helper iterates through a sequence of numbers and keeps only
    those elements that can be coerced to floats and are finite (not
    NaN, +inf or -inf). Non-numeric entries are ignored.

    Parameters
    ----------
    series:
        A sequence of values which may include numbers, strings or other
        objects.

    Returns
    -------
    List[float]
        A list containing only finite floats extracted from ``series``.
    """
    out: List[float] = []
    for x in series:
        if isinstance(x, (int, float)):
            try:
                f = float(x)
            except Exception:
                continue
            if math.isfinite(f):
                out.append(f)
    return out


def _extract_series(history: List[Dict[str, Any]], field: str) -> List[float]:
    """Extract a numeric series from history by field name.

    Non numeric values are skipped silently.
    """
    series: List[float] = []
    for entry in history:
        v = _safe_float(entry.get(field))
        if v is not None:
            series.append(v)
    return series


def _rolling_mean(series: List[float], window: int) -> List[Optional[float]]:
    """Compute rolling mean with given window size.

    Aligns each mean with the last index in the window. Values before
    the first complete window are None.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if not series:
        return []
    out: List[Optional[float]] = [None] * len(series)
    acc = 0.0
    for i, v in enumerate(series):
        acc += v
        if i >= window:
            acc -= series[i - window]
        if i >= window - 1:
            out[i] = acc / float(window)
    return out


def _rolling_std(series: List[float], window: int) -> List[Optional[float]]:
    """Rolling standard deviation with window > 1.

    Uses population style denominator (n) for stability.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if not series:
        return []
    out: List[Optional[float]] = [None] * len(series)
    buf: List[float] = []
    for i, v in enumerate(series):
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) == window:
            mean_val = sum(buf) / float(window)
            var = sum((x - mean_val) ** 2 for x in buf) / float(window)
            out[i] = math.sqrt(max(var, 0.0))
    return out


def _subsequence(series: List[float], start: int, end: int) -> List[float]:
    """Safe slice [start, end) that never raises."""
    if start < 0:
        start = 0
    if end < 0:
        end = 0
    if start >= len(series):
        return []
    if end > len(series):
        end = len(series)
    if end <= start:
        return []
    return series[start:end]


# ---------------------------------------------------------------------------
# Stability index and recovery momentum
# ---------------------------------------------------------------------------

def stability_index_from_rye(
    rye_series: List[float],
    window: int = 20,
) -> Optional[float]:
    """Compute a stability index in [0, 1] from a RYE series.

    Heuristic design:

        - High stability if:
            * variance is low
            * sign of small changes is mixed (no strong drift)
            * no large jumps

        - Low stability if:
            * variance is high
            * changes are strongly directional
            * many large jumps or oscillations

    The exact formula is not unique. This one is chosen for robustness
    and simple interpretability.

    Returns None if there is not enough data.
    """
    rye_series = _finite_series(rye_series)
    if not rye_series:
        return None
    # Require at least three samples for a reliable stability estimate.
    if len(rye_series) < 3:
        return None

    w = max(3, min(window, len(rye_series)))
    tail = rye_series[-w:]

    mean_val = statistics.fmean(tail)
    var = statistics.pvariance(tail) if len(tail) > 1 else 0.0
    std = math.sqrt(max(var, 0.0))

    # Normalize variance relative to magnitude of mean
    scale = abs(mean_val) + 1e-6
    norm_std = std / scale

    # Look at step differences to estimate directional drift
    deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if deltas:
        mean_delta = statistics.fmean(deltas)
        abs_delta = statistics.fmean([abs(d) for d in deltas])
    else:
        mean_delta = 0.0
        abs_delta = 0.0

    if abs_delta > 0:
        drift_ratio = abs(mean_delta) / (abs_delta + 1e-6)
    else:
        drift_ratio = 0.0

    # Look for big jumps relative to std
    jumps = 0
    jump_threshold = std * 3.0 if std > 0 else scale * 0.05
    for d in deltas:
        if abs(d) > jump_threshold:
            jumps += 1
    jump_rate = jumps / max(len(deltas), 1)

    # If the mean is nonâpositive, consider the series unstable.
    if mean_val <= 0:
        return 0.0
    # Aggregate penalties into instability score
    instability = 0.0
    instability += min(norm_std, 3.0) / 3.0       # 0 to 1
    instability += min(drift_ratio, 1.0)          # 0 to 1
    instability += min(jump_rate * 3.0, 1.0)      # 0 to 1
    instability = min(max(instability / 3.0, 0.0), 1.0)

    # Convert to stability
    stability = 1.0 - instability
    return stability


def recovery_momentum_from_rye(
    rye_series: List[float],
    lookback: int = 30,
) -> Optional[float]:
    """Estimate recovery momentum from a RYE time series.

    Idea:

        1. Identify recent negative shocks (dips) in the tail.
        2. Measure how much RYE recovers after those dips.
        3. Combine them into a momentum value.

    Returns:

        Positive value:
            RYE tends to climb after setbacks.
        Negative value:
            RYE tends to keep falling after setbacks.
        0 or near 0:
            Mixed behavior or not enough data.
    """
    rye_series = _finite_series(rye_series)
    if not rye_series or len(rye_series) < 4:
        return None

    n = len(rye_series)
    lb = max(4, min(lookback, n))
    tail = rye_series[-lb:]

    deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if not deltas:
        return 0.0

    # Identify local negative shocks
    negative_indices = [
        i for i, d in enumerate(deltas, start=1)  # offset due to diff
        if d < 0
    ]
    if not negative_indices:
        # No obvious negative shocks, treat as neutral
        return 0.0

    recovery_values: List[float] = []
    for idx in negative_indices:
        # Compare value just after shock to average of a few points ahead
        before = tail[idx]
        ahead_slice = _subsequence(tail, idx + 1, idx + 4)
        if not ahead_slice:
            continue
        after_avg = statistics.fmean(ahead_slice)
        recovery = after_avg - before
        recovery_values.append(recovery)

    if not recovery_values:
        return 0.0

    # Normalize by magnitude of RYE to keep scale consistent
    mag = abs(statistics.fmean(tail)) + 1e-6
    norm_recoveries = [r / mag for r in recovery_values]
    momentum = statistics.fmean(norm_recoveries)

    # Clamp to a reasonable range to avoid wild outliers
    return max(min(momentum, 1.0), -1.0)

# ---------------------------------------------------------------------------
# Trajectory signals for compounding readiness
# ---------------------------------------------------------------------------

def trajectory_signals(
    history: List[Dict[str, Any]],
    window: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Compute simple signals that indicate whether a run is transitioning from
    exploration into compounding discovery mode.  These metrics look at
    hypothesis persistence, novelty variability, and citation reuse over
    recent cycles.

    Parameters
    ----------
    history: List[Dict[str, Any]]
        Sequence of cycle log entries as produced by TGRMLoop.  Each entry
        should include hypotheses (list of dicts), citations (list of dicts),
        and optional novelty metrics.
    window: int, optional
        Number of recent cycles to consider.  Defaults to 10.  Fewer cycles
        reduce noise when runs are short; more cycles provide smoother
        estimates when runs are long.

    Returns
    -------
    Dict[str, Optional[float]]
        A dictionary with keys:
        - ``hypothesis_survival_rate``: the average fraction of hypotheses
          that persist from one cycle to the next across the window.  Higher
          values suggest that the system is refining existing ideas rather
          than starting from scratch each time.
        - ``novelty_variance``: variance of novelty scores over the window.
          Decreasing variance may indicate convergence on a stable line of
          inquiry.  Returns None when novelty scores are unavailable.
        - ``constraint_reuse_rate``: the average fraction of citations
          reused in consecutive cycles.  Higher values suggest that
          constraints and evidence are being integrated rather than
          discarded.

    Notes
    -----
    These signals are intended as heuristics only.  They do not alter the
    stability index or volatility measures but can be consumed by higher
    level controllers or visualizations to decide when to extend runs.
    """
    if not isinstance(history, list) or not history:
        return {
            "hypothesis_survival_rate": None,
            "novelty_variance": None,
            "constraint_reuse_rate": None,
        }
    # Trim to the last ``window`` entries
    tail = history[-window:]
    # Hypothesis survival
    survival_rates: List[float] = []
    for prev, curr in zip(tail[:-1], tail[1:]):
        try:
            prev_hyps = prev.get("hypotheses") or []
            curr_hyps = curr.get("hypotheses") or []
        except Exception:
            prev_hyps, curr_hyps = [], []
        # Normalize to strings for simple comparison
        def _hyp_key(h: Any) -> str:
            if isinstance(h, dict):
                return str(h.get("title") or h.get("text") or h)
            return str(h)
        prev_keys = {_hyp_key(h).strip().lower() for h in prev_hyps if _hyp_key(h).strip()}
        curr_keys = {_hyp_key(h).strip().lower() for h in curr_hyps if _hyp_key(h).strip()}
        if not prev_keys:
            continue
        persisted = prev_keys.intersection(curr_keys)
        survival_rates.append(len(persisted) / float(len(prev_keys)))
    hypothesis_survival_rate: Optional[float] = None
    if survival_rates:
        try:
            hypothesis_survival_rate = float(sum(survival_rates)) / float(len(survival_rates))
        except Exception:
            hypothesis_survival_rate = None
    # Novelty variance
    novelty_scores: List[float] = []
    for entry in tail:
        try:
            n = entry.get("novelty_score") or entry.get("novelty") or entry.get("noveltyScore")
            if isinstance(n, (int, float)):
                novelty_scores.append(float(n))
        except Exception:
            continue
    novelty_variance: Optional[float] = None
    if novelty_scores:
        try:
            # Compute variance; require at least two samples
            if len(novelty_scores) >= 2:
                mean_val = statistics.fmean(novelty_scores)
                var = statistics.pvariance(novelty_scores)
                novelty_variance = float(var)
            else:
                novelty_variance = 0.0
        except Exception:
            novelty_variance = None
    # Constraint reuse (citation reuse)
    reuse_rates: List[float] = []
    for prev, curr in zip(tail[:-1], tail[1:]):
        try:
            prev_cites = prev.get("citations") or prev.get("sources") or []
            curr_cites = curr.get("citations") or curr.get("sources") or []
        except Exception:
            prev_cites, curr_cites = [], []
        def _cite_key(c: Any) -> str:
            if isinstance(c, dict):
                return str(c.get("url") or c.get("title") or c)
            return str(c)
        prev_keys = {_cite_key(c).strip().lower() for c in prev_cites if _cite_key(c).strip()}
        curr_keys = {_cite_key(c).strip().lower() for c in curr_cites if _cite_key(c).strip()}
        if not curr_keys:
            continue
        reused = curr_keys.intersection(prev_keys)
        reuse_rates.append(len(reused) / float(len(curr_keys)))
    constraint_reuse_rate: Optional[float] = None
    if reuse_rates:
        try:
            constraint_reuse_rate = float(sum(reuse_rates)) / float(len(reuse_rates))
        except Exception:
            constraint_reuse_rate = None
    return {
        "hypothesis_survival_rate": hypothesis_survival_rate,
        "novelty_variance": novelty_variance,
        "constraint_reuse_rate": constraint_reuse_rate,
    }


# ---------------------------------------------------------------------------
# Volatility and oscillation signatures
# ---------------------------------------------------------------------------

def volatility_signature_from_rye(
    rye_series: List[float],
    window: int = 20,
) -> Dict[str, Any]:
    """Compute volatility statistics and a compact score for a RYE sequence.

    Returns a dict:

        The dictionary contains both classical and robust statistics:

        * ``std``: population standard deviation of the tail segment
        * ``mean``: arithmetic mean of the tail segment
        * ``median``: median of the tail segment
        * ``mad``: median absolute deviation (a robust dispersion measure)
        * ``min``: minimum value in the tail
        * ``max``: maximum value in the tail
        * ``range``: difference between max and min
        * ``volatility_score``: heuristic volatility metric based on normalized std and range
        * ``regime``: qualitative regime label derived from the volatility_score
        * ``norm_std``: normalized standard deviation (std / abs(mean))
        * ``norm_range``: normalized range (range / abs(mean))
        * ``norm_mad``: normalized median absolute deviation (mad / abs(median))
    """
    rye_series = _finite_series(rye_series)
    if not rye_series:
        return {
            "std": None,
            "mean": None,
            "min": None,
            "max": None,
            "range": None,
            "volatility_score": None,
            "regime": "unknown",
        }

    if len(rye_series) < 2:
        v = rye_series[0]
        return {
            "std": 0.0,
            "mean": float(v),
            "min": float(v),
            "max": float(v),
            "range": 0.0,
            "volatility_score": 0.0,
            "regime": "low",
        }

    w = max(5, min(window, len(rye_series)))
    tail = rye_series[-w:]

    # Basic statistics
    mean_val = statistics.fmean(tail)
    std_val = statistics.pstdev(tail)
    min_val = min(tail)
    max_val = max(tail)
    rng_val = max_val - min_val

    # Robust statistics based on the median
    try:
        median_val = statistics.median(tail)
    except Exception:
        median_val = mean_val
    try:
        mad_val = statistics.median([abs(x - median_val) for x in tail])
    except Exception:
        mad_val = 0.0
    # Use the average absolute value as the scaling denominator for standard
    # deviation and range to avoid excessive blow-ups when the mean is near zero.
    avg_abs = statistics.fmean([abs(x) for x in tail]) + 1e-6
    # Use the median of absolute values to scale MAD (more robust than abs(median))
    median_abs = statistics.median([abs(x) for x in tail]) + 1e-6

    norm_std = std_val / avg_abs
    norm_range = rng_val / avg_abs
    norm_mad = mad_val / median_abs

    # Compact volatility metric (original heuristic)
    volatility_score = norm_std + 0.5 * norm_range

    # Assign regime based on volatility_score
    if volatility_score < 0.2:
        regime = "low"
    elif volatility_score < 0.6:
        regime = "medium"
    elif volatility_score < 1.5:
        regime = "high"
    else:
        regime = "extreme"

    return {
        "std": std_val,
        "mean": mean_val,
        "median": median_val,
        "mad": mad_val,
        "min": min_val,
        "max": max_val,
        "range": rng_val,
        "volatility_score": volatility_score,
        "regime": regime,
        "norm_std": norm_std,
        "norm_range": norm_range,
        "norm_mad": norm_mad,
    }


def oscillation_profile_from_rye(
    rye_series: List[float],
    window: int = 30,
) -> Dict[str, Any]:
    """Describe oscillation behavior of RYE in the tail.

    This is a lightweight alternative to full spectral analysis.

    Measures:

        - sign_flip_rate:
            How often the direction of change flips.
        - average_step_size:
            Average absolute change.
        - peak_to_trough:
            Distance between local peaks and troughs.
    """
    rye_series = _finite_series(rye_series)
    if not rye_series or len(rye_series) < 4:
        return {
            "sign_flip_rate": None,
            "average_step_size": None,
            "peak_to_trough": None,
        }

    w = max(4, min(window, len(rye_series)))
    tail = rye_series[-w:]

    deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if not deltas:
        return {
            "sign_flip_rate": 0.0,
            "average_step_size": 0.0,
            "peak_to_trough": 0.0,
        }

    # Sign flip rate
    flips = 0
    last_sign = 0
    for d in deltas:
        s = 1 if d > 0 else -1 if d < 0 else 0
        if s != 0 and last_sign != 0 and s != last_sign:
            flips += 1
        if s != 0:
            last_sign = s

    sign_flip_rate = flips / max(len(deltas) - 1, 1)

    # Average step size
    avg_step = statistics.fmean([abs(d) for d in deltas])

    # Peak to trough variation inside the window
    peak_to_trough = max(tail) - min(tail)

    return {
        "sign_flip_rate": sign_flip_rate,
        "average_step_size": avg_step,
        "peak_to_trough": peak_to_trough,
    }


# ---------------------------------------------------------------------------
# Equilibrium detection
# ---------------------------------------------------------------------------

def detect_equilibrium_window(
    rye_series: List[float],
    window: int = 25,
    tolerance: float = 0.20,
) -> Dict[str, Any]:
    """
    Detect an approximate equilibrium segment in a RYE series.

    This function searches for a contiguous plateau where the series oscillates
    within a relatively narrow band.  It uses a relative range criterion:
    a segment is considered an equilibrium if

    ``(max(segment) - min(segment)) / (abs(mean(segment)) + eps) <= tolerance``.

    The algorithm begins with a minimum required window length but then
    attempts to extend candidate windows as far as possible.  The longest
    qualifying segment is selected.  If no segment meets the criteria,
    the result indicates that no equilibrium was found.

    Parameters
    ----------
    rye_series:
        Sequence of numeric RYE values.  Short or empty sequences return
        immediately with an explanation.

    window:
        Minimum size of the equilibrium candidate window.  Values shorter
        than this are not considered unless the whole series is shorter,
        in which case the entire series is examined.

    tolerance:
        Relative amplitude allowed inside the equilibrium band.  Smaller
        values demand flatter plateaus.  Expressed as a fraction of the
        absolute mean of the segment.

    Returns
    -------
    Dict[str, Any]
        Summary describing whether an equilibrium was found and, if so,
        where it lies.  Keys include ``in_equilibrium``, ``start_index``,
        ``end_index`` (exclusive), ``equilibrium_fraction`` (the fraction of
        the series covered), ``mean``, ``range`` and ``reason``.
    """
    rye_series = _finite_series(rye_series)
    n = len(rye_series)
    # Handle empty input
    if n == 0:
        return {
            "in_equilibrium": False,
            "start_index": None,
            "end_index": None,
            "equilibrium_fraction": None,
            "mean": None,
            "range": None,
            "reason": "no_data",
        }

    # For very short series we cannot infer a meaningful plateau
    if n < max(5, window):
        return {
            "in_equilibrium": False,
            "start_index": None,
            "end_index": None,
            "equilibrium_fraction": None,
            "mean": statistics.fmean(rye_series),
            "range": max(rye_series) - min(rye_series),
            "reason": "too_short",
        }

    # Clamp window to sensible bounds
    min_window = max(3, min(window, n))

    best_start: Optional[int] = None
    best_end: Optional[int] = None
    best_fraction = 0.0
    best_mean: Optional[float] = None
    best_range: Optional[float] = None

    # Search for the largest contiguous segment satisfying the tolerance
    for start in range(0, n - min_window + 1):
        # Expand candidate until the tolerance is violated
        segment: List[float] = []
        for end in range(start, n):
            segment.append(rye_series[end])
            if len(segment) < min_window:
                continue
            mean_val = statistics.fmean(segment)
            rng = max(segment) - min(segment)
            denom = abs(mean_val) + 1e-6
            rel_range = rng / denom
            if rel_range <= tolerance:
                # Valid equilibrium candidate of current length
                eq_fraction = len(segment) / float(n)
                if eq_fraction > best_fraction:
                    best_fraction = eq_fraction
                    best_start = start
                    best_end = end + 1  # end is inclusive, convert to exclusive
                    best_mean = mean_val
                    best_range = rng
                # Continue extending; a longer segment might still satisfy tolerance
                continue
            else:
                # Tolerance violated; break early for this start
                break

    # Provide fallback if no plateau found
    if best_start is None:
        return {
            "in_equilibrium": False,
            "start_index": None,
            "end_index": None,
            "equilibrium_fraction": 0.0,
            "mean": statistics.fmean(rye_series),
            "range": max(rye_series) - min(rye_series),
            "reason": "no_plateau_found",
        }

    return {
        "in_equilibrium": True,
        "start_index": best_start,
        "end_index": best_end,
        "equilibrium_fraction": best_fraction,
        "mean": best_mean,
        "range": best_range,
        "reason": "plateau_detected",
    }


# ---------------------------------------------------------------------------
# Critical transitions and regime shifts
# ---------------------------------------------------------------------------

def detect_critical_transitions(
    rye_series: List[float],
    threshold_std: float = 2.5,
) -> List[Dict[str, Any]]:
    """Detect critical transitions where RYE jumps strongly.

    Uses a simple method:

        1. Compute differences d_t = RYE_t - RYE_{t-1}.
        2. Compute std of differences.
        3. Any step with |d_t| > threshold_std * std_diff is a critical step.

    Returns a list of dicts:

        {
            "index": int,         # index in rye_series where change ends
            "delta": float,       # step difference
            "kind": "up" | "down",
        }
    """
    rye_series = _finite_series(rye_series)
    if not rye_series or len(rye_series) < 3:
        return []

    deltas = [rye_series[i] - rye_series[i - 1] for i in range(1, len(rye_series))]
    if not deltas:
        return []

    std_diff = statistics.pstdev(deltas) or 0.0
    if std_diff <= 0:
        return []

    crits: List[Dict[str, Any]] = []
    thresh = threshold_std * std_diff
    for i, d in enumerate(deltas, start=1):
        if abs(d) > thresh:
            crits.append(
                {
                    "index": i,
                    "delta": d,
                    "kind": "up" if d > 0 else "down",
                }
            )
    return crits


# ---------------------------------------------------------------------------
# Top level stability profile
# ---------------------------------------------------------------------------

def build_stability_profile(
    history: List[Dict[str, Any]],
    window_short: int = 10,
    window_long: int = 50,
) -> Dict[str, Any]:
    """Build a rich stability profile from full cycle history.

    This is a convenience aggregator that callers can use directly or
    feed into higher level diagnostics.

    Returns a dict with:

        {
            "stability_index": float or None,
            "recovery_momentum": float or None,
            "volatility": { ... },
            "oscillation": { ... },
            "equilibrium": { ... },
            "critical_transitions": [ ... ],
            "series": {
                "rye": [...],
                "delta_r": [...],
                "energy": [...],
            },
            "windows": {
                "short": int,
                "long": int,
            },
        }
    """
    rye_series = _extract_series(history, "RYE")
    delta_series = _extract_series(history, "delta_R")
    energy_series = _extract_series(history, "energy_E")

    # Use long window for stability and volatility, short for finer detail
    w_long = max(5, min(window_long, len(rye_series) or window_long))
    w_short = max(3, min(window_short, len(rye_series) or window_short))

    stability = stability_index_from_rye(rye_series, window=w_long) if rye_series else None
    recovery = recovery_momentum_from_rye(rye_series, lookback=w_long) if rye_series else None
    volatility = volatility_signature_from_rye(rye_series, window=w_long) if rye_series else {}
    oscillation = oscillation_profile_from_rye(rye_series, window=w_short) if rye_series else {}
    equilibrium = detect_equilibrium_window(rye_series, window=w_long) if rye_series else {
        "in_equilibrium": False,
        "start_index": None,
        "end_index": None,
        "equilibrium_fraction": None,
        "mean": None,
        "range": None,
        "reason": "no_data",
    }
    crits = detect_critical_transitions(rye_series) if rye_series else []

    return {
        "stability_index": stability,
        "recovery_momentum": recovery,
        "volatility": volatility,
        "oscillation": oscillation,
        "equilibrium": equilibrium,
        "critical_transitions": crits,
        "series": {
            "rye": rye_series,
            "delta_r": delta_series,
            "energy": energy_series,
        },
        "windows": {
            "short": w_short,
            "long": w_long,
        },
    }


# ---------------------------------------------------------------------------
# Regime classification and compact summaries
# ---------------------------------------------------------------------------

def classify_stability_regime(
    profile: Dict[str, Any],
    expectations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify a stability profile into a high level regime.

    expectations is an optional dict such as:

        {
            "stability_index_target": 0.6,
            "recovery_momentum_target": 0.1,
            "max_oscillation_std": 0.25,
        }

    Returns a dict:

        {
            "label": "unstable" | "fragile" | "stable" | "robust",
            "score": float in [0, 1],
            "reasons": [str, ...],
        }
    """
    stability = profile.get("stability_index")
    recovery = profile.get("recovery_momentum")
    volatility = profile.get("volatility", {})
    oscillation = profile.get("oscillation", {})

    vol_score = volatility.get("volatility_score")
    osc_step = oscillation.get("average_step_size")

    if expectations is None:
        expectations = {}

    target_stab = expectations.get("stability_index_target", 0.6)
    target_rec = expectations.get("recovery_momentum_target", 0.1)
    max_osc_std = expectations.get("max_oscillation_std", 0.25)

    reasons: List[str] = []
    score_pieces: List[float] = []

    if isinstance(stability, (int, float)):
        if stability >= target_stab:
            score_pieces.append(1.0)
            reasons.append(f"stability_index {stability:.3f} at or above target {target_stab:.3f}")
        else:
            rel = stability / max(target_stab, 1e-6)
            score_pieces.append(max(rel, 0.0))
            reasons.append(f"stability_index {stability:.3f} below target {target_stab:.3f}")
    else:
        score_pieces.append(0.5)
        reasons.append("stability_index missing")

    if isinstance(recovery, (int, float)):
        if recovery >= target_rec:
            score_pieces.append(1.0)
            reasons.append(f"recovery_momentum {recovery:.3f} at or above target {target_rec:.3f}")
        else:
            # Allow mild negative recovery but penalize strongly if very negative
            if recovery <= -0.1:
                rel = 0.0
            else:
                rel = (recovery + 0.1) / (target_rec + 0.1 + 1e-6)
            score_pieces.append(max(min(rel, 1.0), 0.0))
            reasons.append(f"recovery_momentum {recovery:.3f} below target {target_rec:.3f}")
    else:
        score_pieces.append(0.5)
        reasons.append("recovery_momentum missing")

    if isinstance(vol_score, (int, float)):
        # Lower volatility is better
        if vol_score <= max_osc_std:
            score_pieces.append(1.0)
            reasons.append(f"volatility_score {vol_score:.3f} within preferred bound {max_osc_std:.3f}")
        else:
            rel = max(max_osc_std / (vol_score + 1e-6), 0.0)
            score_pieces.append(min(rel, 1.0))
            reasons.append(f"volatility_score {vol_score:.3f} above preferred bound {max_osc_std:.3f}")
    else:
        score_pieces.append(0.5)
        reasons.append("volatility_score missing")

    if isinstance(osc_step, (int, float)):
        # Smaller average step size hints at smoother dynamics
        if osc_step <= max_osc_std:
            score_pieces.append(1.0)
            reasons.append(f"average_step_size {osc_step:.3f} within preferred bound {max_osc_std:.3f}")
        else:
            rel = max(max_osc_std / (osc_step + 1e-6), 0.0)
            score_pieces.append(min(rel, 1.0))
            reasons.append(f"average_step_size {osc_step:.3f} above preferred bound {max_osc_std:.3f}")
    else:
        score_pieces.append(0.5)
        reasons.append("average_step_size missing")

    score = statistics.fmean(score_pieces) if score_pieces else 0.5

    if score < 0.25:
        label = "unstable"
    elif score < 0.5:
        label = "fragile"
    elif score < 0.8:
        label = "stable"
    else:
        label = "robust"

    return {
        "label": label,
        "score": score,
        "reasons": reasons,
    }


def summarize_for_ui(
    history: List[Dict[str, Any]],
    expectations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact summary suitable for dashboards or MSIL overlays.

    Combines stability profile and regime classification into a single
    dict with only numeric and short string fields so it is easy to
    serialize and display.
    """
    profile = build_stability_profile(history)
    regime = classify_stability_regime(profile, expectations=expectations)

    eq = profile.get("equilibrium", {}) or {}
    volatility = profile.get("volatility", {}) or {}

    out: Dict[str, Any] = {
        "stability_index": profile.get("stability_index"),
        "recovery_momentum": profile.get("recovery_momentum"),
        "equilibrium_fraction": eq.get("equilibrium_fraction"),
        "equilibrium_in_equilibrium": eq.get("in_equilibrium"),
        "equilibrium_reason": eq.get("reason"),
        "volatility_score": volatility.get("volatility_score"),
        "volatility_regime": volatility.get("regime"),
        "regime_label": regime.get("label"),
        "regime_score": regime.get("score"),
        "regime_reasons": regime.get("reasons", []),
        "critical_transition_count": len(profile.get("critical_transitions") or []),
    }
    return out


# ---------------------------------------------------------------------------
# Stateful StabilityKernel class
# ---------------------------------------------------------------------------

class StabilityKernel:
    """
    Stateful wrapper around the pure stability functions defined in this module.

    The `StabilityKernel` class tracks a rolling history of cycle metrics and
    produces a stability snapshot on each update.  It is designed to be
    instantiated by higher level components (for example the TGRM loop) and
    called once per cycle.  Internally it collects the supplied `RYE`,
    `delta_R`, and `energy_E` values, along with optional equilibrium
    information and arbitrary metadata, then derives a rich stability profile
    using the pure analytic helpers in this module.

    Only a fixed amount of history is retained (default 1000 cycles) to bound
    memory usage in long autonomous runs.  If more cycles are processed the
    oldest entries are dropped.

    Parameters
    ----------
    window_short: int, optional
        Size of the short window used for oscillation metrics.  Defaults to
        10.  Must be at least 3.
    window_long: int, optional
        Size of the long window used for stability and volatility metrics.
        Defaults to 50.  Must be at least 5.
    max_history: int, optional
        Maximum number of history entries to retain.  Defaults to 1000.

    Examples
    --------

    >>> sk = StabilityKernel()
    >>> snapshot = sk.update_from_cycle(rye=1.0, delta_r=2.0, energy_e=1.0, equilibrium_info={'equilibrium_label': 'high_equilibrium'})
    >>> 'stability_index' in snapshot
    True
    """

    def __init__(
        self,
        *,
        window_short: int = 10,
        window_long: int = 50,
        max_history: int = 1000,
    ) -> None:
        # Validate and clamp window sizes
        self.window_short = max(3, int(window_short))
        self.window_long = max(5, int(window_long))
        # Track history of cycles (each entry is a dict)
        self._history: List[Dict[str, Any]] = []
        # Bound how many entries we keep to avoid unbounded growth
        self._max_history = max(10, int(max_history))

    def update_from_cycle(
        self,
        *,
        rye: Optional[float] = None,
        delta_r: Optional[float] = None,
        energy_e: Optional[float] = None,
        equilibrium_info: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update the internal history with a new cycle and return a stability snapshot.

        All numeric inputs are sanitized via `_safe_float` to guard against
        invalid or non-finite values.  Missing values are permitted and will
        simply be skipped in the resulting series.

        Parameters
        ----------
        rye: float, optional
            The raw RYE value for the cycle.  If provided and finite this
            populates the ``RYE`` field of the stored history entry.  When
            omitted or non-numeric the entry will lack a ``RYE`` field.
        delta_r: float, optional
            The delta improvement value for the cycle.  Sanitized and stored
            under ``delta_R`` when finite.
        energy_e: float, optional
            The energy cost for the cycle.  Sanitized and stored under
            ``energy_E`` when finite.
        equilibrium_info: dict, optional
            Any equilibrium metadata computed by other components.  Copied
            directly into the history entry under the key ``equilibrium``.
        meta: dict, optional
            Arbitrary metadata associated with the cycle.  Stored under
            ``meta`` for later inspection (not used by stability metrics).

        Returns
        -------
        Dict[str, Any]
            A stability snapshot containing at least ``stability_index`` and
            ``recovery_momentum`` keys, along with volatility, oscillation,
            equilibrium and critical transitions.  Additional fields may be
            included for diagnostics.
        """
        # Build a new entry and sanitize numeric fields
        entry: Dict[str, Any] = {}
        v_rye = _safe_float(rye)
        if v_rye is not None:
            entry["RYE"] = v_rye
        v_dr = _safe_float(delta_r)
        if v_dr is not None:
            entry["delta_R"] = v_dr
        v_e = _safe_float(energy_e)
        if v_e is not None:
            entry["energy_E"] = v_e

        # Attach equilibrium info if provided and dict-like
        if isinstance(equilibrium_info, dict) and equilibrium_info:
            entry["equilibrium"] = dict(equilibrium_info)
        # Attach meta if provided
        if isinstance(meta, dict) and meta:
            entry["meta"] = dict(meta)

        # Append entry to history and enforce history cap
        if entry:
            self._history.append(entry)
            if len(self._history) > self._max_history:
                # Drop oldest entries to bound memory
                excess = len(self._history) - self._max_history
                if excess > 0:
                    self._history = self._history[excess:]

        # Build stability profile based on accumulated history
        profile = build_stability_profile(
            self._history,
            window_short=self.window_short,
            window_long=self.window_long,
        )
        # Prepare snapshot with key metrics
        snapshot: Dict[str, Any] = {
            "stability_index": profile.get("stability_index"),
            "recovery_momentum": profile.get("recovery_momentum"),
            "volatility": profile.get("volatility"),
            "oscillation": profile.get("oscillation"),
            "equilibrium": profile.get("equilibrium"),
            "critical_transitions": profile.get("critical_transitions"),
            "windows": profile.get("windows"),
            # Optionally include raw series length for diagnostics
            "history_length": len(self._history),
        }
        return snapshot
