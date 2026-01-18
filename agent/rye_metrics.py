"""
RYE Metrics (Level 3, Swarm Aware, 90 Day Safe, Learning Aware 10x Ready)

This upgraded module introduces:
--------------------------------
 - Delta R with deeper signal components
 - E with RYE aware cost fields for swarms and multi agent parallelization
 - Noise resistant RYE
 - Rolling RYE (windowed)
 - Robust trend detection
 - Regression slope
 - Outlier robust median RYE
 - Domain weighted RYE (math, longevity, general)
 - Multi agent energy normalization
 - Stability Index (NEW)
 - Recovery Momentum (NEW)
 - Repair Efficiency Signature (NEW)
 - Per cycle RYE summaries (NEW)
 - Run level diagnostics bundle (NEW)
 - Tool RYE diagnostics for tool_events (NEW)
 - Learning adjusted RYE fields for 10x style cognitive speed factors (NEW)

Backwards compatibility:
    Existing callers that only pass the original arguments still work:
    - compute_delta_r(...) can be called without novelty_score or coherence_gain.
    - compute_energy(...) can be called without swarm_size or swarm_layer.
    - build_cycle_rye_summary(...) can be called without learning_speed_factor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import statistics
import math


# ---------------------------------------------------------------------------
# Tier configuration constants for tuning
# ---------------------------------------------------------------------------

# In the original RYE metrics implementation a minimum of four cycles were
# required before any tiering or advanced run diagnostics would be
# computed.  This meant that short runs (e.g. three cycles) produced
# ``n/a`` values in the learning dashboards and prevented the autonomy
# view from showing higher tiers.  To make the diagnostics usable for
# short, finite runs Ã¢ÂÂ which are common when exploring longevity
# questions Ã¢ÂÂ the longevityÃ¢ÂÂonly build lowers the minimum cycles
# threshold to 3.  Runs with three or more cycles will now produce
# full diagnostics instead of being flagged as insufficient.
# Reduce the cycle threshold to allow diagnostics to appear even for very short
# runs.  When set to 1 the learning dashboards will compute tiering and other
# runÃ¢ÂÂlevel signals as soon as a single cycle is available.  This change
# avoids "n/a" results when exploring quick, finite runs.
# To avoid prematurely classifying very short runs, require at least
# three cycles before tiering heuristics are applied.  Previously this
# threshold was lowered to 1 to show diagnostics even on tiny runs, but
# this can produce noisy or misleading tiers.  A minimum of three cycles
# strikes a balance between immediacy and reliability.
MIN_CYCLES_FOR_TIERING: int = 3

# Run level Tier 2 thresholds
TIER2_MIN_AVG_RYE: float = 0.2
TIER2_MIN_STABILITY: float = 0.4

# Run level Tier 3 thresholds
TIER3_MIN_AVG_RYE: float = 0.4
TIER3_MIN_STABILITY: float = 0.6
TIER3_MIN_TREND: float = 0.0
TIER3_MIN_MID_PERCENTILE: float = 0.2
TIER3_MIN_BREAKTHROUGH_PROB: float = 0.30

# Per cycle Tier thresholds
CYCLE_TIER1_MIN_RYE: float = 0.0
CYCLE_TIER2_MIN_RYE: float = 0.3
CYCLE_TIER2_MIN_DELTA_R: float = 1.0
CYCLE_TIER3_MIN_RYE: float = 0.6
CYCLE_TIER3_MIN_DELTA_R: float = 2.0
CYCLE_TIER3_MIN_NOVELTY: float = 0.4
CYCLE_TIER3_MIN_COHERENCE: float = 0.4

# ---------------------------------------------------------------------------
# RYE sanity caps
#
# During long runs with multiple agents, erroneous calculations or runaway
# feedback can inflate RYE values far beyond realistic bounds.  Extremely
# large RYE values poison averages and diagnostic metrics, producing
# misleading "max" and "avg" statistics.  To guard against this, the
# perâcycle RYE is clamped to a reasonable maximum.  You can adjust this
# constant if your domain demands higher theoretical RYE values.
MAX_CYCLE_RYE: float = 10.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """
    Convert to float whenever possible.

    Accepts ints, floats, and numeric strings. Returns None when conversion
    fails, or when the result is NaN/Inf.

    This makes the module more robust to mixed-type histories.
    """
    try:
        if value is None:
            return None
        v = float(value)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _extract_rye_from_entry(entry: Dict[str, Any]) -> Optional[float]:
    """
    Extract RYE from a history entry.

    Priority:
      1) direct RYE fields ("RYE" or "rye")
      2) derived from delta_r / energy_e (or common alternates)
    """
    v = _safe_float(entry.get("RYE"))
    if v is None:
        v = _safe_float(entry.get("rye"))
    if v is not None:
        return v

    # Derived fields (common keys used across agent logs)
    delta = _safe_float(entry.get("delta_r"))
    if delta is None:
        delta = _safe_float(entry.get("delta_R"))
    if delta is None:
        delta = _safe_float(entry.get("rye_delta"))

    energy = _safe_float(entry.get("energy_e"))
    if energy is None:
        energy = _safe_float(entry.get("energy_cost"))
    if energy is None:
        energy = _safe_float(entry.get("E"))
    if energy is None:
        energy = _safe_float(entry.get("energy"))

    # Only consider improvements that actually reduce the defect count.
    # Treat cycles with non-positive delta_R as non-contributory. Likewise, skip
    # if energy is missing or non-positive to avoid skewing averages.
    if delta is None or delta <= 0 or energy is None or energy <= 0:
        return None

    try:
        ratio = float(delta) / float(energy)
    except Exception:
        return None
    # Clamp runaway RYE values to a sane maximum.  Without this guard,
    # extremely small energy values (or spurious delta calculations) can
    # produce huge RYE numbers that distort summaries.  See MAX_CYCLE_RYE
    # above for the limit.
    if ratio > MAX_CYCLE_RYE:
        ratio = MAX_CYCLE_RYE
    return ratio


def normalize_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize history to entries that contain numeric RYE.

    Upgrade:
      - If "RYE" is missing, derive it from delta_r/energy_e when possible.

    This makes downstream metrics more robust and predictable for long runs
    where some cycles might be missing RYE or contain partial data.

    Note:
        This operates on the canonical "RYE" key. Learning adjusted
        variants are handled separately and do not change this function
        to keep legacy behavior stable.
    """
    if not isinstance(history, list):
        return []

    norm: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue

        rye = _extract_rye_from_entry(entry)
        if rye is None:
            continue

        e = dict(entry)
        e["RYE"] = float(rye)
        norm.append(e)

    return norm


def _extract_rye_series(values_or_history: List[Any]) -> List[float]:
    """
    Flexible extractor used by many metrics.

    Accepts either:
        - a list of history dicts with "RYE" fields (or with delta_r/energy_e)
        - a list of raw numeric values
        - a mixed list of both

    Returns a clean list[float] of RYE values in the original order.

    This is what makes the module compatible with both:
        - history style callers (TGRM, CoreAgent, Streamlit UI)
        - vector style callers (msil.py, IQ probes, etc)
    """
    if not isinstance(values_or_history, list) or not values_or_history:
        return []

    out: List[float] = []
    for item in values_or_history:
        if isinstance(item, dict):
            v = _extract_rye_from_entry(item)
        else:
            v = _safe_float(item)

        if v is not None:
            out.append(float(v))

    return out


# ---------------------------------------------------------------------------
# Delta R (Improvement) - Level 3
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
    Compute Delta R using Level 3 multi-signal improvement analysis.

    New improvements:
    - novelty_score      -> reward discovering something new
    - coherence_gain     -> reward resolving contradictions cleanly

    All original parameters still work as before.
    """
    base = max(issues_before - issues_after, 0)

    # Improve delta R weightings to better reward highâvalue actions.
    # Contradiction resolution and novelty/coherence gains are key signals of
    # genuine progress. Increase their contribution to delta R.  Hypothesis
    # generation still adds value but with a moderate weight to discourage
    # uncontrolled idea spamming.  Sources provide evidence and thus scale
    # the improvement more strongly than before.  Bonus terms reward
    # novelty and coherence more aggressively.
    # Increase the contribution from resolved contradictions.  Discovery often
    # hinges on reconciling conflicting evidence, so reward each
    # resolved contradiction more strongly.  A higher factor boosts
    # delta R when the agent makes sense of disparate sources.
    # Increase the contribution from resolved contradictions further.  Discovery often
    # hinges on reconciling conflicting evidence, so reward each
    # resolved contradiction more strongly.  A higher factor boosts
    # delta R when the agent makes sense of disparate sources.
    # Increase the contribution from resolved contradictions, hypotheses, and sources.
    # Resolving contradictions is particularly highâvalue because it indicates the agent
    # reconciled conflicting evidence, so give each resolution a weight of 4.0.
    contradiction_gain = contradictions_resolved * 4.0
    # Hypothesis generation is more valuable than before; boost its weight to 2.0 to reward
    # creative thinking while avoiding runaway idea spamming.
    hypothesis_gain = hypotheses_generated * 2.0
    # Each unique source now contributes 1.0 instead of 0.5, encouraging thorough citation.
    source_gain = min(max(sources_used, 0), 20) * 1.0

    # Reward discovering new mechanisms and improved coherence even more
    # aggressively.  Novel findings and coherent integration of evidence
    # are hallmarks of meaningful discovery.  Higher multipliers here
    # encourage exploration and synthesis while still keeping the bonus
    # term bounded.  Increase novelty and coherence multipliers.
    # Amplify novelty and coherence bonuses to reward groundbreaking findings and
    # wellâintegrated evidence. Novelty is now multiplied by 3.0 and coherence by 2.0.
    bonus = max(novelty_score * 3.0, 0.0) + max(coherence_gain * 2.0, 0.0)

    delta = float(base + contradiction_gain + hypothesis_gain + source_gain + bonus)

    # Maintenance credit (avoid delta R = 0 with real work)
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
    swarm_size: Optional[int] = None,
    swarm_layer: Optional[int] = None,
    **kwargs: Any,
) -> float:
    """
    Compute the energy cost of a cycle with swarm normalization.

    Energy represents the effort expended by the agent.  This revised
    implementation reduces the base cost of actions and lowers the
    penalties for external calls, encouraging richer interactions without
    causing the energy term to overwhelm deltaâR.  Swarm penalties are
    moderated to acknowledge efficiency gains from parallelism.

    Parameters
    ----------
    actions_taken : list of dict
        Sequence of tool actions executed during the cycle.
    web_calls : int, optional
        Number of web search API calls made.
    pubmed_calls : int, optional
        Number of PubMed API calls made.
    semantic_calls : int, optional
        Number of Semantic Scholar API calls made.
    pdf_ingestions : int, optional
        Number of PDF ingestion operations performed.
    tokens_estimate : int, optional
        Estimated number of LLM tokens consumed.
    swarm_size : int, optional
        Number of agents concurrently active in the swarm.
    swarm_layer : int, optional
        Depth of the swarm hierarchy (for multiâlayer swarms).

    Returns
    -------
    float
        The computed energy cost.
    """
    # Base cost from actions.  Lowering the base cost to 0.05 per action
    # rewards lowâoverhead interactions and mitigates the denominator in
    # early cycles.  A minimal nonâzero default avoids divideâbyâzero.
    base_cost = 0.05 * float(len(actions_taken)) if actions_taken else 0.05

    # Lower energy multipliers for external calls.  Web, PubMed and
    # Semantic Scholar searches are valuable but now cost only 0.2 each.
    # PDF ingestion is reduced to 0.4 to reflect improved efficiencies.
    cost_web = max(web_calls, 0) * 0.2
    cost_pubmed = max(pubmed_calls, 0) * 0.2
    cost_sem = max(semantic_calls, 0) * 0.2
    cost_pdf = max(pdf_ingestions, 0) * 0.4

    total = base_cost + cost_web + cost_pubmed + cost_sem + cost_pdf

    # Token costing: a larger denominator makes long prompts cheaper,
    # encouraging richer context without ballooning energy.  Keep the
    # divisor at 4000 as before.
    if tokens_estimate is not None and tokens_estimate > 0:
        total += float(tokens_estimate) / 4000.0

    # Swarm penalty discourages unbounded agent spawning but recognises
    # parallel efficiency.  Start with a lower base factor of 0.6 and use
    # smaller increments per additional agent and layer.  This reflects
    # better coordination within swarms.
    swarm_size_eff = swarm_size if isinstance(swarm_size, int) and swarm_size > 0 else 1
    swarm_layer_eff = swarm_layer if isinstance(swarm_layer, int) and swarm_layer > 0 else 1
    swarm_penalty = 0.6 + ((swarm_size_eff - 1) * 0.005) + ((swarm_layer_eff - 1) * 0.010)
    total *= swarm_penalty

    # Safety clamps: ensure energy is positive and not trivially small.
    if total <= 0:
        total = 1.0
    if total < 0.05:
        total = 0.05

    return float(total)


# ---------------------------------------------------------------------------
# Core RYE
# ---------------------------------------------------------------------------

def compute_rye(delta_r: float, energy_e: float) -> float:
    """RYE = Delta R / E, with a defensive zero floor for bad E."""
    dr = _safe_float(delta_r) or 0.0
    e = _safe_float(energy_e)
    if e is None or e <= 0:
        return 0.0
    return float(dr) / float(e)

# ---------------------------------------------------------------------------
# Qualityâweighted RYE (optional)
# ---------------------------------------------------------------------------

def compute_effective_rye(delta_r: float, energy_e: float, quality_score: float = 1.0) -> float:
    """
    Compute a qualityâweighted RYE.

    Parameters
    ----------
    delta_r : float
        The delta R value for the cycle.
    energy_e : float
        The energy cost for the cycle.
    quality_score : float, optional
        A quality factor between 0.0 and 1.0 representing evidence strength. A value
        of 1.0 leaves the RYE unchanged, whereas 0.0 collapses RYE to zero. Values
        outside the [0, 1] range are clamped.

    Returns
    -------
    float
        The qualityâadjusted RYE value.

    Notes
    -----
    The default `compute_rye` remains unchanged to preserve existing behaviour.
    This helper multiplies the raw RYE by a userâprovided quality factor. It is
    safe to call even when no quality_score is known; the default of 1.0 yields
    the unmodified RYE. Callers are expected to derive an appropriate quality
    factor (e.g. based on citation count, support ratio, or other verification
    diagnostics) and pass it in.
    """
    # Clamp quality_score into [0.0, 1.0]
    try:
        q = float(quality_score)
    except Exception:
        q = 1.0
    if q < 0.0:
        q = 0.0
    if q > 1.0:
        q = 1.0
    raw_rye = compute_rye(delta_r, energy_e)
    return raw_rye * q


# ---------------------------------------------------------------------------
# Rolling RYE (mean)
# ---------------------------------------------------------------------------

def rolling_rye(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """
    Compute a simple rolling average of RYE over the last N cycles.

    Compatible with either:
        - full history entries (dicts with "RYE")
        - bare RYE series (list[float])
    """
    if not history or window <= 0:
        return None

    vals = _extract_rye_series(history)
    if not vals:
        return None

    recent = vals[-window:]
    if not recent:
        return None

    return sum(recent) / len(recent)


# ---------------------------------------------------------------------------
# Robust rolling RYE (median style)
# ---------------------------------------------------------------------------

def robust_rolling_rye(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """
    Median based rolling RYE.

    This is more robust to occasional spikes or outliers, and useful
    for noisy long runs with irregular repair bursts.

    Accepts history style or bare RYE lists.
    """
    if not history or window <= 0:
        return None

    vals = _extract_rye_series(history)
    if not vals:
        return None

    recent = vals[-window:]
    if len(recent) == 0:
        return None

    return statistics.median(recent)


# ---------------------------------------------------------------------------
# Median RYE - Noise Resistant
# ---------------------------------------------------------------------------

def median_rye(history: List[Dict[str, Any]]) -> Optional[float]:
    """Median RYE across all cycles, robust to outliers."""
    vals = _extract_rye_series(history)
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

    Accepts history style or bare RYE lists.
    """
    vals = _extract_rye_series(history)
    n = len(vals)
    if n < 4:
        return None

    mid = n // 2
    old = vals[:mid]
    recent = vals[mid:]

    if not old or not recent:
        return None

    avg_old = sum(old) / len(old)
    avg_recent = sum(recent) / len(recent)

    return avg_recent - avg_old


# ---------------------------------------------------------------------------
# Regression Based Trend - Best for Long Runs
# ---------------------------------------------------------------------------

def regression_rye_slope(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Linear regression slope of RYE over cycles.

    Accepts either a full history or a bare list of RYE floats.

    Returns:
        slope (float) or None if not enough data.
    """
    vals = _extract_rye_series(history)
    # Skip leading zeros or near zero values which often correspond to idle or
    # preârepair cycles. These can artificially flatten the regression slope.
    filtered_vals: List[float] = []
    found_non_zero = False
    for v in vals:
        if not found_non_zero and abs(v) < 1e-12:
            continue
        found_non_zero = True
        filtered_vals.append(v)
    if not filtered_vals:
        filtered_vals = vals
    n = len(filtered_vals)
    if n < 2:
        return None

    xs: List[float] = [float(i) for i in range(n)]
    mean_x = sum(xs) / n
    mean_y = sum(filtered_vals) / n

    num = sum((xs[i] - mean_x) * (filtered_vals[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))

    if den == 0:
        return None

    return num / den


# ---------------------------------------------------------------------------
# Rolling RYE slope (linear regression on recent window)
# ---------------------------------------------------------------------------

def rolling_rye_slope(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """
    Compute the slope of the RYE series over the most recent ``window`` cycles.

    This helper fits a simple linear regression line y = a + b*x to the
    last N RYE values (where x is the zero-based index) and returns the
    slope coefficient ``b``.  If insufficient data exists (fewer than two
    numeric RYE values) or ``window`` <= 1, it returns ``None``.

    The slope provides a local trend estimate that is less sensitive to
    the entire run history than the global regression slope.  Positive
    values indicate increasing efficiency, negative values indicate
    declining efficiency, and zero indicates flat performance.

    Parameters
    ----------
    history:
        List of cycle dictionaries or raw RYE values.  Mixed types are
        accepted; non-numeric entries are ignored.
    window:
        Number of recent cycles to consider.  Values <= 1 disable the
        computation and return ``None``.

    Returns
    -------
    float or None
        The slope of the local regression line, or ``None`` if it cannot
        be computed.
    """
    # Defensive checks
    if not history or window is None or window <= 1:
        return None
    vals = _extract_rye_series(history)
    if not vals:
        return None
    recent = vals[-window:]
    n = len(recent)
    if n < 2:
        return None
    # Compute mean of x indices (0..n-1) and y values
    mean_x = (n - 1) / 2.0
    mean_y = sum(recent) / float(n)
    num = 0.0
    den = 0.0
    for i, y in enumerate(recent):
        dx = i - mean_x
        dy = y - mean_y
        num += dx * dy
        den += dx * dx
    if den == 0.0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Stability Index (NEW)
# ---------------------------------------------------------------------------

def stability_index(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Measures variance normalized stability of RYE values.

    A value of 1.0 indicates perfectly stable improvements and 0.0
    corresponds to chaotic swings.  Originally this metric required at
    least four RYE observations to compute a meaningful variance; runs
    shorter than that returned ``None``.  To support short finite
    experiments (e.g. two or three cycles) we relax this requirement and
    compute a stability score whenever at least one RYE value exists.
    When only a single value is available the variance is zero and the
    stability is interpreted as fully stable (1.0 if the value is
    nonÃ¢ÂÂnegative, otherwise 0.0).  For two or three values the sample
    variance provides a coarse estimate of volatility.

    Args:
        history: List of cycle history dicts or raw RYE values.

    Returns:
        A float in [0, 1] indicating stability, or ``None`` when no RYE
        data is available.
    """
    vals = _extract_rye_series(history)
    if not vals:
        return None
    # Require at least three data points for a meaningful stability estimate.
    # Shorter runs return None to avoid misleading high or low scores.
    if len(vals) < 3:
        return None
    mean = sum(vals) / len(vals)
    var = statistics.pvariance(vals)
    # If there is no variance, the sequence is constant; treat as fully stable.
    if var == 0:
        return 1.0
    # If the mean is non-positive, the run is not improving; treat as unstable.
    if mean <= 0:
        return 0.0
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

    Works with either full history or bare RYE lists.

    The original implementation returned ``None`` when there were not
    enough data points to compute a regression slope.  To allow early
    diagnostics on short runs, treat a missing slope as zero momentum.
    """
    slope = regression_rye_slope(history)
    if slope is None:
        # With insufficient data the acceleration is unknown; assume no
        # momentum rather than omitting the metric entirely.
        return 0.0
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

    Compatible with both history dicts and bare RYE lists.
    """
    vals = sorted(_extract_rye_series(history))
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
# Per cycle helper: full RYE summary with domain weighting and learning factor
# ---------------------------------------------------------------------------

def build_cycle_rye_summary(
    *,
    delta_r: float,
    energy_e: float,
    domain: Optional[str] = None,
    role: Optional[str] = None,
    cycle_index: Optional[int] = None,
    extra_signals: Optional[Dict[str, Any]] = None,
    # Learning aware fields (for 10x modes)
    learning_speed_factor: Optional[float] = None,
    learning_profile_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a structured per cycle RYE summary used by TGRM and CoreAgent.

    This helper is the canonical place to:
    - compute raw RYE
    - apply domain weighting
    - attach basic metadata (domain, role, cycle_index)
    - attach learning aware RYE (10x modes) when a learning_speed_factor is provided
    - carry forward any extra signals (novelty, coherence, biomarkers, etc)

    Learning logic:
        rye_raw = Delta R / E
        rye_learning_adjusted = rye_raw * learning_speed_factor

    Where learning_speed_factor > 1.0 represents faster learning
    (more repair yield per unit energy).
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

    # Learning aware RYE enrichment (10x ready)
    if learning_speed_factor is not None:
        try:
            factor = float(learning_speed_factor)
        except Exception:
            factor = 1.0

        factor = max(0.1, min(factor, 10.0))

        summary["learning_speed_factor"] = factor
        if learning_profile_hint is not None:
            summary["learning_profile_hint"] = str(learning_profile_hint)

        rye_learning_adjusted = rye_raw * factor
        summary["RYE_learning_adjusted"] = rye_learning_adjusted
        summary["RYE_learning_adjusted_weighted"] = apply_domain_weight(
            rye_learning_adjusted, domain
        )

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
          "rye_avg_weighted": Optional[float],
          "learning_adjusted": { ... }  # only present if RYE_learning_adjusted exists
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

    # Learning adjusted diagnostics (10x aware) if the history contains RYE_learning_adjusted
    la_hist: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        v = _safe_float(entry.get("RYE_learning_adjusted"))
        if v is None:
            continue
        # Keep chronological order but use canonical RYE key internally
        la_hist.append({"RYE": v})

    if la_hist:
        la_norm = normalize_history(la_hist)
        la_count = len(la_norm)

        la_rolling = rolling_rye(la_norm, window=window)
        la_robust_rolling = robust_rolling_rye(la_norm, window=window)
        la_med = median_rye(la_norm)
        la_trend = efficiency_trend(la_norm)
        la_slope = regression_rye_slope(la_norm)
        la_stab = stability_index(la_norm)
        la_rec = recovery_momentum(la_norm)
        la_low, la_mid, la_high = rye_percentiles(la_norm)

        la_vals = [float(e["RYE"]) for e in la_norm]
        la_avg = sum(la_vals) / len(la_vals) if la_vals else None
        la_last = la_vals[-1] if la_vals else None

        learning_diag: Dict[str, Any] = {
            "count": la_count,
            "rye_avg": la_avg,
            "rye_median": la_med,
            "rye_last": la_last,
            "rolling_rye": la_rolling,
            "robust_rolling_rye": la_robust_rolling,
            "trend_simple": la_trend,
            "trend_slope": la_slope,
            "stability_index": la_stab,
            "recovery_momentum": la_rec,
            "low_percentile": la_low,
            "mid_percentile": la_mid,
            "high_percentile": la_high,
        }

        diagnostics["learning_adjusted"] = learning_diag

    return diagnostics


# Backwards-compatible alias for older code / MSIL imports
def run_diagnostics(
    history: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    window: int = 10,
) -> Dict[str, Any]:
    """
    Backwards compatible alias for build_run_diagnostics.

    Some callers expect `run_diagnostics(...)` to exist either in a separate
    module or here. This keeps MSIL and UI code simple.
    """
    return build_run_diagnostics(history, domain=domain, window=window)


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
          "learning_speed_factor": Optional[float],  # optional 10x style hint
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
        - rye_median_learning: median per event ratio * learning_speed_factor (if available)
        - rye_last_learning: last per event ratio * learning_speed_factor (if available)
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
            "ratios": [],
            "ratios_learning": [],
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
        lsf = _safe_float(ev.get("learning_speed_factor"))

        if delta_r is not None:
            t["sum_delta_r"] = float(t.get("sum_delta_r", 0.0)) + delta_r
        if energy_e is not None and energy_e > 0:
            t["sum_energy"] = float(t.get("sum_energy", 0.0)) + energy_e

        if delta_r is not None and energy_e is not None and energy_e > 0:
            ratio = delta_r / energy_e
            t["ratios"].append(ratio)

            if lsf is not None:
                factor = max(0.1, min(float(lsf), 10.0))
                t["ratios_learning"].append(ratio * factor)

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

        ratios_learning_raw = t.get("ratios_learning", [])
        ratios_learning: List[float] = [
            float(r) for r in ratios_learning_raw if isinstance(r, (int, float))
        ]

        sum_energy = float(t.get("sum_energy", 0.0))
        sum_delta_r = float(t.get("sum_delta_r", 0.0))

        rye_avg: Optional[float] = None
        rye_median_val: Optional[float] = None
        rye_last: Optional[float] = None

        rye_median_learning: Optional[float] = None
        rye_last_learning: Optional[float] = None

        # NOTE: compute avg even if sum_delta_r == 0 (so avg can be 0.0)
        if sum_energy > 0:
            rye_avg = sum_delta_r / sum_energy

        if ratios:
            rye_median_val = statistics.median(ratios)
            rye_last = ratios[-1]

        if ratios_learning:
            rye_median_learning = statistics.median(ratios_learning)
            rye_last_learning = ratios_learning[-1]

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
            "rye_median_learning": rye_median_learning,
            "rye_last_learning": rye_last_learning,
            "last_timestamp": last_timestamp,
        }

    return {"tools": tools_out}


# ======================================================================
#                         OPTION C EXTRAS
# ======================================================================

# RYE Volatility Signature (local self-diagnosis)

def rye_volatility_signature(
    history_or_diagnostics: Any,
    *,
    window: int = 20,
    domain: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Optional[float]]:
    """
    Local volatility signature of RYE over a recent window.

    Accepts either:
        - full history (list of dicts or numeric RYE values)
        - diagnostics dict produced by build_run_diagnostics

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
    # Prefer explicit history kwarg if provided
    if "history" in kwargs and kwargs["history"] is not None:
        data = kwargs["history"]
    else:
        data = history_or_diagnostics

    # Diagnostics mode: approximate volatility from percentiles and averages
    if isinstance(data, dict):
        diag = data
        low = _safe_float(diag.get("low_percentile"))
        high = _safe_float(diag.get("high_percentile"))
        mid = _safe_float(diag.get("mid_percentile"))
        avg = _safe_float(diag.get("rye_avg"))

        if low is None or high is None:
            return {
                "std": None,
                "range": None,
                "cv": None,
                "volatility_score": None,
            }

        r = high - low
        # Simple heuristic: treat range as about four standard deviations
        std = r / 4.0 if r is not None else None

        center = avg if avg is not None else mid
        if center is None or center == 0 or std is None:
            cv = None
        else:
            cv = abs(std / center)

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

    # History mode: original series based implementation
    vals = _extract_rye_series(data if isinstance(data, list) else [])
    if len(vals) == 0 or window <= 1:
        return {
            "std": None,
            "range": None,
            "cv": None,
            "volatility_score": None,
        }

    recent = vals[-window:]
    # When fewer than two observations are available treat the series as
    # perfectly stable (zero variance).  This produces std=0, range=0,
    # cv=None and a maximum volatility score of 1.0, avoiding ``None``
    # signals that propagate as n/a in the UI.
    if len(recent) < 2:
        return {
            "std": 0.0,
            "range": 0.0,
            "cv": None,
            "volatility_score": 1.0,
        }

    mean = sum(recent) / len(recent)
    std = statistics.pstdev(recent)
    vmin = min(recent)
    vmax = max(recent)
    r = vmax - vmin

    if mean == 0:
        cv = None
    else:
        cv = abs(std / mean)

    if cv is None:
        # When the coefficient of variation is undefined (mean==0), treat
        # volatility as maximum (1.0) to indicate neutrality.  Setting
        # cv=None here would propagate None and result in n/a signals.
        volatility_score = 1.0
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

# Minimum number of data points required to evaluate equilibrium.
# If fewer than this number of RYE values are available in the history or
# diagnostics bundle, the equilibrium detector will always return
# ``in_equilibrium = False`` with ``reason = 'insufficient_data'``.
# In the original implementation at least ten RYE observations were
# required before equilibrium detection could be performed.  For short
# exploratory runs this prevented the equilibrium detector from ever
# engaging, resulting in ``n/a`` and ``insufficient_data`` flags.  The
# longevityÃ¢ÂÂonly build reduces this requirement to three cycles so that
# the equilibrium heuristics can produce a meaningful result on short
# runs.  Users should treat these early equilibrium indicators as
# provisional.
# Similarly lower the number of RYE observations required before equilibrium
# detection engages.  By setting this to 1 the equilibrium detector will no
# longer immediately return 'insufficient_data' on short runs.  It will
# evaluate the available data and report nonÃ¢ÂÂequilibrium reasons rather
# than withholding diagnostics entirely.
MIN_EQUILIBRIUM_CYCLES: int = 1

def detect_rye_equilibrium(
    history_or_diagnostics: Any,
    *,
    window: int = 30,
    slope_tolerance: float = 0.01,
    **kwargs: Any,
) -> Dict[str, Optional[Any]]:
    """
    Detect whether the system appears to be in a RYE equilibrium zone.

    This function tries to distinguish between a genuine plateau in repair
    efficiency and a degenerate "flatline" where nothing is happening.  In
    addition to the original conditions (regression slope within
    ``slope_tolerance`` and nonÃ¢ÂÂexcessive volatility), it enforces a few
    additional gating criteria:

      * **Minimum data requirement** Ã¢ÂÂ at least ``MIN_EQUILIBRIUM_CYCLES``
        recent RYE observations are needed before equilibrium can be
        considered.  With fewer points the detector returns
        ``in_equilibrium = False`` and ``reason = 'insufficient_data'``.

      * **NonÃ¢ÂÂdegenerate volatility** Ã¢ÂÂ the recent window must exhibit some
        range (i.e., max(RYE) Ã¢ÂÂ min(RYE) > 0) and a finite volatility
        score.  A completely flat sequence (volatility range Ã¢ÂÂ 0) is
        treated as stasis rather than equilibrium.  When the range is zero
        or the volatility signature cannot be computed, the detector
        returns ``in_equilibrium = False`` and ``reason = 'no_volatility'``.

      * **Evidence of progress** Ã¢ÂÂ the run must show some improvement in
        efficiency.  This is estimated via ``trend_simple``, the difference
        between average RYE in the second half of the history and the first
        half.  If ``trend_simple`` is not available or is nonÃ¢ÂÂpositive,
        equilibrium is not declared and the reason will be
        ``'no_progress'``.  Callers that wish to override this gate can
        precompute a custom progress metric and pass it via the
        ``history_or_diagnostics`` parameter.

    Accepts either:
        * full history (list)
        * diagnostics dict produced by ``build_run_diagnostics``

    Returns a dict with keys:
        ``in_equilibrium`` (bool),
        ``reason`` (str),
        ``window_size`` (int or None),
        ``local_slope`` (float or None),
        ``local_volatility_score`` (float or None),
        and, when in diagnostics mode, ``progress`` (float or None).
    """
    # Prefer explicit history kwarg if provided
    if "history" in kwargs and kwargs["history"] is not None:
        data = kwargs["history"]
    else:
        data = history_or_diagnostics

    # Diagnostics mode: use trend_slope and volatility_signature
    if isinstance(data, dict):
        diag = data
        slope = _safe_float(diag.get("trend_slope"))
        vol_sig = rye_volatility_signature(diag)
        vol_score = vol_sig.get("volatility_score")

        count = diag.get("count")
        window_size = int(count) if isinstance(count, int) else None

        # Always require a minimum number of samples
        if window_size is None or window_size < MIN_EQUILIBRIUM_CYCLES:
            return {
                "in_equilibrium": False,
                "reason": "insufficient_data",
                "window_size": window_size,
                "local_slope": None,
                "local_volatility_score": None,
                "progress": None,
            }

        # Extract additional signals
        progress = None
        try:
            # trend_simple is positive when recent performance exceeds early performance
            progress = _safe_float(diag.get("trend_simple"))
        except Exception:
            progress = None

        vol_sig = rye_volatility_signature(diag)
        vol_score = vol_sig.get("volatility_score")
        vol_range = vol_sig.get("range")

        # Gating: require slope to be defined
        if slope is None:
            return {
                "in_equilibrium": False,
                "reason": "no_slope",
                "window_size": window_size,
                "local_slope": None,
                "local_volatility_score": vol_score,
                "progress": progress,
            }

        # Gating: require volatility signature to exist and have nonÃ¢ÂÂzero range
        if vol_score is None or vol_range is None or vol_range <= 0:
            return {
                "in_equilibrium": False,
                "reason": "no_volatility",
                "window_size": window_size,
                "local_slope": slope,
                "local_volatility_score": vol_score,
                "progress": progress,
            }

        # Gating: require progress (trend_simple > 0) to avoid flatline stasis
        if progress is None or progress <= 0:
            return {
                "in_equilibrium": False,
                "reason": "no_progress",
                "window_size": window_size,
                "local_slope": slope,
                "local_volatility_score": vol_score,
                "progress": progress,
            }

        # Core equilibrium heuristic: slope within tolerance and volatility high enough
        if abs(slope) <= slope_tolerance and vol_score >= 0.4:
            return {
                "in_equilibrium": True,
                "reason": "equilibrium",
                "window_size": window_size,
                "local_slope": slope,
                "local_volatility_score": vol_score,
                "progress": progress,
            }
        else:
            return {
                "in_equilibrium": False,
                "reason": "non_equilibrium",
                "window_size": window_size,
                "local_slope": slope,
                "local_volatility_score": vol_score,
                "progress": progress,
            }

    # History mode: original series based implementation
    vals = _extract_rye_series(data if isinstance(data, list) else [])
    n = len(vals)
    # Require at least the minimal number of samples for equilibrium detection.  Unlike
    # the original implementation, do not insist on a large fixed window; use
    # whatever data is available beyond the minimum.  This allows the detector
    # to operate on short runs instead of immediately returning insufficient_data.
    if n < MIN_EQUILIBRIUM_CYCLES:
        return {
            "in_equilibrium": False,
            "reason": "insufficient_data",
            "window_size": n,
            "local_slope": None,
            "local_volatility_score": None,
        }

    # Only evaluate the most recent portion of the series.  Use the smaller
    # of the requested window or the available data so that short runs still
    # produce a meaningful equilibrium signal.  Always use at least
    # MIN_EQUILIBRIUM_CYCLES observations.
    recent_window = max(MIN_EQUILIBRIUM_CYCLES, min(window, n))
    recent_vals = vals[-recent_window:]
    slope = regression_rye_slope(recent_vals)
    vol_sig = rye_volatility_signature(recent_vals, window=recent_window)
    vol_score = vol_sig.get("volatility_score")
    vol_range = vol_sig.get("range")

    # Compute simple progress over the recent window: compare first and last halves
    progress = efficiency_trend(recent_vals) if isinstance(recent_vals, list) else None

    if slope is None:
        return {
            "in_equilibrium": False,
            "reason": "no_slope",
            "window_size": len(recent_vals),
            "local_slope": None,
            "local_volatility_score": vol_score,
        }

    # Gating: require volatility signature and nonÃ¢ÂÂzero range
    if vol_score is None or vol_range is None or vol_range <= 0:
        return {
            "in_equilibrium": False,
            "reason": "no_volatility",
            "window_size": len(recent_vals),
            "local_slope": slope,
            "local_volatility_score": vol_score,
        }

    # Gating: require progress
    if progress is None or progress <= 0:
        return {
            "in_equilibrium": False,
            "reason": "no_progress",
            "window_size": len(recent_vals),
            "local_slope": slope,
            "local_volatility_score": vol_score,
        }

    # Final equilibrium heuristic
    if abs(slope) <= slope_tolerance and vol_score >= 0.4:
        return {
            "in_equilibrium": True,
            "reason": "equilibrium",
            "window_size": len(recent_vals),
            "local_slope": slope,
            "local_volatility_score": vol_score,
        }
    else:
        return {
            "in_equilibrium": False,
            "reason": "non_equilibrium",
            "window_size": len(recent_vals),
            "local_slope": slope,
            "local_volatility_score": vol_score,
        }


# TGRM Harmonic Index (proxy)

def tgrm_harmonic_index(
    history_or_diagnostics: Any,
    *,
    window: int = 10,
    domain: Optional[str] = None,
    **kwargs: Any,
) -> Optional[float]:
    """
    Proxy for TGRM harmonic resonance.

    Accepts either:
        - full history (list)
        - diagnostics dict produced by build_run_diagnostics

    Approximated via:
        - stability_index
        - recovery_momentum

    Returns:
        float in [0, 1] or None
    """
    # Prefer explicit history kwarg if provided
    if "history" in kwargs and kwargs["history"] is not None:
        data = kwargs["history"]
    else:
        data = history_or_diagnostics

    # Diagnostics mode: read stability_index and recovery_momentum directly
    if isinstance(data, dict):
        diag = data
        stab = _safe_float(diag.get("stability_index"))
        rec = _safe_float(diag.get("recovery_momentum"))

        if stab is None and rec is None:
            return None

        stab_norm = stab if stab is not None else 0.0
        rec_raw = rec if rec is not None else 0.0
        rec_norm = min(max(rec_raw / 5.0, 0.0), 1.0)

        idx = 0.6 * stab_norm + 0.4 * rec_norm
        return max(0.0, min(idx, 1.0))

    # History mode: use history based stability and momentum
    hist = data if isinstance(data, list) else []
    stab = stability_index(hist)
    rec = recovery_momentum(hist)

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
    Estimate a heuristic probability-like score that this run configuration
    could yield a meaningful breakthrough within a given horizon.

    Important:
        - This is NOT a calibrated real-world probability.
        - Treat `probability` as a confidence-style gauge derived from
          RYE dynamics (trend, momentum, stability, tails), not as
          "52.1% chance of Nobel-tier discovery".
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


# Autonomy Stability Safety Envelope

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
    # Basic metrics extracted from diagnostics
    stab = diagnostics.get("stability_index")
    slope = diagnostics.get("trend_slope")
    low_p = diagnostics.get("low_percentile")
    high_p = diagnostics.get("high_percentile")
    count = diagnostics.get("count") or 0

    # If any essential metrics are missing, report unknown state
    if stab is None or slope is None or low_p is None or high_p is None:
        return {
            "state": "unknown",
            "details": {
                "reason": "insufficient_metrics",
            },
        }

    # For very short histories we cannot confidently classify the safety envelope.
    # Require at least 5 cycles of data before making a directional judgment.
    if count < 5:
        spread = high_p - low_p if (high_p is not None and low_p is not None) else None
        return {
            "state": "neutral",
            "details": {
                "stability_index": stab,
                "trend_slope": slope,
                "low_percentile": low_p,
                "high_percentile": high_p,
                "spread": spread,
                "reason": "insufficient_history",
            },
        }

    spread = high_p - low_p

    # Classify envelope based on stability, trend, and percentile spread
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

        Tier 0: unstable or chaotic
        Tier 1: normal working agent (positive average RYE)
        Tier 2: self repairing, long run stable
        Tier 3: "Major Breakthrough Zone" candidate

    Note:
        This uses the heuristic breakthrough probability, not a calibrated
        real-world chance. Treat it as "zone tagging" rather than a
        literal probability label.
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

    if count < MIN_CYCLES_FOR_TIERING:
        tier = "Tier 0"
        reason = "too_few_cycles"
    elif stab < 0.2 and avg <= 0.0:
        tier = "Tier 0"
        reason = "unstable_negative_or_zero_rye"
    elif avg > 0.0 and stab >= 0.2:
        tier = "Tier 1"
        reason = "positive_rye_basic_stability"

    if (
        count >= MIN_CYCLES_FOR_TIERING
        and avg > TIER2_MIN_AVG_RYE
        and stab >= TIER2_MIN_STABILITY
        and trend >= 0.0
        and mid_p > 0.0
    ):
        tier = "Tier 2"
        reason = "self_repairing_long_run_candidate"

    if is_tier3_run_candidate(diagnostics, breakthrough_prob=bp):
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


def is_tier3_run_candidate(
    diagnostics: Dict[str, Any],
    *,
    breakthrough_prob: Optional[float] = None,
) -> bool:
    """
    Convenience helper for Tier 3 hunts.

    Returns True when the run appears to be in a Tier 3 zone according to
    current heuristic thresholds.
    """
    count = diagnostics.get("count") or 0
    if count < MIN_CYCLES_FOR_TIERING:
        return False

    stab = diagnostics.get("stability_index") or 0.0
    avg = diagnostics.get("rye_avg") or 0.0
    trend = diagnostics.get("trend_slope") or 0.0
    mid_p = diagnostics.get("mid_percentile") or 0.0

    if breakthrough_prob is None:
        bp = estimate_breakthrough_probability(diagnostics).get("probability") or 0.0
    else:
        bp = breakthrough_prob

    if (
        avg > TIER3_MIN_AVG_RYE
        and stab >= TIER3_MIN_STABILITY
        and trend > TIER3_MIN_TREND
        and mid_p > TIER3_MIN_MID_PERCENTILE
        and bp >= TIER3_MIN_BREAKTHROUGH_PROB
    ):
        return True

    return False


# Critical Failure Early Warning Score

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
    count = diagnostics.get("count") or 0

    # If any required metrics are missing, return an unknown score.
    if stab is None or trend is None or avg is None or low_p is None or last is None:
        return {
            "score": None,
            "factors": {
                "reason": "insufficient_metrics",
            },
        }

    # For very short runs (< 5 cycles), treat failure risk as neutral.
    if count < 5:
        return {
            "score": 0.0,
            "factors": {
                "inv_stability": None,
                "downward": None,
                "avg_risk": None,
                "last_risk": None,
                "reason": "insufficient_history",
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


# 90 Day Breakthrough Likelihood Score

def breakthrough_likelihood_90d(
    diagnostics: Dict[str, Any],
    *,
    domain: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: estimate breakthrough likelihood specifically for
    a 90 day horizon, optionally conditioned on how long the system has
    already been running.

    The output is still a heuristic score, not a calibrated probability.
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


# Per cycle tier classifier

def classify_cycle_tier(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify a single cycle into a tier based on its RYE and extra signals.

    Expected fields in summary (from build_cycle_rye_summary and extra_signals):
        - RYE
        - delta_r
        - domain (optional)
        - novelty_score (optional, from extra_signals)
        - coherence_gain (optional, from extra_signals)

    Tiers:
        Tier 0: non productive or negative cycle
        Tier 1: basic positive repair
        Tier 2: strong repair with good yield
        Tier 3: candidate for high value discovery
    """
    rye = _safe_float(summary.get("RYE")) or 0.0
    delta_r = _safe_float(summary.get("delta_r")) or 0.0
    novelty = _safe_float(summary.get("novelty_score")) or 0.0
    coherence = _safe_float(summary.get("coherence_gain")) or 0.0

    tier = "Tier 0"
    reason = "non_productive_or_negative_cycle"

    if rye <= CYCLE_TIER1_MIN_RYE or delta_r <= 0.0:
        tier = "Tier 0"
        reason = "non_positive_rye_or_delta_r"
    elif rye > CYCLE_TIER1_MIN_RYE and delta_r > 0.0:
        tier = "Tier 1"
        reason = "basic_positive_repair"

    if rye > CYCLE_TIER2_MIN_RYE and delta_r >= CYCLE_TIER2_MIN_DELTA_R:
        tier = "Tier 2"
        reason = "strong_repair_good_yield"

    if (
        rye > CYCLE_TIER3_MIN_RYE
        and delta_r >= CYCLE_TIER3_MIN_DELTA_R
        and novelty >= CYCLE_TIER3_MIN_NOVELTY
        and coherence >= CYCLE_TIER3_MIN_COHERENCE
    ):
        tier = "Tier 3"
        reason = "high_value_discovery_candidate"

    return {
        "tier": tier,
        "reason": reason,
        "metrics": {
            "RYE": rye,
            "delta_r": delta_r,
            "novelty_score": novelty,
            "coherence_gain": coherence,
        },
    }


def is_tier3_cycle_candidate(summary: Dict[str, Any]) -> bool:
    """
    Convenience helper for Tier 3 hunts at cycle level.

    Returns True when a single cycle looks like a Tier 3 candidate
    according to current per cycle thresholds.
    """
    info = classify_cycle_tier(summary)
    return info.get("tier") == "Tier 3"


# Option C master bundle for a run

def build_option_c_signature(
    history: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
    window: int = 10,
) -> Dict[str, Any]:
    """
    High level Option C self diagnosis snapshot.

    Combines:
        - run diagnostics (including learning adjusted sub block if present)
        - volatility signature
        - equilibrium detection
        - TGRM harmonic index
        - breakthrough probability (heuristic)
        - 90 day breakthrough likelihood (heuristic)
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


# Backwards-compatible alias for Option C
def option_c_signature(
    history: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
    window: int = 10,
) -> Dict[str, Any]:
    """
    Backwards compatible alias for build_option_c_signature.

    Some offline analytics or MSIL probes may import `option_c_signature`
    directly.
    """
    return build_option_c_signature(
        history,
        domain=domain,
        hours_run_so_far=hours_run_so_far,
        window=window,
    )


# ======================================================================
#                    OPTIONAL UTILITIES FOR MSIL AND OFFLINE ANALYTICS
# ======================================================================

def compact_history(
    history: List[Dict[str, Any]],
    *,
    max_points: int = 500,
) -> List[Dict[str, Any]]:
    """
    Optional helper: downsample a long history to at most max_points entries.

    Keeps the first and last entries and samples the middle region
    approximately uniformly. This is useful for:
        - exporting long runs
        - building light msil snapshots
        - plotting without huge payloads
    """
    if not isinstance(history, list) or len(history) <= max_points:
        return list(history or [])

    n = len(history)
    if max_points <= 2:
        return [history[0], history[-1]]

    # Always keep first and last
    result: List[Dict[str, Any]] = [history[0]]
    inner = history[1:-1]
    inner_points = max_points - 2
    step = max(len(inner) / float(inner_points), 1.0)

    idx = 0.0
    while int(idx) < len(inner) and len(result) < max_points - 1:
        result.append(inner[int(idx)])
        idx += step

    result.append(history[-1])
    return result


def history_window(
    history: List[Dict[str, Any]],
    *,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    last_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Optional helper: slice a history by indices or take the last N entries.

    This is WYSIWYG and does not touch RYE values. It is designed to be used
    before calling build_option_c_signature or build_run_diagnostics in
    msil style probes.
    """
    if not isinstance(history, list):
        return []

    n = len(history)
    if n == 0:
        return []

    if last_n is not None and last_n > 0:
        return history[max(0, n - last_n) :]

    si = 0 if start_index is None else max(start_index, 0)
    ei = n if end_index is None else min(end_index, n)
    if si >= ei:
        return []
    return history[si:ei]


def domain_slice(
    history: List[Dict[str, Any]],
    *,
    domain: str,
) -> List[Dict[str, Any]]:
    """
    Optional helper: filter history to entries for a single domain.

    This expects history entries with a "domain" key. If the key is missing
    the entry is skipped.
    """
    if not isinstance(history, list) or not domain:
        return []

    target = domain.lower()
    out: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        d = entry.get("domain")
        if isinstance(d, str) and d.lower() == target:
            out.append(entry)
    return out


def summarize_rye_series(
    values_or_history: List[Any],
) -> Dict[str, Optional[float]]:
    """
    Optional helper: quick summary for bare RYE series or full history.

    Returns a compact summary:
        {
          "count": int,
          "avg": float or None,
          "median": float or None,
          "min": float or None,
          "max": float or None,
        }
    """
    vals = _extract_rye_series(values_or_history)
    n = len(vals)
    if n == 0:
        return {
            "count": 0,
            "avg": None,
            "median": None,
            "min": None,
            "max": None,
        }

    avg = sum(vals) / n
    med = statistics.median(vals)
    vmin = min(vals)
    vmax = max(vals)

    return {
        "count": n,
        "avg": avg,
        "median": med,
        "min": vmin,
        "max": vmax,
    }


def build_msil_ready_snapshot(
    history: List[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    label: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
    window: int = 10,
    max_points: int = 500,
) -> Dict[str, Any]:
    """
    Optional high level helper for msil.py and IQ style probes.

    This gives a compact, export friendly snapshot with:
        - a compacted history segment
        - option C signature
        - a minimal front panel summary

    It does not change any core logic, it just composes existing primitives.
    """
    compact = compact_history(history, max_points=max_points)
    option_c = build_option_c_signature(
        compact,
        domain=domain,
        hours_run_so_far=hours_run_so_far,
        window=window,
    )
    diag = option_c["diagnostics"]
    tier_info = option_c["run_tier"]

    summary = {
        "label": label,
        "domain": domain or diag.get("domain"),
        "tier": tier_info.get("tier"),
        "rye_avg": diag.get("rye_avg"),
        "rye_median": diag.get("rye_median"),
        "rye_last": diag.get("rye_last"),
        "stability_index": diag.get("stability_index"),
        "trend_slope": diag.get("trend_slope"),
        "breakthrough_probability": option_c["breakthrough_probability"].get("probability"),
        "breakthrough_likelihood_90d": option_c["breakthrough_likelihood_90d"].get("probability"),
        "early_failure_score": option_c["early_failure_warning"].get("score"),
    }

    return {
        "summary": summary,
        "option_c_signature": option_c,
        "history_compact": compact,
    }
