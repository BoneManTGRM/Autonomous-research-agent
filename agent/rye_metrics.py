"""Metrics for computing RYE (Repair Yield per Energy).

This module defines helper functions to compute:

- ΔR (improvement) for each cycle
- E  (effort / energy cost) for each cycle
- RYE = ΔR / E (Repair Yield per Energy)
- Optional rolling RYE and efficiency trends

Reparodynamics interpretation:
    The research agent is a reparodynamic system trying to raise RYE over
    time. Each TGRM cycle (Test → Detect → Repair → Verify) attempts to
    reduce defects (missing info, contradictions, TODOs) with minimal
    effort. Higher RYE means more efficient self repair.

Backwards compatibility:
    Existing calls that use the simple signatures:

        compute_delta_r(issues_before, issues_after, repairs_applied)
        compute_energy(actions_taken)

    still work exactly as before. The extra parameters are optional and
    only used when provided by newer parts of the agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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
    """Compute the improvement ΔR for a cycle.

    Args:
        issues_before:
            Number of issues detected before repairs (gaps, TODOs,
            unanswered questions, contradictions).
        issues_after:
            Number of issues remaining after repairs.
        repairs_applied:
            Count of repair actions taken (web searches, ingestion,
            notes added, etc.).
        contradictions_resolved:
            Optional count of contradictions explicitly resolved in this
            cycle (e.g., two papers disagreed and this cycle clarified).
        hypotheses_generated:
            Optional count of new, non-trivial hypotheses produced this
            cycle by the hypothesis engine.
        sources_used:
            Optional count of distinct sources (web pages, papers, etc.)
            that contributed to the repair.

    Returns:
        float: The improvement score ΔR.

    Heuristic logic:
        - Base improvement is the number of issues resolved.
        - Additional credit for resolved contradictions and hypotheses.
        - Very small bonus for number of sources used (diverse evidence).
    """
    # Base improvement: number of issues resolved (cannot be negative)
    base = max(issues_before - issues_after, 0)

    # Each resolved contradiction is quite valuable: +0.5
    contradiction_gain = contradictions_resolved * 0.5

    # Each hypothesis is useful but may be noisy: +0.2
    hypothesis_gain = hypotheses_generated * 0.2

    # Diverse sources help, but with very small per-source gain: +0.05
    source_gain = min(max(sources_used, 0), 20) * 0.05  # cap to avoid explosion

    delta = float(base + contradiction_gain + hypothesis_gain + source_gain)

    # If no issues existed at all, we still give a tiny credit so
    # "maintenance cycles" that only refine knowledge are not zero.
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
    """Estimate the effort (E) expended during the cycle.

    Args:
        actions_taken:
            A list of action dictionaries representing the operations
            executed during the cycle. Each action contributes at least
            1 unit of cost.
        web_calls:
            Optional explicit count of general web/Tavily calls.
        pubmed_calls:
            Optional explicit count of PubMed API queries.
        semantic_calls:
            Optional explicit count of Semantic Scholar queries.
        pdf_ingestions:
            Optional explicit count of PDF downloads/parse operations.
        tokens_estimate:
            Optional estimate of language model tokens processed in this
            cycle (if you later integrate an LLM). If provided, each
            1000 tokens adds 1 unit of cost.

    Returns:
        float: A numeric energy cost.

    Heuristic cost model:
        - Each action: +1
        - Each web call: +1.5
        - Each PubMed / Semantic call: +2
        - Each PDF ingestion: +2.5 (heavier operation)
        - Each 1000 tokens: +1
    """
    # Base effort: each logged action costs 1
    base_cost = float(len(actions_taken)) if actions_taken else 1.0

    # Add weighted external calls
    cost_web = max(web_calls, 0) * 1.5
    cost_pubmed = max(pubmed_calls, 0) * 2.0
    cost_sem = max(semantic_calls, 0) * 2.0
    cost_pdf = max(pdf_ingestions, 0) * 2.5

    total = base_cost + cost_web + cost_pubmed + cost_sem + cost_pdf

    # Token-based cost if available
    if tokens_estimate is not None and tokens_estimate > 0:
        total += float(tokens_estimate) / 1000.0

    # Ensure energy is never zero (avoid division by zero)
    if total <= 0:
        total = 1.0

    return float(total)


# ---------------------------------------------------------------------------
# RYE computation
# ---------------------------------------------------------------------------


def compute_rye(delta_r: float, energy_e: float) -> float:
    """Compute the Repair Yield per Energy (RYE) metric.

    Args:
        delta_r:
            Improvement achieved during the cycle (ΔR).
        energy_e:
            Effort spent during the cycle (E).

    Returns:
        float: The RYE value. Defaults to 0 if energy is zero.

    Reparodynamics interpretation:
        RYE is the core efficiency law for this agent: how much verified
        repair or improvement you get per unit of energy. The goal of
        the entire system is to raise RYE over time while maintaining
        stability and correctness.
    """
    if energy_e <= 0:
        return 0.0
    return float(delta_r) / float(energy_e)


# ---------------------------------------------------------------------------
# Helpers for long-running sessions
# ---------------------------------------------------------------------------


def rolling_rye(history: List[Dict[str, Any]], window: int = 10) -> Optional[float]:
    """Compute a rolling average of RYE over the last N cycles.

    Args:
        history:
            List of cycle logs, each ideally containing a "RYE" field.
        window:
            Number of most recent cycles to include.

    Returns:
        float or None:
            Rolling average RYE over the requested window, or None if
            there is not enough data.
    """
    if not history:
        return None

    recent = history[-window:]
    values: List[float] = []
    for entry in recent:
        val = entry.get("RYE")
        if isinstance(val, (int, float)):
            values.append(float(val))

    if not values:
        return None
    return sum(values) / len(values)


def efficiency_trend(history: List[Dict[str, Any]]) -> Optional[float]:
    """Compute a simple trend indicator of RYE over time.

    This is a very lightweight slope-like metric: it compares the
    average RYE of the first half of the history and the second half.
    Positive values suggest improving efficiency; negative values
    suggest deteriorating efficiency.

    Args:
        history:
            List of cycle logs.

    Returns:
        float or None:
            Difference (avg_recent - avg_old) or None if there is not
            enough data.
    """
    n = len(history)
    if n < 4:
        return None  # not enough data to say anything meaningful

    mid = n // 2
    old = history[:mid]
    recent = history[mid:]

    def _avg(h: List[Dict[str, Any]]) -> Optional[float]:
        vals: List[float] = []
        for e in h:
            v = e.get("RYE")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return None
        return sum(vals) / len(vals)

    avg_old = _avg(old)
    avg_recent = _avg(recent)
    if avg_old is None or avg_recent is None:
        return None

    return avg_recent - avg_old
