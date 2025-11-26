"""Gold Notebook for the Autonomous Research Agent (Option C friendly).

This module is a high level analysis and "lab notebook" helper that
focuses on:

- Learning speed (how fast RYE and delta_R improve)
- Reparodynamic stability and equilibrium
- Breakthrough probabilities over different horizons
- Swarm and role performance
- Discovery highlights

It is designed to be safe to import from Streamlit or a notebook.
There are no side effects during import.

Key entry points
----------------
- build_gold_snapshot(...)
    Returns a structured dict summarizing the current run.
- gold_notebook_markdown(...)
    Returns a Markdown notebook style report string.
- export_gold_notebook_json(...)
    Convenience helper to save a snapshot as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_store import MemoryStore

from agent.rye_metrics import (
    build_run_diagnostics,
    rye_volatility_signature,
    detect_rye_equilibrium,
    tgrm_harmonic_index,
    estimate_breakthrough_probability,
    breakthrough_likelihood_90d,
    autonomy_safety_envelope,
    early_failure_warning_score,
    classify_run_tier,
    build_option_c_signature,
)


# --------------------------------------------------------------------
# Internal helpers and data classes
# --------------------------------------------------------------------


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if x is None:
            return default
        return float(str(x))
    except Exception:
        return default


def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _learning_speed_label(slope: Optional[float]) -> str:
    """Translate RYE slope into a human friendly learning speed label."""
    if slope is None:
        return "unknown"

    s = slope
    if s > 0.02:
        return "very fast learning"
    if s > 0.005:
        return "steady learning"
    if s > 0.0:
        return "slow learning"
    if s == 0.0:
        return "flat or stalled"
    if s > -0.005:
        return "slight degradation"
    return "clear degradation"


def _equilibrium_label(eq: Dict[str, Any]) -> str:
    """Compact human view of equilibrium status."""
    if not eq:
        return "no equilibrium data recorded"

    frac = eq.get("equilibrium_fraction")
    stab = eq.get("stability_index")
    if isinstance(frac, (int, float)) and isinstance(stab, (int, float)):
        if frac >= 0.7 and stab >= 0.7:
            return "high equilibrium with strong stability"
        if frac >= 0.4 and stab >= 0.5:
            return "partial equilibrium, still consolidating"
        if stab < 0.3:
            return "highly volatile regime"
        return "mixed regime, unclear equilibrium"
    if isinstance(stab, (int, float)):
        if stab >= 0.7:
            return "stable regime, equilibrium fraction unknown"
        if stab < 0.3:
            return "unstable or exploratory regime"
    return "equilibrium status unclear"


def _breakthrough_label(prob: float) -> str:
    if prob >= 0.8:
        return "very high breakthrough chance"
    if prob >= 0.6:
        return "high breakthrough chance"
    if prob >= 0.4:
        return "moderate breakthrough chance"
    if prob >= 0.2:
        return "low breakthrough chance"
    return "very low breakthrough chance"


def _swarm_role_stats(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate RYE by role so swarm performance can be inspected."""
    stats: Dict[str, Dict[str, Any]] = {}
    for e in history:
        role = str(e.get("role", "agent") or "agent")
        rye = e.get("RYE")
        if not isinstance(rye, (int, float)):
            continue
        r = stats.setdefault(role, {"count": 0, "rye_sum": 0.0, "rye_vals": []})
        r["count"] += 1
        r["rye_sum"] += float(rye)
        r["rye_vals"].append(float(rye))

    for role, r in list(stats.items()):
        cnt = max(int(r.get("count", 0)), 1)
        avg = r["rye_sum"] / float(cnt)
        med = _median(r["rye_vals"])
        stats[role] = {
            "count": cnt,
            "avg_rye": avg,
            "median_rye": med,
        }
    return stats


def _goal_summary(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    goals: Dict[str, int] = {}
    for e in history:
        g = str(e.get("goal") or "").strip()
        if not g:
            continue
        short = g if len(g) <= 140 else g[:137] + "..."
        goals[short] = goals.get(short, 0) + 1
    sorted_items = sorted(goals.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "unique_goals": len(goals),
        "top_goals": [{"text": g, "cycles": c} for g, c in sorted_items[:5]],
    }


@dataclass
class LearningSpeedBlock:
    slope_per_cycle: Optional[float]
    trend_simple: Optional[float]
    rolling_rye: Optional[float]
    volatility: Optional[Dict[str, Any]]
    label: str


@dataclass
class EquilibriumBlock:
    metrics: Dict[str, Any]
    label: str


@dataclass
class BreakthroughBlock:
    probability_short_term: Optional[float]
    label_short_term: str
    probability_90d: Optional[float]
    label_90d: str
    run_tier: Optional[str]


@dataclass
class SwarmBlock:
    roles: Dict[str, Dict[str, Any]]


@dataclass
class GoldNotebookSnapshot:
    timestamp_utc: str
    domain: Optional[str]
    total_cycles: int
    goals: Dict[str, Any]
    diagnostics: Dict[str, Any]
    learning: LearningSpeedBlock
    equilibrium: EquilibriumBlock
    breakthroughs: BreakthroughBlock
    safety_envelope: Dict[str, Any]
    failure_warning: Dict[str, Any]
    option_c_signature: Dict[str, Any]
    swarm: SwarmBlock
    notes: Dict[str, Any]


# --------------------------------------------------------------------
# Core snapshot builder
# --------------------------------------------------------------------


def build_gold_snapshot(
    *,
    memory_store: MemoryStore,
    domain: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
) -> GoldNotebookSnapshot:
    """Build a structured snapshot of the current run for analysis."""

    history = memory_store.get_cycle_history()
    total_cycles = len(history)

    diagnostics = build_run_diagnostics(history=history, domain=domain)

    vol = rye_volatility_signature(history)
    eq = detect_rye_equilibrium(history)
    harm = tgrm_harmonic_index(history)

    bp = estimate_breakthrough_probability(diagnostics, domain=domain, horizon_hours=None)
    bp_prob = _safe_float(bp.get("probability")) if isinstance(bp, dict) else None

    bp90 = breakthrough_likelihood_90d(
        diagnostics,
        domain=domain,
        hours_run_so_far=hours_run_so_far,
    )
    bp90_prob = None
    if isinstance(bp90, dict):
        bp90_prob = _safe_float(bp90.get("probability"))

    env = autonomy_safety_envelope(diagnostics)
    fail = early_failure_warning_score(diagnostics)

    tier_info = classify_run_tier(diagnostics, breakthrough_prob=bp_prob)
    if isinstance(tier_info, dict):
        run_tier = tier_info.get("tier")
    else:
        run_tier = None

    option_c_sig = build_option_c_signature(
        history,
        domain=domain,
        hours_run_so_far=hours_run_so_far,
    )

    # Learning speed from diagnostics
    slope = diagnostics.get("trend_slope")
    trend_simple = diagnostics.get("trend_simple")
    rolling = diagnostics.get("rolling_rye")

    learning_block = LearningSpeedBlock(
        slope_per_cycle=slope if isinstance(slope, (int, float)) else None,
        trend_simple=trend_simple if isinstance(trend_simple, (int, float)) else None,
        rolling_rye=rolling if isinstance(rolling, (int, float)) else None,
        volatility=vol or {},
        label=_learning_speed_label(slope if isinstance(slope, (int, float)) else None),
    )

    eq_block = EquilibriumBlock(
        metrics=eq or {},
        label=_equilibrium_label(eq or {}),
    )

    breakthroughs_block = BreakthroughBlock(
        probability_short_term=bp_prob if isinstance(bp_prob, float) else None,
        label_short_term=_breakthrough_label(bp_prob) if isinstance(bp_prob, float) else "unknown",
        probability_90d=bp90_prob if isinstance(bp90_prob, float) else None,
        label_90d=_breakthrough_label(bp90_prob) if isinstance(bp90_prob, float) else "unknown",
        run_tier=str(run_tier) if run_tier is not None else None,
    )

    swarm_block = SwarmBlock(
        roles=_swarm_role_stats(history),
    )

    goals_info = _goal_summary(history)

    notes = {
        "total_discoveries": None,
        "top_discovery_labels": [],
        "harmonic_index": harm,
    }

    # Optional: pull discovery log if MemoryStore exposes helper
    try:
        if hasattr(memory_store, "get_discoveries"):
            disc = memory_store.get_discoveries(goal=None)
        else:
            disc = []
    except Exception:
        disc = []

    if isinstance(disc, list):
        notes["total_discoveries"] = len(disc)
        labels: List[str] = []
        for d in disc[:6]:
            label = d.get("label") or d.get("title") or d.get("summary")
            if label:
                labels.append(str(label))
        notes["top_discovery_labels"] = labels

    snapshot = GoldNotebookSnapshot(
        timestamp_utc=_iso_now(),
        domain=domain,
        total_cycles=total_cycles,
        goals=goals_info,
        diagnostics=diagnostics,
        learning=learning_block,
        equilibrium=eq_block,
        breakthroughs=breakthroughs_block,
        safety_envelope=env or {},
        failure_warning=fail or {},
        option_c_signature=option_c_sig or {},
        swarm=swarm_block,
        notes=notes,
    )
    return snapshot


# --------------------------------------------------------------------
# Markdown notebook view
# --------------------------------------------------------------------


def gold_notebook_markdown(snapshot: GoldNotebookSnapshot) -> str:
    """Render a Gold Notebook snapshot as human readable Markdown."""
    s = snapshot  # short alias

    lines: List[str] = []
    lines.append("# Gold Notebook\n")
    lines.append(f"- Generated at: `{s.timestamp_utc}`")
    if s.domain:
        lines.append(f"- Domain: `{s.domain}`")
    lines.append(f"- Total cycles: **{s.total_cycles}**")
    lines.append("")

    # Goals
    lines.append("## Goals\n")
    lines.append(f"- Unique goals touched: **{s.goals.get('unique_goals', 0)}**")
    top_goals = s.goals.get("top_goals") or []
    if top_goals:
        lines.append("- Top goals by cycle count:")
        for g in top_goals:
            lines.append(f"  - {g['cycles']} cycles on: {g['text']}")
    lines.append("")

    # Learning speed
    lines.append("## Learning speed\n")
    lines.append(f"- Learning speed label: **{s.learning.label}**")
    if s.learning.slope_per_cycle is not None:
        lines.append(f"- RYE slope per cycle: **{s.learning.slope_per_cycle:.5f}**")
    else:
        lines.append("- RYE slope per cycle: n/a")
    if s.learning.trend_simple is not None:
        lines.append(f"- Trend (first half vs second half): **{s.learning.trend_simple:.3f}**")
    else:
        lines.append("- Trend (first half vs second half): n/a")
    if s.learning.rolling_rye is not None:
        lines.append(f"- Rolling RYE (last window): **{s.learning.rolling_rye:.3f}**")
    else:
        lines.append("- Rolling RYE (last window): n/a")
    if s.learning.volatility:
        vol = s.learning.volatility
        vol_desc = {
            "std_dev": vol.get("std_dev"),
            "coefficient_of_variation": vol.get("cv"),
            "spikes": vol.get("spikes"),
        }
        lines.append("- Volatility snapshot:")
        lines.append(f"  - {json.dumps(vol_desc, indent=2)}")
    lines.append("")

    # Equilibrium
    lines.append("## Equilibrium and stability\n")
    lines.append(f"- Equilibrium label: **{s.equilibrium.label}**")
    eqm = s.equilibrium.metrics
    if eqm:
        lines.append("- Raw equilibrium metrics:")
        lines.append("```json")
        lines.append(json.dumps(eqm, indent=2))
        lines.append("```")
    else:
        lines.append("- No equilibrium metrics available.")
    lines.append("")

    # Breakthroughs
    lines.append("## Breakthrough outlook\n")
    lines.append(f"- Short term probability: **{s.breakthroughs.probability_short_term}**")
    lines.append(f"- Short term label: **{s.breakthroughs.label_short_term}**")
    lines.append(f"- 90 day probability: **{s.breakthroughs.probability_90d}**")
    lines.append(f"- 90 day label: **{s.breakthroughs.label_90d}**")
    if s.breakthroughs.run_tier:
        lines.append(f"- Run tier classification: **{s.breakthroughs.run_tier}**")
    lines.append("")

    # Swarm
    lines.append("## Swarm and roles\n")
    if not s.swarm.roles:
        lines.append("Single role or swarm not used.")
    else:
        lines.append("| Role | Cycles | Avg RYE | Median RYE |")
        lines.append("|------|--------|---------|------------|")
        for role, r in sorted(s.swarm.roles.items(), key=lambda kv: kv[0]):
            avg = r.get("avg_rye")
            med = r.get("median_rye")
            lines.append(
                f"| {role} | {r['count']} | "
                f"{avg:.3f if isinstance(avg, (int, float)) else 'n/a'} | "
                f"{med:.3f if isinstance(med, (int, float)) else 'n/a'} |"
            )
    lines.append("")

    # Safety envelope
    lines.append("## Autonomy safety envelope\n")
    if s.safety_envelope:
        lines.append("```json")
        lines.append(json.dumps(s.safety_envelope, indent=2))
        lines.append("```")
    else:
        lines.append("No safety envelope metrics available.")
    lines.append("")

    # Failure warning
    lines.append("## Early failure warning\n")
    if s.failure_warning:
        lines.append("```json")
        lines.append(json.dumps(s.failure_warning, indent=2))
        lines.append("```")
    else:
        lines.append("No early failure metrics available.")
    lines.append("")

    # Option C signature
    lines.append("## Option C signature\n")
    if s.option_c_signature:
        lines.append("```json")
        lines.append(json.dumps(s.option_c_signature, indent=2))
        lines.append("```")
    else:
        lines.append("No Option C signature available.")
    lines.append("")

    # Notes and discoveries
    lines.append("## Notes and discovery hints\n")
    lines.append(f"- Total discoveries recorded: **{s.notes.get('total_discoveries')}**")
    if s.notes.get("harmonic_index") is not None:
        lines.append(f"- Harmonic index: **{s.notes['harmonic_index']:.3f}**")
    labels = s.notes.get("top_discovery_labels") or []
    if labels:
        lines.append("- Top discovery candidates:")
        for lbl in labels:
            lines.append(f"  - {lbl}")
    lines.append("")

    # Diagnostics (raw)
    lines.append("## Raw diagnostics bundle\n")
    lines.append("```json")
    lines.append(json.dumps(s.diagnostics, indent=2))
    lines.append("```")

    return "\n".join(lines)


# --------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------


def export_gold_notebook_json(
    snapshot: GoldNotebookSnapshot,
    output_path: str | Path,
) -> Path:
    """Save the snapshot as a JSON file and return the Path."""
    path = Path(output_path)
    payload = asdict(snapshot)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
