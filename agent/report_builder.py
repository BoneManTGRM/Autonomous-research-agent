"""
Advanced Report Builder for the Autonomous Research Agent (Option C Edition)

This combines:
- Full run diagnostics (RYE Level 3, Stability, Momentum, Trends)
- Option C signatures (equilibrium, harmonic index, volatility)
- Breakthrough estimators (24h, 7d, 90d)
- Swarm role summaries
- Intelligence profile summary
- Biomarker snapshots (if enabled)
- PDF-ready structured fields
- Markdown builder for Streamlit + headless use

This replaces nothing.
It is a new module that supercharges your existing report_generator.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rye_metrics import (
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

from .memory_store import MemoryStore


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _md_section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n\n---\n\n"


def _safe_get_cycle_history(memory_store: MemoryStore) -> List[Dict[str, Any]]:
    """Defensive wrapper for MemoryStore.get_cycle_history."""
    try:
        hist = memory_store.get_cycle_history()
        if isinstance(hist, list):
            return hist
    except Exception:
        pass
    return []


def _safe_get_tool_stats(memory_store: MemoryStore) -> Dict[str, Any]:
    """Defensive wrapper for MemoryStore.get_tool_stats."""
    try:
        if hasattr(memory_store, "get_tool_stats"):
            stats = memory_store.get_tool_stats()  # type: ignore[attr-defined]
            if isinstance(stats, dict):
                return stats
    except Exception:
        pass
    return {}


# -------------------------------------------------------------
# MAIN REPORT BUILDER
# -------------------------------------------------------------

def build_agent_report(
    *,
    memory_store: MemoryStore,
    goal: Optional[str] = None,
    domain: Optional[str] = None,
    hours_run_so_far: Optional[float] = None,
    swarm_stats: Optional[Dict[str, Any]] = None,
    intelligence_profile: Optional[Dict[str, Any]] = None,
    biomarker_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a full Option C scientific report:

    - Cycle history summary
    - RYE diagnostics bundle
    - Stability and trend analysis
    - Option C composite signature
    - Breakthrough estimators
    - Run tier classification
    - Tool stats (from MemoryStore)
    - Swarm mode summary
    - Intelligence profile summary
    - Biomarkers (anti-aging mode)
    """

    # History and tool stats with defensive wrappers
    history = _safe_get_cycle_history(memory_store)
    tool_stats = _safe_get_tool_stats(memory_store)

    # Core metrics with best effort guards
    try:
        diag_raw = build_run_diagnostics(history, domain=domain)
        diag: Dict[str, Any] = diag_raw if isinstance(diag_raw, dict) else {}
    except Exception:
        diag = {}

    try:
        vol = rye_volatility_signature(history)
    except Exception:
        vol = {}

    try:
        eq = detect_rye_equilibrium(history)
    except Exception:
        eq = {}

    try:
        harm = tgrm_harmonic_index(history)
    except Exception:
        harm = None

    try:
        bp_raw = estimate_breakthrough_probability(diag, domain=domain, horizon_hours=None)
        bp: Dict[str, Any] = bp_raw if isinstance(bp_raw, dict) else {}
    except Exception:
        bp = {}

    try:
        bp90_raw = breakthrough_likelihood_90d(
            diag,
            domain=domain,
            hours_run_so_far=hours_run_so_far,
        )
        bp90: Dict[str, Any] = bp90_raw if isinstance(bp90_raw, dict) else {}
    except Exception:
        bp90 = {}

    try:
        env = autonomy_safety_envelope(diag)
    except Exception:
        env = {}

    try:
        fail = early_failure_warning_score(diag)
    except Exception:
        fail = {}

    # Safe breakthrough probability extraction for tier classification
    bp_prob_raw = bp.get("probability") if isinstance(bp, dict) else None
    try:
        bp_prob = float(bp_prob_raw) if bp_prob_raw is not None else 0.0
    except Exception:
        bp_prob = 0.0

    try:
        tier_raw = classify_run_tier(diag, breakthrough_prob=bp_prob)
        tier: Dict[str, Any] = tier_raw if isinstance(tier_raw, dict) else {}
    except Exception:
        tier = {}

    try:
        option_c_signature = build_option_c_signature(
            history,
            domain=domain,
            hours_run_so_far=hours_run_so_far,
        )
    except Exception:
        option_c_signature = {}

    # Lightweight learning speed summary for quick inspection
    learning_speed_summary: Dict[str, Any] = {
        "trend_slope": diag.get("trend_slope") if isinstance(diag, dict) else None,
        "recovery_momentum": diag.get("recovery_momentum") if isinstance(diag, dict) else None,
        "stability_index": diag.get("stability_index") if isinstance(diag, dict) else None,
        "breakthrough_probability": bp.get("probability") if isinstance(bp, dict) else None,
        "breakthrough_likelihood_90d": bp90.get("probability") if isinstance(bp90, dict) else None,
        "run_tier": tier.get("tier") if isinstance(tier, dict) else None,
    }

    # Last 20 cycles condensed view
    if isinstance(history, list) and history:
        condensed_history = history[-20:]
    else:
        condensed_history = []

    # -------------------------------------------------------
    # Build Markdown
    # -------------------------------------------------------

    lines: List[str] = []

    # Header
    lines.append("# Autonomous Research Agent - Full Option C Report")
    lines.append(f"**Timestamp:** { _iso_now() }")
    if goal:
        lines.append(f"**Goal:** {goal}")
    if domain:
        lines.append(f"**Domain:** `{domain}`")
    lines.append("")
    lines.append("---\n")

    # 1. Cycle Overview
    lines.append(
        _md_section(
            "Cycle Overview",
            _json({
                "total_cycles": len(history),
                "domain": domain,
                "hours_run_so_far": hours_run_so_far,
            })
        )
    )

    # 2. RYE Diagnostics
    lines.append(_md_section("RYE Diagnostics", _json(diag)))

    # 3. Learning Speed Summary (10x signals)
    lines.append(_md_section("Learning Speed Summary", _json(learning_speed_summary)))

    # 4. Volatility
    lines.append(_md_section("Volatility Signature", _json(vol)))

    # 5. Equilibrium Detection
    lines.append(_md_section("Equilibrium Detection", _json(eq)))

    # 6. TGRM Harmonic Index
    lines.append(_md_section("TGRM Harmonic Index", _json({"harmonic_index": harm})))

    # 7. Breakthrough Probability (short term)
    lines.append(_md_section("Breakthrough Probability (Short-Term)", _json(bp)))

    # 8. 90 Day Breakthrough Likelihood
    lines.append(_md_section("90-Day Breakthrough Likelihood", _json(bp90)))

    # 9. Autonomy Stability Envelope
    lines.append(_md_section("Autonomy Stability Envelope", _json(env)))

    # 10. Critical Failure Early Warning
    lines.append(_md_section("Critical-Failure Early Warning", _json(fail)))

    # 11. Run Tier Classification
    lines.append(_md_section("Run Tier Classification", _json(tier)))

    # 12. Option C Composite Signature
    lines.append(_md_section("Option C Composite Signature", _json(option_c_signature)))

    # 13. Tool Diagnostics
    lines.append(_md_section("Tool Diagnostics", _json(tool_stats)))

    # 14. Swarm Stats
    lines.append(_md_section("Swarm Stats", _json(swarm_stats or {})))

    # 15. Intelligence Profile
    lines.append(_md_section("Intelligence Profile", _json(intelligence_profile or {})))

    # 16. Biomarker Snapshot (Longevity Mode)
    lines.append(_md_section("Biomarker Snapshot", _json(biomarker_snapshot or {})))

    # 17. Raw Cycle History (condensed)
    lines.append(
        _md_section(
            "Cycle History (condensed)",
            (
                f"Cycles: {len(history)}\n\n"
                "Only the last 20 entries shown:\n\n"
                f"```json\n{_json(condensed_history)}\n```"
            ),
        )
    )

    return "\n".join(lines)
