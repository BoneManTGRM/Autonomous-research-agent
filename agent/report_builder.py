# agent_report_builder.py
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

This replaces NOTHING.
It is a NEW MODULE that supercharges your existing report_generator.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from agent.memory_store import MemoryStore


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
    Generates a FULL Option C scientific report:

    - Cycle history summary
    - RYE diagnostics bundle
    - Stability & trend analysis
    - Option C composite signature
    - Breakthrough estimators
    - Run tier classification
    - Tool stats (from MemoryStore)
    - Swarm mode summary
    - Intelligence profile summary
    - Biomarkers (anti-aging mode)
    """

    history = memory_store.get_cycle_history()
    tool_stats = memory_store.get_tool_stats()

    # Core metrics
    diag = build_run_diagnostics(history, domain=domain)
    vol = rye_volatility_signature(history)
    eq = detect_rye_equilibrium(history)
    harm = tgrm_harmonic_index(history)
    bp = estimate_breakthrough_probability(diag, domain=domain, horizon_hours=None)
    bp90 = breakthrough_likelihood_90d(diag, domain=domain, hours_run_so_far=hours_run_so_far)
    env = autonomy_safety_envelope(diag)
    fail = early_failure_warning_score(diag)
    tier = classify_run_tier(diag, breakthrough_prob=bp["probability"])

    option_c_signature = build_option_c_signature(
        history,
        domain=domain,
        hours_run_so_far=hours_run_so_far,
    )

    # -------------------------------------------------------
    # Build Markdown
    # -------------------------------------------------------

    lines: List[str] = []

    # Header
    lines.append(f"# Autonomous Research Agent — Full Option C Report")
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

    # 3. Volatility
    lines.append(_md_section("Volatility Signature", _json(vol)))

    # 4. Equilibrium Detection
    lines.append(_md_section("Equilibrium Detection", _json(eq)))

    # 5. TGRM Harmonic Index
    lines.append(_md_section("TGRM Harmonic Index", _json({"harmonic_index": harm})))

    # 6. Breakthrough Probability (near term)
    lines.append(_md_section("Breakthrough Probability (Short-Term)", _json(bp)))

    # 7. 90-Day Breakthrough Likelihood
    lines.append(_md_section("90-Day Breakthrough Likelihood", _json(bp90)))

    # 8. Autonomy Safety Envelope
    lines.append(_md_section("Autonomy–Stability Envelope", _json(env)))

    # 9. Early Failure Warning Score
    lines.append(_md_section("Critical-Failure Early Warning", _json(fail)))

    # 10. Run Tier Classification
    lines.append(_md_section("Run Tier Classification", _json(tier)))

    # 11. Option C Composite Signature
    lines.append(_md_section("Option C Composite Signature", _json(option_c_signature)))

    # 12. Tool Diagnostics
    lines.append(_md_section("Tool Diagnostics", _json(tool_stats)))

    # 13. Swarm Stats
    lines.append(_md_section("Swarm Stats", _json(swarm_stats or {})))

    # 14. Intelligence Profile
    lines.append(_md_section("Intelligence Profile", _json(intelligence_profile or {})))

    # 15. Biomarker Snapshot (Longevity Mode)
    lines.append(_md_section("Biomarker Snapshot", _json(biomarker_snapshot or {})))

    # 16. Raw Cycle History (Optional)
    lines.append(
        _md_section(
            "Cycle History (condensed)",
            f"Cycles: {len(history)}\n\nOnly the last 20 entries shown:\n\n```json\n{_json(history[-20:])}\n```"
        )
    )

    return "\n".join(lines)
