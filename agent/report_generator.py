"""Extended Report Generator for the Autonomous Research Agent.

Outputs:
1. Full Reparodynamics Report (rich RYE and swarm analytics)
2. Targeted Findings Report (cures, treatments, mechanisms, interventions)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .rye_metrics import (
    rolling_rye,
    efficiency_trend,
    median_rye,
    stability_index,
    recovery_momentum,
    regression_rye_slope,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    """Best effort conversion to float."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if x is None:
            return default
        return float(str(x))
    except Exception:
        return default


def _extract_session_runtime(timestamps: List[str]) -> Optional[str]:
    """Return human readable runtime if timestamps exist."""
    if not timestamps:
        return None

    from datetime import datetime

    # Accept both microsecond and plain second ISO formats
    fmts = ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ")

    def _parse(ts: str) -> Optional[datetime]:
        for fmt in fmts:
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
        return None

    try:
        parsed = [p for t in sorted(timestamps) if (p := _parse(t)) is not None]
        if not parsed:
            return None
        first = parsed[0]
        last = parsed[-1]
        diff = last - first
        hours = diff.total_seconds() / 3600.0
        minutes = diff.total_seconds() / 60.0
        return f"{hours:.2f} hours ({minutes:.1f} minutes)"
    except Exception:
        return None


def _domain_and_role_stats(
    cycles: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Compute per domain and per role summary stats.

    Returns:
        domain_stats: {domain: {"count": int, "avg_rye": float}}
        role_stats:   {role:   {"count": int, "avg_rye": float}}
    """
    domain_stats: Dict[str, Dict[str, Any]] = {}
    role_stats: Dict[str, Dict[str, Any]] = {}

    for c in cycles:
        rye = c.get("RYE")
        if not isinstance(rye, (int, float)):
            continue
        rye_f = float(rye)

        domain = str(c.get("domain", "general") or "general")
        role = str(c.get("role", "agent") or "agent")

        ds = domain_stats.setdefault(domain, {"sum": 0.0, "count": 0})
        ds["sum"] += rye_f
        ds["count"] += 1

        rs = role_stats.setdefault(role, {"sum": 0.0, "count": 0})
        rs["sum"] += rye_f
        rs["count"] += 1

    # Convert sums to averages
    for d, v in list(domain_stats.items()):
        cnt = max(int(v.get("count", 0)), 1)
        domain_stats[d] = {
            "count": cnt,
            "avg_rye": float(v.get("sum", 0.0)) / float(cnt),
        }

    for r, v in list(role_stats.items()):
        cnt = max(int(v.get("count", 0)), 1)
        role_stats[r] = {
            "count": cnt,
            "avg_rye": float(v.get("sum", 0.0)) / float(cnt),
        }

    return domain_stats, role_stats


def _classify_phase(
    slope: Optional[float],
    trend: Optional[float],
    stab: Optional[float],
) -> str:
    """Heuristic phase label using slope, trend, and stability index."""
    if slope is None and trend is None:
        return "insufficient data"

    s = slope or 0.0
    t = trend or 0.0
    st = stab if isinstance(stab, (int, float)) else None

    improving = s > 0.0 or t > 0.0
    declining = s < 0.0 and t < 0.0

    if st is not None:
        if improving and st >= 0.7:
            return "stable improving repair phase"
        if improving and st < 0.7:
            return "noisy but improving repair phase"
        if declining and st >= 0.7:
            return "stable decline phase"
        if declining and st < 0.7:
            return "chaotic decline phase"
        if st >= 0.8:
            return "high stability equilibrium zone"
        if st <= 0.3:
            return "highly volatile exploration zone"

    if improving:
        return "improving efficiency"
    if declining:
        return "declining efficiency"
    return "mixed or flat efficiency"


# ---------------------------------------------------------
# FULL REPORT (Advanced)
# ---------------------------------------------------------
def generate_report(memory_store: Any, goal: Optional[str] = None) -> str:
    """Generate full Reparodynamics markdown report with advanced metrics."""

    all_cycles: List[Dict[str, Any]] = memory_store.get_cycle_history()

    if goal:
        cycles = [c for c in all_cycles if (c.get("goal") or "") == goal]
    else:
        cycles = list(all_cycles)

    n_cycles = len(cycles)

    if n_cycles == 0:
        return "# Autonomous Research Agent Report\n\nNo cycles logged."

    # Metric stores
    rye_values: List[float] = []
    delta_values: List[float] = []
    energy_values: List[float] = []
    domains: set = set()
    goals_seen: set = set()
    timestamps: List[str] = []
    all_hypotheses: List[Dict[str, Any]] = []
    all_citations: List[Dict[str, Any]] = []

    for c in cycles:
        rye_values.append(_safe_float(c.get("RYE"), 0.0))
        delta_values.append(_safe_float(c.get("delta_R"), 0.0))
        energy_values.append(_safe_float(c.get("energy_E"), 0.0))

        domains.add(c.get("domain", "general"))
        goals_seen.add(c.get("goal", ""))

        ts = c.get("timestamp")
        if isinstance(ts, str) and ts:
            timestamps.append(ts)

        # Hypotheses and citations attached to cycles
        hyps = c.get("hypotheses") or []
        cits = c.get("citations") or []

        for h in hyps:
            if isinstance(h, dict):
                all_hypotheses.append(h)
            else:
                all_hypotheses.append({"text": str(h), "confidence": None})

        for ct in cits:
            if isinstance(ct, dict):
                all_citations.append(ct)

    # Aggregate metrics
    avg_rye = sum(rye_values) / len(rye_values) if rye_values else 0.0
    avg_delta = sum(delta_values) / len(delta_values) if delta_values else 0.0
    avg_energy = sum(energy_values) / len(energy_values) if energy_values else 0.0

    roll = rolling_rye(cycles, window=10)
    trend = efficiency_trend(cycles)
    med_rye = median_rye(cycles)
    stab = stability_index(cycles)
    momentum = recovery_momentum(cycles)
    slope = regression_rye_slope(cycles)
    runtime = _extract_session_runtime(timestamps)

    domain_stats, role_stats = _domain_and_role_stats(cycles)

    # Optional goal index and discoveries from memory store
    goal_index_entry: Optional[Dict[str, Any]] = None
    try:
        goal_index_entry = memory_store.get_goal_index(goal) if hasattr(memory_store, "get_goal_index") else None
    except Exception:
        goal_index_entry = None

    discoveries: List[Dict[str, Any]] = []
    try:
        discoveries = memory_store.get_discoveries(goal=goal) if hasattr(memory_store, "get_discoveries") else []
    except Exception:
        discoveries = []

    # Build report
    lines: List[str] = []
    lines.append("# Autonomous Research Agent Report\n")

    # Runtime
    if runtime:
        lines.append(f"**Session runtime:** {runtime}\n")

    # Goals
    if goal:
        lines.append(f"**Filtered goal:** {goal}\n")
    else:
        goals_list = [g for g in goals_seen if g]
        if goals_list:
            lines.append("**Goals touched during session:**")
            for g in goals_list[:10]:
                trimmed = g if len(g) <= 100 else g[:97] + "..."
                lines.append(f"- {trimmed}")
            if len(goals_list) > 10:
                lines.append(f"- ... and {len(goals_list) - 10} more")
        lines.append("")

    # Time span
    if timestamps:
        first_ts = sorted(timestamps)[0]
        last_ts = sorted(timestamps)[-1]
        lines.append("**Time span (UTC):**")
        lines.append(f"- First cycle: `{first_ts}`")
        lines.append(f"- Last cycle: `{last_ts}`\n")

    # Basic stats
    lines.append("## Overall statistics\n")
    lines.append(f"- Total cycles: **{n_cycles}**")
    lines.append(f"- Domains: **{', '.join(sorted(str(d) for d in domains))}**")
    lines.append(f"- Avg RYE: **{avg_rye:.3f}**")
    lines.append(f"- Avg ΔR: **{avg_delta:.3f}**")
    lines.append(f"- Avg Energy: **{avg_energy:.3f}**")

    if roll is not None:
        lines.append(f"- Rolling RYE (last 10): **{roll:.3f}**")

    if med_rye is not None:
        lines.append(f"- Median RYE (noise resistant): **{med_rye:.3f}**")

    if trend is not None:
        direction = "improving" if trend > 0 else "declining" if trend < 0 else "flat"
        lines.append(f"- RYE trend (first half vs second half): **{trend:.3f}** ({direction})")

    if slope is not None:
        lines.append(f"- Regression slope of RYE over cycles: **{slope:.5f}**")

    if stab is not None:
        lines.append(f"- Stability index: **{stab:.3f}** (1.0 is highly stable, 0.0 is chaotic)")

    if momentum is not None:
        lines.append(f"- Recovery momentum: **{momentum:.3f}** (higher means late stage acceleration)")

    phase_label = _classify_phase(slope, trend, stab)
    lines.append(f"- Phase classification: **{phase_label}**")
    lines.append("")

    # Domain level view
    lines.append("## Domain level RYE profile\n")
    if not domain_stats:
        lines.append("No RYE data per domain.\n")
    else:
        lines.append("| Domain | Cycles | Avg RYE |")
        lines.append("|--------|--------|---------|")
        for d, stats in sorted(domain_stats.items(), key=lambda kv: kv[0]):
            lines.append(f"| {d} | {stats['count']} | {stats['avg_rye']:.3f} |")
        lines.append("")

    # Swarm role view
    lines.append("## Swarm role efficiency\n")
    if not role_stats or (len(role_stats) == 1 and "agent" in role_stats):
        lines.append("Single role run or insufficient role diversity for swarm analysis.\n")
    else:
        lines.append("| Role | Cycles | Avg RYE |")
        lines.append("|------|--------|---------|")
        for r, stats in sorted(role_stats.items(), key=lambda kv: kv[0]):
            lines.append(f"| {r} | {stats['count']} | {stats['avg_rye']:.3f} |")
        lines.append("")

    # Goal index, if available
    if goal_index_entry:
        lines.append("## Goal index snapshot\n")
        gi = goal_index_entry
        lines.append(f"- Created at: `{gi.get('created_at', 'unknown')}`")
        lines.append(f"- Last updated: `{gi.get('last_updated', 'unknown')}`")
        lines.append(f"- Total cycles counted: **{gi.get('cycle_count', 0)}**")
        lines.append(f"- Total notes counted: **{gi.get('note_count', 0)}**")
        if isinstance(gi.get("avg_rye"), (int, float)):
            lines.append(f"- Streaming avg RYE: **{float(gi['avg_rye']):.3f}**")
        if isinstance(gi.get("rye_count"), int):
            lines.append(f"- RYE samples tracked: **{gi['rye_count']}**")
        lines.append("")

    # Hypotheses
    lines.append("## Generated hypotheses\n")
    if not all_hypotheses:
        lines.append("No hypotheses generated.\n")
    else:
        for i, h in enumerate(all_hypotheses[:60], start=1):
            t = h.get("text", "")
            conf = h.get("confidence")
            score = h.get("score") if "score" in h else None
            label_parts = []
            if isinstance(conf, (int, float)):
                label_parts.append(f"conf {conf}")
            if isinstance(score, (int, float)):
                label_parts.append(f"score {score:.2f}")
            if label_parts:
                meta_str = ", ".join(label_parts)
                lines.append(f"{i}. {t} _({meta_str})_")
            else:
                lines.append(f"{i}. {t}")
        if len(all_hypotheses) > 60:
            lines.append(f"... and {len(all_hypotheses) - 60} more.\n")

    # Citations
    lines.append("\n## Key citations\n")
    if not all_citations:
        lines.append("No citations recorded.\n")
    else:
        seen = set()
        unique_cites: List[Dict[str, Any]] = []
        for ct in all_citations:
            key = (ct.get("source"), ct.get("title"), ct.get("url"))
            if key not in seen:
                seen.add(key)
                unique_cites.append(ct)

        for i, ct in enumerate(unique_cites[:50], start=1):
            src = ct.get("source", "web")
            title = ct.get("title", "Untitled")
            url = ct.get("url", "")
            lines.append(f"{i}. **[{src}]** {title}")
            if url:
                lines.append(f"   - {url}")
        if len(unique_cites) > 50:
            lines.append(f"... and {len(unique_cites) - 50} more.\n")

    # Discoveries
    lines.append("\n## Structured discoveries\n")
    if not discoveries:
        lines.append("No structured discoveries have been recorded yet.\n")
    else:
        kind_counts: Dict[str, int] = {}
        for d in discoveries:
            k = str(d.get("kind", "unknown") or "unknown")
            kind_counts[k] = kind_counts.get(k, 0) + 1
        lines.append("Summary by kind:")
        for k, cnt in sorted(kind_counts.items(), key=lambda kv: kv[0]):
            lines.append(f"- {k}: **{cnt}**")
        lines.append("\nRecent discoveries:")
        for d in discoveries[-10:]:
            ts = d.get("timestamp", "")
            kind = d.get("kind", "")
            label = d.get("label", "")
            score = d.get("score")
            if isinstance(score, (int, float)):
                lines.append(f"- [{ts}] [{kind}] ({score:.2f}) {label}")
            else:
                lines.append(f"- [{ts}] [{kind}] {label}")
        lines.append("")

    # Interpretation
    lines.append("## Reparodynamics interpretation\n")
    lines.append(
        "This session reflects a sequence of TGRM cycles (Test → Detect → Repair → Verify). "
        "RYE quantifies how much verified improvement (ΔR) occurred per unit energy (E). "
        "The stability index, momentum, and slope together indicate whether the system is moving "
        "toward a stable high repair yield equilibrium or oscillating in a more exploratory regime."
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# TARGETED FINDINGS REPORT
# ---------------------------------------------------------
def generate_findings_report(memory_store: Any, goal: Optional[str] = None) -> str:
    """
    Extract actionable findings such as:
    - Possible cures
    - Potential treatments
    - Interventions
    - Mechanisms
    - High value biomarkers or targets
    """

    all_cycles = memory_store.get_cycle_history()

    if goal:
        cycles = [c for c in all_cycles if (c.get("goal") or "") == goal]
    else:
        cycles = list(all_cycles)

    if not cycles:
        return "# Findings Report\n\nNo cycles found."

    timestamps: List[str] = []
    for c in cycles:
        ts = c.get("timestamp")
        if ts:
            timestamps.append(ts)

    runtime = _extract_session_runtime(timestamps)

    # Structured discoveries
    structured: List[Dict[str, Any]] = []
    try:
        structured = memory_store.get_discoveries(goal=goal) if hasattr(memory_store, "get_discoveries") else []
    except Exception:
        structured = []

    # Text mined findings from hypotheses
    findings_text: List[str] = []
    KEYWORDS = [
        "treatment",
        "cure",
        "therapy",
        "intervention",
        "mechanism",
        "pathway",
        "target",
        "biomarker",
        "drug",
        "compound",
        "protocol",
        "longevity",
        "anti-aging",
        "anti aging",
    ]

    for c in cycles:
        hyps = c.get("hypotheses") or []
        for h in hyps:
            text = h["text"] if isinstance(h, dict) else str(h)
            if any(k.lower() in text.lower() for k in KEYWORDS):
                findings_text.append(text)

    # Build report
    lines: List[str] = []
    lines.append("# Targeted Findings Report\n")

    if runtime:
        lines.append(f"**Session runtime:** {runtime}\n")

    if goal:
        lines.append(f"**Filtered goal:** {goal}\n")

    # Structured discoveries first
    lines.append("## Structured discoveries from agent memory\n")
    if not structured:
        lines.append("No structured discoveries recorded.\n")
    else:
        # Sort by score then timestamp
        def _disc_key(d: Dict[str, Any]) -> Tuple[float, str]:
            score = d.get("score")
            s_val = float(score) if isinstance(score, (int, float)) else 0.0
            ts = str(d.get("timestamp", ""))
            return (-s_val, ts)

        sorted_disc = sorted(structured, key=_disc_key)
        for i, d in enumerate(sorted_disc[:50], start=1):
            kind = d.get("kind", "")
            label = d.get("label", "")
            score = d.get("score")
            ts = d.get("timestamp", "")
            score_str = f" (score {score:.2f})" if isinstance(score, (int, float)) else ""
            lines.append(f"{i}. [{kind}] {label}{score_str}")
            if ts:
                lines.append(f"   - recorded at `{ts}`")
        if len(sorted_disc) > 50:
            lines.append(f"... and {len(sorted_disc) - 50} more structured discoveries.\n")

    # Text mined findings
    lines.append("\n## Text mined findings from hypotheses\n")
    if not findings_text:
        lines.append("No cure or treatment related hypotheses detected.\n")
    else:
        seen_text = set()
        unique_findings: List[str] = []
        for f in findings_text:
            f_norm = f.strip()
            if f_norm and f_norm not in seen_text:
                seen_text.add(f_norm)
                unique_findings.append(f_norm)

        for i, f in enumerate(unique_findings[:80], start=1):
            lines.append(f"{i}. {f}")
        if len(unique_findings) > 80:
            lines.append(f"... and {len(unique_findings) - 80} more hypotheses mentioning treatments or mechanisms.\n")

    return "\n".join(lines)
