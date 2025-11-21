"""Report generator for the Autonomous Research Agent.

Builds a human-readable markdown report from the logged cycle history.
Uses RYE metrics as a core efficiency lens (Reparodynamics view).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rye_metrics import rolling_rye, efficiency_trend


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float, without throwing."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if x is None:
            return default
        return float(str(x))
    except Exception:
        return default


def generate_report(memory_store: Any, goal: Optional[str] = None) -> str:
    """Generate a markdown report from the agent's history.

    Args:
        memory_store:
            The MemoryStore instance used by the agent.
        goal:
            Optional filter by goal string. If provided, only cycles
            matching that goal are considered. If None, the full history
            is used.

    Returns:
        str: Markdown-formatted report text.
    """
    # For now we only rely on get_cycle_history(), which you already have.
    all_cycles: List[Dict[str, Any]] = memory_store.get_cycle_history()

    if goal:
        cycles = [c for c in all_cycles if (c.get("goal") or "") == goal]
    else:
        cycles = list(all_cycles)

    n_cycles = len(cycles)

    if n_cycles == 0:
        return "# Autonomous Research Agent Report\n\nNo cycles have been logged yet."

    # Aggregate basic stats
    rye_values: List[float] = []
    delta_values: List[float] = []
    energy_values: List[float] = []

    domains = set()
    goals_seen = set()
    timestamps: List[str] = []

    all_hypotheses: List[Dict[str, Any]] = []
    all_citations: List[Dict[str, Any]] = []

    for c in cycles:
        # Metrics (robust to weird types)
        rye_values.append(_safe_float(c.get("RYE"), 0.0))
        delta_values.append(_safe_float(c.get("delta_R"), 0.0))
        energy_values.append(_safe_float(c.get("energy_E"), 0.0))

        # Meta
        domains.add(c.get("domain", "general"))
        goals_seen.add(c.get("goal", ""))

        ts = c.get("timestamp")
        if isinstance(ts, str) and ts:
            timestamps.append(ts)

        # Hypotheses + citations
        hyps = c.get("hypotheses") or []
        cits = c.get("citations") or []

        # Normalize hypotheses to dicts for display
        for h in hyps:
            if isinstance(h, dict):
                all_hypotheses.append(h)
            else:
                all_hypotheses.append({"text": str(h), "confidence": None})

        for ct in cits:
            if isinstance(ct, dict):
                all_citations.append(ct)

    # Strip zeros if they were all defaulted
    rye_values = [v for v in rye_values if v != 0.0] or [0.0]
    delta_values = [v for v in delta_values if v != 0.0] or [0.0]
    energy_values = [v for v in energy_values if v != 0.0] or [0.0]

    avg_rye = sum(rye_values) / len(rye_values) if rye_values else 0.0
    avg_delta = sum(delta_values) / len(delta_values) if delta_values else 0.0
    avg_energy = sum(energy_values) / len(energy_values) if energy_values else 0.0

    roll = rolling_rye(cycles, window=10)
    trend = efficiency_trend(cycles)

    # Build report lines
    lines: List[str] = []
    lines.append("# Autonomous Research Agent Report")
    lines.append("")

    if goal:
        lines.append(f"**Filtered goal:** {goal}")
    else:
        # Summarize distinct goals briefly
        goals_list = [g for g in goals_seen if g]
        if goals_list:
            lines.append("**Goals touched in this session:**")
            for g in goals_list[:10]:
                trimmed = g if len(g) <= 100 else g[:97] + "..."
                lines.append(f"- {trimmed}")
            if len(goals_list) > 10:
                lines.append(f"- ... and {len(goals_list) - 10} more")
        lines.append("")

    # Session time span (best effort from timestamps)
    if timestamps:
        first_ts = sorted(timestamps)[0]
        last_ts = sorted(timestamps)[-1]
        lines.append("**Session time span (UTC, best-effort from logs):**")
        lines.append(f"- First cycle: `{first_ts}`")
        lines.append(f"- Last cycle: `{last_ts}`")
        lines.append("")

    # Core stats
    lines.append("## Overall statistics")
    lines.append("")
    lines.append(f"- Total cycles: **{n_cycles}**")
    lines.append(f"- Domains involved: **{', '.join(sorted(domains))}**")
    lines.append(f"- Average RYE: **{avg_rye:.3f}**")
    lines.append(f"- Average ΔR per cycle: **{avg_delta:.3f}**")
    lines.append(f"- Average energy per cycle: **{avg_energy:.3f}**")
    if roll is not None:
        lines.append(f"- Rolling RYE (last 10 cycles): **{roll:.3f}**")
    if trend is not None:
        direction = "improving" if trend > 0 else "declining" if trend < 0 else "flat"
        lines.append(f"- RYE trend (recent - old): **{trend:.3f}** ({direction} efficiency)")
    lines.append("")

    # Hypotheses section
    lines.append("## Generated hypotheses")
    lines.append("")
    if not all_hypotheses:
        lines.append("No hypotheses were logged in this session.")
    else:
        for i, h in enumerate(all_hypotheses[:50], start=1):
            text = h.get("text", "")
            conf = h.get("confidence")
            if conf is not None:
                lines.append(f"{i}. {text} _(confidence ~ {conf})_")
            else:
                lines.append(f"{i}. {text}")
        if len(all_hypotheses) > 50:
            lines.append(f"... and {len(all_hypotheses) - 50} more hypotheses.")
    lines.append("")

    # Citations section
    lines.append("## Key citations (sources used)")
    lines.append("")
    if not all_citations:
        lines.append("No external citations were recorded.")
    else:
        # Collapse duplicates by (source, title, url)
        seen = set()
        unique_cites: List[Dict[str, Any]] = []
        for ct in all_citations:
            key = (
                str(ct.get("source", "")),
                str(ct.get("title", "")),
                str(ct.get("url", "")),
            )
            if key in seen:
                continue
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
            lines.append(f"... and {len(unique_cites) - 50} more sources.")
    lines.append("")

    # Reparodynamic interpretation
    lines.append("## Reparodynamic interpretation")
    lines.append("")
    lines.append(
        "From a Reparodynamics perspective, this session represents a sequence of TGRM cycles "
        "(Test → Detect → Repair → Verify) where the agent attempted to reduce defects "
        "(gaps, TODOs, contradictions) while minimizing effort."
    )
    lines.append(
        f"The average RYE of **{avg_rye:.3f}** captures how much verified improvement (ΔR) "
        "was achieved per unit of energy (E). A positive RYE trend suggests that the system "
        "is learning to repair itself more efficiently over time, while a negative trend "
        "indicates diminishing returns or increasing repair difficulty."
    )
    lines.append("")

    return "\n".join(lines)
