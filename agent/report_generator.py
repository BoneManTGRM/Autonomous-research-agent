"""Extended Report Generator for the Autonomous Research Agent.

Outputs:
1. Full Reparodynamics Report (original format)
2. Targeted Findings Report (cures/treatments/interventions only)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from .rye_metrics import rolling_rye, efficiency_trend


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if x is None:
            return default
        return float(str(x))
    except Exception:
        return default


def _extract_session_runtime(timestamps: List[str]) -> Optional[str]:
    """Return human-readable runtime if timestamps exist."""
    if not timestamps:
        return None

    from datetime import datetime

    try:
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        first = datetime.strptime(sorted(timestamps)[0], fmt)
        last = datetime.strptime(sorted(timestamps)[-1], fmt)
        diff = last - first
        hours = diff.total_seconds() / 3600
        minutes = diff.total_seconds() / 60

        return f"{hours:.2f} hours ({minutes:.1f} minutes)"
    except Exception:
        return None


# ---------------------------------------------------------
# FULL REPORT (Original)
# ---------------------------------------------------------
def generate_report(memory_store: Any, goal: Optional[str] = None) -> str:
    """Generate full Reparodynamics markdown report."""

    all_cycles: List[Dict[str, Any]] = memory_store.get_cycle_history()

    if goal:
        cycles = [c for c in all_cycles if (c.get("goal") or "") == goal]
    else:
        cycles = list(all_cycles)

    n_cycles = len(cycles)

    if n_cycles == 0:
        return "# Autonomous Research Agent Report\n\nNo cycles logged."

    # Metric stores
    rye_values, delta_values, energy_values = [], [], []
    domains, goals_seen, timestamps = set(), set(), []
    all_hypotheses, all_citations = [], []

    for c in cycles:
        rye_values.append(_safe_float(c.get("RYE"), 0.0))
        delta_values.append(_safe_float(c.get("delta_R"), 0.0))
        energy_values.append(_safe_float(c.get("energy_E"), 0.0))

        domains.add(c.get("domain", "general"))
        goals_seen.add(c.get("goal", ""))

        ts = c.get("timestamp")
        if isinstance(ts, str) and ts:
            timestamps.append(ts)

        # Hypotheses and citations
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

    # Compute aggregates
    avg_rye = sum(rye_values) / len(rye_values) if rye_values else 0.0
    avg_delta = sum(delta_values) / len(delta_values) if delta_values else 0.0
    avg_energy = sum(energy_values) / len(energy_values) if energy_values else 0.0

    roll = rolling_rye(cycles, window=10)
    trend = efficiency_trend(cycles)
    runtime = _extract_session_runtime(timestamps)

    # Build report
    lines = []
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

    # Timestamps
    if timestamps:
        first_ts = sorted(timestamps)[0]
        last_ts = sorted(timestamps)[-1]
        lines.append("**Time span (UTC):**")
        lines.append(f"- First cycle: `{first_ts}`")
        lines.append(f"- Last cycle: `{last_ts}`\n")

    # Stats
    lines.append("## Overall statistics\n")
    lines.append(f"- Total cycles: **{n_cycles}**")
    lines.append(f"- Domains: **{', '.join(sorted(domains))}**")
    lines.append(f"- Avg RYE: **{avg_rye:.3f}**")
    lines.append(f"- Avg ΔR: **{avg_delta:.3f}**")
    lines.append(f"- Avg Energy: **{avg_energy:.3f}**")

    if roll is not None:
        lines.append(f"- Rolling RYE (last 10): **{roll:.3f}**")

    if trend is not None:
        direction = "improving" if trend > 0 else "declining" if trend < 0 else "flat"
        lines.append(f"- RYE trend: **{trend:.3f}** ({direction})")

    lines.append("")

    # Hypotheses
    lines.append("## Generated hypotheses\n")
    if not all_hypotheses:
        lines.append("No hypotheses generated.\n")
    else:
        for i, h in enumerate(all_hypotheses[:60], start=1):
            t = h.get("text", "")
            conf = h.get("confidence")
            if conf is not None:
                lines.append(f"{i}. {t} _(confidence {conf})_")
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
        unique_cites = []
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

    # Interpretation
    lines.append("\n## Reparodynamics interpretation\n")
    lines.append(
        "This session reflects a sequence of TGRM cycles (Test → Detect → Repair → Verify). "
        "RYE expresses how much verified improvement (ΔR) occurred per unit energy (E). "
        "Trend > 0 indicates increasing repair efficiency."
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# TARGETED FINDINGS REPORT
# ---------------------------------------------------------
def generate_findings_report(memory_store: Any, goal: Optional[str] = None) -> str:
    """
    NEW: Extracts actionable findings such as:
    - Possible cures
    - Potential treatments
    - Interventions
    - Mechanisms
    - Priority-ranked items
    """

    all_cycles = memory_store.get_cycle_history()

    if goal:
        cycles = [c for c in all_cycles if (c.get("goal") or "") == goal]
    else:
        cycles = list(all_cycles)

    if not cycles:
        return "# Findings Report\n\nNo cycles found."

    findings = []
    timestamps = []

    KEYWORDS = [
        "treatment", "cure", "therapy",
        "intervention", "mechanism",
        "pathway", "target", "biomarker",
        "drug", "compound", "protocol",
        "longevity", "anti-aging"
    ]

    for c in cycles:
        ts = c.get("timestamp")
        if ts:
            timestamps.append(ts)

        hyps = c.get("hypotheses") or []
        for h in hyps:
            text = h["text"] if isinstance(h, dict) else str(h)
            if any(k.lower() in text.lower() for k in KEYWORDS):
                findings.append(text)

    runtime = _extract_session_runtime(timestamps)

    lines = []
    lines.append("# Targeted Findings Report\n")

    if runtime:
        lines.append(f"**Session runtime:** {runtime}\n")

    lines.append("## Extracted actionable findings\n")

    if not findings:
        lines.append("No cure/treatment/intervention findings detected.\n")
    else:
        for i, f in enumerate(findings[:80], start=1):
            lines.append(f"{i}. {f}")

        if len(findings) > 80:
            lines.append(f"... and {len(findings) - 80} more.\n")

    return "\n".join(lines)
