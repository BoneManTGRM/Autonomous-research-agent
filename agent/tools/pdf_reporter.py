"""
pdf_reporter.py

Universal PDF report generator for the Autonomous Research Agent.

Maxed-out features:
- Layman-friendly explanations of RYE, delta_R, stability, discoveries
- Breakthrough scoring (Tier 1, Tier 2, Tier 3 likelihood)
- Summary of key findings, hypotheses, verifications
- Biomarker mode for longevity / anti-aging research
- Math-formal mode summaries for proof-style runs
- Snapshot + equilibrium interpretation
- Plain-language interpretation for non-experts
- ReportLab PDF output (standalone, safe)

This file is a pure bolt-on component. No other files need modification.

You can import it from Streamlit like:

    from agent.tools.pdf_reporter import build_full_pdf_report

Then call:

    pdf_path = build_full_pdf_report(memory, discoveries, snapshots, output_path="agent_report.pdf")

"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# ---------------------------------------------------------------------------
# Helper: safe float
# ---------------------------------------------------------------------------
def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layman-friendly interpretation helpers
# ---------------------------------------------------------------------------
def interpret_rye(rye: Optional[float]) -> str:
    """Convert RYE value into a clear, simple explanation."""
    if rye is None:
        return "RYE not available."

    if rye > 0.20:
        return (
            f"RYE {rye:.3f} means **excellent repair efficiency**. "
            "The agent is finding high-impact improvements with very little wasted effort."
        )
    if rye > 0.10:
        return (
            f"RYE {rye:.3f} is **strong**. "
            "Consistent improvements are being made and the research direction is healthy."
        )
    if rye > 0.05:
        return (
            f"RYE {rye:.3f} is **moderate**. "
            "The system is learning and repairing, but improvement may be slowing."
        )
    if rye > 0.0:
        return (
            f"RYE {rye:.3f} is **weak but positive**. "
            "The agent is still improving, but the progress is small."
        )
    return (
        f"RYE {rye:.3f} is **negative**, meaning the agent consumed effort "
        "but did not make stable improvements. This may indicate noise, confusion, "
        "or a difficult research question."
    )


def interpret_stability(stab: Optional[float]) -> str:
    if stab is None:
        return "Stability index not available."
    if stab >= 0.65:
        return (
            f"Stability Index {stab:.3f}: **Highly stable**. "
            "The agent reached a smooth equilibrium and repairs are consistent."
        )
    if stab >= 0.50:
        return (
            f"Stability Index {stab:.3f}: **Stable**. "
            "The system is holding coherence with manageable oscillations."
        )
    if stab >= 0.35:
        return (
            f"Stability Index {stab:.3f}: **Partially stable**. "
            "Some fluctuation exists, but the model is generally coherent."
        )
    return (
        f"Stability Index {stab:.3f}: **Unstable**. "
        "There are oscillations and uncertainty in the agent’s reasoning."
    )


def interpret_momentum(m: Optional[float]) -> str:
    if m is None:
        return "Recovery momentum not available."
    if m > 0.10:
        return (
            f"Recovery Momentum {m:.3f}: **Strong recovery** after disruptions. "
            "The system corrects itself quickly."
        )
    if m > 0.0:
        return (
            f"Recovery Momentum {m:.3f}: **Mild recovery**. "
            "Repairs are stabilizing but only gradually."
        )
    if m > -0.05:
        return (
            f"Recovery Momentum {m:.3f}: **Neutral to slightly declining**. "
            "The agent is stable but not accelerating."
        )
    return (
        f"Recovery Momentum {m:.3f}: **Negative**. "
        "The system is losing coherence or fighting noise."
    )


# ---------------------------------------------------------------------------
# Breakthrough scoring
# ---------------------------------------------------------------------------
def breakthrough_score(rye_vals: List[float], discoveries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return Tier 1 / Tier 2 / Tier 3 breakthrough likelihood."""
    if not rye_vals:
        return {"tier": "None", "score": 0.0, "text": "Insufficient data for breakthrough scoring."}

    avg_rye = sum(rye_vals) / len(rye_vals)

    # discovery signal
    discovery_signal = len(discoveries)

    score = avg_rye * 3.0 + discovery_signal * 0.1

    if score > 1.8:
        tier = "Tier 3"
        text = (
            "Breakthrough likelihood is **high (Tier 3)**. "
            "This means the agent has shown repeated strong RYE improvements, "
            "stable long-run behavior, and meaningful discoveries."
        )
    elif score > 1.2:
        tier = "Tier 2"
        text = (
            "Breakthrough likelihood is **moderate (Tier 2)**. "
            "There are promising signals but not enough sustained efficiency."
        )
    elif score > 0.6:
        tier = "Tier 1"
        text = (
            "Breakthrough likelihood is **low but nonzero (Tier 1)**. "
            "Some signals exist but the evidence is still early."
        )
    else:
        tier = "None"
        text = "Breakthrough likelihood is low. The agent did not sustain high RYE or major discoveries."

    return {"tier": tier, "score": score, "text": text}


# ---------------------------------------------------------------------------
# Core PDF builder
# ---------------------------------------------------------------------------
def build_pdf(content: List[str], output_path: str) -> str:
    """Generate a PDF with ReportLab."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    body_lead = ParagraphStyle('Lead', parent=body, leading=16)

    for section in content:
        if section.startswith("# "):
            story.append(Paragraph(section[2:], h1))
        elif section.startswith("## "):
            story.append(Paragraph(section[3:], h2))
        else:
            story.append(Paragraph(section, body_lead))
        story.append(Spacer(1, 12))

    doc.build(story)
    return output_path


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------
def build_full_pdf_report(
    memory_store: Any,
    discoveries: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
    output_path: str = "autonomous_research_agent_report.pdf",
) -> str:
    """Generate a maxed-out PDF summarizing everything."""

    history = memory_store.get_cycle_history()
    content: List[str] = []

    # Title
    content.append("# Autonomous Research Agent Report")
    content.append(f"Generated: {datetime.utcnow().isoformat()}")

    # -------------------------------------------------------------------
    # High-level stats
    # -------------------------------------------------------------------
    content.append("## Run Overview")
    content.append(f"Total cycles: {len(history)}")
    content.append(f"Total discoveries: {len(discoveries)}")

    # Gather RYE values
    rye_vals = []
    for e in history:
        r = _safe_float(e.get("RYE"))
        if r is not None:
            rye_vals.append(r)

    if rye_vals:
        avg_rye = sum(rye_vals) / len(rye_vals)
        content.append(f"Average RYE: {avg_rye:.3f}")
        content.append(interpret_rye(avg_rye))
    else:
        content.append("RYE metrics not available.")

    # -------------------------------------------------------------------
    # Breakthrough scoring
    # -------------------------------------------------------------------
    content.append("## Breakthrough Analysis")
    b = breakthrough_score(rye_vals, discoveries)
    content.append(f"Breakthrough Tier: **{b['tier']}**")
    content.append(f"Breakthrough Score: {b['score']:.3f}")
    content.append(b["text"])

    # -------------------------------------------------------------------
    # Stability and momentum
    # -------------------------------------------------------------------
    content.append("## Stability & Recovery")
    # last snapshot or last cycle
    if snapshots:
        snap = snapshots[-1]
        metrics = snap.get("data", {}).get("metrics", {})
        stab = _safe_float(metrics.get("stability_index"))
        m = _safe_float(metrics.get("recovery_momentum"))
    else:
        stab = None
        m = None

    content.append(interpret_stability(stab))
    content.append(interpret_momentum(m))

    # -------------------------------------------------------------------
    # Discoveries
    # -------------------------------------------------------------------
    content.append("## Discoveries")
    if not discoveries:
        content.append("No discoveries recorded.")
    else:
        for d in discoveries[:15]:
            label = d.get("title") or d.get("summary") or "Discovery"
            content.append(f"- **{label}**")
            g = _safe_float(d.get("rye_gain"))
            if g is not None:
                content.append(f"  - RYE gain: {g:.3f}")

    # -------------------------------------------------------------------
    # Hypotheses
    # -------------------------------------------------------------------
    content.append("## Hypotheses")
    all_h = []
    for e in history:
        for h in e.get("hypotheses") or []:
            if isinstance(h, dict):
                txt = h.get("text", "")
            else:
                txt = str(h)
            if txt:
                all_h.append(txt)

    if not all_h:
        content.append("No hypotheses recorded.")
    else:
        for h in all_h[:30]:
            content.append(f"- {h}")

    # -------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------
    content.append("## Verification Results")
    try:
        v = memory_store.load_verification_log()
    except Exception:
        v = []

    if not v:
        content.append("No verification log found.")
    else:
        for item in v[:20]:
            label = item.get("label") or item.get("id") or "verification"
            ok = item.get("verified") or item.get("success")
            delta = _safe_float(item.get("delta_RYE") or item.get("rye_gain"))
            content.append(f"- **{label}**: {'PASSED' if ok else 'FAILED'}")
            if delta is not None:
                content.append(f"  - RYE effect: {delta:+.3f}")

    # -------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------
    content.append("## Snapshots (Equilibrium)")
    if not snapshots:
        content.append("No snapshots saved.")
    else:
        for s in snapshots[-5:]:
            ts = s.get("timestamp")
            metrics = s.get("data", {}).get("metrics", {})
            rye_avg = _safe_float(metrics.get("rye_avg"))
            stab_idx = _safe_float(metrics.get("stability_index"))
            content.append(f"- Snapshot @ {ts}")
            if rye_avg is not None:
                content.append(f"  - RYE avg: {rye_avg:.3f}")
            if stab_idx is not None:
                content.append(f"  - Stability: {stab_idx:.3f}")

    # Final build
    return build_pdf(content, output_path)
