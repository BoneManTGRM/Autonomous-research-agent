# pdf_report.py
"""
PDF report exporter for the Autonomous Research Agent.

This module takes a markdown-style report (for example from
agent_report_builder.build_agent_report) and turns it into a
simple, readable PDF.

Design goals:
- Safe: if reportlab is not installed, it falls back to writing a .md file.
- Simple: one main function generate_pdf_report(...) that returns a Path.
- Friendly: includes an optional "Plain English Summary" section so a
  non technical reader can understand what the metrics mean.
- Transparent: can optionally append a "Sources and Citations" section
  so a reader can see where information was obtained.

To fully enable PDF export, add this to requirements.txt:

    reportlab

Then install:

    pip install reportlab
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# Try to import reportlab. If not available, we degrade gracefully.
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_reports_dir(base: str = "reports") -> Path:
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_output_path(reports_dir: Path, stem: str = "agent_report") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return reports_dir / f"{stem}_{ts}.pdf"


def _split_markdown_into_sections(markdown_text: str) -> Tuple[str, str]:
    """
    Split markdown into:
      - a short header/intro
      - the rest of the content

    This gives us something to treat as the "technical" part, while
    we optionally prepend a Plain English Summary.
    """
    if not markdown_text:
        return "", ""

    lines = markdown_text.splitlines()
    if not lines:
        return "", ""

    header_lines: List[str] = []
    body_lines: List[str] = []
    passed_header = False

    for line in lines:
        if not passed_header:
            header_lines.append(line)
            if line.strip() == "---":
                passed_header = True
        else:
            body_lines.append(line)

    header = "\n".join(header_lines).strip()
    body = "\n".join(body_lines).strip()

    return header, body


def _build_plain_english_summary(
    *,
    goal: Optional[str],
    domain: Optional[str],
    diagnostics: Optional[Dict[str, Any]] = None,
    option_c_signature: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a short, human friendly explanation of what the agent did and
    what the metrics roughly mean.

    This does not try to be numerically exact. It is a clear
    interpretation layer for humans.
    """
    lines: List[str] = []

    lines.append("Plain English Summary")
    lines.append("")
    lines.append(
        "This report describes what your autonomous research agent has been doing "
        "over time and how well it has been repairing and improving its own knowledge."
    )

    if goal:
        lines.append("")
        lines.append(f"Main goal: {goal}")

    if domain:
        lines.append(f"Domain focus: {domain}")

    # Basic diagnostics interpretation
    if diagnostics:
        count = diagnostics.get("count")
        avg = diagnostics.get("rye_avg")
        stab = diagnostics.get("stability_index")
        trend = diagnostics.get("trend_slope")
        high_p = diagnostics.get("high_percentile")

        if isinstance(count, int):
            lines.append("")
            lines.append(
                f"The agent has completed about {count} tracked cycles of Test, Detect, Repair, and Verify."
            )

        if isinstance(avg, (int, float)):
            if avg > 0.4:
                lines.append(
                    "On average, the agent is making strong progress per unit effort (high RYE)."
                )
            elif avg > 0.1:
                lines.append(
                    "On average, the agent is making steady but moderate progress per unit effort."
                )
            elif avg > 0:
                lines.append(
                    "The agent is making slight improvements per unit effort, but the signal is weak."
                )
            else:
                lines.append(
                    "Average RYE is near zero or negative, which suggests the agent is not gaining much net ground."
                )

        if isinstance(stab, (int, float)):
            if stab >= 0.7:
                lines.append(
                    "Stability is high: the agent's performance is consistent and does not swing wildly."
                )
            elif stab >= 0.4:
                lines.append(
                    "Stability is moderate: the agent sometimes swings, but overall stays under control."
                )
            else:
                lines.append(
                    "Stability is low: the efficiency of the runs is noisy or chaotic and may need tuning."
                )

        if isinstance(trend, (int, float)):
            if trend > 0.0:
                lines.append(
                    "The long term trend is upward, which means the agent is gradually getting more efficient."
                )
            elif trend < 0.0:
                lines.append(
                    "The long term trend is downward, so efficiency is slowly decaying and may require intervention."
                )
            else:
                lines.append(
                    "The long term trend is roughly flat, so efficiency is not clearly improving or worsening."
                )

        if isinstance(high_p, (int, float)):
            if high_p > 2.0:
                lines.append(
                    "At its best, the agent can achieve very high RYE bursts, which are candidates for breakthrough work."
                )

    # Option C tier and safety envelope
    if option_c_signature:
        run_tier = (option_c_signature.get("run_tier") or {}).get("tier")
        env = option_c_signature.get("autonomy_safety_envelope") or {}
        env_state = env.get("state")

        if run_tier:
            lines.append("")
            lines.append(f"Run tier: {run_tier}.")
            if run_tier == "Tier 0":
                lines.append(
                    "Tier 0 means the run is unstable or too early to call. "
                    "It behaves like a normal tool, not a self repairing system yet."
                )
            elif run_tier == "Tier 1":
                lines.append(
                    "Tier 1 means the agent is working and making positive progress, "
                    "but it is still in a basic, early stage."
                )
            elif run_tier == "Tier 2":
                lines.append(
                    "Tier 2 means the agent behaves like a genuine self repairing system over long runs. "
                    "It is consistently improving its own knowledge."
                )
            elif run_tier == "Tier 3":
                lines.append(
                    "Tier 3 means the run is in a major breakthrough zone: high stability, "
                    "positive trend, and strong repair yield."
                )

        if env_state:
            lines.append("")
            lines.append(f"Autonomy safety envelope: {env_state}.")
            if env_state == "stable":
                lines.append(
                    "Stable means the system is running in a safe, predictable band with no signs of collapse "
                    "or runaway behavior."
                )
            elif env_state == "healthy_growth":
                lines.append(
                    "Healthy growth means the system is both stable and trending upward, "
                    "which is the ideal regime for long experiments."
                )
            elif env_state == "oscillatory":
                lines.append(
                    "Oscillatory means the system swings between good and bad patches. "
                    "It may still be useful, but needs monitoring."
                )
            elif env_state == "collapsing":
                lines.append(
                    "Collapsing means performance is trending downward in an unstable way. "
                    "This usually needs human intervention."
                )
            elif env_state == "explosive":
                lines.append(
                    "Explosive means the spread of RYE values is very high. "
                    "This can be a sign of risky or uncontrolled behavior."
                )
            else:
                lines.append(
                    "Neutral or unknown means the signals are mixed or not strong enough to classify clearly."
                )

    lines.append("")
    lines.append(
        "In short: this summary tells you whether the agent is mostly stable, mostly improving, "
        "and whether it ever reaches strong bursts of efficiency that could hide real discoveries."
    )

    return "\n".join(lines)


def _wrap_text_for_pdf(text: str, max_width_chars: int = 90) -> str:
    """
    Wrap text to roughly fit a PDF page width. The PDF canvas uses a
    fixed width font style for simplicity, so this approximation is
    enough for readable output.
    """
    wrapped_lines: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=max_width_chars))
    return "\n".join(wrapped_lines)


def _draw_text_block(
    c: "canvas.Canvas",
    text: str,
    *,
    x_margin: float = 0.75 * inch,
    y_margin: float = 0.75 * inch,
    line_height: float = 12.0,
) -> None:
    """
    Draw a block of text on multiple pages if needed.
    """
    width, height = letter
    max_y = height - y_margin
    min_y = y_margin

    c.setFont("Helvetica", 10)

    y = max_y
    for line in text.splitlines():
        if y <= min_y:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = max_y
        c.drawString(x_margin, y, line)
        y -= line_height


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def generate_pdf_report(
    *,
    markdown_report: str,
    output_path: Optional[str] = None,
    goal: Optional[str] = None,
    domain: Optional[str] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    option_c_signature: Optional[Dict[str, Any]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Generate a PDF report from a markdown style report plus optional
    diagnostics for a plain English summary.

    Args:
        markdown_report:
            Full markdown report string (for example from build_agent_report).
        output_path:
            Optional path for the resulting PDF. If not provided, a new
            file is created in reports/ with a timestamped name.
        goal:
            Optional main goal used in the summary.
        domain:
            Optional domain label (general, longevity, math, etc).
        diagnostics:
            Optional diagnostics dict from rye_metrics.build_run_diagnostics.
        option_c_signature:
            Optional bundle from rye_metrics.build_option_c_signature.
        citations:
            Optional list of citation dicts with at least title and url fields.
            If provided, a "Sources and Citations" section is appended to
            the report so humans can see where information came from.

    Returns:
        Path to the generated PDF file. If reportlab is not installed,
        the function writes a .md file instead and returns its path.
    """
    reports_dir = _ensure_reports_dir()

    # If no explicit output path is given, create a default one.
    if output_path is not None:
        out_path = Path(output_path)
        if out_path.suffix.lower() != ".pdf":
            out_path = out_path.with_suffix(".pdf")
    else:
        out_path = _default_output_path(reports_dir)

    # If reportlab is not available, fall back to markdown file export.
    if not _REPORTLAB_AVAILABLE:
        fallback = out_path.with_suffix(".md")
        # If citations are provided, append them in markdown so we still
        # preserve transparency in fallback mode.
        combined_markdown = markdown_report
        if citations:
            combined_markdown += "\n\n---\n\n## Sources and Citations\n\n"
            for idx, c in enumerate(citations, start=1):
                title = str(c.get("title") or "").strip()
                url = str(c.get("url") or "").strip()
                src = str(c.get("source") or "").strip()
                line = f"{idx}. {title}"
                if src:
                    line += f" [{src}]"
                if url:
                    line += f" - {url}"
                combined_markdown += line + "\n"
        fallback.write_text(combined_markdown, encoding="utf-8")
        return fallback

    # Build a plain english summary section
    plain_summary = _build_plain_english_summary(
        goal=goal,
        domain=domain,
        diagnostics=diagnostics,
        option_c_signature=option_c_signature,
    )

    header, body = _split_markdown_into_sections(markdown_report)

    combined_text_parts: List[str] = []

    # Top level header
    combined_text_parts.append("Autonomous Research Agent - PDF Report")
    combined_text_parts.append(f"Generated at: {_iso_now()}")
    if goal:
        combined_text_parts.append(f"Goal: {goal}")
    if domain:
        combined_text_parts.append(f"Domain: {domain}")
    combined_text_parts.append("")
    combined_text_parts.append("=" * 70)
    combined_text_parts.append("")

    # Plain English summary first
    combined_text_parts.append(plain_summary)
    combined_text_parts.append("")
    combined_text_parts.append("=" * 70)
    combined_text_parts.append("")

    # Then original header and body from the markdown report
    if header:
        combined_text_parts.append("Original Report Header")
        combined_text_parts.append("")
        combined_text_parts.append(header)
        combined_text_parts.append("")
        combined_text_parts.append("=" * 70)
        combined_text_parts.append("")

    if body:
        combined_text_parts.append("Full Technical Report")
        combined_text_parts.append("")
        combined_text_parts.append(body)
        combined_text_parts.append("")
        combined_text_parts.append("=" * 70)
        combined_text_parts.append("")

    # Optional explicit citation section so the PDF always tells
    # where the information was obtained.
    if citations:
        combined_text_parts.append("Sources and Citations")
        combined_text_parts.append("")
        for idx, c in enumerate(citations, start=1):
            title = str(c.get("title") or "").strip()
            url = str(c.get("url") or "").strip()
            src = str(c.get("source") or "").strip()
            row = f"{idx}. {title}" if title else f"{idx}. (untitled)"
            if src:
                row += f" [{src}]"
            if url:
                row += f" - {url}"
            combined_text_parts.append(row)

    full_text = "\n".join(combined_text_parts)
    wrapped_text = _wrap_text_for_pdf(full_text)

    # Write PDF
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=letter)
    _draw_text_block(c, wrapped_text)
    c.showPage()
    c.save()

    return out_path
