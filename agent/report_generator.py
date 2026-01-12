"""Extended Report Generator for the Autonomous Research Agent.

Outputs:
1. Full Reparodynamics Report (rich RYE and swarm analytics, including Option C meta segments)
2. Targeted Findings Report (cures, treatments, mechanisms, interventions)
3. Optional PDF exports for both reports (if reportlab is available)

10x learning mode:
    This module now consumes the full Option C RYE diagnostics bundle
    (run tier, safety envelope, early failure warning, breakthrough
    probability, equilibrium and volatility) so the agent and human
    operator can see how quickly the system is learning and how safe
    the autonomy regime is.

Learning speed diagnostics:
    In addition to Option C signals, this module estimates cycles per
    hour, RYE per hour, and a qualitative learning speed grade that
    combines stability, recovery momentum, and efficiency. This helps
    operators compare different training runs for how fast and how
    safely they learn.

Discovery and equilibrium view:
    This module also aggregates breakthrough signals and equilibrium
    snapshots from individual cycles when available. It produces a
    simple tier histogram for discovery intensity and a compact view
    of equilibrium labels and scores across the run.

Citation coverage:
    The report estimates how many cycles carried citations, how many
    unique sources and papers were used, and how dense the supporting
    evidence is for the run as a whole.

MSIL and meta intelligence:
    When available, this module reads MSIL snapshots and meta controller
    states from the memory store. It exposes msil_score, skill dimensions,
    domain profiles, discovery density, and curriculum hints as an
    additional layer above raw RYE metrics.

Optional fields:
    Wherever possible this module will:
    - fall back between flattened and summary nested cycle structures
    - pick up optional bundles like learning_meta, safety_meta,
      spread_of_learning, and any extra diagnostics values
    - render generic key value views for unknown optional dictionaries
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
    robust_rolling_rye,
    build_run_diagnostics,
    rye_percentiles,
    build_option_c_signature,
)

import textwrap
import json
import os
import re
from pathlib import Path
from collections import Counter

# -----------------------------------------------------------------------
# Vague and placeholder detection helpers
#
# HEDGING_PATTERNS captures hedging words/phrases that weaken claims.
HEDGING_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bcan\b",
    r"\bwould\b",
    r"\bshould\b",
    r"\bsuggests?\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bpotential\b",
    r"\bimplications?\b",
    r"\bfurther\b",
    r"\bfuture\b",
    r"\bappears\b",
    r"\bindicates?\b",
    r"\bunclear\b",
    r"\blikely\b",
    r"\bunlikly\b",
]

def _is_vague(text: str) -> bool:
    """Return True if the sentence contains hedging language that weakens the claim."""
    if not text:
        return False
    # Normalize to lowercase and repair any mojibake
    try:
        s = normalize_text(text).lower()
    except Exception:
        s = str(text).lower() if text else ""
    for pat in HEDGING_PATTERNS:
        try:
            if re.search(pat, s):
                return True
        except Exception:
            continue
    return False

# Patterns to detect and exclude internal or template-like discovery entries.
# These match fragments of placeholder entries (maintenance modes, templates, logs).
BANNED_DISCOVERY_PATTERNS = [
    "maintenance_mode",
    "maintenance mode",
    "discovery_log",
    "discovery log",
    "placeholder discovery",
    "template entry",
    "template",
    "example",
    "inconclusive verification",
    "rejected hypothesis",
    "performed targeted research",
    "initial discovery_log.json",
    "initial discovery log",
    "used only to show",
    "shows how an",
]

def _contains_banned_pattern(text: str) -> bool:
    """Return True if the string contains any banned placeholder pattern."""
    if not text:
        return False
    try:
        lower = normalize_text(text).lower()
    except Exception:
        lower = str(text).lower() if text else ""
    for pat in BANNED_DISCOVERY_PATTERNS:
        if pat in lower:
            return True
    return False


def normalize_text(text: Any) -> str:
    """Best-effort fix for common mojibake sequences and numeric range dashes.

    This function cleans up mojibake artifacts that commonly appear when UTFâ8
    text has been decoded using a single-byte encoding such as Windows-1252.
    It also ensures that any dash or minus sign occurring between two digits
    (for example, in numeric ranges like "15â20" or misdecoded forms like
    "15Ã¢ï¿½ï¿½20") is replaced with a plain ASCII hyphen ('-').  This range
    normalization happens even if no mojibake markers are detected, so that
    genuine en dashes in ranges don't leak into downstream outputs.
    """
    if text is None:
        return ""
    # Coerce non-strings to strings
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    # Always normalize dash/minus variants between digits.  By performing
    # this step before checking for mojibake markers, we ensure ranges like
    # "15â20" are canonicalized to "15-20" even when no mojibake markers
    # are present.  We also handle common misdecoded sequences.
    try:
        for seq in ("Ã¢ï¿½ï¿½", "Ã¢??", "Ã¢â¬â", "Ã¢\u0080\u0093"):
            text = re.sub(rf"(?<=\d){re.escape(seq)}(?=\d)", "-", text)
        text = re.sub(r"(?<=\d)\s*[âââââï¹ï¹£ï¼]\s*(?=\d)", "-", text)
    except Exception:
        pass
    # Only attempt mojibake repair when common markers are present
    if not any(tok in text for tok in ("Ã", "Ã¢", "Ã")):
        return text
    # Count markers helper
    def count_markers(val: str) -> int:
        return sum(val.count(ch) for ch in ("Ã", "Ã¢", "Ã"))
    before = count_markers(text)
    for enc in ("cp1252", "latin1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except Exception:
            continue
        if fixed and count_markers(fixed) < before:
            # Normalize numeric ranges again on the fixed string
            try:
                for seq in ("Ã¢ï¿½ï¿½", "Ã¢??", "Ã¢â¬â", "Ã¢\u0080\u0093"):
                    fixed = re.sub(rf"(?<=\d){re.escape(seq)}(?=\d)", "-", fixed)
                fixed = re.sub(r"(?<=\d)\s*[âââââï¹ï¹£ï¼]\s*(?=\d)", "-", fixed)
            except Exception:
                pass
            return fixed
    # Fallback: convert misdecoded bullet if present
    if "Ã¢â¬Â¢" in text:
        text = text.replace("Ã¢â¬Â¢", "â¢")
    # Final pass: ensure any remaining dash variants between digits are normalized
    try:
        for seq in ("Ã¢ï¿½ï¿½", "Ã¢??", "Ã¢â¬â", "Ã¢\u0080\u0093"):
            text = re.sub(rf"(?<=\d){re.escape(seq)}(?=\d)", "-", text)
        text = re.sub(r"(?<=\d)\s*[âââââï¹ï¹£ï¼]\s*(?=\d)", "-", text)
    except Exception:
        pass
    return text


# ---------------------------------------------------------
# Event stream helpers (per-run events.jsonl)
# ---------------------------------------------------------
def _resolve_runs_root() -> Optional[Path]:
    """Resolve the runs root directory where per-run folders live.

    This is intentionally best-effort and mirrors the rest of the codebase:
    - Prefer ARA_RUNS_DIR when set.
    - Fall back to a local ./runs folder if nothing is configured.
    """
    env_val = os.getenv("ARA_RUNS_DIR")
    if isinstance(env_val, str) and env_val.strip():
        try:
            return Path(env_val.strip()).expanduser().resolve()
        except Exception:
            pass
    try:
        return (Path.cwd() / "runs").resolve()
    except Exception:
        return None


def _infer_run_id(
    memory_store: Any,
    cycles: List[Dict[str, Any]],
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Infer the active run_id from explicit arg, memory_store, or cycle history."""
    if explicit:
        try:
            return str(explicit)
        except Exception:
            return explicit  # type: ignore[return-value]

    # Common attributes / helpers
    for attr in ("run_id", "active_run_id", "current_run_id"):
        try:
            v = getattr(memory_store, attr, None)
            if v:
                return str(v)
        except Exception:
            continue

    try:
        getter = getattr(memory_store, "get_run_id", None)
        if callable(getter):
            v = getter()
            if v:
                return str(v)
    except Exception:
        pass

    # Mode of run_id values present in cycle history
    try:
        ids = [str(c.get("run_id")) for c in cycles if c.get("run_id") is not None]
        if ids:
            return Counter(ids).most_common(1)[0][0]
    except Exception:
        pass

    return None


def _read_jsonl_events(path: Path, max_lines: int = 20000) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dict events (tail-limited)."""
    out: List[Dict[str, Any]] = []
    try:
        if not path.exists():
            return out
        data = path.read_bytes()
        lines = data.splitlines()
        if max_lines and len(lines) > max_lines:
            lines = lines[-max_lines:]
        for raw in lines:
            try:
                s = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    except Exception:
        return out
    return out


def _load_run_events(run_id: str, runs_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load per-run events.jsonl for a run_id."""
    if not run_id:
        return []
    root = runs_root or _resolve_runs_root()
    if root is None:
        return []
    # Primary per-run path
    p = root / str(run_id) / "events.jsonl"
    events = _read_jsonl_events(p)
    if events:
        return events

    # Optional global mirror (if present)
    alt = root / "logs" / "events_global.jsonl"
    evs = _read_jsonl_events(alt)
    if not evs:
        return []

    # Filter global mirror to run_id
    filtered: List[Dict[str, Any]] = []
    for e in evs:
        try:
            e_run = e.get("run_id")
            if e_run is not None and str(e_run) == str(run_id):
                filtered.append(e)
        except Exception:
            continue
    return filtered


def _events_to_discovery_log(events: List[Dict[str, Any]], goal: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert events.jsonl entries into the discovery-log shape this module expects."""
    out: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        kind = str(e.get("kind") or "").strip()
        if kind not in ("candidate_hypothesis", "discovery", "discovery_candidate"):
            continue

        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        # Optional goal filter if goal exists in event payload
        if goal:
            ev_goal = e.get("goal") or data.get("goal")
            if isinstance(ev_goal, str) and ev_goal and ev_goal != goal:
                continue

        label = (
            data.get("title")
            or data.get("label")
            or data.get("thesis")
            or data.get("headline")
            or e.get("message")
            or kind
        )
        ts = e.get("timestamp") or e.get("ts") or ""
        score = data.get("score") or e.get("score")
        tier = data.get("tier") or data.get("discovery_tier")
        out.append(
            {
                "timestamp": ts,
                "kind": kind,
                "label": str(label) if label is not None else kind,
                "score": score,
                "tier": tier,
                "raw_event": e,
            }
        )
    return out


def _events_to_structured_discoveries(events: List[Dict[str, Any]], goal: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert events to a structured discoveries list (closer to get_discoveries())."""
    structured: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        kind = str(e.get("kind") or "").strip()
        if kind not in ("candidate_hypothesis", "discovery", "discovery_candidate"):
            continue
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        if goal:
            ev_goal = e.get("goal") or data.get("goal")
            if isinstance(ev_goal, str) and ev_goal and ev_goal != goal:
                continue
        structured.append(
            {
                "timestamp": e.get("timestamp") or e.get("ts") or "",
                "kind": kind,
                "label": data.get("title") or data.get("thesis") or data.get("label") or "",
                "score": data.get("score") or e.get("score"),
                "tier": data.get("tier") or data.get("discovery_tier"),
                "data": data,
            }
        )
    return structured


def _render_event_only_report(events: List[Dict[str, Any]], run_id: Optional[str] = None, goal: Optional[str] = None) -> str:
    """Fallback report when cycle history is empty but events exist."""
    disc = _events_to_discovery_log(events, goal=goal)
    lines: List[str] = []
    lines.append("# Autonomous Research Agent Report\n")
    if run_id:
        lines.append(f"**Run ID:** `{run_id}`\n")
    if goal:
        lines.append(f"**Goal:** `{goal}`\n")
    lines.append("## Event-only summary\n")
    if not disc:
        lines.append("No candidate hypotheses or discoveries found in the event stream.\n")
    else:
        lines.append(f"Found **{len(disc)}** candidate/discovery events.\n")
        for d in disc[-20:]:
            ts = d.get("timestamp", "")
            kind = d.get("kind", "")
            label = d.get("label", "")
            lines.append(f"- [{ts}] [{kind}] {label}")
        lines.append("")
    return "\n".join(lines)

def _render_event_only_findings_report(
    events: List[Dict[str, Any]],
    run_id: Optional[str] = None,
    goal: Optional[str] = None,
) -> str:
    """Fallback findings report when cycle history is empty but events exist."""
    md = _render_event_only_report(events, run_id=run_id, goal=goal)
    return md.replace("# Autonomous Research Agent Report", "# Targeted Findings Report", 1)


# Optional PDF support
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - optional dependency
    canvas = None  # type: ignore[assignment]
    letter = None  # type: ignore[assignment]


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


def _compute_session_hours(timestamps: List[str]) -> Optional[float]:
    """Return numeric session length in hours if timestamps exist."""
    if not timestamps:
        return None

    from datetime import datetime

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
        return diff.total_seconds() / 3600.0
    except Exception:
        return None


def _safe_get_cycle_history(memory_store: Any) -> List[Dict[str, Any]]:
    """Defensive wrapper around memory_store.get_cycle_history."""
    try:
        hist = memory_store.get_cycle_history()
        if isinstance(hist, list):
            return hist
    except Exception:
        return []
    return []


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
        # Support both flattened and nested summary structures
        rye_val = c.get("RYE")
        if rye_val is None and isinstance(c.get("summary"), dict):
            rye_val = c["summary"].get("RYE")

        if not isinstance(rye_val, (int, float)):
            continue
        rye_f = float(rye_val)

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


def _primary_domain_from_stats(domain_stats: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Pick a primary domain based on cycle counts."""
    if not domain_stats:
        return None
    best_domain = None
    best_count = -1
    for d, stats in domain_stats.items():
        cnt = int(stats.get("count", 0))
        if cnt > best_count:
            best_count = cnt
            best_domain = d
    return best_domain


def _markdown_to_plain_lines(text: str, width: int = 90) -> List[str]:
    """Very simple markdown to plain converter for PDF text.

    Strips leading markdown markers and wraps to fixed width.
    """
    lines: List[str] = []
    for raw in text.splitlines():
        s = raw.lstrip()
        # strip some simple markdown markers
        # Strip common markdown markers.  The bullet prefix must be the actual
        # Unicode bullet rather than a mojibake sequence (e.g. "Ã¢ÂÂ¢ "), which
        # can appear if the source file was mis-encoded.  Using "â¢ " here
        # ensures that list items are detected and the bullet itself is removed
        # before wrapping.
        for prefix in ("#", "* ", "- ", "â¢ ", "> "):
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip()
                break
        if not s:
            lines.append("")  # preserve blank lines
            continue
        wrapped = textwrap.wrap(s, width=width) or [""]
        lines.extend(wrapped)
    return lines


def _write_pdf(text: str, output_path: str, title: Optional[str] = None) -> str:
    """Write plain text content into a simple PDF file.

    Requires reportlab. If reportlab is not installed, raises a RuntimeError.
    """
    if canvas is None or letter is None:
        raise RuntimeError(
            "PDF generation requires the 'reportlab' package. "
            "Install it with: pip install reportlab"
        )

    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    if title:
        c.setTitle(title)

    text_obj = c.beginText()
    left_margin = 40
    top_margin = height - 50
    bottom_margin = 40
    line_height = 14

    text_obj.setTextOrigin(left_margin, top_margin)
    text_obj.setFont("Helvetica", 10)

    for line in _markdown_to_plain_lines(text, width=100):
        if text_obj.getY() <= bottom_margin:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText()
            text_obj.setTextOrigin(left_margin, top_margin)
            text_obj.setFont("Helvetica", 10)
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()
    return output_path


def _meta_segment_stats(
    cycles: List[Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """Aggregate Option C meta controller segments from per cycle run_metadata."""
    segments: Dict[int, Dict[str, Any]] = {}

    for c in cycles:
        meta = c.get("run_metadata") or {}
        if not isinstance(meta, dict):
            continue

        seg_idx = meta.get("segment_index")
        if not isinstance(seg_idx, int):
            continue

        rye_val = c.get("RYE")
        if rye_val is None and isinstance(c.get("summary"), dict):
            rye_val = c["summary"].get("RYE")

        rye_f: Optional[float] = None
        if isinstance(rye_val, (int, float)):
            rye_f = float(rye_val)

        seg = segments.setdefault(
            seg_idx,
            {
                "phase": meta.get("phase"),
                "mode": meta.get("mode"),
                "runtime_profile": meta.get("runtime_profile"),
                "count": 0,
                "rye_sum": 0.0,
                "min_rye": None,
                "max_rye": None,
            },
        )

        seg["count"] = int(seg.get("count", 0)) + 1
        if rye_f is not None:
            seg["rye_sum"] = float(seg.get("rye_sum", 0.0)) + rye_f
            cur_min = seg.get("min_rye")
            cur_max = seg.get("max_rye")
            if cur_min is None or rye_f < cur_min:
                seg["min_rye"] = rye_f
            if cur_max is None or rye_f > cur_max:
                seg["max_rye"] = rye_f

    for idx, seg in list(segments.items()):
        cnt = int(seg.get("count", 0)) or 0
        if cnt <= 0:
            seg["avg_rye"] = None
        else:
            seg["avg_rye"] = float(seg.get("rye_sum", 0.0)) / float(cnt)
        seg.pop("rye_sum", None)

    return segments


def _map_breakthrough_score_to_tier(score: float) -> str:
    """Map a breakthrough_score in [0, 1] to an internal discovery tier label."""
    if score < 0.3:
        return "none"
    if score < 0.55:
        return "tier_3_hint"
    if score < 0.8:
        return "tier_2_candidate"
    return "tier_1_candidate"


def _breakthrough_and_equilibrium_stats(
    cycles: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate breakthrough and equilibrium signals over all cycles."""
    total_cycles = len(cycles)
    breakthrough_count = 0
    best_breakthrough_score: Optional[float] = None
    best_breakthrough_cycle: Optional[Dict[str, Any]] = None

    tier_counts: Dict[str, int] = {
        "none": 0,
        "tier_3_hint": 0,
        "tier_2_candidate": 0,
        "tier_1_candidate": 0,
    }

    eq_label_counts: Dict[str, int] = {}
    eq_scores: List[float] = []
    osc_scores: List[float] = []

    for c in cycles:
        summary = c.get("summary")
        if not isinstance(summary, dict):
            summary = {}

        breakthrough = summary.get("breakthrough") or c.get("breakthrough") or {}
        if isinstance(breakthrough, dict):
            bs = breakthrough.get("breakthrough_score")
            flag = breakthrough.get("is_breakthrough")
            tier = breakthrough.get("tier") or breakthrough.get("discovery_tier")

            if isinstance(bs, (int, float)):
                bs_float = float(bs)
                if best_breakthrough_score is None or bs_float > best_breakthrough_score:
                    best_breakthrough_score = bs_float
                    best_breakthrough_cycle = c
                if not tier:
                    tier = _map_breakthrough_score_to_tier(bs_float)
                if tier:
                    tier = str(tier)
                    if tier not in tier_counts:
                        tier_counts[tier] = 0
                    tier_counts[tier] += 1
                breakthrough_count += 1
            elif isinstance(flag, bool) and flag:
                breakthrough_count += 1
                if not tier:
                    tier = "tier_3_hint"
                if tier not in tier_counts:
                    tier_counts[tier] = 0
                tier_counts[tier] += 1

        equilibrium = summary.get("equilibrium") or c.get("equilibrium") or {}
        if isinstance(equilibrium, dict):
            label = equilibrium.get("equilibrium_label") or equilibrium.get("label")
            eq_score = equilibrium.get("equilibrium_score")
            osc_score = equilibrium.get("oscillation_score")

            if isinstance(label, str) and label:
                eq_label_counts[label] = eq_label_counts.get(label, 0) + 1
            if isinstance(eq_score, (int, float)):
                eq_scores.append(float(eq_score))
            if isinstance(osc_score, (int, float)):
                osc_scores.append(float(osc_score))

    avg_eq_score = float(sum(eq_scores) / len(eq_scores)) if eq_scores else None
    avg_osc_score = float(sum(osc_scores) / len(osc_scores)) if osc_scores else None

    tier_order = ["none", "tier_3_hint", "tier_2_candidate", "tier_1_candidate"]
    rank = {name: idx for idx, name in enumerate(tier_order)}
    overall_tier = "none"
    for t, cnt in tier_counts.items():
        if cnt <= 0:
            continue
        if rank.get(t, 0) > rank.get(overall_tier, 0):
            overall_tier = t

    return {
        "total_cycles": total_cycles,
        "breakthrough_cycles": breakthrough_count,
        "tier_counts": tier_counts,
        "overall_tier": overall_tier,
        "best_breakthrough_score": best_breakthrough_score,
        "best_breakthrough_cycle": best_breakthrough_cycle,
        "equilibrium_label_counts": eq_label_counts,
        "avg_equilibrium_score": avg_eq_score,
        "avg_oscillation_score": avg_osc_score,
    }


def _citation_stats(
    cycles: List[Dict[str, Any]],
    all_citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Estimate citation coverage and unique sources over the run."""
    total_cycles = len(cycles)
    cycles_with_cites = 0

    for c in cycles:
        cits = c.get("citations") or []
        if isinstance(cits, list) and cits:
            cycles_with_cites += 1
        else:
            summary = c.get("summary")
            if isinstance(summary, dict):
                sc = summary.get("citations") or []
                if isinstance(sc, list) and sc:
                    cycles_with_cites += 1

    total_cites = len(all_citations)
    unique_sources = set()
    unique_papers = set()

    for ct in all_citations:
        src = str(ct.get("source", "unknown") or "unknown")
        title = str(ct.get("title", "Untitled") or "Untitled")
        url = str(ct.get("url", "") or "")
        unique_sources.add(src)
        unique_papers.add((title, url))

    coverage = float(cycles_with_cites) / float(total_cycles) if total_cycles > 0 else 0.0
    avg_per_cycle = float(total_cites) / float(total_cycles) if total_cycles > 0 else 0.0

    return {
        "total_cycles": total_cycles,
        "cycles_with_citations": cycles_with_cites,
        "coverage_fraction": coverage,
        "total_citations": total_cites,
        "avg_citations_per_cycle": avg_per_cycle,
        "unique_source_count": len(unique_sources),
        "unique_paper_count": len(unique_papers),
    }


def _learning_speed_grade(
    cycles_per_hour: Optional[float],
    rye_per_hour: Optional[float],
    stab: Optional[float],
    momentum: Optional[float],
) -> Tuple[str, str]:
    """Qualitative learning speed grade for 10x tuning."""
    cph = float(cycles_per_hour) if isinstance(cycles_per_hour, (int, float)) else 0.0
    rph = float(rye_per_hour) if isinstance(rye_per_hour, (int, float)) else 0.0
    st = float(stab) if isinstance(stab, (int, float)) else None
    mo = float(momentum) if isinstance(momentum, (int, float)) else None

    if cph <= 0.0 or rph <= 0.0:
        return (
            "insufficient data",
            "Too few cycles or unknown runtime length to estimate learning speed.",
        )

    if cph >= 60 and rph >= 2.5:
        base_label = "very fast learning"
    elif cph >= 30 and rph >= 1.0:
        base_label = "fast learning"
    elif cph >= 15 and rph >= 0.4:
        base_label = "moderate learning"
    else:
        base_label = "slow learning"

    if st is not None:
        improving = mo is None or mo >= 0.0
        if improving and st >= 0.65:
            return (
                f"{base_label}, stable",
                "Learning speed is high with a stable repair pattern and nonnegative recovery momentum.",
            )
        if st <= 0.35 and (mo is not None and mo < 0.0):
            return (
                f"{base_label}, unstable",
                "Learning speed is limited by volatility and negative recovery momentum. Consider lowering swarm size or tightening verification.",
            )

    if mo is not None and mo > 0.1 and rph >= 0.4:
        return (
            f"{base_label}, accelerating",
            "Recovery momentum is positive and RYE per hour is improving, suggesting an accelerating learning regime.",
        )

    return (
        base_label,
        "Learning speed is within expected bounds for this configuration. Use RYE trends and stability index for finer tuning.",
    )


def _render_generic_optional_dict(
    d: Dict[str, Any],
    lines: List[str],
    prefix: str = "- ",
    max_items: int = 12,
) -> None:
    """Render a generic optional dict in key colon value form.

    Only prints simple scalar values and short lists or dicts, and
    truncates the list for readability.
    """
    if not isinstance(d, dict):
        return
    count = 0
    for key, val in d.items():
        if count >= max_items:
            lines.append(f"{prefix}... {len(d) - max_items} more keys not shown")
            break
        if isinstance(val, (int, float, str, bool)) or val is None:
            lines.append(f"{prefix}{key}: {val}")
            count += 1
        elif isinstance(val, (list, dict)):
            try:
                short = str(val)
                if len(short) > 120:
                    short = short[:117] + "..."
                lines.append(f"{prefix}{key}: {short}")
                count += 1
            except Exception:
                continue


# ---------------------------------------------------------
# FULL REPORT
# ---------------------------------------------------------
def generate_report(memory_store: Any, goal: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """Generate full Reparodynamics markdown report with advanced metrics."""

    all_cycles: List[Dict[str, Any]] = _safe_get_cycle_history(memory_store)

    if goal:
        cycles = [c for c in all_cycles if (c.get("goal") or "") == goal]
    else:
        cycles = list(all_cycles)

    # Run scoping: prefer explicit run_id, then infer from memory/cycle history.
    inferred_run_id = _infer_run_id(memory_store, cycles, explicit=run_id)

    # If cycles carry run_id fields, filter to the inferred run_id to avoid blending runs.
    if inferred_run_id and any(c.get("run_id") is not None for c in cycles):
        try:
            filtered = [c for c in cycles if str(c.get("run_id")) == str(inferred_run_id)]
            if filtered:
                cycles = filtered
        except Exception:
            pass

    n_cycles = len(cycles)

    if n_cycles == 0:
        # Fall back to the per-run event stream if cycles are not available.
        if inferred_run_id:
            try:
                _events = _load_run_events(str(inferred_run_id))
                if _events:
                    return _render_event_only_report(_events, run_id=str(inferred_run_id), goal=goal)
            except Exception:
                pass
        return "# Autonomous Research Agent Report\n\nNo cycles logged."

    # Load per-run events once (used for discovery/candidate sections and robustness).
    run_events: List[Dict[str, Any]] = []
    if inferred_run_id:
        try:
            run_events = _load_run_events(str(inferred_run_id))
        except Exception:
            run_events = []

    # Optional control and watchdog snapshots
    run_state: Optional[Dict[str, Any]] = None
    worker_state: Optional[Dict[str, Any]] = None
    watchdog_info: Optional[Dict[str, Any]] = None
    msil_snapshot: Optional[Dict[str, Any]] = None
    meta_state: Optional[Dict[str, Any]] = None

    try:
        if hasattr(memory_store, "load_run_state"):
            run_state = memory_store.load_run_state()
    except Exception:
        run_state = None

    try:
        if hasattr(memory_store, "get_worker_state"):
            worker_state = memory_store.get_worker_state()
    except Exception:
        worker_state = None

    try:
        if hasattr(memory_store, "get_watchdog_info"):
            watchdog_info = memory_store.get_watchdog_info()
    except Exception:
        watchdog_info = None

    # MSIL snapshot if available
    try:
        if hasattr(memory_store, "get_msil_snapshot"):
            try:
                msil_snapshot = memory_store.get_msil_snapshot(goal=goal)
            except TypeError:
                msil_snapshot = memory_store.get_msil_snapshot()
    except Exception:
        msil_snapshot = None

    # Meta controller state if available
    try:
        if hasattr(memory_store, "get_meta_state"):
            meta_state = memory_store.get_meta_state()
    except Exception:
        meta_state = None

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
        rye_val = c.get("RYE")
        delta_val = c.get("delta_R")
        energy_val = c.get("energy_E")

        summary = c.get("summary")
        if isinstance(summary, dict):
            if rye_val is None:
                rye_val = summary.get("RYE")
            if delta_val is None:
                delta_val = summary.get("delta_R")
            if energy_val is None:
                energy_val = summary.get("energy_E")

        rye_values.append(_safe_float(rye_val, 0.0))
        delta_values.append(_safe_float(delta_val, 0.0))
        energy_values.append(_safe_float(energy_val, 0.0))

        domains.add(c.get("domain", "general"))
        goals_seen.add(c.get("goal", ""))

        ts = c.get("timestamp")
        if isinstance(ts, str) and ts:
            timestamps.append(ts)

        hyps = c.get("hypotheses") or []
        cits = c.get("citations") or []

        if isinstance(summary, dict):
            hyps = hyps or summary.get("hypotheses") or []
            cits = cits or summary.get("citations") or []

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

    # Domain and role stats
    domain_stats, role_stats = _domain_and_role_stats(cycles)
    primary_domain = _primary_domain_from_stats(domain_stats)

    # Diagnostics bundle from Option C
    hours_run = _compute_session_hours(timestamps)

    try:
        option_c_sig_raw = build_option_c_signature(
            cycles,
            domain=primary_domain,
            hours_run_so_far=hours_run,
        )
    except Exception:
        option_c_sig_raw = {}

    option_c_sig = option_c_sig_raw if isinstance(option_c_sig_raw, dict) else {}
    diagnostics = option_c_sig.get("diagnostics", {}) or {}

    roll = diagnostics.get("rolling_rye")
    robust_roll = diagnostics.get("robust_rolling_rye")
    trend = diagnostics.get("trend_simple")
    med_rye = diagnostics.get("rye_median")
    stab = diagnostics.get("stability_index")
    momentum = diagnostics.get("recovery_momentum")
    slope = diagnostics.get("trend_slope")
    low_p = diagnostics.get("low_percentile")
    mid_p = diagnostics.get("mid_percentile")
    high_p = diagnostics.get("high_percentile")
    osc_std = diagnostics.get("oscillation_std")

    runtime = _extract_session_runtime(timestamps)

    # Optional goal index and discoveries
    goal_index_entry: Optional[Dict[str, Any]] = None
    try:
        goal_index_entry = (
            memory_store.get_goal_index(goal) if hasattr(memory_store, "get_goal_index") else None
        )
    except Exception:
        goal_index_entry = None

    discoveries: List[Dict[str, Any]] = []
    try:
        discoveries = (
            memory_store.get_discoveries(goal=goal)
            if hasattr(memory_store, "get_discoveries")
            else []
        )
    except Exception:
        discoveries = []

    # If agent memory has no structured discoveries yet, fall back to the event stream.
    if not discoveries and run_events:
        try:
            discoveries = _events_to_discovery_log(run_events, goal=goal) or discoveries
        except Exception:
            pass

    # Option C meta segment stats
    meta_segments = _meta_segment_stats(cycles)

    # Breakthrough and equilibrium stats across cycles
    be_stats = _breakthrough_and_equilibrium_stats(cycles)

    # Citation coverage stats
    cite_stats = _citation_stats(cycles, all_citations)

    # Learning speed metrics
    cycles_per_hour: Optional[float] = None
    rye_per_hour: Optional[float] = None
    if hours_run is not None and hours_run > 0.0:
        cycles_per_hour = float(n_cycles) / float(hours_run)
        rye_per_hour = avg_rye * cycles_per_hour

    # Option C top level bundles
    run_tier = option_c_sig.get("run_tier") or {}
    env = option_c_sig.get("autonomy_safety_envelope") or {}
    early_fail = option_c_sig.get("early_failure_warning") or {}
    bp = option_c_sig.get("breakthrough_probability") or {}
    bp90 = option_c_sig.get("breakthrough_likelihood_90d") or {}
    equilibrium = option_c_sig.get("equilibrium") or {}
    volatility = option_c_sig.get("volatility") or {}
    harmonic = option_c_sig.get("tgrm_harmonic_index")
    learning_meta = option_c_sig.get("learning_meta") or {}
    spread_meta = option_c_sig.get("spread_of_learning") or {}
    safety_meta = option_c_sig.get("safety_meta") or {}
    curriculum_meta = option_c_sig.get("curriculum_meta") or {}
    msil_meta = option_c_sig.get("msil_meta") or {}

    # Compute list of extra diagnostic keys not already rendered
    known_diag_keys = {
        "rolling_rye",
        "robust_rolling_rye",
        "trend_simple",
        "rye_median",
        "stability_index",
        "recovery_momentum",
        "trend_slope",
        "low_percentile",
        "mid_percentile",
        "high_percentile",
        "oscillation_std",
    }
    extra_diag = {
        k: v
        for k, v in diagnostics.items()
        if k not in known_diag_keys and isinstance(v, (int, float))
    }

    # Build report
    lines: List[str] = []
    lines.append("# Autonomous Research Agent Report\n")

    # Runtime
    if runtime:
        lines.append(f"**Session runtime:** {runtime}\n")
    if inferred_run_id:
        lines.append(f"**Run ID:** `{inferred_run_id}`\n")

    # Run control snapshot
    if run_state or worker_state or watchdog_info:
        lines.append("## Run control snapshot\n")
        if run_state:
            lines.append("**Last saved run state:**")
            rs_goal = run_state.get("goal")
            rs_mode = run_state.get("mode")
            rs_domain = run_state.get("domain")
            rs_role = run_state.get("role")
            rs_min_left = run_state.get("minutes_remaining")
            rs_last_cycle = run_state.get("last_cycle_index")
            rs_updated = run_state.get("updated_at")

            if rs_goal:
                lines.append(f"- Goal: `{rs_goal}`")
            if rs_domain:
                lines.append(f"- Domain: `{rs_domain}`")
            if rs_role:
                lines.append(f"- Primary role: `{rs_role}`")
            if rs_mode:
                lines.append(f"- Mode: `{rs_mode}`")
            if rs_min_left is not None:
                try:
                    lines.append(f"- Approx minutes remaining at save: **{float(rs_min_left):.1f}**")
                except Exception:
                    lines.append(f"- Approx minutes remaining at save: **{rs_min_left}**")
            if rs_last_cycle is not None:
                lines.append(f"- Last completed cycle index: **{rs_last_cycle}**")
            if rs_updated:
                lines.append(f"- Run state updated at: `{rs_updated}`")
            lines.append("")

        if worker_state:
            lines.append("**Live worker status (last snapshot):**")
            ws_status = worker_state.get("status")
            ws_mode = worker_state.get("mode")
            ws_goal = worker_state.get("goal")
            ws_domain = worker_state.get("domain")
            ws_roles = worker_state.get("roles") or []
            ws_profile = worker_state.get("runtime_profile")
            ws_stop_rye = worker_state.get("stop_rye")
            ws_max_minutes = worker_state.get("max_minutes")
            ws_updated = worker_state.get("updated_at")
            ws_learning_mode = worker_state.get("learning_mode")
            ws_swarm_size = worker_state.get("swarm_size")

            if ws_status:
                lines.append(f"- Status: **{ws_status}**")
            if ws_mode:
                lines.append(f"- Mode: `{ws_mode}`")
            if ws_goal:
                lines.append(f"- Goal: `{ws_goal}`")
            if ws_domain:
                lines.append(f"- Domain: `{ws_domain}`")
            if ws_roles:
                lines.append(f"- Roles: {', '.join(str(r) for r in ws_roles)}")
            if ws_profile:
                lines.append(f"- Runtime profile: `{ws_profile}`")
            if isinstance(ws_stop_rye, (int, float)):
                lines.append(f"- Stop RYE threshold: **{float(ws_stop_rye):.3f}**")
            if ws_max_minutes is not None:
                try:
                    lines.append(f"- Max run minutes: **{float(ws_max_minutes):.1f}**")
                except Exception:
                    lines.append(f"- Max run minutes: **{ws_max_minutes}**")
            if ws_learning_mode:
                lines.append(f"- Learning mode: `{ws_learning_mode}`")
            if isinstance(ws_swarm_size, (int, float)):
                lines.append(f"- Swarm agents active: **{int(ws_swarm_size)}**")
            if ws_updated:
                lines.append(f"- Worker state updated at: `{ws_updated}`")
            lines.append("")

        if watchdog_info:
            lines.append("**Watchdog heartbeat:**")
            wd_last = watchdog_info.get("last_beat")
            wd_count = watchdog_info.get("count")
            wd_seconds = watchdog_info.get("seconds_since_last")
            if wd_last:
                lines.append(f"- Last heartbeat: `{wd_last}`")
            if wd_count is not None:
                lines.append(f"- Total heartbeats recorded: **{int(wd_count)}**")
            if isinstance(wd_seconds, (int, float)):
                minutes_since = wd_seconds / 60.0
                lines.append(f"- Time since last heartbeat: **{minutes_since:.2f} minutes**")
            lines.append("")

    # MSIL snapshot section
    if isinstance(msil_snapshot, dict) and msil_snapshot:
        lines.append("## MSIL snapshot\n")
        msil_score = msil_snapshot.get("msil_score")
        if isinstance(msil_score, (int, float)):
            lines.append(
                f"- MSIL score: **{float(msil_score):.3f}** (0 to 1 meta stability and intelligence level)"
            )
        skill_dims = msil_snapshot.get("skill_dimensions") or {}
        if isinstance(skill_dims, dict) and skill_dims:
            lines.append("- Skill dimensions:")
            for name, val in skill_dims.items():
                if isinstance(val, (int, float)):
                    lines.append(f"  - {name}: **{float(val):.3f}**")
                else:
                    lines.append(f"  - {name}: {val}")
        dom_prof = msil_snapshot.get("domain_profiles") or {}
        if isinstance(dom_prof, dict) and dom_prof:
            lines.append("- Domain profiles:")
            _render_generic_optional_dict(dom_prof, lines, prefix="  - ", max_items=8)
        msil_break = msil_snapshot.get("breakthrough_density")
        if isinstance(msil_break, (int, float)):
            lines.append(
                f"- Breakthrough density estimate: **{float(msil_break):.3f}**"
            )
        msil_text = msil_snapshot.get("comment")
        if isinstance(msil_text, str) and msil_text.strip():
            lines.append(f"- MSIL comment: {msil_text}")
        lines.append("")

    # Meta controller state section
    if isinstance(meta_state, dict) and meta_state:
        lines.append("## Meta controller state\n")
        meta_mode = meta_state.get("mode")
        if meta_mode:
            lines.append(f"- Meta mode: `{meta_mode}`")
        active_profile = meta_state.get("active_profile")
        if active_profile:
            lines.append(f"- Active training profile: `{active_profile}`")
        target_goal = meta_state.get("target_goal")
        if target_goal:
            lines.append(f"- Target meta goal: `{target_goal}`")
        meta_note = meta_state.get("note")
        if isinstance(meta_note, str) and meta_note.strip():
            lines.append(f"- Meta note: {meta_note}")
        extra_meta = {
            k: v
            for k, v in meta_state.items()
            if k not in {"mode", "active_profile", "target_goal", "note"}
        }
        if extra_meta:
            _render_generic_optional_dict(extra_meta, lines, prefix="- ", max_items=10)
        lines.append("")

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
    lines.append(f"- Avg ÃÂR: **{avg_delta:.3f}**")
    lines.append(f"- Avg Energy: **{avg_energy:.3f}**")

    if cycles_per_hour is not None:
        lines.append(f"- Cycles per hour (approx): **{cycles_per_hour:.2f}**")
    if rye_per_hour is not None:
        lines.append(f"- RYE per hour (approx): **{rye_per_hour:.3f}**")

    if roll is not None:
        lines.append(f"- Rolling RYE (last 10): **{roll:.3f}**")
    if robust_roll is not None:
        lines.append(
            f"- Robust rolling RYE (median, last 10): **{robust_roll:.3f}**"
        )

    if med_rye is not None:
        lines.append(f"- Median RYE (noise resistant): **{med_rye:.3f}**")

    if trend is not None:
        direction = "improving" if trend > 0 else "declining" if trend < 0 else "flat"
        lines.append(
            f"- RYE trend (first half vs second half): **{trend:.3f}** ({direction})"
        )

    if slope is not None:
        lines.append(
            f"- Regression slope of RYE over cycles: **{slope:.5f}**"
        )

    if stab is not None:
        lines.append(
            f"- Stability index: **{stab:.3f}** (1.0 highly stable, 0.0 chaotic)"
        )

    if momentum is not None:
        lines.append(
            f"- Recovery momentum: **{momentum:.3f}** (higher means late stage acceleration)"
        )

    if osc_std is not None:
        lines.append(
            f"- RYE oscillation standard deviation: **{osc_std:.3f}**"
        )

    if low_p is not None and mid_p is not None and high_p is not None:
        lines.append(
            f"- RYE distribution: low ~**{low_p:.3f}**, median ~**{mid_p:.3f}**, high ~**{high_p:.3f}**"
        )

    if extra_diag:
        lines.append("- Extra diagnostics:")
        for k, v in list(extra_diag.items())[:10]:
            lines.append(f"  - {k}: **{float(v):.4f}**")

    phase_label = _classify_phase(slope, trend, stab)
    lines.append(f"- Phase classification: **{phase_label}**")
    lines.append("")

    # Discovery and equilibrium view
    lines.append("## Discovery and equilibrium signals\n")
    lines.append(
        f"- Cycles with breakthrough signatures: **{be_stats['breakthrough_cycles']}** "
        f"of **{be_stats['total_cycles']}**"
    )
    tier_counts = be_stats["tier_counts"]
    lines.append(
        f"- Discovery tier counts: "
        f"tier 1 candidates **{tier_counts.get('tier_1_candidate', 0)}**, "
        f"tier 2 candidates **{tier_counts.get('tier_2_candidate', 0)}**, "
        f"tier 3 hints **{tier_counts.get('tier_3_hint', 0)}**"
    )
    lines.append(
        f"- Overall discovery tier for this run: **{be_stats['overall_tier']}**"
    )

    if isinstance(be_stats.get("best_breakthrough_score"), (int, float)):
        lines.append(
            f"- Best breakthrough score observed: **{be_stats['best_breakthrough_score']:.3f}**"
        )

    eq_counts = be_stats["equilibrium_label_counts"]
    if eq_counts:
        lines.append("- Equilibrium labels seen across cycles:")
        for lbl, cnt in sorted(eq_counts.items(), key=lambda kv: kv[0]):
            lines.append(f"  - `{lbl}`: **{cnt}** cycles")

    if isinstance(be_stats.get("avg_equilibrium_score"), (int, float)):
        lines.append(
            f"- Avg equilibrium score: **{be_stats['avg_equilibrium_score']:.3f}**"
        )
    if isinstance(be_stats.get("avg_oscillation_score"), (int, float)):
        lines.append(
            f"- Avg oscillation score: **{be_stats['avg_oscillation_score']:.3f}**"
        )
    lines.append("")

    # Citation coverage view
    lines.append("## Citation coverage\n")
    lines.append(
        f"- Cycles with citations: **{cite_stats['cycles_with_citations']}** of "
        f"**{cite_stats['total_cycles']}** "
        f"({cite_stats['coverage_fraction'] * 100.0:.1f}% coverage)"
    )
    lines.append(
        f"- Total citations recorded: **{cite_stats['total_citations']}** "
        f"(avg **{cite_stats['avg_citations_per_cycle']:.2f}** per cycle)"
    )
    lines.append(
        f"- Unique sources: **{cite_stats['unique_source_count']}**, "
        f"unique papers or URLs: **{cite_stats['unique_paper_count']}**"
    )
    lines.append("")

    # Option C self diagnosis
    lines.append("## Option C self diagnosis (10x learning signals)\n")

    tier_label = run_tier.get("tier")
    tier_reason = run_tier.get("reason")
    if tier_label:
        lines.append(f"- Run tier: **{tier_label}** ({tier_reason})")

    env_state = env.get("state")
    env_details = env.get("details") or {}
    if env_state:
        reason = env_details.get("reason", "")
        lines.append(f"- Autonomy safety envelope: **{env_state}** ({reason})")

    early_score = early_fail.get("score")
    if isinstance(early_score, (int, float)):
        lines.append(
            f"- Early failure warning score: **{early_score:.3f}** (higher more risk)"
        )

    bp_val = bp.get("probability")
    if isinstance(bp_val, (int, float)):
        lines.append(
            f"- Near term breakthrough probability (heuristic): **{bp_val:.3f}**"
        )

    bp90_val = bp90.get("probability")
    if isinstance(bp90_val, (int, float)):
        lines.append(
            f"- 90 day breakthrough likelihood (heuristic): **{bp90_val:.3f}**"
        )

    eq_flag = equilibrium.get("in_equilibrium")
    eq_reason = equilibrium.get("reason")
    if eq_flag is not None:
        state_txt = "yes" if eq_flag else "no"
        lines.append(
            f"- RYE equilibrium detected: **{state_txt}** (reason: {eq_reason})"
        )

    vol_score = volatility.get("volatility_score")
    if isinstance(vol_score, (int, float)):
        lines.append(
            f"- Local volatility score: **{vol_score:.3f}** (1.0 very stable, 0.0 very noisy)"
        )

    if isinstance(harmonic, (int, float)):
        lines.append(
            f"- TGRM harmonic index: **{harmonic:.3f}** (proxy for coherent self repair)"
        )

    if hours_run is not None:
        lines.append(f"- Hours run so far (approx): **{hours_run:.2f} h**")

    # Learning meta
    lm_speed = learning_meta.get("learning_speed_label")
    lm_comment = learning_meta.get("learning_speed_comment")
    lm_curriculum = learning_meta.get("curriculum_stage")
    lm_profile = learning_meta.get("profile")
    if lm_speed:
        lines.append(f"- Option C learning speed label: **{lm_speed}**")
    if lm_curriculum:
        lines.append(f"- Curriculum stage: **{lm_curriculum}**")
    if lm_profile:
        lines.append(f"- Learning profile: **{lm_profile}**")
    if lm_comment:
        lines.append(f"  - {lm_comment}")

    sol_mode = spread_meta.get("mode")
    sol_parents = spread_meta.get("parents_used")
    if sol_mode:
        lines.append(f"- Spread of learning mode: **{sol_mode}**")
    if isinstance(sol_parents, int):
        lines.append(
            f"- Parent goals used for cross copying: **{sol_parents}**"
        )

    # Safety meta
    hard_violations = safety_meta.get("hard_violations")
    soft_warnings = safety_meta.get("soft_warnings")
    if isinstance(hard_violations, int) and hard_violations > 0:
        lines.append(
            f"- Hard safety violations recorded: **{hard_violations}**"
        )
    if isinstance(soft_warnings, int) and soft_warnings > 0:
        lines.append(
            f"- Soft safety warnings recorded: **{soft_warnings}**"
        )

    # Curriculum meta
    if isinstance(curriculum_meta, dict) and curriculum_meta:
        lines.append("- Curriculum meta overview:")
        _render_generic_optional_dict(curriculum_meta, lines, prefix="  - ", max_items=8)

    # MSIL meta from Option C
    if isinstance(msil_meta, dict) and msil_meta:
        lines.append("- MSIL meta overview:")
        _render_generic_optional_dict(msil_meta, lines, prefix="  - ", max_items=8)

    # Local learning speed grade
    lines.append(f"- Learning speed grade: **{speed_grade_label}**")
    lines.append(f"  - {speed_grade_text}")
    lines.append("")

    # Domain level view
    lines.append("## Domain level RYE profile\n")
    if not domain_stats:
        lines.append("No RYE data per domain.\n")
    else:
        lines.append("| Domain | Cycles | Avg RYE |")
        lines.append("|--------|--------|---------|")
        for d, stats in sorted(domain_stats.items(), key=lambda kv: kv[0]):
            lines.append(
                f"| {d} | {stats['count']} | {stats['avg_rye']:.3f} |"
            )
        lines.append("")

    # Swarm role view
    lines.append("## Swarm role efficiency\n")
    if not role_stats or (len(role_stats) == 1 and "agent" in role_stats):
        lines.append(
            "Single role run or insufficient role diversity for swarm analysis.\n"
        )
    else:
        lines.append("| Role | Cycles | Avg RYE |")
        lines.append("|------|--------|---------|")
        for r, stats in sorted(role_stats.items(), key=lambda kv: kv[0]):
            lines.append(
                f"| {r} | {stats['count']} | {stats['avg_rye']:.3f} |"
            )
        lines.append("")

    # Option C meta segments
    lines.append("## Meta controller segments (Option C)\n")
    if not meta_segments:
        lines.append(
            "No meta segments detected in run_metadata. This may mean classic mode was used, "
            "or the worker did not record segment indices.\n"
        )
    else:
        lines.append(
            "These segment level statistics are inferred from per cycle run_metadata. "
            "Each segment usually corresponds to a meta phase such as exploration, "
            "stabilization, or refinement."
        )
        lines.append("")
        lines.append("| Segment | Phase | Mode | Runtime profile | Cycles | Avg RYE | Best RYE |")
        lines.append("|---------|-------|------|-----------------|--------|---------|----------|")
        for idx in sorted(meta_segments.keys()):
            seg = meta_segments[idx]
            phase = seg.get("phase") or ""
            mode = seg.get("mode") or ""
            rp = seg.get("runtime_profile") or ""
            cnt = int(seg.get("count", 0))
            avg_seg = seg.get("avg_rye")
            max_seg = seg.get("max_rye")
            avg_str = f"{avg_seg:.3f}" if isinstance(avg_seg, (int, float)) else "n/a"
            best_str = f"{max_seg:.3f}" if isinstance(max_seg, (int, float)) else "n/a"
            lines.append(
                f"| {idx} | {phase} | {mode} | {rp} | {cnt} | {avg_str} | {best_str} |"
            )
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
            lines.append(
                f"- Streaming avg RYE: **{float(gi['avg_rye']):.3f}**"
            )
        if isinstance(gi.get("rye_count"), int):
            lines.append(f"- RYE samples tracked: **{gi['rye_count']}**")
        if gi.get("equilibrium_label"):
            lines.append(
                f"- Equilibrium label: `{gi['equilibrium_label']}`"
            )
        if gi.get("curriculum_stage"):
            lines.append(
                f"- Curriculum stage: `{gi['curriculum_stage']}`"
            )
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
            lines.append(
                f"... and {len(all_hypotheses) - 60} more.\n"
            )

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
            lines.append(
                f"... and {len(unique_cites) - 50} more.\n"
            )

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
            score = d.get("score", None)
            tier = d.get("tier") or d.get("discovery_tier")
            tier_str = f", tier {tier}" if tier else ""
            if isinstance(score, (int, float)):
                lines.append(
                    f"- [{ts}] [{kind}{tier_str}] ({score:.2f}) {label}"
                )
            else:
                lines.append(f"- [{ts}] [{kind}{tier_str}] {label}")
        lines.append("")

    # Interpretation
    lines.append("## Reparodynamics interpretation\n")
    lines.append(
        "This session reflects a sequence of TGRM cycles (Test -> Detect -> Repair -> Verify). "
        "RYE quantifies how much verified improvement (delta R) occurred per unit energy (E). "
        "The stability index, momentum, trend, oscillation level, Option C safety envelope, "
        "and MSIL snapshot together indicate whether the system is moving toward a stable high "
        "repair yield equilibrium or oscillating in a more exploratory regime. Run tier, learning "
        "speed grade, discovery tiers, breakthrough likelihood, and citation coverage give a fast, "
        "human-friendly summary of where this experiment sits in the Reparodynamics landscape and "
        "how aggressively it is learning without leaving its safety envelope."
    )

    return normalize_text("\n".join(lines))


def generate_publishable_report(
    memory_store: Any,
    goal: Optional[str] = None,
    run_id: Optional[str] = None,
    max_findings: int = 12,
    max_sources: int = 25,
) -> str:
    """Generate a *high-signal* report intended for external readers.

    Unlike `generate_report`, this mode avoids per-cycle diagnostics and raw dumps.
    It aims to be suitable for sharing, grading, or investor review.
    """
    cycles_all = _safe_get_cycle_history(memory_store)
    if goal:
        cycles_all = [c for c in cycles_all if (c.get("goal") or "") == goal]
    if run_id:
        cycles_all = [c for c in cycles_all if str(c.get("run_id") or "") == str(run_id)]
    cycles = list(cycles_all)

    if not cycles:
        # Fall back to the full diagnostic report if we have no cycle history.
        try:
            return generate_report(memory_store=memory_store, goal=goal, run_id=run_id)
        except Exception:
            return "# Autonomous research report\n\nNo cycle history available."

    inferred_run_id = run_id or cycles[-1].get("run_id") or cycles[0].get("run_id")
    inferred_domain = cycles[-1].get("domain") or cycles[0].get("domain") or "general"
    inferred_goal = goal or cycles[-1].get("goal") or cycles[0].get("goal") or ""

    timestamps = [str(c.get("timestamp")) for c in cycles if c.get("timestamp")]
    runtime = _extract_session_runtime(timestamps) or "(runtime unavailable)"

    # Pull structured discoveries from events.jsonl if possible.
    structured: List[Dict[str, Any]] = []
    try:
        events = _load_run_events(str(inferred_run_id) if inferred_run_id else "")
        structured = _events_to_structured_discoveries(events, goal=inferred_goal)
    except Exception:
        structured = []

    # Extract citations from the cycle history (best-effort).
    citations: List[Dict[str, Any]] = []
    seen_cite: set = set()

    def _cite_key(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        url = str(obj.get("url") or obj.get("link") or "").strip()
        doi = str(obj.get("doi") or "").strip()
        title = str(obj.get("title") or "").strip()
        key = url or doi or title
        key = key.strip().lower()
        return key or None

    def _ingest_cites(cite_list: Any) -> None:
        if not isinstance(cite_list, list):
            return
        for c in cite_list:
            if not isinstance(c, dict):
                continue
            k = _cite_key(c)
            if not k or k in seen_cite:
                continue
            seen_cite.add(k)
            citations.append(c)

    for c in cycles:
        _ingest_cites(c.get("citations"))
        _ingest_cites(c.get("sources"))
        summ = c.get("summary") if isinstance(c.get("summary"), dict) else {}
        if isinstance(summ, dict):
            _ingest_cites(summ.get("citations"))
            _ingest_cites(summ.get("sources"))

    # Heuristic citation matching for inline refs.
    def _match_citations(text: str, max_refs: int = 2) -> List[int]:
        if not citations:
            return []
        t = (text or "").lower()
        # Extract a few meaningful tokens
        tokens = [w for w in re.findall(r"[a-zA-Z]{6,}", t) if w not in {"because", "within", "across", "system", "effect", "effects", "during"}]
        if not tokens:
            return []
        tokens = tokens[:8]
        matches: List[int] = []
        for idx, c in enumerate(citations, start=1):
            title = str(c.get("title") or "").lower()
            if not title:
                continue
            if any(tok in title for tok in tokens):
                matches.append(idx)
            if len(matches) >= max_refs:
                break
        return matches

    # Rank findings: prefer higher score/tier and later timestamps.
    def _finding_score(d: Dict[str, Any]) -> float:
        base = float(d.get("score") or 0.0) if isinstance(d.get("score"), (int, float)) else 0.0
        tier = d.get("tier") or d.get("discovery_tier")
        try:
            if tier is not None:
                base += float(tier) * 0.5
        except Exception:
            pass
        return base

    findings = sorted(structured, key=_finding_score, reverse=True)
    findings = findings[: max(0, int(max_findings))]

    # Filter out low-quality or placeholder-like discoveries.  We drop any
    # finding whose label contains internal logs, templates, or hedging
    # language, or that lacks any matching citation in the current run.
    filtered_findings: List[Dict[str, Any]] = []
    for d in findings:
        label = str(d.get("label") or "").strip()
        if not label:
            continue
        # Skip internal or template entries (e.g. maintenance logs, examples)
        if _contains_banned_pattern(label) or _is_vague(label):
            continue
        # Only keep findings that can be matched to at least one citation
        try:
            matched = _match_citations(label)
        except Exception:
            matched = []
        if not matched:
            continue
        filtered_findings.append(d)
    findings = filtered_findings

    # Group findings by kind for readability.
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for d in findings:
        kind = str(d.get("kind") or "other")
        buckets.setdefault(kind, []).append(d)

    lines: List[str] = []
    lines.append("# Autonomous research report\n")
    lines.append("## Run overview")
    lines.append(f"- Run id: `{inferred_run_id}`" if inferred_run_id else "- Run id: (unavailable)")
    lines.append(f"- Domain: **{inferred_domain}**")
    lines.append(f"- Cycles captured: **{len(cycles)}**")
    lines.append(f"- Runtime: **{runtime}**")
    if inferred_goal:
        lines.append("- Goal:")
        lines.append(f"  > {normalize_text(inferred_goal).strip()}")
    lines.append("")

    # Abstract
    lines.append("## Abstract")
    if findings:
        top_labels = [str(f.get("label") or "").strip() for f in findings[:3] if str(f.get("label") or "").strip()]
    else:
        top_labels = []
    lines.append(
        "This document is an automated synthesis of an ARA run. It summarizes the highest-signal "
        "discoveries/hypotheses extracted from the run artifacts (cycle history and, when available, "
        "structured discovery events)."
    )
    if top_labels:
        lines.append("")
        lines.append("Key outputs include:")
        for lbl in top_labels:
            refs = _match_citations(lbl)
            ref_txt = f" [{', '.join(str(r) for r in refs)}]" if refs else ""
            lines.append(f"- {normalize_text(lbl)}{ref_txt}")
    lines.append("")

    # Findings
    lines.append("## Key findings")
    if not findings:
        lines.append("No structured findings were available for this run.")
    else:
        kind_order = ["mechanism", "intervention", "biomarker", "causal", "other"]
        for kind in kind_order:
            if kind not in buckets:
                continue
            lines.append(f"### {kind.capitalize()}")
            for d in buckets.get(kind, []):
                label = str(d.get("label") or "").strip()
                if not label:
                    continue
                refs = _match_citations(label)
                ref_txt = f" [{', '.join(str(r) for r in refs)}]" if refs else ""
                lines.append(f"- {normalize_text(label)}{ref_txt}")
            lines.append("")

    # Evidence base
    lines.append("## Sources")
    if not citations:
        lines.append("No citations were captured in cycle history.")
    else:
        use = citations[: max(0, int(max_sources))]
        for i, c in enumerate(use, start=1):
            title = normalize_text(c.get("title") or c.get("raw") or "(untitled)").strip()
            url = str(c.get("url") or c.get("link") or "").strip()
            if url:
                lines.append(f"{i}. {title} â {url}")
            else:
                lines.append(f"{i}. {title}")

    # Limitations
    lines.append("\n## Limitations")
    lines.append(
        "- This is an automated synthesis; verify important claims against the primary sources.\n"
        "- If the run produced few structured discovery events or citations, the report may underrepresent useful details from raw notes."
    )

    return normalize_text("\n".join(lines))


# ---------------------------------------------------------
# TARGETED FINDINGS REPORT
# ---------------------------------------------------------
def generate_findings_report(memory_store: Any, goal: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """
    Extract actionable findings such as:
    - Possible cures
    - Potential treatments
    - Interventions
    - Mechanisms
    - High value biomarkers or targets
    """

    all_cycles = _safe_get_cycle_history(memory_store)

    if goal:
        cycles = [c for c in all_cycles if (c.get("goal") or "") == goal]
    else:
        cycles = list(all_cycles)

    # Run scoping: prefer explicit run_id, then infer from memory/cycle history.
    inferred_run_id = _infer_run_id(memory_store, cycles, explicit=run_id)

    # If cycles carry run_id fields, filter to the inferred run_id to avoid blending runs.
    if inferred_run_id and any(c.get("run_id") is not None for c in cycles):
        try:
            filtered = [c for c in cycles if str(c.get("run_id")) == str(inferred_run_id)]
            if filtered:
                cycles = filtered
        except Exception:
            pass

    if not cycles:
        if inferred_run_id:
            try:
                _events = _load_run_events(str(inferred_run_id))
                if _events:
                    return _render_event_only_findings_report(_events, run_id=str(inferred_run_id), goal=goal)
            except Exception:
                pass
        return "# Targeted Findings Report\n\nNo cycles logged."

    # Load per-run events once (used for discovery/candidate sections and robustness).
    run_events: List[Dict[str, Any]] = []
    if inferred_run_id:
        try:
            run_events = _load_run_events(str(inferred_run_id))
        except Exception:
            run_events = []

    timestamps: List[str] = []
    all_citations: List[Dict[str, Any]] = []

    for c in cycles:
        ts = c.get("timestamp")
        if ts:
            timestamps.append(ts)

        summary = c.get("summary")
        cits = c.get("citations") or []
        if isinstance(summary, dict):
            cits = cits or summary.get("citations") or []
        for ct in cits:
            if isinstance(ct, dict):
                all_citations.append(ct)

    runtime = _extract_session_runtime(timestamps)

    # Structured discoveries
    structured: List[Dict[str, Any]] = []
    try:
        structured = (
            memory_store.get_discoveries(goal=goal)
            if hasattr(memory_store, "get_discoveries")
            else []
        )
    except Exception:
        structured = []

    # If agent memory has no structured discoveries yet, fall back to the event stream.
    if not structured and run_events:
        try:
            structured = _events_to_structured_discoveries(run_events, goal=goal) or structured
        except Exception:
            pass

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
        summary = c.get("summary")
        hyps = c.get("hypotheses") or []
        if isinstance(summary, dict):
            hyps = hyps or summary.get("hypotheses") or []
        for h in hyps:
            text = h["text"] if isinstance(h, dict) else str(h)
            if any(k.lower() in text.lower() for k in KEYWORDS):
                findings_text.append(text)

    # If cycle summaries have no hypothesis text, mine the event stream for keyword hits.
    if not findings_text and run_events:
        try:
            for e in run_events:
                if not isinstance(e, dict):
                    continue
                k = str(e.get("kind") or "").strip()
                data = e.get("data") if isinstance(e.get("data"), dict) else {}
                text_blob = ""
                if k in ("candidate_hypothesis", "discovery", "discovery_candidate"):
                    text_blob = (
                        data.get("mechanism_chain")
                        or data.get("hidden_constraint")
                        or data.get("thesis")
                        or data.get("title")
                        or ""
                    )
                elif k == "agent_output":
                    text_blob = data.get("text") or ""
                if text_blob and any(kw.lower() in str(text_blob).lower() for kw in KEYWORDS):
                    findings_text.append(str(text_blob))
                if len(findings_text) >= 200:
                    break
        except Exception:
            pass

    # Build report
    lines: List[str] = []
    lines.append("# Targeted Findings Report\n")

    if runtime:
        lines.append(f"**Session runtime:** {runtime}\n")
    if inferred_run_id:
        lines.append(f"**Run ID:** `{inferred_run_id}`\n")

    if goal:
        lines.append(f"**Filtered goal:** {goal}\n")

    # Structured discoveries first
    lines.append("## Structured discoveries from agent memory\n")
    if not structured:
        lines.append("No structured discoveries recorded.\n")
    else:

        def _disc_key(d: Dict[str, Any]) -> Tuple[float, str]:
            score = d.get("score")
            s_val = float(score) if isinstance(score, (int, float)) else 0.0
            ts_local = str(d.get("timestamp", ""))
            return (-s_val, ts_local)

        sorted_disc = sorted(structured, key=_disc_key)
        for i, d in enumerate(sorted_disc[:50], start=1):
            kind = d.get("kind", "")
            label = d.get("label", "")
            score = d.get("score")
            ts_local = d.get("timestamp", "")
            tier = d.get("tier") or d.get("discovery_tier")
            tier_str = f", tier {tier}" if tier else ""
            score_str = (
                f" (score {score:.2f})" if isinstance(score, (int, float)) else ""
            )
            lines.append(f"{i}. [{kind}{tier_str}] {label}{score_str}")
            if ts_local:
                lines.append(f"   - recorded at `{ts_local}`")
        if len(sorted_disc) > 50:
            lines.append(
                f"... and {len(sorted_disc) - 50} more structured discoveries.\n"
            )

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
            lines.append(
                f"... and {len(unique_findings) - 80} more hypotheses mentioning treatments or mechanisms.\n"
            )

    # Citations that back these findings
    lines.append("\n## Key citations supporting findings\n")
    if not all_citations:
        lines.append("No citations recorded for these cycles.\n")
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
            lines.append(
                f"... and {len(unique_cites) - 50} more.\n"
            )

    return "\n".join(lines)


# ---------------------------------------------------------
# PDF WRAPPERS
# ---------------------------------------------------------
def generate_report_pdf(
    memory_store: Any,
    goal: Optional[str] = None,
    run_id: Optional[str] = None,
    output_path: str = "autonomous_agent_report.pdf",
) -> str:
    """Generate the full report and write it to a PDF file."""
    md = generate_report(memory_store, goal=goal, run_id=run_id)
    title = "Autonomous Research Agent Report"
    if goal:
        title = f"{title} - Goal Snapshot"
    return _write_pdf(md, output_path=output_path, title=title)


def generate_findings_report_pdf(
    memory_store: Any,
    goal: Optional[str] = None,
    run_id: Optional[str] = None,
    output_path: str = "autonomous_agent_findings_report.pdf",
) -> str:
    """Generate the targeted findings report and write it to a PDF file."""
    md = generate_findings_report(memory_store, goal=goal, run_id=run_id)
    title = "Autonomous Agent Findings Report"
    if goal:
        title = f"{title} - Goal Snapshot"
    return _write_pdf(md, output_path=output_path, title=title)
