"""
report_builder.py
================

Build a run report *from the structured event stream*, not from templates.

Key behaviour:
  * Reads the per-run JSONL stream: runs_root/<run_id>/events.jsonl
  * Falls back to legacy JSON-array logs when JSONL is unavailable
  * Ranks cycles by RYE signals (delta_R, E, RYE)
  * Produces a narrative Markdown report with citations and **full agent outputs**
    (no truncation of the output text)

The entrypoint is `build_agent_report(...)` which keeps backwards compatibility
with earlier callers that pass (goal, domain, diagnostics, history).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple, Union


JsonObj = Dict[str, Any]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def _resolve_runs_root() -> Path:
    for k in ("ARA_RUNS_DIR", "ARA_RUNS_ROOT", "RUNS_DIR"):
        v = os.getenv(k)
        if v:
            try:
                return Path(v).expanduser().resolve()
            except Exception:
                pass

    # Optional: align with run_jobs if available
    try:
        from .run_jobs import BASE_DIR as _BASE_DIR  # type: ignore
        if isinstance(_BASE_DIR, Path):
            return _BASE_DIR
        if isinstance(_BASE_DIR, str) and _BASE_DIR.strip():
            return Path(_BASE_DIR).expanduser().resolve()
    except Exception:
        pass

    try:
        here = Path(__file__).resolve()
        return (here.parent.parent / "runs").resolve()
    except Exception:
        return Path("./runs").resolve()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[JsonObj]:
    out: List[JsonObj] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _source_key(src: Any) -> str:
    if isinstance(src, dict):
        return _safe_str(src.get("url") or src.get("id") or src.get("title") or json.dumps(src, sort_keys=True, ensure_ascii=False))
    return _safe_str(src)


def _format_source(src: Any) -> str:
    if isinstance(src, dict):
        title = _safe_str(src.get("title") or src.get("name") or src.get("source") or src.get("url") or "")
        url = _safe_str(src.get("url") or "")
        if url and title and title != url:
            return f"{title} - {url}"
        return title or url or _safe_str(src)
    return _safe_str(src)


def _load_events_for_run(run_id: str, *, runs_root: Optional[Path] = None) -> List[JsonObj]:
    """Load events for a run from the best available location."""
    rid = _safe_str(run_id).strip()
    if not rid:
        return []
    root = runs_root if isinstance(runs_root, Path) else _resolve_runs_root()
    logs = root / "logs"

    candidates: List[Path] = [
        root / rid / "events.jsonl",                 # primary (per-run JSONL)
        logs / "events_global.jsonl",                # global mirror JSONL
        logs / f"{rid}_event_log.json",              # legacy per-run JSON array
        logs / f"{rid}_events.json",                 # another legacy naming
        logs / "event_log.json",                     # legacy global
        logs / "events.json",                        # legacy global alt
    ]

    events: List[JsonObj] = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".jsonl":
                loaded = _read_jsonl(p)
            else:
                loaded = _read_json(p)
            if isinstance(loaded, list):
                events.extend([e for e in loaded if isinstance(e, dict)])
            elif isinstance(loaded, dict):
                # common wrappers
                if isinstance(loaded.get("events"), list):
                    events.extend([e for e in loaded.get("events") if isinstance(e, dict)])
        except Exception:
            continue

        # If we loaded from the primary per-run file, we can stop early.
        if p == (root / rid / "events.jsonl") and events:
            break

    # Filter to run_id (important when reading from global mirrors)
    filtered: List[JsonObj] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        if _safe_str(e.get("run_id") or "").strip() == rid:
            filtered.append(e)

    # Sort by timestamp if available
    def _ts_key(ev: JsonObj) -> str:
        return _safe_str(ev.get("timestamp") or ev.get("time") or "")

    filtered.sort(key=_ts_key)
    return filtered


def _extract_text_from_event(ev: JsonObj) -> str:
    """Extract the primary text content from an event.

    This helper searches common keys in the event's data payload and
    topâlevel for a text string.  It also filters out placeholder or
    incomplete messages (e.g. 'TODO', 'TBD', 'placeholder') which should
    not appear in final reports.  If no valid text is found the empty
    string is returned.
    """
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    # Candidate keys in order of precedence
    for key in ("text", "output", "content", "message"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            # Filter out placeholders; skip if common placeholder keywords appear
            low = text.lower()
            if any(p in low for p in ("todo", "tbd", "placeholder", "insert research here")):
                return ""
            return text
    # Some writers store the text at top-level
    for key in ("message", "msg", "text"):
        v = ev.get(key)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            low = text.lower()
            if any(p in low for p in ("todo", "tbd", "placeholder", "insert research here")):
                return ""
            return text
    return ""


def _extract_sources_from_event(ev: JsonObj) -> List[Any]:
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    sources = data.get("sources")
    if isinstance(sources, list):
        return sources
    citations = data.get("citations")
    if isinstance(citations, list):
        return citations
    # top-level fallbacks
    if isinstance(ev.get("sources"), list):
        return ev.get("sources")  # type: ignore[return-value]
    if isinstance(ev.get("citations"), list):
        return ev.get("citations")  # type: ignore[return-value]
    return []


def _cycle_int(ev: JsonObj) -> Optional[int]:
    c = ev.get("cycle")
    try:
        return None if c is None else int(c)
    except Exception:
        return None


def _float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _compute_cycle_metrics(events: List[JsonObj]) -> Dict[int, Dict[str, Any]]:
    """Aggregate delta_R, E, RYE per cycle from rye_update events."""
    metrics: Dict[int, Dict[str, Any]] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("kind") or "") != "rye_update":
            continue
        c = _cycle_int(ev)
        if c is None:
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        dr = _float_or_none(data.get("delta_R") or data.get("deltaR") or data.get("delta_r"))
        en = _float_or_none(data.get("E") or data.get("energy"))
        ry = _float_or_none(data.get("RYE") or data.get("rye") or data.get("rye_value"))

        m = metrics.setdefault(c, {"delta_R_sum": 0.0, "E_sum": 0.0, "RYE_values": []})
        if dr is not None:
            m["delta_R_sum"] += dr
        if en is not None:
            m["E_sum"] += en
        if ry is not None:
            m["RYE_values"].append(ry)

    for c, m in metrics.items():
        vals = m.get("RYE_values") or []
        if isinstance(vals, list) and vals:
            try:
                m["RYE_avg"] = sum(vals) / float(len(vals))
                m["RYE_max"] = max(vals)
            except Exception:
                m["RYE_avg"] = None
                m["RYE_max"] = None
        else:
            m["RYE_avg"] = None
            m["RYE_max"] = None
    return metrics


def _rank_cycles(cycle_metrics: Dict[int, Dict[str, Any]]) -> List[int]:
    """Return cycles ranked by RYE signals (desc)."""
    def key(c: int) -> Tuple[float, float, float]:
        m = cycle_metrics.get(c, {})
        rye = m.get("RYE_avg")
        dr = m.get("delta_R_sum")
        en = m.get("E_sum")
        rye_f = float(rye) if isinstance(rye, (int, float)) else float("-inf")
        dr_f = float(dr) if isinstance(dr, (int, float)) else 0.0
        en_f = float(en) if isinstance(en, (int, float)) else 0.0
        return (rye_f, dr_f, en_f)

    return sorted(cycle_metrics.keys(), key=key, reverse=True)


def _collect_source_index(events: List[JsonObj]) -> Dict[str, int]:
    """Assign stable 1-based indices for unique sources across the run."""
    seen: Dict[str, int] = {}
    for ev in events:
        for src in _extract_sources_from_event(ev):
            k = _source_key(src)
            if not k:
                continue
            if k not in seen:
                seen[k] = len(seen) + 1
    return seen


def _format_source_refs(ev: JsonObj, source_index: Dict[str, int]) -> str:
    idxs: List[int] = []
    for src in _extract_sources_from_event(ev):
        k = _source_key(src)
        if not k or k not in source_index:
            continue
        idxs.append(source_index[k])
    if not idxs:
        return ""
    idxs = sorted(set(idxs))
    return " [" + ", ".join(str(i) for i in idxs) + "]"


def build_agent_report(
    goal: str,
    domain: str,
    diagnostics: Optional[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]],
    *,
    run_id: Optional[str] = None,
    runs_root: Optional[Path] = None,
    top_cycles: int = 5,
) -> str:
    """Build a narrative Markdown report for a run.

    Parameters
    ----------
    goal/domain/diagnostics/history:
        Kept for backwards compatibility with older callers.
    run_id:
        If provided, uses the event stream for that run_id. If not provided, we will
        attempt to infer it from diagnostics or history entries.
    top_cycles:
        Number of cycles to highlight (ranked by RYE signals). The appendix includes
        full agent outputs for *all* cycles present in the event stream.

    Returns
    -------
    Markdown string.
    """
    rid = _safe_str(run_id or (diagnostics or {}).get("run_id") or "").strip()
    if not rid and history:
        for h in history:
            if isinstance(h, dict) and _safe_str(h.get("run_id") or "").strip():
                rid = _safe_str(h.get("run_id")).strip()
                break

    events: List[JsonObj] = _load_events_for_run(rid, runs_root=runs_root) if rid else []

    # If we cannot load events, fall back to the old history-based dump.
    if not events:
        # Very small, deterministic fallback to avoid empty reports.
        lines: List[str] = []
        lines.append(f"# Run Report")
        if rid:
            lines.append(f"- run_id: `{rid}`")
        lines.append(f"- generated_at: `{_utc_iso()}`")
        lines.append(f"- domain: `{domain}`")
        lines.append("")
        lines.append("## Goal")
        lines.append(goal or "")
        lines.append("")
        lines.append("## Diagnostics")
        lines.append("```json")
        try:
            lines.append(json.dumps(diagnostics or {}, ensure_ascii=False, indent=2))
        except Exception:
            lines.append("{}")
        lines.append("```")
        lines.append("")
        lines.append("## Cycle history (fallback)")
        if history:
            for i, h in enumerate(history[-20:]):
                lines.append(f"### Cycle {i}")
                try:
                    lines.append("```json")
                    lines.append(json.dumps(h, ensure_ascii=False, indent=2))
                    lines.append("```")
                except Exception:
                    lines.append(str(h))
        return "\n".join(lines)

    # Organize events
    agent_outputs = [e for e in events if str(e.get("kind") or "") == "agent_output"]
    funnel_stages = [e for e in events if (e.get("kind") or "").lower() == "funnel_stage"]
    discoveries = [e for e in events if str(e.get("kind") or "") == "discovery"]
    hypotheses = [e for e in events if str(e.get("kind") or "") == "candidate_hypothesis"]
    verifications = [e for e in events if str(e.get("kind") or "") == "verification"]
    rye_events = [e for e in events if str(e.get("kind") or "") == "rye_update"]

    cycle_metrics = _compute_cycle_metrics(events)
    ranked_cycles = _rank_cycles(cycle_metrics)
    highlight_cycles = ranked_cycles[: max(1, int(top_cycles))] if ranked_cycles else []

    # Sources index (global, stable)
    source_index = _collect_source_index(events)

    # Group by cycle for narrative sections
    by_cycle_outputs: DefaultDict[int, List[JsonObj]] = defaultdict(list)
    by_cycle_discoveries: DefaultDict[int, List[JsonObj]] = defaultdict(list)
    by_cycle_hyp: DefaultDict[int, List[JsonObj]] = defaultdict(list)
    by_cycle_ver: DefaultDict[int, List[JsonObj]] = defaultdict(list)
    by_cycle_funnel: DefaultDict[int, List[JsonObj]] = defaultdict(list)

    def _cycle_bucket(ev: JsonObj) -> int:
        c = _cycle_int(ev)
        return int(c) if c is not None else -1

    for ev in agent_outputs:
        by_cycle_outputs[_cycle_bucket(ev)].append(ev)
    for ev in discoveries:
        by_cycle_discoveries[_cycle_bucket(ev)].append(ev)
    for ev in hypotheses:
        by_cycle_hyp[_cycle_bucket(ev)].append(ev)
    for ev in verifications:
        by_cycle_ver[_cycle_bucket(ev)].append(ev)
    for ev in funnel_stages:
        by_cycle_funnel[_cycle_bucket(ev)].append(ev)

    # Build report
    out: List[str] = []
    out.append(f"# Run Report")
    out.append(f"- run_id: `{rid}`")
    out.append(f"- generated_at: `{_utc_iso()}`")
    out.append(f"- domain: `{domain}`")
    out.append(f"- total_events: `{len(events)}`")
    out.append(f"- total_agent_outputs: `{len(agent_outputs)}`")
    out.append(f"- total_discoveries: `{len(discoveries)}`")
    out.append(f"- total_candidate_hypotheses: `{len(hypotheses)}`")
    out.append(f"- total_verifications: `{len(verifications)}`")
    out.append("")

    out.append("## Goal")
    out.append(goal or "")
    out.append("")

    # Executive summary driven by discoveries/hypotheses/verifications
    out.append("## Executive summary")
    if discoveries:
        out.append(f"- **Discoveries logged:** {len(discoveries)}")
    if hypotheses:
        out.append(f"- **Candidate hypotheses logged:** {len(hypotheses)}")
    if verifications:
        passed = 0
        total = 0
        for v in verifications:
            data = v.get("data") if isinstance(v.get("data"), dict) else {}
            if data.get("passed") is True:
                passed += 1
            if data.get("passed") is not None:
                total += 1
        if total:
            out.append(f"- **Verifications:** {passed}/{total} marked as passed")
        else:
            out.append(f"- **Verifications logged:** {len(verifications)}")
    if cycle_metrics:
        out.append(f"- **Cycles with RYE updates:** {len(cycle_metrics)}")
    out.append("")

    # RYE highlight table
    out.append("## RYE highlights")
    if not cycle_metrics:
        out.append("_No rye_update events were found._")
    else:
        out.append("| Cycle | RYE_avg | delta_R_sum | E_sum |")
        out.append("|---:|---:|---:|---:|")
        for c in (highlight_cycles or ranked_cycles):
            m = cycle_metrics.get(c, {})
            out.append(
                f"| {c} | {m.get('RYE_avg') if m.get('RYE_avg') is not None else ''} | {m.get('delta_R_sum', '')} | {m.get('E_sum', '')} |"
            )
    out.append("")

    # Key discoveries
    out.append("## Key discoveries")
    if not discoveries:
        out.append("_No discoveries were logged in the event stream._")
    else:
        # Show up to 15 most recent discoveries with citations
        for ev in discoveries[-15:]:
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            title = data.get("title") or data.get("name") or ev.get("message") or "discovery"
            desc = data.get("description") or data.get("details") or data.get("text") or ""
            role = ev.get("role") or data.get("role") or "agent"
            cycle = _cycle_int(ev)
            cite = _format_source_refs(ev, source_index)
            out.append(f"- **{title}** (cycle {cycle}, role {role}){cite}")
            if desc:
                out.append(f"  - {desc}")
    out.append("")

    # Hypotheses and verifications
    out.append("## Hypotheses and verification")
    if not hypotheses and not verifications:
        out.append("_No hypotheses or verification events were logged._")
    else:
        if hypotheses:
            out.append("### Candidate hypotheses")
            for ev in hypotheses[-15:]:
                data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
                text = data.get("text") or data.get("hypothesis") or _extract_text_from_event(ev)
                role = ev.get("role") or "agent"
                cycle = _cycle_int(ev)
                cite = _format_source_refs(ev, source_index)
                if text:
                    out.append(f"- (cycle {cycle}, role {role}){cite}")
                    out.append(f"  - {text}")
        if verifications:
            out.append("")
            out.append("### Verification results")
            for ev in verifications[-20:]:
                data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
                passed = data.get("passed")
                rationale = data.get("rationale") or data.get("reason") or data.get("notes") or ""
                role = ev.get("role") or "agent"
                cycle = _cycle_int(ev)
                cite = _format_source_refs(ev, source_index)
                out.append(f"- (cycle {cycle}, role {role}){cite} - **passed={passed}**")
                if rationale:
                    out.append(f"  - {rationale}")
    out.append("")

    # Cycle-by-cycle narrative for top cycles
    out.append("## Narrative by highlighted cycle")
    if not highlight_cycles:
        out.append("_No cycle metrics available to select highlights._")
    for c in highlight_cycles:
        out.append(f"### Cycle {c}")
        fs_events = by_cycle_funnel.get(c) or []
        if fs_events:
            fs = fs_events[-1]  # last writer wins
            fs_data = fs.get("data") or {}
            stage_name = (fs_data.get("funnel_stage") or fs.get("message") or "").strip()
            stage_tag = (fs_data.get("stage") or "").strip()
            if stage_name:
                if stage_tag and stage_tag not in stage_name:
                    out.append(f"**Funnel stage:** {stage_name}  *(stage tag: {stage_tag})*")
                else:
                    out.append(f"**Funnel stage:** {stage_name}")
        m = cycle_metrics.get(c, {})
        if m:
            out.append(
                f"- RYE_avg: `{m.get('RYE_avg')}`; delta_R_sum: `{m.get('delta_R_sum')}`; E_sum: `{m.get('E_sum')}`"
            )
        # discoveries/hyp/ver for this cycle
        disc_c = by_cycle_discoveries.get(c, [])
        hyp_c = by_cycle_hyp.get(c, [])
        ver_c = by_cycle_ver.get(c, [])
        if disc_c:
            out.append("")
            out.append("**Discoveries:**")
            for ev in disc_c:
                data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
                title = data.get("title") or data.get("name") or ev.get("message") or "discovery"
                desc = data.get("description") or data.get("details") or data.get("text") or ""
                cite = _format_source_refs(ev, source_index)
                out.append(f"- {title}{cite}")
                if desc:
                    out.append(f"  - {desc}")
        if hyp_c:
            out.append("")
            out.append("**Candidate hypotheses:**")
            for ev in hyp_c:
                data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
                text = data.get("text") or data.get("hypothesis") or _extract_text_from_event(ev)
                cite = _format_source_refs(ev, source_index)
                if text:
                    out.append(f"- {text}{cite}")
        if ver_c:
            out.append("")
            out.append("**Verifications:**")
            for ev in ver_c:
                data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
                passed = data.get("passed")
                rationale = data.get("rationale") or data.get("reason") or data.get("notes") or ""
                cite = _format_source_refs(ev, source_index)
                out.append(f"- passed={passed}{cite}")
                if rationale:
                    out.append(f"  - {rationale}")

        # A short narrative paragraph built from agent outputs (roles + first sentences)
        outs_c = by_cycle_outputs.get(c, [])
        if outs_c:
            out.append("")
            out.append("**Agent contributions (preview):**")
            for ev in outs_c:
                role = ev.get("role") or "agent"
                text = _extract_text_from_event(ev)
                if not text:
                    continue
                first_line = text.strip().splitlines()[0]
                cite = _format_source_refs(ev, source_index)
                out.append(f"- {role}: {first_line}{cite}")
        out.append("")

    # Appendix with full agent outputs (no truncation)
    out.append("## Appendix: full agent outputs (no truncation)")
    # Determine sorted cycles encountered in outputs
    all_cycles = sorted([c for c in by_cycle_outputs.keys() if c is not None])
    for c in all_cycles:
        out.append(f"### Cycle {c}")
        outs = by_cycle_outputs.get(c, [])
        if not outs:
            out.append("_No agent_output events for this cycle._")
            out.append("")
            continue

        # Group by role for readability
        by_role: DefaultDict[str, List[JsonObj]] = defaultdict(list)
        for ev in outs:
            r = _safe_str(ev.get("role") or "agent")
            by_role[r].append(ev)

        for role, evs in sorted(by_role.items(), key=lambda kv: kv[0]):
            out.append(f"#### Role: {role}")
            for ev in evs:
                ts = ev.get("timestamp")
                cite = _format_source_refs(ev, source_index)
                out.append(f"- timestamp: `{ts}`{cite}")
                text = _extract_text_from_event(ev)
                out.append("```")
                out.append(text or "")
                out.append("```")
            out.append("")

    # Global sources / citations index
    out.append("## Sources")
    if source_index:
        # Invert index to keep stable order
        inv = {v: k for k, v in source_index.items()}
        for i in range(1, len(inv) + 1):
            k = inv.get(i)
            if not k:
                continue
            # Find one representative src object to format nicely
            src_obj: Any = None
            for ev in events:
                for src in _extract_sources_from_event(ev):
                    if _source_key(src) == k:
                        src_obj = src
                        break
                if src_obj is not None:
                    break
            try:
                # Use normalized citation formatting when possible
                from .citation_utils import normalize_citation, format_citation_markdown  # type: ignore
                norm_cite = normalize_citation(src_obj)
                if norm_cite:
                    out.append(f"{i}. {format_citation_markdown(norm_cite)}")
                else:
                    out.append(f"{i}. {_format_source(src_obj) if src_obj is not None else k}")
            except Exception:
                out.append(f"{i}. {_format_source(src_obj) if src_obj is not None else k}")
    else:
        out.append("_No sources or citations were logged for this run. The output may be unverified._")

    # Assemble the report string
    report_text = "\n".join(out)
    # Normalize any misâdecoded UTFâ8 characters (e.g., bullets appearing as Ã¢â¬Â¢).
    # First attempt to use the citation_utils.normalize_text helper if available.
    try:
        from .citation_utils import normalize_text  # type: ignore
        report_text = normalize_text(report_text)
    except Exception:
        pass
    # Fallback: if common mojibake markers remain, try a local repair.
    try:
        if any(m in report_text for m in ("Ã", "Ã¢", "Ã")):
            # Round-trip through cp1252 first, then latin1, similar to report_generator.normalize_text
            def _count_markers(val: str) -> int:
                return sum(val.count(ch) for ch in ("Ã", "Ã¢", "Ã"))

            before = _count_markers(report_text)
            for enc in ("cp1252", "latin1"):
                try:
                    fixed = report_text.encode(enc).decode("utf-8")
                except Exception:
                    continue
                if fixed and _count_markers(fixed) < before:
                    report_text = fixed
                    break
            # Also replace bullet sequences if still present
            if "Ã¢â¬Â¢" in report_text:
                report_text = report_text.replace("Ã¢â¬Â¢", "â¢")
    except Exception:
        pass
    return report_text


__all__ = ["build_agent_report"]
