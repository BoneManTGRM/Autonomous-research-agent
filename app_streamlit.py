# -*- coding: utf-8 -*-
"""
Enhanced Streamlit interface for the Autonomous Research Agent.

Features:
- Finite mode only with manual cycle budgets (no timed presets in this build)
- Researcher plus Critic multi agent mode
- Swarm mode with up to dozens of mini agents
- Domain presets (General, Longevity, Math)
- PubMed / Semantic Scholar ingestion controls
- Biomarker analysis toggle (for anti aging teams)
- Hypothesis generation viewer
- PDF ingestion for real scientific papers
- RYE, delta_R, and Energy charts
- Real Tavily search support detection
- Source citation viewer
- Tools status panel for web browser and sandbox tools
- Discovery log viewer and autonomous discovery panel
- Snapshot timeline with equilibrium and stability view
- Hypothesis manager with confidence and domain filters
- Memory pruning controls (if supported by MemoryStore)
- Verification panel for cures, treatments, and stability checks
- Multi agent insight graph for roles, hypotheses, and discoveries
- Report generation from full cycle history
- Optional PDF report export (if reportlab is installed)
- Optional MSIL meta skill intelligence view when msil module is available

Live console upgrades (this update):
- Sticky heartbeat/status bar at the top
- Autonomy level indicator
- Agent presence chips (single / two-stage / swarm)
- Narrative timeline feed (uses event log if present; otherwise synthesizes from history)
- Discovery confidence cards (top discovery candidates)

Run diagnostics upgrades (this update):
- Unified loaders: read from MemoryStore when available, with file-based fallbacks
- Progress normalization (supports both cycle progress and phase progress, if emitted by worker)
    - Optional auto-refresh so the UI can actually show 1/3 -> 2/3 -> 3/3 while the worker runs

Reparodynamics:
    The UI is a front panel on a reparodynamic system:
    - It never runs TGRM cycles directly.
    - Each run request is written as a job in a file based queue and picked up by an external engine worker.
    - The worker runs TGRM loops, computes RYE = delta_R / E, and logs results.

Time:
    In this finite only build, the Streamlit UI only sends explicit cycle budgets.
    The engine worker decides how to map these to wall clock time if needed.

Note:
    This keeps the Streamlit app thin and safe:
    - It only queues jobs through run_jobs.py
    - It never imports or invokes the core loop
    - It only reads finished JSON artifacts and MemoryStore.
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import sys
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import streamlit as st
import yaml
import unicodedata
from pathlib import Path

from typing import List

# Optional auto-refresh component (preferred over sleep+rereun if installed)
try:  # pragma: no cover
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import]
except Exception:  # pragma: no cover
    st_autorefresh = None  # type: ignore[assignment]

def tail_lines(path: Path, max_lines: int = 200) -> List[str]:
    """Return the last ``max_lines`` lines from a text file.

    Reads the entire file into memory in order to retrieve the tail
    efficiently for moderately sized event logs.  Lines are returned
    without trailing newline characters.  On any error (file missing,
    decode error, etc.), an empty list is returned.

    Args:
        path: Filesystem path to the file to tail.
        max_lines: Maximum number of lines to return. Defaults to 200.

    Returns:
        A list of the last ``max_lines`` lines, in order, stripped of
        trailing newline characters.
    """
    try:
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists() or not path.is_file():
            return []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # Strip trailing newlines and take the tail
        tail = [l.rstrip("\n") for l in lines[-max_lines:]]
        return tail
    except Exception:
        return []

# IMPORTANT: st.set_page_config must be the FIRST Streamlit command executed
# (cached decorators count as Streamlit commands). Keep this at module top level.
# (comment trimmed to keep this file renderable in GitHub)
# (comment trimmed to keep this file renderable in GitHub)
st.set_page_config(page_title="ARA powered by Reparodynamics", page_icon="ÃÂÃÂÃÂÃÂ°ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂ ", layout="wide")

# Ensure repository root is on sys.path so imports work on Render and local
# This is robust whether this file lives in repo root or in a subfolder (for example app/)
_THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_FILE_DIR
if not (REPO_ROOT / "agent").is_dir() and (_THIS_FILE_DIR.parent / "agent").is_dir():
    REPO_ROOT = _THIS_FILE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_ui_asset_lines(rel_path: str) -> List[str]:
    """Load small text assets shipped with the repo (one item per line)."""
    try:
        p = (REPO_ROOT / rel_path).resolve()
        if not p.exists() or not p.is_file():
            return []
        out: List[str] = []
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
        return out
    except Exception:
        return []


# These will be filled from run_jobs imports when available
RUNS_BASE_DIR: Optional[Path] = None
RUNS_QUEUE_ROOT: Optional[Path] = None
RUNS_PENDING_DIR: Optional[Path] = None
RUNS_ACTIVE_DIR: Optional[Path] = None
RUNS_FINISHED_DIR: Optional[Path] = None
RUNS_ERROR_DIR: Optional[Path] = None


def _normalize_event_container(obj: Any) -> List[Dict[str, Any]]:
    """Normalize common event-log container shapes to a flat list of event dicts."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        # Common shapes: {"events": [...]}, {"log": [...]}, {"items": [...]}
        for k in ("events", "log", "items"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _load_event_dicts_tail(path: Union[str, Path], *, max_lines: int = 4000) -> List[Dict[str, Any]]:
    """Best-effort loader for event logs (JSONL or JSON) without reading entire files."""
    try:
        p = Path(path)
    except Exception:
        return []
    if not p.exists() or not p.is_file():
        return []

    # Prefer full JSON parse for small files; otherwise tail-parse to stay fast.
    try:
        size = p.stat().st_size
    except Exception:
        size = None

    if size is not None and size <= 2_000_000:
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            events = _normalize_event_container(obj)
            if events:
                return events
        except Exception:
            pass

    events: List[Dict[str, Any]] = []
    for line in tail_lines(p, max_lines=max_lines):
        s = line.strip()
        if not s:
            continue
        if s[0] not in "{[":
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        events.extend(_normalize_event_container(obj))
    return events


def _infer_cycle_progress_from_events(
    events: List[Dict[str, Any]],
    *,
    fallback_total: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Infer (current_cycle_1_based, total_cycles) from a list of event dicts."""
    if not events:
        return (None, fallback_total)

    def _as_int(v: Any) -> Optional[int]:
        try:
            if isinstance(v, bool):
                return None
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return int(v)
            if isinstance(v, str):
                s = v.strip()
                if s.isdigit():
                    return int(s)
        except Exception:
            return None
        return None

    saw_zero = False
    max_cycle_raw: Optional[int] = None
    best_total: Optional[int] = None
    best_progress_current: Optional[int] = None
    best_progress_total: Optional[int] = None

    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = str(ev.get("kind") or ev.get("event") or "").lower()
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}

        # Explicit cycle progress events with current/total
        if kind in {"cycle_progress", "progress_cycle", "cycle"} and isinstance(data, dict):
            cur_v = _as_int(data.get("current")) or _as_int(data.get("current_cycle"))
            tot_v = _as_int(data.get("total")) or _as_int(data.get("total_cycles"))
            if cur_v is not None:
                best_progress_current = cur_v
            if tot_v is not None and tot_v > 0:
                best_progress_total = tot_v

        # Total hints (event-level or data-level)
        for k in ("total_cycles", "total", "cycles_total", "max_cycles", "n_cycles"):
            tot_v = _as_int(ev.get(k))
            if tot_v is None and isinstance(data, dict):
                tot_v = _as_int(data.get(k))
            if tot_v is not None and tot_v > 0 and tot_v <= 100000:
                if best_total is None or tot_v > best_total:
                    best_total = tot_v

        # Cycle hints
        cycle_v = _as_int(ev.get("cycle"))
        if cycle_v is None and isinstance(data, dict):
            cycle_v = _as_int(data.get("cycle"))
        if cycle_v is None and isinstance(data, dict):
            cycle_v = _as_int(data.get("current_cycle"))
        if cycle_v is None and isinstance(data, dict) and kind in {"job_running", "progress", "agent_progress"}:
            cycle_v = _as_int(data.get("current"))

        if cycle_v is not None and 0 <= cycle_v <= 100000:
            if cycle_v == 0:
                saw_zero = True
            if max_cycle_raw is None or cycle_v > max_cycle_raw:
                max_cycle_raw = cycle_v

    total = best_progress_total or best_total or fallback_total
    cur_raw = best_progress_current if best_progress_current is not None else max_cycle_raw

    if cur_raw is None:
        return (None, total)

    # Convert to 1-based for display.
    cur_1_based = cur_raw
    if total is not None and total > 0:
        if saw_zero:
            cur_1_based = cur_raw + 1
        else:
            if 0 <= cur_raw <= total - 1:
                cur_1_based = cur_raw + 1
            elif 1 <= cur_raw <= total:
                cur_1_based = cur_raw
    else:
        if saw_zero:
            cur_1_based = cur_raw + 1

    return (cur_1_based, total)


def _infer_cycle_progress_from_event_logs(
    base_dir: Union[str, Path],
    run_id: str,
    *,
    fallback_total: Optional[int] = None,
    max_lines: int = 4000,
) -> Tuple[Optional[int], Optional[int]]:
    """Try multiple known event-log paths and infer cycle progress."""
    try:
        candidates = _candidate_state_paths(base_dir, run_id)
        paths = list(candidates.get("event_log", [])) + list(candidates.get("event_log_jsonl", []))
    except Exception:
        paths = []

    events: List[Dict[str, Any]] = []
    seen: set = set()
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        events.extend(_load_event_dicts_tail(p, max_lines=max_lines))
        if len(events) >= 2000:
            break

    # IMPORTANT: Some deployments only write a shared/global events log (e.g. logs/events_global.jsonl).
    # If we infer progress from that file without filtering, the UI will "replay history" from other runs.
    if run_id:
        rid = str(run_id)

        def _event_run_id(ev: Any) -> Optional[str]:
            if not isinstance(ev, dict):
                return None
            # Common placements
            r = ev.get("run_id")
            if r:
                return str(r)
            data = ev.get("data")
            if isinstance(data, dict) and data.get("run_id"):
                return str(data.get("run_id"))
            extra = ev.get("extra")
            if isinstance(extra, dict) and extra.get("run_id"):
                return str(extra.get("run_id"))
            return None

        events = [ev for ev in events if _event_run_id(ev) == rid]

    return _infer_cycle_progress_from_events(events, fallback_total=fallback_total)

def _to_ascii(text: Any) -> str:
    """
    Normalize and strip all non-ASCII characters from a string.

    This helper will take arbitrary input, convert it to a string and then
    normalize unicode characters into their closest ASCII equivalents. It then
    encodes the result to ASCII and ignores any characters that cannot be
    represented. This prevents mojibake from leaking into the UI when runtime
    data contains invalid or double-encoded sequences.
    """
    try:
        s = str(text) if text is not None else ""
        normalized = unicodedata.normalize("NFKD", s)
        ascii_bytes = normalized.encode("ascii", errors="ignore")
        return ascii_bytes.decode("ascii")
    except Exception:
        return str(text) if text is not None else ""

# -----------------------------------------------------------------------------
# History helpers
# -----------------------------------------------------------------------------

def _get_cycle_history_for_run(history: List[Dict[str, Any]], run_id: Optional[str]) -> List[Dict[str, Any]]:
    """
    Filter a cycle history list to only include entries belonging to a specific
    run_id.  Cycle entries are expected to be dictionaries that may include
    identifiers such as "run_id" or "id".  When run_id is None or when no
    entries match, the original history is returned.  This allows the UI to
    display per-run histories instead of mixing cycles from multiple runs.

    Args:
        history: A list of cycle entry dictionaries.
        run_id: The run identifier to filter by.

    Returns:
        A filtered list containing only entries whose run_id/id matches the
        specified run_id, or the original history when no match is found.
    """
    if not run_id or not history:
        return history
    filtered: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        # Common keys that might contain the run identifier
        entry_run_id = entry.get("run_id") or entry.get("id") or entry.get("job_id")
        if entry_run_id is not None and str(entry_run_id) == str(run_id):
            filtered.append(entry)
    # If matches were found, return them; otherwise return the original history
    return filtered if filtered else history

# Optional: run_jobs result loader (covers legacy filenames too)
_queue_load_job_result = None  # type: ignore[assignment]

# Optional: report generators (some deployments omit these; UI will fall back to history-based reports)
generate_report = None  # type: ignore[assignment]
generate_report_pdf = None  # type: ignore[assignment]
generate_findings_report = None  # type: ignore[assignment]
generate_findings_report_pdf = None  # type: ignore[assignment]

# -------------------------------------------------------------------
# Imports: prefer package layout agent.*, guarded flat fallback
# -------------------------------------------------------------------
try:
    # Package layout (recommended, what you have on Render)
    from agent.memory_store import MemoryStore
    from agent.presets import PRESETS, RUNTIME_PROFILES, get_preset

    # Report generator is optional; degrade gracefully if missing
    try:
        from agent.report_generator import (  # type: ignore[import]
            generate_findings_report,
            generate_findings_report_pdf,
            generate_report,
            generate_report_pdf,
        )
    except Exception:  # pragma: no cover
        generate_report = None  # type: ignore[assignment]
        generate_report_pdf = None  # type: ignore[assignment]
        generate_findings_report = None  # type: ignore[assignment]
        generate_findings_report_pdf = None  # type: ignore[assignment]

    # Only import the rye_metrics symbols that are actually used here
    from agent.rye_metrics import (
        autonomy_safety_envelope,
        breakthrough_likelihood_90d,
        build_run_diagnostics,
        classify_run_tier,
        detect_rye_equilibrium,
        early_failure_warning_score,
        estimate_breakthrough_probability,
        rye_volatility_signature,
        tgrm_harmonic_index,
    )

    # Optional discovery and verification helpers (imported lazily if present)
    try:  # type: ignore[import]
        from agent import discovery_log as _discovery_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _discovery_module = None  # type: ignore[assignment]

    try:  # type: ignore[import]
        from agent import verification_engine as _verification_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _verification_module = None  # type: ignore[assignment]

    try:  # type: ignore[import]
        from agent import hypothesis_manager as _hypo_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _hypo_module = None  # type: ignore[assignment]

    try:  # type: ignore[import]
        from agent import memory_pruner as _pruner_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _pruner_module = None  # type: ignore[assignment]

    # Optional MSIL meta skill intelligence layer
    try:  # type: ignore[import]
        from agent import msil as _msil_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _msil_module = None  # type: ignore[assignment]

    # Optional tools registry (for web browser and sandbox status)
    try:
        from agent.tools import TOOL_REGISTRY  # type: ignore[import]
    except Exception:  # pragma: no cover
        TOOL_REGISTRY = {}  # type: ignore[assignment]

    # Job queue abstraction: import paths from run_jobs so UI and worker share the exact same directories
    try:
        from agent.run_jobs import (  # type: ignore[import]
            ACTIVE_DIR as RUNS_ACTIVE_DIR,
            BASE_DIR as RUNS_BASE_DIR,
            ERROR_DIR as RUNS_ERROR_DIR,
            FINISHED_DIR as RUNS_FINISHED_DIR,
            PENDING_DIR as RUNS_PENDING_DIR,
            QUEUE_ROOT as RUNS_QUEUE_ROOT,
            create_job,
            list_jobs as list_run_jobs,
            load_job_result as _queue_load_job_result,
            result_path,
        )
    except Exception:
        create_job = None  # type: ignore[assignment]
        list_run_jobs = None  # type: ignore[assignment]
        result_path = None  # type: ignore[assignment]
        RUNS_BASE_DIR = None
        RUNS_QUEUE_ROOT = None
        RUNS_PENDING_DIR = None
        RUNS_ACTIVE_DIR = None
        RUNS_FINISHED_DIR = None
        RUNS_ERROR_DIR = None
        _queue_load_job_result = None  # type: ignore[assignment]

except ModuleNotFoundError as e:
    # If the *agent package itself* is missing, allow flat layout fallback.
    # If a submodule or dependency is missing, re-raise so we see the real error.
    missing_name = getattr(e, "name", None)
    if missing_name not in (None, "agent"):
        raise
    if missing_name is None:
        msg = str(e)
        if (
            "No module named 'agent'" not in msg
            and 'No module named "agent"' not in msg
            and "No module named agent" not in msg
        ):
            raise

    # Flat layout fallback: all modules live next to this file
    # Import memory_store components if available; otherwise define a stub.  Defining
    # `MemoryStore` as `None` allows the UI to detect the absence of a working
    # memory implementation and degrade gracefully.
    try:
        from memory_store import MemoryStore
    except Exception:
        MemoryStore = None  # type: ignore[assignment]

    # Import preset helpers if available; otherwise fall back to empty defaults.
    try:
        from presets import PRESETS, RUNTIME_PROFILES, get_preset  # type: ignore[no-redef]
    except Exception:
        PRESETS = {}  # type: ignore[assignment]
        RUNTIME_PROFILES = {}  # type: ignore[assignment]
        def get_preset(name: str, default: Any = None) -> Any:  # type: ignore[no-redef]
            return default

    # Report generator is optional; degrade gracefully if missing
    try:
        from report_generator import (  # type: ignore[no-redef]
            generate_findings_report,
            generate_findings_report_pdf,
            generate_report,
            generate_report_pdf,
        )
    except Exception:  # pragma: no cover
        generate_report = None  # type: ignore[assignment]
        generate_report_pdf = None  # type: ignore[assignment]
        generate_findings_report = None  # type: ignore[assignment]
        generate_findings_report_pdf = None  # type: ignore[assignment]

    # Import RYE metrics helpers if available; otherwise fall back to safe stubs.  These
    # functions are used in try/except blocks throughout the UI. Returning an empty
    # dictionary or None preserves behaviour when the metrics module is absent.
    try:  # type: ignore[no-redef]
        from rye_metrics import (  # type: ignore[no-redef]
            autonomy_safety_envelope,
            breakthrough_likelihood_90d,
            build_run_diagnostics,
            classify_run_tier,
            detect_rye_equilibrium,
            early_failure_warning_score,
            estimate_breakthrough_probability,
            rye_volatility_signature,
            tgrm_harmonic_index,
        )
    except Exception:
        def autonomy_safety_envelope(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def breakthrough_likelihood_90d(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def build_run_diagnostics(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def classify_run_tier(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def detect_rye_equilibrium(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def early_failure_warning_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def estimate_breakthrough_probability(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def rye_volatility_signature(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}
        def tgrm_harmonic_index(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {}

    try:  # type: ignore[import]
        import discovery_log as _discovery_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _discovery_module = None  # type: ignore[assignment]

    try:  # type: ignore[import]
        import verification_engine as _verification_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _verification_module = None  # type: ignore[assignment]

    try:  # type: ignore[import]
        import hypothesis_manager as _hypo_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _hypo_module = None  # type: ignore[assignment]

    try:  # type: ignore[import]
        import memory_pruner as _pruner_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _pruner_module = None  # type: ignore[assignment]

    # Optional MSIL meta skill intelligence layer
    try:  # type: ignore[import]
        import msil as _msil_module  # pragma: no cover
    except Exception:  # pragma: no cover
        _msil_module = None  # type: ignore[assignment]

    try:
        from tools import TOOL_REGISTRY  # type: ignore[import]
    except Exception:  # pragma: no cover
        TOOL_REGISTRY = {}  # type: ignore[assignment]

    # Flat layout run_jobs fallback (also import paths)
    try:
        from run_jobs import (  # type: ignore[no-redef]
            ACTIVE_DIR as RUNS_ACTIVE_DIR,
            BASE_DIR as RUNS_BASE_DIR,
            ERROR_DIR as RUNS_ERROR_DIR,
            FINISHED_DIR as RUNS_FINISHED_DIR,
            PENDING_DIR as RUNS_PENDING_DIR,
            QUEUE_ROOT as RUNS_QUEUE_ROOT,
            create_job,
            list_jobs as list_run_jobs,
            load_job_result as _queue_load_job_result,
            result_path,
        )
    except Exception:
        create_job = None  # type: ignore[assignment]
        list_run_jobs = None  # type: ignore[assignment]
        result_path = None  # type: ignore[assignment]
        RUNS_BASE_DIR = None
        RUNS_QUEUE_ROOT = None
        RUNS_PENDING_DIR = None
        RUNS_ACTIVE_DIR = None
        RUNS_FINISHED_DIR = None
        RUNS_ERROR_DIR = None
        _queue_load_job_result = None  # type: ignore[assignment]


def _coerce_path(p: Any) -> Optional[Path]:
    """Coerce run_jobs exported paths (Path/str/PathLike) into Path, or None."""
    if p is None:
        return None
    if isinstance(p, Path):
        return p
    if isinstance(p, str):
        return Path(p)
    try:
        return Path(os.fspath(p))
    except Exception:
        return None


# Normalize run_jobs path constants (they may be Path OR str depending on deployment)
RUNS_BASE_DIR = _coerce_path(RUNS_BASE_DIR)
RUNS_QUEUE_ROOT = _coerce_path(RUNS_QUEUE_ROOT)
RUNS_PENDING_DIR = _coerce_path(RUNS_PENDING_DIR)
RUNS_ACTIVE_DIR = _coerce_path(RUNS_ACTIVE_DIR)
RUNS_FINISHED_DIR = _coerce_path(RUNS_FINISHED_DIR)
RUNS_ERROR_DIR = _coerce_path(RUNS_ERROR_DIR)

# Use absolute path for default config relative to repo root
CONFIG_PATH_DEFAULT = str(REPO_ROOT / "config" / "settings.yaml")

# Rough estimate for cycles per hour in continuous mode.
# Used historically, now only advisory metadata handed to the worker.
CYCLES_PER_HOUR_ESTIMATE = 120

# Swarm roles: base archetypes for mini agents
SWARM_ROLES: List[Tuple[str, str]] = [
    ("researcher", "Deep literature and web researcher"),
    ("critic", "Methodology critic and refiner"),
    ("explorer", "Out of distribution explorer (new angles, analogies)"),
    ("theorist", "Model builder and unifier"),
    ("integrator", "Synthesizer that integrates and summarizes"),
]

# Safe upper bound for swarm size on typical Render or Streamlit setups.
# All swarm agents are still run sequentially in a single process by the worker.
MAX_SWARM_AGENTS: int = 64

# Limit points in charts so the frontend does not hit RangeError on very long runs.
MAX_POINTS_FOR_CHARTS: int = 1000

# Live console defaults
LIVE_EVENTS_LIMIT: int = 30
DISCOVERY_CARDS_LIMIT: int = 6


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _parse_timestamp_str(ts: str) -> Optional[datetime]:
    """Parse ISO style timestamps, including those with a trailing Z.

    Returns a naive UTC datetime when possible:
    - "2025-01-01T00:00:00Z" -> naive UTC
    - "2025-01-01T00:00:00+00:00" -> converted to UTC then tzinfo stripped
    """
    if not isinstance(ts, str):
        return None
    ts = ts.strip()
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1]
            dt = datetime.fromisoformat(ts)
            # Z indicates UTC; keep naive UTC representation
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _event_ts_to_dt(ts_val: Any) -> Optional[datetime]:
    """Convert event timestamps to a naive UTC datetime.

    Accepts:
    - ISO 8601 strings (with or without a trailing 'Z' / timezone offset)
    - epoch seconds (int/float) or epoch milliseconds
    - datetime objects

    Returns:
    - naive datetime in UTC, or None if not parseable
    """
    if ts_val is None:
        return None

    # Already a datetime
    if isinstance(ts_val, datetime):
        dt = ts_val
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    # Epoch seconds / milliseconds
    if isinstance(ts_val, (int, float)) and not isinstance(ts_val, bool):
        try:
            v = float(ts_val)
            # Heuristic: anything above ~1e11 is almost certainly epoch ms
            if v > 1e11:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None

    # Strings: numeric epoch or ISO
    if isinstance(ts_val, str):
        s = ts_val.strip()
        if not s:
            return None

        # Numeric epoch encoded as a string
        try:
            v = float(s)
            if v > 1e11:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            pass

        dt = _parse_timestamp_str(s)
        if dt is not None:
            return dt

    return None

def _coalesce(*values: Any) -> Any:
    """Return the first value that is not None (preserves 0/False)."""
    for v in values:
        if v is not None:
            return v
    return None


def _maybe_float(v: Any) -> Optional[float]:
    """Best-effort convert to float. Returns None if not convertible."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    """Best-effort convert to int."""
    if v is None:
        return default
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return default
            return int(float(s))
        return int(v)
    except Exception:
        return default


def _clamp_float(x: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    if x is None:
        return None
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return None


def _humanize_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    try:
        s = float(seconds)
    except Exception:
        return "n/a"
    if s < 0:
        s = 0.0
    if s < 60:
        return f"{s:.0f}s"
    m = s / 60.0
    if m < 60:
        return f"{m:.0f}m"
    h = m / 60.0
    if h < 24:
        return f"{h:.1f}h"
    d = h / 24.0
    return f"{d:.1f}d"



def _abbrev_id(value: Any, head: int = 8, tail: int = 4) -> str:
    """Abbreviate a long identifier for compact UIs (especially mobile)."""
    if value is None:
        return ""
    s = str(value)
    if len(s) <= head + tail + 1:
        return s
    # Use simple dots instead of a unicode ellipsis to avoid encoding issues
    return f"{s[:head]}...{s[-tail:]}"
def _format_metric_value(v: Any, decimals: int = 3) -> str:
    """Format a metric value safely for st.metric."""
    num = _maybe_float(v)
    if num is not None:
        try:
            return f"{num:.{decimals}f}"
        except Exception:
            return str(num)
    if v is None:
        return "n/a"
    return str(v)


def load_settings(config_path: str = CONFIG_PATH_DEFAULT) -> Dict[str, Any]:
    """Load YAML settings file into a dictionary."""
    # Allow env override (useful on platforms like Render)
    env_path = os.getenv("ARA_SETTINGS_PATH")
    if env_path:
        config_path = env_path

    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_runs_root() -> str:
    """Return the base directory used for ARA run artifacts.

    Primary source is run_jobs.BASE_DIR so UI and worker are always in sync.
    If that is not available, fall back to ARA_RUNS_DIR or <repo_root>/runs.
    """
    if isinstance(RUNS_BASE_DIR, Path):
        return str(RUNS_BASE_DIR)
    root = os.getenv("ARA_RUNS_DIR")
    if root:
        return root
    return str(REPO_ROOT / "runs")


def get_queue_root() -> str:
    """Return the canonical queue root directory used for job files.

    Primary source is run_jobs.QUEUE_ROOT (or run_jobs.PENDING_DIR parent).
    Falls back to ARA_QUEUE_ROOT or <runs_root>/queue.
    """
    if isinstance(RUNS_QUEUE_ROOT, Path):
        return str(RUNS_QUEUE_ROOT)
    if isinstance(RUNS_PENDING_DIR, Path):
        try:
            return str(RUNS_PENDING_DIR.parent)
        except Exception:
            pass
    env = os.getenv("ARA_QUEUE_ROOT")
    if env:
        return env
    return str(Path(get_runs_root()) / "queue")


def ensure_directories() -> None:
    """Ensure that log directories exist.

    Creates both:
    - Repo-local logs (useful for local dev)
    - Shared runs-root logs (preferred for worker/UI shared artifacts)
    """
    # Repo-local (legacy / dev convenience)
    repo_logs_path = REPO_ROOT / "logs"
    repo_sessions_path = repo_logs_path / "sessions"
    repo_logs_path.mkdir(parents=True, exist_ok=True)
    repo_sessions_path.mkdir(parents=True, exist_ok=True)

    # Shared runs-root logs (recommended)
    try:
        runs_logs_path = Path(get_runs_root()) / "logs"
        runs_sessions_path = runs_logs_path / "sessions"
        runs_logs_path.mkdir(parents=True, exist_ok=True)
        runs_sessions_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Non-fatal (e.g. permission issues in constrained environments)
        pass


@st.cache_resource
def init_memory_store(config_path: str = CONFIG_PATH_DEFAULT) -> MemoryStore:
    """Create a single MemoryStore instance for the Streamlit app (read only).

    IMPORTANT:
    Prefer a MemoryStore file under the runs root so the UI and the engine worker
    naturally share the same state/cycle history on disk.
    """
    # If there is no MemoryStore implementation available, return None.  This allows
    # the remainder of the UI to handle missing memory gracefully without crashing.
    if MemoryStore is None:  # type: ignore[comparison-overlap]
        return None  # type: ignore[return-value]

    ensure_directories()
    config = load_settings(config_path)

    # Default relative location (resolved below)
    memory_file_cfg = config.get("memory_file", "logs/sessions/default_memory.json")

    # Resolve memory path:
    # - absolute paths are used as-is
    # - relative paths prefer runs-root (shared), but keep backward-compat by using
    #   an existing repo-root file if that is where the current memory lives.
    if isinstance(memory_file_cfg, str) and os.path.isabs(memory_file_cfg):
        resolved = Path(memory_file_cfg)
    else:
        rel = str(memory_file_cfg) if memory_file_cfg is not None else "logs/sessions/default_memory.json"
        runs_candidate = Path(get_runs_root()) / rel
        repo_candidate = REPO_ROOT / rel

        if runs_candidate.exists():
            resolved = runs_candidate
        elif repo_candidate.exists():
            resolved = repo_candidate
        else:
            # Prefer shared runs-root for new deployments
            resolved = runs_candidate

    # Ensure parent directory exists
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    memory = MemoryStore(str(resolved))
    return memory


def load_job_result(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a finished job result.

    Uses run_jobs.load_job_result when available (supports legacy result filenames),
    with a strict fallback to run_jobs.result_path when needed.
    """
    if callable(_queue_load_job_result):
        try:
            data = _queue_load_job_result(run_id)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    if "result_path" not in globals() or result_path is None:
        return None
    fp = result_path(run_id)
    if not fp.exists():
        return None
    try:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def tavily_status() -> Dict[str, Any]:
    """Check whether a Tavily API key is available (per user or env)."""
    # 1) Prefer per user key stored in session state (from sidebar input)
    key = st.session_state.get("tavily_key", None)
    if isinstance(key, str):
        key = key.strip()

    # 2) Fallback to environment variable (in case you set it on the server)
    if not key:
        key = os.getenv("TAVILY_API_KEY")
        if isinstance(key, str):
            key = key.strip()

    # 3) Optional final fallback to secrets (owner only use, can be empty)
    if not key:
        try:
            key = st.secrets.get("TAVILY_API_KEY", None)  # type: ignore[attr-defined]
            if isinstance(key, str):
                key = key.strip()
        except Exception:
            key = None

    if key:
        tail = key[-4:]
        return {"has_key": True, "display": f"Tavily key detected (...{tail})", "tail": tail}
    return {
        "has_key": False,
        "display": "No Tavily API key found. Web search will use stubbed results.",
        "tail": None,
    }


def detect_tools() -> Dict[str, bool]:
    """Detect presence of web browser and sandbox tools from TOOL_REGISTRY."""
    if not isinstance(TOOL_REGISTRY, dict):
        return {"web": False, "sandbox": False}

    # Flexible detection by common keys (case-insensitive)
    keys_lower = {str(k).lower() for k in TOOL_REGISTRY.keys()}
    web_keys = {"web_search", "browser", "web", "internet"}
    sandbox_keys = {"sandbox", "code_sandbox", "python_sandbox", "exec_sandbox"}

    has_web = any(k in keys_lower for k in web_keys)
    has_sandbox = any(k in keys_lower for k in sandbox_keys)

    return {"web": has_web, "sandbox": has_sandbox}


def render_cycle_summary(cycle_summary: Dict[str, Any]) -> None:
    """Pretty print cycle summary output."""
    role = cycle_summary.get("role", "agent")
    domain = cycle_summary.get("domain") or "general"

    # Respect the cycle index written by the worker (do not add 1 again)
    cycle_index = cycle_summary.get("cycle")
    if cycle_index is None:
        cycle_index = cycle_summary.get("cycle_index")
    if cycle_index is None:
        cycle_index = 1

    st.markdown(f"### Cycle {cycle_index} (role: {role}, domain: {domain})")

    # Metrics (safe handling for None / strings)
    delta_val = cycle_summary.get("delta_R")
    if delta_val is None:
        delta_val = cycle_summary.get("delta_r")

    energy_val = cycle_summary.get("energy_E")
    if energy_val is None:
        energy_val = cycle_summary.get("energy")

    rye_val = cycle_summary.get("RYE")
    if rye_val is None:
        rye_val = cycle_summary.get("rye")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("delta_R", _format_metric_value(delta_val, decimals=4))
    with col2:
        st.metric("Energy E", _format_metric_value(energy_val, decimals=4))
    with col3:
        st.metric("RYE", _format_metric_value(rye_val, decimals=3))

    # Issues
    issues_before = cycle_summary.get("issues_before", [])
    if issues_before:
        st.write("Issues before repair:")
        for issue in issues_before:
            st.write(f"- {issue}")
    else:
        st.write("No issues detected before repair.")

    # Repairs
    repairs = cycle_summary.get("repairs", [])
    if repairs:
        st.write("Repairs applied:")
        for rep in repairs:
            st.write(f"- {rep}")
    else:
        st.write("No repairs performed.")

    # Notes
    if cycle_summary.get("notes_added"):
        with st.expander("Notes added"):
            for note in cycle_summary["notes_added"]:
                st.write(f"- {note}")

    # Hypotheses
    hypotheses = cycle_summary.get("hypotheses") or []
    if hypotheses:
        with st.expander("Generated hypotheses"):
            for h in hypotheses:
                if isinstance(h, dict):
                    text = h.get("text", "")
                    conf = h.get("confidence")
                    if conf is not None:
                        st.write(f"- {text} (confidence ~ {conf})")
                    else:
                        st.write(f"- {text}")
                else:
                    st.write(f"- {h}")

    # Citations
    if cycle_summary.get("citations") or cycle_summary.get("sources") or cycle_summary.get("source_list"):
        with st.expander("Citations for this cycle"):
            cites = (
                cycle_summary.get("citations")
                or cycle_summary.get("sources")
                or cycle_summary.get("source_list")
                or []
            )
            for c in cites:
                if not isinstance(c, dict):
                    st.write(f"- {c}")
                    continue
                src = c.get("source", "") or c.get("provider", "")
                title = c.get("title", "")
                url = c.get("url", "") or c.get("link", "")
                st.write(f"- [{src}] {title} - {url}")


def build_swarm_roles(enabled: bool, swarm_size: int) -> List[Tuple[str, str]]:
    """Return the active swarm roles (name, description) given total swarm agents.

    If swarm_size <= len(SWARM_ROLES), we just take the first N base roles.
    If swarm_size > len(SWARM_ROLES), we create multiple agents per base role
    with role names like researcher_1, critic_2, etc.
    """
    if not enabled or swarm_size <= 1:
        return []

    total = max(1, min(swarm_size, MAX_SWARM_AGENTS))
    agents: List[Tuple[str, str]] = []

    for idx in range(total):
        base_role, base_desc = SWARM_ROLES[idx % len(SWARM_ROLES)]
        if total <= len(SWARM_ROLES):
            role_name = base_role
            desc = base_desc
        else:
            # Distinguish clones of the same archetype
            role_name = f"{base_role}_{idx + 1}"
            desc = f"{base_desc} (agent {idx + 1}/{total})"
        agents.append((role_name, desc))

    return agents


def role_specific_goal(base_goal: str, role: str) -> str:
    """Specialize the goal text slightly for each swarm role."""
    base_goal = base_goal.strip()

    # Strip any clone suffix like _3 so we map back to the archetype
    archetype = role.split("_", 1)[0] if "_" in role else role

    if archetype == "researcher":
        return (
            f"Primary deep research agent for goal: {base_goal}.\n"
            "Focus on high quality sources, detailed notes, and clear summaries."
        )
    if archetype == "critic":
        return (
            f"Critically review, cross check, and refine all existing Reparodynamic notes and hypotheses for: {base_goal}.\n"
            "Identify weaknesses, gaps, and overclaims."
        )
    if archetype == "explorer":
        return (
            f"Exploration agent for goal: {base_goal}.\n"
            "Look for unusual angles, analogies, adjacent fields, and surprising connections."
        )
    if archetype == "theorist":
        return (
            f"Theory building agent for goal: {base_goal}.\n"
            "Try to organize findings into coherent models, equations, or structured frameworks."
        )
    if archetype == "integrator":
        return (
            f"Integration agent for goal: {base_goal}.\n"
            "Synthesize results from all prior agents into clear narratives, tables, and distilled insights."
        )
    # Fallback: use the original goal
    return base_goal


def _get_job_id(job: Any) -> str:
    """Extract a run id or job id from RunJob or legacy dict."""
    if hasattr(job, "run_id"):
        return str(getattr(job, "run_id"))
    if isinstance(job, dict):
        return str(job.get("run_id") or job.get("job_id") or "unknown")
    return "unknown"


def _get_job_config(job: Any) -> Dict[str, Any]:
    """Get config dict from RunJob or legacy dict, safe default."""
    cfg = getattr(job, "config", None)
    if isinstance(job, dict):
        cfg = job.get("config", cfg)
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _get_job_meta(job: Any) -> Dict[str, Any]:
    """Get meta dict from RunJob or legacy dict, safe default."""
    meta = getattr(job, "meta", None)
    if isinstance(job, dict):
        meta = job.get("meta", meta)
    if not isinstance(meta, dict):
        return {}
    return meta


def _get_job_label(job: Any) -> str:
    """Human friendly label for a job."""
    cfg = _get_job_config(job)
    meta = _get_job_meta(job)
    label = (
        meta.get("run_label")
        or meta.get("label")
        or cfg.get("notes")
        or cfg.get("goal")
        or _get_job_id(job)
    )
    return str(label)


def _job_created_at_ts(job: Any) -> float:
    """Best-effort numeric timestamp for sorting jobs."""
    ts_raw = None
    if hasattr(job, "created_at"):
        ts_raw = getattr(job, "created_at", None)
    elif isinstance(job, dict):
        ts_raw = job.get("created_at")

    if isinstance(ts_raw, (int, float)):
        return float(ts_raw)

    if isinstance(ts_raw, str):
        dt = _parse_timestamp_str(ts_raw)
        if dt is not None:
            try:
                return dt.timestamp()
            except Exception:
                return 0.0
        # Last-ditch parse
        try:
            return datetime.fromisoformat(ts_raw.replace("Z", "")).timestamp()
        except Exception:
            return 0.0

    return 0.0


def render_job_summary(job: Any) -> None:
    """Compact header row for a job (queued or finished)."""
    job_id = _get_job_id(job)
    if hasattr(job, "status"):
        status = getattr(job, "status", "unknown")
        created_at_raw = getattr(job, "created_at", None)
    elif isinstance(job, dict):
        status = job.get("status", "unknown")
        created_at_raw = job.get("created_at")
    else:
        status = "unknown"
        created_at_raw = None

    config = _get_job_config(job)
    domain = config.get("domain", "general")

    if isinstance(domain, dict):
        domain = domain.get("tag") or domain.get("name") or "general"

    # Mode detection from config
    mode = config.get("mode")
    swarm_cfg = config.get("swarm_config") or config.get("swarm") or {}
    if not mode:
        if swarm_cfg.get("swarm_size", 1) and swarm_cfg.get("swarm_size", 1) > 1:
            mode = "swarm"
        elif config.get("multi_agent_pair"):
            mode = "two_stage"
        else:
            mode = "single"

    label = _get_job_label(job)

    if isinstance(created_at_raw, (int, float)):
        created_at = datetime.utcfromtimestamp(created_at_raw).isoformat() + "Z"
    else:
        created_at = str(created_at_raw or "unknown")

    cols = st.columns([3, 2, 2, 2])
    cols[0].markdown(f"**{label}**")
    cols[1].markdown(f"Run: `{job_id}`")
    cols[2].markdown(f"Status: `{status}`")
    cols[3].markdown(f"Domain: `{str(domain).title()}`")

    # Replace misencoded bullet with a simple ASCII separator
    st.caption(f"Mode: {mode} | Created at: {created_at}")


# -------------------------------------------------------------------
# run_jobs compatibility wrappers
# -------------------------------------------------------------------
def _safe_list_jobs(status: Optional[str] = None) -> List[Any]:
    """List jobs with best-effort support for differing run_jobs.list_jobs signatures."""
    if list_run_jobs is None:
        return []
    if status is None:
        try:
            out = list_run_jobs()  # type: ignore[misc]
            return out if isinstance(out, list) else []
        except Exception:
            return []
    # Prefer keyword argument
    try:
        out = list_run_jobs(status=status)  # type: ignore[misc]
        return out if isinstance(out, list) else []
    except TypeError:
        # Positional fallback
        try:
            out = list_run_jobs(status)  # type: ignore[misc]
            return out if isinstance(out, list) else []
        except Exception:
            return []
    except Exception:
        return []


def _list_jobs_by_status_candidates(status_candidates: List[str]) -> List[Any]:
    """Union jobs across multiple status labels (dedup by run_id)."""
    seen: Set[str] = set()
    out: List[Any] = []
    for s in status_candidates:
        for job in _safe_list_jobs(status=s):
            jid = _get_job_id(job)
            if jid in seen:
                continue
            seen.add(jid)
            out.append(job)
    return out


def _safe_create_job(config: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Create job with best-effort support for differing run_jobs.create_job signatures."""
    if create_job is None:
        return None

    # Prefer keyword signature (most common)
    try:
        rid = create_job(config=config, meta=meta)  # type: ignore[misc]
        return str(rid) if rid is not None else None
    except TypeError:
        pass
    except Exception:
        return None

    # Positional fallbacks
    try:
        rid = create_job(config, meta)  # type: ignore[misc]
        return str(rid) if rid is not None else None
    except TypeError:
        pass
    except Exception:
        return None

    # Minimal fallback: config only
    try:
        rid = create_job(config)  # type: ignore[misc]
        return str(rid) if rid is not None else None
    except Exception:
        return None


# -------------------------------------------------------------------
# Run-result -> cycle-history helpers (used for citation table fallback)
# -------------------------------------------------------------------
def _extract_cycles_from_run_result(
    run_result: Dict[str, Any],
    run_id: Optional[str] = None,
    default_timestamp: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Normalize cycles from a finished run result into history style entries.

    This is used as a fallback when MemoryStore.get_cycle_history() is empty,
    so the History and Citations panels can still populate directly from
    finished run JSON files written by the engine worker.

    default_timestamp:
        A best effort timestamp for the run, for example job.created_at.
        Used when no per cycle or run level timestamp is present.
    """
    if not isinstance(run_result, dict):
        return []

    payload = run_result.get("result")
    if isinstance(payload, dict):
        base = payload
    else:
        base = run_result

    cfg = run_result.get("config") if isinstance(run_result.get("config"), dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    goal = base.get("goal") or cfg.get("goal")
    domain = base.get("domain") or cfg.get("domain") or "general"
    default_role = base.get("role") or "agent"

    # Normalize a usable default timestamp string up front
    normalized_default_ts: Optional[str] = None
    ts_candidate = base.get("timestamp") or run_result.get("timestamp") or default_timestamp
    if isinstance(ts_candidate, (int, float)):
        try:
            normalized_default_ts = datetime.utcfromtimestamp(ts_candidate).isoformat() + "Z"
        except Exception:
            normalized_default_ts = None
    elif isinstance(ts_candidate, str):
        normalized_default_ts = ts_candidate

    # Find the first key that looks like cycle history
    cycles_raw: Optional[List[Dict[str, Any]]] = None
    for key in ("cycles", "cycle_history", "history", "tgrm_history", "run_history", "per_cycle"):
        val = base.get(key)
        if isinstance(val, list) and val:
            cycles_raw = [c for c in val if isinstance(c, dict)]
            if cycles_raw:
                break

    if not cycles_raw:
        return []

    cycles_out: List[Dict[str, Any]] = []
    for idx, c in enumerate(cycles_raw):
        c2: Dict[str, Any] = dict(c)

        # Normalize cycle number
        if c2.get("cycle") is None and c2.get("cycle_index") is not None:
            c2["cycle"] = c2.get("cycle_index")
        if c2.get("cycle") is None:
            c2["cycle"] = idx + 1

        # Domain, role, goal fallbacks
        if not c2.get("domain"):
            c2["domain"] = domain
        if not c2.get("role"):
            c2["role"] = default_role
        if goal is not None and "goal" not in c2:
            c2["goal"] = goal

        # Timestamp fallback
        ts_val = c2.get("timestamp")
        if ts_val is None or ts_val == "":
            ts_val = normalized_default_ts
        if isinstance(ts_val, (int, float)):
            try:
                ts_val = datetime.utcfromtimestamp(ts_val).isoformat() + "Z"
            except Exception:
                ts_val = None
        if ts_val is not None:
            c2["timestamp"] = ts_val

        # Attach run id if we have it
        if run_id is not None and "run_id" not in c2:
            c2["run_id"] = run_id

        cycles_out.append(c2)

    return cycles_out


def load_history_from_finished_runs(limit_runs: int = 20) -> List[Dict[str, Any]]:
    """Rebuild a synthetic history from finished run JSON files.

    This is a fallback for History and Citations when MemoryStore has no
    cycle history, for example when the queue mode worker only writes
    per run result JSON and does not stream cycles into MemoryStore.
    """
    jobs = _list_jobs_by_status_candidates(["finished", "done", "completed", "complete", "success"])

    if not jobs:
        return []

    # Sort jobs by created_at so we can take the most recent N
    try:
        jobs_sorted = sorted(jobs, key=_job_created_at_ts)
    except Exception:
        jobs_sorted = jobs

    # Only look at the most recent N jobs to keep things light
    jobs_slice = jobs_sorted[-limit_runs:]

    history: List[Dict[str, Any]] = []
    for job in jobs_slice:
        run_id = _get_job_id(job)

        # Best effort created_at timestamp from the job header
        created_at_raw = None
        if hasattr(job, "created_at"):
            created_at_raw = getattr(job, "created_at", None)
        elif isinstance(job, dict):
            created_at_raw = job.get("created_at")

        default_ts = None
        if isinstance(created_at_raw, (int, float)):
            try:
                default_ts = datetime.utcfromtimestamp(created_at_raw).isoformat() + "Z"
            except Exception:
                default_ts = None
        elif isinstance(created_at_raw, str):
            default_ts = created_at_raw

        result = load_job_result(run_id)
        if not isinstance(result, dict):
            continue
        cycles = _extract_cycles_from_run_result(result, run_id=run_id, default_timestamp=default_ts)
        history.extend(cycles)

    if not history:
        return []

    # Sort by timestamp if present then by cycle
    def _sort_key(e: Dict[str, Any]):
        ts = e.get("timestamp")
        sort_ts = ts if isinstance(ts, str) else ""
        return sort_ts, int(e.get("cycle") or 0)

    history.sort(key=_sort_key)
    return history


def extract_unique_citations_from_history(
    history: List[Dict[str, Any]],
    max_items: int = 200,
) -> List[Dict[str, Any]]:
    """Flatten citations from a list of cycle history entries and deduplicate them."""
    if not history:
        return []

    collected: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[Optional[str], Optional[str]]] = set()

    for entry in history:
        for key in ("citations", "sources", "source_list"):
            items = entry.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("name") or ""
                    url = item.get("url") or item.get("link") or ""
                    provider = item.get("source") or item.get("provider") or ""
                    key_tuple = (title.strip() or None, url.strip() or None)
                    if key_tuple in seen_keys:
                        continue
                    seen_keys.add(key_tuple)
                    collected.append(
                        {
                            "title": title,
                            "url": url,
                            "provider": provider,
                            "source": provider,
                            "snippet": item.get("snippet") or item.get("summary") or "",
                        }
                    )
                else:
                    text = str(item).strip()
                    if not text:
                        continue
                    key_tuple = (text, None)
                    if key_tuple in seen_keys:
                        continue
                    seen_keys.add(key_tuple)
                    collected.append(
                        {
                            "title": text,
                            "url": "",
                            "provider": "",
                            "source": "",
                            "snippet": "",
                        }
                    )

                if len(collected) >= max_items:
                    return collected

    return collected


def render_result_details(result: Dict[str, Any]) -> None:
    """Safe read only result viewer for a finished job."""
    payload = result.get("result")
    base = payload if isinstance(payload, dict) else result

    st.markdown("### Run summary")

    summary = base.get("summary") or base.get("human_summary") or base.get("run_summary")
    if summary:
        st.write(summary)
    else:
        # Fallback: synthesize a lightweight summary from cycles/metrics when the engine didn't provide one.
        synthesized = ""
        try:
            if cycles:
                synthesized = build_outcome_summary(cycles)
        except Exception:
            synthesized = ""
        if synthesized:
            st.markdown(synthesized)
        else:
            st.info("No summary was provided by the engine.")

    key_findings = base.get("key_findings") or base.get("discoveries") or base.get("discovery_candidates")
    if isinstance(key_findings, list) and key_findings:
        st.markdown("#### Key findings and discovery candidates")
        for item in key_findings:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("summary") or item.get("title") or str(item)
            else:
                txt = str(item)
            st.markdown(f"- {txt}")

    rye_metrics = base.get("rye_metrics") or base.get("rye") or base.get("run_rye_metrics") or base.get("metrics")
    if isinstance(rye_metrics, dict):
        st.markdown("#### RYE metrics")
        cols = st.columns(3)
        avg_rye = rye_metrics.get("avg_rye") or rye_metrics.get("rye_avg")
        if isinstance(avg_rye, (int, float)):
            cols[0].metric("Average RYE", f"{avg_rye:.4f}")
        trend = rye_metrics.get("trend_slope")
        if isinstance(trend, (int, float)):
            cols[1].metric("RYE trend slope", f"{trend:.4f}")
        stability = rye_metrics.get("stability_index")
        if isinstance(stability, (int, float)):
            cols[2].metric("Stability index", f"{stability:.3f}")

    cycles: Optional[List[Dict[str, Any]]] = None
    for key in ("cycles", "cycle_history", "history", "tgrm_history", "run_history", "per_cycle"):
        val = base.get(key)
        if isinstance(val, list) and val:
            cycles = [c for c in val if isinstance(c, dict)]
            if cycles:
                break

    if cycles:
        st.markdown("#### Cycle timeline")

        cycle_numbers: List[Any] = []
        delta_r_values: List[Any] = []
        energy_values: List[Any] = []
        rye_values: List[Any] = []

        for idx, c in enumerate(cycles):
            c_num = c.get("cycle") or c.get("cycle_index") or (idx + 1)
            cycle_numbers.append(c_num)

            d_val = c.get("delta_r")
            if d_val is None:
                d_val = c.get("delta_R")
            delta_r_values.append(d_val)

            e_val = c.get("energy")
            if e_val is None:
                e_val = c.get("energy_E")
            energy_values.append(e_val)

            r_val = c.get("rye")
            if r_val is None:
                r_val = c.get("RYE")
            rye_values.append(r_val)

        chart_data: Dict[str, List[Any]] = {"cycle": cycle_numbers}
        if any(v is not None for v in delta_r_values):
            chart_data["delta_R"] = delta_r_values
        if any(v is not None for v in energy_values):
            chart_data["energy"] = energy_values
        if any(v is not None for v in rye_values):
            chart_data["RYE"] = rye_values

        if len(chart_data) > 1:
            df = pd.DataFrame(chart_data).set_index("cycle")
            # Ensure numeric columns (Streamlit charts can render blank when dtype becomes 'object').
            for _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce")
            st.line_chart(df)
            st.caption("Timeline of delta_R, energy, and RYE per cycle.")

        with st.expander("Per cycle details"):
            for c in cycles:
                render_cycle_summary(c)

    sources = base.get("sources") or base.get("citations") or base.get("source_list")
    if not sources and cycles:
        flattened_citations = extract_unique_citations_from_history(cycles)
        if flattened_citations:
            sources = flattened_citations

    if isinstance(sources, list) and sources:
        # Toggle to hide/show citations inline (this list can be long)
        rid_for_toggle = str(base.get("run_id") or result.get("run_id") or "")
        toggle_key = f"show_sources_inline__{rid_for_toggle}" if rid_for_toggle else "show_sources_inline"
        show_sources_inline = st.toggle(
            "Show sources and citations",
            value=bool(st.session_state.get(toggle_key, False)),
            key=toggle_key,
            help="Hide/show the live sources list embedded in the report. Use the Source citation viewer tab for full details.",
        )

        if show_sources_inline:
            st.markdown("#### Sources and citations")

            # De-dupe sources (Tavily errors and repeated URLs are common)
            deduped_sources: List[Any] = []
            seen: Set[Any] = set()
            for s in sources:
                # Skip any sources that clearly represent an error entry (e.g. Tavily errors)
                if isinstance(s, dict):
                    provider_val = str(s.get("source") or s.get("provider") or "").strip().lower()
                    title_val = str(s.get("title") or "").strip().lower()
                    # Filter out entries where provider or title indicates an error; allow stub entries to be shown
                    if provider_val == "error" or "error" in title_val:
                        continue
                    url = str(s.get("url") or s.get("link") or "").strip()
                    title = str(s.get("title") or "Source").strip()
                    provider = str(s.get("source") or s.get("provider") or "").strip()
                    key = (url or title, provider)
                else:
                    # Skip simple strings that are error messages
                    simple_val = str(s).strip()
                    if "error" in simple_val.lower():
                        continue
                    key = simple_val
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped_sources.append(s)

            if len(deduped_sources) != len(sources):
                st.caption(f"De-duplicated sources: showing {len(deduped_sources)} unique of {len(sources)} total.")

            for s in deduped_sources:
                if not isinstance(s, dict):
                    st.markdown(f"- {s}")
                    continue
                title = s.get("title", "Source")
                url = s.get("url") or s.get("link")
                snippet = s.get("snippet") or s.get("summary") or ""
                provider = s.get("source") or s.get("provider") or ""
                line = ""
                if provider:
                    line += f"[{provider}] "
                if url:
                    line += f"[{title}]({url})"
                else:
                    line += title
                if snippet:
                    line += f"  \n  {snippet}"
                st.markdown(f"- {line}")
        else:
            # When citations are hidden, do not render the section or caption at all.
            pass

    debug = base.get("debug") or base.get("diagnostics") or result.get("debug") or result.get("diagnostics")
    if debug:
        with st.expander("Diagnostics and debug"):
            st.json(debug)


# -------------------------------------------------------------------
# Outcome focused summary helper
# -------------------------------------------------------------------
def build_outcome_summary(history: List[Dict[str, Any]]) -> str:
    """Create a markdown summary focused on outcomes and run time."""
    if not history:
        return "# Outcome summary\n\nNo cycles have been recorded yet."

    total_cycles = len(history)
    roles = sorted({str(e.get("role", "agent")) for e in history})
    domains = sorted({str(e.get("domain", "general")) for e in history})

    timestamps: List[datetime] = []
    for e in history:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            dt = _parse_timestamp_str(ts)
            if dt is not None:
                timestamps.append(dt)

    runtime_text = "Runtime not available"
    if len(timestamps) >= 2:
        start = min(timestamps)
        end = max(timestamps)
        delta = end - start
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        runtime_text = f"Approx runtime: {hours} hours {minutes} minutes (from first to last cycle)."

    rye_vals: List[float] = []
    for e in history:
        v = e.get("RYE")
        if not isinstance(v, (int, float)):
            v = e.get("rye")
        if isinstance(v, (int, float)):
            rye_vals.append(float(v))

    rye_text = "RYE statistics not available."
    if rye_vals:
        avg_rye = sum(rye_vals) / len(rye_vals)
        rye_text = (
            "RYE statistics:\n"
            f"- Min RYE: {min(rye_vals):.3f}\n"
            f"- Max RYE: {max(rye_vals):.3f}\n"
            f"- Average RYE: {avg_rye:.3f}"
        )

    findings: List[str] = []
    for e in history:
        for n in (e.get("notes_added") or []):
            findings.append(str(n))
        for r in e.get("repairs") or []:
            findings.append(str(r))
        for h in e.get("hypotheses") or []:
            txt = h.get("text", "") if isinstance(h, dict) else str(h)
            if txt:
                findings.append(txt)

    seen: Set[str] = set()
    unique_findings: List[str] = []
    for f in findings:
        f_clean = f.strip()
        if not f_clean:
            continue
        if f_clean in seen:
            continue
        seen.add(f_clean)
        unique_findings.append(f_clean)
        if len(unique_findings) >= 80:
            break

    lines: List[str] = []
    lines.append("# Outcome summary\n")
    lines.append("## Run overview\n")
    lines.append(f"- Total cycles: {total_cycles}")
    lines.append(f"- Roles used: {', '.join(roles) if roles else 'None recorded'}")
    lines.append(f"- Domains used: {', '.join(domains) if domains else 'None recorded'}")
    lines.append(f"- {runtime_text}\n")
    lines.append("## RYE and efficiency\n")
    lines.append(rye_text + "\n")
    lines.append("## Candidate findings\n")
    if not unique_findings:
        lines.append("No candidate findings extracted from notes, repairs, or hypotheses.")
    else:
        lines.append(
            "Below are candidate interventions, mechanisms, treatments, or key ideas extracted "
            "from notes, repairs, and hypotheses. This is a raw list to review, not medical advice."
        )
        for f in unique_findings:
            if len(f) > 400:
                f = f[:400] + "..."
            lines.append(f"- {f}")

    return "\n".join(lines)


def build_findings_report_from_history(history: List[Dict[str, Any]]) -> str:
    """Build a findings style report directly from synthetic history."""
    if not history:
        return "# Findings Report\n\nNo cycles found."

    total_cycles = len(history)
    domains = sorted({str(e.get("domain", "general")) for e in history})
    roles = sorted({str(e.get("role", "agent")) for e in history})

    keywords = ["treatment", "therapy", "intervention", "protocol", "cure", "mechanism"]
    findings: List[Tuple[float, str, Dict[str, Any]]] = []

    def _score_text(text: str) -> float:
        text_low = text.lower()
        score = 0.0
        for kw in keywords:
            if kw in text_low:
                score += 1.0
        if len(text) < 200:
            score += 0.3
        return score

    for e in history:
        base_meta = {
            "cycle": e.get("cycle"),
            "role": e.get("role", "agent"),
            "domain": e.get("domain", "general"),
            "timestamp": e.get("timestamp"),
        }
        for field in ("notes_added", "repairs", "hypotheses"):
            items = e.get(field) or []
            for item in items:
                text = (item.get("text") or item.get("summary") or "") if isinstance(item, dict) else str(item)
                text = text.strip()
                if not text:
                    continue
                score = _score_text(text)
                if score <= 0.0 and len(text) > 260:
                    continue
                meta = dict(base_meta)
                meta["source_field"] = field
                findings.append((score, text, meta))

    findings.sort(key=lambda x: x[0], reverse=True)
    top_findings = findings[:80]

    lines: List[str] = []
    lines.append("# Findings Report\n")
    lines.append("This autonomous report lists candidate cures, treatments, and mechanisms.\n")
    lines.append("It is a research artifact only and not medical advice.\n")
    lines.append("## Run context\n")
    lines.append(f"- Total cycles scanned: {total_cycles}")
    lines.append(f"- Domains seen: {', '.join(domains) if domains else 'None recorded'}")
    lines.append(f"- Roles seen: {', '.join(roles) if roles else 'None recorded'}\n")

    if not top_findings:
        lines.append("No candidate cure or treatment style findings were extracted.\n")
        return "\n".join(lines)

    lines.append("## Candidate cures, treatments, and mechanisms\n")
    for score, text, meta in top_findings:
        cycle = meta.get("cycle")
        role = meta.get("role", "agent")
        domain = meta.get("domain", "general")
        field = meta.get("source_field", "notes")
        ts = meta.get("timestamp")
        header_parts = [f"[{domain}/{role}"]
        if cycle is not None:
            header_parts.append(f"cycle {cycle}")
        header = " ".join(header_parts) + f" from {field}]"
        t = text[:480] + "..." if len(text) > 480 else text
        if ts:
            lines.append(f"- {header} ({ts})")
        else:
            lines.append(f"- {header}")
        lines.append(f"  - {t}")

    return "\n".join(lines)


def compute_run_hours(history: List[Dict[str, Any]]) -> Optional[float]:
    """Approximate total hours between first and last cycle timestamps."""
    timestamps: List[datetime] = []
    for e in history:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            dt = _parse_timestamp_str(ts)
            if dt is not None:
                timestamps.append(dt)
    if len(timestamps) < 2:
        return None
    start = min(timestamps)
    end = max(timestamps)
    delta = end - start
    return max(delta.total_seconds() / 3600.0, 0.0)


def compute_msil_profile(
    history: List[Dict[str, Any]],
    goal: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call optional MSIL layer if available to compute meta skill intelligence profile."""
    if not history or _msil_module is None:
        return None

    if goal is None:
        goal = str(history[-1].get("goal") or "unknown_goal")

    try:
        analyze_run = getattr(_msil_module, "analyze_run", None)
        if callable(analyze_run):
            return analyze_run(history=history, goal=goal, config=None)

        layer_cls = getattr(_msil_module, "MetaSkillIntelligenceLayer", None)
        store_cls = getattr(_msil_module, "_HistoryBackedMemoryStore", None)
        if layer_cls is not None and store_cls is not None:
            store = store_cls(history)  # type: ignore[call-arg]
            layer = layer_cls(memory_store=store, config={})  # type: ignore[call-arg]
            return layer.summarise_run(goal=goal, run_id=None, limit=len(history))  # type: ignore[call-arg]
    except Exception:
        return None

    return None


# -------------------------------------------------------------------
# Advanced log and snapshot helpers
# -------------------------------------------------------------------
def _load_json_file(path: Path) -> Optional[Any]:
    """Small helper to load a JSON/JSONL file and return the decoded data.

    - For *.json: returns the decoded JSON value (dict/list/etc.)
    - For *.jsonl: returns a list of decoded JSON objects (one per line)
    """
    if not path.exists():
        return None
    try:
        # JSONL support (append-only event streams)
        if path.suffix.lower() == ".jsonl":
            try:
                txt = path.read_text(errors="ignore")
            except Exception:
                return None
            out: List[Dict[str, Any]] = []
            for line in txt.splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    obj = json.loads(line_s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
            return out

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_state_timestamp_seconds(state: Any, path: Optional[Path] = None) -> Optional[float]:
    """Extract a best-effort timestamp (seconds) from a worker/run/progress dict.

    Different components emit different fields depending on version.
    We support:
      - Numeric: ts, timestamp
      - Strings: utc, updated_at, timestamp_utc

    If the dict doesn't contain any known timestamp field, we fall back to the
    file mtime when a path is provided.
    """
    if not isinstance(state, dict):
        try:
            if path is not None and path.exists():
                return float(path.stat().st_mtime)
        except Exception:
            return None
        return None

    # 1) Numeric timestamps are the most reliable.
    for key in ("ts", "timestamp"):
        try:
            v = _maybe_float(state.get(key))
            if v is not None and v > 0:
                return float(v)
        except Exception:
            pass

    # 2) ISO-ish timestamps.
    for key in ("utc", "updated_at", "timestamp_utc", "time_utc"):
        raw = state.get(key)
        if isinstance(raw, str) and raw.strip():
            dt = _parse_timestamp_str(raw)
            if dt is not None:
                try:
                    return float(dt.replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    pass

    # 3) Fallback to file mtime.
    try:
        if path is not None and path.exists():
            return float(path.stat().st_mtime)
    except Exception:
        return None

    return None


def _first_existing_json(paths: List[Path]) -> Tuple[Optional[Any], Optional[Path]]:
    """Return (json_data, path) for the first readable JSON in paths."""
    for p in paths:
        data = _load_json_file(p)
        if data is not None:
            return data, p
    return None, None


def _candidate_state_paths(run_id: Optional[str] = None) -> Dict[str, List[Path]]:
    """Common places the worker may write state/diagnostics files."""
    runs_root = Path(get_runs_root())
    queue_root = Path(get_queue_root())

    logs = runs_root / "logs"

    # Some deployments (e.g., Render) can override where logs are written.
    # Keep Streamlit in sync with the worker by checking the same env overrides.
    logs_dirs: List[Path] = [logs]
    for _k in ("ARA_RUNS_LOGS_DIR", "ARA_RUNS_LOG_DIR", "ARA_LOGS_DIR"):
        _v = os.getenv(_k)
        if _v:
            try:
                _p = Path(_v).expanduser()
                if _p not in logs_dirs:
                    logs_dirs.append(_p)
            except Exception:
                pass
    q_pending = queue_root / "pending"
    q_active = queue_root / "active"
    q_finished = queue_root / "finished"

    # Generic filenames (shared)
    worker_state = [
        # Root-level (run_jobs defaults)
        runs_root / "worker_state.json",
        runs_root / "engine_worker_state.json",
        logs / "worker_state.json",
        logs / "engine_worker_state.json",
        logs / "worker_status.json",
        queue_root / "worker_state.json",
        q_active / "worker_state.json",
    ]
    run_state = [
        # Root-level (some deployments write run_state.json here)
        runs_root / "run_state.json",
        logs / "run_state.json",
        logs / "last_run_state.json",
        queue_root / "run_state.json",
        q_active / "run_state.json",
    ]
    heartbeat = [
        # Root-level (run_jobs watchdog heartbeat default)
        runs_root / "watchdog_heartbeat.json",
        logs / "watchdog_heartbeat.json",
        logs / "heartbeat.json",
        logs / "worker_heartbeat.json",
        logs / "watchdog.json",
        logs / "watchdog_state.json",
        queue_root / "watchdog_heartbeat.json",
        q_active / "watchdog_heartbeat.json",
    ]
    events = [
        # JSONL (append-only) event streams (preferred)
        logs / "events_global.jsonl",
        logs / "events.jsonl",
        # Legacy JSON (kept for backward compatibility)
        logs / "event_log.json",
        logs / "events.json",
        logs / "timeline.json",
        queue_root / "event_log.json",
    ]

    # Include the same filenames under any env-overridden log directories.
    if len(logs_dirs) > 1:
        for _ld in logs_dirs[1:]:
            worker_state.extend([
                _ld / "worker_state.json",
                _ld / "engine_worker_state.json",
                _ld / "worker_status.json",
            ])
            run_state.extend([
                _ld / "run_state.json",
                _ld / "last_run_state.json",
            ])
            heartbeat.extend([
                _ld / "watchdog_heartbeat.json",
                _ld / "heartbeat.json",
                _ld / "worker_heartbeat.json",
                _ld / "watchdog.json",
                _ld / "watchdog_state.json",
            ])
            events.extend([
                _ld / "events_global.jsonl",
                _ld / "events.jsonl",
                _ld / "event_log.json",
                _ld / "events.json",
                _ld / "timeline.json",
            ])

    # Per-run filenames (if emitted)
    if run_id:
        worker_state.extend(
            [
                logs / f"{run_id}_worker_state.json",
                logs / f"{run_id}_state.json",
                runs_root / run_id / "worker_state.json",
                runs_root / run_id / "state.json",
            ]
        )
        run_state.extend(
            [
                logs / f"{run_id}_run_state.json",
                logs / f"{run_id}_runstate.json",
                runs_root / run_id / "run_state.json",
            ]
        )
        heartbeat.extend(
            [
                logs / f"{run_id}_heartbeat.json",
                runs_root / run_id / "heartbeat.json",
                runs_root / run_id / "watchdog_heartbeat.json",
                runs_root / run_id / "watchdog.json",
            ]
        )
        events.extend(
            [
                # Per-run JSONL (append-only; current default)
                logs / f"{run_id}_events.jsonl",
                logs / f"{run_id}_event_log.jsonl",
                runs_root / run_id / "events.jsonl",
                runs_root / run_id / "event_log.jsonl",
                # Per-run legacy JSON (backward compatibility)
                logs / f"{run_id}_events.json",
                logs / f"{run_id}_event_log.json",
                runs_root / run_id / "events.json",
                runs_root / run_id / "event_log.json",
            ]
        )

        # Also consider per-run files under env-overridden log dirs.
        if len(logs_dirs) > 1:
            for _ld in logs_dirs[1:]:
                worker_state.extend([
                    _ld / f"{run_id}_worker_state.json",
                    _ld / f"{run_id}_state.json",
                ])
                run_state.extend([
                    _ld / f"{run_id}_run_state.json",
                    _ld / f"{run_id}_runstate.json",
                ])
                heartbeat.extend([
                    _ld / f"{run_id}_heartbeat.json",
                ])
                events.extend([
                    _ld / f"{run_id}_events.jsonl",
                    _ld / f"{run_id}_event_log.jsonl",
                    _ld / f"{run_id}_events.json",
                    _ld / f"{run_id}_event_log.json",
                ])

    progress = []
    if run_id:
        progress = [
            logs / f"{run_id}_progress.json",
            queue_root / f"{run_id}_progress.json",
            q_active / f"{run_id}_progress.json",
            q_finished / f"{run_id}_progress.json",
            runs_root / run_id / "progress.json",
        ]

        if len(logs_dirs) > 1:
            for _ld in logs_dirs[1:]:
                progress.append(_ld / f"{run_id}_progress.json")

    return {
        "worker_state": worker_state,
        "run_state": run_state,
        "heartbeat": heartbeat,
        "events": events,
        "progress": progress,
        "queue_pending": [q_pending],
        "queue_active": [q_active],
        "queue_finished": [q_finished],
        "runs_logs": logs_dirs,
    }



def _normalize_watchdog_info(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize watchdog heartbeat payloads from various backends.

    Fixes:
    - If last_beat is a numeric epoch (float/int), compute seconds_since_last automatically.
    """
    if not isinstance(raw, dict):
        return {}
    last_beat = (
        raw.get("last_beat")
        or raw.get("lastBeat")
        or raw.get("ts")
        or raw.get("timestamp")
        or raw.get("utc")
        or raw.get("time")
    )
    seconds_since = raw.get("seconds_since_last") or raw.get("secondsSinceLast") or raw.get("age_seconds") or raw.get("age")
    count = raw.get("count") or raw.get("heartbeat_count") or raw.get("beats")

    # If seconds_since wasn't provided, compute it from last_beat when possible.
    if seconds_since is None and last_beat is not None:
        dt = _as_datetime_utc(last_beat)
        if dt is not None:
            try:
                seconds_since = (datetime.utcnow() - dt).total_seconds()
            except Exception:
                seconds_since = None

    try:
        if seconds_since is not None:
            seconds_since = float(seconds_since)
            if seconds_since < 0:
                seconds_since = 0.0
    except Exception:
        seconds_since = None

    return {
        "last_beat": last_beat,
        "seconds_since_last": seconds_since,
        "count": _safe_int(count, 0),
    }
def _is_meaningful_watchdog(info: Optional[Dict[str, Any]]) -> bool:
    """Return True only if the watchdog payload contains real signals.

    This prevents an "empty shell" dict (count=0, last_beat=None) coming from MemoryStore
    from blocking file-based fallbacks.
    """
    if not isinstance(info, dict):
        return False

    last = info.get("last_beat")
    if isinstance(last, str) and last.strip():
        return True
    if isinstance(last, (int, float)):
        try:
            return float(last) > 0.0
        except Exception:
            pass

    count = _safe_int(info.get("count"), 0) or 0
    if count > 0:
        return True

    seconds = _maybe_float(info.get("seconds_since_last"))
    if seconds is not None:
        return True

    return False


def _watchdog_dt(info: Optional[Dict[str, Any]], fallback_path: Optional[Path] = None) -> Optional[datetime]:
    """Best-effort datetime for 'freshness' comparisons."""
    if isinstance(info, dict):
        lb = info.get("last_beat")
        if isinstance(lb, (int, float)):
            try:
                return datetime.utcfromtimestamp(float(lb))
            except Exception:
                pass
        if isinstance(lb, str):
            dt = _parse_timestamp_str(lb)
            if dt is not None:
                return dt

    if isinstance(fallback_path, Path):
        try:
            return datetime.utcfromtimestamp(float(fallback_path.stat().st_mtime))
        except Exception:
            return None

    return None


def load_watchdog_info_unified(
    memory: MemoryStore,
    run_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load heartbeat/watchdog info.

    Fix:
    - Do NOT treat an empty MemoryStore watchdog dict as authoritative.
      If MemoryStore returns {count:0,last_beat:None}, we fall back to file-based heartbeat,
      and if that also doesn't exist, we return None so the UI can synthesize "last activity".
    - If both MemoryStore and a file exist, pick the fresher one by timestamp/mtime.
    """
    mem_info: Optional[Dict[str, Any]] = None
    mem_src = "MemoryStore.get_watchdog_info"

    func = getattr(memory, "get_watchdog_info", None)
    if callable(func):
        raw = None
        try:
            if run_id is not None:
                try:
                    raw = func(run_id=run_id)  # type: ignore[misc]
                except TypeError:
                    try:
                        raw = func(run_id)  # type: ignore[misc]
                    except TypeError:
                        raw = func()  # type: ignore[misc]
            else:
                raw = func()  # type: ignore[misc]
        except Exception:
            raw = None

        cand = _normalize_watchdog_info(raw)
        if _is_meaningful_watchdog(cand):
            mem_info = cand

    file_info: Optional[Dict[str, Any]] = None
    file_src: str = "not found"
    file_path: Optional[Path] = None

    paths = _candidate_state_paths(run_id=run_id)["heartbeat"]
    raw2, p = _first_existing_json(paths)
    cand2 = _normalize_watchdog_info(raw2)
    if _is_meaningful_watchdog(cand2) and p is not None:
        file_info = cand2
        file_path = p
        file_src = str(p)

    # Choose freshest when both exist
    if mem_info and file_info:
        dt_mem = _watchdog_dt(mem_info)
        dt_file = _watchdog_dt(file_info, fallback_path=file_path)

        use_file = dt_file is not None and (dt_mem is None or dt_file >= dt_mem)
        primary, primary_src, secondary = (file_info, file_src, mem_info) if use_file else (mem_info, mem_src, file_info)

        merged = dict(primary)
        merged["count"] = max(_safe_int(mem_info.get("count"), 0) or 0, _safe_int(file_info.get("count"), 0) or 0)
        merged["last_beat"] = merged.get("last_beat") or secondary.get("last_beat")
        merged["seconds_since_last"] = _coalesce(merged.get("seconds_since_last"), secondary.get("seconds_since_last"))
        return merged, primary_src

    if file_info:
        return file_info, file_src
    if mem_info:
        return mem_info, mem_src

    return None, "not found"


def _as_datetime_utc(v: Any) -> Optional[datetime]:
    """Convert common timestamp formats into a naive UTC datetime."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(v))
        except Exception:
            return None
    if isinstance(v, str):
        return _parse_timestamp_str(v)
    return None



def _format_watchdog_last_beat(last_beat: Any) -> str:
    """Human-friendly formatting for watchdog timestamps."""
    if last_beat in (None, ""):
        return "None recorded"
    dt = _as_datetime_utc(last_beat)
    if dt is not None:
        try:
            return dt.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            return str(last_beat)
    return str(last_beat)
def _latest_timestamp_in_dict(d: Optional[Dict[str, Any]], keys: List[str]) -> Optional[datetime]:
    if not isinstance(d, dict):
        return None
    best: Optional[datetime] = None
    for k in keys:
        dt = _as_datetime_utc(d.get(k))
        if dt is None:
            continue
        if best is None or dt > best:
            best = dt
    return best


def synthesize_watchdog_from_activity(
    worker_state: Optional[Dict[str, Any]],
    run_state: Optional[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Synthesize a 'watchdog-like' heartbeat when real heartbeat isn't emitted.

    Used only when:
    - MemoryStore watchdog is empty, AND
    - no heartbeat JSON is found on disk.

    It derives 'last_beat' from the freshest timestamp it can find in:
    - worker_state (updated_at/timestamp/etc)
    - run_state (updated_at/timestamp/etc)
    - cycle history (last cycle timestamp)

    Returns (watchdog_dict_or_none, source_label).
    """
    candidates: List[Tuple[datetime, str]] = []

    ws_dt = _latest_timestamp_in_dict(
        worker_state,
        keys=["updated_at", "updatedAt", "timestamp", "ts", "last_update", "lastUpdate", "last_seen", "lastSeen"],
    )
    if ws_dt is not None:
        candidates.append((ws_dt, "derived:worker_state"))

    rs_dt = _latest_timestamp_in_dict(
        run_state,
        keys=["updated_at", "updatedAt", "timestamp", "ts", "last_update", "lastUpdate", "saved_at", "savedAt"],
    )
    if rs_dt is not None:
        candidates.append((rs_dt, "derived:run_state"))

    hist_dt: Optional[datetime] = None
    if isinstance(history, list) and history:
        ts = history[-1].get("timestamp")
        if isinstance(ts, str):
            hist_dt = _parse_timestamp_str(ts)
    if hist_dt is not None:
        candidates.append((hist_dt, "derived:history_last_cycle"))

    if not candidates:
        return None, "not found"

    best_dt, src = max(candidates, key=lambda x: x[0])

    try:
        seconds_since = (datetime.utcnow() - best_dt).total_seconds()
    except Exception:
        seconds_since = None

    # Heuristic count: use cycle count if we have it (best effort)
    count_guess = 0
    if isinstance(history, list) and history:
        count_guess = len(history)
    else:
        count_guess = _safe_int((worker_state or {}).get("cycle") or (worker_state or {}).get("current"), 0) or 0

    last_beat_str = best_dt.isoformat(timespec="seconds") + "Z"
    return (
        {"last_beat": last_beat_str, "count": count_guess, "seconds_since_last": seconds_since},
        src,
    )


def _normalize_worker_state(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    ws = dict(raw)

    # Normalize common run id keys
    run_id_val = ws.get("run_id") or ws.get("job_id") or ws.get("id")
    if run_id_val is not None:
        ws["run_id"] = str(run_id_val)

    # Normalize progress keys (both phase and cycle)
    # Phase progress (if worker emits it)
    if "phase_total" not in ws and "total_phases" in ws:
        ws["phase_total"] = ws.get("total_phases")
    if "phase_index" not in ws and "phase" in ws:
        ws["phase_index"] = ws.get("phase")

    # Cycle progress
    if "total" not in ws and "total_cycles" in ws:
        ws["total"] = ws.get("total_cycles")

    # IMPORTANT: preserve 0 values
    if "current" not in ws or ws.get("current") is None:
        ws["current"] = _coalesce(ws.get("current_cycle"), ws.get("cycle"), ws.get("cycle_index"))

    return ws


def load_worker_state_unified(
    memory: MemoryStore,
    run_id_hint: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load worker state, preferring MemoryStore but with file fallbacks."""
    ws_mem: Optional[Dict[str, Any]] = None
    ws_mem_src: Optional[str] = None

    # MemoryStore methods (try multiple names)
    for name in ("get_worker_state", "read_worker_state", "load_worker_state"):
        func = getattr(memory, name, None)
        if not callable(func):
            continue
        raw = None
        try:
            if run_id_hint:
                # try keyword then positional then no-arg fallback
                try:
                    raw = func(run_id=run_id_hint)  # type: ignore[misc]
                except TypeError:
                    try:
                        raw = func(run_id_hint)  # type: ignore[misc]
                    except TypeError:
                        raw = func()  # type: ignore[misc]
            else:
                raw = func()  # type: ignore[misc]
        except Exception:
            raw = None

        ws = _normalize_worker_state(raw)
        if ws:
            ws_mem = ws
            ws_mem_src = f"MemoryStore.{name}"
            break

    ws_disk: Optional[Dict[str, Any]] = None
    ws_disk_src: Optional[str] = None
    ws_disk_path: Optional[Path] = None
    paths = _candidate_state_paths(run_id=run_id_hint)["worker_state"]
    raw2, p = _first_existing_json(paths)
    ws2 = _normalize_worker_state(raw2)
    if ws2 and p is not None:
        ws_disk = ws2
        ws_disk_src = str(p)
        ws_disk_path = p

    # If both exist, prefer the freshest payload.
    if ws_mem and ws_disk:
        ts_mem = _extract_state_timestamp_seconds(ws_mem, None)
        ts_disk = _extract_state_timestamp_seconds(ws_disk, ws_disk_path)
        if ts_disk is not None and (ts_mem is None or ts_disk >= ts_mem):
            return ws_disk, ws_disk_src or "disk"
        return ws_mem, ws_mem_src or "MemoryStore"

    if ws_disk:
        return ws_disk, ws_disk_src or "disk"
    if ws_mem:
        return ws_mem, ws_mem_src or "MemoryStore"

    return None, "not found"


def load_run_state_unified(
    memory: MemoryStore,
    run_id_hint: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load last saved run state, preferring MemoryStore but with file fallbacks."""
    rs_mem: Optional[Dict[str, Any]] = None
    rs_mem_src: Optional[str] = None
    func = getattr(memory, "load_run_state", None)
    if callable(func):
        raw = None
        try:
            if run_id_hint:
                try:
                    raw = func(run_id=run_id_hint)  # type: ignore[misc]
                except TypeError:
                    try:
                        raw = func(run_id_hint)  # type: ignore[misc]
                    except TypeError:
                        raw = func()  # type: ignore[misc]
            else:
                raw = func()  # type: ignore[misc]
        except Exception:
            raw = None

        if isinstance(raw, dict) and raw:
            rs_mem = raw
            rs_mem_src = "MemoryStore.load_run_state"

    rs_disk: Optional[Dict[str, Any]] = None
    rs_disk_src: Optional[str] = None
    rs_disk_path: Optional[Path] = None
    paths = _candidate_state_paths(run_id=run_id_hint)["run_state"]
    raw2, p = _first_existing_json(paths)
    if isinstance(raw2, dict) and raw2 and p is not None:
        rs_disk = raw2
        rs_disk_src = str(p)
        rs_disk_path = p

    if rs_mem and rs_disk:
        ts_mem = _extract_state_timestamp_seconds(rs_mem, None)
        ts_disk = _extract_state_timestamp_seconds(rs_disk, rs_disk_path)
        if ts_disk is not None and (ts_mem is None or ts_disk >= ts_mem):
            return rs_disk, rs_disk_src or "disk"
        return rs_mem, rs_mem_src or "MemoryStore"

    if rs_disk:
        return rs_disk, rs_disk_src or "disk"
    if rs_mem:
        return rs_mem, rs_mem_src or "MemoryStore"

    return None, "not found"


def load_progress_unified(run_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load progress JSON if the worker emits one (recommended for smooth 1/3->2/3->3/3 updates)."""
    if not run_id:
        return None, "no run_id"

    paths = _candidate_state_paths(run_id=run_id)["progress"]
    raw, p = _first_existing_json(paths)
    if isinstance(raw, dict) and raw and p is not None:
        return raw, str(p)

    # Loose glob fallback (in case the worker uses custom naming)
    try:
        runs_root = Path(get_runs_root())
        queue_root = Path(get_queue_root())
        glob_candidates: List[Path] = []
        glob_candidates.extend(list(runs_root.glob(f"**/{run_id}*progress*.json")))
        glob_candidates.extend(list(queue_root.glob(f"**/{run_id}*progress*.json")))
        for gp in glob_candidates[:10]:
            data = _load_json_file(gp)
            if isinstance(data, dict) and data:
                return data, str(gp)
    except Exception:
        pass

    return None, "not found"


def _derive_active_run_id_from_queue() -> Optional[str]:
    """Try to infer an active run id by looking at queue/active or runs/active."""
    # Prefer canonical queue/active
    candidates: List[Path] = []
    try:
        qr = Path(get_queue_root())
        candidates.append(qr / "active")
        candidates.append(qr)
    except Exception:
        pass
    # Also check run_jobs active dir if present
    if isinstance(RUNS_ACTIVE_DIR, Path):
        candidates.append(RUNS_ACTIVE_DIR)

    for base in candidates:
        if not base.exists() or not base.is_dir():
            continue
        # Look for *_job.json
        try:
            job_files = sorted(base.glob("*_job.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            job_files = sorted(base.glob("*_job.json"))
        for fp in job_files[:10]:
            stem = fp.name.replace("_job.json", "")
            if stem:
                return stem
    return None


def load_job_payload_from_disk(run_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load the job JSON (config/meta) for a given run_id.

    This enables autonomy level + agent presence even when MemoryStore doesn't expose it.
    """
    if not run_id:
        return None, "no run_id"

    candidates: List[Path] = []

    # canonical dirs
    try:
        qr = Path(get_queue_root())
        candidates.extend(
            [
                qr / "pending" / f"{run_id}_job.json",
                qr / "active" / f"{run_id}_job.json",
                qr / "finished" / f"{run_id}_job.json",
                qr / f"{run_id}_job.json",
                qr / f"{run_id}.json",  # legacy
            ]
        )
    except Exception:
        pass

    # run_jobs dirs if exposed
    for d in (RUNS_PENDING_DIR, RUNS_ACTIVE_DIR, RUNS_FINISHED_DIR, RUNS_ERROR_DIR):
        if isinstance(d, Path):
            candidates.append(d / f"{run_id}_job.json")
            candidates.append(d / f"{run_id}.json")

    # Also check runs_root
    try:
        rr = Path(get_runs_root())
        candidates.extend(
            [
                rr / "pending" / f"{run_id}_job.json",
                rr / "active" / f"{run_id}_job.json",
                rr / "finished" / f"{run_id}_job.json",
            ]
        )
    except Exception:
        pass

    for fp in candidates:
        data = _load_json_file(fp)
        if isinstance(data, dict) and data:
            return data, str(fp)

    return None, "not found"


def compute_progress_view(
    worker_state: Optional[Dict[str, Any]],
    progress_state: Optional[Dict[str, Any]],
    watchdog: Optional[Dict[str, Any]],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a robust (current, total, fraction, label) progress view.

    Supports:
    - phase_index/phase_total (preferred if present)
    - current/total cycles
    - fallbacks based on progress JSON naming variants

    Update: fixes common off-by-one for phase progress by remembering whether a given run
    appears to be zero-indexed (per run_id) based on observing an initial 0 while running.
    """
    ws = worker_state or {}
    ps = progress_state or {}

    status = _coalesce(ws.get("status"), ps.get("status"), "unknown")
    status_s = str(status).lower()

    if run_id is None:
        run_id_val = _coalesce(ws.get("run_id"), ps.get("run_id"), ws.get("job_id"), ps.get("job_id"))
        if run_id_val is not None:
            run_id = str(run_id_val)

    # Consider additional final statuses as finished-like.  Some job engines
    # reset the worker status to "idle" or "stopped" after completing a
    # run.  Without including these in the finished set, the UI may
    # display a partially filled progress bar even though no further
    # cycles will run.  Include "idle" and "stopped" to treat
    # those states as terminal.
    finished_like = status_s in {
        "finished",
        "done",
        "completed",
        "complete",
        "success",
        "idle",
        "stopped",
    }

    # Signals that the worker has started doing real work (status running OR fresh-ish heartbeat)
    hb_count = _safe_int((watchdog or {}).get("count"), 0) or 0
    hb_last = (watchdog or {}).get("last_beat")
    hb_age = _maybe_float((watchdog or {}).get("seconds_since_last"))
    heartbeat_fresh = hb_age is None or hb_age <= 600.0
    running_like = status_s in {"running", "active", "in_progress", "working"}
    started_like = bool(running_like or hb_count > 0 or (hb_last and heartbeat_fresh))

    # Preferred: phase progress (3-phase pipeline etc.) (preserve 0 values)
    phase_cur = _coalesce(ws.get("phase_index"), ps.get("phase_index"), ps.get("phase_current"))
    phase_tot = _coalesce(ws.get("phase_total"), ps.get("phase_total"), ps.get("phase_count"))
    phase_name = _coalesce(ws.get("phase_name"), ps.get("phase_name"), "") or ""

    # Cycle progress (preserve 0 values)
    cur = _coalesce(
        ws.get("effective_current"),
        ps.get("effective_current"),
        ws.get("current"),
        ps.get("current"),
        ps.get("current_cycle"),
        ps.get("cycle"),
        ps.get("cycle_index"),
    )
    tot = _coalesce(
        ws.get("effective_total"),
        ps.get("effective_total"),
        ws.get("total"),
        ps.get("total"),
        ps.get("total_cycles"),
        ps.get("max_cycles"),
    )

    # Select which progress track to display
# (comment trimmed to keep this file renderable in GitHub)
    phase_total_int = _safe_int(phase_tot, None)
    use_phase = phase_total_int
    # When phase_total is 1 or less, fall back to cycle progress instead of using phase progress.  This
# (comment trimmed to keep this file renderable in GitHub)
    if phase_total_int is not None and phase_total_int > 1:
        phase_cur_raw = phase_cur
        c = _safe_int(phase_cur_raw, 0) or 0
        t = use_phase

        # If finished, clamp to full
        if finished_like:
            c_disp = t
        else:
            # Remember whether this run looks zero-indexed for phase progress
            key = f"ara_phase_zero_indexed::{run_id or 'global'}"
            zero_indexed = bool(st.session_state.get(key, False))

            # If we observe phase_cur==0 (and phase_cur actually exists), treat as 0-based and mark it
            if phase_cur_raw is not None and c == 0 and started_like:
                st.session_state[key] = True
                zero_indexed = True

            if zero_indexed:
                # 0-based indices 0..t-1
                if c < 0:
                    c_disp = 0
                elif c >= t:
                    c_disp = t
                else:
                    # display 1..t
                    c_disp = min(c + 1, t)
            else:
                # 1-based or already-display-ready
                c_disp = min(max(c, 0), t)

            # UX: if started and still at 0, show 1 as "in progress"
            if started_like and c_disp == 0 and t > 0:
                c_disp = 1

        frac = float(c_disp) / float(t) if t > 0 else None
        return {
            "kind": "phase",
            "current": c_disp,
            "total": t,
            "fraction": _clamp_float(frac),
            "label": phase_name or "phases",
        }

    # Otherwise show cycle progress
    c2 = _safe_int(cur, None)
    t2 = _safe_int(tot, None)

    # If the job is finished but the reported cycle progress has not
    # reached the total, treat it as complete.  This prevents the
    # progress bar from appearing partially filled when the run
    # terminated early (e.g. in swarm mode where early stopping
# (comment trimmed to keep this file renderable in GitHub)
    # configured maximum).  By promoting the current value to
    # equal the total for finished runs, the UI more clearly
    # communicates that no more cycles remain.
    try:
        if finished_like and c2 is not None and t2 is not None and t2 > 0 and c2 < t2:
            c2 = t2
    except Exception:
        pass
    # Remap internal step-based progress to user cycles when a macro cycle count
    # is available from the prompt details.  Agents sometimes report progress in
    # terms of a large internal step count (e.g. 18) instead of the user requested
    # number of cycles (e.g. 3).  When a macro_total can be derived from the
    # prompt_details (max_cycles or max_rounds), compute the effective cycle
    # index by dividing the step count by the number of steps per cycle and
    # rounding up.  This preserves the final total cycles while allowing the
    # progress bar to show 1/3->2/3->3/3 instead of 1/18->2/18->... etc.
    try:
        macro_total_ui: Optional[int] = None
        pd: Any = None
        mode_pd: Optional[str] = None  # capture mode for use outside remapping block
        # prompt_details may exist in progress_state (ps) or worker_state (ws)
        if isinstance(ps, dict):
            pd = ps.get("prompt_details")  # type: ignore[attr-defined]
        if not pd and isinstance(ws, dict):
            pd = ws.get("prompt_details")  # type: ignore[attr-defined]
        if isinstance(pd, dict):
            # Determine macro_total based on mode; use max_rounds for swarm modes
            mode_pd = pd.get("mode")
            mc = None
            if mode_pd == "swarm":
                # In swarm mode a single round consists of one cycle per role.
                # When only ``max_rounds`` is used, the UI remaps hundreds of
# (comment trimmed to keep this file renderable in GitHub)
                # obscuring the true progress through all agents.  To better
# (comment trimmed to keep this file renderable in GitHub)
                # number of roles.  For example, 3 rounds with 32 roles
                # produces 96 logical cycles.  If roles is missing, assume 1.
                mc = pd.get("max_rounds") or pd.get("max_cycles")
                # When rendering the swarm progress bar, prefer to multiply
                # the number of requested rounds by the total number of agents
                # (swarm_size) if available.  If swarm_size is not present,
                # fall back to the length of the roles list.  This avoids
                # miscounting cycles when the prompt details only include a
                # subset of roles (e.g. unique role names) rather than one entry
                # per agent.
                swarm_sz: Optional[int] = None
                try:
                    swarm_sz_val = pd.get("swarm_size")
                    if swarm_sz_val is not None:
                        # Coerce to int when possible
                        swarm_sz = _safe_int(swarm_sz_val, None)
                except Exception:
                    swarm_sz = None

                roles_list = pd.get("roles") if isinstance(pd.get("roles"), list) else None
                roles_count = len(roles_list) if roles_list else 1
                try:
                    mc_int = _safe_int(mc, None)
                    if mc_int is not None:
                        if isinstance(swarm_sz, int) and swarm_sz > 0:
                            # Use swarm size (number of agents) when available
                            macro_total_ui = mc_int * swarm_sz
                        elif roles_count > 0:
                            # Otherwise use the count of roles entries
                            macro_total_ui = mc_int * roles_count
                        else:
                            macro_total_ui = mc_int
                    else:
                        macro_total_ui = None
                except Exception:
                    macro_total_ui = None
            else:
                mc = pd.get("max_cycles") or pd.get("max_rounds")
                macro_total_ui = _safe_int(mc, None)
# (comment trimmed to keep this file renderable in GitHub)
# (comment trimmed to keep this file renderable in GitHub)
        # be hundreds of steps) instead of collapsing to the requested number of rounds.
        if (
            mode_pd != "swarm"
            and macro_total_ui is not None
            and t2 is not None
            and c2 is not None
            and t2 > 0
            and macro_total_ui > 0
            and t2 > macro_total_ui
        ):
            import math as _math
            try:
                steps_per = float(t2) / float(macro_total_ui)
                if steps_per > 0.0:
                    c2 = int(_math.ceil(float(c2) / steps_per))
                    t2 = macro_total_ui
            except Exception:
                pass

        # In swarm mode some older backends may report total cycles based on the
        # number of unique roles (e.g. 32) rather than the requested swarm size
        # (e.g. 64).  When prompt_details provides a larger macro_total_ui for
        # swarm mode, prefer that over the reported total to ensure the cycle
# (comment trimmed to keep this file renderable in GitHub)
        # current value here; leave c2 untouched so it still reflects the
        # actual number of steps executed so far.
        try:
            if (
                mode_pd == "swarm"
                and macro_total_ui is not None
                and t2 is not None
                and isinstance(t2, (int, float))
                and isinstance(macro_total_ui, (int, float))
                and macro_total_ui > t2
            ):
                t2 = int(macro_total_ui)
        except Exception:
            pass
    except Exception:
        pass

    # If state-based progress looks "stuck" (e.g., stops updating around ~25),
    # fall back to live event logs which often carry per-cycle markers.
    if run_id:
        try:
            base_dir = get_runs_root()
            fallback_total_ev = t2
            if fallback_total_ev is None:
                m = locals().get("macro_total_ui")
                if isinstance(m, int):
                    fallback_total_ev = m
            ev_cur, ev_tot = _infer_cycle_progress_from_event_logs(
                base_dir,
                run_id,
                fallback_total=fallback_total_ev,
                max_lines=4000,
            )
            if ev_tot is not None and (t2 is None or (isinstance(t2, int) and ev_tot > t2)):
                # Prefer the larger total if it still looks reasonable.
                if 0 < ev_tot <= 100000:
                    t2 = ev_tot
            if ev_cur is not None:
                # Prefer event-derived current when it is ahead of the state-derived one.
                if c2 is None:
                    c2 = ev_cur
                elif isinstance(c2, int) and ev_cur > c2:
                    # Classic symptom: progress derived from a capped history window stalls early.
                    if c2 <= 25 or ev_cur <= (t2 or ev_cur):
                        c2 = ev_cur
        except Exception:
            pass

    if c2 is None or t2 is None or t2 <= 0:
        return {"kind": "none", "current": None, "total": None, "fraction": None, "label": ""}

    # If finished, clamp to full
    if finished_like:
        c_disp2 = t2
    else:
        c_disp2 = min(max(c2, 0), t2)

        # UX: if worker has started and current still reads 0, show 1 as "in progress"
        if started_like and c_disp2 == 0 and t2 > 0:
            c_disp2 = 1

    frac2 = float(c_disp2) / float(t2) if t2 > 0 else None
    return {
        "kind": "cycle",
        "current": c_disp2,
        "total": t2,
        "fraction": _clamp_float(frac2),
        "label": "cycles",
    }



def compute_activity_pulse_view(
    events: Optional[List[Dict[str, Any]]],
    watchdog: Optional[Dict[str, Any]],
    worker_state: Optional[Dict[str, Any]],
    *,
    event_log_src: str = "",
) -> Dict[str, Any]:
    """Compute an activity pulse score for the top bar.

    This intentionally does **not** try to count cycles. Instead it derives a
    liveness / activity signal from:
      - how recently the event log produced entries
      - how many events occurred recently (1m + 5m windows)
      - watchdog heartbeat freshness

    Returns a dict with stable keys:
      - score: 0..1
      - label: High | Medium | Low | Idle
      - events_last_60s / events_last_5m
      - last_event_age_s / last_event_ts (best effort)
    """
    evs: List[Dict[str, Any]] = []
    if isinstance(events, list):
        evs = [e for e in events if isinstance(e, dict)]

    now = datetime.utcnow()

    last_dt: Optional[datetime] = None
    c60 = 0
    c300 = 0

    for ev in evs:
        dt = _event_ts_to_dt(
            _coalesce(
                ev.get("ts"),
                ev.get("timestamp"),
                ev.get("utc"),
                ev.get("time"),
                ev.get("created_at"),
                ev.get("createdAt"),
                ev.get("updated_at"),
                ev.get("updatedAt"),
            )
        )
        if dt is None:
            continue

        if last_dt is None or dt > last_dt:
            last_dt = dt

        try:
            age_s = float((now - dt).total_seconds())
        except Exception:
            continue
        if age_s < 0:
            continue
        if age_s <= 60.0:
            c60 += 1
        if age_s <= 300.0:
            c300 += 1

    last_event_age: Optional[float] = None
    last_event_ts: Optional[str] = None
    if last_dt is not None:
        try:
            last_event_age = max(0.0, float((now - last_dt).total_seconds()))
            last_event_ts = last_dt.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            last_event_age = None
            last_event_ts = None

    # Fallback: if we couldn't parse timestamps from events, use the event log file mtime.
    if last_event_age is None and isinstance(event_log_src, str) and event_log_src and event_log_src != "not found":
        try:
            p = Path(event_log_src)
            if p.exists():
                mdt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)
                last_event_age = max(0.0, float((now - mdt).total_seconds()))
                last_event_ts = mdt.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass

    heartbeat_age = _maybe_float((watchdog or {}).get("seconds_since_last"))

    # Best age signal: prefer last event, otherwise heartbeat.
    best_age: Optional[float] = None
    try:
        candidates = [a for a in [last_event_age, heartbeat_age] if isinstance(a, (int, float))]
        if candidates:
            best_age = float(min(candidates))
    except Exception:
        best_age = None

    # Event rate per minute, using the more stable of 1m and 5m windows.
    rate_1m = float(c60)
    rate_5m = float(c300) / 5.0 if c300 else 0.0
    event_rate = max(rate_1m, rate_5m)

    # ------------------------------------------------------------------
    # Adjustments for heartbeatâonly activity
    #
    # In many runs the underlying agent may work silently for long periods
    # (e.g. waiting on network or large tool calls) while still emitting
    # heartbeat signals. Historically this caused a "Low" pulse because the
    # event_rate was zero even though the worker was alive. To better
    # reflect liveness in such cases, when there are no recent events but
    # a fresh heartbeat is available, bump the event_rate to a small
    # baseline so the pulse reads as Medium instead of Low. The cutoff
    # window (120 seconds) matches the typical heartbeat interval.
    if (
        event_rate <= 0.1
        and isinstance(heartbeat_age, (int, float))
        and heartbeat_age is not None
        and heartbeat_age <= 120.0
    ):
        # Use a baseline rate of 3 events per minute, which yields a
        # noticeable activity signal without looking overly busy. This
        # baseline scales naturally with the HIGH_RATE constant below.
        event_rate = max(event_rate, 3.0)

    # Normalize to 0..1 (tunable constants)
    HIGH_RATE = 6.0  # events/min that feels "busy" in most deployments
    activity = _clamp_float(event_rate / HIGH_RATE, 0.0, 1.0) or 0.0

    # Freshness mapping (piecewise; avoids twitchiness)
    freshness = 0.0
    if best_age is not None:
        try:
            a = float(best_age)
            if a <= 15.0:
                freshness = 1.0
            elif a <= 60.0:
                freshness = 0.85
            elif a <= 180.0:
                freshness = 0.65
            elif a <= 600.0:
                freshness = 0.35
            elif a <= 1800.0:
                freshness = 0.15
            else:
                freshness = 0.05
        except Exception:
            freshness = 0.0

    # Weight the freshness signal slightly more than raw event activity.
    # If the agent is still beating (heartbeat) but event activity is low,
    # we want the pulse to reflect that the system is alive. Increasing
    # freshness weight to 0.6 helps avoid spurious "Low" labels when
    # heartbeat signals are frequent.
    score = (0.40 * activity) + (0.60 * freshness)

    # If the worker is finished/idle, dampen pulse so it doesn't look "alive".
    status_s = str((worker_state or {}).get("status") or "").lower()
    if status_s in {"finished", "done", "completed", "complete", "success", "idle", "stopped"}:
        score = min(score, 0.12)

    score = _clamp_float(score, 0.0, 1.0) or 0.0

    if score >= 0.75:
        label = "High"
    elif score >= 0.45:
        label = "Medium"
    elif score >= 0.20:
        label = "Low"
    else:
        label = "Idle"

    return {
        "score": score,
        "label": label,
        "events_last_60s": int(c60),
        "events_last_5m": int(c300),
        "last_event_age_s": last_event_age,
        "last_event_ts": last_event_ts,
        "heartbeat_age_s": heartbeat_age,
        "event_rate_per_min": float(event_rate),
    }


def derive_health_class(
    worker_state: Optional[Dict[str, Any]],
    watchdog: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """Return (health_class, human_label)."""
    status_raw = (worker_state or {}).get("status") or "unknown"
    status = str(status_raw).lower()

    seconds_since = _maybe_float((watchdog or {}).get("seconds_since_last"))
    hb_count = _safe_int((watchdog or {}).get("count"), 0) or 0

    # Worker status driven
    if status in {"queued", "pending"}:
        return "idle", "Queued"
    if status in {"finished", "done", "completed"}:
        return "idle", "Completed"
    if status in {"idle", "stopped", "standby", "waiting"}:
        return "idle", "Idle"
    if status in {"error", "failed"}:
        return "offline", "Error"

    # Heartbeat driven
    if seconds_since is None:
        if status in {"running", "active", "in_progress", "working", "running_job", "running_cycle", "processing", "busy"}:
            return "stale", "Running (no heartbeat)"
        return "unknown", "Unknown"

    # These thresholds assume heartbeat interval ~60s (your config uses 60)
    if seconds_since <= 90:
        return "healthy", "Healthy"
    if seconds_since <= 300:
        return "stale", "Stale"
    if status in {"running", "active", "in_progress", "working", "running_job", "running_cycle", "processing", "busy"} and hb_count > 0:
        return "offline", "Heartbeat lost"
    return "offline", "Offline"


def inject_base_styles() -> None:
    """Global UI polish + styles for heartbeat bar, cards, chips, and event feed."""
    st.markdown(
        textwrap.dedent(
            """
<style>
/* Layout rhythm */
.block-container { padding-top: 0.75rem; padding-bottom: 2.5rem; max-width: 1180px; }

/* Sticky topbar */
.ara-topbar-wrap{
  position: sticky;
  top: 0;
  z-index: 9999;
  margin: -0.75rem -1rem 0.85rem -1rem;
  padding: 0.65rem 1rem 0.6rem 1rem;
  background: rgba(8, 10, 18, 0.78);
  border-bottom: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(10px);
}
.ara-topbar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 1rem;
}
.ara-topbar-left, .ara-topbar-mid, .ara-topbar-right{
  display:flex; align-items:center; gap: 0.75rem; min-width: 0;
}

    /* Fix sticky topbar layout: allow wrapping and truncate long text */
    .ara-topbar {
      flex-wrap: wrap;
    }
    .ara-topbar-left, .ara-topbar-mid {
      flex: 1 1 auto;
    }
    .ara-topbar-right {
      /* allow the right section to grow and stack items on small screens */
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 0.25rem;
    }
    .ara-topbar-left > div,
    .ara-topbar-mid > div,
    .ara-topbar-right > div {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
    }
.ara-topbar-mid { opacity: 0.9; }
.ara-topbar-title{
  font-weight: 650;
  letter-spacing: 0.2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ara-dot{
  width: 10px; height: 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.15);
  box-shadow: 0 0 0 0 rgba(34,197,94,0.0);
}
.ara-dot.healthy { background: #22c55e; animation: araPulse 1.8s infinite; }
.ara-dot.stale   { background: #f59e0b; }
.ara-dot.offline { background: #ef4444; }
.ara-dot.idle    { background: #60a5fa; }
.ara-dot.unknown { background: #94a3b8; }

@keyframes araPulse{
  0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.35); }
  70%  { box-shadow: 0 0 0 10px rgba(34,197,94,0.0); }
  100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.0); }
}

.ara-kv{
  font-size: 0.85rem;
  opacity: 0.86;
  white-space: nowrap;
}
.ara-kv code{
  font-size: 0.8rem;
  padding: 0.1rem 0.35rem;
  border-radius: 8px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
}

/* Mini progress bar */
    .ara-mini-progress{
      /* widen progress bar on small screens; let it fill available space */
      width: 100%;
      min-width: 160px;
      height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.10);
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
    }
.ara-mini-progress > div{
  height: 100%;
  width: 0%;
  background: rgba(110,231,183,0.9);
}

/* Chips */
.ara-chip{
  display:inline-flex;
  align-items:center;
  gap: 0.4rem;
  padding: 0.30rem 0.60rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  margin: 0.15rem 0.35rem 0.15rem 0;
  font-size: 0.85rem;
  opacity: 0.92;
}
.ara-chip.active{
  border-color: rgba(110,231,183,0.55);
  box-shadow: 0 0 0 3px rgba(110,231,183,0.10);
}

/* Cards */
.ara-card{
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  background: rgba(255,255,255,0.03);
  padding: 0.9rem 1rem;
}
.ara-card-title{
  font-weight: 700;
  letter-spacing: 0.2px;
  margin-bottom: 0.25rem;
}
.ara-card-sub{
  font-size: 0.85rem;
  opacity: 0.78;
  margin-bottom: 0.65rem;
}
.ara-card-mono{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.82rem;
  opacity: 0.85;
}

/* Event feed */
.ara-feed{
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  background: rgba(255,255,255,0.02);
  padding: 0.25rem 0.9rem;
}
.ara-event{
  padding: 0.7rem 0.25rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.ara-event:last-child{ border-bottom: none; }
.ara-event-meta{
  font-size: 0.78rem;
  opacity: 0.70;
  margin-bottom: 0.15rem;
}
.ara-event-msg{
  font-size: 0.92rem;
  opacity: 0.92;
}

/* Reduce clutter of Streamlit default anchors in markdown headers */
h1 a, h2 a, h3 a, h4 a { display:none !important; }
</style>
            """
        ).strip("\n"),
        unsafe_allow_html=True,
    )




def render_topbar(
    worker_state: Optional[Dict[str, Any]],
    watchdog: Optional[Dict[str, Any]],
    pulse_view: Dict[str, Any],
    autonomy_view: Dict[str, Any],
):
    """Render the fixed top header bar."""
    status = str((worker_state or {}).get("status") or "").lower()
    seconds_since = _maybe_float((watchdog or {}).get("seconds_since_last"))

    health_class, health_label = derive_health_class(worker_state, watchdog)

    beat_txt = _humanize_seconds(seconds_since) if seconds_since is not None else "n/a"

    # Compact run id for mobile while keeping full id on hover (desktop).
    run_id = (worker_state or {}).get("run_id")
    run_id_full = str(run_id) if run_id else ""
    run_id_short = _abbrev_id(run_id_full, head=8, tail=4) if run_id_full else ""

    mode = (worker_state or {}).get("mode") or (worker_state or {}).get("run_mode")

    mid_parts: List[str] = []
    if run_id_short:
        rid_html = html.escape(run_id_short)
        rid_title = html.escape(run_id_full)
        mid_parts.append(f'Run <code title="{rid_title}">{rid_html}</code>')
        if mode:
            mid_parts.append(f"Mode <code>{html.escape(str(mode))}</code>")
    # Use a simple ASCII separator to avoid mojibake
    mid_txt = " | ".join(mid_parts) if mid_parts else "No active run detected"

    # Activity pulse (event-driven; does not attempt cycle counting)
    pv = pulse_view if isinstance(pulse_view, dict) else {}
    pulse_score = _maybe_float(pv.get("score")) or 0.0
    try:
        pulse_score = max(0.0, min(1.0, float(pulse_score)))
    except Exception:
        pulse_score = 0.0
    width_pct = pulse_score * 100.0

    pulse_label = str(pv.get("label") or "-").strip() or "-"
    right_txt = f"Pulse {pulse_label}" if pulse_label != "-" else "-"

    # Detail line (kept short)
    detail_parts: List[str] = []
    last_age = _maybe_float(pv.get("last_event_age_s"))
    if last_age is not None:
        detail_parts.append(f"Last event {_humanize_seconds(last_age)}")
    ev1m = _safe_int(pv.get("events_last_60s"), None)
    ev5m = _safe_int(pv.get("events_last_5m"), None)
    if isinstance(ev1m, int):
        detail_parts.append(f"1m {ev1m}")
    if isinstance(ev5m, int):
        detail_parts.append(f"5m {ev5m}")
    detail_txt = " | ".join(detail_parts)

    # Sanitize dynamic text to avoid stray mojibake
    autonomy_label = str(autonomy_view.get("label") or "Assisted")
    autonomy_score = _safe_int(autonomy_view.get("score"), 0)
    autonomy_html = html.escape(_to_ascii(f"{autonomy_label} ({autonomy_score}/4)"))

    status_html = html.escape(_to_ascii(status or "unknown"))

    detail_html = html.escape(_to_ascii(detail_txt)) if detail_txt else ""
    detail_line = f'<div class="ara-kv">{detail_html}</div>' if detail_html else ""

    st.markdown(
        textwrap.dedent(
            f"""
<div class="ara-topbar-wrap">
<div class="ara-topbar {health_class}">
<div class="ara-topbar-left">
<div class="ara-dot {health_class}"></div>
<div class="ara-topbar-title">{html.escape(_to_ascii(health_label))}</div>
<div class="ara-kv">Beat {html.escape(_to_ascii(beat_txt))} ago</div>
<div class="ara-kv">Status <code>{status_html}</code></div>
</div>

<div class="ara-topbar-mid">
<div class="ara-kv">{html.escape(_to_ascii(mid_txt))}</div>
</div>

<div class="ara-topbar-right">
<div class="ara-kv">{autonomy_html}</div>
<div class="ara-kv"><code>{html.escape(_to_ascii(right_txt))}</code></div>
{detail_line}
<div class="ara-mini-progress" title="activity pulse">
<div style="width:{width_pct}%"></div>
</div>
</div>
</div>
</div>
            """
        ).strip("\n"),
        unsafe_allow_html=True,
    )


def compute_autonomy_view(
    job_payload: Optional[Dict[str, Any]],
    worker_state: Optional[Dict[str, Any]],
    diagnostics_preview: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute a simple autonomy score (0..4) and label.

    Important UX goal:
    - If we *don't* have an active job/config context, avoid showing high autonomy levels based on
      stale diagnostics or defaults.
    """
    # Extract config (if present)
    cfg: Dict[str, Any] = {}
    if isinstance(job_payload, dict):
        cfg_raw = job_payload.get("config")
        if isinstance(cfg_raw, dict):
            cfg = cfg_raw

    has_cfg = bool(cfg)

    source_controls = cfg.get("source_controls") if has_cfg and isinstance(cfg.get("source_controls"), dict) else {}
    monitoring = cfg.get("monitoring") if has_cfg and isinstance(cfg.get("monitoring"), dict) else {}
    swarm_cfg = cfg.get("swarm") if has_cfg and isinstance(cfg.get("swarm"), dict) else {}

    status = str((worker_state or {}).get("status") or "").lower()
    running_like = status in {"running", "active", "in_progress", "working"}

    # Only treat "signals" as meaningful if we have an active job/config context or we're running.
    has_active_context = bool(job_payload) or running_like

    stable_signal = False
    if has_active_context and isinstance(diagnostics_preview, dict):
        stable_signal = bool(
            diagnostics_preview.get("stable_signal")
            or diagnostics_preview.get("equilibrium_detected")
            or diagnostics_preview.get("self_stabilizing")
        )
    # Fallback: if no stable signal was detected from diagnostics, attempt to
    # derive stability from the worker_state progress.  In some cases
    # diagnostics_preview may not yet include stable flags, but the
    # worker_state's extra.progress section may contain them.  This
    # ensures the autonomy score can advance to "Self-stabilizing" (4/4)
    # when the engine sets stability flags in the worker_state, even
    # if diagnostics are lagging.  Errors are swallowed.
    if not stable_signal and has_active_context and isinstance(worker_state, dict):
        try:
            extra = worker_state.get("extra") or {}
            if isinstance(extra, dict):
                progress = extra.get("progress") or {}
                if isinstance(progress, dict):
                    if progress.get("stable_signal") or progress.get("equilibrium_detected") or progress.get("self_stabilizing"):
                        stable_signal = True
        except Exception:
            pass

    # Elevate to self-stabilizing when the run appears complete.  If no
    # stability signal has been detected but the progress has reached or
    # exceeded its total, or if the worker status is finished-like, then
    # treat the run as self-stabilizing.  This ensures the autonomy
    # score can advance to 4/4 when the progress bar hits 100% even if
    # diagnostics and worker state have not yet emitted explicit stability
    # signals.  For swarm runs with many micro-cycles, current and total
    # may reflect internal step counts rather than macro cycles; this
    # fallback triggers when current >= total, which corresponds to a
    # completed run.
    finished_like_labels = {"finished", "done", "completed", "complete", "success"}
    if not stable_signal:
        try:
            # Compute progress fraction from worker_state if possible
            current = None
            total_v = None
            if isinstance(worker_state, dict):
                current = worker_state.get("effective_current") or worker_state.get("current")
                total_v = worker_state.get("effective_total") or worker_state.get("total")
            # Force numeric
            frac_complete = None
            try:
                if current is not None and total_v is not None:
                    c_val = float(current)
                    t_val = float(total_v)
                    if t_val > 0:
                        frac_complete = c_val / t_val
            except Exception:
                frac_complete = None
            if (frac_complete is not None and frac_complete >= 1.0) or status in finished_like_labels:
                stable_signal = True
        except Exception:
            pass

    # Features are only credited when config is present.
    has_tools = False
    is_monitored = False
    is_multi = False

    if has_cfg:
        has_tools = bool(
            source_controls.get("allow_remote_sources")
            or source_controls.get("allow_local_sources")
            or source_controls.get("enable_web")
            or source_controls.get("enable_web_search")
            or source_controls.get("enable_retrieval")
        )

        hb_enabled = monitoring.get("heartbeat_enabled")
        log_watchdog = monitoring.get("log_watchdog")
        is_monitored = bool(hb_enabled) or bool(log_watchdog)

        worker_count = _safe_int(swarm_cfg.get("worker_count"), 1)
        roles = cfg.get("roles") if isinstance(cfg.get("roles"), list) else []
        is_multi = bool(worker_count and worker_count > 1) or len(roles) > 1

    score = 0
    label = "Assisted"

    if is_multi:
        score = max(score, 1)
        label = "Collaborative"

    if has_tools:
        score = max(score, 2)
        label = "Tool-enabled"

    if is_monitored:
        score = max(score, 3)
        label = "Self-monitoring"

    if stable_signal:
        score = max(score, 4)
        label = "Self-stabilizing"

    return {"score": score, "label": label}
def _infer_agents_from_job_config(job_payload: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(job_payload, dict):
        return ["agent"]
    cfg = job_payload.get("config") if isinstance(job_payload.get("config"), dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    mode = str(cfg.get("mode") or "").lower()

    # Extract swarm configuration (may be under swarm_config or swarm)
    swarm_cfg = cfg.get("swarm_config") or cfg.get("swarm") or {}
    if not isinstance(swarm_cfg, dict):
        swarm_cfg = {}

    swarm_size = _safe_int(swarm_cfg.get("swarm_size"), 1) or 1

# (comment trimmed to keep this file renderable in GitHub)
    # (including the patched Streamlit UI) set roles on the top level of
    # the run configuration to include every mini agent in the swarm.  If
    # provided, use this list as the authoritative set of agents so that
    # all agents are rendered in the UI.  Fall back to swarm_config roles
# (comment trimmed to keep this file renderable in GitHub)
    roles_top = cfg.get("roles")
    if isinstance(roles_top, list) and roles_top:
        role_names = [str(r) for r in roles_top if str(r).strip()]
        if role_names:
            return role_names

    roles = swarm_cfg.get("roles")
    if isinstance(roles, list) and roles:
        role_names = [str(r) for r in roles if str(r).strip()]
        if role_names:
            return role_names

    if mode == "swarm" or swarm_size > 1:
        # Provide at least the canonical base roles (count = swarm_size)
        pairs = build_swarm_roles(True, swarm_size)
        return [p[0] for p in pairs] if pairs else ["agent"]

    if bool(cfg.get("multi_agent_pair")) or mode == "two_stage":
        return ["researcher", "critic"]

    return ["agent"]


def render_agent_presence(
    agents: List[str],
    active_agent: Optional[str] = None,
) -> None:
    if not agents:
        agents = ["agent"]

    st.markdown('<div class="ara-card">', unsafe_allow_html=True)
    st.markdown('<div class="ara-card-title">Agent presence</div>', unsafe_allow_html=True)
    st.markdown('<div class="ara-card-sub">Roles currently configured for the active run.</div>', unsafe_allow_html=True)

    # Deduplicate agent names to avoid duplicate chips in the UI
    seen_agents: Set[str] = set()
    unique_agents: List[str] = []
    for name in agents:
        name_str = str(name)
        if name_str not in seen_agents:
            seen_agents.add(name_str)
            unique_agents.append(name_str)

    chips: List[str] = []
    for a in unique_agents[:MAX_SWARM_AGENTS]:
        a_clean = str(a)
        is_active = active_agent and (
            a_clean == active_agent or a_clean.split("_", 1)[0] == str(active_agent)
        )
        cls = "ara-chip active" if is_active else "ara-chip"
        # Use a plain ASCII marker instead of a misencoded Unicode bullet
        chips.append(f'<span class="{cls}">* {html.escape(a_clean)}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def load_discovery_log(run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load discovery candidates for the UI.

    Preferred source: structured event stream (events.jsonl / event_log.json),
    filtered to the active run_id when provided.

    Fallback: legacy discovery_log.json / discoveries.json files.
    """
    # Best-effort inference of run_id from session (keeps backward compatibility)
    if not run_id:
        try:
            rid = st.session_state.get("active_run_id") or st.session_state.get("run_id")
        except Exception:
            rid = None
        if rid:
            run_id = str(rid)

    def _read_jsonl_events(p: Path, max_chars: int = 5_000_000) -> List[Dict[str, Any]]:
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            return []
        if not isinstance(txt, str) or not txt:
            return []
        # If the file is very large, keep the tail so we don't blow memory.
        if len(txt) > max_chars:
            txt = txt[-max_chars:]
        out: List[Dict[str, Any]] = []
        for line in txt.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            try:
                obj = json.loads(line_s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def _extract_candidates(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not events:
            return []
        rid = str(run_id) if run_id else None

        # Build a small cycle->RYE gain map if rye_update events exist.
        rye_gain_by_cycle: Dict[int, float] = {}
        for e in events:
            if not isinstance(e, dict):
                continue
            k = str(e.get("kind") or e.get("type") or "").lower()
            if k not in ("rye_update", "rye", "metrics", "rye_metrics"):
                continue
            cyc = e.get("cycle")
            try:
                cyc_i = int(cyc)
            except Exception:
                continue
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            val = d.get("delta_R") or d.get("delta_rye") or d.get("delta_RYE") or d.get("rye_gain")
            try:
                rye_gain_by_cycle[cyc_i] = float(val)
            except Exception:
                continue

        want_kinds = {
            "candidate_hypothesis",
            "candidate",
            "hypothesis_candidate",
            "discovery_candidate",
            "discovery",
        }

        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for e in events:
            if not isinstance(e, dict):
                continue

            # Filter explicit mismatches when run_id is known.
            if rid:
                e_rid = e.get("run_id") or e.get("job_id")
                if e_rid is not None and str(e_rid) != rid:
                    continue

            kind = str(e.get("kind") or e.get("type") or "").strip()
            kind_l = kind.lower()
            if kind_l not in want_kinds:
                continue

            data = e.get("data") if isinstance(e.get("data"), dict) else {}

            title = (
                data.get("title")
                or data.get("thesis")
                or data.get("name")
                or data.get("constraint")
                or e.get("title")
                or e.get("msg")
                or e.get("message")
                or kind
                or "candidate"
            )

            # Description: prefer constraint/mechanism/summary, then full text, then message
            desc = (
                data.get("constraint")
                or data.get("mechanism")
                or data.get("summary")
                or data.get("text")
                or e.get("msg")
                or e.get("message")
                or ""
            )

            dom = data.get("domain") or e.get("domain") or "general"
            cyc = e.get("cycle")
            role = e.get("role") or data.get("role") or "swarm"
            ts = e.get("timestamp") or e.get("ts") or ""

            evidence = data.get("citations") or data.get("sources") or data.get("evidence") or []
            ev_n = 0
            if isinstance(evidence, list):
                ev_n = len(evidence)
            elif isinstance(evidence, dict):
                ev_n = len(evidence)
            elif evidence:
                ev_n = 1

            gain = data.get("rye_gain") or data.get("delta_rye") or data.get("delta_R") or data.get("delta_RYE")
            if gain is None:
                try:
                    cyc_i = int(cyc)
                    if cyc_i in rye_gain_by_cycle:
                        gain = rye_gain_by_cycle[cyc_i]
                except Exception:
                    pass

            conf = data.get("confidence") or e.get("confidence")

            cand = {
                "title": title,
                "domain": dom,
                "description": desc,
                "evidence_count": ev_n,
                "rye_gain": gain,
                "confidence": conf,
                "cycle": cyc,
                "role": role,
                "timestamp": ts,
                # Keep the original payload for expanders/debugging if needed.
                "_event": e,
            }

            key = f"{str(title)[:120]}|{cyc}|{kind_l}"
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)

        return out

    # 1) Event-driven candidates (preferred)
    events: List[Dict[str, Any]] = []
    try:
        # JSON event logs (if present)
        ev_json, _src = load_event_log_unified(run_id)  # type: ignore[name-defined]
        if isinstance(ev_json, list):
            events.extend([e for e in ev_json if isinstance(e, dict)])
    except Exception:
        pass

    # JSONL per-run events (if present)
    try:
        if run_id:
            runs_root = Path(get_runs_root())
            p = runs_root / str(run_id) / "events.jsonl"
            if p.exists():
                events.extend(_read_jsonl_events(p))

        # Also check env-overridden log dirs (Render-friendly)
        for _k in ("ARA_RUNS_LOGS_DIR", "ARA_RUNS_LOG_DIR", "ARA_LOGS_DIR"):
            _v = os.getenv(_k)
            if not _v:
                continue
            try:
                ld = Path(_v).expanduser()
            except Exception:
                continue
            if run_id:
                p1 = ld / f"{str(run_id)}_events.jsonl"
                if p1.exists():
                    events.extend(_read_jsonl_events(p1))
            p2 = ld / "events.jsonl"
            if p2.exists():
                events.extend(_read_jsonl_events(p2))
    except Exception:
        pass

    candidates = _extract_candidates(events)
    if candidates:
        return candidates

    # 2) Legacy discovery logs (fallback)
    legacy_paths: List[Path] = []
    try:
        runs_logs = Path(get_runs_root()) / "logs"
        legacy_paths.extend(
            [
                runs_logs / "discovery_log.json",
                runs_logs / "discovery" / "discovery_log.json",
                runs_logs / "discovery" / "discoveries.json",
            ]
        )
    except Exception:
        pass

    legacy_paths.extend(
        [
            REPO_ROOT / "logs" / "discovery_log.json",
            REPO_ROOT / "logs" / "discovery" / "discovery_log.json",
            REPO_ROOT / "logs" / "discovery" / "discoveries.json",
        ]
    )

    for p in legacy_paths:
        data = _load_json_file(p)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]

    if _discovery_module is not None:
        try:
            func = getattr(_discovery_module, "load_discovery_log", None)
            if callable(func):
                data = func()
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass

    return []
def _filter_events_for_run(
    events: List[Dict[str, Any]],
    run_id: Optional[str],
    source_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Filter an events list to a specific run_id when possible.

    Many global logs contain events from multiple runs; this keeps the UI
    narrative feed pinned to the active run.
    """
    if not run_id:
        return events
    rid = str(run_id)
    out: List[Dict[str, Any]] = []
    sp = str(source_path) if source_path is not None else ""
    for e in events:
        if not isinstance(e, dict):
            continue
        erid = e.get("run_id") or e.get("job_id") or e.get("run") or e.get("runId")
        if erid is not None and str(erid) == rid:
            out.append(e)
        elif erid is None and sp and rid in sp:
            # Per-run files sometimes omit run_id on each event; keep them.
            out.append(e)
    if out:
        return out
    # If the source file itself is clearly per-run, return the events even
    # when no per-event run_id matched.
    if sp and rid in sp:
        return events
    return out

def load_event_log_unified(run_id: Optional[str], logs_dir_hint: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], str]:
    """Load event log from common paths.

    Supports both JSONL (append-only) and legacy JSON containers.

    If no event log exists yet, this returns an empty list (and the UI
    can synthesize a minimal timeline from history/state).
    """
    paths = _candidate_state_paths(run_id=run_id)["events"]

    # If the worker state exposes the actual logs directory it is writing to,
    # prefer that location so the UI can stay live.
    if isinstance(logs_dir_hint, Path):
        hinted: List[Path] = []
        try:
            ld = logs_dir_hint
            if run_id:
                rid = str(run_id)
                hinted.extend(
                    [
                        # Per-run JSONL
                        ld / f"{rid}_events.jsonl",
                        ld / f"{rid}_event_log.jsonl",
                        # Per-run legacy JSON
                        ld / f"{rid}_event_log.json",
                        ld / f"{rid}_events.json",
                        ld / f"{rid}_timeline.json",
                    ]
                )
            hinted.extend(
                [
                    # Global JSONL
                    ld / "events_global.jsonl",
                    ld / "events.jsonl",
                    # Global legacy JSON
                    ld / "event_log.json",
                    ld / "events.json",
                    ld / "timeline.json",
                ]
            )
        except Exception:
            hinted = []

        if hinted:
            # Prepend while preserving order and avoiding duplicates.
            new_paths: List[Path] = []
            for p in hinted + paths:
                if p not in new_paths:
                    new_paths.append(p)
            paths = new_paths

    # Prefer per-run log files when run_id is known. This avoids showing
    # events from a previous run when the global log file exists.
    per_run_paths: List[Path] = []
    global_paths: List[Path] = []
    if run_id:
        rid = str(run_id)
        for pp in paths:
            s = str(pp)
            if rid in pp.name or f"/{rid}/" in s:
                per_run_paths.append(pp)
            else:
                global_paths.append(pp)
    else:
        global_paths = paths

    def _load_events_from_file(pth: Path, max_lines: int = 4000) -> List[Dict[str, Any]]:
        """Return a list of event dicts from either JSONL or legacy JSON."""
        try:
            if str(pth).lower().endswith(".jsonl"):
                return _load_event_dicts_tail(pth, max_lines=max_lines)
        except Exception:
            pass

        raw = _load_json_file(pth)
        if isinstance(raw, dict):
            maybe = raw.get("events") or raw.get("timeline") or raw.get("items")
            if isinstance(maybe, list):
                return [e for e in maybe if isinstance(e, dict)]
            # Some emitters place the event list directly at the top level.
            evs = _normalize_event_container(raw)
            if evs:
                return evs
            return []

        if isinstance(raw, list):
            return [e for e in raw if isinstance(e, dict)]

        # Fallback: attempt tail parsing (covers some "one JSON object per line" dumps).
        return _load_event_dicts_tail(pth, max_lines=max_lines)

    def _first_existing_events(
        candidates: List[Path],
        filter_for_run: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
        """Return the first candidate file that yields events (optionally filtered to run_id)."""
        for pth in candidates:
            try:
                if not pth.exists():
                    continue
            except Exception:
                continue
            try:
                evs = _load_events_from_file(pth)
                if not evs:
                    continue
                if filter_for_run and run_id:
                    evs_f = _filter_events_for_run(evs, run_id=run_id, source_path=pth)
                    if not evs_f:
                        continue
                    return evs_f, pth
                return evs, pth
            except Exception:
                continue
        return [], None

    events: List[Dict[str, Any]] = []
    p: Optional[Path] = None

    if per_run_paths:
        events, p = _first_existing_events(per_run_paths, filter_for_run=bool(run_id))

    if not events:
        events, p = _first_existing_events(global_paths, filter_for_run=bool(run_id))

    if not events:
        return [], "not found"


    return events, str(p) if p else "unknown"


def _event_ts_to_str(ts_val: Any) -> str:
    if isinstance(ts_val, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(ts_val)).isoformat(timespec="seconds") + "Z"
        except Exception:
            return ""
    if isinstance(ts_val, str):
        return ts_val
    return ""



def _now_iso() -> str:
    """Current UTC time as ISO-8601 with 'Z' suffix."""
    try:
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    except Exception:
        # Last resort: avoid crashing if datetime/timezone is unavailable
        return ''

def build_narrative_events_from_history(history: List[Dict[str, Any]], limit: int = LIVE_EVENTS_LIMIT) -> List[Dict[str, Any]]:
    """Create a narrative event feed from cycle history."""
    if not history:
        return []

    events: List[Dict[str, Any]] = []
    tail = history[-limit:]
    for e in tail:
        cycle = e.get("cycle")
        role = e.get("role", "agent")
        domain = e.get("domain", "general")
        rye = e.get("RYE")
        if rye is None:
            rye = e.get("rye")
        d_r = e.get("delta_R")
        if d_r is None:
            d_r = e.get("delta_r")

        notes_n = len(e.get("notes_added") or [])
        hyps_n = len(e.get("hypotheses") or [])
        repairs_n = len(e.get("repairs") or [])

        parts = [f"Cycle {cycle} [{domain}/{role}]"]
        if isinstance(rye, (int, float)):
            parts.append(f"RYE {float(rye):.3f}")
        if isinstance(d_r, (int, float)):
            # Use a readable delta symbol instead of a misencoded character
            parts.append(f"\u0394R {float(d_r):.3f}")  # (trimmed)
        if repairs_n:
            parts.append(f"{repairs_n} repairs")
        if notes_n:
            parts.append(f"{notes_n} notes")
        if hyps_n:
            parts.append(f"{hyps_n} hypotheses")

        # Use a simple pipe separator instead of a misencoded bullet
        msg = " | ".join(parts)
        events.append(
            {
                "ts": e.get("timestamp") or "",
                "kind": "cycle",
                "message": msg,
            }
        )
    return events


def build_cycle_count_events_from_history(
    history: List[Dict[str, Any]],
    total_cycles: Optional[int] = None,
    limit: int = LIVE_EVENTS_LIMIT,
    current_cycle_hint: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Create a minimal event feed that only reports cycle count.

    This is useful when you want to hide low-signal worker/job events and keep the
    UI focused on progress (Cycle X/Y).
    """
    if not history:
        return []

    tail = history[-limit:]
    start_index = max(0, len(history) - len(tail))

    tot: Optional[int] = None
    try:
        if isinstance(total_cycles, int) and total_cycles > 0:
            tot = int(total_cycles)
    except Exception:
        tot = None
    # Fall back to the number of cycles we have so far (best effort).
    if tot is None:
        try:
            tot = int(len(history))
        except Exception:
            tot = None

    def _as_int(v: Any) -> Optional[int]:
        try:
            if isinstance(v, bool):
                return None
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                if v != v:  # NaN
                    return None
                return int(v)
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                    return int(s)
                return int(float(s))
        except Exception:
            return None
        return None

    # Prefer explicit cycle numbers when present so a truncated history window
    # doesn't make the UI appear to "stop" at a small fixed number (e.g. 25).
    cyc_vals: List[int] = []
    for e in tail:
        if not isinstance(e, dict):
            continue
        for k in ("cycle", "cycle_index", "current_cycle", "current"):
            iv = _as_int(e.get(k))
            if iv is not None:
                cyc_vals.append(iv)
                break

    # Detect 0-based counters and shift to 1-based for display.
    shift = 0
    try:
        if cyc_vals:
            mx = max(cyc_vals)
            if 0 in cyc_vals:
                shift = 1
            elif isinstance(tot, int) and tot > 0 and mx == (tot - 1):
                shift = 1
    except Exception:
        shift = 0

    # If cycles are not embedded in the history entries, use a stable start
    # derived from the current progress value when available.
    fallback_start: Optional[int] = None
    if not cyc_vals:
        try:
            cur_hint = _as_int(current_cycle_hint)
            if isinstance(cur_hint, int) and cur_hint > 0:
                fallback_start = max(1, int(cur_hint) - len(tail) + 1)
            else:
                fallback_start = start_index + 1
        except Exception:
            fallback_start = start_index + 1

    events: List[Dict[str, Any]] = []
    for i, e in enumerate(tail):
        # Determine display cycle
        cyc: Optional[int] = None
        if isinstance(e, dict):
            for k in ("cycle", "cycle_index", "current_cycle", "current"):
                cyc = _as_int(e.get(k))
                if cyc is not None:
                    break
        if cyc is not None:
            display_cycle = cyc + shift
            if display_cycle <= 0:
                display_cycle = 1
        else:
            display_cycle = (fallback_start or 1) + i

        # Timestamp
        ts = ""
        try:
            if isinstance(e, dict):
                ts_val = e.get("timestamp") or e.get("ts") or e.get("time") or ""
                ts = _event_ts_to_str(ts_val)
        except Exception:
            ts = ""
        if not ts:
            ts = _now_iso()

        if isinstance(tot, int) and tot > 0:
            msg = f"Cycle {display_cycle}/{tot}"
        else:
            msg = f"Cycle {display_cycle}"

        events.append(
            {
                "ts": ts,
                "kind": "cycle_progress",
                "message": msg,
                "cycle": display_cycle,
                "total": tot,
            }
        )
    return events



def build_cycle_count_events_from_event_log(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synthesize `cycle_progress` events from cycle-indexed entries in a raw event log.

    Some workers do not emit explicit "Cycle X/Y" events and may only emit a coarse
    progress counter (e.g. 0/N then N/N). However, most per-cycle work products
    (candidate hypotheses, metrics, etc.) carry a `cycle` field. This helper
    collapses those into one high-signal row per cycle.

    Returns a list of events shaped like:
      {"ts": "...", "kind": "cycle_progress", "message": "Cycle 3/6", ...}
    """
    def _as_int(v: Any) -> Optional[int]:
        """Best-effort int coercion; returns None when coercion fails."""
        try:
            if v is None:
                return None
            # Avoid surprising behaviour: bool is a subclass of int
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                # Guard NaN / inf
                if v != v or v == float("inf") or v == float("-inf"):
                    return None
                return int(v)
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                # Extract first integer-like token (handles "5/6", "cycle=3", etc.)
                m = re.search(r"-?\d+", s)
                return int(m.group(0)) if m else None
            return int(v)
        except Exception:
            return None

    if not events:
        return []

    # Collect earliest timestamp per raw cycle index.
    raw_cycles: set[int] = set()
    cycle_dt: Dict[int, Optional[datetime]] = {}
    saw_zero = False

    for ev in events:
        if not isinstance(ev, dict):
            continue

        kind = str(ev.get("kind") or ev.get("event") or ev.get("domain") or "").lower()
        if kind == "cycle_progress":
            continue

        raw = _as_int(ev.get("cycle"))
        if raw is None:
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            raw = _as_int(data.get("cycle"))

        if raw is None:
            continue

        if raw == 0:
            saw_zero = True

        raw_cycles.add(raw)

        dt = _event_ts_to_dt(ev.get("ts"))
        prev = cycle_dt.get(raw)
        if prev is None:
            cycle_dt[raw] = dt
        else:
            if dt is not None and (prev is None or dt < prev):
                cycle_dt[raw] = dt

    if not raw_cycles:
        return []

    # Best effort total cycle hint from the same event log.
    _, total_hint = _infer_cycle_progress_from_events(events)

    # Normalize to 1-based display if we detect a 0-based cycle index.
    display_cycles = sorted(
        {
            (c + 1 if saw_zero else c)
            for c in raw_cycles
            if (c + 1 if saw_zero else c) > 0
        }
    )
    if not display_cycles:
        return []

    total = total_hint if isinstance(total_hint, int) and total_hint > 0 else max(display_cycles)

    out: List[Dict[str, Any]] = []
    for dc in display_cycles:
        raw = dc - 1 if saw_zero else dc
        dt = cycle_dt.get(raw)

        ts = _now_iso()
        if isinstance(dt, datetime):
            # `_event_ts_to_dt` returns UTC (often naive). Format with a Z suffix.
            ts = dt.isoformat()
            if dt.tzinfo is None:
                ts = f"{ts}Z"
            elif ts.endswith("+00:00"):
                ts = ts.replace("+00:00", "Z")

        out.append(
            {
                "ts": ts,
                "kind": "cycle_progress",
                "message": f"Cycle {dc}/{total}",
                "cycle": dc,
                "total": total,
                "_synthetic": True,
                "_source": "event_log",
            }
        )

    return out

def build_cycle_count_events_from_progress(
    progress_view: Optional[Dict[str, Any]],
    run_id: Optional[str],
    limit: int = LIVE_EVENTS_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Build a small rolling timeline strictly from live counters.
    Never uses history and never fabricates intermediate cycles.

    This helper keeps a per-run rolling list in ``st.session_state`` keyed by the run ID,
    so the timeline grows as cycles advance even when the Streamlit script reruns frequently.
    """
    # require both a dict progress view and a run_id
    if not isinstance(progress_view, dict) or not run_id:
        return []

    # Helper to normalize arbitrary values to integers when possible
    def _as_int(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip() != "":
                return int(float(v.strip()))
        except Exception:
            return None
        return None

    # Extract current and total cycle counters, preferring explicit cycle fields
    cur = _as_int(progress_view.get("current"))
    tot = _as_int(progress_view.get("total"))
    cur_cycle = _as_int(progress_view.get("current_cycle"))
    tot_cycles = _as_int(progress_view.get("total_cycles"))
    if cur_cycle is not None:
        cur = cur_cycle
    if tot_cycles is not None:
        tot = tot_cycles

    # Bail if no current cycle information is available
    if cur is None:
        return []

    ss = st.session_state
    key_last = f"_ara_live_cycle_last::{run_id}"
    key_events = f"_ara_live_cycle_events::{run_id}"
    last = _as_int(ss.get(key_last))
    events = ss.get(key_events)
    if not isinstance(events, list):
        events = []
    events = [e for e in events if isinstance(e, dict)]

    # If the cycle count goes backwards (new run), reset state
    if last is not None and cur < last:
        last = None
        events = []

    # Message formatter
    def _msg(c: int) -> str:
        if isinstance(tot, int) and tot > 0:
            return f"Cycle {c}/{tot}"
        return f"Cycle {c}"

    # If this is the first observation on a new run and cycle 0, seed the timeline
    if (last is None) and cur == 0 and not events:
        events.append({"ts": _now_iso(), "kind": "cycle_progress", "message": _msg(0), "cycle": 0})
        last = 0

    # Append only what we observe; never fabricate intermediate cycles
    if last is None:
        last = -1
    if cur > last:
        events.append({"ts": _now_iso(), "kind": "cycle_progress", "message": _msg(cur), "cycle": cur})
        last = cur

    # Persist state back to the session
    ss[key_last] = last
    ss[key_events] = events

    # Return the tail of the events list, constrained by the limit
    if isinstance(limit, int) and limit > 0 and len(events) > limit:
        return events[-limit:]
    return events

def _filter_cycle_only_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only per-cycle progress events from a mixed event stream."""
    out: List[Dict[str, Any]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("kind") or ev.get("type") or ev.get("domain")
        kind_s = str(kind).strip().lower() if kind is not None else ""
        if kind_s in {"cycle_progress", "cycle"}:
            out.append(ev)
            continue
        # Back-compat: treat messages that look like "Cycle X/Y" as cycle progress.
        msg = ev.get("message") or ev.get("msg") or ev.get("text")
        if isinstance(msg, str) and msg.strip().lower().startswith("cycle "):
            out.append(ev)
    return out


def build_high_signal_events_from_event_log(
    events: List[Dict[str, Any]],
    *,
    limit: int = LIVE_EVENTS_LIMIT,
) -> List[Dict[str, Any]]:
    """Build a compact, human-friendly event feed from a raw event log.

    This favors *high-signal* events and skips noisy lifecycle/progress chatter.
    It does not attempt to infer cycle counts.

    The output is shaped for `render_narrative_feed`:
      {"ts": "...", "kind": "...", "message": "..."}
    """
    if not events:
        return []

    noise_domains = {"progress", "heartbeat", "watchdog"}
    noise_kinds = {
        "cycle_progress",
        "progress",
        "progress_update",
        "heartbeat",
        "watchdog_heartbeat",
        "job_running",
        "job_claimed",
        "worker_state",
        "run_state",
    }

    out: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue

        dom = str(ev.get("domain") or "").strip().lower()
        kind_raw = ev.get("kind") or ev.get("type") or ev.get("event") or ev.get("domain") or "event"
        kind_s = str(kind_raw).strip().lower()

        # Skip noise first
        if dom in noise_domains:
            continue
        if kind_s in noise_kinds:
            continue
        if "heartbeat" in kind_s:
            continue
        if "progress" in kind_s:
            continue

        msg = ev.get("message") or ev.get("msg") or ev.get("text") or ev.get("summary")
        if msg is None:
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            if isinstance(data, dict):
                msg = data.get("note") or data.get("message") or data.get("summary")

        if isinstance(msg, str):
            msg_s = msg.strip()
        else:
            msg_s = str(msg).strip() if msg is not None else ""

        if not msg_s:
            continue

        # Avoid duplicative lines even when domain/kind is missing.
        msg_low = msg_s.lower()
        if msg_low.startswith("progress "):
            continue
        if msg_low.startswith("[hb]") or msg_low.startswith("hb "):
            continue

        ts_val = _coalesce(ev.get("ts"), ev.get("timestamp"), ev.get("utc"), ev.get("time"))
        ts_s = _event_ts_to_str(ts_val) if ts_val is not None else ""
        if not ts_s:
            ts_s = _now_iso()

        out.append({"ts": ts_s, "kind": str(kind_raw), "message": msg_s})

    if not out:
        return []

    if isinstance(limit, int) and limit > 0 and len(out) > limit:
        return out[-limit:]
    return out



def render_narrative_feed(events: List[Dict[str, Any]], source_label: str = "") -> None:
    st.markdown('<div class="ara-card">', unsafe_allow_html=True)
    title = "Recent activity"
    if source_label and source_label != "not found":
        title += f" (source: {html.escape(source_label)})"
    st.markdown(f'<div class="ara-card-title">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="ara-card-sub">A readable timeline. Raw logs stay optional.</div>', unsafe_allow_html=True)

    if not events:
        st.markdown(
            '<div class="ara-feed"><div class="ara-event"><div class="ara-event-msg">No events yet.</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return
    # Keep only last N and reverse so newest at top
    tail = events[-LIVE_EVENTS_LIMIT:]
    tail = list(reversed(tail))

    rows = []
    for ev in tail:
        ts = _event_ts_to_str(ev.get("ts") or ev.get("timestamp") or "")
        kind = str(ev.get("kind") or ev.get("domain") or ev.get("type") or "event")
        msg = str(ev.get("message") or ev.get("text") or ev.get("summary") or "")
        if not msg:
            continue
        # Use a simple pipe separator in place of a misencoded bullet
        meta = " | ".join([x for x in [ts, kind] if x])
        rows.append(
            f"""
<div class="ara-event">
  <div class="ara-event-meta">{html.escape(meta)}</div>
  <div class="ara-event-msg">{html.escape(msg)}</div>
</div>
            """
        )

    st.markdown(f'<div class="ara-feed">{"".join(rows)}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _infer_discovery_confidence(d: Dict[str, Any]) -> Optional[float]:
    conf = d.get("confidence")
    if isinstance(conf, (int, float)):
        return _clamp_float(float(conf), 0.0, 1.0)
    # Infer from rye_gain if present
    gain = d.get("rye_gain") or d.get("delta_rye") or d.get("delta_RYE")
    if isinstance(gain, (int, float)):
        # Soft scale: 0.0..1.0 as gain 0..1 (cap)
        return _clamp_float(float(gain), 0.0, 1.0)
    return None


def render_discovery_cards(discoveries: List[Dict[str, Any]]) -> None:
    st.markdown("### Discovery candidates")

    if not discoveries:
        st.info("No discovery entries found yet.")
        return

    # Sort by confidence then gain
    def _score(d: Dict[str, Any]) -> float:
        c = _infer_discovery_confidence(d)
        gain = d.get("rye_gain") or d.get("delta_rye") or 0.0
        try:
            g = float(gain)
        except Exception:
            g = 0.0
        return (c or 0.0) * 1.2 + g * 0.2

    sorted_disc = sorted(discoveries, key=_score, reverse=True)
    top = sorted_disc[:DISCOVERY_CARDS_LIMIT]

    cols = st.columns(3)
    for i, d in enumerate(top):
        with cols[i % 3]:
            title = d.get("title") or d.get("summary") or d.get("id") or "Discovery"
            title = str(title).strip()
            if len(title) > 70:
                title = title[:70] + "..."
            domain = str(d.get("domain") or "general")
            conf = _infer_discovery_confidence(d)
            gain = d.get("rye_gain") or d.get("delta_rye") or d.get("delta_RYE")
            try:
                gain_f = float(gain) if gain is not None else None
            except Exception:
                gain_f = None
            evidence = d.get("evidence") or d.get("citations") or d.get("sources") or []
            ev_n = len(evidence) if isinstance(evidence, list) else 0

            desc = d.get("description") or d.get("details") or d.get("text") or ""
            desc = str(desc).strip()
            if len(desc) > 160:
                desc = desc[:160] + "..."

            conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
            gain_txt = f"{gain_f:.3f}" if isinstance(gain_f, (int, float)) else "n/a"

            # Render discovery candidate card with ASCII separators to avoid mojibake.
            st.markdown(
                f"""
<div class="ara-card">
  <div class="ara-card-title">{html.escape(_to_ascii(title))}</div>
  <div class="ara-card-sub">{html.escape(_to_ascii(domain))} | confidence {html.escape(_to_ascii(conf_txt))} | RYE gain {html.escape(_to_ascii(gain_txt))} | evidence {ev_n}</div>
  <div class="ara-card-mono">{html.escape(_to_ascii(desc)) if desc else "-"}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def safe_json_preview(
    obj: Any,
    max_chars: int = 200_000,
    max_items: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Convert an object to JSON for display with size limits.

    Returns (json_string_or_none, info_message_or_none).
    """
    note_parts: List[str] = []

    if max_items is not None and isinstance(obj, list) and len(obj) > max_items:
        obj = obj[-max_items:]
        note_parts.append(f"Showing last {max_items} items from a larger array.")

    try:
        s = json.dumps(obj, indent=2)
    except TypeError:
        return None, "Object contains non JSON serializable entries."

    if len(s) > max_chars:
        s = s[:max_chars] + "\n... (truncated)"
        note_parts.append(f"Output truncated to {max_chars} characters for display.")

    note = " ".join(note_parts) if note_parts else None
    return s, note


# -------------------------------------------------------------------
# Snapshot + hypotheses + citations + verification helpers (unchanged)
# -------------------------------------------------------------------
def load_snapshots() -> List[Dict[str, Any]]:
    """Load snapshot JSON files as a list of {name, timestamp, data}."""
    snapshot_dir_candidates: List[Path] = []

    try:
        runs_logs = Path(get_runs_root()) / "logs"
        snapshot_dir_candidates.extend([runs_logs / "snapshots", runs_logs / "snapshot"])
    except Exception:
        pass

    snapshot_dir_candidates.extend([REPO_ROOT / "logs" / "snapshots", REPO_ROOT / "logs" / "snapshot"])

    try:
        runs_root_path = Path(get_runs_root())
        if runs_root_path.exists() and runs_root_path.is_dir():
            for run_dir in runs_root_path.iterdir():
                if not run_dir.is_dir():
                    continue
                snap_dir = run_dir / "snapshots"
                if snap_dir.exists() and snap_dir.is_dir():
                    snapshot_dir_candidates.append(snap_dir)
            # Additionally include snapshots created via MemoryStore.  These live
            # under "runs_root/snapshots/<run_id>" and will be missed by the
            # simple per-run scan above.  Append each run-specific folder here.
            try:
                snap_root = runs_root_path / "snapshots"
                if snap_root.exists() and snap_root.is_dir():
                    for sub_dir in snap_root.iterdir():
                        if sub_dir.is_dir():
                            snapshot_dir_candidates.append(sub_dir)
            except Exception:
                pass
    except Exception:
        pass

    if isinstance(RUNS_FINISHED_DIR, Path):
        try:
            for run_dir in RUNS_FINISHED_DIR.iterdir():
                if not run_dir.is_dir():
                    continue
                snap_dir = run_dir / "snapshots"
                if snap_dir.exists() and snap_dir.is_dir():
                    snapshot_dir_candidates.append(snap_dir)
        except Exception:
            pass

    seen_dirs: Set[str] = set()
    unique_snapshot_dirs: List[Path] = []
    for d in snapshot_dir_candidates:
        try:
            key = str(d.resolve())
        except Exception:
            key = str(d)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        unique_snapshot_dirs.append(d)

    snapshots: List[Dict[str, Any]] = []
    for base in unique_snapshot_dirs:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.glob("*.json")):
            data = _load_json_file(path)
            if not isinstance(data, dict):
                continue
            ts_val = data.get("timestamp") or data.get("timestamp_utc") or data.get("created_at")
            try:
                ts = _parse_timestamp_str(ts_val) if isinstance(ts_val, str) else None
            except Exception:
                ts = None
            snapshots.append(
                {"name": path.name, "path": str(path), "timestamp": ts, "raw_timestamp": ts_val, "data": data}
            )
    snapshots.sort(key=lambda s: s["timestamp"] or datetime.min)
    return snapshots


def extract_hypotheses_from_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten hypotheses across all cycles into a list with cycle info."""
    results: List[Dict[str, Any]] = []
    for entry in history:
        cycle_idx = entry.get("cycle")
        role = entry.get("role", "agent")
        domain = entry.get("domain", "general")
        ts = entry.get("timestamp")
        hyps = entry.get("hypotheses") or []
        for h in hyps:
            if isinstance(h, dict):
                text = h.get("text", "")
                conf = h.get("confidence")
            else:
                text = str(h)
                conf = None
            if not text:
                continue
            results.append(
                {"cycle": cycle_idx, "role": role, "domain": domain, "timestamp": ts, "text": text, "confidence": conf}
            )
    return results


def extract_citation_rows_from_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten citations across all cycles into rows with cycle info for the citation viewer."""
    results: List[Dict[str, Any]] = []
    for entry in history:
        cycle_idx = entry.get("cycle")
        role = entry.get("role", "agent")
        domain = entry.get("domain", "general")
        ts = entry.get("timestamp")

        raw_cites = entry.get("citations") or entry.get("sources") or entry.get("source_list") or []
        for c in raw_cites:
            if not isinstance(c, dict):
                url = str(c)
                results.append(
                    {
                        "cycle": cycle_idx,
                        "role": role,
                        "domain": domain,
                        "timestamp": ts,
                        "source": "",
                        "title": url,
                        "url": url,
                        "snippet": "",
                    }
                )
                continue

            source = c.get("source") or c.get("provider") or ""
            title = c.get("title") or ""
            url = c.get("url") or c.get("link") or ""
            snippet = c.get("snippet") or c.get("summary") or ""
            results.append(
                {
                    "cycle": cycle_idx,
                    "role": role,
                    "domain": domain,
                    "timestamp": ts,
                    "source": source,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
    return results


def load_verification_log() -> List[Dict[str, Any]]:
    """Try to load verification log entries from standard locations."""
    candidates: List[Path] = []
    try:
        runs_logs = Path(get_runs_root()) / "logs"
        candidates.extend(
            [
                runs_logs / "verification_log.json",
                runs_logs / "verification" / "verification_log.json",
                runs_logs / "verification" / "results.json",
            ]
        )
    except Exception:
        pass

    candidates.extend(
        [
            REPO_ROOT / "logs" / "verification_log.json",
            REPO_ROOT / "logs" / "verification" / "verification_log.json",
            REPO_ROOT / "logs" / "verification" / "results.json",
        ]
    )

    for p in candidates:
        data = _load_json_file(p)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    if _verification_module is not None:
        try:
            func = getattr(_verification_module, "load_verification_log", None)
            if callable(func):
                data = func()
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass
    return []


def equilibrium_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract equilibrium related metrics from a snapshot if present."""
    # Extract metrics from a snapshot.  In some cases the snapshot may not include a
    # dedicated 'metrics' key, but diagnostic values may be flattened into
    # 'diagnostics' or the top-level snapshot.  Fall back through several
    # potential sources.
    metrics: Any = snapshot.get("metrics") or snapshot.get("rye_metrics")
    if metrics is None:
        # Fallback to diagnostics dict if present
        diag_val = snapshot.get("diagnostics")
        if isinstance(diag_val, dict):
            metrics = diag_val
        # As a last resort, consider the snapshot itself as the metrics dict if it
        # contains expected fields; this allows equilibrium views to work even when
        # metrics are flattened.
        elif isinstance(snapshot, dict):
            metrics = snapshot
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "rye_avg": metrics.get("rye_avg"),
        "stability_index": metrics.get("stability_index"),
        "coherence_plateau": metrics.get("coherence_plateau"),
        "equilibrium_fraction": metrics.get("equilibrium_fraction"),
    }


def _safe_gv_id(prefix: str, raw: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    if not clean:
        clean = "node"
    return f"{prefix}{clean}"


def _clean_label_text(text: str, max_len: int = 60) -> str:
    text = str(text)
    text = text.replace('"', "'").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def build_insight_graph(history: List[Dict[str, Any]], discoveries: List[Dict[str, Any]]) -> str:
    """Build a Graphviz DOT string linking goals, roles, hypotheses, and discoveries."""
    nodes: List[str] = []
    edges: List[str] = []
    nodes.append('run [label="Run", shape=box, style=filled, fillcolor="#eeeeee"]')

    domains = sorted({str(e.get("domain", "general")) for e in history})
    domain_ids: Dict[str, str] = {}
    for d in domains:
        safe_d_label = _clean_label_text(f"Domain: {d}")
        node_id = _safe_gv_id("domain_", d)
        domain_ids[d] = node_id
        nodes.append(f'{node_id} [label="{safe_d_label}", shape=box]')
        edges.append(f"run -> {node_id}")

    roles = sorted({str(e.get("role", "agent")) for e in history})
    role_ids: Dict[str, str] = {}
    for r in roles:
        safe_r_label = _clean_label_text(f"Role: {r}")
        node_id = _safe_gv_id("role_", r)
        role_ids[r] = node_id
        nodes.append(f'{node_id} [label="{safe_r_label}", shape=ellipse]')
        edges.append(f"run -> {node_id}")

    hyps = extract_hypotheses_from_history(history)
    scored = []
    for h in hyps:
        conf = h.get("confidence")
        score = float(conf) if isinstance(conf, (int, float)) else 0.0
        scored.append((score, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_hyps = [h for _, h in scored[:8]]

    for idx, h in enumerate(top_hyps):
        label_text = _clean_label_text(h["text"])
        hyp_id = f"hyp_{idx}"
        nodes.append(f'{hyp_id} [label="H: {label_text}", shape=note]')
        d = str(h.get("domain", "general"))
        r = str(h.get("role", "agent"))
        d_id = domain_ids.get(d, _safe_gv_id("domain_", d))
        r_id = role_ids.get(r, _safe_gv_id("role_", r))
        if d_id not in domain_ids.values():
            safe_d2_label = _clean_label_text(f"Domain: {d}")
            nodes.append(f'{d_id} [label="{safe_d2_label}", shape=box]')
        if r_id not in role_ids.values():
            safe_r2_label = _clean_label_text(f"Role: {r}")
            nodes.append(f'{r_id} [label="{safe_r2_label}", shape=ellipse]')
        edges.append(f"{d_id} -> {hyp_id}")
        edges.append(f"{r_id} -> {hyp_id}")

    top_disc: List[Dict[str, Any]] = []
    if discoveries:
        scored_disc = []
        for d in discoveries:
            gain = d.get("rye_gain") or d.get("delta_rye") or 0.0
            try:
                gain_f = float(gain)
            except Exception:
                gain_f = 0.0
            scored_disc.append((gain_f, d))
        scored_disc.sort(key=lambda x: x[0], reverse=True)
        top_disc = [d for _, d in scored_disc[:8]]

    for idx, d in enumerate(top_disc):
        label_src = d.get("title") or d.get("summary") or d.get("id") or f"Discovery {idx + 1}"
        label_text = _clean_label_text(label_src)
        disc_id = f"disc_{idx}"
        nodes.append(f'{disc_id} [label="D: {label_text}", shape=diamond]')
        edges.append(f"run -> {disc_id}")

    dot_lines = ["digraph G {", "rankdir=LR;", 'node [fontname="Helvetica"];']
    dot_lines.extend(nodes)
    dot_lines.extend(edges)
    dot_lines.append("}")
    return "\n".join(dot_lines)


def build_breakthrough_report(history: List[Dict[str, Any]], discoveries: List[Dict[str, Any]]) -> str:
    """Build a markdown style breakthrough snapshot report from history and discovery log."""
    lines: List[str] = []
    lines.append("# Breakthrough snapshot report\n")

    if not history and not discoveries:
        lines.append("No cycles or discoveries recorded yet.")
        return "\n".join(lines)

    lines.append(
        "This report highlights candidate breakthroughs based on high RYE, strong delta_R, "
        "and discovery log entries. It is an autonomous research artifact, not medical advice.\n"
    )

    scored_cycles: List[Tuple[float, Dict[str, Any]]] = []
    for e in history:
        rye_val = e.get("RYE")
        if rye_val is None:
            rye_val = e.get("rye")
        d_r = e.get("delta_R")
        if d_r is None:
            d_r = e.get("delta_r")

        if isinstance(rye_val, (int, float)):
            score = float(rye_val)
        elif isinstance(d_r, (int, float)):
            score = float(d_r)
        else:
            continue
        scored_cycles.append((score, e))

    scored_cycles.sort(key=lambda x: x[0], reverse=True)
    top_cycles = [e for _, e in scored_cycles[:10]]

    lines.append("## Top cycles by efficiency and improvement\n")
    if not top_cycles:
        lines.append("No cycles with numeric RYE or delta_R found.\n")
    else:
        for e in top_cycles:
            cycle_idx = e.get("cycle")
            role = e.get("role", "agent")
            domain = e.get("domain", "general")

            rye_val = e.get("RYE")
            if rye_val is None:
                rye_val = e.get("rye")

            d_r = e.get("delta_R")
            if d_r is None:
                d_r = e.get("delta_r")

            energy_e = e.get("energy_E")
            if energy_e is None:
                energy_e = e.get("energy")

            ts = e.get("timestamp")

            header = f"- Cycle {cycle_idx} [{domain}/{role}]"
            metrics_parts = []
            if isinstance(rye_val, (int, float)):
                metrics_parts.append(f"RYE={rye_val:.3f}")
            if isinstance(d_r, (int, float)):
                metrics_parts.append(f"delta_R={d_r:.3f}")
            if isinstance(energy_e, (int, float)):
                metrics_parts.append(f"E={energy_e:.3f}")
            if ts:
                metrics_parts.append(f"time={ts}")
            if metrics_parts:
                header += " (" + ", ".join(metrics_parts) + ")"
            lines.append(header)

            notes = e.get("notes_added") or []
            hyps = e.get("hypotheses") or []
            details_added = 0
            for n in notes:
                txt = str(n).strip()
                if not txt:
                    continue
                if len(txt) > 220:
                    txt = txt[:220] + "..."
                lines.append(f"  - Note: {txt}")
                details_added += 1
                if details_added >= 2:
                    break
            if details_added < 2:
                for h in hyps:
                    txt = h.get("text", "") if isinstance(h, dict) else str(h)
                    txt = txt.strip()
                    if not txt:
                        continue
                    if len(txt) > 220:
                        txt = txt[:220] + "..."
                    lines.append(f"  - Hypothesis: {txt}")
                    details_added += 1
                    if details_added >= 2:
                        break

    lines.append("\n## Discovery log highlights\n")
    if not discoveries:
        lines.append("No discovery log entries found.\n")
    else:
        scored_disc: List[Tuple[float, Dict[str, Any]]] = []
        for d in discoveries:
            gain = d.get("rye_gain") or d.get("delta_rye") or 0.0
            try:
                gain_f = float(gain)
            except Exception:
                gain_f = 0.0
            scored_disc.append((gain_f, d))
        scored_disc.sort(key=lambda x: x[0], reverse=True)
        top_disc = [d for _, d in scored_disc[:10]]

        for d in top_disc:
            dom = d.get("domain", "general")
            label = d.get("title") or d.get("summary") or d.get("id") or "discovery"
            gain = d.get("rye_gain") or d.get("delta_rye") or 0.0
            try:
                gain_f = float(gain)
                gain_txt = f"{gain_f:.3f}"
            except Exception:
                gain_txt = str(gain)

            header = f"- [{dom}] {label} (approx RYE gain {gain_txt})"
            lines.append(header)

            desc = d.get("description") or d.get("details") or ""
            if desc:
                txt = str(desc).strip()
                if len(txt) > 260:
                    txt = txt[:260] + "..."
                lines.append(f"  - Description: {txt}")

    lines.append(
        "\nThis snapshot is designed to give a human reviewer a short list of high impact cycles "
        "and discovery candidates to investigate further."
    )

    return "\n".join(lines)


def load_citations_from_finished_runs(limit_runs: int = 20) -> List[Dict[str, Any]]:
    """Extract citations from finished run JSONs as a fallback for the citation viewer."""
    jobs = _list_jobs_by_status_candidates(["finished", "done", "completed", "complete", "success"])
    if not jobs:
        return []

    try:
        jobs_sorted = sorted(jobs, key=_job_created_at_ts)
    except Exception:
        jobs_sorted = jobs

    jobs_slice = jobs_sorted[-limit_runs:]

    all_history: List[Dict[str, Any]] = []
    top_level_citations: List[Dict[str, Any]] = []

    for job in jobs_slice:
        run_id = _get_job_id(job)

        created_at_raw = (
            getattr(job, "created_at", None)
            if hasattr(job, "created_at")
            else (job.get("created_at") if isinstance(job, dict) else None)
        )
        default_ts = None
        if isinstance(created_at_raw, (int, float)):
            try:
                default_ts = datetime.utcfromtimestamp(created_at_raw).isoformat() + "Z"
            except Exception:
                default_ts = None
        elif isinstance(created_at_raw, str):
            default_ts = created_at_raw

        result = load_job_result(run_id)
        if not isinstance(result, dict):
            continue

        payload = result.get("result")
        base = payload if isinstance(payload, dict) else result

        cycles = _extract_cycles_from_run_result(result, run_id=run_id, default_timestamp=default_ts)
        all_history.extend(cycles)

        cites = base.get("citations") or base.get("sources") or base.get("source_list") or []
        if isinstance(cites, list):
            for c in cites:
                if not isinstance(c, dict):
                    url = str(c)
                    top_level_citations.append(
                        {
                            "cycle": None,
                            "role": "run",
                            "domain": base.get("domain") or "general",
                            "timestamp": default_ts,
                            "source": "",
                            "title": url,
                            "url": url,
                            "snippet": "",
                        }
                    )
                    continue

                source = c.get("source") or c.get("provider") or ""
                title = c.get("title") or ""
                url = c.get("url") or c.get("link") or ""
                snippet = c.get("snippet") or c.get("summary") or ""

                ts_val = base.get("timestamp") or result.get("timestamp") or default_ts
                if isinstance(ts_val, (int, float)):
                    try:
                        ts_val = datetime.utcfromtimestamp(ts_val).isoformat() + "Z"
                    except Exception:
                        ts_val = None

                top_level_citations.append(
                    {
                        "cycle": None,
                        "role": c.get("role") or "run",
                        "domain": c.get("domain") or base.get("domain") or "general",
                        "timestamp": ts_val,
                        "source": source,
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                )

    flattened = extract_citation_rows_from_history(all_history)
    flattened.extend(top_level_citations)
    return flattened


def load_discoveries_from_finished_runs(limit_runs: int = 20) -> List[Dict[str, Any]]:
    """Extract discovery entries from finished run JSONs as a fallback for the discovery tab."""
    jobs = _list_jobs_by_status_candidates(["finished", "done", "completed", "complete", "success"])
    if not jobs:
        return []

    try:
        jobs_sorted = sorted(jobs, key=_job_created_at_ts)
    except Exception:
        jobs_sorted = jobs

    jobs_slice = jobs_sorted[-limit_runs:]

    discoveries: List[Dict[str, Any]] = []

    for job in jobs_slice:
        run_id = _get_job_id(job)

        created_at_raw = (
            getattr(job, "created_at", None)
            if hasattr(job, "created_at")
            else (job.get("created_at") if isinstance(job, dict) else None)
        )
        default_ts = None
        if isinstance(created_at_raw, (int, float)):
            try:
                default_ts = datetime.utcfromtimestamp(created_at_raw).isoformat() + "Z"
            except Exception:
                default_ts = None
        elif isinstance(created_at_raw, str):
            default_ts = created_at_raw

        result = load_job_result(run_id)
        if not isinstance(result, dict):
            continue

        payload = result.get("result")
        base = payload if isinstance(payload, dict) else result

        candidates = base.get("discoveries") or base.get("discovery_candidates") or base.get("discovery_log") or []
        if not isinstance(candidates, list):
            continue

        ts_val = base.get("timestamp") or result.get("timestamp") or default_ts
        if isinstance(ts_val, (int, float)):
            try:
                ts_val = datetime.utcfromtimestamp(ts_val).isoformat() + "Z"
            except Exception:
                ts_val = None

        for d in candidates:
            if not isinstance(d, dict):
                continue
            d2 = dict(d)
            if "run_id" not in d2:
                d2["run_id"] = run_id
            if "domain" not in d2:
                d2["domain"] = base.get("domain", "general")
            if "timestamp" not in d2:
                d2["timestamp"] = ts_val
            discoveries.append(d2)

    return discoveries


def _looks_like_job_payload_json(obj: Any) -> bool:
    """Heuristic: detect legacy job JSONs (avoid deleting results)."""
    if not isinstance(obj, dict):
        return False
    # Results often include "result" or "results"
    if "result" in obj or "results" in obj:
        return False
    cfg = obj.get("config")
    if isinstance(cfg, dict) and cfg:
        return True
    # Flat/legacy
    if "goal" in obj and any(k in obj for k in ("mode", "total_cycles", "max_cycles", "runtime_hints", "source_controls")):
        return True
    return False


# -------------------------------------------------------------------
# Main Streamlit app
# -------------------------------------------------------------------
def main() -> None:
    inject_base_styles()

    # Resolve base + queue paths once for UI display
    runs_base_dir = get_runs_root()
    queue_root_dir = get_queue_root()
    queue_pending_dir = str(Path(queue_root_dir) / "pending")

    # Shared MemoryStore instance (read-only UI)
    memory = init_memory_store()

    # Load worker state and global run_state early (used for active run detection)
    ws0, ws_src0 = load_worker_state_unified(memory)
    run_state_global, run_state_global_src = load_run_state_unified(memory, run_id_hint=None)

    # Determine active run id as early as possible.
    # Prefer run_state.active_run_id (authoritative), then a running worker's run_id, then UI hint, then queue scan.
    active_run_hint = st.session_state.get("active_run_id_hint")
    if active_run_hint is not None:
        active_run_hint = str(active_run_hint)

    status0 = str((ws0 or {}).get("status") or "").lower()
    # Treat additional engine-worker statuses as "running" for active run detection.
    running_like = status0 in {
        "running",
        "active",
        "in_progress",
        "working",
        "running_job",
        "running_cycle",
        "processing",
        "busy",
    }

    active_run_from_run_state: Optional[str] = None
    if isinstance(run_state_global, dict):
        ar0 = run_state_global.get("active_run_id")
        if ar0 not in (None, "", "null", "NULL"):
            active_run_from_run_state = str(ar0)

    ws_run_id = (ws0 or {}).get("run_id")
    ws_run_id = str(ws_run_id) if ws_run_id is not None else None

    active_run_id = (
        (ws_run_id if ws_run_id else None)
        or active_run_from_run_state
        or active_run_hint
        or _derive_active_run_id_from_queue()
    )

    # If the worker is actively running a run_id, prefer it over a stale
    # global run_state.active_run_id. This prevents the UI topbar and
    # narrative feed from drifting to different runs.
    if ws_run_id:
        try:
            if active_run_id and str(active_run_id) != str(ws_run_id):
                active_run_id = str(ws_run_id)
        except Exception:
            active_run_id = str(ws_run_id)

    # Light history preview for narrative synthesis and stability/autonomy (last ~25 only)
    history_preview: List[Dict[str, Any]] = []
    get_cycle_history_preview = getattr(memory, "get_cycle_history", None)
    if callable(get_cycle_history_preview):
        try:
            hist = get_cycle_history_preview() or []
            if isinstance(hist, list):
                history_preview = [e for e in hist if isinstance(e, dict)][-25:]
        except Exception:
            history_preview = []
    if not history_preview:
        # fallback to finished runs (small)
        history_preview = load_history_from_finished_runs(limit_runs=5)[-25:]

    # Filter the preview to cycles from the active run only when available
    if active_run_id:
        history_preview = _get_cycle_history_for_run(history_preview, active_run_id)

    # Diagnostics sources (for top bar + live console)
    if active_run_id:
        run_state0, run_state_src0 = load_run_state_unified(memory, run_id_hint=active_run_id)
    else:
        run_state0, run_state_src0 = run_state_global, run_state_global_src
    watchdog0, watchdog_src0 = load_watchdog_info_unified(memory, run_id=active_run_id)

    # If no real watchdog heartbeat is available, synthesize one from last activity
    if not _is_meaningful_watchdog(watchdog0):
        synth_wd, synth_src = synthesize_watchdog_from_activity(ws0, run_state0, history_preview)
        if synth_wd:
            watchdog0 = synth_wd
            watchdog_src0 = synth_src

    progress0_raw, progress_src0 = load_progress_unified(active_run_id)
    progress_view0 = compute_progress_view(
        ws0 if active_run_id else None,
        progress0_raw if active_run_id else None,
        watchdog0,
        run_id=active_run_id,
    )

    diagnostics_preview: Optional[Dict[str, Any]] = None
    if history_preview:
        try:
            diagnostics_preview = build_run_diagnostics(history=history_preview, domain=None, window=10)
        except Exception:
            diagnostics_preview = None

    # Autonomy + agents (from job payload if available)
    job_payload0, job_payload_src0 = (
        load_job_payload_from_disk(active_run_id) if active_run_id else (None, "no run_id")
    )
    autonomy_view0 = compute_autonomy_view(job_payload0, ws0, diagnostics_preview)
    agents0 = _infer_agents_from_job_config(job_payload0)

    # Event log (best effort) + activity pulse (event-driven)
    # Hint for where the worker is writing logs (helps avoid "stale" views when
    # Streamlit can't otherwise infer the logs directory).
    logs_dir_hint0: Optional[Path] = None
    try:
        _ld_hint = (ws0 or {}).get("runs_logs_dir") or (ws0 or {}).get("logs_dir") or (ws0 or {}).get("runs_log_dir")
        if _ld_hint:
            logs_dir_hint0 = Path(str(_ld_hint)).expanduser()
    except Exception:
        logs_dir_hint0 = None

    event_log0, event_src0 = load_event_log_unified(active_run_id, logs_dir_hint=logs_dir_hint0)

    # Activity pulse for the topbar (does not attempt cycle counting)
    pulse_view0 = compute_activity_pulse_view(
        event_log0 if active_run_id else [],
        watchdog0,
        ws0,
        event_log_src=event_src0,
    )

    # Recent activity: high-signal events from the event log (no cycle counting)
    narrative_events: List[Dict[str, Any]] = []
    feed_source_label0: str = ""

    if event_log0:
        narrative_events = build_high_signal_events_from_event_log(event_log0, limit=LIVE_EVENTS_LIMIT)
        if narrative_events:
            try:
                if event_src0 and event_src0 != "not found":
                    feed_source_label0 = f"{Path(str(event_src0)).name} (high-signal)"
                else:
                    feed_source_label0 = "event log (high-signal)"
            except Exception:
                feed_source_label0 = "event log (high-signal)"

    # Fallback (active run): if phase progress exists but the event log is quiet.
    if (not narrative_events) and active_run_id:
        try:
            if isinstance(progress_view0, dict) and str(progress_view0.get("kind") or "") == "phase":
                c = _safe_int(progress_view0.get("current"), None)
                t = _safe_int(progress_view0.get("total"), None)
                if isinstance(c, int) and isinstance(t, int) and t > 0:
                    narrative_events = [{"ts": _now_iso(), "kind": "phase", "message": f"Phase {c}/{t}"}]
                    feed_source_label0 = "run state (phase)"
        except Exception:
            pass

    # Sticky topbar (heartbeat/status/progress)
    ws0_for_topbar: Optional[Dict[str, Any]] = ws0
    if not active_run_id:
        # Avoid displaying stale run/progress information when the worker is idle.
        ws0_for_topbar = dict(ws0 or {})
        for _k in (
            "run_id",
            "job_id",
            "mode",
            "run_mode",
            "current",
            "total",
            "phase_index",
            "phase_total",
            "phase_name",
        ):
            ws0_for_topbar.pop(_k, None)

    render_topbar(ws0_for_topbar, watchdog0, pulse_view0, autonomy_view0)

    # Header: ARA with gradient and powered by pill
    st.markdown(
        textwrap.dedent(
            """
<style>
.ara-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.20rem 0 1.05rem 0;
}
.ara-logo-text {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #FFD93B, #FF9A1F, #FF3B3B);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.ara-powered-pill {
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.25);
    background: rgba(10, 10, 20, 0.6);
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}
.ara-powered-pill span.label { opacity: 0.8; }
.ara-powered-pill span.brand { font-weight: 600; opacity: 1.0; }
</style>
<div class="ara-header">
    <div class="ara-logo-text">ARA</div>
    <div class="ara-powered-pill">
        <span class="label">powered by</span>
        <span class="brand">Reparodynamics</span>
    </div>
</div>
            """
        ).strip("\n"),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Finite mode only | Queue based runs | Engine worker processes jobs from `{queue_pending_dir}` for `*_job.json` files.\n"
        "This UI never runs TGRM loops directly. It only queues jobs and visualizes artifacts."
    )

    # Show MemoryStore path in sidebar for sanity (diagnostics depend on this being shared)
    try:
        mem_path = getattr(memory, "path", None) or getattr(memory, "memory_file", None)
        if isinstance(mem_path, str):
            st.sidebar.caption(f"Memory file: `{mem_path}`")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Sidebar: Live updates (this fixes the "0/3 then 3/3" perception issue)
    # ------------------------------------------------------------------
    st.sidebar.subheader("Live updates")
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh while worker is running",
        value=True,
        help="Enables live dashboard updates so progress can show 1/3 -> 2/3 -> 3/3 during runs.",
    )
    refresh_seconds = 5
    if auto_refresh:
        refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", min_value=2, max_value=30, value=5, step=1)

    st.sidebar.markdown("---")

    # Small always visible sidebar progress (unified + normalized)
    if ws0:
        pv = progress_view0
        cur = pv.get("current")
        tot = pv.get("total")
        frac = pv.get("fraction")
        label = pv.get("label") or ""
        if isinstance(cur, int) and isinstance(tot, int) and tot > 0:
            # Display progress differently when total count is one or unknown.
            status_cur = str((ws0.get("status") or "")).lower()
            running_like = status_cur in {
                "running",
                "active",
                "in_progress",
                "working",
                "running_job",
                "running_cycle",
                "processing",
                "busy",
            }
            if tot > 1:
                st.sidebar.caption(f"Progress: {cur}/{tot} {label}".strip())
            else:
                if cur:
                    st.sidebar.caption(f"Progress: {cur} {label}".strip())
                elif running_like:
                    st.sidebar.caption("Progress: running")
                else:
                    st.sidebar.caption(f"Progress: {cur} {label}".strip())
            if isinstance(frac, (int, float)):
                try:
                    st.sidebar.progress(min(max(float(frac), 0.0), 1.0))
                except Exception:
                    pass
        else:
            # Show the worker status when no progress is available
            st.sidebar.caption(f"Worker status: {(ws0.get('status') or 'unknown')}")

    # ------------------------------------------------------------------
    # Live Console (visual upgrade set)
    # ------------------------------------------------------------------
    st.markdown("### Live console")
    left_console, right_console = st.columns([1, 2], gap="large")

    with left_console:
        # Autonomy level bar removed per user request. Display only agent presence.
        active_agent = None
        if ws0:
            active_agent = ws0.get("role") or ws0.get("active_role") or ws0.get("current_role")
            if active_agent is not None:
                active_agent = str(active_agent)
        render_agent_presence(agents0, active_agent=active_agent)

    with right_console:

        # Display a filtered live event feed in addition to a narrative summary.
        # Determine whether the run is active to decide if the console should auto refresh.
        try:
            run_id_feed = active_run_id  # type: ignore[name-defined]
        except Exception:
            run_id_feed = None

        run_active = False
        if st_autorefresh and ws0:
            try:
                status0 = str((ws0.get("status") or "")).lower()  # type: ignore[name-defined]
            except Exception:
                status0 = ""
            run_active = status0 in {
                "running",
                "active",
                "in_progress",
                "working",
                "busy",
                "running_job",
                "running_cycle",
                "processing",
            }
            if run_active and run_id_feed:
                # Refresh periodically to keep the console up to date.
                st_autorefresh(interval=750, key=f"console_refresh_{run_id_feed}")  # type: ignore[misc]

        # Live events feed controls: fixed number of messages to show.
        # This keeps the UI clean and avoids an extra slider on mobile.
        max_event_lines = 30
        # Use a fixed list of noisy keywords to exclude from the live event feed.  The
        # keyword input box has been removed per user request, but these defaults
        # help filter out internal progress messages.
        exclude_keywords: List[str] = ["cycle_progress", "job_claimed", "job_running"]

        # Attempt to tail the events.jsonl log for this run and filter out unhelpful progress events.
        messages: List[str] = []
        if run_id_feed:
            try:
                run_dir_path = Path(get_runs_root()) / str(run_id_feed)
                events_path = run_dir_path / "events.jsonl"
                # Use user-selected limit for the number of lines to tail
                lines = tail_lines(events_path, max_lines=int(max_event_lines))
            except Exception:
                lines = []
            # Parse and filter lines
            for _ev_line in lines:
                try:
                    _ev = json.loads(_ev_line)
                    # Skip progress events to reduce noise; show everything else.
                    if str(_ev.get("domain")) == "progress":
                        continue
                    # Filter out noisy lifecycle/status events...
                    _kind = str(_ev.get("kind") or _ev.get("type") or "").strip().lower()
                    if _kind in {"job_running", "job_claimed"}:
                        continue
                    _msg = _ev.get("msg") or _ev.get("message")
                    if isinstance(_msg, str) and _msg.strip():
                        # Apply keyword-based exclusions (case-insensitive) on the message.
                        _msg_low = _msg.lower()
                        if exclude_keywords and any(kw in _msg_low for kw in exclude_keywords):
                            continue
                        messages.append(str(_msg))
                        continue
                    # Fallback: build a message from domain and level.
                    dom = _ev.get("domain")
                    lvl = _ev.get("level")
                    msg_s = ""
                    if dom or lvl:
                        msg_s = f"{lvl or 'info'}: {dom}"
                    else:
                        msg_s = str(_ev)
                    # Apply keyword-based exclusions on fallback message.
                    if exclude_keywords and any(kw in msg_s.lower() for kw in exclude_keywords):
                        continue
                    messages.append(msg_s)
                except Exception:
                    # Non-JSON lines are appended directly if they pass exclusions.
                    line_s = _ev_line.strip()
                    if exclude_keywords and any(kw in line_s.lower() for kw in exclude_keywords):
                        continue
                    messages.append(line_s)
        # Render the filtered live events if any exist.
        if False and messages:
            st.markdown("#### Live events")
            for m in messages:
                st.write(m)

        # Render the narrative feed summarizing recent activity (high-signal, human friendly).
        label0 = feed_source_label0 or ""
        render_narrative_feed(narrative_events, source_label=label0)

        # Provide access to the raw event log JSON if available.  Users can expand this
        # section to inspect low-level event details.
        with st.expander("Raw event log JSON (if available)"):
            if event_log0:
                preview, note = safe_json_preview(event_log0, max_items=120)
                if preview:
                    st.code(preview, language="json")
                    if note:
                        st.caption(note)
            else:
                st.info("No event log file found yet.")

    # Discovery cards under live console
    discoveries_live = load_discovery_log(run_id=active_run_id)
    if not discoveries_live:
        discoveries_live = load_discoveries_from_finished_runs()
    render_discovery_cards(discoveries_live)

    # -------------------------------------------------------------------
    # Sidebar: Run configuration (original)
    # -------------------------------------------------------------------
    st.sidebar.title("Run configuration")

    # Optional Tavily key input (per user, not stored on disk)
    st.sidebar.subheader("Tavily API key (optional)")
    current_key = st.session_state.get("tavily_key", "")
    new_key = st.sidebar.text_input(
        "Tavily API key",
        type="password",
        value=current_key or "",
        help="Optional. If provided, allows real web search through Tavily in the engine worker.",
    )
    if new_key != current_key:
        st.session_state["tavily_key"] = new_key

    # Preset and domain selection
    st.sidebar.subheader("Preset and domain")

    if not PRESETS:
        st.sidebar.warning("No presets are defined. Using a basic default configuration.")
        preset = {
            "label": "Default",
            "domain": "general",
            "default_goal": "Explore Reparodynamics, define RYE and TGRM, and compare with related frameworks.",
        }
        selected_label = "Default"
        selected_key = "default"
    else:
        preset_labels = list(PRESETS.keys())
        # Only include the longevity preset; remove other presets such as general or math
        preset_labels = [key for key in preset_labels if str(key).lower() == "longevity"]
        selected_label = st.sidebar.selectbox(
            "Select preset",
            options=preset_labels,
            index=0,
            help="Choose a domain oriented preset for this run.",
        )
        selected_key = selected_label
        try:
            preset = get_preset(selected_key)  # type: ignore[arg-type]
        except Exception:
            preset = PRESETS.get(selected_key, {})
        if not isinstance(preset, dict):
            preset = {}

    domain_tag = preset.get("domain") or preset.get("domain_tag") or preset.get("domain_key") or "general"
    if isinstance(domain_tag, dict):
        domain_tag = domain_tag.get("tag") or domain_tag.get("name") or "general"
    # Override domain_tag to longevity since other presets are disabled.
    domain_tag = "longevity"
    st.sidebar.caption(f"Active domain: **{str(domain_tag).title()}**")

    # Runtime profile view (finite only, advisory)
    st.sidebar.subheader("Runtime profile (advisory, finite only)")
    runtime_profile = None
    runtime_profile_key = preset.get("runtime_profile") or preset.get("runtime_profile_key")
    if runtime_profile_key and isinstance(RUNTIME_PROFILES, dict):
        runtime_profile = RUNTIME_PROFILES.get(runtime_profile_key)

    if isinstance(runtime_profile, dict):
        label = runtime_profile.get("label") or runtime_profile_key
        description = runtime_profile.get("description") or runtime_profile.get("desc")
        max_minutes = runtime_profile.get("max_minutes")
        max_cycles = runtime_profile.get("max_cycles")

        lines = []
        if label:
            lines.append(f"Profile: **{label}**")
        if description:
            lines.append(str(description))
        if max_minutes is not None or max_cycles is not None:
            caps = []
            if max_minutes is not None:
                caps.append(f"~{max_minutes} minutes")
            if max_cycles is not None:
                caps.append(f"~{max_cycles} cycles")
            if caps:
                lines.append("Approx caps: " + ", ".join(caps))
        lines.append(
            "In this UI, the cycle input below is the hard limit. "
            "The profile is only a hint to the engine worker for internal tuning."
        )
        st.sidebar.caption("\n\n".join(lines))
    else:
        st.sidebar.caption("This preset has no runtime profile configured. Manual finite mode uses generic defaults.")

    # Friendly label per run
    st.sidebar.subheader("Run label")
    run_label = st.sidebar.text_input("Run label", value="experiment", help="Human friendly label for this run request.")

    # Tavily status (after handling key input)
    status = tavily_status()
    st.sidebar.subheader("Internet research")
    if status["has_key"]:
        st.sidebar.success(status["display"])
    else:
        st.sidebar.warning(status["display"])
        st.sidebar.write("Paste a Tavily key above to enable real web search. Otherwise, stubbed results are used.")

    # Tool status (web browser and sandbox)
    st.sidebar.subheader("Tools status")
    tool_flags = detect_tools()

    if tool_flags["web"]:
        st.sidebar.success("Web browser tool is available in tools.py.")
    else:
        st.sidebar.info("Web browser tool not detected in tools.py. Core engine may still use Tavily directly.")

    if tool_flags["sandbox"]:
        st.sidebar.success("Sandbox tool is available for safe code execution.")
    else:
        st.sidebar.info("Sandbox tool not detected in tools.py.")

    # Web browser and sandbox toggles
    use_web_tool = st.sidebar.checkbox(
        "Use web browser tool",
        value=status["has_key"],
        help="If enabled, the engine can use the web browser tool for searches.",
    )
    allow_sandbox = st.sidebar.checkbox(
        "Allow sandbox code execution",
        value=tool_flags["sandbox"],
        help="If enabled and present, the engine can run code in a bounded sandbox.",
    )

    # Swarm toggle and size
    st.sidebar.subheader("Swarm configuration")
    enable_swarm = st.sidebar.checkbox(
        "Enable Swarm (multi role mini agents)",
        value=False,
        help="Request multiple specialized agents. The worker runs them sequentially for safety.",
    )

    swarm_size = 1
    swarm_roles: List[Tuple[str, str]] = []
    if enable_swarm:
        swarm_size = st.sidebar.slider(
            "Total swarm agents",
            min_value=2,
            max_value=MAX_SWARM_AGENTS,
            value=min(5, MAX_SWARM_AGENTS),
            help="Total number of mini agents in the swarm.",
        )
        swarm_roles = build_swarm_roles(True, swarm_size)
        st.sidebar.write("Active swarm agents:")
        for name, desc in swarm_roles:
            st.sidebar.write(f"- **{name}**: {desc}")

    # Multi agent toggle (classic researcher plus critic)
    multi_agent = False
    if not enable_swarm:
        multi_agent = st.sidebar.checkbox(
            "Enable classic Multi Agent (Researcher + Critic)",
            value=False,
            help="If swarm is disabled, request a simple researcher + critic pair.",
        )
    else:
        st.sidebar.info("Classic Multi Agent is disabled when Swarm is enabled.")

    # Source controls
    sc_defaults = preset.get("source_controls", {})
    use_pubmed = st.sidebar.checkbox("Use PubMed (scientific literature)", value=bool(sc_defaults.get("pubmed", False)))
    use_semantic = st.sidebar.checkbox("Use Semantic Scholar ingestion", value=bool(sc_defaults.get("semantic", False)))
    use_pdf = st.sidebar.checkbox("Enable PDF ingestion (upload papers below)", value=bool(sc_defaults.get("pdf", True)))

    uploaded_pdf = None
    if use_pdf:
        uploaded_pdf = st.sidebar.file_uploader("Upload a PDF paper", type=["pdf"])

    use_biomarkers = st.sidebar.checkbox(
        "Biomarker / Longevity Mode (anti aging teams)",
        value=bool(sc_defaults.get("biomarkers", False)),
    )

    # Snapshot configuration
    st.sidebar.subheader("Snapshots")
    enable_snapshots = st.sidebar.checkbox(
        "Enable snapshot generation",
        value=True,
        help="Snapshots and heartbeat are most useful for long runs.",
    )
    snapshot_interval = st.sidebar.number_input(
        "Snapshot interval in cycles",
        min_value=1,
        max_value=1_000_000,
        value=1,
        step=1,
        help="How often to capture a snapshot (hint to engine worker).",
    )

    run_mode = st.sidebar.radio(
        "Run mode",
        ["Manual (finite cycles)"],
        index=0,
        help="This build uses finite mode only.",
    )

    # ------------------------------------------------------------------
    # Longevity biomarkers summary
    #
    # When Biomarker / Longevity mode is enabled in the sidebar, display a
    # concise overview of fifteen key blood-based biomarkers associated with
    # healthy aging.  Provide a toggle so users can optionally view the
    # underlying citation details.  The toggle remembers its state via
    # session_state and defaults to hidden to avoid cluttering the UI.
    # ------------------------------------------------------------------
    # Biomarker section removed: disable the display of the Longevity Biomarker Summary
    # by forcing the condition to false.  Previously this section displayed a detailed
    # summary of fifteen biomarkers and their citations when the sidebar toggle was enabled.
    if False and use_biomarkers:
        st.markdown("## Longevity Biomarker Summary")
        st.write(
            "This summary outlines fifteen important biomarkers used in longevity blood tests and why they matter for healthy aging."
        )

        # Toggle to show/hide citation details.  Off by default.
        show_biomarker_citations = st.toggle(
            "Show citation details",
            value=False,
            key="show_biomarker_citations",
            help="Toggle to display or hide detailed source and citation information."
        )

        # Define biomarker names and descriptions.
        biomarker_items = [
            ("Albumin", "Liver-produced protein that helps maintain fluid balance and transport molecules; low levels can suggest inflammation or liver/kidney issues."),
            ("ALP & ALT", "Liver enzymes; elevated ALT can suggest liver injury and elevated ALP can reflect bile duct disease or bone turnover (interpret with context)."),
            ("Creatinine (CRE)", "Kidney filtration marker influenced by muscle mass; rising creatinine can suggest reduced kidney function."),
            ("Creatine Kinase (CK)", "Marker of muscle breakdown; can rise after intense exercise or injury."),
            ("Reactive Oxygen Metabolites (ROM)", "Proxy for oxidative stress burden; higher values can indicate increased oxidative damage."),
            ("Total Antioxidant Capacity (TAC)", "Summary measure of antioxidant defenses; low values can indicate reduced oxidative protection."),
            ("DNA Damage (8-OHdG)", "Oxidative DNA damage marker; higher levels can reflect increased oxidative stress."),
            ("Intracellular NAD+", "Key cofactor for energy metabolism and DNA repair; levels may decline with age and metabolic stress."),
            ("Vitamin D (25-OH)", "Common vitamin D status measure; low levels are associated with bone and immune health risks."),
            ("Glycated Serum Protein (GSP)", "Short-term glycation marker reflecting recent blood sugar control (weeks)."),
            ("Blood Lipids (HDL, LDL, Triglycerides)", "Cardiometabolic risk indicators; patterns matter more than a single value."),
            ("Uric Acid", "At high levels can contribute to gout and cardiometabolic risk; also acts as an antioxidant at physiological levels."),
            ("Klotho", "Hormone-like protein linked to kidney and cardiovascular health; lower levels are associated with aging and disease risk."),
            ("Inflammation Markers (hs-CRP, IL-6, TNF-ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ±)", "Chronic low-grade inflammation (ÃÂÃÂÃÂÃÂ¢ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂinflammagingÃÂÃÂÃÂÃÂ¢ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ) correlates with higher disease and mortality risk."),
            ("Senescence-Associated Markers (SASP)", "Signals related to senescent-cell burden and secreted inflammatory factors; elevated markers can indicate higher senescence activity."),
        ]
        # Render each biomarker item as a bullet point.
        for name, desc in biomarker_items:
            st.markdown(f"* **{name}:** {desc}")

        # Only render citations when the toggle is on.
        if show_biomarker_citations:
            st.markdown("#### Sources and citations")
            citation_lines = _load_ui_asset_lines("ui_assets/biomarker_citations.txt")
            if not citation_lines:
                citation_lines = ["(no citation asset found: ui_assets/biomarker_citations.txt)"]
            for cl in citation_lines:
                st.markdown(f"- {cl}")

    stop_rye_threshold: Optional[float] = None

    # -----------------------------
    # Main area: goal + queue run
    # -----------------------------
    st.subheader("Research goal")

    default_goal = preset.get("default_goal") or (
        "Research and summarize Reparodynamics, define RYE and TGRM, identify similar frameworks, "
        "and produce a structured comparison table."
    )

    if "goal_text" not in st.session_state:
        st.session_state["goal_text"] = default_goal

    goal = st.text_area("Enter research goal:", value=st.session_state["goal_text"], height=160)
    st.session_state["goal_text"] = goal

    cycles = st.number_input(
        "Number of TGRM cycles to request (manual mode)",
        min_value=1,
        max_value=1_000_000,
        value=3,
        step=1,
        help="Finite run cycle budget.",
    )

    run_button = st.button("Queue run request")

    if run_button:
        goal_clean = goal.strip()
        if not goal_clean:
            st.error("Please provide a research goal or question before queuing a run.")
        elif create_job is None:
            st.error("Job queue backend (run_jobs.py) is not available. Make sure agent/run_jobs.py exists.")
        else:
            source_controls = {
                "web": bool(use_web_tool),
                "pubmed": bool(use_pubmed),
                "semantic": bool(use_semantic),
                "pdf": bool(use_pdf and uploaded_pdf is not None),
                "biomarkers": bool(use_biomarkers),
                "sandbox": bool(allow_sandbox and tool_flags["sandbox"]),
                "tavily_enabled": bool(status["has_key"]),
            }

            pdf_payload: Optional[Dict[str, Any]] = None
            if use_pdf and uploaded_pdf is not None:
                try:
                    pdf_bytes = uploaded_pdf.getvalue()
                    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
                    pdf_payload = {"name": uploaded_pdf.name, "base64": pdf_b64}
                except Exception:
                    pdf_payload = {"name": uploaded_pdf.name, "base64": None}

            if enable_swarm and swarm_size > 1:
                mode = "swarm"
            elif multi_agent:
                mode = "two_stage"
            else:
                mode = "single"

            runtime_hints: Dict[str, Any] = {
                "run_mode": "finite_manual",
                "manual_cycles": int(cycles),
                "max_cycles": int(cycles) if mode != "swarm" else None,
                "max_rounds": int(cycles) if mode == "swarm" else None,
                "stop_rye_threshold": stop_rye_threshold,
                "cycles_per_hour_estimate": CYCLES_PER_HOUR_ESTIMATE,
            }

            snapshots_target_dir = str(Path(get_runs_root()) / "logs" / "snapshots")
            snapshot_config: Dict[str, Any] = {
                "enabled": bool(enable_snapshots),
                "every_n_cycles": int(snapshot_interval),
                "target_dir": snapshots_target_dir,
            }
            runtime_hints["snapshot_config"] = snapshot_config

            monitoring_config: Dict[str, Any] = {
                "snapshots_enabled": bool(enable_snapshots),
                "snapshot_interval_cycles": int(snapshot_interval) if enable_snapshots else None,
                "snapshot_interval_minutes": None,
                "snapshot_max_to_keep": 50,
                "heartbeat_enabled": True,
                "heartbeat_interval_seconds": 60,
                "run_state_enabled": True,
            }

            if enable_swarm:
                swarm_config: Dict[str, Any] = {
                    "swarm_size": int(swarm_size),
                    "roles": [name for name, _ in swarm_roles] if swarm_roles else ["agent"],
# (comment trimmed to keep this file renderable in GitHub)
# (comment trimmed to keep this file renderable in GitHub)
                    # the cycles slider.  Set to the requested cycles value.
                    "max_cycles_per_agent": int(cycles),
                    "stagger_start": False,
# (comment trimmed to keep this file renderable in GitHub)
                    # all agents can run each round.  Some backends interpret
                    # zero as a default limit (often 32), which would reduce
# (comment trimmed to keep this file renderable in GitHub)
                    "max_agents_per_tick": int(swarm_size),
                    "role_goals": {name: role_specific_goal(goal_clean, name) for name, _ in swarm_roles} if swarm_roles else {},
                }
            else:
                swarm_config = {
                    "swarm_size": 1,
                    "roles": ["agent"],
                    "max_cycles_per_agent": int(cycles),
                    "stagger_start": False,
# (comment trimmed to keep this file renderable in GitHub)
                    "max_agents_per_tick": 1,
                }

            longevity_config: Dict[str, Any] = {}
            if str(domain_tag).lower() in {"longevity", "aging", "anti_aging"}:
                longevity_defaults = preset.get("longevity_config", {})
                if isinstance(longevity_defaults, dict):
                    longevity_config = {
                        "hallmark_targets": longevity_defaults.get("hallmark_targets", []),
                        "curriculum_profile": longevity_defaults.get("curriculum_profile"),
                    }

            # In swarm mode interpret the cycles slider as the number of rounds
# (comment trimmed to keep this file renderable in GitHub)
# (comment trimmed to keep this file renderable in GitHub)
            # multiplying the number of rounds by the number of participating
            # agents.  This change avoids a mismatch where the UI requests
# (comment trimmed to keep this file renderable in GitHub)
            # interprets it as global cycles, leading to runs that end early or
            # overshoot the intended budget.
            total_cycles_requested = int(cycles)

            run_config: Dict[str, Any] = {
                "goal": goal_clean,
                # Always set the domain to longevity
                "domain": "longevity",
                "mode": mode,
                "total_cycles": total_cycles_requested,
                "max_cycles": int(cycles) if mode != "swarm" else None,
                "max_rounds": int(cycles) if mode == "swarm" else None,
                "max_seconds": None,
                "rye_stop_threshold": stop_rye_threshold,
                "equilibrium_stop_label": None,
                "min_cycles_before_stop": min(3, int(cycles)),
                "source_controls": source_controls,
                "runtime_hints": runtime_hints,
                "swarm": swarm_config,
                "swarm_config": swarm_config,
                "longevity_config": longevity_config,
                "snapshot_config": snapshot_config,
                "snapshots_enabled": bool(enable_snapshots),
                "use_biomarkers": bool(use_biomarkers),
                "multi_agent_pair": bool(multi_agent),
                "notes": (run_label or "experiment").strip(),
                "monitoring": monitoring_config,
            }

            # In swarm mode include explicit top-level roles and swarm sizing hints.
            # The engine worker uses the top-level "roles" list to determine the
            # number of agents when computing progress.  Without this, it falls back
            # to the number of unique role names which may be fewer than the total
            # number of mini agents.  Additionally propagate swarm_size and
            # max_agents_per_tick so worker defaults (e.g. 32) do not override the
            # requested swarm size.
            if enable_swarm:
                try:
                    # Flatten the role names into a simple list for the worker.
                    base_roles = [name for name, _desc in swarm_roles] if swarm_roles else ["agent"]
                    # Expand or truncate the roles list to match the requested swarm size.
                    try:
                        target_size = int(swarm_size)
                    except Exception:
                        target_size = len(base_roles)
                    if target_size <= 0:
                        target_size = len(base_roles)
                    # If fewer base roles than agents, repeat them until reaching the target size.
                    if len(base_roles) < target_size:
                        extended: List[str] = []
                        while len(extended) < target_size:
                            extended.extend(base_roles)
                        role_names_extended = extended[:target_size]
                    else:
                        role_names_extended = base_roles[:target_size]
                    run_config["roles"] = role_names_extended
                except Exception:
                    run_config["roles"] = ["agent"]
                # Propagate swarm_size and per tick agent cap to the top level
                run_config["swarm_size"] = int(swarm_size)
                run_config["max_agents_per_tick"] = int(swarm_size)
                # Also include max_cycles_per_agent at the top level for clarity
                run_config["max_cycles_per_agent"] = int(cycles)
            else:
                # For non-swarm runs, define a single agent role if not already present.
                run_config.setdefault("roles", ["agent"])

            # Back-compat: some engine/agent versions use `cycles`/`rounds` keys.
            # We keep these consistent with max_cycles/max_rounds when present.
            if mode != "swarm":
                run_config.setdefault("cycles", int(cycles))
                run_config.setdefault("total_cycles", int(total_cycles_requested))
            else:
                run_config.setdefault("rounds", int(cycles))
                run_config.setdefault("total_rounds", int(cycles))

            if pdf_payload is not None:
                run_config["pdf"] = pdf_payload
                run_config["pdf_payload"] = pdf_payload

            t_key = st.session_state.get("tavily_key")
            t_tail = t_key[-4:] if isinstance(t_key, str) and len(t_key) >= 4 else None

            meta: Dict[str, Any] = {
                "run_label": (run_label or "experiment").strip(),
                "preset_key": selected_key,
                "preset_label": preset.get("label", selected_label),
                # Always set the domain to longevity
                "domain": "longevity",
                "mode": mode,
                "tavily_enabled": bool(status["has_key"]),
                "tavily_key_tail": t_tail,
                "ui_metadata": {"requested_from": "streamlit", "client_version": "v4-live-console-diagnostics"},
            }

            run_id = _safe_create_job(config=run_config, meta=meta)
            if not run_id:
                st.error("Failed to queue job (create_job did not return a run id). Check server logs.")
            else:
                # Persist as a hint so the topbar can immediately track it even before worker_state appears
                st.session_state["active_run_id_hint"] = str(run_id)
                st.session_state["last_queued_run_id"] = str(run_id)

                st.success(f"Run request queued with run id `{run_id}`.")
                if RUNS_PENDING_DIR is not None:
                    st.caption(f"Pending job written to `{RUNS_PENDING_DIR / (str(run_id) + '_job.json')}`")
                else:
                    st.caption(
                        f"Job was queued via run_jobs.create_job. The engine worker should watch `{queue_pending_dir}` for new `*_job.json` files."
                    )

    # ------------------------------
    # Runs and job queue (queued or finished via run_jobs.py)
    # ------------------------------
    st.markdown("---")
    st.subheader("Runs and job queue")

    # Debug view moved behind expander to keep UI clean
    with st.expander("Debug: queue directories", expanded=False):
        st.caption(f"DEBUG runs base dir: `{runs_base_dir}`")
        st.caption(f"DEBUG queue root: `{queue_root_dir}`")
        st.caption("Active can include stale files if a worker stopped before cleaning up.")

        def _debug_list_dir(label: str, specific: Optional[Path]) -> None:
            if isinstance(specific, Path):
                base = specific
            else:
                # Prefer queue-root for queue dirs when run_jobs paths aren't available
                if label in ("pending", "active", "finished", "error"):
                    base = Path(queue_root_dir) / label
                else:
                    base = Path(runs_base_dir) / label
            try:
                if not base.exists() or not base.is_dir():
                    items: List[str] = []
                else:
                    items = []
                    for p in sorted(base.iterdir()):
                        if p.is_dir():
                            items.append(p.name + "/")
                        elif p.suffix.lower() == ".json":
                            items.append(p.name)

                    if label in ("pending", "active"):
                        items = [x for x in items if x.endswith("_job.json") or x.endswith(".json")]

                    if len(items) > 60:
                        items = items[:60] + ["..."]
            except Exception as e:
                items = [f"error: {e}"]
            st.text(f"DEBUG {label}: {items}")

        _debug_list_dir("pending", RUNS_PENDING_DIR)
        _debug_list_dir("active", RUNS_ACTIVE_DIR)
        _debug_list_dir("finished", RUNS_FINISHED_DIR)
        _debug_list_dir("error", RUNS_ERROR_DIR)

    finished_jobs: List[Any] = _list_jobs_by_status_candidates(["finished", "done", "completed", "complete", "success"])
    pending_jobs: List[Any] = _list_jobs_by_status_candidates(["queued", "pending", "waiting"])
    active_jobs: List[Any] = _list_jobs_by_status_candidates(["active", "running", "in_progress", "working"])

    try:
        finished_jobs = sorted(finished_jobs, key=_job_created_at_ts, reverse=True)
    except Exception:
        pass
    try:
        pending_jobs = sorted(pending_jobs, key=_job_created_at_ts)
    except Exception:
        pass
    try:
        active_jobs = sorted(active_jobs, key=_job_created_at_ts)
    except Exception:
        pass

    col_runs_left, col_runs_right = st.columns([2, 1])

    with col_runs_left:
        st.markdown("#### Finished runs")
        if not finished_jobs:
            st.info("No finished runs found yet.")
        else:
            run_ids = [_get_job_id(j) for j in finished_jobs]
            id_to_job = {_get_job_id(j): j for j in finished_jobs}

            def _format_run(jid: str) -> str:
                j = id_to_job.get(jid)
                return _get_job_label(j) if j is not None else jid

            selected_run_id = st.selectbox("Select a finished run", options=run_ids, format_func=_format_run)
            if selected_run_id:
                selected_job_header = id_to_job.get(selected_run_id)
                if selected_job_header:
                    render_job_summary(selected_job_header)
                result = load_job_result(selected_run_id)
                if result is None:
                    st.warning("Result file missing or unreadable for this run. The worker may not have written it yet.")
                else:
                    st.markdown("---")
                    render_result_details(result)

    with col_runs_right:
        st.markdown("#### Active runs")
        pending_dir_path = RUNS_PENDING_DIR if isinstance(RUNS_PENDING_DIR, Path) else (Path(queue_root_dir) / "pending")
        pending_dir = str(pending_dir_path)

        if not active_jobs:
            st.info("No active runs detected by run_jobs.")
        else:
            seen_ids: Set[str] = set()
            for job in active_jobs:
                jid = _get_job_id(job)
                if jid in seen_ids:
                    continue
                seen_ids.add(jid)
                with st.container():
                    render_job_summary(job)
                    st.caption("Engine worker is currently processing this run.")
                    # Offer a stop button for each active job.  When clicked, create
                    # a stop flag file within the run directory.  The engine
                    # worker checks for this flag and aborts the run on the next
                    # progress callback.  We scope the key to the run ID to
                    # prevent cross-run clashes.
                    try:
                        jid_str = str(jid)
                        # When the button is clicked, create a stop.flag file
                        # inside <runs_root>/<run_id>.  The directory is
                        # created if it doesn't already exist.  The presence
                        # of this file signals to the worker that it should
                        # gracefully stop the run.
                        if st.button(
                            "Stop this run", key=f"stop_run_{jid_str}", help="Gracefully stop this active run"
                        ):
                            try:
                                run_dir = Path(get_runs_root()) / jid_str
                                run_dir.mkdir(parents=True, exist_ok=True)
                                flag_path = run_dir / "stop.flag"
                                # Touch the stop flag file
                                flag_path.touch(exist_ok=True)
                                st.success(f"Stop signal sent to run {jid_str}. The run will halt shortly.")
                            except Exception as e:
                                st.error(f"Failed to signal stop for run {jid_str}: {e}")
                    except Exception:
                        # If run ID extraction fails, silently skip stop button
                        pass
                    st.markdown("---")

        st.markdown("#### Queued runs")
        st.caption(f"Queue directory: `{pending_dir}`")

        if st.button("Clear job queue", key="clear_queue_btn"):
            removed = 0

            def _is_uuid_stem(stem: str) -> bool:
                try:
                    uuid.UUID(stem)
                    return True
                except Exception:
                    return False

            # Safer: clear only job-like JSONs from pending queues and top-level queue dirs.
            # Scan multiple candidate directories where pending jobs may live:
            #   - The resolved pending_dir (based off RUNS_PENDING_DIR or queue_root)
            #   - The ``pending`` subfolder under the queue root
            #   - The ``pending`` subfolder under the runs root
            #   - The queue root itself (jobs may be written there directly)
            #   - The runs root itself (legacy job files)
            #   - The explicit RUNS_PENDING_DIR if provided
            dirs_to_scan: List[Path] = []
            try:
                dirs_to_scan.append(Path(pending_dir))
            except Exception:
                pass
            try:
                # Queue root/pending
                dirs_to_scan.append(Path(get_queue_root()) / "pending")
            except Exception:
                pass
            try:
                # Runs root/pending
                dirs_to_scan.append(Path(get_runs_root()) / "pending")
            except Exception:
                pass
            try:
                # Top-level queue root (some versions write jobs directly here)
                dirs_to_scan.append(Path(get_queue_root()))
            except Exception:
                pass
            try:
                # Top-level runs root (legacy job submission)
                dirs_to_scan.append(Path(get_runs_root()))
            except Exception:
                pass
            if isinstance(RUNS_PENDING_DIR, Path):
                dirs_to_scan.append(RUNS_PENDING_DIR)

            # Deduplicate
            seen_dirs: Set[str] = set()
            unique_dirs: List[Path] = []
            for d in dirs_to_scan:
                try:
                    key = str(d.resolve())
                except Exception:
                    key = str(d)
                if key in seen_dirs:
                    continue
                seen_dirs.add(key)
                unique_dirs.append(d)

            for dpath in unique_dirs:
                if not isinstance(dpath, Path) or not dpath.exists() or not dpath.is_dir():
                    continue
                for fp in dpath.glob("*.json"):
                    name = fp.name
                    # Never delete result/progress artifacts
                    if name.endswith("_progress.json") or name.endswith("_results.json") or name.endswith("_result.json"):
                        continue

                    # Primary: canonical job files
                    if name.endswith("_job.json"):
                        try:
                            fp.unlink()
                            removed += 1
                        except Exception:
                            pass
                        continue

                    # Legacy: job files named as UUID.json
                    if _is_uuid_stem(fp.stem):
                        try:
                            # If it's small enough, sanity-check it's actually a job payload
                            if fp.stat().st_size <= 5_000_000:
                                data = _load_json_file(fp)
                                if data is not None and not _looks_like_job_payload_json(data):
                                    continue
                            fp.unlink()
                            removed += 1
                        except Exception:
                            pass
                        continue

                    # Optional: non-uuid, but job-shaped JSON
                    try:
                        if fp.stat().st_size <= 2_000_000:
                            data2 = _load_json_file(fp)
                            if data2 is not None and _looks_like_job_payload_json(data2):
                                fp.unlink()
                                removed += 1
                    except Exception:
                        pass

            st.success(f"Cleared {removed} queued job file(s) (pending only).")
            st.rerun()

        if not pending_jobs:
            st.info("No queued runs. Use the form above to queue one.")
        else:
            for job in pending_jobs:
                with st.container():
                    render_job_summary(job)
                    st.caption("Waiting for engine worker to start.")
                    st.markdown("---")

    # ------------------------------
    # History and advanced panels (original)
    # ------------------------------
    st.markdown("---")
    st.subheader("History and advanced analysis")

    get_cycle_history = getattr(memory, "get_cycle_history", None)
    history: List[Dict[str, Any]] = []
    if callable(get_cycle_history):
        try:
            history = get_cycle_history() or []
        except Exception:
            history = []

    if not history:
        history = load_history_from_finished_runs()

    # Filter history to the active run when available.  This prevents cycles
    # from previous runs mixing with the currently selected run in the UI.
    active_run_id_local: Optional[str] = None
    try:
        # Attempt to reuse the previously computed active_run_id from the top-level scope
        active_run_id_local = active_run_id  # type: ignore[name-defined]
    except Exception:
        active_run_id_local = None
    if active_run_id_local:
        history = _get_cycle_history_for_run(history, active_run_id_local)

    if not history:
        st.write("No cycles yet.")
    else:
        msil_profile_full = compute_msil_profile(history)

        tab_history, tab_citations, tab_discovery, tab_snapshots, tab_hypo, tab_memory, tab_verify, tab_graph = st.tabs(
            [
                "Cycle history",
                "Citations",
                "Discovery log",
                "Snapshots and equilibrium",
                "Hypothesis manager",
                "Memory pruning",
                "Verification and cures",
                "Multi agent insight graph",
            ]
        )

        with tab_history:
            rows: List[Dict[str, Any]] = []
            for entry in history:
                goal_text = entry.get("goal", "") or ""

                delta_val = entry.get("delta_R")
                if delta_val is None:
                    delta_val = entry.get("delta_r")

                energy_val = entry.get("energy_E")
                if energy_val is None:
                    energy_val = entry.get("energy")

                rye_val = entry.get("RYE")
                if rye_val is None:
                    rye_val = entry.get("rye")

                rows.append(
                    {
                        "cycle": entry.get("cycle"),
                        "role": entry.get("role", "agent"),
                        "domain": entry.get("domain", "general"),
                        "goal": goal_text[:60] + ("..." if len(goal_text) > 60 else ""),
                        "delta_R": delta_val,
                        "energy_E": energy_val,
                        "RYE": rye_val,
                        "timestamp": entry.get("timestamp"),
                    }
                )

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("### Efficiency Charts")

            plot_rows = rows[-MAX_POINTS_FOR_CHARTS:]
            if plot_rows:
                chart_df = pd.DataFrame(plot_rows)[["cycle", "RYE", "delta_R", "energy_E"]].copy()
                chart_df = chart_df[chart_df["cycle"].notna()]
                if not chart_df.empty:
                    chart_df = chart_df.set_index("cycle")

                    if chart_df["RYE"].notna().any():
                        st.line_chart(chart_df[["RYE"]])
                        st.caption("Higher RYE means more efficient repair (delta_R per unit energy).")

                    if chart_df["delta_R"].notna().any():
                        st.line_chart(chart_df[["delta_R"]])
                        st.caption("delta_R is how much improvement each cycle produced.")

                    if chart_df["energy_E"].notna().any():
                        st.line_chart(chart_df[["energy_E"]])
                        st.caption("Energy per cycle (approximate effort cost).")
                else:
                    st.info("No cycle indices available for charting yet.")

            st.markdown("### Snapshot timeline")

            if history:
                max_cycle = max(int(e.get("cycle", 0) or 0) for e in history)
                if max_cycle < 1:
                    st.info("No cycles to snapshot yet.")
                else:
                    default_interval = max(1, max_cycle // 50)
                    snapshot_interval_display = st.number_input(
                        "Display a snapshot every N cycles",
                        min_value=1,
                        max_value=max_cycle,
                        value=default_interval,
                        step=1,
                        help="Controls how densely cycles are sampled for the snapshot chart (display only).",
                        key="history_snapshot_interval",
                    )

                    snapshot_points: List[Dict[str, Any]] = []
                    for e in history:
                        c_num = int(e.get("cycle", 0) or 0)
                        if c_num <= 0:
                            continue
                        if c_num == 1 or c_num == max_cycle or c_num % snapshot_interval_display == 0:
                            snapshot_points.append(e)

                    if len(snapshot_points) > MAX_POINTS_FOR_CHARTS:
                        snapshot_points = snapshot_points[-MAX_POINTS_FOR_CHARTS:]

                    st.caption(
                        f"Showing {len(snapshot_points)} snapshot points out of {max_cycle} cycles (interval {snapshot_interval_display})."
                    )

                    snapshot_cycles: List[int] = []
                    snapshot_rye: List[Any] = []
                    for e in snapshot_points:
                        c_num = int(e.get("cycle", 0) or 0)
                        snapshot_cycles.append(c_num)
                        v = e.get("RYE")
                        if not isinstance(v, (int, float)):
                            v = e.get("rye")
                        snapshot_rye.append(v)

                    if any(v is not None for v in snapshot_rye):
                        df_snap = pd.DataFrame({"cycle": snapshot_cycles, "RYE": snapshot_rye}).set_index("cycle")
                        st.line_chart(df_snap)
                        st.caption("Snapshot view of RYE across the run.")
                    else:
                        st.info("No RYE values available for snapshot chart.")
            else:
                st.info("No history available yet for snapshot timeline.")

            st.markdown("### Advanced RYE diagnostics")

            try:
                diagnostics = build_run_diagnostics(history=history, domain=None, window=10)
            except Exception:
                diagnostics = {}

            roll_val = diagnostics.get("rolling_rye")
            trend_val = diagnostics.get("trend_simple")
            slope_val = diagnostics.get("trend_slope")
            stability_val = diagnostics.get("stability_index")
            momentum_val = diagnostics.get("recovery_momentum")

            adv_cols = st.columns(5)
            with adv_cols[0]:
                st.metric("Rolling RYE (10)", f"{float(roll_val):.3f}" if isinstance(roll_val, (int, float)) else "n/a")
            with adv_cols[1]:
                st.metric("RYE trend", f"{float(trend_val):.3f}" if isinstance(trend_val, (int, float)) else "n/a")
            with adv_cols[2]:
                st.metric("RYE slope", f"{float(slope_val):.4f}" if isinstance(slope_val, (int, float)) else "n/a")
            with adv_cols[3]:
                st.metric("Stability index", f"{float(stability_val):.3f}" if isinstance(stability_val, (int, float)) else "n/a")
            with adv_cols[4]:
                st.metric("Recovery momentum", f"{float(momentum_val):.3f}" if isinstance(momentum_val, (int, float)) else "n/a")

            st.markdown("### Learning speed and breakthrough profile")

            hours_run = compute_run_hours(history)
            try:
                bp_short = estimate_breakthrough_probability(diagnostics, domain=None, horizon_hours=hours_run)
            except Exception:
                bp_short = None
            try:
                bp90 = breakthrough_likelihood_90d(diagnostics, domain=None, hours_run_so_far=hours_run)
            except Exception:
                bp90 = None
            try:
                env = autonomy_safety_envelope(diagnostics)
            except Exception:
                env = {}
            try:
                fail = early_failure_warning_score(diagnostics)
            except Exception:
                fail = {}

            bp_prob = bp_short.get("probability") if isinstance(bp_short, dict) else None
            bp90_prob = bp90.get("probability") if isinstance(bp90, dict) else None

            try:
                tier_info = classify_run_tier(diagnostics, breakthrough_prob=bp_prob)
            except Exception:
                tier_info = None

            tier_label = tier_info.get("tier") or tier_info.get("label") if isinstance(tier_info, dict) else None

            ls_cols = st.columns(4)
            with ls_cols[0]:
                st.metric("Approx hours run", f"{hours_run:.2f}" if isinstance(hours_run, (int, float)) else "n/a")
            with ls_cols[1]:
                st.metric("Breakthrough signal (near term, 0 to 1)", f"{bp_prob:.3f}" if isinstance(bp_prob, (int, float)) else "n/a")
            with ls_cols[2]:
                st.metric("Breakthrough signal 90d (0 to 1)", f"{bp90_prob:.3f}" if isinstance(bp90_prob, (int, float)) else "n/a")
            with ls_cols[3]:
                st.metric("Run tier", tier_label or "n/a")

            st.caption("Breakthrough signals are heuristic scores (0..1), not calibrated real world probabilities.")

            st.markdown("### Meta skill intelligence (MSIL)")
            if msil_profile_full:
                msil_score = msil_profile_full.get("msil_score")
                skills = msil_profile_full.get("skills") or msil_profile_full.get("dimensions") or {}
                domains_profile = msil_profile_full.get("domains") or msil_profile_full.get("domain_profiles") or []
                dynamics = msil_profile_full.get("dynamics") or {}

                msil_cols = st.columns(3)
                with msil_cols[0]:
                    st.metric("MSIL score", f"{msil_score:.3f}" if isinstance(msil_score, (int, float)) else "n/a")
                with msil_cols[1]:
                    st.metric("Skill dimensions", len(skills) if isinstance(skills, dict) else 0)
                with msil_cols[2]:
                    dom_count = len(domains_profile) if isinstance(domains_profile, list) else (len(domains_profile) if isinstance(domains_profile, dict) else 0)
                    st.metric("Domain profiles", dom_count)

                with st.expander("Skill breakdown"):
                    st.json(skills)
                with st.expander("Domain intelligence profile"):
                    st.json(domains_profile)
                if dynamics:
                    with st.expander("Learning and stability dynamics"):
                        st.json(dynamics)
            else:
                st.info("MSIL module not detected or no MSIL profile available. This panel stays optional.")

            # st.markdown("#### 10x learning dashboard (Option C signals)")

            # Option C signals have been disabled due to persistent 'n/a' readings.
            # domain_for_signals = None
            # if isinstance(diagnostics, dict):
            #     dom0 = diagnostics.get("domain")
            #     if isinstance(dom0, str) and dom0:
            #         domain_for_signals = dom0

            # rye_series_for_option_c: List[float] = []
            # try:
            #     if isinstance(history, list):
            #         for c in history:
            #             if not isinstance(c, dict):
            #                 continue
            #             if domain_for_signals and str(c.get("domain") or "") != domain_for_signals:
            #                 continue
            #             v = _maybe_float(c.get("rye"))
            #             if v is not None:
            #                 rye_series_for_option_c.append(float(v))
            # except Exception:
            #     rye_series_for_option_c = []

            # volatility_info: Dict[str, Any] = {}
            # try:
            #     volatility_info = rye_volatility_signature(rye_series_for_option_c, window=10)  # type: ignore[arg-type]
            # except TypeError:
            #     try:
            #         # Back-compat: older signature may accept history/domain/window
            #         volatility_info = rye_volatility_signature(history=history, domain=domain_for_signals, window=10)  # type: ignore[call-arg]
            #     except Exception:
            #         volatility_info = {}
            # except Exception:
            #     volatility_info = {}

            # equilibrium_info: Dict[str, Any] = {}
            # try:
            #     equilibrium_info = detect_rye_equilibrium(rye_series_for_option_c)  # type: ignore[arg-type]
            # except TypeError:
            #     try:
            #         equilibrium_info = detect_rye_equilibrium(history=history, domain=domain_for_signals, window=10)  # type: ignore[call-arg]
            #     except Exception:
            #         equilibrium_info = {}
            # except Exception:
            #     equilibrium_info = {}

            # harmonic_val: Optional[float] = None
            # try:
            #     if isinstance(diagnostics, dict):
            #         harmonic_val = tgrm_harmonic_index(diagnostics.get("stability_index"), diagnostics.get("recovery_momentum"))  # type: ignore[arg-type]
            #     else:
            #         harmonic_val = tgrm_harmonic_index(None, None)  # type: ignore[arg-type]
            # except TypeError:
            #     try:
            #         harmonic_val = tgrm_harmonic_index(history=history, domain=domain_for_signals, window=10)  # type: ignore[call-arg]
            #     except Exception:
            #         harmonic_val = None
            # except Exception:
            #     harmonic_val = None

            # vol_score = volatility_info.get("volatility_score")
            # vol_regime = volatility_info.get("regime") or volatility_info.get("label")
            # eq_flag = equilibrium_info.get("equilibrium")
            # if eq_flag is None:
            #     eq_flag = equilibrium_info.get("in_equilibrium")
            # eq_reason = equilibrium_info.get("reason")
            # eq_state_text = "yes" if eq_flag is True else ("no" if eq_flag is False else "unknown")
            # if str(eq_reason).lower() in {"not_enough_data", "insufficient_data", "no_data"}:
            #     eq_state_text = "n/a"

            # oc_cols = st.columns(3)
            # with oc_cols[0]:
            #     st.metric("Equilibrium detected", eq_state_text)
            # with oc_cols[1]:
            #     st.metric("Volatility score", f"{vol_score:.3f}" if isinstance(vol_score, (int, float)) else "n/a")
            # with oc_cols[2]:
            #     st.metric("TGRM harmonic index", f"{harmonic_val:.3f}" if isinstance(harmonic_val, (int, float)) else "n/a")

            # if vol_regime:
            #     st.caption(f"Volatility regime: {vol_regime}")
            # if eq_reason:
            #     st.caption(f"Equilibrium reasoning: {eq_reason}")

            # with st.expander("Autonomy safety and failure envelope"):
            #     st.write("Autonomy safety envelope:")
            #     st.json(env)
            #     st.write("Early failure warning score:")
            #     st.json(fail)

            # with st.expander("Raw Option C style signals"):
            #     raw_signals = {
            #         "diagnostics": diagnostics,
            #         "volatility": volatility_info,
            #         "equilibrium": equilibrium_info,
            #         "harmonic_index": harmonic_val,
            #         "breakthrough_near_term": bp_short,
            #         "breakthrough_90d": bp90,
            #         "run_tier": tier_info,
            #         "msil_profile": msil_profile_full,
            #     }
            #     preview, note = safe_json_preview(raw_signals)
            #     if preview is not None:
            #         st.code(preview, language="json")
            #         if note:
            #             st.caption(note)

            # with st.expander("Raw history JSON"):
            #     preview, note = safe_json_preview(history, max_items=MAX_POINTS_FOR_CHARTS)
            #     if preview is not None:
            #         st.code(preview, language="json")
            #         if note:
            #             st.caption(note)

        with tab_citations:
            st.markdown("### Source citation viewer")
            citations = extract_citation_rows_from_history(history)
            if not citations:
                citations = load_citations_from_finished_runs()

            if not citations:
                st.info("No citations recorded yet in cycle history or finished run artifacts.")
            else:
                citations_df = pd.DataFrame(citations)
                expected_cols = ["cycle", "role", "domain", "source", "title", "snippet", "url", "timestamp"]
                for col in expected_cols:
                    if col not in citations_df.columns:
                        citations_df[col] = None

                total_cites = len(citations_df)
                unique_sources = sorted({s for s in citations_df["source"].dropna().astype(str).unique() if s})
                domains_c = sorted({str(d) for d in citations_df["domain"].dropna().astype(str).unique()})
                roles_c = sorted({str(r) for r in citations_df["role"].dropna().astype(str).unique()})

                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    st.metric("Total citation hits", total_cites)
                with col_c2:
                    st.metric("Unique sources", len(unique_sources))
                with col_c3:
                    st.metric("Domains with citations", len(domains_c))

                citations_domain_filter = st.multiselect("Filter by domain", options=domains_c, default=domains_c, key="citations_domain_filter")
                citations_role_filter = st.multiselect("Filter by role", options=roles_c, default=roles_c, key="citations_role_filter")
                citations_source_filter = st.multiselect("Filter by source", options=unique_sources, default=unique_sources, key="citations_source_filter")

                search_query = st.text_input("Search citations (title or snippet)", value="", key="citations_search")
                group_by_option = st.selectbox("Group citations by", options=["None", "Cycle", "Source"], index=0, key="citations_group_by")

                filtered_df = citations_df[
                    citations_df["domain"].astype(str).isin(citations_domain_filter)
                    & citations_df["role"].astype(str).isin(citations_role_filter)
                    & citations_df["source"].astype(str).isin(citations_source_filter)
                ].copy()

                if search_query:
                    q = search_query.lower()
                    mask = (
                        filtered_df["title"].astype(str).str.lower().str.contains(q, na=False)
                        | filtered_df["snippet"].astype(str).str.lower().str.contains(q, na=False)
                    )
                    filtered_df = filtered_df[mask]

                if group_by_option == "Cycle":
                    group_counts = filtered_df.groupby("cycle").size().reset_index(name="citation_count").sort_values("cycle")
                    st.dataframe(group_counts, use_container_width=True)
                elif group_by_option == "Source":
                    group_counts = filtered_df.groupby("source").size().reset_index(name="citation_count").sort_values("citation_count", ascending=False)
                    st.dataframe(group_counts, use_container_width=True)
                else:
                    display_df = filtered_df.copy().reset_index(drop=True)
                    display_df["title_short"] = display_df["title"].fillna("").astype(str).str.slice(0, 80)
                    display_df["snippet_short"] = display_df["snippet"].fillna("").astype(str).str.slice(0, 120)
                    view_df = display_df[["cycle", "role", "domain", "source", "title_short", "snippet_short", "url", "timestamp"]].rename(
                        columns={"title_short": "title", "snippet_short": "snippet"}
                    )
                    st.write(f"Showing {len(view_df)} citations after filters.")
                    st.dataframe(view_df, use_container_width=True)

                    if not view_df.empty:
                        selected_index = st.selectbox(
                            "Select a citation to view details",
                            options=list(range(len(view_df))),
                            format_func=lambda i: f"{view_df.iloc[i]['source']} - {str(view_df.iloc[i]['title'])[:50]}",
                            key="citations_select",
                        )
                        selected_citation = display_df.iloc[selected_index]
                        with st.expander("Citation details", expanded=False):
                            st.write(f"**Cycle:** {selected_citation['cycle']}")
                            st.write(f"**Role:** {selected_citation['role']}")
                            st.write(f"**Domain:** {selected_citation['domain']}")
                            st.write(f"**Source:** {selected_citation['source']}")
                            st.write(f"**Title:** {selected_citation['title']}")
                            st.write(f"**URL:** {selected_citation['url']}")
                            st.write(f"**Timestamp:** {selected_citation['timestamp']}")
                            st.write(f"**Snippet:** {selected_citation['snippet']}")

                    if not filtered_df.empty:
                        csv_data = filtered_df.to_csv(index=False)
                        st.download_button(
                            "Download citations as CSV",
                            data=csv_data,
                            file_name="citations_export.csv",
                            mime="text/csv",
                        )

                with st.expander("Raw citations JSON"):
                    preview, note = safe_json_preview(citations)
                    if preview is not None:
                        st.code(preview, language="json")
                        if note:
                            st.caption(note)

        with tab_discovery:
            st.markdown("### Discovery log")
            discoveries = load_discovery_log(run_id=active_run_id)
            if not discoveries:
                discoveries = load_discoveries_from_finished_runs()

            if not discoveries:
                st.info("No discovery log entries found yet.")
            else:
                total_disc = len(discoveries)
                domains_disc = sorted({str(d.get("domain", "general")) for d in discoveries})
                best_gain = None
                best_label = None
                for d in discoveries:
                    gain = d.get("rye_gain") or d.get("delta_rye") or 0.0
                    try:
                        gain_f = float(gain)
                    except Exception:
                        gain_f = 0.0
                    if best_gain is None or gain_f > best_gain:
                        best_gain = gain_f
                        best_label = d.get("title") or d.get("summary") or d.get("id") or "discovery"

                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Total discoveries", total_disc)
                with col_d2:
                    st.metric("Domains with discoveries", len(domains_disc))
                with col_d3:
                    st.metric("Best RYE gain", f"{best_gain:.3f}" if best_gain is not None else "n/a")

                if best_label is not None and best_gain is not None:
                    st.caption(f"Top discovery candidate: {str(best_label)[:80]}")

                domains_available = sorted({str(d.get("domain", "general")) for d in discoveries})
                discovery_domain_filter = st.multiselect(
                    "Filter by domain",
                    options=domains_available,
                    default=domains_available,
                    key="discovery_domain_filter",
                )
                min_gain = st.number_input("Minimum RYE gain to show", min_value=0.0, max_value=10.0, value=0.0, step=0.01)

                filtered = []
                for d in discoveries:
                    dom = str(d.get("domain", "general"))
                    if discovery_domain_filter and dom not in discovery_domain_filter:
                        continue
                    gain = d.get("rye_gain") or d.get("delta_rye") or 0.0
                    try:
                        gain_f = float(gain)
                    except Exception:
                        gain_f = 0.0
                    if gain_f < min_gain:
                        continue
                    d_view = dict(d)
                    d_view["rye_gain"] = gain_f
                    filtered.append(d_view)

                if filtered:
                    st.dataframe(pd.DataFrame(filtered), use_container_width=True)
                else:
                    st.info("No discoveries matched the current filters.")

                with st.expander("Raw discovery log JSON"):
                    preview, note = safe_json_preview(discoveries)
                    if preview is not None:
                        st.code(preview, language="json")
                        if note:
                            st.caption(note)

        with tab_snapshots:
            st.markdown("### Snapshots and equilibrium")
            # Prefer real snapshot artifacts (json files) for this run.  If we
            # don't have enough (common in short runs when the worker only
            # writes a final snapshot), synthesize snapshots from cycle history
            # so the "Snapshots" tab stays consistent with the per-cycle chart.
            all_snapshots = load_snapshots()
            snapshots_run: List[Dict[str, Any]] = []
            if active_run_id_local:
                rid = str(active_run_id_local)
                for s in all_snapshots:
                    d = s.get("data") if isinstance(s, dict) else None
                    d_rid = None
                    if isinstance(d, dict):
                        d_rid = d.get("run_id") or d.get("run") or d.get("job_id")
                    if d_rid is not None and str(d_rid) == rid:
                        snapshots_run.append(s)
                        continue
                    # Fallback: match by filename/name when run_id isn't in payload
                    if rid and rid in str(s.get("name") or ""):
                        snapshots_run.append(s)
            else:
                snapshots_run = list(all_snapshots)

            snapshot_source = "snapshot artifacts"
            snapshots: List[Dict[str, Any]] = snapshots_run

            # If we only have a final snapshot artifact (common for short runs),
            # or if artifacts lack usable RYE diagnostics, generate synthetic
            # snapshots from cycle history so the charts stay consistent.
            use_synthetic = False
            if history:
                if not snapshots or len(snapshots) < 2:
                    use_synthetic = True
                else:
                    try:
                        usable = 0
                        for _s in snapshots:
                            _eq = equilibrium_from_snapshot(_s.get("data") or {})
                            if _eq.get("rye_avg") is not None:
                                usable += 1
                                if usable >= 2:
                                    break
                        if usable < 2:
                            use_synthetic = True
                    except Exception:
                        # Be conservative; if anything looks off, fall back to
                        # history-derived snapshots.
                        use_synthetic = True

            if use_synthetic:
                snapshot_source = "cycle history"
                snapshots = []

                # Cap synthetic points to keep UI responsive on long runs.
                n_cycles = len(history)
                max_points = 50
                if n_cycles <= max_points:
                    cycle_indices = list(range(1, n_cycles + 1))
                else:
                    step = max(1, n_cycles // max_points)
                    cycle_indices = list(range(step, n_cycles + 1, step))
                    if cycle_indices and cycle_indices[-1] != n_cycles:
                        cycle_indices.append(n_cycles)

                domain_hint: Optional[str] = None
                try:
                    domain_hint = str((ws0 or {}).get("domain") or (run_state0 or {}).get("domain") or "").strip() or None
                except Exception:
                    domain_hint = None

                for cidx in cycle_indices:
                    prefix = history[:cidx]
                    diag: Dict[str, Any] = {}
                    try:
                        diag = build_run_diagnostics(history=prefix, domain=domain_hint, window=10) or {}
                    except Exception:
                        diag = {}

                    last = prefix[-1] if prefix else {}
                    ts_dt = _event_ts_to_dt(last.get("timestamp") or last.get("ts") or last.get("utc"))
                    ts_str = last.get("timestamp") or last.get("ts") or last.get("utc") or ""

                    snapshots.append(
                        {
                            "name": f"cycle_{cidx}",
                            "timestamp": ts_dt,
                            "data": {
                                "run_id": str(active_run_id_local or ""),
                                "timestamp": ts_str,
                                "current_cycle": cidx,
                                "diagnostics": diag,
                            },
                        }
                    )

            if not snapshots:
                st.info("No snapshots found yet.")
            else:
                labels: List[str] = []
                for s in snapshots:
                    ts = s.get("timestamp")
                    name = str(s.get("name") or "snapshot")
                    label = f"{name} - {ts.isoformat(timespec='seconds')}" if isinstance(ts, datetime) else name
                    labels.append(label)

                st.write(f"Total snapshots: {len(snapshots)} ({snapshot_source})")

                # Timeline: prefer RYE avg from snapshot diagnostics.  If it's
                # missing, this run likely hasn't produced diagnostics yet.
                timeline_rows: List[Dict[str, Any]] = []
                for i, s in enumerate(snapshots):
                    data = s.get("data") or {}
                    eq = equilibrium_from_snapshot(data)
                    rye_avg = eq.get("rye_avg")
                    if rye_avg is None:
                        continue
                    cyc = data.get("current_cycle") or data.get("cycle") or data.get("cycle_index")
                    try:
                        cyc_i = int(cyc) if cyc is not None else (i + 1)
                    except Exception:
                        cyc_i = i + 1
                    timeline_rows.append({"cycle": cyc_i, "RYE avg": rye_avg})

                if timeline_rows:
                    st.markdown("#### Snapshot RYE timeline")
                    df_rye = pd.DataFrame(timeline_rows).set_index("cycle")
                    st.line_chart(df_rye)

                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    idx1 = st.selectbox(
                        "Select first snapshot",
                        options=list(range(len(snapshots))),
                        format_func=lambda i: labels[i],
                    )
                with col_sel2:
                    idx2_default = len(snapshots) - 1
                    idx2 = st.selectbox(
                        "Select second snapshot to compare",
                        options=list(range(len(snapshots))),
                        index=idx2_default,
                        format_func=lambda i: labels[i],
                    )

                s1 = snapshots[idx1]
                s2 = snapshots[idx2]

                st.markdown("#### Snapshot 1 equilibrium view")
                eq1 = equilibrium_from_snapshot(s1["data"])
                # Only show implemented metrics: RYE avg and Stability idx.  Additional metrics are not computed yet.
                col_eq1 = st.columns(2)
                col_eq1[0].metric("RYE avg", f"{eq1['rye_avg']:.3f}" if eq1["rye_avg"] is not None else "n/a")
                col_eq1[1].metric("Stability idx", f"{eq1['stability_index']:.3f}" if eq1["stability_index"] is not None else "n/a")

                st.markdown("#### Snapshot 2 equilibrium view")
                eq2 = equilibrium_from_snapshot(s2["data"])
                # Only show implemented metrics: RYE avg and Stability idx.  Additional metrics are not computed yet.
                col_eq2 = st.columns(2)
                col_eq2[0].metric("RYE avg", f"{eq2['rye_avg']:.3f}" if eq2["rye_avg"] is not None else "n/a")
                col_eq2[1].metric("Stability idx", f"{eq2['stability_index']:.3f}" if eq2["stability_index"] is not None else "n/a")

                st.markdown("#### Equilibrium delta (snapshot2 minus snapshot1)")

                def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
                    if a is None or b is None:
                        return None
                    return b - a

                d_rye = _delta(eq1["rye_avg"], eq2["rye_avg"])
                d_stab = _delta(eq1["stability_index"], eq2["stability_index"])
                # Plateau and equilibrium fraction metrics are not implemented, so omit their delta calculations
                # d_plateau = _delta(eq1["coherence_plateau"], eq2["coherence_plateau"])
                # d_eqfrac = _delta(eq1["equilibrium_fraction"], eq2["equilibrium_fraction"])

                col_de = st.columns(2)
                col_de[0].metric("Delta RYE avg", f"{d_rye:+.3f}" if d_rye is not None else "n/a")
                col_de[1].metric("Delta stability", f"{d_stab:+.3f}" if d_stab is not None else "n/a")

                with st.expander("Raw snapshot 1 JSON"):
                    preview, note = safe_json_preview(s1["data"])
                    if preview is not None:
                        st.code(preview, language="json")
                        if note:
                            st.caption(note)
                with st.expander("Raw snapshot 2 JSON"):
                    preview, note = safe_json_preview(s2["data"])
                    if preview is not None:
                        st.code(preview, language="json")
                        if note:
                            st.caption(note)

        with tab_hypo:
            st.markdown("### Hypothesis manager")
            all_hyps = extract_hypotheses_from_history(history)
            if not all_hyps:
                st.info("No hypotheses recorded yet in cycle history.")
            else:
                domains = sorted({str(h["domain"]) for h in all_hyps})
                roles = sorted({str(h["role"]) for h in all_hyps})
                hypo_domain_filter = st.multiselect("Filter by domain", options=domains, default=domains, key="hypo_domain_filter")
                hypo_role_filter = st.multiselect("Filter by role", options=roles, default=roles, key="hypo_role_filter")
                min_conf = st.number_input("Minimum confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

                filtered_h = []
                for h in all_hyps:
                    d = str(h["domain"])
                    r = str(h["role"])
                    if hypo_domain_filter and d not in hypo_domain_filter:
                        continue
                    if hypo_role_filter and r not in hypo_role_filter:
                        continue
                    conf = h.get("confidence")
                    if isinstance(conf, (int, float)) and conf < min_conf:
                        continue
                    filtered_h.append(h)

                def _score(hh: Dict[str, Any]) -> float:
                    c = hh.get("confidence")
                    try:
                        return float(c) if c is not None else 0.0
                    except Exception:
                        return 0.0

                filtered_h.sort(key=_score, reverse=True)
                view_rows = []
                for h in filtered_h:
                    text = h["text"]
                    text_short = text[:120] + ("..." if len(text) > 120 else "")
                    view_rows.append(
                        {
                            "cycle": h["cycle"],
                            "role": h["role"],
                            "domain": h["domain"],
                            "confidence": h["confidence"],
                            "text": text_short,
                            "timestamp": h["timestamp"],
                        }
                    )
                st.dataframe(pd.DataFrame(view_rows), use_container_width=True)

                hypo_md = ["# Hypotheses\n"]
                for h in filtered_h:
                    conf_txt = f" (confidence ~ {h['confidence']:.2f})" if isinstance(h.get("confidence"), (int, float)) else ""
                    hypo_md.append(f"- [{h['domain']}/{h['role']} cycle {h['cycle']}] {h['text']}{conf_txt}")
                st.download_button("Download hypotheses as Markdown", data="\n".join(hypo_md), file_name="hypotheses_export.md", mime="text/markdown")

        with tab_memory:
            st.markdown("### Memory pruning and compaction")
            total_cycles = len(history)
            st.metric("Total cycles in history", total_cycles)

            has_prune_method = hasattr(memory, "prune_low_value_notes") or hasattr(memory, "prune_history")
            has_pruner_module = _pruner_module is not None

            if not has_prune_method and not has_pruner_module:
                st.info("No pruning hooks detected.")
            else:
                threshold = st.number_input("Approx minimum RYE gain to keep entries", min_value=0.0, max_value=1.0, value=0.01, step=0.005)
                max_keep = st.number_input("Maximum entries to keep (0 means no cap)", min_value=0, max_value=100000, value=5000, step=500)

                if st.button("Run pruning now (experimental)"):
                    pruned_count = 0
                    error_msg = None
                    try:
                        if hasattr(memory, "prune_low_value_notes"):
                            func = getattr(memory, "prune_low_value_notes")
                            pruned_count = int(func(threshold=threshold, max_keep=max_keep))  # type: ignore[arg-type]
                        elif hasattr(memory, "prune_history"):
                            func = getattr(memory, "prune_history")
                            pruned_count = int(func(threshold=threshold, max_keep=max_keep))  # type: ignore[arg-type]
                        elif has_pruner_module and hasattr(_pruner_module, "run_memory_pruning"):
                            func = getattr(_pruner_module, "run_memory_pruning")
                            pruned_count = int(func(memory_store=memory, threshold=threshold, max_keep=max_keep))  # type: ignore[arg-type]
                    except Exception as e:
                        error_msg = str(e)

                    if error_msg:
                        st.error(f"Pruning error: {error_msg}")
                    else:
                        st.success(f"Pruning completed. Approx entries removed: {pruned_count}")
                        # Automatically reload the page so updated counts reflect the pruned history
                        st.info("Reloading the page to reflect updated history and diagnostics...")
                        st.rerun()

        with tab_verify:
            st.markdown("### Verification and cure oriented findings")
            verifications = load_verification_log()
            if not verifications:
                st.info("No verification log found yet.")
            else:
                success_flags = []
                rye_deltas = []
                for v in verifications:
                    ok = v.get("verified") or v.get("success")
                    success_flags.append(bool(ok))
                    delta = v.get("rye_gain") or v.get("delta_rye") or v.get("delta_RYE")
                    try:
                        rye_deltas.append(float(delta))
                    except Exception:
                        continue

                total = len(verifications)
                successful = sum(1 for x in success_flags if x)
                st.metric("Total verifications", total)
                st.metric("Successful verifications", successful)
                st.metric("Success rate", f"{(successful / total * 100.0):.1f}%" if total > 0 else "n/a")
                st.metric(
                    "Average RYE change when verified",
                    f"{(sum(rye_deltas) / len(rye_deltas)):.3f}" if rye_deltas else "n/a",
                )

                view_rows_v = []
                for v in verifications:
                    label = v.get("label") or v.get("id") or v.get("target") or "item"
                    hyp = v.get("hypothesis") or v.get("text")
                    ok = bool(v.get("verified") or v.get("success"))
                    d_rye = v.get("rye_gain") or v.get("delta_rye") or v.get("delta_RYE")
                    domain = v.get("domain", "general")
                    view_rows_v.append(
                        {
                            "label": label,
                            "domain": domain,
                            "verified": ok,
                            "delta_RYE": d_rye,
                            "hypothesis": (hyp or "")[:120] + ("..." if hyp and len(hyp) > 120 else ""),
                        }
                    )
                st.dataframe(pd.DataFrame(view_rows_v), use_container_width=True)

                with st.expander("Raw verification log JSON"):
                    preview, note = safe_json_preview(verifications)
                    if preview is not None:
                        st.code(preview, language="json")
                        if note:
                            st.caption(note)

        with tab_graph:
            st.markdown("### Multi agent insight graph")
            discoveries_for_graph = load_discovery_log(run_id=active_run_id)
            if not discoveries_for_graph:
                discoveries_for_graph = load_discoveries_from_finished_runs()

            if not history:
                st.info("No history yet to build a graph.")
            else:
                dot = build_insight_graph(history=history, discoveries=discoveries_for_graph)
                try:
                    st.graphviz_chart(dot)
                except Exception as e:
                    st.info(f"Graphviz could not render this graph: {e}")

    # ------------------------------
    # Run diagnostics (unified loaders + sources + progress)
    # ------------------------------
    st.markdown("---")
    st.subheader("Run diagnostics")

    # Refresh button (manual)
    # Use a plain label without misencoded symbols
    if st.button("Refresh diagnostics now", key="refresh_diag_btn"):
        st.rerun()

    # Reload unified states (fresh for this render)
    ws, ws_src = load_worker_state_unified(memory)
    run_id = (ws or {}).get("run_id") or active_run_id
    watchdog, watchdog_src = load_watchdog_info_unified(memory, run_id=run_id)
    run_state, run_state_src = load_run_state_unified(memory, run_id_hint=run_id)
    progress_raw, progress_src = load_progress_unified(run_id)

    # If watchdog is missing/empty, synthesize from last activity so the UI doesn't show "None recorded"
    if not _is_meaningful_watchdog(watchdog):
        synth_wd2, synth_src2 = synthesize_watchdog_from_activity(ws, run_state, history_preview)
        if synth_wd2:
            watchdog = synth_wd2
            watchdog_src = synth_src2

    progress_view = compute_progress_view(ws, progress_raw, watchdog, run_id=run_id)

    col_state, col_watchdog = st.columns(2)

    with col_state:
        st.markdown("**Last saved run state**")
        if not run_state:
            st.write("No saved run state found yet.")
        else:
            st.caption(f"Source: `{run_state_src}`")
            st.json(run_state)

        # Clear saved run state (only if MemoryStore supports it)
        if callable(getattr(memory, "clear_run_state", None)):
            if st.button("Clear saved run state", key="clear_run_state_btn"):
                try:
                    memory.clear_run_state()  # type: ignore[call-arg]
                    st.success("Saved run state cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Clear run state failed: {e}")

    with col_watchdog:
        st.markdown("**Watchdog heartbeat**")
        if not watchdog:
            st.write("No watchdog heartbeat found yet.")
        else:
            st.caption(f"Source: `{watchdog_src}`")
            last_beat = watchdog.get("last_beat")
            count = watchdog.get("count", 0)
            seconds_since = watchdog.get("seconds_since_last")
            st.write(f"Last beat: {_format_watchdog_last_beat(last_beat)}")
            st.write(f"Heartbeat count: {count}")
            seconds_since_f = _maybe_float(seconds_since)
            if seconds_since_f is not None:
                st.write(
                    f"Seconds since last beat: {seconds_since_f:.1f} ({_humanize_seconds(seconds_since_f)} ago)"
                )
            else:
                st.write("Seconds since last beat: n/a")

        st.markdown("---")
        st.markdown("**Worker state (engine queue)**")
        if not ws:
            st.write("No worker state recorded yet. The worker may not have started or may be writing elsewhere.")
        else:
            st.caption(f"Source: `{ws_src}`")
            status_val = ws.get("status") or "unknown"
            run_id_val = ws.get("run_id") or ws.get("job_id") or "none"
            mode = ws.get("mode") or ws.get("run_mode")
            domain_ws = ws.get("domain") or ""
            goal_ws = ws.get("goal") or ""

            cols_ws = st.columns(3)
            with cols_ws[0]:
                st.write(f"Status: `{status_val}`")
                if mode:
                    st.write(f"Mode: `{mode}`")
            with cols_ws[1]:
                st.write(f"Run id: `{run_id_val}`")
                if domain_ws:
                    st.write(f"Domain: `{domain_ws}`")
            with cols_ws[2]:
                cur_p = progress_view.get("current")
                tot_p = progress_view.get("total")
                if isinstance(cur_p, int) and isinstance(tot_p, int) and tot_p > 0:
                    pct = (cur_p / tot_p) * 100.0
                    st.write(f"Progress: {cur_p}/{tot_p} ({pct:.1f}%)")
                else:
                    st.write("Progress: n/a")

            if goal_ws:
                st.caption(f"Worker goal: {str(goal_ws)[:140]}")

            with st.expander("Raw worker state JSON"):
                preview, note = safe_json_preview(ws)
                if preview is not None:
                    st.code(preview, language="json")
                    if note:
                        st.caption(note)

            with st.expander("Raw progress JSON (if present)"):
                if progress_raw:
                    st.caption(f"Source: `{progress_src}`")
                    preview, note = safe_json_preview(progress_raw)
                    if preview is not None:
                        st.code(preview, language="json")
                        if note:
                            st.caption(note)
                else:
                    st.info("No progress JSON found. If you want smooth 1/3 -> 2/3 updates, have the worker write `<run_id>_progress.json` each phase/cycle.")

    with st.expander("Diagnostics discovery (files checked)"):
        # Move the report inline toggle into this section.  Users expect to
        # control report rendering where they inspect run diagnostics and
        # citations.  The checkbox is declared here with the same key
        # ``show_reports_inline`` to preserve existing session state.
        st.checkbox(
            "Show report text inline (can be very long)",
            key="show_reports_inline",
            help="When enabled, clicking a report button will render the full text on this page.",
        )

        st.write("These are the standard locations the UI checks for diagnostics artifacts.")
        paths = _candidate_state_paths(run_id=run_id)
        for k in ["worker_state", "run_state", "heartbeat", "events", "progress"]:
            st.markdown(f"**{k}**")
            lst = paths.get(k, [])
            shown = []
            for p in lst[:12]:
                # Use plain ASCII markers to indicate existence instead of misencoded symbols
                exists_marker = "[x]" if p.exists() else "[ ]"
                shown.append(f"{exists_marker} `{p}`")
            st.write("\n".join(shown))

    # ------------------------------
    # Report generation
    # ------------------------------
    st.markdown("---")
    st.subheader("Generate report")

    # Large reports can be thousands of lines. By default, avoid dumping
    # them inline (which forces a ton of scrolling) and encourage downloads.
    # The inline toggle is now located in the diagnostics discovery section;
    # retrieve its state from ``st.session_state`` here.
    show_reports_inline = bool(st.session_state.get("show_reports_inline", False))

    def _present_report(md: str, preview_label: str = "Preview") -> None:
        if show_reports_inline:
            st.markdown(md)
        else:
            with st.expander(preview_label, expanded=False):
                st.markdown(md)

    raw_memory_history: List[Dict[str, Any]] = []
    if callable(getattr(memory, "get_cycle_history", None)):
        try:
            raw_memory_history = memory.get_cycle_history() or []
        except Exception:
            raw_memory_history = []

    # IMPORTANT: prevent blended reports across runs.
    # MemoryStore implementations sometimes return a global history; when we have an
    # active_run_id, filter down to that run.
    if raw_memory_history and active_run_id:
        try:
            raw_memory_history = _get_cycle_history_for_run(raw_memory_history, active_run_id)
        except Exception:
            pass

    if raw_memory_history:
        history_for_reports = raw_memory_history
        used_fallback_history = False
    else:
        history_for_reports = load_history_from_finished_runs()
        used_fallback_history = bool(history_for_reports)

    # Filter report history to the active run_id when possible.
    if history_for_reports and active_run_id:
        try:
            history_for_reports = _get_cycle_history_for_run(history_for_reports, active_run_id)
        except Exception:
            pass

    hours_run_for_reports = compute_run_hours(history_for_reports) if history_for_reports else None
    msil_profile_for_reports = compute_msil_profile(history_for_reports) if history_for_reports else None

    discoveries_for_reports = load_discovery_log(run_id=active_run_id)
    if not discoveries_for_reports:
        discoveries_for_reports = load_discoveries_from_finished_runs()

    col_rep1, col_rep2, col_rep3 = st.columns(3)

    with col_rep1:
        if st.button("Full history report", key="full_history_report_btn"):
            if not history_for_reports:
                st.info("No cycles logged yet, nothing to report.")
            else:
                if used_fallback_history or not callable(generate_report):
                    report_md = build_outcome_summary(history_for_reports)
                else:
                    report_md = generate_report(memory_store=memory, goal=None)  # type: ignore[misc]
                _present_report(report_md, "Full history report preview")
                st.download_button("Download full report (Markdown)", data=report_md, file_name="autonomous_research_report.md", mime="text/markdown")

                if not used_fallback_history and callable(generate_report_pdf):
                    try:
                        pdf_path = generate_report_pdf(memory_store=memory, goal=None, output_path="autonomous_research_report.pdf")  # type: ignore[misc]
                        with open(pdf_path, "rb") as f:
                            st.download_button("Download full report (PDF)", data=f, file_name="autonomous_research_report.pdf", mime="application/pdf")
                    except RuntimeError as e:
                        st.info(str(e))
                    except Exception:
                        st.info("PDF generation failed unexpectedly. Check server logs for details.")
                elif not callable(generate_report_pdf):
                    st.info("PDF export not available (report generator module missing or PDF backend unavailable).")

        if st.button("Full Option C learning speed report", key="option_c_report_btn"):
            if not history_for_reports:
                st.info("No cycles logged yet, Option C learning report is empty.")
            else:
                parts: List[str] = []
                parts.append(build_outcome_summary(history_for_reports))
                parts.append("\n\n---\n\n")
                parts.append(build_breakthrough_report(history_for_reports, discoveries_for_reports))

                if msil_profile_for_reports:
                    parts.append("\n\n---\n\n")
                    parts.append("# MSIL meta skill profile snapshot\n\n")
                    try:
                        msil_json = json.dumps(msil_profile_for_reports, indent=2)
                    except Exception:
                        msil_json = str(msil_profile_for_reports)
                    parts.append("```json\n")
                    parts.append(msil_json)
                    parts.append("\n```")

                if isinstance(hours_run_for_reports, (int, float)):
                    parts.append(f"\n\n_Approximate hours between first and last recorded cycle: {hours_run_for_reports:.2f}_\n")

                option_c_md = "".join(parts)
                _present_report(option_c_md, "Option C learning report preview")
                st.download_button("Download Option C learning report (Markdown)", data=option_c_md, file_name="option_c_learning_report.md", mime="text/markdown")

    with col_rep2:
        if st.button("Findings report (cures, mechanisms, treatments)", key="findings_report_btn"):
            if not history_for_reports:
                st.info("No cycles logged yet, findings report is empty.")
            else:
                if used_fallback_history or not callable(generate_findings_report):
                    findings_md = build_findings_report_from_history(history_for_reports)
                else:
                    findings_md = generate_findings_report(memory_store=memory, goal=None)  # type: ignore[misc]
                _present_report(findings_md, "Findings report preview")
                st.download_button("Download findings report (Markdown)", data=findings_md, file_name="findings_report.md", mime="text/markdown")

                if not used_fallback_history and callable(generate_findings_report_pdf):
                    try:
                        pdf_path_f = generate_findings_report_pdf(memory_store=memory, goal=None, output_path="findings_report.pdf")  # type: ignore[misc]
                        with open(pdf_path_f, "rb") as f:
                            st.download_button("Download findings report (PDF)", data=f, file_name="findings_report.pdf", mime="application/pdf")
                    except RuntimeError as e:
                        st.info(str(e))
                    except Exception:
                        st.info("PDF generation failed unexpectedly. Check server logs for details.")
                elif not callable(generate_findings_report_pdf):
                    st.info("PDF export not available (report generator module missing or PDF backend unavailable).")

    with col_rep3:
        if st.button("Breakthrough snapshot report", key="breakthrough_snapshot_btn"):
            if not history_for_reports:
                st.info("No cycles logged yet, breakthrough snapshot is empty.")
            else:
                br_md = build_breakthrough_report(history_for_reports, discoveries_for_reports)
                _present_report(br_md, "Breakthrough snapshot preview")
                st.download_button("Download breakthrough snapshot (Markdown)", data=br_md, file_name="breakthrough_snapshot_report.md", mime="text/markdown")

        if history_for_reports:
            history_json_str, note_hist = safe_json_preview(history_for_reports, max_chars=500_000, max_items=None)
            if history_json_str is not None:
                st.download_button(
                    "Download full cycle history as JSON",
                    data=history_json_str,
                    file_name="cycle_history.json",
                    mime="application/json",
                    key="history_json_export_btn",
                )

    # ------------------------------
    # Auto-refresh (only while worker appears active)
    # ------------------------------
    if auto_refresh:
        # Determine if the worker appears to be running or recently active
        health_class, _ = derive_health_class(ws, watchdog)
        status_now = str((ws or {}).get("status") or "").lower()
        progress_status = str((progress_raw or {}).get("status") or "").lower() if isinstance(progress_raw, dict) else ""

        # Some worker versions emit richer status strings (e.g. "running_job").
        running_statuses = {
            "running",
            "active",
            "in_progress",
            "working",
            "running_job",
            "running_cycle",
            "processing",
            "busy",
        }

        # Detect queue activity even if worker_state/watchdog are missing.
        queue_active_has_jobs = False
        try:
            q_active = Path(get_queue_root()) / "active"
            queue_active_has_jobs = any(q_active.glob("*_job.json"))
        except Exception:
            queue_active_has_jobs = False

        # If progress indicates an unfinished run, keep refreshing.
        progress_frac = None
        try:
            if isinstance(progress_view, dict):
                progress_frac = progress_view.get("fraction")
        except Exception:
            progress_frac = None
        progress_incomplete = isinstance(progress_frac, (int, float)) and float(progress_frac) < 1.0

        running_like2 = (
            status_now in running_statuses
            or progress_status in {"active", "running", "in_progress", "working"}
            or health_class in {"healthy", "stale"}
            or queue_active_has_jobs
            or (run_id is not None and progress_incomplete)
        )
        if running_like2:
            refresh_ms = int(refresh_seconds * 1000)
            if callable(st_autorefresh):
                # Use the autorefresh component when available
                try:
                    st_autorefresh(interval=refresh_ms, key="ara_autorefresh")  # type: ignore[call-arg]
                except Exception:
                    # On failure, fall back to a manual sleep and rerun
                    time.sleep(float(refresh_seconds))
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            else:
                # When autorefresh is not available, fall back to manual sleep and rerun
                time.sleep(float(refresh_seconds))
                try:
                    st.experimental_rerun()
                except Exception:
                    pass


# Streamlit entry point
if __name__ == "__main__":
    main()
