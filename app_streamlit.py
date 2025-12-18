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
- Optional auto-refresh so the UI can actually show 1/3 → 2/3 → 3/3 while the worker runs

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
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
import yaml

# Optional auto-refresh component (preferred over sleep+rereun if installed)
try:  # pragma: no cover
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import]
except Exception:  # pragma: no cover
    st_autorefresh = None  # type: ignore[assignment]

# Ensure repository root is on sys.path so imports work on Render and local
# This is robust whether this file lives in repo root or in a subfolder (for example app/)
_THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_FILE_DIR
if not (REPO_ROOT / "agent").is_dir() and (_THIS_FILE_DIR.parent / "agent").is_dir():
    REPO_ROOT = _THIS_FILE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These will be filled from run_jobs imports when available
RUNS_BASE_DIR: Optional[Path] = None
RUNS_QUEUE_ROOT: Optional[Path] = None
RUNS_PENDING_DIR: Optional[Path] = None
RUNS_ACTIVE_DIR: Optional[Path] = None
RUNS_FINISHED_DIR: Optional[Path] = None
RUNS_ERROR_DIR: Optional[Path] = None

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
    from memory_store import MemoryStore
    from presets import PRESETS, RUNTIME_PROFILES, get_preset  # type: ignore[no-redef]

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
MAX_SWARM_AGENTS: int = 32

# Limit points in charts so the frontend does not hit RangeError on very long runs.
MAX_POINTS_FOR_CHARTS: int = 1000

# Live console defaults
LIVE_EVENTS_LIMIT: int = 30
DISCOVERY_CARDS_LIMIT: int = 6


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _parse_timestamp_str(ts: str) -> Optional[datetime]:
    """Parse ISO style timestamps, including those with a trailing Z."""
    if not isinstance(ts, str):
        return None
    ts = ts.strip()
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1]
        return datetime.fromisoformat(ts)
    except Exception:
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

    # 2) Fallback to environment variable (in case you set it on the server)
    if not key:
        key = os.getenv("TAVILY_API_KEY")

    # 3) Optional final fallback to secrets (owner only use, can be empty)
    if not key:
        try:
            key = st.secrets.get("TAVILY_API_KEY", None)  # type: ignore[attr-defined]
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

    # Flexible detection by common keys
    web_keys = {"web_search", "browser", "web", "internet"}
    sandbox_keys = {"sandbox", "code_sandbox", "python_sandbox", "exec_sandbox"}

    has_web = any(k in TOOL_REGISTRY for k in web_keys)
    has_sandbox = any(k in TOOL_REGISTRY for k in sandbox_keys)

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
                        st.write(f"• {text} (confidence ~ {conf})")
                    else:
                        st.write(f"• {text}")
                else:
                    st.write(f"• {h}")

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

    st.caption(f"Mode: {mode} • Created at: {created_at}")


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
# Run-result → cycle-history helpers (used for citation table fallback)
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
        st.markdown("#### Sources and citations")
        for s in sources:
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
    """Small helper to load a JSON file and return the decoded data."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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
    q_pending = queue_root / "pending"
    q_active = queue_root / "active"
    q_finished = queue_root / "finished"

    # Generic filenames (shared)
    worker_state = [
        logs / "worker_state.json",
        logs / "engine_worker_state.json",
        logs / "worker_status.json",
        queue_root / "worker_state.json",
        q_active / "worker_state.json",
    ]
    run_state = [
        logs / "run_state.json",
        logs / "last_run_state.json",
        queue_root / "run_state.json",
        q_active / "run_state.json",
    ]
    heartbeat = [
        logs / "watchdog_heartbeat.json",
        logs / "heartbeat.json",
        logs / "worker_heartbeat.json",
        queue_root / "watchdog_heartbeat.json",
        q_active / "watchdog_heartbeat.json",
    ]
    events = [
        logs / "event_log.json",
        logs / "events.json",
        logs / "timeline.json",
        queue_root / "event_log.json",
    ]

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
            ]
        )
        events.extend(
            [
                logs / f"{run_id}_events.json",
                logs / f"{run_id}_event_log.json",
                runs_root / run_id / "events.json",
                runs_root / run_id / "event_log.json",
            ]
        )

    progress = []
    if run_id:
        progress = [
            logs / f"{run_id}_progress.json",
            queue_root / f"{run_id}_progress.json",
            q_active / f"{run_id}_progress.json",
            q_finished / f"{run_id}_progress.json",
            runs_root / run_id / "progress.json",
        ]

    return {
        "worker_state": worker_state,
        "run_state": run_state,
        "heartbeat": heartbeat,
        "events": events,
        "progress": progress,
        "queue_pending": [q_pending],
        "queue_active": [q_active],
        "queue_finished": [q_finished],
        "runs_logs": [logs],
    }


def _normalize_watchdog_info(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    last_beat = raw.get("last_beat") or raw.get("lastBeat") or raw.get("timestamp") or raw.get("ts")
    count = raw.get("count")
    if count is None:
        count = raw.get("heartbeat_count") or raw.get("beats") or 0
    seconds_since = raw.get("seconds_since_last") or raw.get("seconds_since") or raw.get("age_seconds")

    # Try compute seconds since if we have an ISO timestamp
    if seconds_since is None and isinstance(last_beat, str):
        dt = _parse_timestamp_str(last_beat)
        if dt is not None:
            try:
                seconds_since = (datetime.utcnow() - dt).total_seconds()
            except Exception:
                seconds_since = None

    return {
        "last_beat": last_beat,
        "count": _safe_int(count, 0) or 0,
        "seconds_since_last": seconds_since,
    }


def load_watchdog_info_unified(memory: MemoryStore, run_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load heartbeat/watchdog info, preferring MemoryStore but with file fallbacks.

    Returns (info_or_none, source_string).
    """
    # MemoryStore
    func = getattr(memory, "get_watchdog_info", None)
    if callable(func):
        try:
            raw = func()
            info = _normalize_watchdog_info(raw)
            if info:
                return info, "MemoryStore.get_watchdog_info"
        except Exception:
            pass

    # File fallback
    paths = _candidate_state_paths(run_id=run_id)["heartbeat"]
    raw, p = _first_existing_json(paths)
    info = _normalize_watchdog_info(raw)
    if info and p is not None:
        return info, str(p)

    return None, "not found"


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
    if "current" not in ws:
        # several worker styles
        ws["current"] = ws.get("current_cycle") or ws.get("cycle") or ws.get("cycle_index")

    return ws


def load_worker_state_unified(memory: MemoryStore, run_id_hint: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load worker state, preferring MemoryStore but with file fallbacks."""
    # MemoryStore methods (try multiple names)
    for name in ("get_worker_state", "read_worker_state", "load_worker_state"):
        func = getattr(memory, name, None)
        if callable(func):
            try:
                raw = func()
                ws = _normalize_worker_state(raw)
                if ws:
                    return ws, f"MemoryStore.{name}"
            except Exception:
                pass

    paths = _candidate_state_paths(run_id=run_id_hint)["worker_state"]
    raw, p = _first_existing_json(paths)
    ws = _normalize_worker_state(raw)
    if ws and p is not None:
        return ws, str(p)

    return None, "not found"


def load_run_state_unified(memory: MemoryStore, run_id_hint: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load last saved run state, preferring MemoryStore but with file fallbacks."""
    func = getattr(memory, "load_run_state", None)
    if callable(func):
        try:
            raw = func()
            if isinstance(raw, dict) and raw:
                return raw, "MemoryStore.load_run_state"
        except Exception:
            pass

    paths = _candidate_state_paths(run_id=run_id_hint)["run_state"]
    raw, p = _first_existing_json(paths)
    if isinstance(raw, dict) and raw and p is not None:
        return raw, str(p)

    return None, "not found"


def load_progress_unified(run_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load progress JSON if the worker emits one (recommended for smooth 1/3→2/3→3/3 updates)."""
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
    status = (worker_state or {}).get("status") or (progress_state or {}).get("status") or "unknown"
    status_s = str(status).lower()

    if run_id is None:
        run_id = (
            (worker_state or {}).get("run_id")
            or (progress_state or {}).get("run_id")
            or (worker_state or {}).get("job_id")
            or (progress_state or {}).get("job_id")
        )
        if run_id is not None:
            run_id = str(run_id)

    finished_like = status_s in {"finished", "done", "completed", "complete", "success"}

    # Preferred: phase progress (3-phase pipeline etc.)
    phase_cur = (
        (worker_state or {}).get("phase_index")
        or (progress_state or {}).get("phase_index")
        or (progress_state or {}).get("phase_current")
    )
    phase_tot = (
        (worker_state or {}).get("phase_total")
        or (progress_state or {}).get("phase_total")
        or (progress_state or {}).get("phase_count")
    )
    phase_name = (worker_state or {}).get("phase_name") or (progress_state or {}).get("phase_name") or ""

    # Cycle progress
    cur = (
        (worker_state or {}).get("effective_current")
        or (progress_state or {}).get("effective_current")
        or (worker_state or {}).get("current")
        or (progress_state or {}).get("current")
        or (progress_state or {}).get("current_cycle")
        or (progress_state or {}).get("cycle")
        or (progress_state or {}).get("cycle_index")
    )
    tot = (
        (worker_state or {}).get("effective_total")
        or (progress_state or {}).get("effective_total")
        or (worker_state or {}).get("total")
        or (progress_state or {}).get("total")
        or (progress_state or {}).get("total_cycles")
        or (progress_state or {}).get("max_cycles")
    )

    # Select which progress track to display
    use_phase = _safe_int(phase_tot, None)
    if use_phase is not None and use_phase > 0:
        c = _safe_int(phase_cur, 0) or 0
        t = use_phase

        # If finished, clamp to full
        if finished_like:
            c_disp = t
        else:
            # Remember whether this run looks zero-indexed for phase progress
            key = f"ara_phase_zero_indexed::{run_id or 'global'}"
            zero_indexed = bool(st.session_state.get(key, False))

            hb_count = _safe_int((watchdog or {}).get("count"), 0) or 0
            hb_last = (watchdog or {}).get("last_beat")
            started = bool(hb_count > 0 or hb_last)

            # If we observe phase_cur==0 while actively running, treat as 0-based and mark it
            running_like = status_s in {"running", "active", "in_progress", "working"}
            if running_like and c == 0 and started:
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

            # Small UX: if running, started, and c_disp==0, show 1 as "in progress"
            if running_like and started and c_disp == 0 and t > 0:
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
    if c2 is None or t2 is None or t2 <= 0:
        return {"kind": "none", "current": None, "total": None, "fraction": None, "label": ""}

    # If finished, clamp to full
    if finished_like:
        c_disp2 = t2
    else:
        c_disp2 = min(max(c2, 0), t2)

        # Small improvement: if run is active and current == 0 but we have heartbeat activity, show "1" as "in progress"
        hb_count = _safe_int((watchdog or {}).get("count"), 0) or 0
        hb_last = (watchdog or {}).get("last_beat")
        if status_s in {"running", "active", "in_progress", "working"} and c_disp2 == 0 and t2 > 0:
            if hb_count > 0 or hb_last:
                c_disp2 = 1

    frac2 = float(c_disp2) / float(t2) if t2 > 0 else None
    return {
        "kind": "cycle",
        "current": c_disp2,
        "total": t2,
        "fraction": _clamp_float(frac2),
        "label": "cycles",
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
    if status in {"error", "failed"}:
        return "offline", "Error"

    # Heartbeat driven
    if seconds_since is None:
        if status in {"running", "active", "in_progress", "working"}:
            return "stale", "Running (no heartbeat)"
        return "unknown", "Unknown"

    # These thresholds assume heartbeat interval ~60s (your config uses 60)
    if seconds_since <= 90:
        return "healthy", "Healthy"
    if seconds_since <= 300:
        return "stale", "Stale"
    if status in {"running", "active", "in_progress", "working"} and hb_count > 0:
        return "offline", "Heartbeat lost"
    return "offline", "Offline"


def inject_base_styles() -> None:
    """Global UI polish + styles for heartbeat bar, cards, chips, and event feed."""
    st.markdown(
        """
<style>
/* Layout rhythm */
.block-container { padding-top: 0.75rem; padding-bottom: 2.5rem; max-width: 1180px; }

/* Sticky topbar */
.ara-topbar-wrap{
  position: sticky;
  top: 0;
  z-index: 9999;
  margin: -0.75rem -1rem 1rem -1rem;
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
  width: 160px;
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
        """,
        unsafe_allow_html=True,
    )


def render_topbar(
    worker_state: Optional[Dict[str, Any]],
    watchdog: Optional[Dict[str, Any]],
    progress_view: Dict[str, Any],
    autonomy_view: Dict[str, Any],
) -> None:
    health_class, health_label = derive_health_class(worker_state, watchdog)

    run_id = (worker_state or {}).get("run_id") or ""
    mode = (worker_state or {}).get("mode") or (worker_state or {}).get("run_mode") or ""
    status = (worker_state or {}).get("status") or "unknown"

    seconds_since = _maybe_float((watchdog or {}).get("seconds_since_last"))
    beat_txt = _humanize_seconds(seconds_since)

    cur = progress_view.get("current")
    tot = progress_view.get("total")
    frac = progress_view.get("fraction")
    kind_label = progress_view.get("label") or ""
    if isinstance(cur, int) and isinstance(tot, int) and tot > 0:
        progress_text = f"{cur}/{tot}"
    else:
        progress_text = "—"

    autonomy_label = autonomy_view.get("label") or "Autonomy"
    autonomy_score = autonomy_view.get("score")
    if isinstance(autonomy_score, int):
        autonomy_text = f"{autonomy_label} ({autonomy_score}/4)"
    else:
        autonomy_text = autonomy_label

    width_pct = 0
    if isinstance(frac, (int, float)):
        width_pct = int(max(0.0, min(1.0, float(frac))) * 100)

    # Escape user-controlled strings
    run_id_html = html.escape(str(run_id)) if run_id else ""
    mode_html = html.escape(str(mode)) if mode else ""
    status_html = html.escape(str(status))
    autonomy_html = html.escape(str(autonomy_text))

    mid_parts = []
    if run_id_html:
        mid_parts.append(f'Run <code>{run_id_html}</code>')
    if mode_html:
        mid_parts.append(f'Mode <code>{mode_html}</code>')
    mid_txt = " • ".join(mid_parts) if mid_parts else "No active run detected"

    right_txt = f"{progress_text}"
    if kind_label:
        right_txt += f" {html.escape(str(kind_label))}"

    st.markdown(
        f"""
<div class="ara-topbar-wrap">
  <div class="ara-topbar">
    <div class="ara-topbar-left">
      <div class="ara-dot {health_class}"></div>
      <div class="ara-topbar-title">{html.escape(health_label)}</div>
      <div class="ara-kv">Beat {html.escape(beat_txt)} ago</div>
      <div class="ara-kv">Status <code>{status_html}</code></div>
    </div>

    <div class="ara-topbar-mid">
      <div class="ara-kv">{mid_txt}</div>
    </div>

    <div class="ara-topbar-right">
      <div class="ara-kv">{autonomy_html}</div>
      <div class="ara-kv"><code>{html.escape(right_txt)}</code></div>
      <div class="ara-mini-progress" title="progress">
        <div style="width:{width_pct}%"></div>
      </div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def compute_autonomy_view(
    job_payload: Optional[Dict[str, Any]],
    worker_state: Optional[Dict[str, Any]],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute autonomy level based on configured features + observed stability."""
    cfg = {}
    if isinstance(job_payload, dict):
        cfg = job_payload.get("config") if isinstance(job_payload.get("config"), dict) else {}
        if not isinstance(cfg, dict):
            cfg = {}

    source_controls = cfg.get("source_controls") if isinstance(cfg.get("source_controls"), dict) else {}
    monitoring = cfg.get("monitoring") if isinstance(cfg.get("monitoring"), dict) else {}
    swarm_cfg = cfg.get("swarm_config") or cfg.get("swarm") or {}
    if not isinstance(swarm_cfg, dict):
        swarm_cfg = {}

    # Feature flags
    has_tools = bool(source_controls.get("web") or source_controls.get("sandbox") or source_controls.get("tavily_enabled"))
    is_multi = bool(cfg.get("multi_agent_pair") or (_safe_int(swarm_cfg.get("swarm_size"), 1) or 1) > 1)
    is_monitored = bool(
        monitoring.get("heartbeat_enabled", True)
        or monitoring.get("run_state_enabled", True)
        or monitoring.get("snapshots_enabled", False)
    )

    # Observed stability (optional)
    stable_signal = False
    if isinstance(diagnostics, dict):
        stab = diagnostics.get("stability_index")
        eq = diagnostics.get("equilibrium_fraction") or diagnostics.get("equilibrium")
        try:
            if isinstance(stab, (int, float)) and float(stab) >= 0.70:
                stable_signal = True
        except Exception:
            pass
        try:
            if isinstance(eq, (int, float)) and float(eq) >= 0.60:
                stable_signal = True
        except Exception:
            pass

    # Score ladder
    score = 0
    label = "Assisted"
    explain = "Manual finite runs, UI queues jobs. No tool autonomy detected."

    if has_tools:
        score = 1
        label = "Tool-assisted"
        explain = "Uses external tools (web/sandbox) when enabled."

    if is_multi:
        score = 2
        label = "Multi-agent"
        explain = "Multiple roles/agents contribute (two-stage or swarm)."

    if is_monitored:
        score = max(score, 3)
        label = "Self-monitoring"
        explain = "Emits heartbeat/state/snapshots for operational stability."

    if stable_signal:
        score = max(score, 4)
        label = "Self-stabilizing"
        explain = "Signals suggest stability/equilibrium emerging (heuristic)."

    # Add current state context
    status = (worker_state or {}).get("status")
    if status:
        explain = f"{explain} Worker status: {status}."

    return {"score": score, "label": label, "explain": explain}


def _infer_agents_from_job_config(job_payload: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(job_payload, dict):
        return ["agent"]
    cfg = job_payload.get("config") if isinstance(job_payload.get("config"), dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    mode = str(cfg.get("mode") or "").lower()

    swarm_cfg = cfg.get("swarm_config") or cfg.get("swarm") or {}
    if not isinstance(swarm_cfg, dict):
        swarm_cfg = {}

    swarm_size = _safe_int(swarm_cfg.get("swarm_size"), 1) or 1
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

    chips = []
    for a in agents[:MAX_SWARM_AGENTS]:
        a_clean = str(a)
        is_active = active_agent and (a_clean == active_agent or a_clean.split("_", 1)[0] == str(active_agent))
        cls = "ara-chip active" if is_active else "ara-chip"
        chips.append(f'<span class="{cls}">● {html.escape(a_clean)}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def load_discovery_log() -> List[Dict[str, Any]]:
    """Try to load discovery log entries from standard locations."""
    candidates: List[Path] = []

    # Prefer shared runs-root logs, then repo-root logs
    try:
        runs_logs = Path(get_runs_root()) / "logs"
        candidates.extend(
            [
                runs_logs / "discovery_log.json",
                runs_logs / "discovery" / "discovery_log.json",
                runs_logs / "discovery" / "discoveries.json",
            ]
        )
    except Exception:
        pass

    candidates.extend(
        [
            REPO_ROOT / "logs" / "discovery_log.json",
            REPO_ROOT / "logs" / "discovery" / "discovery_log.json",
            REPO_ROOT / "logs" / "discovery" / "discoveries.json",
        ]
    )

    for p in candidates:
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


def load_event_log_unified(run_id: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Load event log from common paths.

    If your worker doesn't emit an event log yet, this will be empty (and we synthesize from history).
    """
    paths = _candidate_state_paths(run_id=run_id)["events"]
    raw, p = _first_existing_json(paths)

    if isinstance(raw, dict):
        maybe = raw.get("events") or raw.get("timeline") or raw.get("items")
        if isinstance(maybe, list):
            events = [e for e in maybe if isinstance(e, dict)]
            return events, str(p) if p else "dict container"
        return [], str(p) if p else "dict container"

    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)], str(p) if p else "list container"

    return [], "not found"


def _event_ts_to_str(ts_val: Any) -> str:
    if isinstance(ts_val, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(ts_val)).isoformat(timespec="seconds") + "Z"
        except Exception:
            return ""
    if isinstance(ts_val, str):
        return ts_val
    return ""


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
            parts.append(f"ΔR {float(d_r):.3f}")
        if repairs_n:
            parts.append(f"{repairs_n} repairs")
        if notes_n:
            parts.append(f"{notes_n} notes")
        if hyps_n:
            parts.append(f"{hyps_n} hypotheses")

        msg = " • ".join(parts)
        events.append(
            {
                "ts": e.get("timestamp") or "",
                "kind": "cycle",
                "message": msg,
            }
        )
    return events


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
        kind = str(ev.get("kind") or ev.get("type") or "event")
        msg = str(ev.get("message") or ev.get("text") or ev.get("summary") or "")
        if not msg:
            continue
        meta = " • ".join([x for x in [ts, kind] if x])
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

            st.markdown(
                f"""
<div class="ara-card">
  <div class="ara-card-title">{html.escape(title)}</div>
  <div class="ara-card-sub">{html.escape(domain)} • confidence {html.escape(conf_txt)} • RYE gain {html.escape(gain_txt)} • evidence {ev_n}</div>
  <div class="ara-card-mono">{html.escape(desc) if desc else "—"}</div>
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
    metrics = snapshot.get("metrics") or snapshot.get("rye_metrics") or {}
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

        created_at_raw = getattr(job, "created_at", None) if hasattr(job, "created_at") else (job.get("created_at") if isinstance(job, dict) else None)
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

        created_at_raw = getattr(job, "created_at", None) if hasattr(job, "created_at") else (job.get("created_at") if isinstance(job, dict) else None)
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


# -------------------------------------------------------------------
# Main Streamlit app
# -------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="ARA powered by Reparodynamics", page_icon="🧪", layout="wide")

    inject_base_styles()

    # Resolve base + queue paths once for UI display
    runs_base_dir = get_runs_root()
    queue_root_dir = get_queue_root()
    queue_pending_dir = str(Path(queue_root_dir) / "pending")

    # Shared MemoryStore instance (read-only UI)
    memory = init_memory_store()

    # Determine active run id as early as possible (worker_state run_id > last queued hint > queue scan)
    ws0, ws_src0 = load_worker_state_unified(memory)
    active_run_hint = st.session_state.get("active_run_id_hint")
    if active_run_hint is not None:
        active_run_hint = str(active_run_hint)
    active_run_id = (ws0 or {}).get("run_id") or active_run_hint or _derive_active_run_id_from_queue()

    # Diagnostics sources (for top bar + live console)
    watchdog0, watchdog_src0 = load_watchdog_info_unified(memory, run_id=active_run_id)
    progress0_raw, progress_src0 = load_progress_unified(active_run_id)
    progress_view0 = compute_progress_view(ws0, progress0_raw, watchdog0, run_id=active_run_id)

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

    diagnostics_preview: Optional[Dict[str, Any]] = None
    if history_preview:
        try:
            diagnostics_preview = build_run_diagnostics(history=history_preview, domain=None, window=10)
        except Exception:
            diagnostics_preview = None

    # Autonomy + agents (from job payload if available)
    job_payload0, job_payload_src0 = load_job_payload_from_disk(active_run_id) if active_run_id else (None, "no run_id")
    autonomy_view0 = compute_autonomy_view(job_payload0, ws0, diagnostics_preview)
    agents0 = _infer_agents_from_job_config(job_payload0)

    # Event log (or synthesized)
    event_log0, event_src0 = load_event_log_unified(active_run_id)
    narrative_events = event_log0 if event_log0 else build_narrative_events_from_history(history_preview, limit=LIVE_EVENTS_LIMIT)

    # Sticky topbar removed (was rendering as raw HTML on some clients)
    # render_topbar(ws0, watchdog0, progress_view0, autonomy_view0)

    # Header: ARA with gradient and powered by pill
    st.markdown(
        """
        <style>
        .ara-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
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
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Finite mode only • Queue based runs • Engine worker processes jobs from `{queue_pending_dir}` for `*_job.json` files.\n"
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
    # Sidebar: Live updates (this fixes the “0/3 then 3/3” perception issue)
    # ------------------------------------------------------------------
    st.sidebar.subheader("Live updates")
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh while worker is running",
        value=True,
        help="Enables live dashboard updates so progress can show 1/3 → 2/3 → 3/3 during runs.",
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
            st.sidebar.caption(f"Progress: {cur}/{tot} {label}".strip())
            if isinstance(frac, (int, float)):
                try:
                    st.sidebar.progress(min(max(float(frac), 0.0), 1.0))
                except Exception:
                    pass
        else:
            st.sidebar.caption(f"Worker status: {(ws0.get('status') or 'unknown')}")

    # ------------------------------------------------------------------
    # Live Console (visual upgrade set)
    # ------------------------------------------------------------------
    st.markdown("### Live console")
    left_console, right_console = st.columns([1, 2], gap="large")

    with left_console:
        st.markdown(
            f"""
<div class="ara-card">
  <div class="ara-card-title">Autonomy level</div>
  <div class="ara-card-sub">{html.escape(str(autonomy_view0.get("label","Unknown")))} • {int(autonomy_view0.get("score",0))}/4</div>
  <div class="ara-card-mono">{html.escape(str(autonomy_view0.get("explain","")))}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        # progress bar for autonomy
        try:
            score = int(autonomy_view0.get("score", 0))
            st.progress(max(0.0, min(1.0, score / 4.0)))
        except Exception:
            pass

        st.write("")  # spacing
        active_agent = None
        if ws0:
            active_agent = ws0.get("role") or ws0.get("active_role") or ws0.get("current_role")
            if active_agent is not None:
                active_agent = str(active_agent)
        render_agent_presence(agents0, active_agent=active_agent)

    with right_console:
        render_narrative_feed(narrative_events, source_label=event_src0 if event_src0 != "not found" else "synthesized")

        with st.expander("Raw event log JSON (if available)"):
            if event_log0:
                preview, note = safe_json_preview(event_log0, max_items=120)
                if preview:
                    st.code(preview, language="json")
                    if note:
                        st.caption(note)
            else:
                st.info("No event log file found. This view is synthesizing events from cycle history.")

    # Discovery cards under live console
    discoveries_live = load_discovery_log()
    if not discoveries_live:
        discoveries_live = load_discoveries_from_finished_runs()
    render_discovery_cards(discoveries_live)

    # ------------------------------------------------------------------
    # Sidebar: Run configuration (original)
    # ------------------------------------------------------------------
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
                    "max_cycles_per_agent": 1,
                    "stagger_start": False,
                    "max_agents_per_tick": 0,
                    "role_goals": {name: role_specific_goal(goal_clean, name) for name, _ in swarm_roles} if swarm_roles else {},
                }
            else:
                swarm_config = {
                    "swarm_size": 1,
                    "roles": ["agent"],
                    "max_cycles_per_agent": int(cycles),
                    "stagger_start": False,
                    "max_agents_per_tick": 0,
                }

            longevity_config: Dict[str, Any] = {}
            if str(domain_tag).lower() in {"longevity", "aging", "anti_aging"}:
                longevity_defaults = preset.get("longevity_config", {})
                if isinstance(longevity_defaults, dict):
                    longevity_config = {
                        "hallmark_targets": longevity_defaults.get("hallmark_targets", []),
                        "curriculum_profile": longevity_defaults.get("curriculum_profile"),
                    }

            total_cycles_requested = int(cycles)
            if mode == "swarm":
                total_cycles_requested = int(cycles) * int(swarm_size)

            run_config: Dict[str, Any] = {
                "goal": goal_clean,
                "domain": domain_tag,
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

            if pdf_payload is not None:
                run_config["pdf"] = pdf_payload
                run_config["pdf_payload"] = pdf_payload

            t_key = st.session_state.get("tavily_key")
            t_tail = t_key[-4:] if isinstance(t_key, str) and len(t_key) >= 4 else None

            meta: Dict[str, Any] = {
                "run_label": (run_label or "experiment").strip(),
                "preset_key": selected_key,
                "preset_label": preset.get("label", selected_label),
                "domain": domain_tag,
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
                    st.markdown("---")

        st.markdown("#### Queued runs")
        st.caption(f"Queue directory: `{pending_dir}`")

        if st.button("🧹 Clear job queue", key="clear_queue_btn"):
            removed = 0

            def _is_uuid_stem(stem: str) -> bool:
                try:
                    uuid.UUID(stem)
                    return True
                except Exception:
                    return False

            dirs_to_scan: List[Tuple[Path, bool]] = []
            try:
                dirs_to_scan.append((Path(pending_dir), False))
            except Exception:
                pass
            try:
                legacy_pending = Path(get_runs_root()) / "pending"
                dirs_to_scan.append((legacy_pending, False))
            except Exception:
                pass
            try:
                queue_root_path = Path(get_queue_root())
                dirs_to_scan.append((queue_root_path, True))
            except Exception:
                queue_root_path = None  # type: ignore[assignment]

            for dpath, root_level in dirs_to_scan:
                if not isinstance(dpath, Path) or not dpath.exists() or not dpath.is_dir():
                    continue
                for fp in dpath.glob("*.json"):
                    name = fp.name
                    if name.endswith("_progress.json") or name.endswith("_results.json") or name.endswith("_result.json"):
                        continue
                    if name.endswith("_job.json"):
                        try:
                            fp.unlink()
                            removed += 1
                        except Exception:
                            pass
                        continue
                    if _is_uuid_stem(fp.stem):
                        try:
                            fp.unlink()
                            removed += 1
                        except Exception:
                            pass
                        continue

            st.success(f"Cleared {removed} queued job file(s) (canonical + legacy).")
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

            st.markdown("#### 10x learning dashboard (Option C signals)")

            volatility_info: Dict[str, Any] = {}
            try:
                volatility_info = rye_volatility_signature(diagnostics)
            except TypeError:
                try:
                    volatility_info = rye_volatility_signature(history=history, domain=None, window=10)  # type: ignore[call-arg]
                except Exception:
                    volatility_info = {}
            except Exception:
                volatility_info = {}

            equilibrium_info: Dict[str, Any] = {}
            try:
                equilibrium_info = detect_rye_equilibrium(diagnostics)
            except TypeError:
                try:
                    equilibrium_info = detect_rye_equilibrium(history=history, domain=None, window=10)  # type: ignore[call-arg]
                except Exception:
                    equilibrium_info = {}
            except Exception:
                equilibrium_info = {}

            harmonic_val: Optional[float] = None
            try:
                harmonic_val = tgrm_harmonic_index(diagnostics)
            except TypeError:
                try:
                    harmonic_val = tgrm_harmonic_index(history=history, domain=None, window=10)  # type: ignore[call-arg]
                except Exception:
                    harmonic_val = None
            except Exception:
                harmonic_val = None

            vol_score = volatility_info.get("volatility_score")
            vol_regime = volatility_info.get("regime") or volatility_info.get("label")
            eq_flag = equilibrium_info.get("in_equilibrium")
            eq_reason = equilibrium_info.get("reason")
            eq_state_text = "yes" if eq_flag is True else ("no" if eq_flag is False else "unknown")

            oc_cols = st.columns(3)
            with oc_cols[0]:
                st.metric("Equilibrium detected", eq_state_text)
            with oc_cols[1]:
                st.metric("Volatility score", f"{vol_score:.3f}" if isinstance(vol_score, (int, float)) else "n/a")
            with oc_cols[2]:
                st.metric("TGRM harmonic index", f"{harmonic_val:.3f}" if isinstance(harmonic_val, (int, float)) else "n/a")

            if vol_regime:
                st.caption(f"Volatility regime: {vol_regime}")
            if eq_reason:
                st.caption(f"Equilibrium reasoning: {eq_reason}")

            with st.expander("Autonomy safety and failure envelope"):
                st.write("Autonomy safety envelope:")
                st.json(env)
                st.write("Early failure warning score:")
                st.json(fail)

            with st.expander("Raw Option C style signals"):
                raw_signals = {
                    "diagnostics": diagnostics,
                    "volatility": volatility_info,
                    "equilibrium": equilibrium_info,
                    "harmonic_index": harmonic_val,
                    "breakthrough_near_term": bp_short,
                    "breakthrough_90d": bp90,
                    "run_tier": tier_info,
                    "msil_profile": msil_profile_full,
                }
                preview, note = safe_json_preview(raw_signals)
                if preview is not None:
                    st.code(preview, language="json")
                    if note:
                        st.caption(note)

            with st.expander("Raw history JSON"):
                preview, note = safe_json_preview(history, max_items=MAX_POINTS_FOR_CHARTS)
                if preview is not None:
                    st.code(preview, language="json")
                    if note:
                        st.caption(note)

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
            discoveries = load_discovery_log()
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
            snapshots = load_snapshots()
            if not snapshots:
                st.info("No snapshots found yet.")
            else:
                labels = []
                for s in snapshots:
                    ts = s["timestamp"]
                    label = f"{s['name']} - {ts.isoformat(timespec='seconds')}" if isinstance(ts, datetime) else s["name"]
                    labels.append(label)

                st.write(f"Total snapshots: {len(snapshots)}")

                rye_series = []
                for s in snapshots:
                    eq = equilibrium_from_snapshot(s["data"])
                    if eq["rye_avg"] is not None:
                        rye_series.append(eq["rye_avg"])
                if rye_series:
                    st.markdown("#### Snapshot RYE timeline")
                    # Use a dataframe so Streamlit renders consistently
                    df_rye = pd.DataFrame({"snapshot_index": list(range(1, len(rye_series) + 1)), "RYE avg": rye_series}).set_index("snapshot_index")
                    st.line_chart(df_rye)

                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    idx1 = st.selectbox("Select first snapshot", options=list(range(len(snapshots))), format_func=lambda i: labels[i])
                with col_sel2:
                    idx2_default = len(snapshots) - 1
                    idx2 = st.selectbox("Select second snapshot to compare", options=list(range(len(snapshots))), index=idx2_default, format_func=lambda i: labels[i])

                s1 = snapshots[idx1]
                s2 = snapshots[idx2]

                st.markdown("#### Snapshot 1 equilibrium view")
                eq1 = equilibrium_from_snapshot(s1["data"])
                col_eq1 = st.columns(4)
                col_eq1[0].metric("RYE avg", f"{eq1['rye_avg']:.3f}" if eq1["rye_avg"] is not None else "n/a")
                col_eq1[1].metric("Stability idx", f"{eq1['stability_index']:.3f}" if eq1["stability_index"] is not None else "n/a")
                col_eq1[2].metric("Coherence plateau", f"{eq1['coherence_plateau']:.3f}" if eq1["coherence_plateau"] is not None else "n/a")
                col_eq1[3].metric("Equilibrium fraction", f"{eq1['equilibrium_fraction']:.3f}" if eq1["equilibrium_fraction"] is not None else "n/a")

                st.markdown("#### Snapshot 2 equilibrium view")
                eq2 = equilibrium_from_snapshot(s2["data"])
                col_eq2 = st.columns(4)
                col_eq2[0].metric("RYE avg", f"{eq2['rye_avg']:.3f}" if eq2["rye_avg"] is not None else "n/a")
                col_eq2[1].metric("Stability idx", f"{eq2['stability_index']:.3f}" if eq2["stability_index"] is not None else "n/a")
                col_eq2[2].metric("Coherence plateau", f"{eq2['coherence_plateau']:.3f}" if eq2["coherence_plateau"] is not None else "n/a")
                col_eq2[3].metric("Equilibrium fraction", f"{eq2['equilibrium_fraction']:.3f}" if eq2["equilibrium_fraction"] is not None else "n/a")

                st.markdown("#### Equilibrium delta (snapshot2 minus snapshot1)")

                def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
                    if a is None or b is None:
                        return None
                    return b - a

                d_rye = _delta(eq1["rye_avg"], eq2["rye_avg"])
                d_stab = _delta(eq1["stability_index"], eq2["stability_index"])
                d_plateau = _delta(eq1["coherence_plateau"], eq2["coherence_plateau"])
                d_eqfrac = _delta(eq1["equilibrium_fraction"], eq2["equilibrium_fraction"])

                col_de = st.columns(4)
                col_de[0].metric("Delta RYE avg", f"{d_rye:+.3f}" if d_rye is not None else "n/a")
                col_de[1].metric("Delta stability", f"{d_stab:+.3f}" if d_stab is not None else "n/a")
                col_de[2].metric("Delta plateau", f"{d_plateau:+.3f}" if d_plateau is not None else "n/a")
                col_de[3].metric("Delta equilibrium", f"{d_eqfrac:+.3f}" if d_eqfrac is not None else "n/a")

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
                        st.info("Reload the page to reflect updated history and diagnostics.")

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
            discoveries_for_graph = load_discovery_log()
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
    if st.button("↻ Refresh diagnostics now", key="refresh_diag_btn"):
        st.rerun()

    # Reload unified states (fresh for this render)
    ws, ws_src = load_worker_state_unified(memory)
    run_id = (ws or {}).get("run_id") or active_run_id
    watchdog, watchdog_src = load_watchdog_info_unified(memory, run_id=run_id)
    run_state, run_state_src = load_run_state_unified(memory, run_id_hint=run_id)
    progress_raw, progress_src = load_progress_unified(run_id)
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
            st.write(f"Last beat: {last_beat if last_beat else 'None recorded'}")
            st.write(f"Heartbeat count: {count}")
            st.write(f"Seconds since last beat: {_humanize_seconds(_maybe_float(seconds_since))}")

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
                    st.info("No progress JSON found. If you want smooth 1/3 → 2/3 updates, have the worker write `<run_id>_progress.json` each phase/cycle.")

    with st.expander("Diagnostics discovery (files checked)"):
        st.write("These are the standard locations the UI checks for diagnostics artifacts.")
        paths = _candidate_state_paths(run_id=run_id)
        for k in ["worker_state", "run_state", "heartbeat", "events", "progress"]:
            st.markdown(f"**{k}**")
            lst = paths.get(k, [])
            shown = []
            for p in lst[:12]:
                exists = "✅" if p.exists() else "—"
                shown.append(f"{exists} `{p}`")
            st.write("\n".join(shown))

    # ------------------------------
    # Report generation
    # ------------------------------
    st.markdown("---")
    st.subheader("Generate report")

    raw_memory_history: List[Dict[str, Any]] = []
    if callable(getattr(memory, "get_cycle_history", None)):
        try:
            raw_memory_history = memory.get_cycle_history() or []
        except Exception:
            raw_memory_history = []

    if raw_memory_history:
        history_for_reports = raw_memory_history
        used_fallback_history = False
    else:
        history_for_reports = load_history_from_finished_runs()
        used_fallback_history = bool(history_for_reports)

    hours_run_for_reports = compute_run_hours(history_for_reports) if history_for_reports else None
    msil_profile_for_reports = compute_msil_profile(history_for_reports) if history_for_reports else None

    discoveries_for_reports = load_discovery_log()
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
                st.markdown(report_md)
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
                st.markdown(option_c_md)
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
                st.markdown(findings_md)
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
                st.markdown(br_md)
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
        health_class, _ = derive_health_class(ws, watchdog)
        status_now = str((ws or {}).get("status") or "").lower()
        running_like = status_now in {"running", "active", "in_progress", "working"} or health_class in {"healthy", "stale"}
        if running_like:
            if callable(st_autorefresh):
                try:
                    st_autorefresh(interval=int(refresh_seconds * 1000), key="ara_autorefresh")
                except Exception:
                    # fallback to sleep+rereun
                    time.sleep(float(refresh_seconds))
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
            else:
                time.sleep(float(refresh_seconds))
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass


# Streamlit entry point
if __name__ == "__main__":
    main()
