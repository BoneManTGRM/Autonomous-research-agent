"""Enhanced Streamlit interface for the Autonomous Research Agent.

Features:
- Finite mode only with manual cycle budgets (no timed presets in this build)
- Researcher + Critic multi agent mode
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
import json
import os
import re
import sys
import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import streamlit as st
import yaml

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
RUNS_PENDING_DIR: Optional[Path] = None
RUNS_ACTIVE_DIR: Optional[Path] = None
RUNS_FINISHED_DIR: Optional[Path] = None
RUNS_ERROR_DIR: Optional[Path] = None

# -------------------------------------------------------------------
# Imports: prefer package layout agent.*, guarded flat fallback
# -------------------------------------------------------------------
try:
    # Package layout (recommended, what you have on Render)
    from agent.memory_store import MemoryStore
    from agent.presets import PRESETS, get_preset, RUNTIME_PROFILES
    from agent.report_generator import (
        generate_report,
        generate_findings_report,
        generate_report_pdf,
        generate_findings_report_pdf,
    )
    # Only import the rye_metrics symbols that are actually used here
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
    )
    from agent.report_builder import build_agent_report

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
        from agent.run_jobs import (
            create_job,
            list_jobs as list_run_jobs,
            result_path,
            BASE_DIR as RUNS_BASE_DIR,
            PENDING_DIR as RUNS_PENDING_DIR,
            ACTIVE_DIR as RUNS_ACTIVE_DIR,
            FINISHED_DIR as RUNS_FINISHED_DIR,
            ERROR_DIR as RUNS_ERROR_DIR,
        )
    except Exception:
        create_job = None  # type: ignore[assignment]
        list_run_jobs = None  # type: ignore[assignment]
        result_path = None  # type: ignore[assignment]
        RUNS_BASE_DIR = None
        RUNS_PENDING_DIR = None
        RUNS_ACTIVE_DIR = None
        RUNS_FINISHED_DIR = None
        RUNS_ERROR_DIR = None

except ModuleNotFoundError as e:
    # If the agent package itself is missing, allow flat layout fallback.
    # If something inside agent.* is missing, re raise so we see the real error.
    if "agent" not in str(e):
        raise

    # Flat layout fallback: all modules live next to this file
    from memory_store import MemoryStore
    from presets import PRESETS, get_preset, RUNTIME_PROFILES  # type: ignore[no-redef]
    from report_generator import (  # type: ignore[no-redef]
        generate_report,
        generate_findings_report,
        generate_report_pdf,
        generate_findings_report_pdf,
    )
    from rye_metrics import (  # type: ignore[no-redef]
        build_run_diagnostics,
        rye_volatility_signature,
        detect_rye_equilibrium,
        tgrm_harmonic_index,
        estimate_breakthrough_probability,
        breakthrough_likelihood_90d,
        autonomy_safety_envelope,
        early_failure_warning_score,
        classify_run_tier,
    )
    from report_builder import build_agent_report  # type: ignore[no-redef]

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
            create_job,
            list_jobs as list_run_jobs,
            result_path,
            BASE_DIR as RUNS_BASE_DIR,
            PENDING_DIR as RUNS_PENDING_DIR,
            ACTIVE_DIR as RUNS_ACTIVE_DIR,
            FINISHED_DIR as RUNS_FINISHED_DIR,
            ERROR_DIR as RUNS_ERROR_DIR,
        )
    except Exception:
        create_job = None  # type: ignore[assignment]
        list_run_jobs = None  # type: ignore[assignment]
        result_path = None  # type: ignore[assignment]
        RUNS_BASE_DIR = None
        RUNS_PENDING_DIR = None
        RUNS_ACTIVE_DIR = None
        RUNS_FINISHED_DIR = None
        RUNS_ERROR_DIR = None


# Use absolute path for default config relative to repo root
CONFIG_PATH_DEFAULT = str(REPO_ROOT / "config" / "settings.yaml")

# Rough estimate for cycles per hour in continuous mode.
# Used historically; now only advisory metadata handed to the worker.
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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_settings(config_path: str = CONFIG_PATH_DEFAULT) -> Dict[str, Any]:
    """Load YAML settings file into a dictionary."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_directories() -> None:
    """Ensure that log directories exist."""
    logs_path = Path("logs")
    sessions_path = logs_path / "sessions"
    logs_path.mkdir(exist_ok=True)
    sessions_path.mkdir(exist_ok=True)


def get_runs_root() -> str:
    """Return the root directory used for ARA run jobs.

    Primary source is run_jobs.BASE_DIR so that UI and worker are always in sync.
    If that is not available, fall back to ARA_RUNS_DIR or <repo_root>/runs.
    """
    if isinstance(RUNS_BASE_DIR, Path):
        return str(RUNS_BASE_DIR)
    root = os.getenv("ARA_RUNS_DIR")
    if root:
        return root
    return str(REPO_ROOT / "runs")


@st.cache_resource
def init_memory_store(config_path: str = CONFIG_PATH_DEFAULT) -> MemoryStore:
    """Create a single MemoryStore instance for the Streamlit app (read only)."""
    ensure_directories()
    config = load_settings(config_path)
    memory_file = config.get("memory_file", "logs/sessions/default_memory.json")
    memory = MemoryStore(memory_file)
    return memory


def load_job_result(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a finished job result using run_jobs.result_path."""
    if result_path is None:
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
        return {"has_key": True, "display": f"Tavily key detected (...{tail})"}
    return {
        "has_key": False,
        "display": "No Tavily API key found. Web search will use stubbed results.",
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
    st.markdown(
        f"### Cycle {cycle_summary['cycle'] + 1} "
        f"(role: {role}, domain: {domain})"
    )

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("delta_R", cycle_summary.get("delta_R", 0.0))
    with col2:
        st.metric("Energy E", cycle_summary.get("energy_E", 0.0))
    with col3:
        st.metric("RYE", round(cycle_summary.get("RYE", 0.0), 3))

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
    if cycle_summary.get("citations"):
        with st.expander("Citations for this cycle"):
            for c in cycle_summary["citations"]:
                src = c.get("source", "")
                title = c.get("title", "")
                url = c.get("url", "")
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


def render_result_details(result: Dict[str, Any]) -> None:
    """Safe read-only result viewer for a finished job.

    Assumes worker writes JSON with fields such as:
    - summary or human_summary: str
    - key_findings or discoveries: list[str]
    - cycles: list[dict] with delta_r, energy, rye, etc.
    - rye_metrics or rye: dict with aggregates
    - sources or citations: list[dict] with title/url/snippet
    Adjust this to match your worker schema.
    """
    st.markdown("### Run summary")

    summary = result.get("summary") or result.get("human_summary")
    if summary:
        st.write(summary)
    else:
        st.info("No summary was provided by the engine.")

    key_findings = result.get("key_findings") or result.get("discoveries")
    if isinstance(key_findings, list) and key_findings:
        st.markdown("#### Key findings")
        for item in key_findings:
            st.markdown(f"- {item}")

    rye_metrics = result.get("rye_metrics") or result.get("rye")
    if isinstance(rye_metrics, dict):
        st.markdown("#### RYE metrics")
        cols = st.columns(3)
        avg_rye = rye_metrics.get("avg_rye")
        if avg_rye is not None:
            cols[0].metric("Average RYE", f"{avg_rye:.4f}")
        trend = rye_metrics.get("trend_slope")
        if trend is not None:
            cols[1].metric("RYE trend slope", f"{trend:.4f}")
        stability = rye_metrics.get("stability_index")
        if stability is not None:
            cols[2].metric("Stability index", f"{stability:.3f}")

    cycles = result.get("cycles")
    if isinstance(cycles, list) and cycles:
        st.markdown("#### Cycle timeline")
        cycle_numbers: List[Any] = []
        delta_r_values: List[Any] = []
        energy_values: List[Any] = []
        rye_values: List[Any] = []

        for idx, c in enumerate(cycles):
            # Cycle index
            c_num = c.get("cycle")
            if c_num is None:
                c_num = c.get("cycle_index")
            if c_num is None:
                c_num = idx + 1
            cycle_numbers.append(c_num)

            # delta_R
            d_val = c.get("delta_r")
            if d_val is None:
                d_val = c.get("delta_R")
            delta_r_values.append(d_val)

            # energy
            e_val = c.get("energy")
            if e_val is None:
                e_val = c.get("energy_E")
            energy_values.append(e_val)

            # RYE
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
            st.line_chart(chart_data)

    sources = result.get("sources") or result.get("citations")
    if isinstance(sources, list) and sources:
        st.markdown("#### Sources and citations")
        for s in sources:
            title = s.get("title", "Source")
            url = s.get("url") or s.get("link")
            snippet = s.get("snippet") or s.get("summary") or ""
            if url:
                st.markdown(f"- [{title}]({url})  \n  {snippet}")
            else:
                st.markdown(f"- {title}  \n  {snippet}")

    debug = result.get("debug") or result.get("diagnostics")
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

    # Parse timestamps if possible
    timestamps: List[datetime] = []
    for e in history:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            try:
                timestamps.append(datetime.fromisoformat(ts))
            except Exception:
                continue

    runtime_text = "Runtime not available"
    if len(timestamps) >= 2:
        start = min(timestamps)
        end = max(timestamps)
        delta = end - start
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        runtime_text = f"Approx runtime: {hours} hours {minutes} minutes (from first to last cycle)."

    # RYE stats
    rye_vals: List[float] = []
    for e in history:
        v = e.get("RYE")
        if isinstance(v, (int, float)):
            rye_vals.append(float(v))
    rye_text = "RYE statistics not available."
    if rye_vals:
        avg_rye = sum(rye_vals) / len(rye_vals)
        rye_text = (
            f"RYE statistics:\n"
            f"- Min RYE: {min(rye_vals):.3f}\n"
            f"- Max RYE: {max(rye_vals):.3f}\n"
            f"- Average RYE: {avg_rye:.3f}"
        )

    # Collect candidate findings from notes, repairs, and hypotheses
    findings: List[str] = []
    for e in history:
        for n in (e.get("notes_added") or []):
            findings.append(str(n))
        for r in e.get("repairs") or []:
            findings.append(str(r))
        for h in e.get("hypotheses") or []:
            if isinstance(h, dict):
                txt = h.get("text", "")
            else:
                txt = str(h)
            if txt:
                findings.append(txt)

    # Deduplicate while preserving order
    seen: set[str] = set()
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


# Helper used by learning speed panels and Option C report
def compute_run_hours(history: List[Dict[str, Any]]) -> Optional[float]:
    """Approximate total hours between first and last cycle timestamps."""
    timestamps: List[datetime] = []
    for e in history:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            try:
                timestamps.append(datetime.fromisoformat(ts))
            except Exception:
                continue
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
    """Call optional MSIL layer if available to compute meta skill intelligence profile.

    This is wired to the msil.analyze_run helper when present, and falls back
    to constructing an internal MetaSkillIntelligenceLayer using the
    msil._HistoryBackedMemoryStore wrapper if needed.
    """
    if not history or _msil_module is None:
        return None

    if goal is None:
        goal = str(history[-1].get("goal") or "unknown_goal")

    try:
        # Preferred simple function style (matches msil.analyze_run signature)
        analyze_run = getattr(_msil_module, "analyze_run", None)
        if callable(analyze_run):
            return analyze_run(history=history, goal=goal, config=None)

        # Class based API fallback using msil._HistoryBackedMemoryStore
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


def load_discovery_log() -> List[Dict[str, Any]]:
    """Try to load discovery log entries from standard locations."""
    candidates = [
        Path("logs/discovery_log.json"),
        Path("logs/discovery/discovery_log.json"),
        Path("logs/discovery/discoveries.json"),
    ]
    for p in candidates:
        data = _load_json_file(p)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    # If discovery module exposes a helper, use it
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


def load_snapshots() -> List[Dict[str, Any]]:
    """Load snapshot JSON files as a list of {name, timestamp, data}."""
    snapshot_dir_candidates = [
        Path("logs/snapshots"),
        Path("logs/snapshot"),
    ]
    snapshots: List[Dict[str, Any]] = []
    for base in snapshot_dir_candidates:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.glob("*.json")):
            data = _load_json_file(path)
            if not isinstance(data, dict):
                continue
            ts_val = data.get("timestamp") or data.get("timestamp_utc") or data.get("created_at")
            try:
                if isinstance(ts_val, str):
                    ts = datetime.fromisoformat(ts_val)
                else:
                    ts = None
            except Exception:
                ts = None
            snapshots.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "timestamp": ts,
                    "raw_timestamp": ts_val,
                    "data": data,
                }
            )
    # Sort by timestamp if available
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
                {
                    "cycle": cycle_idx,
                    "role": role,
                    "domain": domain,
                    "timestamp": ts,
                    "text": text,
                    "confidence": conf,
                }
            )
    return results


def extract_citations_from_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten citations across all cycles into a list with cycle info for the citation viewer."""
    results: List[Dict[str, Any]] = []
    for entry in history:
        cycle_idx = entry.get("cycle")
        role = entry.get("role", "agent")
        domain = entry.get("domain", "general")
        ts = entry.get("timestamp")
        cites = entry.get("citations") or []
        for c in cites:
            if not isinstance(c, dict):
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
    candidates = [
        Path("logs/verification_log.json"),
        Path("logs/verification/verification_log.json"),
        Path("logs/verification/results.json"),
    ]
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
    """Sanitize Graphviz node id to avoid spaces and punctuation issues."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    if not clean:
        clean = "node"
    return f"{prefix}{clean}"


def _clean_label_text(text: str, max_len: int = 60) -> str:
    """Clean label text for Graphviz: strip quotes, newlines, and compress spaces."""
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

    # Single root node for the run
    nodes.append('run [label="Run", shape=box, style=filled, fillcolor="#eeeeee"]')

    # Domains
    domains = sorted({str(e.get("domain", "general")) for e in history})
    domain_ids: Dict[str, str] = {}
    for d in domains:
        safe_d_label = _clean_label_text(f"Domain: {d}")
        node_id = _safe_gv_id("domain_", d)
        domain_ids[d] = node_id
        nodes.append(f'{node_id} [label="{safe_d_label}", shape=box]')
        edges.append(f"run -> {node_id}")

    # Roles
    roles = sorted({str(e.get("role", "agent")) for e in history})
    role_ids: Dict[str, str] = {}
    for r in roles:
        safe_r_label = _clean_label_text(f"Role: {r}")
        node_id = _safe_gv_id("role_", r)
        role_ids[r] = node_id
        nodes.append(f'{node_id} [label="{safe_r_label}", shape=ellipse]')
        edges.append(f"run -> {node_id}")

    # Hypotheses, take top 8 by confidence if available
    hyps = extract_hypotheses_from_history(history)
    if hyps:
        scored = []
        for h in hyps:
            conf = h.get("confidence")
            if isinstance(conf, (int, float)):
                score = float(conf)
            else:
                score = 0.0
            scored.append((score, h))
        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        scored = []
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
        if r_id not in domain_ids.values():
            safe_r2_label = _clean_label_text(f"Role: {r}")
            nodes.append(f'{r_id} [label="{safe_r2_label}", shape=ellipse]')
        edges.append(f"{d_id} -> {hyp_id}")
        edges.append(f"{r_id} -> {hyp_id}")

    # Discoveries, top 8 by rye_gain if present
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

    # Top cycles by RYE
    scored_cycles: List[Tuple[float, Dict[str, Any]]] = []
    for e in history:
        rye_val = e.get("RYE")
        d_r = e.get("delta_R")
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
            d_r = e.get("delta_R")
            energy_e = e.get("energy_E")
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

            # Attach one or two short notes or hypotheses as evidence
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
                    if isinstance(h, dict):
                        txt = h.get("text", "")
                    else:
                        txt = str(h)
                    txt = txt.strip()
                    if not txt:
                        continue
                    if len(txt) > 220:
                        txt = txt[:220] + "..."
                    lines.append(f"  - Hypothesis: {txt}")
                    details_added += 1
                    if details_added >= 2:
                        break

    # Pull best discoveries from discovery log
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


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
def main() -> None:
    st.title("Autonomous Research Agent")
    st.caption(
        "Reparodynamics, RYE, and TGRM powered research loop "
        "(UI only queues jobs and reads results; all cycles run in a separate worker)."
    )

    memory = init_memory_store()

    # -----------------------------
    # Sidebar settings
    # -----------------------------
    st.sidebar.header("Run settings")

    # Tavily key input (per user)
    st.sidebar.subheader("Tavily API key")
    existing_key = st.session_state.get("tavily_key", "")
    tavily_key_input = st.sidebar.text_input(
        "Enter your Tavily key",
        value=existing_key,
        type="password",
        help=(
            "Each user should paste their own Tavily API key here. "
            "If left empty, the agent will use stubbed (offline) web results."
        ),
    )
    if tavily_key_input:
        st.session_state["tavily_key"] = tavily_key_input
        os.environ["TAVILY_API_KEY"] = tavily_key_input
    else:
        st.session_state["tavily_key"] = ""
        os.environ.pop("TAVILY_API_KEY", None)

    # Preset selector (General, Longevity, Math, etc.)
    preset_keys = list(PRESETS.keys())
    preset_labels = [PRESETS[k]["label"] for k in preset_keys]

    default_preset_index = 0
    if "general" in preset_keys:
        default_preset_index = preset_keys.index("general")

    selected_label = st.sidebar.selectbox(
        "Domain preset",
        options=preset_labels,
        index=default_preset_index,
        help="Choose a domain preset. You can still edit all settings below.",
    )
    selected_key = preset_keys[preset_labels.index(selected_label)]
    preset = get_preset(selected_key)
    domain_tag = preset.get("domain", selected_key)

    # Runtime profile info from presets if configured
    st.sidebar.subheader("Runtime profile")
    default_runtime_profile = preset.get("default_runtime_profile")
    if default_runtime_profile:
        rp_cfg = RUNTIME_PROFILES.get(default_runtime_profile, {})
        rp_label = rp_cfg.get("label", default_runtime_profile)
        rp_desc = rp_cfg.get("description", "")
        rp_est_cycles = rp_cfg.get("estimated_cycles")
        st.sidebar.write(f"Preset profile: **{rp_label}**")
        if rp_est_cycles is not None:
            st.sidebar.caption(f"Estimated cycles target: {rp_est_cycles}")
        if rp_desc:
            st.sidebar.caption(rp_desc)
    else:
        st.sidebar.caption(
            "This preset has no runtime profile configured. "
            "Manual finite mode uses generic defaults."
        )

    # Friendly label per run
    st.sidebar.subheader("Run label")
    run_label = st.sidebar.text_input(
        "Run label",
        value="experiment",
        help="Human friendly label for this run request.",
    )

    # Tavily status (after handling key input)
    status = tavily_status()
    st.sidebar.subheader("Internet research")
    if status["has_key"]:
        st.sidebar.success(status["display"])
    else:
        st.sidebar.warning(status["display"])
        st.sidebar.write(
            "Paste a Tavily key above to enable real web search. "
            "Otherwise, stubbed results are used."
        )

    # Tool status (web browser and sandbox)
    st.sidebar.subheader("Tools status")
    tool_flags = detect_tools()

    if tool_flags["web"]:
        st.sidebar.success("Web browser tool is available in tools.py.")
    else:
        st.sidebar.info(
            "Web browser tool not detected in tools.py. "
            "Core engine may still use Tavily directly."
        )

    if tool_flags["sandbox"]:
        st.sidebar.success("Sandbox tool is available for safe code execution.")
    else:
        st.sidebar.info("Sandbox tool not detected in tools.py.")

    # Web browser and sandbox toggles
    use_web_tool = st.sidebar.checkbox(
        "Use web browser tool",
        value=status["has_key"],
        help=(
            "If enabled, the engine can use the web browser tool for searches. "
            "If disabled, only local notes and PDFs are used."
        ),
    )

    allow_sandbox = st.sidebar.checkbox(
        "Allow sandbox code execution",
        value=tool_flags["sandbox"],
        help=(
            "If enabled and the sandbox tool is present, the engine can run code in a bounded sandbox. "
            "If disabled, code execution tools are not used."
        ),
    )

    # Swarm toggle and size
    st.sidebar.subheader("Swarm configuration")
    enable_swarm = st.sidebar.checkbox(
        "Enable Swarm (multi role mini agents)",
        value=False,
        help=(
            "Request up to dozens of specialized agents (researchers, critics, explorers, "
            "theorists, integrators). The worker runs them sequentially for safety."
        ),
    )

    swarm_size = 1
    swarm_roles: List[Tuple[str, str]] = []
    if enable_swarm:
        swarm_size = st.sidebar.slider(
            "Total swarm agents",
            min_value=2,
            max_value=MAX_SWARM_AGENTS,
            value=min(5, MAX_SWARM_AGENTS),
            help=(
                "Total number of mini agents in the swarm. "
                "Higher values mean more total cycles and more API usage."
            ),
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
            help="If swarm is disabled, you can still request a simple researcher plus critic pair.",
        )
    else:
        st.sidebar.info("Classic Multi Agent is disabled when Swarm is enabled.")

    # Source controls (defaults from preset, but user can override)
    sc_defaults = preset.get("source_controls", {})
    use_pubmed = st.sidebar.checkbox(
        "Use PubMed (scientific literature)",
        value=bool(sc_defaults.get("pubmed", False)),
    )
    use_semantic = st.sidebar.checkbox(
        "Use Semantic Scholar ingestion",
        value=bool(sc_defaults.get("semantic", False)),
    )
    use_pdf = st.sidebar.checkbox(
        "Enable PDF ingestion (upload papers below)",
        value=bool(sc_defaults.get("pdf", True)),
    )

    uploaded_pdf = None
    if use_pdf:
        uploaded_pdf = st.sidebar.file_uploader("Upload a PDF paper", type=["pdf"])

    # Biomarker mode
    use_biomarkers = st.sidebar.checkbox(
        "Biomarker / Longevity Mode (anti aging teams)",
        value=bool(sc_defaults.get("biomarkers", False)),
    )

    # Run mode presets: finite only in this build
    run_mode = st.sidebar.radio(
        "Run mode",
        ["Manual (finite cycles)"],
        index=0,
        help=(
            "This build uses finite mode only. "
            "Each run requests a fixed number of cycles from the engine worker."
        ),
    )

    # No RYE based stop for finite only build
    stop_rye_threshold: Optional[float] = None

    # -----------------------------
    # Main area
    # -----------------------------
    st.subheader("Research goal")

    default_goal = preset.get("default_goal") or (
        "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
        "identify similar frameworks in the literature, and produce a structured comparison table."
    )

    if "goal_text" not in st.session_state:
        st.session_state["goal_text"] = default_goal

    goal = st.text_area("Enter research goal:", value=st.session_state["goal_text"], height=160)
    st.session_state["goal_text"] = goal

    # Cycles only matter in manual mode (hint to worker)
    cycles = st.number_input(
        "Number of TGRM cycles to request (manual mode)",
        min_value=1,
        max_value=200,
        value=3,
        step=1,
        help="Used when Run mode is Manual. There are no timed presets in this build.",
    )

    run_button = st.button("Queue run request")

    # ------------------------------
    # Queue job (never run cycles here)
    # ------------------------------
    if run_button:
        goal_clean = goal.strip()
        if not goal_clean:
            st.error("Please provide a research goal or question before queuing a run.")
        elif create_job is None:
            st.error(
                "Job queue backend (run_jobs.py) is not available. "
                "Make sure agent/run_jobs.py exists and is importable."
            )
        else:
            # Source controls for the engine worker
            source_controls = {
                "web": bool(use_web_tool),
                "pubmed": bool(use_pubmed),
                "semantic": bool(use_semantic),
                "pdf": bool(use_pdf and uploaded_pdf is not None),
                "biomarkers": bool(use_biomarkers),
                "sandbox": bool(allow_sandbox and tool_flags["sandbox"]),
            }

            # Optional PDF embedded as base64 (worker can decode)
            pdf_payload: Optional[Dict[str, Any]] = None
            if use_pdf and uploaded_pdf is not None:
                try:
                    pdf_bytes = uploaded_pdf.getvalue()
                    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
                    pdf_payload = {
                        "name": uploaded_pdf.name,
                        "base64": pdf_b64,
                    }
                except Exception:
                    pdf_payload = {
                        "name": uploaded_pdf.name,
                        "base64": None,
                    }

            # Runtime hints for the worker (finite only, advisory)
            runtime_hints: Dict[str, Any] = {
                "run_mode": "finite_manual",
                "manual_cycles": int(cycles),
                "max_cycles": int(cycles),
                "stop_rye_threshold": stop_rye_threshold,
                "cycles_per_hour_estimate": CYCLES_PER_HOUR_ESTIMATE,
            }

            # Swarm configuration for the worker
            if enable_swarm:
                swarm_config: Dict[str, Any] = {
                    "swarm_size": int(swarm_size),
                    "roles": [name for name, _ in swarm_roles] if swarm_roles else ["agent"],
                    "max_cycles_per_agent": int(cycles),
                    "stagger_start": False,
                    "max_agents_per_tick": 0,
                }
            else:
                swarm_config = {
                    "swarm_size": 1,
                    "roles": ["agent"],
                    "max_cycles_per_agent": int(cycles),
                    "stagger_start": False,
                    "max_agents_per_tick": 0,
                }

            # Optional longevity config stub for worker
            longevity_config: Dict[str, Any] = {}
            if str(domain_tag).lower() in {"longevity", "aging", "anti_aging"}:
                longevity_defaults = preset.get("longevity_config", {})
                if isinstance(longevity_defaults, dict):
                    longevity_config = {
                        "hallmark_targets": longevity_defaults.get("hallmark_targets", []),
                        "curriculum_profile": longevity_defaults.get("curriculum_profile"),
                    }

            # Decide mode for RunManager
            if enable_swarm and swarm_config.get("swarm_size", 1) > 1:
                mode = "swarm"
            elif multi_agent:
                mode = "two_stage"
            else:
                mode = "single"

            # Core run configuration that engine_worker can map to RunConfig
            run_config: Dict[str, Any] = {
                "goal": goal_clean,
                "domain": domain_tag,
                "mode": mode,
                "total_cycles": int(cycles),
                "max_seconds": None,
                "rye_stop_threshold": stop_rye_threshold,
                "equilibrium_stop_label": None,
                "min_cycles_before_stop": 3,
                "source_controls": source_controls,
                "runtime_hints": runtime_hints,
                "swarm": swarm_config,
                "longevity_config": longevity_config,
                "use_biomarkers": bool(use_biomarkers),
                "multi_agent_pair": bool(multi_agent),
                "notes": (run_label or "experiment").strip(),
            }

            if pdf_payload is not None:
                run_config["pdf"] = pdf_payload

            # Meta for UI and analytics (does not go into RunConfig directly)
            meta: Dict[str, Any] = {
                "run_label": (run_label or "experiment").strip(),
                "preset_key": selected_key,
                "preset_label": preset.get("label", selected_label),
                "domain": domain_tag,
                "tavily_enabled": bool(status["has_key"]),
                "ui_metadata": {
                    "requested_from": "streamlit",
                    "client_version": "v3-run-manager-finite-only",
                },
            }

            # Single source of truth: register job through run_jobs.create_job.
            # This writes the JSON into PENDING_DIR (and legacy queue) using the same BASE_DIR as the worker.
            run_id = create_job(config=run_config, meta=meta)

            st.success(f"Run request queued with run id `{run_id}`.")
            if RUNS_PENDING_DIR is not None:
                st.caption(f"Pending job written to `{RUNS_PENDING_DIR / (str(run_id) + '.json')}`")
            else:
                st.caption(
                    "Job was queued via run_jobs.create_job. "
                    "The engine worker should watch ARA_RUNS_DIR/pending for new jobs."
                )

    # ------------------------------
    # Runs and job queue (queued / finished via run_jobs.py)
    # ------------------------------
    st.markdown("---")
    st.subheader("Runs and job queue")

    # Debug view of what the UI actually sees on disk for the queue
    runs_root = get_runs_root()
    st.caption(f"DEBUG runs root: `{runs_root}`")

    def _debug_list_dir(label: str, specific: Optional[Path]) -> None:
        if isinstance(specific, Path):
            base = specific
        else:
            base = Path(runs_root) / label
        try:
            items = sorted(p.name for p in base.glob("*.json"))
        except Exception as e:
            items = [f"error: {e}"]
        st.text(f"DEBUG {label}: {items}")

    _debug_list_dir("pending", RUNS_PENDING_DIR)
    _debug_list_dir("active", RUNS_ACTIVE_DIR)
    _debug_list_dir("finished", RUNS_FINISHED_DIR)
    _debug_list_dir("error", RUNS_ERROR_DIR)

    if list_run_jobs is not None:
        try:
            finished_jobs = list_run_jobs(status="finished")
        except TypeError:
            # Fallback if signature is list_jobs() without args
            finished_jobs = list_run_jobs()  # type: ignore[call-arg]
        try:
            pending_jobs = list_run_jobs(status="queued")
        except TypeError:
            pending_jobs = []
    else:
        finished_jobs = []
        pending_jobs = []

    col_runs_left, col_runs_right = st.columns([2, 1])

    with col_runs_left:
        st.markdown("#### Finished runs")
        if not finished_jobs:
            st.info("No finished runs found yet.")
        else:
            run_ids = [_get_job_id(j) for j in finished_jobs]

            def _format_run(jid: str) -> str:
                for j in finished_jobs:
                    if _get_job_id(j) == jid:
                        return _get_job_label(j)
                return jid

            selected_run_id = st.selectbox(
                "Select a finished run",
                options=run_ids,
                format_func=_format_run,
            )
            if selected_run_id:
                selected_job_header = next(
                    (j for j in finished_jobs if _get_job_id(j) == selected_run_id),
                    None,
                )
                if selected_job_header:
                    render_job_summary(selected_job_header)
                result = load_job_result(selected_run_id)
                if result is None:
                    st.warning(
                        "Result file missing or unreadable for this run. "
                        "The engine worker may not have written it yet."
                    )
                else:
                    st.markdown("---")
                    render_result_details(result)

    with col_runs_right:
        st.markdown("#### Queued runs")

        # Show the canonical pending directory used by run_jobs
        if isinstance(RUNS_PENDING_DIR, Path):
            pending_dir = str(RUNS_PENDING_DIR)
        else:
            runs_root = get_runs_root()
            pending_dir = os.path.join(runs_root, "pending")

        st.caption(f"Queue directory: `{pending_dir}`")

        # Clear queue button (file based jobs under ARA_RUNS_DIR/pending)
        if st.button("🧹 Clear job queue", key="clear_queue_btn"):
            pattern = os.path.join(pending_dir, "*.json")
            removed = 0
            for fp in glob.glob(pattern):
                try:
                    os.remove(fp)
                    removed += 1
                except Exception:
                    # Ignore deletion errors and continue
                    pass
            st.success(f"Cleared {removed} queued job file(s) from {pending_dir}.")
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
    # History and advanced panels
    # ------------------------------
    st.markdown("---")
    st.subheader("History and advanced analysis")

    get_cycle_history = getattr(memory, "get_cycle_history", None)
    if callable(get_cycle_history):
        history = get_cycle_history()
    else:
        history = []

    if not history:
        st.write("No cycles yet.")
    else:
        # Pre compute optional MSIL profile
        msil_profile_full = compute_msil_profile(history)

        # Top level tabs
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

        # ----------------- Cycle history tab -----------------
        with tab_history:
            rows: List[Dict[str, Any]] = []
            for entry in history:
                goal_text = entry.get("goal", "") or ""
                rows.append(
                    {
                        "cycle": entry.get("cycle"),
                        "role": entry.get("role", "agent"),
                        "domain": entry.get("domain", "general"),
                        "goal": goal_text[:60] + ("..." if len(goal_text) > 60 else ""),
                        "delta_R": entry.get("delta_R"),
                        "energy_E": entry.get("energy_E"),
                        "RYE": entry.get("RYE"),
                        "timestamp": entry.get("timestamp"),
                    }
                )

            st.dataframe(rows, use_container_width=True)

            st.markdown("### Efficiency Charts")

            plot_rows = rows[-MAX_POINTS_FOR_CHARTS:]

            cycles_x = [r["cycle"] for r in plot_rows if r["cycle"] is not None]
            rye_y = [r["RYE"] for r in plot_rows]
            delta_y = [r["delta_R"] for r in plot_rows]
            energy_y = [r["energy_E"] for r in plot_rows]

            if cycles_x:
                st.line_chart({"RYE": rye_y})
                st.caption("Higher RYE means more efficient repair (delta_R per unit energy).")

                st.line_chart({"delta_R": delta_y})
                st.caption("delta_R is how much improvement each cycle produced.")

                st.line_chart({"energy_E": energy_y})
                st.caption("Energy per cycle (approximate effort cost).")

            # Advanced RYE diagnostics using rye_metrics build_run_diagnostics
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
                st.metric("Rolling RYE (10)", f"{roll_val:.3f}" if roll_val is not None else "n/a")
            with adv_cols[1]:
                st.metric("RYE trend", f"{trend_val:.3f}" if trend_val is not None else "n/a")
            with adv_cols[2]:
                st.metric("RYE slope", f"{slope_val:.4f}" if slope_val is not None else "n/a")
            with adv_cols[3]:
                st.metric("Stability index", f"{stability_val:.3f}" if stability_val is not None else "n/a")
            with adv_cols[4]:
                st.metric("Recovery momentum", f"{momentum_val:.3f}" if momentum_val is not None else "n/a")

            # Learning speed and breakthrough profile plus 10x Option C view
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

            bp_prob = None
            if isinstance(bp_short, dict):
                bp_prob = bp_short.get("probability")

            bp90_prob = None
            if isinstance(bp90, dict):
                bp90_prob = bp90.get("probability")

            try:
                tier_info = classify_run_tier(diagnostics, breakthrough_prob=bp_prob)
            except Exception:
                tier_info = None

            tier_label = None
            if isinstance(tier_info, dict):
                tier_label = tier_info.get("tier") or tier_info.get("label")

            ls_cols = st.columns(4)
            with ls_cols[0]:
                st.metric(
                    "Approx hours run",
                    f"{hours_run:.2f}" if isinstance(hours_run, (int, float)) else "n/a",
                )
            with ls_cols[1]:
                if isinstance(bp_prob, (int, float)):
                    st.metric("Breakthrough signal (near term, 0–1)", f"{bp_prob:.3f}")
                else:
                    st.metric("Breakthrough signal (near term, 0–1)", "n/a")
            with ls_cols[2]:
                if isinstance(bp90_prob, (int, float)):
                    st.metric("Breakthrough signal 90d (0–1)", f"{bp90_prob:.3f}")
                else:
                    st.metric("Breakthrough signal 90d (0–1)", "n/a")
            with ls_cols[3]:
                st.metric("Run tier", tier_label or "n/a")

            st.caption(
                "Breakthrough signals are heuristic scores on a 0 to 1 scale derived from RYE and stability trends, "
                "not calibrated real world probabilities."
            )

            # Optional MSIL meta intelligence view
            st.markdown("### Meta skill intelligence (MSIL)")

            if msil_profile_full:
                msil_score = msil_profile_full.get("msil_score")
                skills = msil_profile_full.get("skills") or msil_profile_full.get("dimensions") or {}
                domains_profile = (
                    msil_profile_full.get("domains")
                    or msil_profile_full.get("domain_profiles")
                    or []
                )
                dynamics = msil_profile_full.get("dynamics") or {}

                msil_cols = st.columns(3)
                with msil_cols[0]:
                    if isinstance(msil_score, (int, float)):
                        st.metric("MSIL score", f"{msil_score:.3f}")
                    else:
                        st.metric("MSIL score", "n/a")
                with msil_cols[1]:
                    st.metric("Skill dimensions", len(skills) if isinstance(skills, dict) else 0)
                with msil_cols[2]:
                    dom_count = 0
                    if isinstance(domains_profile, list):
                        dom_count = len(domains_profile)
                    elif isinstance(domains_profile, dict):
                        dom_count = len(domains_profile)
                    st.metric("Domain profiles", dom_count)

                with st.expander("Skill breakdown"):
                    st.json(skills)
                with st.expander("Domain intelligence profile"):
                    st.json(domains_profile)
                if dynamics:
                    with st.expander("Learning and stability dynamics"):
                        st.json(dynamics)
            else:
                st.info(
                    "MSIL module not detected or no MSIL profile available. "
                    "This panel stays optional and non blocking."
                )

            # New Option C style 10x learning dashboard
            st.markdown("#### 10x learning dashboard (Option C signals)")

            # Volatility
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

            # Equilibrium detection
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

            # Harmonic index
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
            eq_state_text = "unknown"
            if isinstance(eq_flag, bool):
                eq_state_text = "yes" if eq_flag else "no"

            oc_cols = st.columns(3)
            with oc_cols[0]:
                st.metric("Equilibrium detected", eq_state_text)
            with oc_cols[1]:
                if isinstance(vol_score, (int, float)):
                    st.metric("Volatility score", f"{vol_score:.3f}")
                else:
                    st.metric("Volatility score", "n/a")
            with oc_cols[2]:
                if isinstance(harmonic_val, (int, float)):
                    st.metric("TGRM harmonic index", f"{harmonic_val:.3f}")
                else:
                    st.metric("TGRM harmonic index", "n/a")

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
                st.code(json.dumps(raw_signals, indent=2), language="json")

            with st.expander("Raw history JSON"):
                st.code(json.dumps(history, indent=2), language="json")

            with st.expander("Raw diagnostics JSON"):
                st.code(json.dumps(diagnostics, indent=2), language="json")

        # ----------------- Citations tab -----------------
        with tab_citations:
            st.markdown("### Source citation viewer")

            citations = extract_citations_from_history(history)
            if not citations:
                st.info("No citations recorded yet in cycle history.")
            else:
                total_cites = len(citations)
                unique_sources = sorted({c["source"] for c in citations if c.get("source")})
                domains_c = sorted({c["domain"] for c in citations})
                roles_c = sorted({c["role"] for c in citations})

                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    st.metric("Total citation hits", total_cites)
                with col_c2:
                    st.metric("Unique sources", len(unique_sources))
                with col_c3:
                    st.metric("Domains with citations", len(domains_c))

                citations_domain_filter = st.multiselect(
                    "Filter by domain",
                    options=domains_c,
                    default=domains_c,
                    key="citations_domain_filter",
                )
                citations_role_filter = st.multiselect(
                    "Filter by role",
                    options=roles_c,
                    default=roles_c,
                    key="citations_role_filter",
                )
                citations_source_filter = st.multiselect(
                    "Filter by source",
                    options=unique_sources,
                    default=unique_sources,
                    key="citations_source_filter",
                )

                filtered_cites: List[Dict[str, Any]] = []
                for c in citations:
                    d = c["domain"]
                    r = c["role"]
                    s = c["source"]
                    if citations_domain_filter and d not in citations_domain_filter:
                        continue
                    if citations_role_filter and r not in citations_role_filter:
                        continue
                    if citations_source_filter and s not in citations_source_filter:
                        continue
                    filtered_cites.append(c)

                st.write(f"Showing {len(filtered_cites)} citations after filters.")

                view_rows_c: List[Dict[str, Any]] = []
                for c in filtered_cites:
                    title = c["title"] or ""
                    snippet = c["snippet"] or ""
                    if len(title) > 80:
                        title = title[:80] + "..."
                    if len(snippet) > 120:
                        snippet = snippet[:120] + "..."
                    view_rows_c.append(
                        {
                            "cycle": c["cycle"],
                            "role": c["role"],
                            "domain": c["domain"],
                            "source": c["source"],
                            "title": title,
                            "snippet": snippet,
                            "url": c["url"],
                            "timestamp": c["timestamp"],
                        }
                    )

                st.dataframe(view_rows_c, use_container_width=True)

                with st.expander("Raw citations JSON"):
                    st.code(json.dumps(citations, indent=2), language="json")

        # ----------------- Discovery log tab -----------------
        with tab_discovery:
            st.markdown("### Discovery log")
            discoveries = load_discovery_log()
            if not discoveries:
                st.info(
                    "No discovery log found yet. The worker will populate it once discovery logging is enabled."
                )
            else:
                # High level stats for discoveries
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
                    if best_gain is not None:
                        st.metric("Best RYE gain", f"{best_gain:.3f}")
                    else:
                        st.metric("Best RYE gain", "n/a")

                if best_label is not None and best_gain is not None:
                    st.caption(f"Top discovery candidate: {str(best_label)[:80]}")

                # Simple filters
                domains_available = sorted(
                    {str(d.get("domain", "general")) for d in discoveries}
                )
                discovery_domain_filter = st.multiselect(
                    "Filter by domain",
                    options=domains_available,
                    default=domains_available,
                    key="discovery_domain_filter",
                )
                min_gain = st.number_input(
                    "Minimum RYE gain to show",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.01,
                )

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
                    st.write(f"Showing {len(filtered)} discoveries after filters.")
                    st.dataframe(filtered, use_container_width=True)
                else:
                    st.info("No discoveries matched the current filters.")

                with st.expander("Raw discovery log JSON"):
                    st.code(json.dumps(discoveries, indent=2), language="json")

        # ----------------- Snapshots and equilibrium tab -----------------
        with tab_snapshots:
            st.markdown("### Snapshots and equilibrium")

            snapshots = load_snapshots()
            if not snapshots:
                st.info(
                    "No snapshots found yet. The worker will create them when snapshot generation is enabled."
                )
            else:
                labels = []
                for s in snapshots:
                    ts = s["timestamp"]
                    if isinstance(ts, datetime):
                        label = f"{s['name']} - {ts.isoformat(timespec='seconds')}"
                    else:
                        label = s["name"]
                    labels.append(label)

                st.write(f"Total snapshots: {len(snapshots)}")

                # Timeline view of equilibrium RYE across snapshots
                rye_series = []
                ts_series = []
                for s in snapshots:
                    eq = equilibrium_from_snapshot(s["data"])
                    if eq["rye_avg"] is not None:
                        rye_series.append(eq["rye_avg"])
                        ts = s["timestamp"]
                        if isinstance(ts, datetime):
                            ts_series.append(ts.isoformat(timespec="seconds"))
                        else:
                            ts_series.append(s["name"])
                if rye_series:
                    st.markdown("#### Snapshot RYE timeline")
                    timeline_data = {"RYE avg": rye_series}
                    st.line_chart(timeline_data)
                    st.caption("Approximate evolution of equilibrium RYE across saved snapshots.")

                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    idx1 = st.selectbox(
                        "Select first snapshot",
                        options=list(range(len(snapshots))),
                        format_func=lambda i: labels[i],
                    )
                with col_sel2:
                    idx2 = st.selectbox(
                        "Select second snapshot to compare",
                        options=list(range(len(snapshots))),
                        index=min(len(snapshots) - 1, max(1, len(snapshots) - 1)),
                        format_func=lambda i: labels[i],
                    )

                s1 = snapshots[idx1]
                s2 = snapshots[idx2]

                st.markdown("#### Snapshot 1 equilibrium view")
                eq1 = equilibrium_from_snapshot(s1["data"])
                col_eq1 = st.columns(4)
                col_eq1[0].metric(
                    "RYE avg",
                    f"{eq1['rye_avg']:.3f}" if eq1["rye_avg"] is not None else "n/a",
                )
                col_eq1[1].metric(
                    "Stability idx",
                    f"{eq1['stability_index']:.3f}" if eq1["stability_index"] is not None else "n/a",
                )
                col_eq1[2].metric(
                    "Coherence plateau",
                    f"{eq1['coherence_plateau']:.3f}" if eq1["coherence_plateau"] is not None else "n/a",
                )
                col_eq1[3].metric(
                    "Equilibrium fraction",
                    f"{eq1['equilibrium_fraction']:.3f}" if eq1["equilibrium_fraction"] is not None else "n/a",
                )

                st.markdown("#### Snapshot 2 equilibrium view")
                eq2 = equilibrium_from_snapshot(s2["data"])
                col_eq2 = st.columns(4)
                col_eq2[0].metric(
                    "RYE avg",
                    f"{eq2['rye_avg']:.3f}" if eq2["rye_avg"] is not None else "n/a",
                )
                col_eq2[1].metric(
                    "Stability idx",
                    f"{eq2['stability_index']:.3f}" if eq2["stability_index"] is not None else "n/a",
                )
                col_eq2[2].metric(
                    "Coherence plateau",
                    f"{eq2['coherence_plateau']:.3f}" if eq2["coherence_plateau"] is not None else "n/a",
                )
                col_eq2[3].metric(
                    "Equilibrium fraction",
                    f"{eq2['equilibrium_fraction']:.3f}" if eq2["equilibrium_fraction"] is not None else "n/a",
                )

                st.markdown("#### Equilibrium delta (snapshot2 - snapshot1)")

                def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
                    if a is None or b is None:
                        return None
                    return b - a

                d_rye = _delta(eq1["rye_avg"], eq2["rye_avg"])
                d_stab = _delta(eq1["stability_index"], eq2["stability_index"])
                d_plateau = _delta(eq1["coherence_plateau"], eq2["coherence_plateau"])
                d_eqfrac = _delta(eq1["equilibrium_fraction"], eq2["equilibrium_fraction"])

                col_de = st.columns(4)
                col_de[0].metric(
                    "Delta RYE avg", f"{d_rye:+.3f}" if d_rye is not None else "n/a"
                )
                col_de[1].metric(
                    "Delta stability", f"{d_stab:+.3f}" if d_stab is not None else "n/a"
                )
                col_de[2].metric(
                    "Delta plateau",
                    f"{d_plateau:+.3f}" if d_plateau is not None else "n/a",
                )
                col_de[3].metric(
                    "Delta equilibrium",
                    f"{d_eqfrac:+.3f}" if d_eqfrac is not None else "n/a",
                )

                with st.expander("Raw snapshot 1 JSON"):
                    st.code(json.dumps(s1["data"], indent=2), language="json")
                with st.expander("Raw snapshot 2 JSON"):
                    st.code(json.dumps(s2["data"], indent=2), language="json")

        # ----------------- Hypothesis manager tab -----------------
        with tab_hypo:
            st.markdown("### Hypothesis manager")

            all_hyps = extract_hypotheses_from_history(history)
            if not all_hyps:
                st.info("No hypotheses recorded yet in cycle history.")
            else:
                # Filters
                domains = sorted({str(h["domain"]) for h in all_hyps})
                roles = sorted({str(h["role"]) for h in all_hyps})
                hypo_domain_filter = st.multiselect(
                    "Filter by domain",
                    options=domains,
                    default=domains,
                    key="hypo_domain_filter",
                )
                hypo_role_filter = st.multiselect(
                    "Filter by role",
                    options=roles,
                    default=roles,
                    key="hypo_role_filter",
                )

                min_conf = st.number_input(
                    "Minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                )

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

                st.write(
                    f"Total hypotheses: {len(all_hyps)}, after filters: {len(filtered_h)}"
                )

                # Sort by confidence if present
                def _score(h: Dict[str, Any]) -> float:
                    c = h.get("confidence")
                    try:
                        return float(c) if c is not None else 0.0
                    except Exception:
                        return 0.0

                filtered_h.sort(key=_score, reverse=True)
                view_rows = []
                for h in filtered_h:
                    view_rows.append(
                        {
                            "cycle": h["cycle"],
                            "role": h["role"],
                            "domain": h["domain"],
                            "confidence": h["confidence"],
                            "text": h["text"][:120] + ("..." if len(h["text"]) > 120 else ""),
                            "timestamp": h["timestamp"],
                        }
                    )
                st.dataframe(view_rows, use_container_width=True)

                hypo_md = ["# Hypotheses\n"]
                for h in filtered_h:
                    conf_txt = ""
                    if isinstance(h.get("confidence"), (int, float)):
                        conf_txt = f" (confidence ~ {h['confidence']:.2f})"
                    hypo_md.append(
                        f"- [{h['domain']}/{h['role']} cycle {h['cycle']}] {h['text']}{conf_txt}"
                    )
                hypo_md_text = "\n".join(hypo_md)
                st.download_button(
                    "Download hypotheses as Markdown",
                    data=hypo_md_text,
                    file_name="hypotheses_export.md",
                    mime="text/markdown",
                )

        # ----------------- Memory pruning tab -----------------
        with tab_memory:
            st.markdown("### Memory pruning and compaction")

            total_cycles = len(history)
            st.metric("Total cycles in history", total_cycles)

            # Check for optional pruning hooks on MemoryStore or pruner module
            has_prune_method = hasattr(memory, "prune_low_value_notes") or hasattr(
                memory, "prune_history"
            )
            has_pruner_module = _pruner_module is not None

            if not has_prune_method and not has_pruner_module:
                st.info(
                    "No pruning hooks detected on MemoryStore or memory_pruner module. "
                    "You can still manually clear history by deleting the memory file on disk."
                )
            else:
                threshold = st.number_input(
                    "Approximate minimum RYE gain to keep entries (used if supported)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.01,
                    step=0.005,
                )
                max_keep = st.number_input(
                    "Maximum entries to keep in detailed history (0 means no cap)",
                    min_value=0,
                    max_value=100000,
                    value=5000,
                    step=500,
                )

                if st.button("Run pruning now (experimental)"):
                    pruned_count = 0
                    error_msg = None
                    try:
                        if hasattr(memory, "prune_low_value_notes"):
                            func = getattr(memory, "prune_low_value_notes")
                            pruned_count = int(
                                func(threshold=threshold, max_keep=max_keep)  # type: ignore[arg-type]
                            )
                        elif hasattr(memory, "prune_history"):
                            func = getattr(memory, "prune_history")
                            pruned_count = int(
                                func(threshold=threshold, max_keep=max_keep)  # type: ignore[arg-type]
                            )
                        elif has_pruner_module and hasattr(
                            _pruner_module, "run_memory_pruning"
                        ):
                            func = getattr(_pruner_module, "run_memory_pruning")
                            pruned_count = int(
                                func(
                                    memory_store=memory,
                                    threshold=threshold,
                                    max_keep=max_keep,
                                )  # type: ignore[arg-type]
                            )
                    except Exception as e:
                        error_msg = str(e)

                    if error_msg:
                        st.error(f"Pruning call raised an error: {error_msg}")
                    else:
                        st.success(
                            f"Pruning completed. Approximate entries removed: {pruned_count}"
                        )
                        st.info(
                            "Reload the page to reflect updated history and diagnostics."
                        )

        # ----------------- Verification and cures tab -----------------
        with tab_verify:
            st.markdown("### Verification and cure oriented findings")

            verifications = load_verification_log()
            if not verifications:
                st.info(
                    "No verification log found yet. Verification engine has not written results or file is empty."
                )
            else:
                # Summaries
                success_flags = []
                rye_deltas = []
                for v in verifications:
                    ok = v.get("verified") or v.get("success")
                    success_flags.append(bool(ok))
                    delta = (
                        v.get("rye_gain")
                        or v.get("delta_rye")
                        or v.get("delta_RYE")
                    )
                    try:
                        rye_deltas.append(float(delta))
                    except Exception:
                        continue

                total = len(verifications)
                successful = sum(1 for x in success_flags if x)
                st.metric("Total verifications", total)
                st.metric("Successful verifications", successful)
                st.metric(
                    "Success rate",
                    f"{(successful / total * 100.0):.1f}%" if total > 0 else "n/a",
                )
                if rye_deltas:
                    st.metric(
                        "Average RYE change when verified",
                        f"{(sum(rye_deltas) / len(rye_deltas)):.3f}",
                    )
                else:
                    st.metric(
                        "Average RYE change when verified",
                        "n/a",
                    )

                # Table
                view_rows_v = []
                for v in verifications:
                    label = v.get("label") or v.get("id") or v.get("target") or "item"
                    hyp = v.get("hypothesis") or v.get("text")
                    ok = bool(v.get("verified") or v.get("success"))
                    d_rye = (
                        v.get("rye_gain")
                        or v.get("delta_rye")
                        or v.get("delta_RYE")
                    )
                    domain = v.get("domain", "general")
                    view_rows_v.append(
                        {
                            "label": label,
                            "domain": domain,
                            "verified": ok,
                            "delta_RYE": d_rye,
                            "hypothesis": (hyp or "")[:120]
                            + ("..." if hyp and len(hyp) > 120 else ""),
                        }
                    )
                st.dataframe(view_rows_v, use_container_width=True)

                with st.expander("Raw verification log JSON"):
                    st.code(json.dumps(verifications, indent=2), language="json")

        # ----------------- Multi agent insight graph tab -----------------
        with tab_graph:
            st.markdown("### Multi agent insight graph")

            discoveries_for_graph = load_discovery_log()
            if not history:
                st.info("No history yet to build a graph.")
            else:
                dot = build_insight_graph(
                    history=history, discoveries=discoveries_for_graph
                )
                st.graphviz_chart(dot)

    # ------------------------------
    # Run diagnostics (state from MemoryStore)
    # ------------------------------
    st.markdown("---")
    st.subheader("Run diagnostics")

    col_state, col_watchdog = st.columns(2)

    with col_state:
        st.markdown("**Last saved run state (MemoryStore)**")
        load_run_state = getattr(memory, "load_run_state", None)
        if callable(load_run_state):
            state = load_run_state()
        else:
            state = {}
        if not state:
            st.write("No saved run state yet.")
        else:
            st.json(state)
            if callable(getattr(memory, "clear_run_state", None)):
                if st.button("Clear saved run state", key="clear_run_state_btn"):
                    memory.clear_run_state()  # type: ignore[call-arg]
                    st.success(
                        "Saved run state cleared. It will be rebuilt on the next run "
                        "by your engine worker."
                    )
            else:
                st.info("MemoryStore.clear_run_state not available in this build.")

    with col_watchdog:
        st.markdown("**Watchdog heartbeat (MemoryStore)**")
        get_watchdog_info = getattr(memory, "get_watchdog_info", None)
        if callable(get_watchdog_info):
            info = get_watchdog_info()
            last_beat = info.get("last_beat")
            count = info.get("count", 0)
            seconds_since = info.get("seconds_since_last")

            st.write(f"Last beat: {last_beat if last_beat else 'None recorded'}")
            st.write(f"Heartbeat count: {count}")
            if isinstance(seconds_since, (int, float)):
                st.write(f"Seconds since last beat: {seconds_since:.1f}")
            else:
                st.write("Seconds since last beat: not available")
        else:
            st.write("MemoryStore.get_watchdog_info not available in this build.")

    # ------------------------------
    # Report generation
    # ------------------------------
    st.markdown("---")
    st.subheader("Generate report")

    if callable(getattr(memory, "get_cycle_history", None)):
        history_for_reports = memory.get_cycle_history()
    else:
        history_for_reports = []

    hours_run_for_reports = (
        compute_run_hours(history_for_reports) if history_for_reports else None
    )
    msil_profile_for_reports = (
        compute_msil_profile(history_for_reports) if history_for_reports else None
    )

    col_rep1, col_rep2, col_rep3 = st.columns(3)

    with col_rep1:
        if st.button("Full history report"):
            report_md = generate_report(memory_store=memory, goal=None)
            st.markdown(report_md)
            st.download_button(
                "Download full report (Markdown)",
                data=report_md,
                file_name="autonomous_research_report.md",
                mime="text/markdown",
            )
            # Optional PDF
            try:
                pdf_path = generate_report_pdf(
                    memory_store=memory,
                    goal=None,
                    output_path="autonomous_research_report.pdf",
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download full report (PDF)",
                        data=f,
                        file_name="autonomous_research_report.pdf",
                        mime="application/pdf",
                    )
            except RuntimeError as e:
                st.info(str(e))
            except Exception:
                st.info(
                    "PDF generation failed unexpectedly. Check server logs for details."
                )

        if st.button("Full Option C learning speed report", key="option_c_report_btn"):
            option_md = build_agent_report(
                memory_store=memory,
                goal=None,
                domain=None,
                hours_run_so_far=hours_run_for_reports,
                swarm_stats=None,
                intelligence_profile=msil_profile_for_reports,
                biomarker_snapshot=None,
            )
            st.markdown(option_md)
            st.download_button(
                "Download Option C report (Markdown)",
                data=option_md,
                file_name="autonomous_option_c_report.md",
                mime="text/markdown",
            )

    with col_rep2:
        if st.button("Outcome focused summary"):
            outcome_md = build_outcome_summary(
                history_for_reports if history_for_reports else []
            )
            st.markdown(outcome_md)
            st.download_button(
                "Download outcome summary",
                data=outcome_md,
                file_name="autonomous_outcome_summary.md",
                mime="text/markdown",
            )

    with col_rep3:
        if st.button("Findings report (cures, treatments)"):
            findings_md = generate_findings_report(memory_store=memory, goal=None)
            st.markdown(findings_md)
            st.download_button(
                "Download findings report (Markdown)",
                data=findings_md,
                file_name="autonomous_findings_report.md",
                mime="text/markdown",
            )
            # Optional PDF
            try:
                pdf_path = generate_findings_report_pdf(
                    memory_store=memory,
                    goal=None,
                    output_path="autonomous_agent_findings_report.pdf",
                )
            except RuntimeError as e:
                pdf_path = None
                st.info(str(e))
            except Exception:
                pdf_path = None
                st.info(
                    "PDF generation failed unexpectedly. Check server logs for details."
                )
            if pdf_path is not None:
                try:
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Download findings report (PDF)",
                            data=f,
                            file_name="autonomous_agent_findings_report.pdf",
                            mime="application/pdf",
                        )
                except Exception:
                    st.info("PDF file not available after generation attempt.")

        if st.button("Breakthrough snapshot report"):
            discoveries = load_discovery_log()
            breakthrough_md = build_breakthrough_report(
                history_for_reports if history_for_reports else [],
                discoveries,
            )
            st.markdown(breakthrough_md)
            st.download_button(
                "Download breakthrough snapshot",
                data=breakthrough_md,
                file_name="autonomous_breakthrough_snapshot.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
