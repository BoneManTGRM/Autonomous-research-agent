"""Enhanced Streamlit interface for the Autonomous Research Agent.

Features:
- Continuous Mode with duration presets (1h, 8h, 24h, 90 days, Forever)
- Researcher + Critic multi-agent mode
- Swarm mode with up to dozens of specialized mini agents
- Domain presets (General, Longevity, Math)
- PubMed / Semantic Scholar ingestion controls
- Biomarker analysis toggle (for anti-aging teams)
- Hypothesis generation viewer
- PDF ingestion for real scientific papers
- RYE, delta_R, and Energy charts
- Real Tavily search support detection
- Source citation viewer
- Report generation from full cycle history

Reparodynamics:
    The UI is a front panel on a reparodynamic system:
    - Each click runs the TGRM loop (Test, Detect, Repair, Verify).
    - Each cycle computes RYE = delta_R / E and is logged.

Time:
    All continuous run presets (1h, 8h, 24h, 90 days) map to real
    wall-clock minutes via the CoreAgent's max_minutes budget.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import streamlit as st
import yaml

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore
from agent.presets import PRESETS, get_preset  # domain presets
from agent.report_generator import generate_report  # report builder

CONFIG_PATH_DEFAULT = "config/settings.yaml"

# Rough estimate for how many cycles you expect per hour in continuous mode.
# This is now just a safety cap; real time is controlled by max_minutes.
CYCLES_PER_HOUR_ESTIMATE = 120

# Swarm roles: base archetypes for mini agents
SWARM_ROLES: List[Tuple[str, str]] = [
    ("researcher", "Deep literature and web researcher"),
    ("critic", "Methodology critic and refiner"),
    ("explorer", "Out-of-distribution explorer (new angles, analogies)"),
    ("theorist", "Model builder and unifier"),
    ("integrator", "Synthesizer that integrates and summarizes"),
]

# Safe upper bound for swarm size on typical Render or Streamlit setups.
# All swarm agents are still run sequentially in a single process.
MAX_SWARM_AGENTS: int = 32

# Limit points in charts so the frontend does not hit RangeError on very long runs.
MAX_POINTS_FOR_CHARTS: int = 1000

# Where UI and worker coordinate long runs
CONTROL_STATE_PATH = Path("logs/control_state.json")


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


def load_control_state() -> Dict[str, Any]:
    """Load the shared control state used by the background worker."""
    ensure_directories()
    if not CONTROL_STATE_PATH.exists():
        return {}
    try:
        with CONTROL_STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_control_state(state: Dict[str, Any]) -> None:
    """Persist the shared control state for the background worker."""
    ensure_directories()
    try:
        with CONTROL_STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        # Control state failure should not crash the UI
        return


@st.cache_resource
def init_agent(config_path: str = CONFIG_PATH_DEFAULT) -> Tuple[CoreAgent, MemoryStore]:
    """Create a single CoreAgent instance for the Streamlit app."""
    ensure_directories()
    config = load_settings(config_path)
    memory_file = config.get("memory_file", "logs/sessions/default_memory.json")
    memory = MemoryStore(memory_file)
    agent = CoreAgent(memory_store=memory, config=config)
    return agent, memory


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
    return {"has_key": False, "display": "No Tavily API key found. Web search will use stubbed results."}


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
        for n in e.get("notes_added", []) or []:
            findings.append(str(n))
        for r in e.get("repairs", []) or []:
            findings.append(str(r))
        for h in e.get("hypotheses", []) or []:
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
            # Keep bullets short
            if len(f) > 400:
                f = f[:400] + "..."
            lines.append(f"- {f}")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
def main() -> None:
    st.title("Autonomous Research Agent")
    st.caption("Reparodynamics, RYE, and TGRM powered research loop (with swarm mode and background worker)")

    agent, memory = init_agent()

    # -----------------------------
    # Sidebar - settings
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

    default_preset_index = 0  # default to first key
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

    # Tavily status (after handling key input)
    status = tavily_status()
    st.sidebar.subheader("Internet research")
    if status["has_key"]:
        st.sidebar.success(status["display"])
    else:
        st.sidebar.warning(status["display"])
        st.sidebar.write("Paste a Tavily key above to enable real web search. Otherwise, stubbed results are used.")

    # Swarm toggle and size
    st.sidebar.subheader("Swarm configuration")
    enable_swarm = st.sidebar.checkbox(
        "Enable Swarm (multi-role mini agents)",
        value=False,
        help=(
            "Run up to dozens of specialized agents (researchers, critics, explorers, "
            "theorists, integrators). All agents run sequentially in one process for safety."
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
                "They share memory but take on specialized roles. "
                "Higher values mean more total cycles and more API usage."
            ),
        )
        swarm_roles = build_swarm_roles(True, swarm_size)
        st.sidebar.write("Active swarm agents:")
        for name, desc in swarm_roles:
            st.sidebar.write(f"- **{name}**: {desc}")

    # Multi agent toggle (classic researcher plus critic)
    # Disabled when swarm is on, because swarm already includes critic logic.
    multi_agent = False
    if not enable_swarm:
        multi_agent = st.sidebar.checkbox(
            "Enable classic Multi Agent (Researcher + Critic)",
            value=False,
            help="If swarm is disabled, you can still run a simple researcher plus critic pair.",
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
        "Biomarker / Longevity Mode (anti-aging teams)",
        value=bool(sc_defaults.get("biomarkers", False)),
    )

    # Run mode presets
    run_mode = st.sidebar.radio(
        "Run mode",
        [
            "Manual (finite cycles)",
            "1 hour (real clock)",
            "8 hours (real clock)",
            "24 hours (real clock)",
            "90 days (real clock)",
            "Forever (until stopped)",
        ],
        index=0,
        help="Timed modes respect real wall clock minutes via the agent time budget.",
    )

    # Optional RYE stop for continuous modes
    stop_rye_threshold: Optional[float] = None
    if run_mode != "Manual (finite cycles)":
        stop_rye_threshold = st.sidebar.number_input(
            "Optional stop when RYE falls below (0 = ignore)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )
        if stop_rye_threshold <= 0:
            stop_rye_threshold = None

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

    # Cycles only matter in manual mode
    cycles = st.number_input(
        "Number of TGRM cycles to run (manual mode, UI only)",
        min_value=1,
        max_value=200,
        value=3,
        step=1,
        help="Used when Run mode is Manual. Timed modes rely on the background worker.",
    )

    run_button = st.button("Run agent (manual or configure worker)")

    # ------------------------------
    # Run cycles or configure worker
    # ------------------------------
    if run_button:
        st.write(f"Running or configuring agent with preset: {preset.get('label', selected_label)} (domain: {domain_tag})")
        history = memory.get_cycle_history()
        next_index = len(history)
        results: List[Dict[str, Any]] = []

        # Source controls for TGRM loop
        source_controls = {
            "web": True,
            "pubmed": bool(use_pubmed),
            "semantic": bool(use_semantic),
            "pdf": bool(use_pdf and uploaded_pdf is not None),
            "biomarkers": bool(use_biomarkers),
        }

        # PDF bytes (if provided)
        pdf_bytes: Optional[bytes] = None
        if use_pdf and uploaded_pdf is not None:
            try:
                pdf_bytes = uploaded_pdf.getvalue()
            except Exception:
                pdf_bytes = None

        # Manual finite mode stays inside Streamlit for quick tests.
        if run_mode == "Manual (finite cycles)":
            # Swarm manual mode
            if enable_swarm and swarm_roles:
                total_cycles = len(swarm_roles) * int(cycles)
                st.info(
                    f"Manual Swarm mode: {len(swarm_roles)} agents x {int(cycles)} cycles "
                    f"(total {total_cycles} mini cycles)."
                )
                for i in range(int(cycles)):
                    base_index = next_index + i * len(swarm_roles)
                    for j, (role_name, _) in enumerate(swarm_roles):
                        ci = base_index + j
                        role_goal = role_specific_goal(goal, role_name)
                        out = agent.run_cycle(
                            goal=role_goal,
                            cycle_index=ci,
                            role=role_name,
                            source_controls=source_controls,
                            pdf_bytes=pdf_bytes,
                            biomarker_snapshot=None,
                            domain=domain_tag,
                        )
                        results.append(out["summary"])
            else:
                # Classic single or researcher plus critic manual mode
                if not multi_agent:
                    for i in range(int(cycles)):
                        ci = next_index + i
                        out = agent.run_cycle(
                            goal=goal,
                            cycle_index=ci,
                            role="agent",
                            source_controls=source_controls,
                            pdf_bytes=pdf_bytes,
                            biomarker_snapshot=None,
                            domain=domain_tag,
                        )
                        results.append(out["summary"])
                else:
                    for i in range(int(cycles)):
                        base = next_index + 2 * i

                        # Researcher
                        r = agent.run_cycle(
                            goal=goal,
                            cycle_index=base,
                            role="researcher",
                            source_controls=source_controls,
                            pdf_bytes=pdf_bytes,
                            biomarker_snapshot=None,
                            domain=domain_tag,
                        )
                        results.append(r["summary"])

                        # Critic
                        critic_goal = f"Critically review and refine notes for: {goal}"
                        c = agent.run_cycle(
                            goal=critic_goal,
                            cycle_index=base + 1,
                            role="critic",
                            source_controls=source_controls,
                            pdf_bytes=None,
                            biomarker_snapshot=None,
                            domain=domain_tag,
                        )
                        results.append(c["summary"])

            # Show cycle summaries for manual runs
            st.subheader("Cycle Summaries (manual)")
            for cs in results:
                render_cycle_summary(cs)

        else:
            # Continuous modes no longer run inside Streamlit.
            # Instead, we configure control_state for the background worker.
            max_minutes: Optional[float] = None
            forever_flag: bool = False
            runtime_profile: Optional[str] = None

            if run_mode == "1 hour (real clock)":
                max_minutes = 60.0
                runtime_profile = "1_hour"
            elif run_mode == "8 hours (real clock)":
                max_minutes = 8 * 60.0
                runtime_profile = "8_hours"
            elif run_mode == "24 hours (real clock)":
                max_minutes = 24 * 60.0
                runtime_profile = "24_hours"
            elif run_mode == "90 days (real clock)":
                max_minutes = 90.0 * 24.0 * 60.0
                runtime_profile = "90_days"
            elif run_mode == "Forever (until stopped)":
                max_minutes = None
                forever_flag = True
                runtime_profile = "forever"

            # Worker will use its own safety caps on cycles.
            control_state = load_control_state()
            new_state: Dict[str, Any] = dict(control_state)

            # Base fields
            new_state.update(
                {
                    "status": "running",
                    "mode": "swarm" if enable_swarm else ("multi" if multi_agent else "single"),
                    "run_profile": runtime_profile,
                    "goal": goal,
                    "domain": domain_tag,
                    "max_minutes": max_minutes,
                    "forever": forever_flag,
                    "stop_rye": stop_rye_threshold,
                    "source_controls": source_controls,
                    "use_biomarkers": bool(use_biomarkers),
                    "timestamp_utc": datetime.utcnow().isoformat(),
                }
            )

            # Swarm configuration for the worker
            if enable_swarm and swarm_roles:
                new_state["swarm"] = {
                    "enabled": True,
                    "roles": [name for name, _ in swarm_roles],
                    "size": len(swarm_roles),
                }
            else:
                new_state["swarm"] = {"enabled": False, "roles": [], "size": 0}

            # Classic multi agent flag
            new_state["multi_agent_pair"] = bool(multi_agent)

            # Optional attached pdf flag (worker can decide whether to re ingests files)
            new_state["has_pdf"] = bool(use_pdf and uploaded_pdf is not None)

            save_control_state(new_state)

            if max_minutes is not None:
                st.success(
                    f"Configured background worker for continuous mode {run_mode} "
                    f"with time budget ~ {max_minutes:.1f} minutes. "
                    "The worker will now run cycles and update history while this UI just monitors."
                )
            else:
                st.success(
                    "Configured background worker for continuous mode Forever. "
                    "The worker will run until stopped by status or environment limits."
                )

    # ------------------------------
    # Engine control panel (for worker)
    # ------------------------------
    st.markdown("---")
    st.subheader("Engine control panel (for background worker)")

    control_state = load_control_state()
    current_status = control_state.get("status", "idle")
    st.write(f"Current engine status: **{current_status}**")

    if control_state:
        with st.expander("Raw control state"):
            st.code(json.dumps(control_state, indent=2), language="json")
    else:
        st.write("No control state file yet. Configure a timed run to create one.")

    col_start, col_pause, col_stop = st.columns(3)

    with col_start:
        if st.button("Set status: running"):
            new_state = load_control_state()
            new_state["status"] = "running"
            new_state["timestamp_utc"] = datetime.utcnow().isoformat()
            save_control_state(new_state)
            st.success("Engine status set to running.")

    with col_pause:
        if st.button("Set status: paused"):
            new_state = load_control_state()
            new_state["status"] = "paused"
            new_state["timestamp_utc"] = datetime.utcnow().isoformat()
            save_control_state(new_state)
            st.info("Engine status set to paused.")

    with col_stop:
        if st.button("Set status: stopped"):
            new_state = load_control_state()
            new_state["status"] = "stopped"
            new_state["timestamp_utc"] = datetime.utcnow().isoformat()
            save_control_state(new_state)
            st.warning("Engine status set to stopped. Worker should halt after the current cycle.")

    # ------------------------------
    # History + Charts
    # ------------------------------
    st.markdown("---")
    st.subheader("Cycle history")

    history = memory.get_cycle_history()
    if not history:
        st.write("No cycles yet.")
    else:
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

        # To avoid frontend RangeError on very long runs,
        # only send the most recent MAX_POINTS_FOR_CHARTS points to the charts.
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

        with st.expander("Raw JSON"):
            st.code(json.dumps(history, indent=2), language="json")

    # ------------------------------
    # Run diagnostics (continuous mode support from MemoryStore)
    # ------------------------------
    st.markdown("---")
    st.subheader("Run diagnostics")

    col_state, col_watchdog = st.columns(2)

    with col_state:
        st.markdown("**Last saved run state (MemoryStore)**")
        state = memory.load_run_state()
        if not state:
            st.write("No saved run state yet.")
        else:
            st.json(state)
            if st.button("Clear saved run state", key="clear_run_state_btn"):
                memory.clear_run_state()
                st.success("Saved run state cleared. It will be rebuilt on the next continuous run by the worker.")

    with col_watchdog:
        st.markdown("**Watchdog heartbeat (MemoryStore)**")
        info = memory.get_watchdog_info()
        last_beat = info.get("last_beat")
        count = info.get("count", 0)
        seconds_since = info.get("seconds_since_last")

        st.write(f"Last beat: {last_beat if last_beat else 'None recorded'}")
        st.write(f"Heartbeat count: {count}")
        if isinstance(seconds_since, (int, float)):
            st.write(f"Seconds since last beat: {seconds_since:.1f}")
        else:
            st.write("Seconds since last beat: not available")

    # ------------------------------
    # Report generation
    # ------------------------------
    st.markdown("---")
    st.subheader("Generate report")

    st.caption(
        "Build summarized reports from the current cycle history. "
        "You can re run these after long autonomous sessions."
    )

    col_rep1, col_rep2 = st.columns(2)

    with col_rep1:
        if st.button("Generate full history report"):
            report_md = generate_report(memory_store=memory, goal=None)
            st.markdown(report_md)
            st.download_button(
                "Download report as Markdown",
                data=report_md,
                file_name="autonomous_research_report.md",
                mime="text/markdown",
            )

    with col_rep2:
        if st.button("Generate outcome focused summary"):
            outcome_md = build_outcome_summary(memory.get_cycle_history())
            st.markdown(outcome_md)
            st.download_button(
                "Download outcome summary",
                data=outcome_md,
                file_name="autonomous_outcome_summary.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
