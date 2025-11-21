"""Enhanced Streamlit interface for the Autonomous Research Agent.

Features:
- Continuous Mode with duration presets (1h, 8h, 24h, 90 days, Forever)
- Researcher + Critic multi-agent mode
- Swarm mode with up to 5 specialized roles
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

# Swarm roles: up to 5 specialized mini agents
SWARM_ROLES: List[Tuple[str, str]] = [
    ("researcher", "Deep literature and web researcher"),
    ("critic", "Methodology critic and refiner"),
    ("explorer", "Out-of-distribution explorer (new angles, analogies)"),
    ("theorist", "Model builder and unifier"),
    ("integrator", "Synthesizer that integrates and summarizes"),
]


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
    """Check whether a Tavily API key is available (per-user or env)."""

    # 1) Prefer per-user key stored in session state (from sidebar input)
    key = st.session_state.get("tavily_key", None)

    # 2) Fallback to environment variable (in case you set it on the server)
    if not key:
        key = os.getenv("TAVILY_API_KEY")

    # 3) Optional final fallback to secrets (owner-only use, can be empty)
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
                st.write(f"- [{src}] {title} — {url}")


def build_swarm_roles(enabled: bool, swarm_size: int) -> List[Tuple[str, str]]:
    """Return the active swarm roles (name, description) given size."""
    if not enabled or swarm_size <= 1:
        return []
    size = max(1, min(swarm_size, len(SWARM_ROLES)))
    return SWARM_ROLES[:size]


def role_specific_goal(base_goal: str, role: str) -> str:
    """Specialize the goal text slightly for each swarm role."""
    base_goal = base_goal.strip()
    if role == "researcher":
        return (
            f"Primary deep research agent for goal: {base_goal}.\n"
            "Focus on high quality sources, detailed notes, and clear summaries."
        )
    if role == "critic":
        return (
            f"Critically review, cross check, and refine all existing Reparodynamic notes and hypotheses for: {base_goal}.\n"
            "Identify weaknesses, gaps, and overclaims."
        )
    if role == "explorer":
        return (
            f"Exploration agent for goal: {base_goal}.\n"
            "Look for unusual angles, analogies, adjacent fields, and surprising connections."
        )
    if role == "theorist":
        return (
            f"Theory building agent for goal: {base_goal}.\n"
            "Try to organize findings into coherent models, equations, or structured frameworks."
        )
    if role == "integrator":
        return (
            f"Integration agent for goal: {base_goal}.\n"
            "Synthesize results from all prior agents into clear narratives, tables, and distilled insights."
        )
    # Fallback: use the original goal
    return base_goal


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
def main() -> None:
    st.title("Autonomous Research Agent")
    st.caption("Reparodynamics, RYE, and TGRM powered research loop (with swarm mode)")

    agent, memory = init_agent()

    # -----------------------------
    # Sidebar - settings
    # -----------------------------
    st.sidebar.header("Run settings")

    # Tavily key input (per-user)
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
        help="Run up to 5 specialized agents (researcher, critic, explorer, theorist, integrator).",
    )
    swarm_size = 1
    swarm_roles: List[Tuple[str, str]] = []
    if enable_swarm:
        swarm_size = st.sidebar.slider(
            "Number of swarm agents",
            min_value=2,
            max_value=len(SWARM_ROLES),
            value=3,
            help="Swarm size is how many specialized roles you activate at once.",
        )
        swarm_roles = build_swarm_roles(True, swarm_size)
        st.sidebar.write("Active roles:")
        for name, desc in swarm_roles:
            st.sidebar.write(f"- **{name}**: {desc}")

    # Multi-agent toggle (classic researcher + critic)
    # Disabled when swarm is on, because swarm already includes critic logic.
    multi_agent = False
    if not enable_swarm:
        multi_agent = st.sidebar.checkbox(
            "Enable classic Multi Agent (Researcher + Critic)",
            value=False,
            help="If swarm is disabled, you can still run a simple researcher+critic pair.",
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
        help="Timed modes respect real wall-clock minutes via the agent's time budget.",
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
        "Number of TGRM cycles to run (manual mode)",
        min_value=1,
        max_value=200,
        value=3,
        step=1,
        help="Used when Run mode is 'Manual (finite cycles)'. Timed modes ignore this and rely on minutes.",
    )

    run_button = st.button("Run agent")

    # ------------------------------
    # Run cycles
    # ------------------------------
    if run_button:
        st.write(f"Running agent with preset: {preset.get('label', selected_label)} (domain: {domain_tag})")
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

        # Manual finite mode
        if run_mode == "Manual (finite cycles)":
            # Swarm manual mode
            if enable_swarm and swarm_roles:
                st.info(
                    f"Manual Swarm mode: {len(swarm_roles)} roles × {int(cycles)} cycles "
                    f"(total {len(swarm_roles) * int(cycles)} mini-cycles)."
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
                # Classic single or researcher+critic manual mode
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
        else:
            # Continuous modes with real time budget via max_minutes
            # Map run_mode to minutes
            max_minutes: Optional[float] = None
            forever_flag: bool = False

            if run_mode == "1 hour (real clock)":
                max_minutes = 60.0
            elif run_mode == "8 hours (real clock)":
                max_minutes = 8 * 60.0
            elif run_mode == "24 hours (real clock)":
                max_minutes = 24 * 60.0
            elif run_mode == "90 days (real clock)":
                max_minutes = 90.0 * 24.0 * 60.0
            elif run_mode == "Forever (until stopped)":
                max_minutes = None
                forever_flag = True

            # Cycles are now a safety cap. Use a large upper bound.
            if forever_flag:
                effective_max_cycles = 10_000_000
            else:
                # For timed runs, cycles are not the primary limiter,
                # so we pick a large cap so time is the real stop condition.
                effective_max_cycles = 10_000_000

            # Swarm + Forever is ambiguous in a single-threaded UI, so we
            # treat it as a single agent run when Forever is selected.
            if enable_swarm and forever_flag:
                st.warning(
                    "Swarm mode with 'Forever' is not supported in this UI. "
                    "Running as a single continuous agent instead."
                )
                swarm_roles = []

            # Show what the agent will try to do
            if max_minutes is not None:
                st.info(
                    f"Continuous mode: target {run_mode} "
                    f"(time budget ~ {max_minutes:.1f} minutes total, up to {effective_max_cycles} cycles). "
                    f"RYE stop condition: {'disabled' if stop_rye_threshold is None else stop_rye_threshold}."
                )
            else:
                st.info(
                    "Continuous mode: Forever (until stopped by environment or RYE threshold). "
                    f"Cycle safety cap: {effective_max_cycles}. "
                    f"RYE stop condition: {'disabled' if stop_rye_threshold is None else stop_rye_threshold}."
                )

            # Continuous swarm: split time budget across roles
            if enable_swarm and swarm_roles and max_minutes is not None:
                minutes_per_agent = max_minutes / float(len(swarm_roles))
                st.info(
                    f"Swarm continuous mode: {len(swarm_roles)} roles, "
                    f"~{minutes_per_agent:.1f} minutes per role."
                )
                for idx, (role_name, _) in enumerate(swarm_roles, start=1):
                    st.write(f"Starting swarm agent {idx}/{len(swarm_roles)} (role: {role_name})...")
                    role_goal = role_specific_goal(goal, role_name)
                    summaries = agent.run_continuous(
                        goal=role_goal,
                        max_cycles=int(effective_max_cycles),
                        stop_rye=stop_rye_threshold,
                        role=role_name,
                        source_controls=source_controls,
                        pdf_bytes=pdf_bytes,
                        biomarker_snapshot=None,
                        domain=domain_tag,
                        max_minutes=minutes_per_agent,
                        forever=False,
                    )
                    results.extend(summaries)
            else:
                # Single-agent or classic continuous run
                summaries = agent.run_continuous(
                    goal=goal,
                    max_cycles=int(effective_max_cycles),
                    stop_rye=stop_rye_threshold,
                    role="agent",
                    source_controls=source_controls,
                    pdf_bytes=pdf_bytes,
                    biomarker_snapshot=None,
                    domain=domain_tag,
                    max_minutes=max_minutes,
                    forever=forever_flag,
                )
                results.extend(summaries)

        # Show cycle summaries
        st.subheader("Cycle Summaries")
        for cs in results:
            render_cycle_summary(cs)

    # ------------------------------
    # History + Charts
    # ------------------------------
    st.markdown("---")
    st.subheader("Cycle history")

    history = memory.get_cycle_history()
    if not history:
        st.write("No cycles yet.")
    else:
        rows = []
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

        cycles_x = [r["cycle"] for r in rows if r["cycle"] is not None]
        rye_y = [r["RYE"] for r in rows]
        delta_y = [r["delta_R"] for r in rows]
        energy_y = [r["energy_E"] for r in rows]

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
    # Report generation
    # ------------------------------
    st.markdown("---")
    st.subheader("Generate report")

    st.caption(
        "Build a summarized report from the current cycle history. "
        "You can re-run this after long autonomous sessions."
    )

    if st.button("Generate report from full history"):
        report_md = generate_report(memory_store=memory, goal=None)
        st.markdown(report_md)

        st.download_button(
            "Download report as Markdown",
            data=report_md,
            file_name="autonomous_research_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
