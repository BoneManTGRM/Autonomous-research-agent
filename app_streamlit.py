"""Enhanced Streamlit interface for the Autonomous Research Agent.

Features:
- Continuous Mode (agent runs up to N cycles or until RYE threshold)
- Researcher + Critic multi-agent mode
- Domain presets (General, Longevity, Math)
- PubMed / Semantic Scholar ingestion controls
- Biomarker analysis toggle (for anti-aging teams)
- Hypothesis generation viewer
- PDF ingestion for real scientific papers
- RYE, delta_R, and Energy charts
- Real Tavily search support detection
- Source citation viewer

Reparodynamics:
    The UI is a front panel on a reparodynamic system:
    - Each click runs the TGRM loop (Test, Detect, Repair, Verify).
    - Each cycle computes RYE = delta_R / E and is logged.
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
from agent.presets import PRESETS, get_preset  # new presets import

CONFIG_PATH_DEFAULT = "config/settings.yaml"


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
    """Check whether a Tavily API key is available."""
    key = None
    try:
        key = st.secrets.get("TAVILY_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = os.getenv("TAVILY_API_KEY")

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


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
def main() -> None:
    st.title("Autonomous Research Agent")
    st.caption("Reparodynamics, RYE, and TGRM powered research loop")

    agent, memory = init_agent()

    # -----------------------------
    # Sidebar – settings
    # -----------------------------
    st.sidebar.header("Run settings")

    # Preset selector (General, Longevity, Math, etc.)
    preset_keys = list(PRESETS.keys())
    preset_labels = [PRESETS[k]["label"] for k in preset_keys]

    # Make "general" the default if it exists
    default_preset_index = 0
    if "general" in preset_keys:
        default_preset_index = preset_keys.index("general")

    selected_label = st.sidebar.selectbox(
        "Preset",
        options=preset_labels,
        index=default_preset_index,
        help="Choose a domain preset. You can still edit all settings below.",
    )
    # Map label back to preset key
    selected_key = preset_keys[preset_labels.index(selected_label)]
    preset = get_preset(selected_key)
    domain_tag = preset.get("domain", selected_key)

    # Tavily status
    status = tavily_status()
    st.sidebar.subheader("Internet research")
    if status["has_key"]:
        st.sidebar.success(status["display"])
    else:
        st.sidebar.warning(status["display"])
        st.sidebar.write("Add TAVILY_API_KEY in Streamlit Secrets to enable real web search.")

    # Multi-agent toggle
    multi_agent = st.sidebar.checkbox("Enable Multi Agent (Researcher + Critic)", value=False)

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

    # Biomarker mode (future use for anti-aging / longevity dashboards)
    use_biomarkers = st.sidebar.checkbox(
        "Biomarker / Longevity Mode (anti-aging teams)",
        value=bool(sc_defaults.get("biomarkers", False)),
    )

    # Continuous mode
    continuous_mode = st.sidebar.checkbox("Continuous mode (up to N cycles with stop condition)", value=False)
    stop_rye_threshold: Optional[float] = None
    if continuous_mode:
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

    # Default goal from preset, but user can override
    default_goal = preset.get("default_goal") or (
        "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
        "identify similar frameworks in the literature, and produce a structured comparison table."
    )

    # Use a key for the goal so the preset is used as default,
    # but user edits are preserved across reruns.
    if "goal_text" not in st.session_state:
        st.session_state["goal_text"] = default_goal

    goal = st.text_area("Enter research goal:", value=st.session_state["goal_text"], height=160)
    # Update session state to keep latest text
    st.session_state["goal_text"] = goal

    cycles = st.number_input(
        "Number of TGRM cycles to run",
        min_value=1,
        max_value=200,
        value=3,
        step=1,
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

        # Build source_controls dict for the agent / TGRM loop
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

        if continuous_mode:
            st.warning("Continuous mode enabled. The agent will run multiple cycles until limit or stop condition.")
            # CoreAgent.run_continuous handles extra kwargs with a safe wrapper
            summaries = agent.run_continuous(
                goal=goal,
                max_cycles=int(cycles),
                stop_rye=stop_rye_threshold,
                role="agent",
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=None,
                domain=domain_tag,
            )
            results.extend(summaries)
        else:
            # Finite cycles mode
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
                # Multi-agent: researcher + critic per logical cycle
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
                        pdf_bytes=None,  # critic does not need to re-ingest PDF
                        biomarker_snapshot=None,
                        domain=domain_tag,
                    )
                    results.append(c["summary"])

        # Display cycle summaries
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


if __name__ == "__main__":
    main()
