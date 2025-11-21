"""Enhanced Streamlit interface for the Autonomous Research Agent.

New features added:
- Continuous Mode (agent runs forever until stopped)
- Researcher + Critic multi-agent mode
- PubMed / Semantic Scholar ingestion controls
- Biomarker analysis mode (for anti-aging teams)
- Hypothesis generation viewer
- PDF ingestion for real scientific papers
- RYE, delta_R, and Energy charts
- Real Tavily search support detection
- Source citation viewer
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
import yaml

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore

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
    st.markdown(f"### Cycle {cycle_summary['cycle'] + 1} ({cycle_summary.get('role', 'agent')})")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("delta_R", cycle_summary["delta_R"])
    with col2:
        st.metric("Energy E", cycle_summary["energy_E"])
    with col3:
        st.metric("RYE", round(cycle_summary["RYE"], 3))

    # Issues
    if cycle_summary["issues_before"]:
        st.write("Issues before repair:")
        for issue in cycle_summary["issues_before"]:
            st.write(f"- {issue}")
    else:
        st.write("No issues detected before repair.")

    # Repairs
    if cycle_summary["repairs"]:
        st.write("Repairs applied:")
        for rep in cycle_summary["repairs"]:
            st.write(f"- {rep}")
    else:
        st.write("No repairs performed.")

    # Notes
    if cycle_summary.get("notes_added"):
        with st.expander("Notes added"):
            for note in cycle_summary["notes_added"]:
                st.write(f"- {note}")

    # Hypotheses (NEW)
    if cycle_summary.get("hypotheses"):
        with st.expander("Generated hypotheses"):
            for h in cycle_summary["hypotheses"]:
                st.write(f"• {h}")

    # Citations
    if cycle_summary.get("citations"):
        with st.expander("Citations for this cycle"):
            for c in cycle_summary["citations"]:
                st.write(f"- [{c.get('source','')}] {c.get('title','')} — {c.get('url','')}")


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

    # Tavily status
    status = tavily_status()
    st.sidebar.subheader("Internet research")
    if status["has_key"]:
        st.sidebar.success(status["display"])
    else:
        st.sidebar.warning(status["display"])
        st.sidebar.write("Add TAVILY_API_KEY in Streamlit Secrets to enable real web search.")

    # Multi-agent toggle
    multi_agent = st.sidebar.checkbox("Enable Multi Agent (Researcher + Critic)")

    # PubMed
    use_pubmed = st.sidebar.checkbox("Use PubMed (scientific literature)")

    # Semantic Scholar
    use_semantic = st.sidebar.checkbox("Use Semantic Scholar ingestion")

    # PDF ingestion
    use_pdf = st.sidebar.checkbox("Enable PDF ingestion (upload papers below)")

    uploaded_pdf = None
    if use_pdf:
        uploaded_pdf = st.sidebar.file_uploader("Upload a PDF paper", type=["pdf"])

    # Biomarker mode
    use_biomarkers = st.sidebar.checkbox("Biomarker / Longevity Mode (anti-aging teams)")

    # Continuous Mode
    continuous_mode = st.sidebar.checkbox("Continuous mode (run without stopping)")

    # -----------------------------
    # Main area
    # -----------------------------
    st.subheader("Research goal")

    default_goal = (
        "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
        "identify similar frameworks in the literature, and produce a structured comparison table."
    )
    goal = st.text_area("Enter research goal:", value=default_goal, height=160)

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
        st.write("Running agent…")
        history = memory.get_cycle_history()
        next_index = len(history)
        results: List[Dict[str, Any]] = []

        # Attach source preferences to agent
        agent.source_controls = {
            "pubmed": use_pubmed,
            "semantic": use_semantic,
            "pdf": use_pdf,
            "biomarkers": use_biomarkers,
        }

        if uploaded_pdf:
            agent.attach_pdf(uploaded_pdf)

        if continuous_mode:
            st.warning("Continuous mode enabled — agent will run until manually stopped.")
            results = agent.run_continuous(goal)
        else:
            if not multi_agent:
                for i in range(int(cycles)):
                    ci = next_index + i
                    out = agent.run_cycle(goal=goal, cycle_index=ci)
                    results.append(out["summary"])
            else:
                for i in range(int(cycles)):
                    base = next_index + 2 * i

                    # Researcher
                    r = agent.run_cycle(goal=goal, cycle_index=base, role="researcher")
                    results.append(r["summary"])

                    # Critic
                    critic_goal = f"Critically review and refine notes for: {goal}"
                    c = agent.run_cycle(goal=critic_goal, cycle_index=base + 1, role="critic")
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
            rows.append(
                {
                    "cycle": entry.get("cycle"),
                    "role": entry.get("role", "agent"),
                    "goal": entry.get("goal", "")[:60] + ("..." if len(entry.get("goal", "")) > 60 else ""),
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
            st.caption("Higher RYE = more efficient repair.")

            st.line_chart({"delta_R": delta_y})
            st.caption("delta_R = how much improvement each cycle produced.")

            st.line_chart({"energy_E": energy_y})
            st.caption("Energy per cycle (approximate effort cost).")

        with st.expander("Raw JSON"):
            st.code(json.dumps(history, indent=2), language="json")


if __name__ == "__main__":
    main()
