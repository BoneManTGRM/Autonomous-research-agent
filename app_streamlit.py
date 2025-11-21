"""Streamlit interface for the Autonomous Research Agent.

This UI lets you:
- Set a research goal
- Choose how many TGRM cycles to run
- See human readable summaries
- Inspect RYE, delta_R, and energy per cycle
- View past cycle history stored in the MemoryStore

Reparodynamics:
    The agent is interpreted as a reparodynamic system that tries to
    maintain and improve the quality of its internal knowledge over time.

RYE (Repair Yield per Energy):
    Each cycle computes delta_R / E, where delta_R is the improvement in detected
    issues and E is an approximate effort cost (number of actions).
    Higher RYE means more efficient self repair.

TGRM (Test, Detect, Repair, Verify):
    Each cycle in CoreAgent/TGRMLoop follows:
        Test   – evaluate current notes / state
        Detect – find gaps, TODOs, unanswered questions
        Repair – perform targeted web and paper actions
        Verify – re test and compute delta_R and RYE
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
        key = None

    if not key:
        key = os.getenv("TAVILY_API_KEY")

    if key:
        tail = key[-4:]
        return {"has_key": True, "display": f"Tavily key detected (...{tail})"}
    return {"has_key": False, "display": "No Tavily API key found. Web search will use stubbed results."}


def render_cycle_summary(cycle_summary: Dict[str, Any]) -> None:
    """Pretty print one cycle summary inside Streamlit."""
    st.markdown(f"### Cycle {cycle_summary['cycle'] + 1} ({cycle_summary.get('role', 'agent')})")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("delta_R", cycle_summary["delta_R"])
    with col2:
        st.metric("Energy E", cycle_summary["energy_E"])
    with col3:
        st.metric("RYE", round(cycle_summary["RYE"], 3))

    if cycle_summary["issues_before"]:
        st.write("Issues before repair:")
        for issue in cycle_summary["issues_before"]:
            st.write(f"- {issue}")
    else:
        st.write("No issues detected before repair.")

    if cycle_summary["repairs"]:
        st.write("Repairs applied:")
        for rep in cycle_summary["repairs"]:
            st.write(f"- {rep}")
    else:
        st.write("No repair actions applied in this cycle.")

    if cycle_summary["notes_added"]:
        with st.expander("Notes added this cycle"):
            for note in cycle_summary["notes_added"]:
                st.write(f"- {note}")

    if cycle_summary.get("citations"):
        with st.expander("Citations for this cycle"):
            for c in cycle_summary["citations"]:
                src = c.get("source", "")
                title = c.get("title", "")
                url = c.get("url", "")
                st.write(f"- [{src}] {title} ({url})")


def main() -> None:
    st.title("Autonomous Research Agent")
    st.caption("Reparodynamics, RYE, and TGRM powered research loop")

    agent, memory = init_agent()

    # Sidebar: settings and environment status
    st.sidebar.header("Run settings")

    status = tavily_status()
    st.sidebar.subheader("Internet research")
    if status["has_key"]:
        st.sidebar.success(status["display"])
        st.sidebar.write("The agent will perform real Tavily web searches during Repair steps.")
    else:
        st.sidebar.warning(status["display"])
        st.sidebar.write("Add TAVILY_API_KEY to Streamlit Secrets to enable real internet research.")

    default_goal = (
        "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
        "identify similar frameworks in the literature, and produce a structured comparison table."
    )
    goal = st.text_area("Research goal", value=default_goal, height=150)

    multi_agent = st.sidebar.checkbox("Enable multi agent (researcher + critic)", value=False)

    cycles = st.number_input(
        "Number of TGRM cycles to run in this session",
        min_value=1,
        max_value=50,
        value=3,
        step=1,
    )

    run_button = st.button("Run agent")

    # Main: run cycles
    if run_button:
        st.write("Running agent...")
        history = memory.get_cycle_history()
        next_index = len(history)

        cycle_results: List[Dict[str, Any]] = []

        if not multi_agent:
            # Single agent mode
            for i in range(int(cycles)):
                ci = next_index + i
                result = agent.run_cycle(goal=goal, cycle_index=ci)
                cycle_results.append(result["summary"])
        else:
            # Multi agent mode: researcher then critic for each logical cycle
            for i in range(int(cycles)):
                base_idx = next_index + 2 * i

                # Primary researcher
                res_researcher = agent.run_cycle(goal=goal, cycle_index=base_idx, role="researcher")
                cycle_results.append(res_researcher["summary"])

                # Critic agent refines existing notes
                critic_goal = f"Critically review and refine notes for: {goal}"
                res_critic = agent.run_cycle(goal=critic_goal, cycle_index=base_idx + 1, role="critic")
                cycle_results.append(res_critic["summary"])

        st.subheader("Cycle summaries")
        for cs in cycle_results:
            render_cycle_summary(cs)

    # History section
    st.markdown("---")
    st.subheader("Cycle history")

    history = memory.get_cycle_history()
    if not history:
        st.write("No past cycles logged yet.")
    else:
        simple_rows = []
        for entry in history:
            simple_rows.append(
                {
                    "cycle": entry.get("cycle"),
                    "role": entry.get("role", "agent"),
                    "goal": entry.get("goal", "")[:60]
                    + ("..." if len(entry.get("goal", "")) > 60 else ""),
                    "delta_R": entry.get("delta_R"),
                    "energy_E": entry.get("energy_E"),
                    "RYE": entry.get("RYE"),
                    "timestamp": entry.get("timestamp"),
                }
            )
        st.dataframe(simple_rows, use_container_width=True)

        # RYE and TGRM efficiency charts
        st.markdown("### RYE and TGRM efficiency charts")

        cycles_x = [r["cycle"] for r in simple_rows if r["cycle"] is not None]
        rye_vals = [r["RYE"] for r in simple_rows]
        delta_vals = [r["delta_R"] for r in simple_rows]
        energy_vals = [r["energy_E"] for r in simple_rows]

        if cycles_x:
            st.line_chart({"RYE": rye_vals})
            st.caption("RYE over cycles (higher is better repair yield per energy).")
            st.line_chart({"delta_R": delta_vals})
            st.caption("delta_R over cycles (how much improvement each cycle achieves).")
            st.line_chart({"energy_E": energy_vals})
            st.caption("Energy E over cycles (approximate effort per cycle).")

        with st.expander("Raw history JSON"):
            st.code(json.dumps(history, indent=2), language="json")


if __name__ == "__main__":
    main()
