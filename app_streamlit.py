"""Streamlit interface for the Autonomous Research Agent.

This UI lets you:
- Set a research goal
- Choose how many TGRM cycles to run
- See human readable summaries
- Inspect RYE, delta_R, and energy per cycle
- View past cycle history stored in the MemoryStore
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


def render_cycle_summary(cycle_summary: Dict[str, Any]) -> None:
    """Pretty print one cycle summary inside Streamlit."""
    st.markdown(f"### Cycle {cycle_summary['cycle'] + 1}")

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


def main() -> None:
    st.title("Autonomous Research Agent")
    st.caption("Reparodynamics, RYE, and TGRM powered research loop")

    agent, memory = init_agent()

    st.sidebar.header("Run settings")
    default_goal = (
        "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
        "identify similar frameworks in the literature, and produce a structured comparison table."
    )
    goal = st.text_area("Research goal", value=default_goal, height=150)

    cycles = st.number_input(
        "Number of TGRM cycles to run",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
    )

    run_button = st.button("Run agent")

    if run_button:
        st.write("Running agent...")
        cycle_results: List[Dict[str, Any]] = []
        for i in range(int(cycles)):
            result = agent.run_cycle(goal=goal, cycle_index=i)
            cycle_results.append(result["summary"])

        st.subheader("Cycle summaries")
        for cs in cycle_results:
            render_cycle_summary(cs)

    st.markdown("---")
    st.subheader("Cycle history")

    history = memory.get_cycle_history()
    if not history:
        st.write("No past cycles logged yet.")
    else:
        # Simple table of history for quick inspection
        simple_rows = []
        for entry in history:
            simple_rows.append(
                {
                    "cycle": entry.get("cycle"),
                    "goal": entry.get("goal", "")[:60] + ("..." if len(entry.get("goal", "")) > 60 else ""),
                    "delta_R": entry.get("delta_R"),
                    "energy_E": entry.get("energy_E"),
                    "RYE": entry.get("RYE"),
                    "timestamp": entry.get("timestamp"),
                }
            )
        st.dataframe(simple_rows, use_container_width=True)

        with st.expander("Raw history JSON"):
            st.code(json.dumps(history, indent=2), language="json")


if __name__ == "__main__":
    main()
