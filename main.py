"""Entry point for the Autonomous Research Agent.

This module orchestrates the research agent by reading configuration,
initialising the core agent, and running a specified number of research cycles.

The agent is designed around the concepts of Reparodynamics, RYE, and TGRM.

Usage:
    python main.py --goal "Your research goal here" --cycles 3

This will run the agent for the given number of cycles and print the results.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Config and filesystem helpers
# ---------------------------------------------------------------------------


def load_settings(config_path: str) -> Dict[str, Any]:
    """Load YAML settings file into a dictionary, safely."""
    import yaml  # imported here to avoid a hard dependency if not used elsewhere

    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        # Never crash the CLI because of a bad config
        return {}


def ensure_directories() -> None:
    """Ensure that necessary directories for logs and sessions exist."""
    logs_path = Path("logs")
    sessions_path = logs_path / "sessions"
    logs_path.mkdir(exist_ok=True)
    sessions_path.mkdir(exist_ok=True)


def init_agent(config_path: str) -> CoreAgent:
    """Create a CoreAgent and its MemoryStore from a config path."""
    ensure_directories()
    config = load_settings(config_path)

    # Allow memory file override in config or env
    memory_file = os.getenv("CLI_MEMORY_FILE", config.get("memory_file", "logs/sessions/default_memory.json"))
    memory_path = Path(memory_file)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    memory = MemoryStore(str(memory_path))
    agent = CoreAgent(memory_store=memory, config=config)
    return agent


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Autonomous Research Agent in simple CLI mode")
    parser.add_argument(
        "--goal",
        type=str,
        required=False,
        default=(
            "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
            "identify similar frameworks in the literature, and produce a structured comparison table."
        ),
        help="Research goal for the agent",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        required=False,
        default=3,
        help="Number of cycles to run the agent",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="config/settings.yaml",
        help="Path to settings YAML file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    agent = init_agent(args.config)

    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    print(f"Starting session {session_id}")
    print(f"Goal: {args.goal}")
    print(f"Cycles: {args.cycles}")
    print("")

    for cycle in range(args.cycles):
        result: Dict[str, Any] = agent.run_cycle(goal=args.goal, cycle_index=cycle)

        # Be tolerant of different return formats from run_cycle
        summary = result.get("summary", result)
        print(f"Cycle {cycle + 1}/{args.cycles} Summary:")

        if isinstance(summary, dict):
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            # Fallback if summary is already a string or something similar
            print(summary)

        print("\n---\n")

    print("Agent run complete. Logs and memory are stored under logs/.")


if __name__ == "__main__":
    main()
