"""Entry point for the autonomous research agent.

This module orchestrates the research agent by reading configuration,
initialising the core agent, and running a specified number of research cycles.

The agent is designed around the concepts of Reparodynamics, RYE, and TGRM.

Usage:
    python main.py --goal "Your research goal here" --cycles 3

This will run the agent for the given number of cycles and print the results.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore


def load_settings(config_path: str) -> dict:
    """Load YAML settings file into a dictionary."""
    import yaml  # imported here to avoid a hard dependency if not used elsewhere
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories():
    """Ensure that necessary directories for logs and sessions exist."""
    logs_path = Path("logs")
    sessions_path = logs_path / "sessions"
    logs_path.mkdir(exist_ok=True)
    sessions_path.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the autonomous research agent")
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


def main():
    args = parse_args()
    ensure_directories()

    # Load configuration if available
    config = {}
    if os.path.exists(args.config):
        config = load_settings(args.config)

    # Initialise memory store and core agent
    memory = MemoryStore(config.get("memory_file", "logs/sessions/default_memory.json"))
    agent = CoreAgent(memory_store=memory, config=config)

    # Run cycles
    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    print(f"Starting session {session_id} with goal: {args.goal}\n")
    for cycle in range(args.cycles):
        result = agent.run_cycle(goal=args.goal, cycle_index=cycle)
        # Print cycle summary
        print(f"Cycle {cycle + 1}/{args.cycles} Summary:")
        print(json.dumps(result["summary"], indent=2))
        print("\n---\n")

    print("Agent run complete. Logs are stored in logs/.")


if __name__ == "__main__":
    main()
