"""
Background engine worker for the Autonomous Research Agent.

This file is the long running engine that can run on Render as a
Background Worker or separate service.

Key ideas:
- Uses the same CoreAgent and MemoryStore as the Streamlit UI.
- Runs continuous mode (single agent or swarm) without any Streamlit session.
- Uses environment variables to control goal, domain, runtime, and swarm.
- Relies on CoreAgent checkpoints and watchdog for crash recovery.

You can start it with commands like:
    WORKER_GOAL="Long run test on reparodynamics" \
    WORKER_RUNTIME_PROFILE="8_hours" \
    python engine_worker.py

On Render, use:
    startCommand: python engine_worker.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore
from agent.presets import PRESETS  # only to get default goals
import yaml


CONFIG_PATH_DEFAULT = "config/settings.yaml"


def load_settings(config_path: str = CONFIG_PATH_DEFAULT) -> Dict[str, Any]:
    """Load YAML settings file into a dictionary."""
    if not Path(config_path).exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def ensure_directories() -> None:
    """Ensure that log directories exist, same pattern as the Streamlit app."""
    logs_path = Path("logs")
    sessions_path = logs_path / "sessions"
    logs_path.mkdir(exist_ok=True)
    sessions_path.mkdir(exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean from environment variables."""
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _env_float(name: str) -> Optional[float]:
    """Parse a float from environment variables, or None if missing."""
    val = os.getenv(name)
    if not val:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _env_list(name: str) -> Optional[List[str]]:
    """Parse a comma separated list from environment variables."""
    val = os.getenv(name)
    if not val:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts or None


def init_agent_from_config() -> CoreAgent:
    """Create a CoreAgent and MemoryStore from config/settings.yaml."""
    ensure_directories()
    config = load_settings(CONFIG_PATH_DEFAULT)

    memory_file = config.get("memory_file", "logs/sessions/default_memory.json")
    memory = MemoryStore(memory_file)
    agent = CoreAgent(memory_store=memory, config=config)
    return agent


def build_goal_and_domain() -> (str, str):
    """
    Decide which goal and domain to use for this worker.

    Priority:
    1. WORKER_GOAL env
    2. A default_worker_goal entry in config (if you add one later)
    3. Longevity or general preset default goal as a fallback
    """
    config = load_settings(CONFIG_PATH_DEFAULT)

    # Goal
    env_goal = os.getenv("WORKER_GOAL")
    if env_goal:
        goal = env_goal
    elif isinstance(config.get("default_worker_goal"), str):
        goal = config["default_worker_goal"]
    else:
        # Try to use the longevity preset first if it exists
        if "longevity" in PRESETS:
            goal = PRESETS["longevity"].get(
                "default_goal",
                "Long run autonomous research on anti aging, longevity, and reparodynamics.",
            )
        else:
            goal = (
                "Long run autonomous research on reparodynamics, RYE, TGRM, "
                "and similar stability or repair frameworks."
            )

    # Domain
    env_domain = os.getenv("WORKER_DOMAIN")
    if env_domain:
        domain = env_domain
    elif isinstance(config.get("default_worker_domain"), str):
        domain = config["default_worker_domain"]
    else:
        domain = "general"

    return goal, domain


def run_single_agent_engine(agent: CoreAgent) -> None:
    """
    Run a long continuous single agent session.

    Controlled by:
    - WORKER_MAX_MINUTES (optional)
    - WORKER_STOP_RYE (optional)
    - WORKER_RUNTIME_PROFILE (optional: 1_hour, 8_hours, 24_hours, forever)
    - WORKER_ROLE (default 'agent')
    """
    goal, domain = build_goal_and_domain()

    max_minutes = _env_float("WORKER_MAX_MINUTES")
    stop_rye = _env_float("WORKER_STOP_RYE")
    runtime_profile = os.getenv("WORKER_RUNTIME_PROFILE")  # example: "8_hours"
    role = os.getenv("WORKER_ROLE", "agent")

    print("=== Autonomous Research Engine - Single Agent Mode ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Role: {role}")
    print(f"Runtime profile: {runtime_profile or 'none'}")
    print(f"Max minutes: {max_minutes if max_minutes is not None else 'None (will use profile or forever)'}")
    print(f"Stop RYE threshold: {stop_rye if stop_rye is not None else 'disabled'}")
    sys.stdout.flush()

    summaries = agent.run_continuous(
        goal=goal,
        max_cycles=10_000_000,
        stop_rye=stop_rye,
        role=role,
        source_controls={"web": True, "pubmed": True, "semantic": True, "pdf": True, "biomarkers": False},
        pdf_bytes=None,
        biomarker_snapshot=None,
        domain=domain,
        max_minutes=max_minutes,
        forever=(runtime_profile == "forever" and max_minutes is None),
        resume_from_checkpoint=True,
        watchdog_interval_minutes=5.0,
        runtime_profile=runtime_profile,
    )

    print("=== Continuous run finished cleanly ===")
    print(f"Total completed cycles: {len(summaries)}")
    sys.stdout.flush()


def run_swarm_engine(agent: CoreAgent) -> None:
    """
    Run a long continuous swarm session.

    Controlled by:
    - WORKER_SWARM_ROLES (comma separated, optional)
    - WORKER_MAX_MINUTES (optional)
    - WORKER_STOP_RYE (optional)
    - WORKER_RUNTIME_PROFILE (optional: 1_hour, 8_hours, 24_hours, forever)
    """
    goal, domain = build_goal_and_domain()

    max_minutes = _env_float("WORKER_MAX_MINUTES")
    stop_rye = _env_float("WORKER_STOP_RYE")
    runtime_profile = os.getenv("WORKER_RUNTIME_PROFILE")
    roles_list = _env_list("WORKER_SWARM_ROLES")

    if roles_list is None:
        # Use all configured roles inside CoreAgent (capped at 32)
        roles_list = agent.get_agent_roles()

    print("=== Autonomous Research Engine - Swarm Mode ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Roles: {roles_list}")
    print(f"Runtime profile: {runtime_profile or 'none'}")
    print(f"Max minutes: {max_minutes if max_minutes is not None else 'None (will use profile or forever)'}")
    print(f"Stop RYE threshold: {stop_rye if stop_rye is not None else 'disabled'}")
    sys.stdout.flush()

    summaries = agent.run_swarm_continuous(
        goal=goal,
        max_rounds=10_000_000,
        stop_rye=stop_rye,
        roles=roles_list,
        source_controls={"web": True, "pubmed": True, "semantic": True, "pdf": True, "biomarkers": False},
        pdf_bytes=None,
        biomarker_snapshot=None,
        domain=domain,
        max_minutes=max_minutes,
        forever=(runtime_profile == "forever" and max_minutes is None),
        resume_from_checkpoint=True,
        watchdog_interval_minutes=5.0,
        runtime_profile=runtime_profile,
    )

    print("=== Swarm run finished cleanly ===")
    print(f"Total summaries produced across all roles and rounds: {len(summaries)}")
    sys.stdout.flush()


def main() -> None:
    """
    Entry point for the background worker.

    Mode selection:
    - WORKER_MODE=swarm   -> run swarm engine
    - anything else       -> run single agent engine
    """
    print("Starting Autonomous Research Agent background engine...")
    sys.stdout.flush()

    agent = init_agent_from_config()

    use_swarm = _env_bool("WORKER_SWARM", default=False)
    mode = os.getenv("WORKER_MODE", "single").strip().lower()

    try:
        if use_swarm or mode == "swarm":
            run_swarm_engine(agent)
        else:
            run_single_agent_engine(agent)
    except KeyboardInterrupt:
        print("Engine interrupted by user or environment.")
        sys.stdout.flush()
    except Exception:
        print("Fatal error in engine_worker:")
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
