"""
Background engine worker for the Autonomous Research Agent.

This file is the long-running engine that can run on Render as a
Background Worker or separate service.

Key ideas:
- Uses the same CoreAgent and MemoryStore as the Streamlit UI.
- Runs continuous mode (single agent or swarm) without any Streamlit session.
- Uses environment variables to control goal, domain, runtime, and swarm.
- Relies on CoreAgent presets, checkpoints, and watchdog for crash recovery.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore
from agent.presets import PRESETS, get_preset  # PRESETS for backward compat defaults


CONFIG_PATH_DEFAULT = "config/settings.yaml"


# ---------------------------------------------------------------------------
# Config + environment helpers
# ---------------------------------------------------------------------------


def load_settings(config_path: str = CONFIG_PATH_DEFAULT) -> Dict[str, Any]:
    """Load YAML settings file into a dictionary."""
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
        # Never crash the worker because of a bad config file
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


def _build_source_controls(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Build source controls for the worker.

    Priority:
    1. WORKER_SOURCES env (comma separated: web,pubmed,semantic,pdf,biomarkers)
       - If present, only those listed are enabled.
    2. config["default_source_controls"] from YAML (if provided)
    3. Hard-coded defaults.
    """
    # Base defaults
    defaults: Dict[str, bool] = {
        "web": True,
        "pubmed": True,
        "semantic": True,
        "pdf": True,
        "biomarkers": False,
    }

    cfg_sc = config.get("default_source_controls")
    if isinstance(cfg_sc, dict):
        merged = defaults.copy()
        for k, v in cfg_sc.items():
            merged[str(k)] = bool(v)
        defaults = merged

    env_sources = _env_list("WORKER_SOURCES")
    if env_sources is None:
        return defaults

    # If WORKER_SOURCES is set, treat it as an allow-list
    allowed = {s.lower() for s in env_sources}
    result: Dict[str, bool] = {}
    for key in defaults.keys():
        result[key] = key.lower() in allowed
    return result


# ---------------------------------------------------------------------------
# Agent initialisation
# ---------------------------------------------------------------------------


def init_agent_from_config() -> Tuple[CoreAgent, Dict[str, Any]]:
    """Create a CoreAgent and MemoryStore from config/settings.yaml."""
    ensure_directories()
    config = load_settings(CONFIG_PATH_DEFAULT)

    # Allow overriding memory file and config path via env for flexible deployments
    memory_file = os.getenv("WORKER_MEMORY_FILE", config.get("memory_file", "logs/sessions/default_memory.json"))
    memory_path = Path(memory_file)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    memory = MemoryStore(str(memory_path))
    agent = CoreAgent(memory_store=memory, config=config)
    return agent, config


def build_goal_and_domain() -> Tuple[str, str]:
    """
    Decide which goal and domain to use for this worker.

    Priority for GOAL:
    1. WORKER_GOAL env
    2. config["default_worker_goal"]
    3. Preset default goal (longevity if available, else general reparodynamics)

    Priority for DOMAIN:
    1. WORKER_DOMAIN env
    2. config["default_worker_domain"]
    3. "longevity" if longevity preset exists, else "general"
    """
    config = load_settings(CONFIG_PATH_DEFAULT)

    # Goal
    env_goal = os.getenv("WORKER_GOAL")
    if env_goal:
        goal = env_goal
    elif isinstance(config.get("default_worker_goal"), str):
        goal = config["default_worker_goal"]
    else:
        # Prefer longevity preset goal if available
        if "longevity" in PRESETS:
            longevity_preset = get_preset("longevity")
            goal = longevity_preset.get(
                "default_goal",
                "Long run autonomous research on anti aging, longevity, and reparodynamics.",
            )
        else:
            general_preset = get_preset("general")
            goal = general_preset.get(
                "default_goal",
                "Long run autonomous research on reparodynamics, RYE, TGRM, and related stability frameworks.",
            )

    # Domain
    env_domain = os.getenv("WORKER_DOMAIN")
    if env_domain:
        domain = env_domain
    elif isinstance(config.get("default_worker_domain"), str):
        domain = config["default_worker_domain"]
    else:
        domain = "longevity" if "longevity" in PRESETS else "general"

    return goal, domain


def _heartbeat(agent: CoreAgent, label: str) -> None:
    """
    Best-effort heartbeat into the MemoryStore so the UI can see that the
    worker is alive. Silently ignored if heartbeat is not implemented.
    """
    try:
        ms = getattr(agent, "memory_store", None)
        if ms is not None and hasattr(ms, "heartbeat"):
            ms.heartbeat(label=label)
    except Exception:
        # Heartbeat must never crash the worker
        return


# ---------------------------------------------------------------------------
# Single-agent engine
# ---------------------------------------------------------------------------


def run_single_agent_engine(agent: CoreAgent, config: Dict[str, Any]) -> None:
    """
    Run a long continuous single agent session.

    Controlled by:
    - WORKER_MAX_MINUTES (optional)
    - WORKER_STOP_RYE (optional)
    - WORKER_RUNTIME_PROFILE (optional: 1_hour, 8_hours, 24_hours, 90_days, forever)
    - WORKER_ROLE (default 'agent')
    - WORKER_SOURCES (optional, comma separated)
    - WORKER_RESUME (bool, default True)
    - WORKER_WATCHDOG_MINUTES (float, default 5.0)
    - WORKER_FOREVER (bool, default False)
    """
    goal, domain = build_goal_and_domain()

    max_minutes = _env_float("WORKER_MAX_MINUTES")
    stop_rye = _env_float("WORKER_STOP_RYE")
    runtime_profile = os.getenv("WORKER_RUNTIME_PROFILE")  # example: "8_hours"
    role = os.getenv("WORKER_ROLE", "agent")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0

    # Determine whether to run "forever" based on profile or explicit env
    forever_env = _env_bool("WORKER_FOREVER", default=False)
    forever = False
    if forever_env and max_minutes is None:
        forever = True
    elif runtime_profile == "forever" and max_minutes is None:
        forever = True

    source_controls = _build_source_controls(config)

    print("=== Autonomous Research Engine - Single Agent Mode ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Role: {role}")
    print(f"Runtime profile (requested): {runtime_profile or 'none (use preset default)'}")
    print(f"Max minutes (explicit): {max_minutes if max_minutes is not None else 'None (profile/preset/forever)'}")
    print(f"Stop RYE threshold (explicit): {stop_rye if stop_rye is not None else 'None (preset/profile)'}")
    print(f"Forever mode: {forever}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    sys.stdout.flush()

    _heartbeat(agent, label="worker_single_start")

    # max_cycles here is effectively unbounded; CoreAgent will apply:
    # - runtime profile estimated_cycles hints
    # - preset stop thresholds
    # - global failsafe max_cycles
    summaries = agent.run_continuous(
        goal=goal,
        max_cycles=10_000_000,
        stop_rye=stop_rye,
        role=role,
        source_controls=source_controls,
        pdf_bytes=None,
        biomarker_snapshot=None,
        domain=domain,
        max_minutes=max_minutes,
        forever=forever,
        resume_from_checkpoint=resume,
        watchdog_interval_minutes=watchdog_minutes,
        runtime_profile=runtime_profile,
    )

    _heartbeat(agent, label="worker_single_finished")

    print("=== Continuous run finished cleanly ===")
    print(f"Total completed cycles: {len(summaries)}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Swarm engine
# ---------------------------------------------------------------------------


def run_swarm_engine(agent: CoreAgent, config: Dict[str, Any]) -> None:
    """
    Run a long continuous swarm session.

    Controlled by:
    - WORKER_SWARM_ROLES (comma separated, optional)
    - WORKER_MAX_MINUTES (optional)
    - WORKER_STOP_RYE (optional)
    - WORKER_RUNTIME_PROFILE (optional: 1_hour, 8_hours, 24_hours, 90_days, forever)
    - WORKER_SOURCES (optional, comma separated)
    - WORKER_RESUME (bool, default True)
    - WORKER_WATCHDOG_MINUTES (float, default 5.0)
    - WORKER_FOREVER (bool, default False)
    """
    goal, domain = build_goal_and_domain()

    max_minutes = _env_float("WORKER_MAX_MINUTES")
    stop_rye = _env_float("WORKER_STOP_RYE")
    runtime_profile = os.getenv("WORKER_RUNTIME_PROFILE")
    roles_list = _env_list("WORKER_SWARM_ROLES")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0

    if roles_list is None:
        # Use all configured roles inside CoreAgent (capped at 32)
        roles_list = agent.get_agent_roles()

    # Determine whether to run "forever" based on profile or explicit env
    forever_env = _env_bool("WORKER_FOREVER", default=False)
    forever = False
    if forever_env and max_minutes is None:
        forever = True
    elif runtime_profile == "forever" and max_minutes is None:
        forever = True

    source_controls = _build_source_controls(config)

    print("=== Autonomous Research Engine - Swarm Mode ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Roles: {roles_list}")
    print(f"Runtime profile (requested): {runtime_profile or 'none (use preset default)'}")
    print(f"Max minutes (explicit): {max_minutes if max_minutes is not None else 'None (profile/preset/forever)'}")
    print(f"Stop RYE threshold (explicit): {stop_rye if stop_rye is not None else 'None (preset/profile)'}")
    print(f"Forever mode: {forever}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    sys.stdout.flush()

    _heartbeat(agent, label="worker_swarm_start")

    # max_rounds is effectively unbounded here; CoreAgent will interpret:
    # - runtime profile estimated_cycles into rounds
    # - preset RYE thresholds
    # - internal safety limits
    summaries = agent.run_swarm_continuous(
        goal=goal,
        max_rounds=10_000_000,
        stop_rye=stop_rye,
        roles=roles_list,
        source_controls=source_controls,
        pdf_bytes=None,
        biomarker_snapshot=None,
        domain=domain,
        max_minutes=max_minutes,
        forever=forever,
        resume_from_checkpoint=resume,
        watchdog_interval_minutes=watchdog_minutes,
        runtime_profile=runtime_profile,
    )

    _heartbeat(agent, label="worker_swarm_finished")

    print("=== Swarm run finished cleanly ===")
    print(f"Total summaries produced across all roles and rounds: {len(summaries)}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Entry point for the background worker.

    Mode selection:
    - WORKER_MODE=swarm   -> run swarm engine
    - WORKER_SWARM=1      -> run swarm engine
    - anything else       -> run single agent engine
    """
    print("Starting Autonomous Research Agent background engine...")
    sys.stdout.flush()

    agent, config = init_agent_from_config()

    use_swarm = _env_bool("WORKER_SWARM", default=False)
    mode = os.getenv("WORKER_MODE", "single").strip().lower()

    try:
        if use_swarm or mode == "swarm":
            run_swarm_engine(agent, config)
        else:
            run_single_agent_engine(agent, config)
    except KeyboardInterrupt:
        print("Engine interrupted by user or environment.")
        sys.stdout.flush()
    except Exception:
        print("Fatal error in engine_worker:")
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
