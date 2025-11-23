"""
Background engine worker for the Autonomous Research Agent.

This file is the long-running engine that can run on Render as a
Background Worker or separate service.

Key ideas:
- Uses the same CoreAgent and MemoryStore as the Streamlit UI.
- Can run in three styles:
    1) Classic single agent continuous mode
    2) Classic swarm continuous mode
    3) Meta-controller mode ("Option C") that plans multiple phases:
       - exploration
       - stabilization
       - refinement
       and adapts using RYE and time used.
- Uses environment variables to control goal, domain, runtime, swarm, and meta behavior.
- Relies on CoreAgent presets, checkpoints, and watchdog for crash recovery.
- Supports web browser and sandbox tools via source_controls flags.
- Streams worker_state, run_manifests, milestones, and basic events into MemoryStore
  so the UI and logs have a first class view of long runs.

You can start it with commands like:
    WORKER_GOAL="Long run test on reparodynamics" \
    WORKER_RUNTIME_PROFILE="8_hours" \
    python engine_worker.py

On Render:
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
from agent.rye_metrics import build_run_diagnostics

# Optional tools registry for web browser and sandbox detection
try:
    from agent.tools import TOOL_REGISTRY  # type: ignore[import]
except Exception:  # pragma: no cover
    TOOL_REGISTRY = {}  # type: ignore[assignment]

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


def detect_tools() -> Dict[str, bool]:
    """Detect presence of web browser and sandbox tools from TOOL_REGISTRY."""
    if not isinstance(TOOL_REGISTRY, dict):
        return {"web": False, "sandbox": False}

    web_keys = {"web_search", "browser", "web", "internet"}
    sandbox_keys = {"sandbox", "code_sandbox", "python_sandbox", "exec_sandbox"}

    has_web = any(k in TOOL_REGISTRY for k in web_keys)
    has_sandbox = any(k in TOOL_REGISTRY for k in sandbox_keys)

    return {"web": has_web, "sandbox": has_sandbox}


def _configure_tavily_from_env() -> None:
    """
    Optional Tavily key configuration.

    If WORKER_TAVILY_KEY is set, propagate it to TAVILY_API_KEY so the
    same engine can be used headless without the Streamlit UI.
    """
    key = os.getenv("WORKER_TAVILY_KEY")
    if key:
        os.environ["TAVILY_API_KEY"] = key


def _build_source_controls(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Build source controls for the worker.

    Keys:
        web, pubmed, semantic, pdf, biomarkers, sandbox

    Priority:
    1. WORKER_SOURCES env (comma separated: web,pubmed,semantic,pdf,biomarkers,sandbox)
       If present, only those listed are enabled.
    2. config["default_source_controls"] from YAML (if provided)
    3. Hard coded defaults.
    4. Optional per source overrides:
       WORKER_WEB, WORKER_PUBMED, WORKER_SEMANTIC, WORKER_PDF,
       WORKER_BIOMARKERS, WORKER_SANDBOX.
    5. Final clamp based on TOOL_REGISTRY detection: if sandbox tool is not
       present, sandbox is forced to False.
    """
    # Base defaults
    defaults: Dict[str, bool] = {
        "web": True,
        "pubmed": True,
        "semantic": True,
        "pdf": True,
        "biomarkers": False,
        "sandbox": False,
    }

    cfg_sc = config.get("default_source_controls")
    if isinstance(cfg_sc, dict):
        merged = defaults.copy()
        for k, v in cfg_sc.items():
            merged[str(k)] = bool(v)
        defaults = merged

    env_sources = _env_list("WORKER_SOURCES")
    if env_sources is not None:
        allowed = {s.lower() for s in env_sources}
        sc: Dict[str, bool] = {}
        for key in defaults.keys():
            sc[key] = key.lower() in allowed
    else:
        sc = defaults.copy()

    # Per source env overrides (optional)
    def _override_bool(key: str, env_name: str) -> None:
        raw = os.getenv(env_name)
        if raw is not None:
            sc[key] = _env_bool(env_name, default=sc.get(key, False))

    _override_bool("web", "WORKER_WEB")
    _override_bool("pubmed", "WORKER_PUBMED")
    _override_bool("semantic", "WORKER_SEMANTIC")
    _override_bool("pdf", "WORKER_PDF")
    _override_bool("biomarkers", "WORKER_BIOMARKERS")
    _override_bool("sandbox", "WORKER_SANDBOX")

    # Final clamp based on TOOL_REGISTRY capabilities
    flags = detect_tools()
    if not flags["sandbox"]:
        sc["sandbox"] = False

    # Web clamp is softer: if web tool is not present but Tavily key exists,
    # the CoreAgent can still use Tavily directly.
    # So we only force web off if TOOL_REGISTRY has no web and no Tavily key.
    if not flags["web"]:
        if not os.getenv("TAVILY_API_KEY"):
            sc["web"] = False

    return sc


# ---------------------------------------------------------------------------
# Agent and memory helpers
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


def _get_memory_store(agent: CoreAgent) -> Optional[MemoryStore]:
    """Best effort accessor for the agents MemoryStore."""
    ms = getattr(agent, "memory_store", None)
    if isinstance(ms, MemoryStore):
        return ms
    return None


def _current_run_id(mode: str) -> str:
    """
    Construct a stable run id for this worker process.

    Priority:
        1) WORKER_RUN_ID env
        2) derived from mode and pid
    """
    base = os.getenv("WORKER_RUN_ID")
    if base:
        return base
    return f"{mode}-{os.getpid()}"


def _heartbeat(agent: CoreAgent, label: str, run_id: Optional[str] = None) -> None:
    """
    Best effort heartbeat into the MemoryStore so the UI can see that the
    worker is alive. Silently ignored if heartbeat is not implemented.
    """
    try:
        ms = _get_memory_store(agent)
        if ms is not None and hasattr(ms, "heartbeat"):
            if run_id is not None:
                ms.heartbeat(label=label, run_id=run_id)
            else:
                ms.heartbeat(label=label)
    except Exception:
        # Heartbeat must never crash the worker
        return


def _log_run_manifest(
    agent: CoreAgent,
    run_id: str,
    *,
    mode: str,
    domain: str,
    goal: str,
    runtime_profile: Optional[str],
    stop_rye: Optional[float],
    max_minutes: Optional[float],
    summaries: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Best effort manifest logging into MemoryStore using the new run_manifests section.
    """
    ms = _get_memory_store(agent)
    if ms is None or not hasattr(ms, "log_run_manifest"):
        return

    try:
        diag = build_run_diagnostics(history=summaries, domain=domain, window=10)
    except Exception:
        diag = {}

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "domain": domain,
        "goal": goal,
        "runtime_profile": runtime_profile,
        "stop_rye": stop_rye,
        "max_minutes": max_minutes,
        "total_items": len(summaries),
        "rye_avg": diag.get("rye_avg"),
        "rye_median": diag.get("rye_median"),
        "rye_last": diag.get("rye_last"),
        "stability_index": diag.get("stability_index"),
        "recovery_momentum": diag.get("recovery_momentum"),
    }
    if extra:
        manifest["extra"] = extra

    try:
        ms.log_run_manifest(run_id, manifest)
    except Exception:
        # Manifest logging must not crash the worker
        return


def _log_milestone(
    agent: CoreAgent,
    *,
    run_id: str,
    goal: str,
    domain: str,
    label: str,
    description: str,
    level: str = "info",
    role: Optional[str] = None,
    cycle_index: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Best effort helper to write milestones for long runs and meta segments."""
    ms = _get_memory_store(agent)
    if ms is None or not hasattr(ms, "log_milestone"):
        return
    try:
        ms.log_milestone(
            run_id=run_id,
            goal=goal,
            domain=domain,
            label=label,
            description=description,
            level=level,
            role=role,
            cycle_index=cycle_index,
            extra=extra,
        )
    except Exception:
        return


def _update_worker_state(
    agent: CoreAgent,
    *,
    status: str,
    mode: str,
    goal: str,
    domain: str,
    roles: Optional[List[str]] = None,
    runtime_profile: Optional[str] = None,
    stop_rye: Optional[float] = None,
    max_minutes: Optional[float] = None,
    run_id: Optional[str] = None,
    experiment_mode: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Thin wrapper around MemoryStore.update_worker_state."""
    ms = _get_memory_store(agent)
    if ms is None or not hasattr(ms, "update_worker_state"):
        return
    try:
        ms.update_worker_state(
            status=status,
            mode=mode,
            goal=goal,
            domain=domain,
            roles=roles,
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            run_id=run_id,
            experiment_mode=experiment_mode,
            extra=extra,
        )
    except Exception:
        return


# ---------------------------------------------------------------------------
# Simple classic engines (single and swarm)
# ---------------------------------------------------------------------------


def run_single_agent_engine(agent: CoreAgent, config: Dict[str, Any]) -> None:
    """
    Run a long continuous single agent session.

    Controlled by:
    - WORKER_MAX_MINUTES (optional)
    - WORKER_STOP_RYE (optional)
    - WORKER_RUNTIME_PROFILE (optional: 1_hour, 8_hours, 24_hours, 90_days, forever)
    - WORKER_ROLE (default "agent")
    - WORKER_SOURCES (optional, comma separated)
    - WORKER_RESUME (bool, default True)
    - WORKER_WATCHDOG_MINUTES (float, default 5.0)
    - WORKER_FOREVER (bool, default False)
    - WORKER_WEB, WORKER_SANDBOX, WORKER_PUBMED, WORKER_SEMANTIC, WORKER_PDF, WORKER_BIOMARKERS
    """
    goal, domain = build_goal_and_domain()
    preset_cfg = get_preset(domain)

    max_minutes = _env_float("WORKER_MAX_MINUTES")
    stop_rye = _env_float("WORKER_STOP_RYE")

    runtime_profile_env = os.getenv("WORKER_RUNTIME_PROFILE")
    runtime_profile = runtime_profile_env or preset_cfg.get("default_runtime_profile")

    role = os.getenv("WORKER_ROLE", "agent")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0

    forever_env = _env_bool("WORKER_FOREVER", default=False)
    forever = False
    if forever_env and max_minutes is None:
        forever = True
    elif runtime_profile == "forever" and max_minutes is None:
        forever = True

    # Domain aware source controls: preset defaults feed into config unless YAML overrides them
    config_for_sources = dict(config)
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    tool_flags = detect_tools()
    run_id = _current_run_id("single")

    print("=== Autonomous Research Engine - Single Agent Mode ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Role: {role}")
    print(f"Run id: {run_id}")
    print(f"Runtime profile (env override): {runtime_profile_env or 'None'}")
    print(f"Runtime profile (effective): {runtime_profile or 'None (engine default)'}")
    print(f"Max minutes (explicit): {max_minutes if max_minutes is not None else 'None (profile/preset/forever)'}")
    print(f"Stop RYE threshold (explicit): {stop_rye if stop_rye is not None else 'None (preset/profile)'}")
    print(f"Forever mode: {forever}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    sys.stdout.flush()

    _update_worker_state(
        agent,
        status="starting",
        mode="single",
        goal=goal,
        domain=domain,
        roles=[role],
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=run_id,
        experiment_mode="classic_single",
    )
    _heartbeat(agent, label="worker_single_start", run_id=run_id)

    _update_worker_state(
        agent,
        status="running",
        mode="single",
        goal=goal,
        domain=domain,
        roles=[role],
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=run_id,
        experiment_mode="classic_single",
    )

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

    _heartbeat(agent, label="worker_single_finished", run_id=run_id)

    print("=== Continuous run finished cleanly ===")
    print(f"Total completed cycles: {len(summaries)}")
    try:
        diag = build_run_diagnostics(history=summaries, domain=domain, window=10)
        print(f"RYE avg: {diag.get('rye_avg')}")
        print(f"RYE median: {diag.get('rye_median')}")
        print(f"RYE last: {diag.get('rye_last')}")
        print(f"Stability index: {diag.get('stability_index')}")
        print(f"Recovery momentum: {diag.get('recovery_momentum')}")
        _log_run_manifest(
            agent,
            run_id,
            mode="single",
            domain=domain,
            goal=goal,
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            summaries=summaries,
            extra={"engine": "single"},
        )
    except Exception:
        print("Diagnostics computation failed, see logs for details.")
    sys.stdout.flush()

    _update_worker_state(
        agent,
        status="stopped",
        mode="single",
        goal=goal,
        domain=domain,
        roles=[role],
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=run_id,
        experiment_mode="classic_single",
    )


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
    - WORKER_WEB, WORKER_SANDBOX, WORKER_PUBMED, WORKER_SEMANTIC, WORKER_PDF, WORKER_BIOMARKERS
    """
    goal, domain = build_goal_and_domain()
    preset_cfg = get_preset(domain)

    max_minutes = _env_float("WORKER_MAX_MINUTES")
    stop_rye = _env_float("WORKER_STOP_RYE")

    runtime_profile_env = os.getenv("WORKER_RUNTIME_PROFILE")
    runtime_profile = runtime_profile_env or preset_cfg.get("default_runtime_profile")

    roles_list = _env_list("WORKER_SWARM_ROLES")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0

    if roles_list is None:
        roles_list = agent.get_agent_roles()

    forever_env = _env_bool("WORKER_FOREVER", default=False)
    forever = False
    if forever_env and max_minutes is None:
        forever = True
    elif runtime_profile == "forever" and max_minutes is None:
        forever = True

    config_for_sources = dict(config)
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    tool_flags = detect_tools()
    run_id = _current_run_id("swarm")

    print("=== Autonomous Research Engine - Swarm Mode ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Roles: {roles_list}")
    print(f"Run id: {run_id}")
    print(f"Runtime profile (env override): {runtime_profile_env or 'None'}")
    print(f"Runtime profile (effective): {runtime_profile or 'None (engine default)'}")
    print(f"Max minutes (explicit): {max_minutes if max_minutes is not None else 'None (profile/preset/forever)'}")
    print(f"Stop RYE threshold (explicit): {stop_rye if stop_rye is not None else 'None (preset/profile)'}")
    print(f"Forever mode: {forever}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    sys.stdout.flush()

    _update_worker_state(
        agent,
        status="starting",
        mode="swarm",
        goal=goal,
        domain=domain,
        roles=roles_list,
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=run_id,
        experiment_mode="classic_swarm",
    )
    _heartbeat(agent, label="worker_swarm_start", run_id=run_id)

    _update_worker_state(
        agent,
        status="running",
        mode="swarm",
        goal=goal,
        domain=domain,
        roles=roles_list,
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=run_id,
        experiment_mode="classic_swarm",
    )

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

    _heartbeat(agent, label="worker_swarm_finished", run_id=run_id)

    print("=== Swarm run finished cleanly ===")
    print(f"Total summaries produced across all roles and rounds: {len(summaries)}")
    try:
        diag = build_run_diagnostics(history=summaries, domain=domain, window=10)
        print(f"RYE avg: {diag.get('rye_avg')}")
        print(f"RYE median: {diag.get('rye_median')}")
        print(f"RYE last: {diag.get('rye_last')}")
        print(f"Stability index: {diag.get('stability_index')}")
        print(f"Recovery momentum: {diag.get('recovery_momentum')}")
        _log_run_manifest(
            agent,
            run_id,
            mode="swarm",
            domain=domain,
            goal=goal,
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            summaries=summaries,
            extra={"engine": "swarm", "roles": roles_list},
        )
    except Exception:
        print("Diagnostics computation failed, see logs for details.")
    sys.stdout.flush()

    _update_worker_state(
        agent,
        status="stopped",
        mode="swarm",
        goal=goal,
        domain=domain,
        roles=roles_list,
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=run_id,
        experiment_mode="classic_swarm",
    )


# ---------------------------------------------------------------------------
# Meta-controller engine (Option C)
# ---------------------------------------------------------------------------


def _compute_segment_stats(segment_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Inspect a list of cycle or swarm summaries and compute stats for the meta-controller.

    Uses build_run_diagnostics so segment stats are consistent with the UI.
    Returns:
        {
            "count": int,
            "avg_rye": Optional[float],
            "max_rye": Optional[float],
            "min_rye": Optional[float],
            "last_rye": Optional[float],
            "stability": Optional[float],
            "momentum": Optional[float],
        }
    """
    if not segment_summaries:
        return {
            "count": 0,
            "avg_rye": None,
            "max_rye": None,
            "min_rye": None,
            "last_rye": None,
            "stability": None,
            "momentum": None,
        }

    diag = build_run_diagnostics(history=segment_summaries, domain=None, window=10)
    rye_vals: List[float] = [
        float(e["RYE"]) for e in segment_summaries if isinstance(e.get("RYE"), (int, float))
    ]
    if not rye_vals:
        return {
            "count": len(segment_summaries),
            "avg_rye": None,
            "max_rye": None,
            "min_rye": None,
            "last_rye": None,
            "stability": diag.get("stability_index"),
            "momentum": diag.get("recovery_momentum"),
        }

    return {
        "count": len(segment_summaries),
        "avg_rye": diag.get("rye_avg"),
        "max_rye": max(rye_vals),
        "min_rye": min(rye_vals),
        "last_rye": rye_vals[-1],
        "stability": diag.get("stability_index"),
        "momentum": diag.get("recovery_momentum"),
    }


def _initial_meta_plan(
    goal: str,
    domain: str,
    preferred_mode: str,
    total_budget_minutes: Optional[float],
) -> Dict[str, Any]:
    """
    Create an initial meta plan with three conceptual phases:
        1) exploration (usually swarm)
        2) stabilization (usually single)
        3) refinement (small targeted segment)

    The exact minutes per phase are distributed from total_budget_minutes.
    If no total budget is given, defaults to a 60 minute plan.
    """
    if total_budget_minutes is None or total_budget_minutes <= 0:
        total_budget_minutes = 60.0

    explore_min = max(5.0, total_budget_minutes * 0.4)
    stabilize_min = max(5.0, total_budget_minutes * 0.4)
    refine_min = max(5.0, total_budget_minutes * 0.2)

    if explore_min + stabilize_min + refine_min > total_budget_minutes:
        scale = total_budget_minutes / (explore_min + stabilize_min + refine_min)
        explore_min *= scale
        stabilize_min *= scale
        refine_min *= scale

    if preferred_mode not in {"single", "swarm"}:
        preferred_mode = "swarm"

    if domain.lower() == "longevity":
        explore_profile = "1_hour"
        stabilize_profile = "8_hours"
        refine_profile = "1_hour"
    elif domain.lower() == "math":
        explore_profile = "1_hour"
        stabilize_profile = "8_hours"
        refine_profile = "8_hours"
    else:
        explore_profile = "1_hour"
        stabilize_profile = "8_hours"
        refine_profile = "1_hour"

    plan = {
        "goal": goal,
        "domain": domain,
        "total_budget_minutes": total_budget_minutes,
        "phases": [
            {
                "name": "exploration",
                "mode": "swarm" if preferred_mode == "swarm" else "single",
                "runtime_profile": explore_profile,
                "target_minutes": explore_min,
                "min_minutes": max(5.0, explore_min * 0.5),
                "max_minutes": explore_min * 1.5,
                "base_stop_rye": 0.02,
            },
            {
                "name": "stabilization",
                "mode": "single",
                "runtime_profile": stabilize_profile,
                "target_minutes": stabilize_min,
                "min_minutes": max(5.0, stabilize_min * 0.5),
                "max_minutes": stabilize_min * 1.5,
                "base_stop_rye": 0.05,
            },
            {
                "name": "refinement",
                "mode": preferred_mode,
                "runtime_profile": refine_profile,
                "target_minutes": refine_min,
                "min_minutes": max(5.0, refine_min * 0.3),
                "max_minutes": refine_min * 1.5,
                "base_stop_rye": 0.08,
            },
        ],
    }
    return plan


def _adjust_phase_from_stats(
    phase_cfg: Dict[str, Any],
    stats: Dict[str, Any],
    recent_avg_rye: Optional[float],
) -> Tuple[float, Optional[float]]:
    """
    Given a phase configuration and recent RYE behavior, choose:
        - effective_minutes for the next segment
        - effective_stop_rye for the next segment

    Rules of thumb:
        - If RYE is very low or trending down, shorten the next segment and lower stop_rye.
        - If RYE is healthy or trending up, keep or extend the next segment and raise stop_rye a bit.
    """
    _ = stats  # placeholder for future use
    base_target = float(phase_cfg.get("target_minutes", 20.0))
    min_minutes = float(phase_cfg.get("min_minutes", 5.0))
    max_minutes = float(phase_cfg.get("max_minutes", base_target))
    base_stop_rye = phase_cfg.get("base_stop_rye")
    effective_stop_rye: Optional[float] = None

    if recent_avg_rye is None:
        effective_minutes = base_target
        effective_stop_rye = base_stop_rye
    else:
        if recent_avg_rye < 0.01:
            effective_minutes = max(min_minutes, base_target * 0.5)
            effective_stop_rye = None
        elif recent_avg_rye < 0.05:
            effective_minutes = base_target
            effective_stop_rye = (base_stop_rye or 0.03) * 0.8
        elif recent_avg_rye < 0.15:
            effective_minutes = base_target
            effective_stop_rye = (base_stop_rye or 0.05) * 1.1
        else:
            effective_minutes = min(max_minutes, base_target * 1.2)
            effective_stop_rye = (base_stop_rye or 0.08) * 1.25

    if effective_minutes < min_minutes:
        effective_minutes = min_minutes
    if effective_minutes > max_minutes:
        effective_minutes = max_minutes

    return effective_minutes, effective_stop_rye


def run_meta_engine(agent: CoreAgent, config: Dict[str, Any]) -> None:
    """
    Meta-controller engine (Option C).

    Instead of one giant continuous call, this orchestrates multiple segments:

        Phase 1: exploration (usually swarm)
        Phase 2: stabilization (usually single)
        Phase 3: refinement (mode depends on domain and env)

    Between segments it:
        - reads RYE behavior
        - adjusts segment duration and stop_rye
        - can stop early if time is almost exhausted or RYE has clearly collapsed

    Controlled by:
    - WORKER_MAX_MINUTES (overall macro budget)
    - WORKER_META_MAX_SEGMENTS (max number of segments, default 6)
    - WORKER_RUNTIME_PROFILE (hint for phase profiles)
    - WORKER_MODE / WORKER_SWARM (preferred mode: single vs swarm)
    - WORKER_SOURCES etc as in classic engines
    """
    goal, domain = build_goal_and_domain()
    preset_cfg = get_preset(domain)

    total_budget_minutes = _env_float("WORKER_MAX_MINUTES")
    meta_max_segments = _env_float("WORKER_META_MAX_SEGMENTS")
    if meta_max_segments is None or meta_max_segments <= 0:
        meta_max_segments = 6.0
    meta_max_segments_int = int(meta_max_segments)

    use_swarm_flag = _env_bool("WORKER_SWARM", default=False)
    mode_env = os.getenv("WORKER_MODE", "single").strip().lower()
    preferred_mode = "swarm" if (use_swarm_flag or mode_env == "swarm") else "single"

    runtime_profile_env = os.getenv("WORKER_RUNTIME_PROFILE")
    stop_rye_env = _env_float("WORKER_STOP_RYE")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0
    forever_env = _env_bool("WORKER_FOREVER", default=False)

    if total_budget_minutes is None:
        if runtime_profile_env == "1_hour":
            total_budget_minutes = 60.0
        elif runtime_profile_env == "8_hours":
            total_budget_minutes = 8 * 60.0
        elif runtime_profile_env == "24_hours":
            total_budget_minutes = 24 * 60.0
        elif runtime_profile_env == "90_days":
            total_budget_minutes = 90 * 24 * 60.0
        elif runtime_profile_env == "forever" or forever_env:
            total_budget_minutes = 24 * 60.0
        else:
            total_budget_minutes = float(preset_cfg.get("runtime_minutes", 60.0))

    if total_budget_minutes <= 0:
        total_budget_minutes = 60.0

    # Domain aware source controls here as well
    config_for_sources = dict(config)
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    tool_flags = detect_tools()
    run_id = _current_run_id("meta")

    print("=== Autonomous Research Engine - Meta Controller Mode (Option C) ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Run id: {run_id}")
    print(f"Preferred mode: {preferred_mode}")
    print(f"Total macro budget (minutes): {total_budget_minutes}")
    print(f"Max meta segments: {meta_max_segments_int}")
    print(f"Runtime profile hint (env): {runtime_profile_env or 'none'}")
    print(f"Explicit stop RYE (env): {stop_rye_env if stop_rye_env is not None else 'None'}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    sys.stdout.flush()

    _update_worker_state(
        agent,
        status="starting",
        mode="meta",
        goal=goal,
        domain=domain,
        roles=None,
        runtime_profile=runtime_profile_env,
        stop_rye=stop_rye_env,
        max_minutes=total_budget_minutes,
        run_id=run_id,
        experiment_mode="meta_controller",
    )
    _heartbeat(agent, label="worker_meta_start", run_id=run_id)

    meta_plan = _initial_meta_plan(
        goal=goal,
        domain=domain,
        preferred_mode=preferred_mode,
        total_budget_minutes=total_budget_minutes,
    )

    segments_run: List[Dict[str, Any]] = []
    recent_avg_rye: Optional[float] = None
    total_elapsed = 0.0

    roles_env = _env_list("WORKER_SWARM_ROLES")
    if roles_env is None:
        roles_for_swarm: Sequence[str] = agent.get_agent_roles()
    else:
        roles_for_swarm = roles_env

    _update_worker_state(
        agent,
        status="running",
        mode="meta",
        goal=goal,
        domain=domain,
        roles=list(roles_for_swarm),
        runtime_profile=runtime_profile_env,
        stop_rye=stop_rye_env,
        max_minutes=total_budget_minutes,
        run_id=run_id,
        experiment_mode="meta_controller",
    )

    for seg_index in range(meta_max_segments_int):
        time_left = total_budget_minutes - total_elapsed
        if time_left <= 1.0:
            print(f"[Meta] Time almost exhausted, stopping before segment {seg_index + 1}.")
            break

        if seg_index < len(meta_plan["phases"]):
            phase_cfg = meta_plan["phases"][seg_index]
        else:
            phase_cfg = meta_plan["phases"][-1]

        phase_name = phase_cfg.get("name", f"segment_{seg_index + 1}")
        phase_mode = phase_cfg.get("mode", preferred_mode)
        phase_profile = phase_cfg.get("runtime_profile")

        if runtime_profile_env and phase_profile is None:
            phase_profile = runtime_profile_env

        effective_minutes, effective_stop_rye = _adjust_phase_from_stats(
            phase_cfg=phase_cfg,
            stats={"count": 0},
            recent_avg_rye=recent_avg_rye,
        )

        if effective_minutes > time_left:
            effective_minutes = time_left

        if stop_rye_env is not None:
            effective_stop_rye = stop_rye_env

        print("")
        print(f"[Meta] Starting segment {seg_index + 1} / {meta_max_segments_int}")
        print(f"[Meta] Phase: {phase_name}")
        print(f"[Meta] Mode: {phase_mode}")
        print(f"[Meta] Segment minutes (requested): {effective_minutes:.2f}")
        print(f"[Meta] Time left after this segment (approx): {time_left - effective_minutes:.2f}")
        print(f"[Meta] Segment stop RYE (auto/explicit): {effective_stop_rye if effective_stop_rye is not None else 'None'}")
        print(f"[Meta] Phase runtime profile: {phase_profile or 'preset default'}")
        sys.stdout.flush()

        if phase_mode == "swarm":
            segment_summaries = agent.run_swarm_continuous(
                goal=goal,
                max_rounds=10_000_000,
                stop_rye=effective_stop_rye,
                roles=roles_for_swarm,
                source_controls=source_controls,
                pdf_bytes=None,
                biomarker_snapshot=None,
                domain=domain,
                max_minutes=effective_minutes,
                forever=False,
                resume_from_checkpoint=resume,
                watchdog_interval_minutes=watchdog_minutes,
                runtime_profile=phase_profile,
            )
        else:
            segment_summaries = agent.run_continuous(
                goal=goal,
                max_cycles=10_000_000,
                stop_rye=effective_stop_rye,
                role=os.getenv("WORKER_ROLE", "agent"),
                source_controls=source_controls,
                pdf_bytes=None,
                biomarker_snapshot=None,
                domain=domain,
                max_minutes=effective_minutes,
                forever=False,
                resume_from_checkpoint=resume,
                watchdog_interval_minutes=watchdog_minutes,
                runtime_profile=phase_profile,
            )

        segments_run.append(
            {
                "phase": phase_name,
                "mode": phase_mode,
                "runtime_profile": phase_profile,
                "minutes_requested": effective_minutes,
                "summaries": segment_summaries,
            }
        )

        seg_stats = _compute_segment_stats(segment_summaries)
        seg_avg_rye = seg_stats["avg_rye"]
        if seg_avg_rye is not None:
            if recent_avg_rye is None:
                recent_avg_rye = seg_avg_rye
            else:
                recent_avg_rye = 0.6 * recent_avg_rye + 0.4 * seg_avg_rye

        segment_elapsed = 0.0
        if segment_summaries:
            meta = segment_summaries[-1].get("run_metadata")
            if isinstance(meta, dict):
                em = meta.get("elapsed_minutes")
                if isinstance(em, (int, float)):
                    segment_elapsed = float(em)

        if segment_elapsed <= 0.0:
            segment_elapsed = effective_minutes

        total_elapsed += segment_elapsed

        print(f"[Meta] Segment {seg_index + 1} stats:")
        print(f"        cycles/summaries: {seg_stats['count']}")
        print(f"        avg RYE: {seg_stats['avg_rye']}")
        print(f"        best RYE: {seg_stats['max_rye']}")
        print(f"        stability: {seg_stats['stability']}")
        print(f"        momentum: {seg_stats['momentum']}")
        print(f"        smoothed recent RYE: {recent_avg_rye}")
        print(f"        total elapsed minutes: {total_elapsed:.2f} / {total_budget_minutes:.2f}")
        sys.stdout.flush()

        _log_milestone(
            agent,
            run_id=run_id,
            goal=goal,
            domain=domain,
            label=f"meta_segment_{seg_index + 1}",
            description=(
                f"Phase {phase_name} ({phase_mode}) finished with avg RYE {seg_stats['avg_rye']} "
                f"and stability {seg_stats['stability']}."
            ),
            level="info",
            extra={
                "segment_index": seg_index + 1,
                "phase_mode": phase_mode,
                "runtime_profile": phase_profile,
                "minutes_requested": effective_minutes,
                "segment_stats": seg_stats,
            },
        )

        _heartbeat(agent, label="worker_meta_segment", run_id=run_id)

        if recent_avg_rye is not None and recent_avg_rye < 0.01 and total_elapsed > total_budget_minutes * 0.6:
            print("[Meta] RYE collapsed and most of the budget is used. Stopping early.")
            break

    _heartbeat(agent, label="worker_meta_finished", run_id=run_id)

    total_segments = len(segments_run)
    total_summaries = sum(len(seg["summaries"]) for seg in segments_run)
    print("")
    print("=== Meta controller run finished ===")
    print(f"Segments executed: {total_segments}")
    print(f"Total summaries across segments: {total_summaries}")
    print(f"Final smoothed recent RYE: {recent_avg_rye}")
    print(f"Total elapsed minutes (approx): {total_elapsed:.2f} / {total_budget_minutes:.2f}")
    sys.stdout.flush()

    # Build a combined history for manifest level diagnostics
    combined_history: List[Dict[str, Any]] = []
    for seg in segments_run:
        combined_history.extend(seg.get("summaries", []))

    _log_run_manifest(
        agent,
        run_id,
        mode="meta",
        domain=domain,
        goal=goal,
        runtime_profile=runtime_profile_env,
        stop_rye=stop_rye_env,
        max_minutes=total_budget_minutes,
        summaries=combined_history,
        extra={
            "engine": "meta",
            "segments": [
                {
                    "phase": seg.get("phase"),
                    "mode": seg.get("mode"),
                    "runtime_profile": seg.get("runtime_profile"),
                    "minutes_requested": seg.get("minutes_requested"),
                    "segment_items": len(seg.get("summaries", [])),
                }
                for seg in segments_run
            ],
        },
    )

    _update_worker_state(
        agent,
        status="stopped",
        mode="meta",
        goal=goal,
        domain=domain,
        roles=list(roles_for_swarm),
        runtime_profile=runtime_profile_env,
        stop_rye=stop_rye_env,
        max_minutes=total_budget_minutes,
        run_id=run_id,
        experiment_mode="meta_controller",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Entry point for the background worker.

    Mode selection:
    - WORKER_META=1          -> run meta controller (Option C, default)
    - WORKER_META=0          -> classic behavior
        - WORKER_MODE=swarm  or WORKER_SWARM=1   -> classic swarm engine
        - anything else                           -> classic single agent engine
    """
    print("Starting Autonomous Research Agent background engine...")
    sys.stdout.flush()

    _configure_tavily_from_env()
    agent, config = init_agent_from_config()

    use_swarm = _env_bool("WORKER_SWARM", default=False)
    mode = os.getenv("WORKER_MODE", "single").strip().lower()
    use_meta = _env_bool("WORKER_META", default=True)

    try:
        if use_meta:
            run_meta_engine(agent, config)
        else:
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
