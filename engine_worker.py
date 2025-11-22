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
# Simple classic engines (single and swarm)
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
# Meta-controller engine (Option C)
# ---------------------------------------------------------------------------

def _compute_segment_stats(segment_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Inspect a list of cycle or swarm summaries and compute simple stats for the meta-controller.

    Returns:
        {
            "count": int,
            "avg_rye": Optional[float],
            "max_rye": Optional[float],
            "min_rye": Optional[float],
            "last_rye": Optional[float],
        }
    """
    ryes: List[float] = []
    for s in segment_summaries:
        rv = s.get("RYE")
        if isinstance(rv, (int, float)):
            ryes.append(float(rv))

    if not ryes:
        return {
            "count": len(segment_summaries),
            "avg_rye": None,
            "max_rye": None,
            "min_rye": None,
            "last_rye": None,
        }

    return {
        "count": len(segment_summaries),
        "avg_rye": sum(ryes) / len(ryes),
        "max_rye": max(ryes),
        "min_rye": min(ryes),
        "last_rye": ryes[-1],
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

    # Rough split: 40 percent explore, 40 percent stabilize, 20 percent refine
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

    # Domain informed default profiles
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
    base_target = float(phase_cfg.get("target_minutes", 20.0))
    min_minutes = float(phase_cfg.get("min_minutes", 5.0))
    max_minutes = float(phase_cfg.get("max_minutes", base_target))
    base_stop_rye = phase_cfg.get("base_stop_rye")
    effective_stop_rye: Optional[float] = None

    if recent_avg_rye is None:
        # No RYE data yet
        effective_minutes = base_target
        effective_stop_rye = base_stop_rye
    else:
        if recent_avg_rye < 0.01:
            # Very low efficiency: short diagnostic segment
            effective_minutes = max(min_minutes, base_target * 0.5)
            effective_stop_rye = None
        elif recent_avg_rye < 0.05:
            # Low but non-zero: moderate segment, low stop threshold
            effective_minutes = base_target
            effective_stop_rye = (base_stop_rye or 0.03) * 0.8
        elif recent_avg_rye < 0.15:
            # Healthy zone: keep base length, slightly higher threshold
            effective_minutes = base_target
            effective_stop_rye = (base_stop_rye or 0.05) * 1.1
        else:
            # Very strong RYE: extend slightly, more demanding threshold
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

    total_budget_minutes = _env_float("WORKER_MAX_MINUTES")
    meta_max_segments = _env_float("WORKER_META_MAX_SEGMENTS")
    if meta_max_segments is None or meta_max_segments <= 0:
        meta_max_segments = 6.0  # float, we cast to int when iterating
    meta_max_segments_int = int(meta_max_segments)

    # Preferred top level mode
    use_swarm_flag = _env_bool("WORKER_SWARM", default=False)
    mode_env = os.getenv("WORKER_MODE", "single").strip().lower()
    preferred_mode = "swarm" if (use_swarm_flag or mode_env == "swarm") else "single"

    # Runtime profile is used as a hint; meta plan still applies
    runtime_profile_env = os.getenv("WORKER_RUNTIME_PROFILE")
    stop_rye_env = _env_float("WORKER_STOP_RYE")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0
    forever_env = _env_bool("WORKER_FOREVER", default=False)

    # Meta engine has a hard time cap: either WORKER_MAX_MINUTES or derived from profile or preset
    preset_cfg = get_preset(domain)
    if total_budget_minutes is None:
        # Use profile hint if present
        if runtime_profile_env == "1_hour":
            total_budget_minutes = 60.0
        elif runtime_profile_env == "8_hours":
            total_budget_minutes = 8 * 60.0
        elif runtime_profile_env == "24_hours":
            total_budget_minutes = 24 * 60.0
        elif runtime_profile_env == "90_days":
            total_budget_minutes = 90 * 24 * 60.0
        elif runtime_profile_env == "forever" or forever_env:
            # If explicitly forever but no minutes, pick a large but finite default
            total_budget_minutes = 24 * 60.0
        else:
            # Fall back to a 60 minute meta plan if nothing else is given
            total_budget_minutes = 60.0

    if total_budget_minutes <= 0:
        total_budget_minutes = 60.0

    source_controls = _build_source_controls(config)

    print("=== Autonomous Research Engine - Meta Controller Mode (Option C) ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Preferred mode: {preferred_mode}")
    print(f"Total macro budget (minutes): {total_budget_minutes}")
    print(f"Max meta segments: {meta_max_segments_int}")
    print(f"Runtime profile hint: {runtime_profile_env or 'none (use defaults)'}")
    print(f"Explicit stop RYE (env): {stop_rye_env if stop_rye_env is not None else 'None'}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    sys.stdout.flush()

    _heartbeat(agent, label="worker_meta_start")

    meta_plan = _initial_meta_plan(
        goal=goal,
        domain=domain,
        preferred_mode=preferred_mode,
        total_budget_minutes=total_budget_minutes,
    )

    segments_run: List[Dict[str, Any]] = []
    recent_avg_rye: Optional[float] = None
    total_elapsed = 0.0

    # Roles for swarm segments: either env override or agent default
    roles_env = _env_list("WORKER_SWARM_ROLES")
    if roles_env is None:
        roles_for_swarm: Sequence[str] = agent.get_agent_roles()
    else:
        roles_for_swarm = roles_env

    for seg_index in range(meta_max_segments_int):
        time_left = total_budget_minutes - total_elapsed
        if time_left <= 1.0:
            print(f"[Meta] Time almost exhausted, stopping before segment {seg_index + 1}.")
            break

        # Pick phase based on segment index
        if seg_index < len(meta_plan["phases"]):
            phase_cfg = meta_plan["phases"][seg_index]
        else:
            # After all phases, loop with refinement behavior
            phase_cfg = meta_plan["phases"][-1]

        phase_name = phase_cfg.get("name", f"segment_{seg_index + 1}")
        phase_mode = phase_cfg.get("mode", preferred_mode)
        phase_profile = phase_cfg.get("runtime_profile")

        # If the user gave a runtime profile env, let it override unless phase explicitly set something else
        if runtime_profile_env and phase_profile is None:
            phase_profile = runtime_profile_env

        effective_minutes, effective_stop_rye = _adjust_phase_from_stats(
            phase_cfg=phase_cfg,
            stats={"count": 0},  # current segment has no stats yet
            recent_avg_rye=recent_avg_rye,
        )

        # Never exceed time_left
        if effective_minutes > time_left:
            effective_minutes = time_left

        # If user explicitly set a stop_rye env, that dominates the automatic one
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

        # Compute stats for this segment and update meta state
        seg_stats = _compute_segment_stats(segment_summaries)
        seg_avg_rye = seg_stats["avg_rye"]
        if seg_avg_rye is not None:
            if recent_avg_rye is None:
                recent_avg_rye = seg_avg_rye
            else:
                # Exponential smoothing
                recent_avg_rye = 0.6 * recent_avg_rye + 0.4 * seg_avg_rye

        # Read elapsed_minutes from run_metadata if available
        segment_elapsed = 0.0
        if segment_summaries:
            meta = segment_summaries[-1].get("run_metadata")
            if isinstance(meta, dict):
                em = meta.get("elapsed_minutes")
                if isinstance(em, (int, float)):
                    segment_elapsed = float(em)

        # Fallback if metadata is missing
        if segment_elapsed <= 0.0:
            segment_elapsed = effective_minutes

        total_elapsed += segment_elapsed

        print(f"[Meta] Segment {seg_index + 1} stats:")
        print(f"        cycles/summaries: {seg_stats['count']}")
        print(f"        avg RYE: {seg_stats['avg_rye']}")
        print(f"        best RYE: {seg_stats['max_rye']}")
        print(f"        smoothed recent RYE: {recent_avg_rye}")
        print(f"        total elapsed minutes: {total_elapsed:.2f} / {total_budget_minutes:.2f}")
        sys.stdout.flush()

        _heartbeat(agent, label="worker_meta_segment")

        # Predictive stopping: if RYE has been very low and time used is high, stop early
        if recent_avg_rye is not None and recent_avg_rye < 0.01 and total_elapsed > total_budget_minutes * 0.6:
            print("[Meta] RYE collapsed and most of the budget is used. Stopping early.")
            break

    _heartbeat(agent, label="worker_meta_finished")

    # Final meta summary to stdout
    total_segments = len(segments_run)
    total_summaries = sum(len(seg["summaries"]) for seg in segments_run)
    print("")
    print("=== Meta controller run finished ===")
    print(f"Segments executed: {total_segments}")
    print(f"Total summaries across segments: {total_summaries}")
    print(f"Final smoothed recent RYE: {recent_avg_rye}")
    print(f"Total elapsed minutes (approx): {total_elapsed:.2f} / {total_budget_minutes:.2f}")
    sys.stdout.flush()


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
