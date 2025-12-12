"""
Background engine worker for the Autonomous Research Agent.

This file is the long-running engine that can run on Render as a
Background Worker or separate service.

Key ideas:
- Uses the same CoreAgent and MemoryStore as the Streamlit UI.
- Can run in three styles:
    1) Single agent engine mode (finite-only)
    2) Swarm engine mode (finite-only)
    3) Meta-controller mode ("Option C") that plans multiple phases:
       - exploration
       - stabilization
       - refinement
       and adapts using RYE and time used (also finite-only).

Finite-only mode:
- "Forever" runs are disabled. All engines must have finite limits:
  * Single agent: bounded by WORKER_MAX_CYCLES and/or WORKER_MAX_MINUTES
  * Swarm: bounded by WORKER_MAX_ROUNDS and/or WORKER_MAX_MINUTES
  * Meta controller: bounded by WORKER_MAX_MINUTES or derived budget
- Runtime profiles are treated as hints for internal tuning only, not as
  permission to run forever.

Queue mode using agent/run_jobs.py:
    - WORKER_QUEUE_MODE=1 (default) -> file-based job queue
    - Worker polls the job queue via agent.run_jobs, executes jobs,
      and writes results via save_job_result / mark_job_error.

You can start it with commands like:
    WORKER_GOAL="Long run test on reparodynamics" \
    WORKER_RUNTIME_PROFILE="8_hours" \
    WORKER_MAX_MINUTES=480 \
    python engine_worker.py

On Render:
    startCommand: python engine_worker.py

Hard safety caps:
- All requested cycles / rounds / minutes are clamped to hard maximums:
    WORKER_HARD_MAX_CYCLES   (default 1,000,000)
    WORKER_HARD_MAX_ROUNDS   (default same as HARD_MAX_CYCLES)
    WORKER_HARD_MAX_MINUTES  (default 1,440 minutes = 24 hours)
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

import yaml

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore
from agent.presets import PRESETS, get_preset  # PRESETS for domain defaults
from agent.rye_metrics import build_run_diagnostics

# Optional tools registry for web browser and sandbox detection
try:
    from agent.tools import TOOL_REGISTRY  # type: ignore[import]
except Exception:  # pragma: no cover
    TOOL_REGISTRY = {}  # type: ignore[assignment]

# File-based run queue integration
try:
    from agent.run_jobs import (
        RunJob,
        load_next_pending_job,
        save_job_result,
        mark_job_error,
        progress_path,
    )
except Exception:
    # If run_jobs is not present, queue mode will not work
    RunJob = None  # type: ignore[assignment]
    load_next_pending_job = None  # type: ignore[assignment]
    save_job_result = None  # type: ignore[assignment]
    mark_job_error = None  # type: ignore[assignment]
    progress_path = None  # type: ignore[assignment]

CONFIG_PATH_DEFAULT = "config/settings.yaml"

# Base folder for runs for worker logs and queue fallbacks.
# This must match agent.run_jobs BASE_DIR which also respects ARA_RUNS_DIR.
BASE_DIR = Path(os.environ.get("ARA_RUNS_DIR", "runs"))

# If agent.run_jobs defines its own BASE_DIR, prefer that so the worker,
# Streamlit app, and queue layer are guaranteed to point at the same place.
try:
    import agent.run_jobs as _run_jobs_mod  # type: ignore[import]

    _rj_base_dir = getattr(_run_jobs_mod, "BASE_DIR", None)
    if _rj_base_dir is not None:
        BASE_DIR = _rj_base_dir  # type: ignore[assignment]
except Exception:
    pass

# NOTE: BASE_DIR is now always derived from agent.run_jobs.BASE_DIR when available,
# so engine_worker, Streamlit, and the queue see the same runs directory tree.


# ---------------------------------------------------------------------------
# Hard safety caps (finite-only guard rails)
# ---------------------------------------------------------------------------


def _parse_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    try:
        v = int(val)
        if v <= 0:
            return default
        return v
    except Exception:
        return default


def _parse_float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    if not val:
        return default
    try:
        v = float(val)
        if v <= 0:
            return default
        return v
    except Exception:
        return default


HARD_MAX_CYCLES: int = _parse_int_env("WORKER_HARD_MAX_CYCLES", 1_000_000)
HARD_MAX_ROUNDS: int = _parse_int_env("WORKER_HARD_MAX_ROUNDS", HARD_MAX_CYCLES)
HARD_MAX_MINUTES: float = _parse_float_env("WORKER_HARD_MAX_MINUTES", 1440.0)


def _clamp_int(value: int, hard_max: int, label: str) -> int:
    if value > hard_max:
        print(
            f"[Safety] {label} requested {value} exceeds hard max {hard_max}. "
            f"Clamping to {hard_max}."
        )
        sys.stdout.flush()
        return hard_max
    if value <= 0:
        print(f"[Safety] {label} requested {value} is non-positive. Using 1.")
        sys.stdout.flush()
        return 1
    return value


def _clamp_minutes(value: Optional[float], label: str) -> Optional[float]:
    if value is None:
        return None
    if value > HARD_MAX_MINUTES:
        print(
            f"[Safety] {label} minutes {value} exceeds hard max {HARD_MAX_MINUTES}. "
            f"Clamping to {HARD_MAX_MINUTES}."
        )
        sys.stdout.flush()
        return HARD_MAX_MINUTES
    if value <= 0:
        print(f"[Safety] {label} minutes {value} is non-positive. Ignoring.")
        sys.stdout.flush()
        return None
    return value


# ---------------------------------------------------------------------------
# Config and environment helpers
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
    """
    Detect presence of web browser and sandbox tools from TOOL_REGISTRY.

    Web detection includes:
    - Generic browser style tools: web_search, browser, web, internet
    - Tavily based tools: tavily_search
    - Option C web wrapper: extreme_web_search (if you registered it with that name)
    """
    if not isinstance(TOOL_REGISTRY, dict):
        return {"web": False, "sandbox": False}

    web_keys = {
        "web_search",
        "browser",
        "web",
        "internet",
        "tavily_search",
        "extreme_web_search",
    }
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

    flags = detect_tools()
    if not flags["sandbox"]:
        sc["sandbox"] = False

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

    memory_file = os.getenv(
        "WORKER_MEMORY_FILE",
        config.get("memory_file", "logs/sessions/default_memory.json"),
    )
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
        return


def _safe_agent_hook(agent: CoreAgent, hook_name: str, **kwargs: Any) -> Optional[Any]:
    """
    Best effort call into optional intelligence hooks on CoreAgent.

    This allows us to integrate learning, discovery, and verification hooks
    without requiring them to exist or changing CoreAgent's contract.
    """
    fn = getattr(agent, hook_name, None)
    if not callable(fn):
        return None
    try:
        return fn(**kwargs)
    except Exception:
        return None


def _build_experiment_fingerprint(
    *,
    goal: str,
    domain: str,
    mode: str,
    runtime_profile: Optional[str],
    source_controls: Dict[str, bool],
    roles: Optional[Sequence[str]] = None,
    env_keys: Optional[Sequence[str]] = None,
) -> str:
    """
    Build a reproducibility fingerprint for this run configuration.

    This does not change behavior, it just records a hash of configuration
    so long runs can be compared or replicated later.
    """
    payload: Dict[str, Any] = {
        "goal": goal,
        "domain": domain,
        "mode": mode,
        "runtime_profile": runtime_profile,
        "source_controls": dict(sorted(source_controls.items())),
    }
    if roles is not None:
        payload["roles"] = sorted(list(roles))

    env_sample: Dict[str, str] = {}
    if env_keys is not None:
        for name in env_keys:
            val = os.getenv(name)
            if val is not None:
                env_sample[name] = val
    if env_sample:
        payload["env"] = env_sample

    try:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return "unknown"


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


def _write_cycles_and_run_state(
    agent: CoreAgent,
    *,
    run_id: str,
    mode: str,
    goal: str,
    domain: str,
    cycles: List[Dict[str, Any]],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Best effort helper so diagnostics and reports see:
      - cycle history
      - run_state snapshot
    This writes once at run end using the full cycle list.
    """
    ms = _get_memory_store(agent)
    if ms is None:
        return

    diag_local = diagnostics or {}

    # Cycle history
    try:
        if hasattr(ms, "write_cycle_history"):
            # Preferred: single call with full history
            ms.write_cycle_history(run_id, cycles)  # type: ignore[arg-type]
        elif hasattr(ms, "append_cycle_log"):
            # Fallback: append one by one
            for idx, c in enumerate(cycles):
                try:
                    ms.append_cycle_log(run_id, c, index=idx)  # type: ignore[arg-type]
                except TypeError:
                    ms.append_cycle_log(run_id, c)  # type: ignore[arg-type]
        elif hasattr(ms, "append_cycle_history"):
            for idx, c in enumerate(cycles):
                try:
                    ms.append_cycle_history(run_id, c, index=idx)  # type: ignore[arg-type]
                except TypeError:
                    ms.append_cycle_history(run_id, c)  # type: ignore[arg-type]
    except Exception:
        # Never break the worker from logging issues
        pass

    # Run state
    try:
        if hasattr(ms, "save_run_state"):
            state: Dict[str, Any] = {
                "run_id": run_id,
                "mode": mode,
                "goal": goal,
                "domain": domain,
                "total_cycles": len(cycles),
                "diagnostics": diag_local,
                "last_update_utc": datetime.utcnow().isoformat() + "Z",
            }
            try:
                ms.save_run_state(run_id, state)  # type: ignore[arg-type]
            except TypeError:
                ms.save_run_state(run_id=run_id, state=state)  # type: ignore[arg-type]
    except Exception:
        pass


def _run_post_run_intelligence(
    agent: CoreAgent,
    *,
    mode: str,
    goal: str,
    domain: str,
    run_id: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Optional learning, discovery, and verification pass after a run.

    This function is strictly additive:
    - It never changes run control flow.
    - It only calls hooks if they exist on CoreAgent.
    - It logs a milestone summarizing any returned intelligence.
    """
    info: Dict[str, Any] = {}
    hooks = [
        ("learn_from_run", "learning"),
        ("run_discovery_pass", "discovery"),
        ("run_verification_pass", "verification"),
    ]

    for hook_name, label in hooks:
        result = _safe_agent_hook(
            agent,
            hook_name,
            history=history,
            mode=mode,
            goal=goal,
            domain=domain,
            run_id=run_id,
        )
        if result is not None:
            info[label] = result

    if info:
        try:
            _log_milestone(
                agent,
                run_id=run_id,
                goal=goal,
                domain=domain,
                label="post_run_intelligence",
                description="Post run learning, discovery, and verification hooks executed.",
                level="info",
                extra={"intelligence": info},
            )
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Result normalization helpers for UI
# ---------------------------------------------------------------------------


def _normalize_cycles_for_ui(cycles_list: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize a list of cycle or round summaries so the UI always sees:
        - index and cycle_index
        - citations, discoveries, sources as lists
    """
    normalized: List[Dict[str, Any]] = []
    for i, entry in enumerate(cycles_list):
        if isinstance(entry, dict):
            c = dict(entry)
        else:
            c = {"raw": entry}
        c.setdefault("index", i + 1)
        c.setdefault("cycle_index", i)
        c.setdefault("citations", [])
        c.setdefault("discoveries", [])
        c.setdefault("sources", [])
        normalized.append(c)
    return normalized


def _aggregate_from_cycles(
    cycles: List[Dict[str, Any]],
    key: str,
) -> List[Any]:
    """
    Aggregate a list field from cycles, for example citations or discoveries.
    """
    out: List[Any] = []
    for c in cycles:
        val = c.get(key)
        if isinstance(val, list):
            out.extend(val)
    return out


def _attach_top_level_defaults(
    result_obj: Dict[str, Any],
    cycles: List[Dict[str, Any]],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Make sure the top level result always exposes:
        cycles, summaries, citations, discoveries, sources, rye_metrics.
    """
    result_obj.setdefault("cycles", cycles)
    result_obj.setdefault("summaries", cycles)

    citations = result_obj.get("citations")
    if not isinstance(citations, list):
        citations = _aggregate_from_cycles(cycles, "citations")
    discoveries = result_obj.get("discoveries")
    if not isinstance(discoveries, list):
        discoveries = _aggregate_from_cycles(cycles, "discoveries")
    sources = result_obj.get("sources")
    if not isinstance(sources, list):
        sources = _aggregate_from_cycles(cycles, "sources")

    result_obj["citations"] = citations
    result_obj["discoveries"] = discoveries
    result_obj["sources"] = sources

    if diagnostics is None:
        diagnostics = {}
    result_obj.setdefault("diagnostics", diagnostics)
    result_obj.setdefault("rye_metrics", diagnostics)

    return result_obj


# ---------------------------------------------------------------------------
# Direct single-job API for engine_worker_queue.py / tests
# ---------------------------------------------------------------------------


def run_engine_job(job: Any) -> Dict[str, Any]:
    """
    Execute a single engine job and return a result dict.

    This is a pure "one-shot" API:
      - It does NOT use the file-based queue helpers.
      - It does NOT write result JSON to disk.
      - It returns a structured result that Streamlit / queue wrappers can save.

    `job` may be:
      - A dict with keys: "run_id", "config", optional "meta"
      - An object with .run_id, .config, optional .meta attributes (e.g. RunJob)
    """
    _configure_tavily_from_env()
    agent, base_config = init_agent_from_config()

    # Normalize job payload
    if isinstance(job, dict):
        cfg: Dict[str, Any] = dict(job.get("config") or {})
        run_id = str(job.get("run_id") or cfg.get("run_id") or f"job-{int(time.time())}")
        job_meta = job.get("meta")
    else:
        cfg = dict(getattr(job, "config", {}) or {})
        run_id = str(getattr(job, "run_id", f"job-{int(time.time())}"))
        job_meta = getattr(job, "meta", None)

    base_goal, base_domain = build_goal_and_domain()

    goal = str(cfg.get("goal", base_goal))
    domain = str(cfg.get("domain", base_domain))

    preset_cfg = get_preset(domain)
    runtime_profile = cfg.get(
        "runtime_profile",
        preset_cfg.get("default_runtime_profile"),
    )

    mode = str(cfg.get("mode", cfg.get("engine_mode", "single"))).lower()
    role = str(cfg.get("role", "agent"))
    roles_list: Optional[List[str]] = None
    if mode == "swarm":
        raw_roles = cfg.get("roles")
        if isinstance(raw_roles, (list, tuple)):
            roles_list = [str(r) for r in raw_roles]
        else:
            try:
                roles_list = agent.get_agent_roles()
            except Exception:
                roles_list = ["agent"]

    # ------------------------------------------------------------------
    # Safe extraction of cycles / rounds (never int(None))
    # ------------------------------------------------------------------
    max_cycles_explicit = (
        ("max_cycles" in cfg and cfg.get("max_cycles") is not None)
        or ("cycles" in cfg and cfg.get("cycles") is not None)
    )
    max_rounds_explicit = (
        ("max_rounds" in cfg and cfg.get("max_rounds") is not None)
        or ("rounds" in cfg and cfg.get("rounds") is not None)
    )

    raw_cycles = cfg.get("max_cycles")
    if raw_cycles is None:
        raw_cycles = cfg.get("cycles")
    if raw_cycles is None:
        raw_cycles = HARD_MAX_CYCLES
    try:
        requested_cycles = int(raw_cycles)
    except Exception:
        requested_cycles = HARD_MAX_CYCLES

    raw_rounds = cfg.get("max_rounds")
    if raw_rounds is None:
        raw_rounds = cfg.get("rounds")
    if raw_rounds is None:
        raw_rounds = requested_cycles
    try:
        requested_rounds = int(raw_rounds)
    except Exception:
        requested_rounds = requested_cycles

    max_cycles = _clamp_int(requested_cycles, HARD_MAX_CYCLES, "max_cycles")
    max_rounds = _clamp_int(requested_rounds, HARD_MAX_ROUNDS, "max_rounds")

    raw_max_minutes = cfg.get("max_minutes")
    if (mode == "single" and max_cycles_explicit) or (mode == "swarm" and max_rounds_explicit):
        max_minutes: Optional[float] = None
    else:
        if raw_max_minutes is not None:
            try:
                max_minutes = float(raw_max_minutes)
            except Exception:
                max_minutes = None
        else:
            max_minutes = None

    max_minutes = _clamp_minutes(max_minutes, "job.max_minutes")

    stop_rye = cfg.get("stop_rye")
    if stop_rye is not None:
        try:
            stop_rye = float(stop_rye)
        except Exception:
            stop_rye = None

    resume = bool(cfg.get("resume", True))
    watchdog_minutes = float(cfg.get("watchdog_minutes", 5.0))

    cfg_for_sources = dict(base_config)
    if "default_source_controls" not in cfg_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            cfg_for_sources["default_source_controls"] = sc_preset

    source_controls = _build_source_controls(cfg_for_sources)
    if isinstance(cfg.get("source_controls"), dict):
        override_sc = cfg["source_controls"]
        for k, v in override_sc.items():
            source_controls[str(k)] = bool(v)

    tool_flags = detect_tools()

    env_keys_for_fingerprint = [
        "WORKER_QUEUE_MODE",
        "WORKER_DOMAIN",
        "WORKER_GOAL",
    ]
    experiment_fingerprint = _build_experiment_fingerprint(
        goal=goal,
        domain=domain,
        mode=f"direct_{mode}",
        runtime_profile=runtime_profile,
        source_controls=source_controls,
        roles=roles_list if mode == "swarm" else [role],
        env_keys=env_keys_for_fingerprint,
    )

    print("")
    print("=== Direct engine job ===")
    print(f"Run id: {run_id}")
    print(f"Mode: {mode}")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Role (single): {role}")
    print(f"Roles (swarm): {roles_list if roles_list is not None else 'auto'}")
    print(f"Max cycles (single, clamped): {max_cycles} (explicit: {max_cycles_explicit})")
    print(f"Max rounds (swarm, clamped): {max_rounds} (explicit: {max_rounds_explicit})")
    print(
        "Max minutes guard (clamped): "
        f"{max_minutes if max_minutes is not None else 'None (cycles/rounds driven)'}"
    )
    print(f"Stop RYE: {stop_rye if stop_rye is not None else 'None'}")
    print(f"Runtime profile: {runtime_profile or 'None'}")
    print(f"Resume: {resume}")
    print(f"Watchdog minutes: {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    print(f"Experiment fingerprint: {experiment_fingerprint}")
    sys.stdout.flush()

    _update_worker_state(
        agent,
        status="running_job",
        mode=mode,
        goal=goal,
        domain=domain,
        roles=roles_list if mode == "swarm" else [role],
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=run_id,
        experiment_mode="direct_job",
        extra={
            "experiment_fingerprint": experiment_fingerprint,
            "job_meta": job_meta,
            "job_config": cfg,
        },
    )
    _heartbeat(agent, label="direct_job_start", run_id=run_id)

    summaries: List[Dict[str, Any]] = []

    try:
        if mode == "swarm":
            if roles_list is None:
                try:
                    roles_list = agent.get_agent_roles()
                except Exception:
                    roles_list = ["agent"]

            summaries = agent.run_swarm_continuous(
                goal=goal,
                max_rounds=max_rounds,
                stop_rye=stop_rye,
                roles=roles_list,
                source_controls=source_controls,
                pdf_bytes=None,
                biomarker_snapshot=None,
                domain=domain,
                max_minutes=max_minutes,
                forever=False,
                resume_from_checkpoint=resume,
                watchdog_interval_minutes=watchdog_minutes,
                runtime_profile=runtime_profile,
            )
        else:
            summaries = agent.run_continuous(
                goal=goal,
                max_cycles=max_cycles,
                stop_rye=stop_rye,
                role=role,
                source_controls=source_controls,
                pdf_bytes=None,
                biomarker_snapshot=None,
                domain=domain,
                max_minutes=max_minutes,
                forever=False,
                resume_from_checkpoint=resume,
                watchdog_interval_minutes=watchdog_minutes,
                runtime_profile=runtime_profile,
            )

        _heartbeat(agent, label="direct_job_finished", run_id=run_id)

        print(f"=== Direct job {run_id} finished cleanly ===")
        print(f"Total summaries: {len(summaries)}")

        diag: Dict[str, Any] = {}
        try:
            diag = build_run_diagnostics(history=summaries, domain=domain, window=10)
            print(f"RYE avg: {diag.get('rye_avg')}")
            print(f"RYE median: {diag.get('rye_median')}")
            print(f"RYE last: {diag.get('rye_last')}")
            print(f"Stability index: {diag.get('stability_index')}")
            print(f"Recovery momentum: {diag.get('recovery_momentum')}")
        except Exception:
            print("Diagnostics computation failed for direct job, see logs for details.")

        # Normalize cycles for UI and diagnostics
        normalized_cycles: List[Dict[str, Any]] = _normalize_cycles_for_ui(summaries)

        # Write cycle history and run_state snapshots for diagnostics panel
        _write_cycles_and_run_state(
            agent,
            run_id=run_id,
            mode=mode,
            goal=goal,
            domain=domain,
            cycles=normalized_cycles,
            diagnostics=diag,
        )

        intelligence_info = _run_post_run_intelligence(
            agent,
            mode=mode,
            goal=goal,
            domain=domain,
            run_id=run_id,
            history=normalized_cycles,
        )

        # Build a simple overall summary string if possible
        overall_summary: Optional[str] = None
        for c in normalized_cycles:
            for key in ("summary", "brief", "title", "description"):
                val = c.get(key)
                if isinstance(val, str) and val.strip():
                    overall_summary = val.strip()
                    break
            if overall_summary:
                break

        extra_manifest: Dict[str, Any] = {
            "engine": f"direct_{mode}",
            "experiment_fingerprint": experiment_fingerprint,
            "job_meta": job_meta,
            "job_config": cfg,
        }
        if diag:
            extra_manifest["diagnostics_snapshot"] = diag
        if intelligence_info:
            extra_manifest["intelligence"] = intelligence_info

        try:
            _log_run_manifest(
                agent,
                run_id,
                mode=mode,
                domain=domain,
                goal=goal,
                runtime_profile=runtime_profile,
                stop_rye=stop_rye,
                max_minutes=max_minutes,
                summaries=normalized_cycles,
                extra=extra_manifest,
            )
        except Exception:
            print("Manifest logging failed for direct job, see logs for details.")

        result_obj: Dict[str, Any] = {
            "status": "ok",
            "run_id": run_id,
            "mode": mode,
            "goal": goal,
            "domain": domain,
            "runtime_profile": runtime_profile,
            "max_minutes": max_minutes,
            "max_cycles": max_cycles if mode == "single" else None,
            "max_rounds": max_rounds if mode == "swarm" else None,
            "stop_rye": stop_rye,
            "summaries": normalized_cycles,
            "cycles": normalized_cycles,
            "diagnostics": diag,
            "intelligence": intelligence_info,
            "experiment_fingerprint": experiment_fingerprint,
            "job_meta": job_meta,
            "job_config": cfg,
        }
        if overall_summary:
            result_obj["summary"] = overall_summary

        # Attach top level defaults including citations / discoveries / sources / rye_metrics
        result_obj = _attach_top_level_defaults(result_obj, normalized_cycles, diag)

        _update_worker_state(
            agent,
            status="idle",
            mode=mode,
            goal=goal,
            domain=domain,
            roles=roles_list if mode == "swarm" else [role],
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            run_id=run_id,
            experiment_mode="direct_job",
            extra={
                "experiment_fingerprint": experiment_fingerprint,
                "final_diagnostics": diag,
                "intelligence": intelligence_info if intelligence_info else None,
                "job_config": cfg,
            },
        )

        return result_obj

    except Exception as e:
        print(f"Fatal error while running direct job {run_id}: {e}")
        tb = traceback.format_exc()
        print(tb)
        sys.stdout.flush()

        error_payload: Dict[str, Any] = {
            "error": str(e),
            "traceback": tb,
            "run_id": run_id,
            "goal": goal,
            "domain": domain,
            "mode": mode,
            "runtime_profile": runtime_profile,
            "stop_rye": stop_rye,
            "max_minutes": max_minutes,
            "max_cycles": max_cycles if mode == "single" else None,
            "max_rounds": max_rounds if mode == "swarm" else None,
            "job_meta": job_meta,
            "job_config": cfg,
            "experiment_fingerprint": experiment_fingerprint,
        }

        try:
            # Log a milestone so the UI can see prompt and error context
            _log_milestone(
                agent,
                run_id=run_id,
                goal=goal,
                domain=domain,
                label="direct_job_error",
                description=f"Direct job failed with error: {e}",
                level="error",
                extra=error_payload,
            )
        except Exception:
            pass

        _update_worker_state(
            agent,
            status="error",
            mode=mode,
            goal=goal,
            domain=domain,
            roles=roles_list if mode == "swarm" else [role],
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            run_id=run_id,
            experiment_mode="direct_job",
            extra=error_payload,
        )

        return {
            "status": "error",
            "run_id": run_id,
            "mode": mode,
            "goal": goal,
            "domain": domain,
            "error": str(e),
            "traceback": tb,
            "job_config": cfg,
            "job_meta": job_meta,
        }


# ---------------------------------------------------------------------------
# Queue mode: process jobs from agent/run_jobs
# ---------------------------------------------------------------------------


def _write_job_progress(
    run_id: str,
    status: str,
    note: str = "",
    current: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """
    Best-effort progress writer for queue jobs.

    Writes runs/active/{run_id}_progress.json using run_jobs.progress_path
    so the Streamlit UI can show basic status, even if we do not have
    per-cycle callbacks.
    """
    if progress_path is None:
        return
    try:
        path = progress_path(run_id)
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "status": status,
            "current_cycle": current,
            "total_cycles": total,
            "last_update_utc": datetime.utcnow().isoformat() + "Z",
            "notes": note,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # Progress should never crash the worker
        return


def _process_single_job(agent: CoreAgent, base_config: Dict[str, Any], job: RunJob) -> None:
    """
    Execute a single RunJob from the file based queue.

    Job config can include:
        mode: "single" or "swarm" (default "single")
        goal: optional override goal
        domain: optional override domain
        role: role for single agent (default "agent")
        roles: list of roles for swarm
        max_cycles: finite cycles for single
        max_rounds: finite rounds for swarm
        max_minutes: optional wall time guard
        stop_rye: optional stop threshold
        runtime_profile: optional profile name
        source_controls: optional explicit source controls dict
        resume: optional resume flag (default True)
        watchdog_minutes: optional watchdog interval

    If explicit cycle or round limits are provided in the job config,
    the worker ignores max_minutes so the job runs until the specified
    cycles or rounds complete (or stop_rye triggers).
    """
    start_ts = time.time()

    base_goal, base_domain = build_goal_and_domain()
    cfg = job.config or {}

    goal = str(cfg.get("goal", base_goal))
    domain = str(cfg.get("domain", base_domain))

    preset_cfg = get_preset(domain)
    runtime_profile = cfg.get(
        "runtime_profile",
        preset_cfg.get("default_runtime_profile"),
    )

    mode = str(cfg.get("mode", cfg.get("engine_mode", "single"))).lower()
    role = str(cfg.get("role", "agent"))
    roles_list: Optional[List[str]] = None
    if mode == "swarm":
        raw_roles = cfg.get("roles")
        if isinstance(raw_roles, (list, tuple)):
            roles_list = [str(r) for r in raw_roles]
        else:
            try:
                roles_list = agent.get_agent_roles()
            except Exception:
                roles_list = ["agent"]

    # ------------------------------------------------------------------
    # Safe extraction of cycles / rounds for queue jobs
    # ------------------------------------------------------------------
    max_cycles_explicit = (
        ("max_cycles" in cfg and cfg.get("max_cycles") is not None)
        or ("cycles" in cfg and cfg.get("cycles") is not None)
    )
    max_rounds_explicit = (
        ("max_rounds" in cfg and cfg.get("max_rounds") is not None)
        or ("rounds" in cfg and cfg.get("rounds") is not None)
    )

    raw_cycles = cfg.get("max_cycles")
    if raw_cycles is None:
        raw_cycles = cfg.get("cycles")
    if raw_cycles is None:
        raw_cycles = HARD_MAX_CYCLES
    try:
        requested_cycles = int(raw_cycles)
    except Exception:
        requested_cycles = HARD_MAX_CYCLES

    raw_rounds = cfg.get("max_rounds")
    if raw_rounds is None:
        raw_rounds = cfg.get("rounds")
    if raw_rounds is None:
        raw_rounds = requested_cycles
    try:
        requested_rounds = int(raw_rounds)
    except Exception:
        requested_rounds = requested_cycles

    max_cycles = _clamp_int(requested_cycles, HARD_MAX_CYCLES, "max_cycles")
    max_rounds = _clamp_int(requested_rounds, HARD_MAX_ROUNDS, "max_rounds")

    raw_max_minutes = cfg.get("max_minutes")
    if (mode == "single" and max_cycles_explicit) or (mode == "swarm" and max_rounds_explicit):
        max_minutes: Optional[float] = None
    else:
        if raw_max_minutes is not None:
            try:
                max_minutes = float(raw_max_minutes)
            except Exception:
                max_minutes = None
        else:
            max_minutes = None

    max_minutes = _clamp_minutes(max_minutes, "job.max_minutes")

    stop_rye = cfg.get("stop_rye")
    if stop_rye is not None:
        try:
            stop_rye = float(stop_rye)
        except Exception:
            stop_rye = None

    resume = bool(cfg.get("resume", True))
    watchdog_minutes = float(cfg.get("watchdog_minutes", 5.0))

    cfg_for_sources = dict(base_config)
    if "default_source_controls" not in cfg_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            cfg_for_sources["default_source_controls"] = sc_preset

    source_controls = _build_source_controls(cfg_for_sources)
    if isinstance(cfg.get("source_controls"), dict):
        override_sc = cfg["source_controls"]
        for k, v in override_sc.items():
            source_controls[str(k)] = bool(v)

    tool_flags = detect_tools()

    env_keys_for_fingerprint = [
        "WORKER_QUEUE_MODE",
        "WORKER_DOMAIN",
        "WORKER_GOAL",
    ]
    experiment_fingerprint = _build_experiment_fingerprint(
        goal=goal,
        domain=domain,
        mode=f"queue_{mode}",
        runtime_profile=runtime_profile,
        source_controls=source_controls,
        roles=roles_list if mode == "swarm" else [role],
        env_keys=env_keys_for_fingerprint,
    )

    job_meta = getattr(job, "meta", None)

    print("")
    print("=== Queue worker: starting job ===")
    print(f"Run id: {job.run_id}")
    print(f"Mode: {mode}")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Role (single): {role}")
    print(f"Roles (swarm): {roles_list if roles_list is not None else 'auto'}")
    print(f"Max cycles (single, clamped): {max_cycles} (explicit: {max_cycles_explicit})")
    print(f"Max rounds (swarm, clamped): {max_rounds} (explicit: {max_rounds_explicit})")
    print(
        "Max minutes guard (clamped): "
        f"{max_minutes if max_minutes is not None else 'None (rounds driven)'}"
    )
    print(f"Stop RYE: {stop_rye if stop_rye is not None else 'None'}")
    print(f"Runtime profile: {runtime_profile or 'None'}")
    print(f"Resume: {resume}")
    print(f"Watchdog minutes: {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    print(f"Experiment fingerprint: {experiment_fingerprint}")
    sys.stdout.flush()

    _update_worker_state(
        agent,
        status="running_job",
        mode=mode,
        goal=goal,
        domain=domain,
        roles=roles_list if mode == "swarm" else [role],
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=job.run_id,
        experiment_mode="queue_worker",
        extra={
            "experiment_fingerprint": experiment_fingerprint,
            "job_meta": job_meta,
            "job_config": cfg,
        },
    )
    _heartbeat(agent, label="queue_job_start", run_id=job.run_id)

    # Initial progress write
    _write_job_progress(
        job.run_id,
        status="active",
        note="Job started",
        current=0,
        total=max_rounds if mode == "swarm" else max_cycles,
    )

    summaries: List[Dict[str, Any]] = []
    full_result: Optional[Dict[str, Any]] = None

    try:
        # Prefer a structured run_goal path if the agent exposes it.
        if hasattr(agent, "run_goal"):
            print("[Queue] Using CoreAgent.run_goal for structured result bundle.")
            sys.stdout.flush()

            def _progress_cb(update: Dict[str, Any]) -> None:
                # Pass through to progress file so UI can live-refresh.
                current = update.get("current_cycle")
                total_local = update.get("total_cycles")
                note = update.get("notes", "")
                _write_job_progress(
                    job.run_id,
                    status=str(update.get("status", "active")),
                    note=str(note),
                    current=int(current) if isinstance(current, (int, float)) else None,
                    total=int(total_local) if isinstance(total_local, (int, float)) else None,
                )

            goal_config: Dict[str, Any] = {
                **cfg,
                "mode": mode,
                "domain": domain,
                "runtime_profile": runtime_profile,
                "max_minutes": max_minutes,
                "max_cycles": max_cycles,
                "max_rounds": max_rounds,
                "stop_rye": stop_rye,
                "source_controls": source_controls,
            }
            if mode == "swarm":
                goal_config["roles"] = roles_list
            else:
                goal_config["role"] = role

            try:
                full_result = agent.run_goal(
                    goal=goal,
                    config=goal_config,
                    progress_callback=_progress_cb,
                )
            except TypeError:
                # Fallback if run_goal does not accept progress_callback
                full_result = agent.run_goal(goal=goal, config=goal_config)  # type: ignore[arg-type]

            if isinstance(full_result, dict):
                cycles_from_result = (
                    full_result.get("cycles")
                    or full_result.get("summaries")
                    or []
                )
                if isinstance(cycles_from_result, list):
                    summaries = cycles_from_result  # for diagnostics below

        # If no run_goal or it failed, fall back to legacy continuous engines.
        if not summaries and full_result is None:
            if mode == "swarm":
                if roles_list is None:
                    try:
                        roles_list = agent.get_agent_roles()
                    except Exception:
                        roles_list = ["agent"]

                summaries = agent.run_swarm_continuous(
                    goal=goal,
                    max_rounds=max_rounds,
                    stop_rye=stop_rye,
                    roles=roles_list,
                    source_controls=source_controls,
                    pdf_bytes=None,
                    biomarker_snapshot=None,
                    domain=domain,
                    max_minutes=max_minutes,
                    forever=False,
                    resume_from_checkpoint=resume,
                    watchdog_interval_minutes=watchdog_minutes,
                    runtime_profile=runtime_profile,
                )
            else:
                summaries = agent.run_continuous(
                    goal=goal,
                    max_cycles=max_cycles,
                    stop_rye=stop_rye,
                    role=role,
                    source_controls=source_controls,
                    pdf_bytes=None,
                    biomarker_snapshot=None,
                    domain=domain,
                    max_minutes=max_minutes,
                    forever=False,
                    resume_from_checkpoint=resume,
                    watchdog_interval_minutes=watchdog_minutes,
                    runtime_profile=runtime_profile,
                )

        _heartbeat(agent, label="queue_job_finished", run_id=job.run_id)

        print(f"=== Queue worker: job {job.run_id} finished cleanly ===")
        print(f"Total summaries: {len(summaries)}")

        # Final progress write (we only know final count now)
        _write_job_progress(
            job.run_id,
            status="finished",
            note="Job finished",
            current=len(summaries),
            total=max_rounds if mode == "swarm" else max_cycles,
        )

        diag: Dict[str, Any] = {}
        try:
            diag = build_run_diagnostics(history=summaries, domain=domain, window=10)
            print(f"RYE avg: {diag.get('rye_avg')}")
            print(f"RYE median: {diag.get('rye_median')}")
            print(f"RYE last: {diag.get('rye_last')}")
            print(f"Stability index: {diag.get('stability_index')}")
            print(f"Recovery momentum: {diag.get('recovery_momentum')}")
        except Exception:
            print("Diagnostics computation failed for job, see logs for details.")

        # Normalize cycles so diagnostics and UI always see a consistent list
        normalized_cycles: List[Dict[str, Any]] = _normalize_cycles_for_ui(summaries)

        # Write cycle history and run_state snapshots for diagnostics panel
        _write_cycles_and_run_state(
            agent,
            run_id=job.run_id,
            mode=mode,
            goal=goal,
            domain=domain,
            cycles=normalized_cycles,
            diagnostics=diag,
        )

        intelligence_info = _run_post_run_intelligence(
            agent,
            mode=mode,
            goal=goal,
            domain=domain,
            run_id=job.run_id,
            history=normalized_cycles,
        )

        # ------------------------------------------------------------------
        # Normalize cycles so the UI always sees cycles + summaries lists
        # with index / cycle_index and an overall summary string.
        # ------------------------------------------------------------------
        overall_summary: Optional[str] = None
        for c in normalized_cycles:
            for key in ("summary", "brief", "title", "description"):
                val = c.get(key)
                if isinstance(val, str) and val.strip():
                    overall_summary = val.strip()
                    break
            if overall_summary:
                break

        extra_manifest: Dict[str, Any] = {
            "engine": f"queue_{mode}",
            "experiment_fingerprint": experiment_fingerprint,
            "job_meta": job_meta,
            "job_config": cfg,
        }
        if diag:
            extra_manifest["diagnostics_snapshot"] = diag
        if intelligence_info:
            extra_manifest["intelligence"] = intelligence_info

        try:
            _log_run_manifest(
                agent,
                job.run_id,
                mode=mode,
                domain=domain,
                goal=goal,
                runtime_profile=runtime_profile,
                stop_rye=stop_rye,
                max_minutes=max_minutes,
                summaries=normalized_cycles,
                extra=extra_manifest,
            )
        except Exception:
            print("Manifest logging failed for job, see logs for details.")

        completed_ts = time.time()

        # Build final result bundle.
        if isinstance(full_result, dict):
            # Ensure both "cycles" and "summaries" keys are present for the UI.
            fr = dict(full_result)

            # Prefer whatever the agent returned, but fall back to our normalized list.
            cycles_src = fr.get("cycles") or fr.get("summaries") or normalized_cycles
            if not isinstance(cycles_src, list):
                cycles_src = normalized_cycles

            norm_fr_cycles: List[Dict[str, Any]] = _normalize_cycles_for_ui(cycles_src)

            fr["cycles"] = norm_fr_cycles
            fr.setdefault("summaries", norm_fr_cycles)

            # Add overall summary if missing.
            if "summary" not in fr and overall_summary:
                fr["summary"] = overall_summary

            result_obj: Dict[str, Any] = {
                "job_id": job.run_id,
                "status": fr.get("status", "finished"),
                "created_at": job.created_at,
                "completed_at": completed_ts,
                "elapsed_seconds": completed_ts - start_ts,
                "meta": job_meta or {},
                "job_config": cfg,
            }
            result_obj.update(fr)

            # Attach top level defaults for the structured full_result case
            result_obj = _attach_top_level_defaults(result_obj, norm_fr_cycles, diag)
        else:
            # Legacy minimal shape, but still expose cycles alias.
            result_obj = {
                "run_id": job.run_id,
                "mode": mode,
                "goal": goal,
                "domain": domain,
                "runtime_profile": runtime_profile,
                "max_minutes": max_minutes,
                "max_cycles": max_cycles if mode == "single" else None,
                "max_rounds": max_rounds if mode == "swarm" else None,
                "stop_rye": stop_rye,
                "summaries": normalized_cycles,
                "cycles": normalized_cycles,
                "diagnostics": diag,
                "intelligence": intelligence_info,
                "experiment_fingerprint": experiment_fingerprint,
                "job_meta": job_meta,
                "job_config": cfg,
            }
            result_obj = _attach_top_level_defaults(result_obj, normalized_cycles, diag)

        if overall_summary:
            result_obj["summary"] = overall_summary

        try:
            if save_job_result is not None:
                save_job_result(job, result_obj)
            else:
                out_dir = BASE_DIR / "finished"
                out_dir.mkdir(parents=True, exist_ok=True)
                rp = out_dir / f"{job.run_id}.json"
                with rp.open("w", encoding="utf8") as f:
                    json.dump(result_obj, f, indent=2)
        except Exception:
            print("Failed to write result JSON for job, see logs for details.")

        _update_worker_state(
            agent,
            status="idle",
            mode=mode,
            goal=goal,
            domain=domain,
            roles=roles_list if mode == "swarm" else [role],
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            run_id=job.run_id,
            experiment_mode="queue_worker",
            extra={
                "experiment_fingerprint": experiment_fingerprint,
                "final_diagnostics": diag,
                "intelligence": intelligence_info if intelligence_info else None,
                "job_config": cfg,
            },
        )

    except Exception as e:
        print(f"Fatal error while running job {job.run_id}: {e}")
        tb = traceback.format_exc()
        print(tb)
        sys.stdout.flush()

        # Progress -> error
        _write_job_progress(
            job.run_id,
            status="error",
            note=str(e),
            current=None,
            total=max_rounds if mode == "swarm" else max_cycles,
        )

        error_payload: Dict[str, Any] = {
            "error": str(e),
            "traceback": tb,
            "run_id": job.run_id,
            "goal": goal,
            "domain": domain,
            "mode": mode,
            "runtime_profile": runtime_profile,
            "stop_rye": stop_rye,
            "max_minutes": max_minutes,
            "max_cycles": max_cycles if mode == "single" else None,
            "max_rounds": max_rounds if mode == "swarm" else None,
            "job_meta": job_meta,
            "job_config": cfg,
            "experiment_fingerprint": experiment_fingerprint,
        }

        try:
            if mark_job_error is not None:
                # Save full context including prompt and config into error record
                mark_job_error(job, error_payload)
            else:
                err_dir = BASE_DIR / "error"
                err_dir.mkdir(parents=True, exist_ok=True)
                ep = err_dir / f"{job.run_id}.json"
                with ep.open("w", encoding="utf8") as f:
                    json.dump(error_payload, f, indent=2)
        except Exception:
            pass

        try:
            _log_milestone(
                agent,
                run_id=job.run_id,
                goal=goal,
                domain=domain,
                label="queue_job_error",
                description=f"Queue job failed with error: {e}",
                level="error",
                extra=error_payload,
            )
        except Exception:
            pass

        _update_worker_state(
            agent,
            status="error",
            mode=mode,
            goal=goal,
            domain=domain,
            roles=roles_list if mode == "swarm" else [role],
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            run_id=job.run_id,
            experiment_mode="queue_worker",
            extra=error_payload,
        )


def run_job_queue_worker() -> None:
    """
    Main loop for queue mode.

    Behavior:
        - Initializes CoreAgent and MemoryStore once.
        - Polls the job queue via agent.run_jobs.load_next_pending_job.
        - For each job:
            * Runs it with _process_single_job
        - Loops continuously so Render worker can stay alive.
    """
    if RunJob is None or load_next_pending_job is None:
        print("Queue mode requested but agent/run_jobs.py is not available.")
        sys.stdout.flush()
        return

    print("Starting Autonomous Research Agent queue worker (file based jobs)...")
    sys.stdout.flush()

    _configure_tavily_from_env()
    agent, config = init_agent_from_config()

    # Extra debug so you can validate path alignment with Streamlit and run_jobs
    ara_runs_env = os.getenv("ARA_RUNS_DIR")
    print(f"[Queue] ARA_RUNS_DIR env: {ara_runs_env!r}")
    try:
        print(f"[Queue] BASE_DIR (engine_worker): {BASE_DIR.resolve()}")
    except Exception:
        print(f"[Queue] BASE_DIR (engine_worker): {BASE_DIR}")
    pending_dir = BASE_DIR / "pending"
    try:
        import agent.run_jobs as run_jobs_mod  # type: ignore[import]
        rj_base = getattr(run_jobs_mod, "BASE_DIR", None)
        rj_pending = getattr(run_jobs_mod, "PENDING_DIR", None)
        if rj_base is not None:
            try:
                print(f"[Queue] run_jobs.BASE_DIR: {rj_base.resolve()}")
            except Exception:
                print(f"[Queue] run_jobs.BASE_DIR: {rj_base}")
        if rj_pending is not None:
            try:
                print(f"[Queue] run_jobs.PENDING_DIR: {rj_pending.resolve()}")
            except Exception:
                print(f"[Queue] run_jobs.PENDING_DIR: {rj_pending}")
            pending_dir = rj_pending
    except Exception:
        print("[Queue] Could not import agent.run_jobs for extra debug info.")
    sys.stdout.flush()

    # Debug: show which directory this worker is actually watching
    try:
        print(f"[Queue] Watching pending dir: {pending_dir.resolve()}")
    except Exception:
        print(f"[Queue] Watching pending dir: {pending_dir}")
    sys.stdout.flush()

    idle_loops = 0

    while True:
        # Debug: list any pending files the worker can see
        try:
            if pending_dir.exists():
                pending_files = sorted(pending_dir.glob("*.json"))
                if pending_files:
                    print(
                        f"[Queue] Pending .json files visible to worker: "
                        f"{[p.name for p in pending_files]}"
                    )
                else:
                    print("[Queue] No pending .json files visible to worker.")
            else:
                try:
                    print(f"[Queue] Pending dir does not exist: {pending_dir.resolve()}")
                except Exception:
                    print(f"[Queue] Pending dir does not exist: {pending_dir}")
        except Exception as e:
            print(f"[Queue] Error listing pending dir: {e}")

        sys.stdout.flush()

        try:
            job = load_next_pending_job()
        except Exception as e:
            print(f"[Queue] load_next_pending_job() raised an exception: {e}")
            print(traceback.format_exc())
            sys.stdout.flush()
            _heartbeat(agent, label="queue_error")
            time.sleep(5.0)
            continue

        if job is None:
            idle_loops += 1
            print("[Queue] No runnable job returned by load_next_pending_job().")
            sys.stdout.flush()
            _heartbeat(agent, label="queue_idle")
            time.sleep(5.0)
            continue

        print(f"[Queue] Loaded job from queue: {getattr(job, 'run_id', 'unknown')}")
        sys.stdout.flush()

        _process_single_job(agent, config, job)
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Single and swarm engines
# ---------------------------------------------------------------------------


def run_single_agent_engine(agent: CoreAgent, config: Dict[str, Any]) -> None:
    """
    Run a long continuous single agent session (finite-only).

    Controlled by:
    - WORKER_MAX_MINUTES (optional wall time guard)
    - WORKER_MAX_CYCLES (optional cycle guard; default very large finite)
    - WORKER_STOP_RYE (optional)
    - WORKER_RUNTIME_PROFILE (optional: hint only)
    - WORKER_ROLE (default "agent")
    - WORKER_SOURCES (optional, comma separated)
    - WORKER_RESUME (bool, default True)
    - WORKER_WATCHDOG_MINUTES (float, default 5.0)

    Forever mode is disabled: the engine always runs with finite limits.
    """
    goal, domain = build_goal_and_domain()
    preset_cfg = get_preset(domain)

    max_minutes_env = _env_float("WORKER_MAX_MINUTES")
    max_minutes = _clamp_minutes(max_minutes_env, "single.max_minutes")

    stop_rye = _env_float("WORKER_STOP_RYE")

    runtime_profile_env = os.getenv("WORKER_RUNTIME_PROFILE")
    runtime_profile = runtime_profile_env or preset_cfg.get("default_runtime_profile")

    role = os.getenv("WORKER_ROLE", "agent")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0

    max_cycles_env = os.getenv("WORKER_MAX_CYCLES")
    if max_cycles_env is not None:
        try:
            requested_cycles = int(max_cycles_env)
        except Exception:
            requested_cycles = HARD_MAX_CYCLES
    else:
        requested_cycles = HARD_MAX_CYCLES

    max_cycles = _clamp_int(requested_cycles, HARD_MAX_CYCLES, "single.max_cycles")
    forever = False

    config_for_sources = dict(config)
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    tool_flags = detect_tools()
    run_id = _current_run_id("single")

    env_keys_for_fingerprint = [
        "WORKER_MAX_MINUTES",
        "WORKER_MAX_CYCLES",
        "WORKER_STOP_RYE",
        "WORKER_RUNTIME_PROFILE",
        "WORKER_MODE",
        "WORKER_SWARM",
        "WORKER_META",
        "WORKER_SOURCES",
        "WORKER_SWARM_ROLES",
        "WORKER_DOMAIN",
        "WORKER_GOAL",
    ]
    experiment_fingerprint = _build_experiment_fingerprint(
        goal=goal,
        domain=domain,
        mode="single",
        runtime_profile=runtime_profile,
        source_controls=source_controls,
        roles=[role],
        env_keys=env_keys_for_fingerprint,
    )

    print("=== Autonomous Research Engine - Single Agent Mode (Finite Only) ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Role: {role}")
    print(f"Run id: {run_id}")
    print(f"Runtime profile (env override): {runtime_profile_env or 'None'}")
    print(
        "Runtime profile (effective, hint only): "
        f"{runtime_profile or 'None (engine default)'}"
    )
    print(
        "Max minutes (explicit, clamped): "
        f"{max_minutes if max_minutes is not None else 'None (cycles-only guard)'}"
    )
    print(f"Max cycles (clamped): {max_cycles}")
    print(
        "Stop RYE threshold (explicit): "
        f"{stop_rye if stop_rye is not None else 'None (preset/profile)'}"
    )
    print(f"Forever mode (disabled in finite-only): {forever}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    print(f"Experiment fingerprint: {experiment_fingerprint}")
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
        experiment_mode="single_engine",
        extra={"experiment_fingerprint": experiment_fingerprint},
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
        experiment_mode="single_engine",
        extra={"experiment_fingerprint": experiment_fingerprint},
    )

    summaries = agent.run_continuous(
        goal=goal,
        max_cycles=max_cycles,
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

    print("=== Continuous run finished cleanly (finite) ===")
    print(f"Total completed cycles: {len(summaries)}")
    diag: Dict[str, Any] = {}
    try:
        diag = build_run_diagnostics(history=summaries, domain=domain, window=10)
        print(f"RYE avg: {diag.get('rye_avg')}")
        print(f"RYE median: {diag.get('rye_median')}")
        print(f"RYE last: {diag.get('rye_last')}")
        print(f"Stability index: {diag.get('stability_index')}")
        print(f"Recovery momentum: {diag.get('recovery_momentum')}")
    except Exception:
        print("Diagnostics computation failed, see logs for details.")

    # Write cycle history and run_state snapshots for diagnostics panel
    _write_cycles_and_run_state(
        agent,
        run_id=run_id,
        mode="single",
        goal=goal,
        domain=domain,
        cycles=_normalize_cycles_for_ui(summaries),
        diagnostics=diag,
    )

    intelligence_info = _run_post_run_intelligence(
        agent,
        mode="single",
        goal=goal,
        domain=domain,
        run_id=run_id,
        history=summaries,
    )

    extra_manifest: Dict[str, Any] = {
        "engine": "single",
        "experiment_fingerprint": experiment_fingerprint,
    }
    if diag:
        extra_manifest["diagnostics_snapshot"] = diag
    if intelligence_info:
        extra_manifest["intelligence"] = intelligence_info

    try:
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
            extra=extra_manifest,
        )
    except Exception:
        print("Manifest logging failed, see logs for details.")
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
        experiment_mode="single_engine",
        extra={
            "experiment_fingerprint": experiment_fingerprint,
            "final_diagnostics": diag,
            "intelligence": intelligence_info if intelligence_info else None,
        },
    )


def run_swarm_engine(agent: CoreAgent, config: Dict[str, Any]) -> None:
    """
    Run a long continuous swarm session (finite-only).

    Controlled by:
    - WORKER_SWARM_ROLES (comma separated, optional)
    - WORKER_MAX_MINUTES (optional wall time guard)
    - WORKER_MAX_ROUNDS (optional round guard; default very large finite)
    - WORKER_STOP_RYE (optional)
    - WORKER_RUNTIME_PROFILE (optional hint)
    - WORKER_SOURCES (optional, comma separated)
    - WORKER_RESUME (bool, default True)
    - WORKER_WATCHDOG_MINUTES (float, default 5.0)
    - WORKER_SHIFT_MINUTES (optional, minutes per shift; enables shift mode if > 0)
    - WORKER_REPEAT_SHIFTS (bool, default False; if True and shift minutes set, runs back to back shifts)

    Forever mode is disabled: swarm always runs with finite limits.
    """
    goal, domain = build_goal_and_domain()
    preset_cfg = get_preset(domain)

    max_minutes_env = _env_float("WORKER_MAX_MINUTES")
    max_minutes = _clamp_minutes(max_minutes_env, "swarm.max_minutes")

    stop_rye = _env_float("WORKER_STOP_RYE")

    runtime_profile_env = os.getenv("WORKER_RUNTIME_PROFILE")
    runtime_profile = runtime_profile_env or preset_cfg.get("default_runtime_profile")

    roles_list = _env_list("WORKER_SWARM_ROLES")
    resume = _env_bool("WORKER_RESUME", default=True)
    watchdog_minutes = _env_float("WORKER_WATCHDOG_MINUTES") or 5.0

    max_rounds_env = os.getenv("WORKER_MAX_ROUNDS")
    if max_rounds_env is not None:
        try:
            requested_rounds = int(max_rounds_env)
        except Exception:
            requested_rounds = HARD_MAX_ROUNDS
    else:
        requested_rounds = HARD_MAX_ROUNDS

    max_rounds = _clamp_int(requested_rounds, HARD_MAX_ROUNDS, "swarm.max_rounds")

    shift_minutes_env = _env_float("WORKER_SHIFT_MINUTES")
    shift_minutes = _clamp_minutes(shift_minutes_env, "swarm.shift_minutes")
    repeat_shifts = _env_bool("WORKER_REPEAT_SHIFTS", default=False)

    if roles_list is None:
        roles_list = agent.get_agent_roles()

    forever = False

    config_for_sources = dict(config)
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    tool_flags = detect_tools()
    run_id = _current_run_id("swarm")

    env_keys_for_fingerprint = [
        "WORKER_MAX_MINUTES",
        "WORKER_MAX_ROUNDS",
        "WORKER_STOP_RYE",
        "WORKER_RUNTIME_PROFILE",
        "WORKER_MODE",
        "WORKER_SWARM",
        "WORKER_META",
        "WORKER_SOURCES",
        "WORKER_SWARM_ROLES",
        "WORKER_DOMAIN",
        "WORKER_GOAL",
        "WORKER_SHIFT_MINUTES",
        "WORKER_REPEAT_SHIFTS",
    ]
    experiment_fingerprint = _build_experiment_fingerprint(
        goal=goal,
        domain=domain,
        mode="swarm",
        runtime_profile=runtime_profile,
        source_controls=source_controls,
        roles=roles_list,
        env_keys=env_keys_for_fingerprint,
    )

    print("=== Autonomous Research Engine - Swarm Mode (Finite Only) ===")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Roles: {roles_list}")
    print(f"Run id: {run_id}")
    print(f"Runtime profile (env override): {runtime_profile_env or 'None'}")
    print(
        "Runtime profile (effective, hint only): "
        f"{runtime_profile or 'None (engine default)'}"
    )
    print(
        "Max minutes (explicit, clamped): "
        f"{max_minutes if max_minutes is not None else 'None (rounds-only guard)'}"
    )
    print(f"Max rounds (clamped): {max_rounds}")
    print(
        "Stop RYE threshold (explicit): "
        f"{stop_rye if stop_rye is not None else 'None (preset/profile)'}"
    )
    print(f"Forever mode (disabled in finite-only): {forever}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    print(f"Shift minutes (clamped): {shift_minutes if shift_minutes is not None else 'None'}")
    print(f"Repeat shifts: {repeat_shifts}")
    print(f"Experiment fingerprint: {experiment_fingerprint}")
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
        experiment_mode="swarm_engine",
        extra={"experiment_fingerprint": experiment_fingerprint},
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
        experiment_mode="swarm_engine",
        extra={"experiment_fingerprint": experiment_fingerprint},
    )

    summaries_all: List[Dict[str, Any]] = []

    if not repeat_shifts or shift_minutes is None or shift_minutes <= 0:
        summaries_all = agent.run_swarm_continuous(
            goal=goal,
            max_rounds=max_rounds,
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
    else:
        total_elapsed = 0.0
        shift_index = 0

        while True:
            if max_minutes is not None:
                remaining = max_minutes - total_elapsed
                if remaining <= 0:
                    break
                this_shift_minutes = min(shift_minutes, remaining)
            else:
                this_shift_minutes = shift_minutes

            shift_index += 1
            print(
                f"--- Swarm shift {shift_index} starting, "
                f"budget {this_shift_minutes:.2f} minutes ---"
            )
            sys.stdout.flush()

            shift_summaries = agent.run_swarm_continuous(
                goal=goal,
                max_rounds=max_rounds,
                stop_rye=stop_rye,
                roles=roles_list,
                source_controls=source_controls,
                pdf_bytes=None,
                biomarker_snapshot=None,
                domain=domain,
                max_minutes=this_shift_minutes,
                forever=False,
                resume_from_checkpoint=resume,
                watchdog_interval_minutes=watchdog_minutes,
                runtime_profile=runtime_profile,
            )

            if not shift_summaries:
                used = this_shift_minutes
            else:
                last_meta = shift_summaries[-1].get("run_metadata")
                em = None
                if isinstance(last_meta, dict):
                    em = last_meta.get("elapsed_minutes")
                if isinstance(em, (int, float)):
                    used = float(em)
                else:
                    used = this_shift_minutes

            total_elapsed += used
            summaries_all.extend(shift_summaries)

            print(
                f"--- Swarm shift {shift_index} finished, "
                f"elapsed this shift {used:.2f} minutes, total {total_elapsed:.2f} ---"
            )
            sys.stdout.flush()

            if max_minutes is not None and total_elapsed >= max_minutes:
                break

            if not repeat_shifts:
                break

    summaries = summaries_all

    _heartbeat(agent, label="worker_swarm_finished", run_id=run_id)

    print("=== Swarm run finished cleanly (finite) ===")
    print(
        "Total summaries produced across all roles and rounds: "
        f"{len(summaries)}"
    )
    diag: Dict[str, Any] = {}
    try:
        diag = build_run_diagnostics(history=summaries, domain=domain, window=10)
        print(f"RYE avg: {diag.get('rye_avg')}")
        print(f"RYE median: {diag.get('rye_median')}")
        print(f"RYE last: {diag.get('rye_last')}")
        print(f"Stability index: {diag.get('stability_index')}")
        print(f"Recovery momentum: {diag.get('recovery_momentum')}")
    except Exception:
        print("Diagnostics computation failed, see logs for details.")

    # Write cycle history and run_state snapshots for diagnostics panel
    _write_cycles_and_run_state(
        agent,
        run_id=run_id,
        mode="swarm",
        goal=goal,
        domain=domain,
        cycles=_normalize_cycles_for_ui(summaries),
        diagnostics=diag,
    )

    intelligence_info = _run_post_run_intelligence(
        agent,
        mode="swarm",
        goal=goal,
        domain=domain,
        run_id=run_id,
        history=summaries,
    )

    extra_manifest: Dict[str, Any] = {
        "engine": "swarm",
        "roles": roles_list,
        "shift_minutes": shift_minutes,
        "repeat_shifts": repeat_shifts,
        "experiment_fingerprint": experiment_fingerprint,
    }
    if diag:
        extra_manifest["diagnostics_snapshot"] = diag
    if intelligence_info:
        extra_manifest["intelligence"] = intelligence_info

    try:
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
            extra=extra_manifest,
        )
    except Exception:
        print("Manifest logging failed, see logs for details.")
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
        experiment_mode="swarm_engine",
        extra={
            "experiment_fingerprint": experiment_fingerprint,
            "final_diagnostics": diag,
            "intelligence": intelligence_info if intelligence_info else None,
        },
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
            "avg_rye": diag.get("rye_avg"),
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

    Note: this is also finite-only; there is always a macro budget.
    """
    if total_budget_minutes is None or total_budget_minutes <= 0:
        total_budget_minutes = 60.0

    # Apply hard clamp for macro budget
    total_budget_minutes = _clamp_minutes(total_budget_minutes, "meta.total_budget") or 60.0

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
    _ = stats
    base_target = float(phase_cfg.get("target_minutes", 20.0))
    min_minutes = float(phase_cfg.get("min_minutes", 5.0))
    max_minutes_phase = float(phase_cfg.get("max_minutes", base_target))
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
            effective_minutes = min(max_minutes_phase, base_target * 1.2)
            effective_stop_rye = (base_stop_rye or 0.08) * 1.25

    if effective_minutes < min_minutes:
        effective_minutes = min_minutes
    if effective_minutes > max_minutes_phase:
        effective_minutes = max_minutes_phase

    effective_minutes = _clamp_minutes(effective_minutes, "meta.segment") or min_minutes

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

    Controlled by (finite-only):
    - WORKER_MAX_MINUTES (overall macro budget; if not set, derived from presets)
    - WORKER_META_MAX_SEGMENTS (max number of segments, default 6)
    - WORKER_RUNTIME_PROFILE (hint for phase profiles only)
    - WORKER_MODE / WORKER_SWARM (preferred mode: single vs swarm)
    - WORKER_SOURCES etc as in single and swarm engines
    """
    goal, domain = build_goal_and_domain()
    preset_cfg = get_preset(domain)

    total_budget_minutes_env = _env_float("WORKER_MAX_MINUTES")
    if total_budget_minutes_env is None:
        total_budget_minutes_env = float(preset_cfg.get("runtime_minutes", 60.0))
    total_budget_minutes = _clamp_minutes(total_budget_minutes_env, "meta.total_budget")
    if total_budget_minutes is None:
        total_budget_minutes = 60.0

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

    config_for_sources = dict(config)
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    tool_flags = detect_tools()
    run_id = _current_run_id("meta")

    env_keys_for_fingerprint = [
        "WORKER_MAX_MINUTES",
        "WORKER_STOP_RYE",
        "WORKER_RUNTIME_PROFILE",
        "WORKER_MODE",
        "WORKER_SWARM",
        "WORKER_META",
        "WORKER_SOURCES",
        "WORKER_SWARM_ROLES",
        "WORKER_DOMAIN",
        "WORKER_GOAL",
        "WORKER_META_MAX_SEGMENTS",
    ]
    experiment_fingerprint = _build_experiment_fingerprint(
        goal=goal,
        domain=domain,
        mode="meta",
        runtime_profile=runtime_profile_env,
        source_controls=source_controls,
        roles=None,
        env_keys=env_keys_for_fingerprint,
    )

    print(
        "=== Autonomous Research Engine - Meta Controller Mode "
        "(Option C, Finite Only) ==="
    )
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Run id: {run_id}")
    print(f"Preferred mode: {preferred_mode}")
    print(f"Total macro budget (minutes, clamped): {total_budget_minutes}")
    print(f"Max meta segments: {meta_max_segments_int}")
    print(f"Runtime profile hint (env): {runtime_profile_env or 'none'}")
    print(
        "Explicit stop RYE (env): "
        f"{stop_rye_env if stop_rye_env is not None else 'None'}"
    )
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    print(f"Experiment fingerprint: {experiment_fingerprint}")
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
        extra={"experiment_fingerprint": experiment_fingerprint},
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
        extra={"experiment_fingerprint": experiment_fingerprint},
    )

    for seg_index in range(meta_max_segments_int):
        time_left = total_budget_minutes - total_elapsed
        if time_left <= 1.0:
            print(
                f"[Meta] Time almost exhausted, "
                f"stopping before segment {seg_index + 1}."
            )
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

        effective_minutes = _clamp_minutes(effective_minutes, "meta.segment") or time_left

        if stop_rye_env is not None:
            effective_stop_rye = stop_rye_env

        print("")
        print(f"[Meta] Starting segment {seg_index + 1} / {meta_max_segments_int}")
        print(f"[Meta] Phase: {phase_name}")
        print(f"[Meta] Mode: {phase_mode}")
        print(f"[Meta] Segment minutes (requested, clamped): {effective_minutes:.2f}")
        print(
            "[Meta] Time left after this segment (approx): "
            f"{time_left - effective_minutes:.2f}"
        )
        print(
            "[Meta] Segment stop RYE (auto/explicit): "
            f"{effective_stop_rye if effective_stop_rye is not None else 'None'}"
        )
        print(
            f"[Meta] Phase runtime profile: "
            f"{phase_profile or 'preset default'}"
        )
        sys.stdout.flush()

        if phase_mode == "swarm":
            segment_summaries = agent.run_swarm_continuous(
                goal=goal,
                max_rounds=HARD_MAX_ROUNDS,
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
                max_cycles=HARD_MAX_CYCLES,
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
        print(
            "        total elapsed minutes: "
            f"{total_elapsed:.2f} / {total_budget_minutes:.2f}"
        )
        sys.stdout.flush()

        _log_milestone(
            agent,
            run_id=run_id,
            goal=goal,
            domain=domain,
            label=f"meta_segment_{seg_index + 1}",
            description=(
                f"Phase {phase_name} ({phase_mode}) finished with avg RYE "
                f"{seg_stats['avg_rye']} and stability {seg_stats['stability']}."
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

        if (
            recent_avg_rye is not None
            and recent_avg_rye < 0.01
            and total_elapsed > total_budget_minutes * 0.6
        ):
            print("[Meta] RYE collapsed and most of the budget is used. Stopping early.")
            break

    _heartbeat(agent, label="worker_meta_finished", run_id=run_id)

    total_segments = len(segments_run)
    total_summaries = sum(len(seg["summaries"]) for seg in segments_run)
    print("")
    print("=== Meta controller run finished (finite) ===")
    print(f"Segments executed: {total_segments}")
    print(f"Total summaries across segments: {total_summaries}")
    print(f"Final smoothed recent RYE: {recent_avg_rye}")
    print(
        "Total elapsed minutes (approx): "
        f"{total_elapsed:.2f} / {total_budget_minutes:.2f}"
    )
    sys.stdout.flush()

    combined_history: List[Dict[str, Any]] = []
    for seg in segments_run:
        combined_history.extend(seg.get("summaries", []))

    # Write aggregated cycle history and run_state snapshots for diagnostics panel
    _write_cycles_and_run_state(
        agent,
        run_id=run_id,
        mode="meta",
        goal=goal,
        domain=domain,
        cycles=_normalize_cycles_for_ui(combined_history),
        diagnostics=None,
    )

    intelligence_info = _run_post_run_intelligence(
        agent,
        mode="meta",
        goal=goal,
        domain=domain,
        run_id=run_id,
        history=combined_history,
    )

    extra_segments = [
        {
            "phase": seg.get("phase"),
            "mode": seg.get("mode"),
            "runtime_profile": seg.get("runtime_profile"),
            "minutes_requested": seg.get("minutes_requested"),
            "segment_items": len(seg.get("summaries", [])),
        }
        for seg in segments_run
    ]

    extra_manifest: Dict[str, Any] = {
        "engine": "meta",
        "segments": extra_segments,
        "experiment_fingerprint": experiment_fingerprint,
    }
    if intelligence_info:
        extra_manifest["intelligence"] = intelligence_info

    try:
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
            extra=extra_manifest,
        )
    except Exception:
        print("Manifest logging failed, see logs for details.")

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
        extra={
            "experiment_fingerprint": experiment_fingerprint,
            "final_recent_rye": recent_avg_rye,
            "total_segments": total_segments,
            "total_summaries": total_summaries,
            "intelligence": intelligence_info if intelligence_info else None,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Entry point for the background worker.

    Mode selection:
    - WORKER_QUEUE_MODE=1 (default) -> file-based queue worker using agent/run_jobs.
      If agent/run_jobs.py is missing or fails to import, queue mode logs an error
      and returns; the queue worker cannot start.
    - WORKER_QUEUE_MODE=0 -> direct engine behavior:
        - WORKER_META=1          -> run meta controller (Option C, finite-only)
        - WORKER_META=0 and WORKER_MODE=swarm or WORKER_SWARM=1 -> swarm engine (finite-only)
        - otherwise             -> single agent engine (finite-only)
    """
    print("Starting Autonomous Research Agent background engine (finite-only mode)...")
    print(
        f"[Safety] HARD_MAX_CYCLES={HARD_MAX_CYCLES}, "
        f"HARD_MAX_ROUNDS={HARD_MAX_ROUNDS}, "
        f"HARD_MAX_MINUTES={HARD_MAX_MINUTES}"
    )
    # Extra debug: show effective runs directory for this worker instance
    print(f"[engine_worker] ARA_RUNS_DIR env: {os.getenv('ARA_RUNS_DIR')!r}")
    try:
        print(f"[engine_worker] BASE_DIR (effective): {BASE_DIR.resolve()}")
    except Exception:
        print(f"[engine_worker] BASE_DIR (effective): {BASE_DIR}")
    sys.stdout.flush()

    queue_mode = _env_bool("WORKER_QUEUE_MODE", default=True)

    if queue_mode:
        if RunJob is None or load_next_pending_job is None:
            print(
                "Queue mode was requested (WORKER_QUEUE_MODE=1), "
                "but agent/run_jobs.py is missing or failed to import. "
                "Queue worker cannot start."
            )
            sys.stdout.flush()
            return
        run_job_queue_worker()
        return

    _configure_tavily_from_env()
    agent, config = init_agent_from_config()

    use_swarm = _env_bool("WORKER_SWARM", default=False)
    mode = os.getenv("WORKER_MODE", "single").strip().lower()
    # IMPORTANT: meta mode is now OFF by default. You must explicitly set
    # WORKER_META=1 to enable the Option C meta-controller.
    use_meta = _env_bool("WORKER_META", default=False)

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
    # Updated guard: if WORKER_MODE is empty, default to "queue" so the
    # queue worker runs and picks up jobs. Only explicit "off" values
    # prevent the worker from starting.
    raw_mode = os.getenv("WORKER_MODE", "").strip().lower()
    effective_mode = raw_mode or "queue"

    print(
        f"[engine_worker] WORKER_MODE raw={raw_mode!r} -> effective={effective_mode!r}"
    )
    sys.stdout.flush()

    if effective_mode in {"off", "disabled", "0", "none"}:
        print(
            "[engine_worker] WORKER_MODE is disabled. "
            "Exiting without starting worker."
        )
        sys.stdout.flush()
        sys.exit(0)

    if effective_mode == "queue":
        os.environ["WORKER_QUEUE_MODE"] = "1"

    main()
