"""
Central configuration helpers for the Autonomous Research Agent.

This module keeps all environment and path handling in one place so that:
- Streamlit UI
- background worker (engine_worker.py)
- maintenance scripts
- learning subsystems (memory -> model adaptation)

can all share the same defaults.

It does NOT depend on Streamlit, so it is safe to import from workers and scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Default paths
DEFAULT_SETTINGS_PATH = Path("config") / "settings.yaml"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_SESSION_DIR = DEFAULT_LOG_DIR / "sessions"
DEFAULT_MEMORY_FILE = DEFAULT_SESSION_DIR / "default_memory.json"

DEFAULT_DISCOVERY_LOG = DEFAULT_LOG_DIR / "discovery" / "discovery_log.json"
DEFAULT_VERIFICATION_LOG = DEFAULT_LOG_DIR / "verification" / "verification_log.json"
DEFAULT_CONTROL_STATE_FILE = DEFAULT_LOG_DIR / "control_state.json"
DEFAULT_SNAPSHOTS_DIR = DEFAULT_LOG_DIR / "snapshots"


@dataclass
class AppConfig:
    """Runtime configuration snapshot for the whole agent stack."""

    # Paths
    settings_path: Path = DEFAULT_SETTINGS_PATH
    log_dir: Path = DEFAULT_LOG_DIR
    session_dir: Path = DEFAULT_SESSION_DIR
    memory_file: Path = DEFAULT_MEMORY_FILE

    # Structured discovery + verification logs
    discovery_log_file: Path = DEFAULT_DISCOVERY_LOG
    verification_log_file: Path = DEFAULT_VERIFICATION_LOG

    # Engine coordination files
    control_state_file: Path = DEFAULT_CONTROL_STATE_FILE
    snapshots_dir: Path = DEFAULT_SNAPSHOTS_DIR

    # Core toggles
    openai_model: str = "gpt-4.1"
    tavily_enabled: bool = True
    tavily_max_cost_usd: float = 1.0

    # Worker defaults
    default_worker_goal: str = (
        "Long run autonomous research on reparodynamics, RYE, TGRM, and stability science."
    )
    default_worker_domain: str = "general"
    runtime_minutes: float = 60.0

    # Source controls
    default_source_controls: Dict[str, bool] = field(
        default_factory=lambda: {
            "web": True,
            "pubmed": True,
            "semantic": True,
            "pdf": True,
            "biomarkers": False,
            "sandbox": False,
        }
    )

    # NEW - Global learning controls (maxed out)
    learning_profile: str = "balanced"  # conservative / balanced / aggressive
    learning_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "summaries": 1.0,
            "discoveries": 1.5,
            "verification_validated": 2.0,
            "verification_rejected": 0.5,
            "cross_run_history": 1.0,
        }
    )
    learning_rate_base: float = 0.10       # base learning rate
    learning_rate_max: float = 0.35        # max RYE-boosted learning rate

    # NEW - memory compaction and noise filters
    memory_compaction_rate: float = 0.10
    coherence_threshold: float = 0.25
    forgetting_half_life_minutes: float = 600.0  # 10h half-life

    # NEW - long run and breakthrough classification
    breakthrough_rye_threshold: float = 0.5      # RYE threshold that suggests a strong intervention
    breakthrough_min_discoveries: int = 3       # minimum discoveries before calling it a breakthrough
    noble_run_minutes: float = 90.0 * 24.0 * 60.0  # 90 days in minutes, used for "Nobel-tier" classification

    # NEW - worker watchdog and snapshot cadence
    watchdog_interval_seconds: float = 60.0     # how often the worker updates heartbeat
    watchdog_timeout_seconds: float = 600.0     # time without heartbeat before "stalled"
    snapshot_interval_minutes: float = 60.0     # how often to write snapshot files on long runs

    # Raw YAML
    raw_settings: Dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return float(val)
    except Exception:
        return default


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return int(val)
    except Exception:
        return default


def build_app_config(settings_path: Optional[str] = None) -> AppConfig:
    """Create a unified AppConfig from defaults + YAML + environment."""

    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    raw = _load_yaml(path)

    cfg = AppConfig()
    cfg.settings_path = path
    cfg.raw_settings = raw

    # Paths from YAML
    memory_yaml = raw.get("memory_file")
    if isinstance(memory_yaml, str):
        cfg.memory_file = Path(memory_yaml)

    discovery_yaml = raw.get("discovery_log_file")
    if isinstance(discovery_yaml, str):
        cfg.discovery_log_file = Path(discovery_yaml)

    verification_yaml = raw.get("verification_log_file")
    if isinstance(verification_yaml, str):
        cfg.verification_log_file = Path(verification_yaml)

    control_state_yaml = raw.get("control_state_file")
    if isinstance(control_state_yaml, str):
        cfg.control_state_file = Path(control_state_yaml)

    snapshots_yaml = raw.get("snapshots_dir")
    if isinstance(snapshots_yaml, str):
        cfg.snapshots_dir = Path(snapshots_yaml)

    # Paths from environment
    mem_env = os.getenv("MEMORY_FILE")
    if mem_env:
        cfg.memory_file = Path(mem_env)

    disc_env = os.getenv("DISCOVERY_LOG_FILE")
    if disc_env:
        cfg.discovery_log_file = Path(disc_env)

    ver_env = os.getenv("VERIFICATION_LOG_FILE")
    if ver_env:
        cfg.verification_log_file = Path(ver_env)

    control_state_env = os.getenv("CONTROL_STATE_FILE")
    if control_state_env:
        cfg.control_state_file = Path(control_state_env)

    snapshots_env = os.getenv("SNAPSHOTS_DIR")
    if snapshots_env:
        cfg.snapshots_dir = Path(snapshots_env)

    # Ensure directories
    cfg.log_dir.mkdir(exist_ok=True)
    cfg.session_dir.mkdir(parents=True, exist_ok=True)
    cfg.discovery_log_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.verification_log_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.control_state_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Worker defaults from YAML (top level)
    if isinstance(raw.get("default_worker_goal"), str):
        cfg.default_worker_goal = raw["default_worker_goal"]
    if isinstance(raw.get("default_worker_domain"), str):
        cfg.default_worker_domain = raw["default_worker_domain"]

    if isinstance(raw.get("runtime_minutes"), (int, float)):
        cfg.runtime_minutes = float(raw["runtime_minutes"])

    # Source controls
    ds = raw.get("default_source_controls")
    if isinstance(ds, dict):
        merged = cfg.default_source_controls.copy()
        for k, v in ds.items():
            merged[str(k)] = bool(v)
        cfg.default_source_controls = merged

    # Learning overrides from YAML
    ls = raw.get("learning")
    if isinstance(ls, dict):
        if isinstance(ls.get("profile"), str):
            cfg.learning_profile = ls["profile"]

        if isinstance(ls.get("weights"), dict):
            cfg.learning_weights.update({
                k: float(v) for k, v in ls["weights"].items()
            })

        if isinstance(ls.get("learning_rate_base"), (int, float)):
            cfg.learning_rate_base = float(ls["learning_rate_base"])

        if isinstance(ls.get("learning_rate_max"), (int, float)):
            cfg.learning_rate_max = float(ls["learning_rate_max"])

        if isinstance(ls.get("memory_compaction_rate"), (int, float)):
            cfg.memory_compaction_rate = float(ls["memory_compaction_rate"])

        if isinstance(ls.get("coherence_threshold"), (int, float)):
            cfg.coherence_threshold = float(ls["coherence_threshold"])

        if isinstance(ls.get("forgetting_half_life_minutes"), (int, float)):
            cfg.forgetting_half_life_minutes = float(ls["forgetting_half_life_minutes"])

    # Engine and breakthrough tuning from YAML
    eng = raw.get("engine")
    if isinstance(eng, dict):
        if isinstance(eng.get("runtime_minutes"), (int, float)):
            cfg.runtime_minutes = float(eng["runtime_minutes"])

        if isinstance(eng.get("breakthrough_rye_threshold"), (int, float)):
            cfg.breakthrough_rye_threshold = float(eng["breakthrough_rye_threshold"])

        if isinstance(eng.get("breakthrough_min_discoveries"), int):
            cfg.breakthrough_min_discoveries = int(eng["breakthrough_min_discoveries"])

        if isinstance(eng.get("noble_run_minutes"), (int, float)):
            cfg.noble_run_minutes = float(eng["noble_run_minutes"])

        if isinstance(eng.get("watchdog_interval_seconds"), (int, float)):
            cfg.watchdog_interval_seconds = float(eng["watchdog_interval_seconds"])

        if isinstance(eng.get("watchdog_timeout_seconds"), (int, float)):
            cfg.watchdog_timeout_seconds = float(eng["watchdog_timeout_seconds"])

        if isinstance(eng.get("snapshot_interval_minutes"), (int, float)):
            cfg.snapshot_interval_minutes = float(eng["snapshot_interval_minutes"])

    # Environment overrides
    model_env = os.getenv("OPENAI_MODEL")
    if model_env:
        cfg.openai_model = model_env

    cfg.tavily_enabled = _env_bool("TAVILY_ENABLED", cfg.tavily_enabled)

    cap = _env_float("TAVILY_MAX_COST_USD", cfg.tavily_max_cost_usd)
    if cap is not None:
        cfg.tavily_max_cost_usd = cap

    goal_env = os.getenv("WORKER_GOAL_DEFAULT")
    if goal_env:
        cfg.default_worker_goal = goal_env

    domain_env = os.getenv("WORKER_DOMAIN_DEFAULT")
    if domain_env:
        cfg.default_worker_domain = domain_env

    rt = _env_float("RUNTIME_MINUTES_DEFAULT", None)
    if rt is not None and rt > 0:
        cfg.runtime_minutes = rt

    # Environment overrides for breakthrough and watchdog
    thr_env = _env_float("BREAKTHROUGH_RYE_THRESHOLD", None)
    if thr_env is not None and thr_env > 0:
        cfg.breakthrough_rye_threshold = thr_env

    min_disc_env = _env_int("BREAKTHROUGH_MIN_DISCOVERIES", None)
    if min_disc_env is not None and min_disc_env > 0:
        cfg.breakthrough_min_discoveries = min_disc_env

    noble_env = _env_float("NOBLE_RUN_MINUTES", None)
    if noble_env is not None and noble_env > 0:
        cfg.noble_run_minutes = noble_env

    wd_int_env = _env_float("WATCHDOG_INTERVAL_SECONDS", None)
    if wd_int_env is not None and wd_int_env > 0:
        cfg.watchdog_interval_seconds = wd_int_env

    wd_to_env = _env_float("WATCHDOG_TIMEOUT_SECONDS", None)
    if wd_to_env is not None and wd_to_env > 0:
        cfg.watchdog_timeout_seconds = wd_to_env

    snap_env = _env_float("SNAPSHOT_INTERVAL_MINUTES", None)
    if snap_env is not None and snap_env > 0:
        cfg.snapshot_interval_minutes = snap_env

    return cfg
