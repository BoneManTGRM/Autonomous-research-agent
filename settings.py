"""
Thin compatibility layer around AppConfig for older imports.

Older code sometimes expects:

    from settings import get_settings

New code should prefer:

    from config import build_app_config

This wrapper adds:
    - full cached AppConfig
    - learning-aware fields for the upgraded engine
    - expanded as_dict() for UI + logging panels
    - forward compatibility for swarm, memory growth, and domain presets
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

# New config system
from .config import AppConfig, build_app_config


# -------------------------------------------------------------------
# Cached singleton settings instance
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """
    Return the single shared AppConfig instance.

    Cached because:
      - workers read settings repeatedly
      - Streamlit reloads modules on every request
      - prevents re-parsing YAML many times
    """
    return build_app_config()


# -------------------------------------------------------------------
# Convert settings to a plain dictionary
# (used by UI panels, debugging routes, worker logging, snapshots)
# -------------------------------------------------------------------

def as_dict() -> Dict[str, Any]:
    """
    Expose settings as a dictionary.

    Includes:
      - all core paths
      - worker defaults
      - learning-aware toggles
      - domain presets
      - source controls
      - model selection
      - Tavily configuration
    """
    cfg = get_settings()

    return {
        # ----- Paths -----
        "settings_path": str(cfg.settings_path),
        "memory_file": str(cfg.memory_file),
        "log_dir": str(cfg.log_dir),
        "session_dir": str(cfg.session_dir),

        # ----- Core defaults -----
        "default_worker_goal": cfg.default_worker_goal,
        "default_worker_domain": cfg.default_worker_domain,
        "runtime_minutes": cfg.runtime_minutes,

        # ----- AI model settings -----
        "openai_model": cfg.openai_model,

        # ----- Tavily -----
        "tavily_enabled": cfg.tavily_enabled,
        "tavily_max_cost_usd": cfg.tavily_max_cost_usd,

        # ----- Source Controls -----
        "default_source_controls": dict(cfg.default_source_controls),

        # ----- Learning-aware flags -----
        # Used by CoreAgent + MemoryStore to enable long-term learning
        "learning_enabled": True,
        "learning_memory_key": "learning_log",
        "learning_min_examples": 3,
        "learning_weight_recent": 0.65,
        "learning_weight_prior": 0.35,

        # ----- Domain tuning -----
        # Exposed for UI + swarm intelligence
        "domain_defaults": {
            "general": "Broad exploration of Reparodynamics, RYE, TGRM, stability systems.",
            "longevity": "Biomarkers, PubMed, anti-aging fundamentals, geroscience.",
            "math": "Formalism, definitions, theorems, proofs, structure search."
        },

        # Include raw YAML so nothing gets lost
        "raw_settings": cfg.raw_settings,
    }


# -------------------------------------------------------------------
# Extended, AGI-style capability view (bolt-on only)
# -------------------------------------------------------------------

def _build_capability_manifest(cfg: AppConfig) -> Dict[str, Any]:
    """
    Human-readable capability summary for UI, debugging, and lay users.

    This does NOT change behavior anywhere. It is a descriptive layer:
      - what model is configured
      - whether Tavily and long-run worker are enabled
      - whether learning and verification scaffolding exist
      - whether the stack is swarm-ready and discovery-ready
    """
    # Long-run ready: anything at or above one hour runtime by default
    long_run_ready = bool(cfg.runtime_minutes and cfg.runtime_minutes >= 60.0)

    exploratory_learning = cfg.learning_profile in {"balanced", "aggressive"}
    conservative_learning = cfg.learning_profile == "conservative"

    raw = cfg.raw_settings if isinstance(cfg.raw_settings, dict) else {}

    # Optional breakthrough / swarm knobs from YAML if present
    breakthrough_raw = raw.get("breakthroughs") if isinstance(raw, dict) else None
    if not isinstance(breakthrough_raw, dict):
        breakthrough_raw = {}

    swarm_raw = raw.get("swarm") if isinstance(raw, dict) else None
    if not isinstance(swarm_raw, dict):
        swarm_raw = {}

    breakthroughs_enabled = bool(breakthrough_raw.get("enabled", True))
    breakthrough_min_rye_gain = float(breakthrough_raw.get("min_rye_gain", 0.10))
    breakthrough_min_repeats = int(breakthrough_raw.get("repeats_required", 2))

    return {
        # Core engine / model
        "model_name": cfg.openai_model,
        "tavily_enabled": cfg.tavily_enabled,
        "tavily_max_cost_usd": cfg.tavily_max_cost_usd,

        # Autonomy + worker
        "supports_long_run_worker": long_run_ready,
        "default_runtime_minutes": cfg.runtime_minutes,

        # Learning behavior (how much the system can adapt over time)
        "learning_profile": cfg.learning_profile,
        "exploratory_learning": exploratory_learning,
        "conservative_learning": conservative_learning,
        "learning_rate_base": cfg.learning_rate_base,
        "learning_rate_max": cfg.learning_rate_max,
        "memory_compaction_rate": cfg.memory_compaction_rate,
        "coherence_threshold": cfg.coherence_threshold,
        "forgetting_half_life_minutes": cfg.forgetting_half_life_minutes,

        # Swarm + multi-agent posture (descriptive only)
        "ready_for_swarm_mode": True,   # codebase supports swarm even if YAML has no swarm block
        "swarm_enabled_by_default": bool(swarm_raw.get("enabled_by_default", False)),
        "swarm_max_agents_configured": int(swarm_raw.get("max_agents", 32)),
        "swarm_default_mode": swarm_raw.get("default_mode", "single_or_pair"),

        # Discovery + verification posture (descriptive)
        "discovery_logging_configured": True,
        "verification_logging_configured": True,
        "breakthrough_detection_enabled": breakthroughs_enabled,
        "breakthrough_min_rye_gain": breakthrough_min_rye_gain,
        "breakthrough_min_repeats": breakthrough_min_repeats,

        # High level: “is this build ready for 90-day style runs?”
        # (This is a qualitative flag for UI + docs, not a guarantee.)
        "ready_for_90_day_experiments": bool(long_run_ready and cfg.tavily_enabled),
    }


def as_rich_dict() -> Dict[str, Any]:
    """
    Extended settings dictionary for advanced panels and logs.

    This is a superset of as_dict() and adds:

      - learning_config:  how the engine updates itself over time
      - swarm_config:     descriptive defaults for swarm behavior (if defined in YAML)
      - logs_config:      paths to discovery + verification logs
      - capability_manifest: a lay-friendly “what can this build do?” summary

    Use this in:
      - Streamlit diagnostic panels
      - PDF reporters
      - snapshot / equilibrium viewers
      - external dashboards
    """
    cfg = get_settings()
    base = as_dict()

    # Learning configuration (clean, engine-wide view)
    learning_config: Dict[str, Any] = {
        "profile": cfg.learning_profile,
        "weights": dict(cfg.learning_weights),
        "learning_rate_base": cfg.learning_rate_base,
        "learning_rate_max": cfg.learning_rate_max,
        "memory_compaction_rate": cfg.memory_compaction_rate,
        "coherence_threshold": cfg.coherence_threshold,
        "forgetting_half_life_minutes": cfg.forgetting_half_life_minutes,
    }

    # Optional swarm block from YAML (if present)
    raw = cfg.raw_settings if isinstance(cfg.raw_settings, dict) else {}
    swarm_raw = raw.get("swarm") if isinstance(raw, dict) else None
    if not isinstance(swarm_raw, dict):
        swarm_raw = {}

    swarm_config: Dict[str, Any] = {
        "enabled_by_default": bool(swarm_raw.get("enabled_by_default", False)),
        "max_agents": int(swarm_raw.get("max_agents", 32)),
        "default_mode": swarm_raw.get("default_mode", "single_or_pair"),
        "notes": (
            "These values describe how many agents a typical swarm run might use. "
            "Actual limits are enforced inside the Streamlit UI and CoreAgent."
        ),
    }

    # Log paths for discovery + verification
    logs_config: Dict[str, Any] = {
        "discovery_log_file": str(cfg.discovery_log_file),
        "verification_log_file": str(cfg.verification_log_file),
    }

    # High-level capability manifest (for humans)
    capability_manifest = _build_capability_manifest(cfg)

    base.update(
        {
            "learning_config": learning_config,
            "swarm_config": swarm_config,
            "logs_config": logs_config,
            "capability_manifest": capability_manifest,
        }
    )
    return base
