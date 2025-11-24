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
