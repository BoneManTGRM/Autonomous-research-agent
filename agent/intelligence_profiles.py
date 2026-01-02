"""
intelligence_profiles.py

Intelligence profile definitions for the Autonomous Research Agent.

Purpose
    Central place to describe "how smart" and "how aggressive"
    different runs should behave, without hard coding behavior
    inside CoreAgent or the worker.

    A profile is a structured bundle of hints that can be used by:
      - CoreAgent
      - engine_worker
      - discovery_engine
      - verification_engine
      - swarm orchestration

    Profiles here are tuned for:
      - stable long runs
      - strong discovery odds
      - high verification quality
      - maximum Tier 3 style breakthrough chances while staying sane

    This file is pure configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

INTELLIGENCE_VERSION: str = "2025-11-24-max2"


# -------------------------------------------------------------------
# Global expectations for advanced RYE metrics
# These mirror the shapes used in presets and memory_store.
# -------------------------------------------------------------------
DEFAULT_ADVANCED_RYE_EXPECTATIONS: Dict[str, Any] = {
    "rolling_window_short": 10,
    "rolling_window_long": 50,
    "stability_index_target": 0.6,       # 0 to 1 scale
    "recovery_momentum_target": 0.1,     # positive values show recovery after perturbation
    "max_oscillation_std": 0.25,
}

LONG_RUN_ADVANCED_RYE_EXPECTATIONS: Dict[str, Any] = {
    "rolling_window_short": 10,
    "rolling_window_long": 100,
    "stability_index_target": 0.55,
    "recovery_momentum_target": 0.15,
    "max_oscillation_std": 0.30,
}

MATH_ADVANCED_RYE_EXPECTATIONS: Dict[str, Any] = {
    "rolling_window_short": 10,
    "rolling_window_long": 40,
    "stability_index_target": 0.65,
    "recovery_momentum_target": 0.08,
    "max_oscillation_std": 0.20,
}


# -------------------------------------------------------------------
# Profile schema
# -------------------------------------------------------------------
# Each profile entry uses the following structure:
#
#   {
#       "label": str,
#       "description": str,
#
#       # Global run behavior hints
#       "depth": "shallow" | "balanced" | "deep" | "ultra_deep",
#       "exploration": float,              # 0 to 1
#       "verification_intensity": float,   # 0 to 1
#       "discovery_focus": float,          # 0 to 1
#       "tier3_bias": float,               # 0 to 1, tuned for major discoveries
#
#       # Autonomy and swarm hints
#       "swarm_recommended": bool,
#       "max_parallel_agents_hint": int,
#       "preferred_swarm_size": int,
#       "use_classic_pair": bool,          # researcher plus critic
#
#       # Engine subsystem hints
#       "use_discovery_engine": bool,
#       "use_verification_engine": bool,
#       "use_meta_controller": bool,
#
#       # Long run behavior
#       "segment_minutes_hint": int,
#       "max_segments_hint": int,
#       "checkpoint_frequency_cycles": int,
#       "checkpoint_frequency_minutes": int,
#
#       # Risk and safety controls
#       "safety_bias": "high" | "medium" | "low",
#       "hallucination_tolerance": float,  # 0 to 1, lower is stricter
#       "require_citations_level": "none" | "normal" | "strict" | "clinical",
#
#       # RYE and Reparodynamics emphasis
#       "rye_weight_factor": float,        # multiplier for RYE in decision making
#       "equilibrium_focus": float,        # 0 to 1, bias toward stability
#       "disruption_budget": float,        # 0 to 1, cycles allowed for risky exploration
#
#       # Advanced RYE expectations and learning behavior
#       "advanced_rye_expectations": dict,
#       "learning_hints": {
#           "use_advanced_rye_metrics": bool,
#           "switch_to_maintenance_if": {...},
#           "trigger_exploration_if": {...},
#       },
#
#       # Swarm behavior preferences (to coordinate with presets swarm_orchestration)
#       "swarm_behavior_hints": {
#           "rotation_strategy": str,      # round_robin, weighted, priority
#           "consensus_model": str,        # majority, weighted_vote, single_best
#       },
#
#       # Diagnostics hints to match memory_store RYE metrics
#       "diagnostics_hints": {
#           "run_diagnostics": bool,
#           "window": int,
#           "frequency_cycles": int,
#           "include_stability_index": bool,
#           "include_recovery_momentum": bool,
#       },
#
#       # Tool usage bias hints that CoreAgent can line up with preset tool_intelligence
#       "tool_bias": {
#           "browser_mode": "conservative" | "normal" | "aggressive" | "adaptive",
#           "sandbox_emphasis": float,     # 0 to 1
#           "data_pipeline_emphasis": float,
#       },
#
#       # Recommended preset and runtime pairing for this profile
#       "preset_recommendations": {
#           "default_preset": "general" | "longevity" | "math",
#           "default_runtime_profile": "1_hour" | "8_hours" | "24_hours" | "90_days" | "forever",
#       },
#   }
#
# These are hints, not hard rules. The engine should interpret them
# as soft guidance when scheduling cycles, choosing tools, and
# deciding how hard to push verification.
# -------------------------------------------------------------------


INTELLIGENCE_PROFILES: Dict[str, Dict[str, Any]] = {
    # ---------------------------------------------------------------
    # Baseline reference profile
    # ---------------------------------------------------------------
    "baseline": {
        "label": "Baseline intelligence",
        "description": (
            "Balanced intelligence profile tuned for general research. "
            "Moderate exploration, moderate verification, safe behavior, "
            "and a good default for early testing and single agent runs."
        ),
        "depth": "balanced",
        "exploration": 0.45,
        "verification_intensity": 0.55,
        "discovery_focus": 0.50,
        "tier3_bias": 0.20,
        "swarm_recommended": False,
        "max_parallel_agents_hint": 4,
        "preferred_swarm_size": 3,
        "use_classic_pair": True,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 30,
        "max_segments_hint": 6,
        "checkpoint_frequency_cycles": 50,
        "checkpoint_frequency_minutes": 20,
        "safety_bias": "high",
        "hallucination_tolerance": 0.10,
        "require_citations_level": "normal",
        "rye_weight_factor": 1.0,
        "equilibrium_focus": 0.50,
        "disruption_budget": 0.20,
        "advanced_rye_expectations": DEFAULT_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.10,
                "stability_index_above": 0.55,
                "oscillation_std_below": 0.22,
                "min_cycles": 25,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": 0.0,
                "window": 20,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "round_robin",
            "consensus_model": "weighted_vote",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 25,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "adaptive",
            "sandbox_emphasis": 0.5,
            "data_pipeline_emphasis": 0.5,
        },
        "preset_recommendations": {
            "default_preset": "general",
            "default_runtime_profile": "8_hours",
        },
    },

    # ---------------------------------------------------------------
    # Conservative profile
    # ---------------------------------------------------------------
    "conservative": {
        "label": "Conservative and safe",
        "description": (
            "High safety, strict verification, low hallucination tolerance. "
            "Use when correctness and robustness matter more than novel discovery."
        ),
        "depth": "deep",
        "exploration": 0.25,
        "verification_intensity": 0.85,
        "discovery_focus": 0.35,
        "tier3_bias": 0.10,
        "swarm_recommended": False,
        "max_parallel_agents_hint": 3,
        "preferred_swarm_size": 2,
        "use_classic_pair": True,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 45,
        "max_segments_hint": 8,
        "checkpoint_frequency_cycles": 40,
        "checkpoint_frequency_minutes": 15,
        "safety_bias": "high",
        "hallucination_tolerance": 0.05,
        "require_citations_level": "strict",
        "rye_weight_factor": 1.2,
        "equilibrium_focus": 0.70,
        "disruption_budget": 0.10,
        "advanced_rye_expectations": DEFAULT_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.12,
                "stability_index_above": 0.60,
                "oscillation_std_below": 0.20,
                "min_cycles": 30,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": 0.0,
                "window": 30,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "round_robin",
            "consensus_model": "majority",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 20,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "conservative",
            "sandbox_emphasis": 0.4,
            "data_pipeline_emphasis": 0.4,
        },
        "preset_recommendations": {
            "default_preset": "general",
            "default_runtime_profile": "24_hours",
        },
    },

    # ---------------------------------------------------------------
    # Aggressive tier 2 style discovery hunter
    # ---------------------------------------------------------------
    "aggressive_discovery": {
        "label": "Aggressive discovery mode",
        "description": (
            "High exploration, strong discovery focus, still coupled to verification. "
            "Good for pushing toward Tier 2 style novel findings while staying inside sane bounds."
        ),
        "depth": "deep",
        "exploration": 0.80,
        "verification_intensity": 0.65,
        "discovery_focus": 0.85,
        "tier3_bias": 0.55,
        "swarm_recommended": True,
        "max_parallel_agents_hint": 12,
        "preferred_swarm_size": 5,
        "use_classic_pair": False,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 60,
        "max_segments_hint": 10,
        "checkpoint_frequency_cycles": 30,
        "checkpoint_frequency_minutes": 15,
        "safety_bias": "medium",
        "hallucination_tolerance": 0.18,
        "require_citations_level": "normal",
        "rye_weight_factor": 1.4,
        "equilibrium_focus": 0.45,
        "disruption_budget": 0.40,
        "advanced_rye_expectations": DEFAULT_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.11,
                "stability_index_above": 0.52,
                "oscillation_std_below": 0.25,
                "min_cycles": 40,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": -0.02,
                "window": 25,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "weighted",
            "consensus_model": "weighted_vote",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 20,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "aggressive",
            "sandbox_emphasis": 0.6,
            "data_pipeline_emphasis": 0.6,
        },
        "preset_recommendations": {
            "default_preset": "general",
            "default_runtime_profile": "24_hours",
        },
    },

    # ---------------------------------------------------------------
    # Tier 3 hunter profile
    # Very aggressive for breakthroughs but with heavy verification
    # and RYE based sanity checks.
    # ---------------------------------------------------------------
    "tier3_hunter": {
        "label": "Tier 3 hunter",
        "description": (
            "Maxed out intelligence profile tuned for Tier 3 level discoveries. "
            "Designed for long runs with swarm, discovery engine, verification engine, "
            "and meta controller fully engaged. Very high exploration and discovery focus, "
            "anchored by strict verification and RYE based sanity checks."
        ),
        "depth": "ultra_deep",
        "exploration": 0.88,
        "verification_intensity": 0.85,
        "discovery_focus": 0.95,
        "tier3_bias": 0.95,
        "swarm_recommended": True,
        "max_parallel_agents_hint": 24,
        "preferred_swarm_size": 8,
        "use_classic_pair": False,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 90,
        "max_segments_hint": 16,
        "checkpoint_frequency_cycles": 25,
        "checkpoint_frequency_minutes": 10,
        "safety_bias": "medium",
        "hallucination_tolerance": 0.15,
        "require_citations_level": "clinical",
        "rye_weight_factor": 1.8,
        "equilibrium_focus": 0.55,
        "disruption_budget": 0.45,
        "advanced_rye_expectations": LONG_RUN_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.14,
                "stability_index_above": 0.55,
                "oscillation_std_below": 0.30,
                "min_cycles": 80,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": -0.03,
                "window": 40,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "weighted",
            "consensus_model": "weighted_vote",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 20,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "aggressive",
            "sandbox_emphasis": 0.7,
            "data_pipeline_emphasis": 0.7,
        },
        "preset_recommendations": {
            "default_preset": "general",
            "default_runtime_profile": "90_days",
        },
    },

    # ---------------------------------------------------------------
    # Longevity clinical profile
    # For anti aging healthspan research and candidate interventions.
    # ---------------------------------------------------------------
    "longevity_clinical": {
        "label": "Longevity clinical intelligence",
        "description": (
            "Clinical grade profile for longevity and anti aging work. "
            "Heavy verification, strict citation requirements, and strong RYE emphasis "
            "on biomarkers and real world evidence. Good candidate for biotech or pharma interest."
        ),
        "depth": "deep",
        "exploration": 0.55,
        "verification_intensity": 0.90,
        "discovery_focus": 0.80,
        "tier3_bias": 0.70,
        "swarm_recommended": True,
        "max_parallel_agents_hint": 16,
        "preferred_swarm_size": 5,
        "use_classic_pair": True,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 60,
        "max_segments_hint": 12,
        "checkpoint_frequency_cycles": 30,
        "checkpoint_frequency_minutes": 15,
        "safety_bias": "high",
        "hallucination_tolerance": 0.08,
        "require_citations_level": "clinical",
        "rye_weight_factor": 1.6,
        "equilibrium_focus": 0.65,
        "disruption_budget": 0.30,
        "advanced_rye_expectations": LONG_RUN_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.10,
                "stability_index_above": 0.50,
                "oscillation_std_below": 0.28,
                "min_cycles": 60,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": 0.0,
                "window": 30,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "weighted",
            "consensus_model": "weighted_vote",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 20,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "aggressive",
            "sandbox_emphasis": 0.6,
            "data_pipeline_emphasis": 0.8,
        },
        "preset_recommendations": {
            "default_preset": "longevity",
            "default_runtime_profile": "24_hours",
        },
    },

    # ---------------------------------------------------------------
    # Math and formalization profile
    # ---------------------------------------------------------------
    "math_formal": {
        "label": "Math and formal theory intelligence",
        "description": (
            "Precision profile for mathematical and formal theoretical work. "
            "Emphasizes structural consistency, proof checking, and low tolerance "
            "for speculative claims without clear reasoning."
        ),
        "depth": "deep",
        "exploration": 0.40,
        "verification_intensity": 0.92,
        "discovery_focus": 0.65,
        "tier3_bias": 0.60,
        "swarm_recommended": True,
        "max_parallel_agents_hint": 6,
        "preferred_swarm_size": 3,
        "use_classic_pair": True,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 75,
        "max_segments_hint": 10,
        "checkpoint_frequency_cycles": 35,
        "checkpoint_frequency_minutes": 20,
        "safety_bias": "high",
        "hallucination_tolerance": 0.06,
        "require_citations_level": "strict",
        "rye_weight_factor": 1.3,
        "equilibrium_focus": 0.70,
        "disruption_budget": 0.20,
        "advanced_rye_expectations": MATH_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.06,
                "stability_index_above": 0.60,
                "oscillation_std_below": 0.18,
                "min_cycles": 40,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": 0.0,
                "window": 25,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "round_robin",
            "consensus_model": "majority",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 30,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "conservative",
            "sandbox_emphasis": 0.7,
            "data_pipeline_emphasis": 0.4,
        },
        "preset_recommendations": {
            "default_preset": "math",
            "default_runtime_profile": "8_hours",
        },
    },

    # ---------------------------------------------------------------
    # Exploration heavy swarm profile
    # Good for early phase sweeps and wide literature mapping.
    # ---------------------------------------------------------------
    "exploration_swarm": {
        "label": "Exploration heavy swarm",
        "description": (
            "Wide sweep swarm profile for mapping large literatures and adjacent domains. "
            "High exploration and swarm size, followed by later verification passes. "
            "Ideal for early phases of a long run before tightening into Tier 3 hunter."
        ),
        "depth": "balanced",
        "exploration": 0.95,
        "verification_intensity": 0.55,
        "discovery_focus": 0.90,
        "tier3_bias": 0.65,
        "swarm_recommended": True,
        "max_parallel_agents_hint": 32,
        "preferred_swarm_size": 10,
        "use_classic_pair": False,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 45,
        "max_segments_hint": 16,
        "checkpoint_frequency_cycles": 40,
        "checkpoint_frequency_minutes": 15,
        "safety_bias": "medium",
        "hallucination_tolerance": 0.22,
        "require_citations_level": "normal",
        "rye_weight_factor": 1.2,
        "equilibrium_focus": 0.40,
        "disruption_budget": 0.55,
        "advanced_rye_expectations": DEFAULT_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.09,
                "stability_index_above": 0.50,
                "oscillation_std_below": 0.28,
                "min_cycles": 50,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": False,
                "recovery_momentum_below": -0.05,
                "window": 30,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "weighted",
            "consensus_model": "weighted_vote",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 30,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "aggressive",
            "sandbox_emphasis": 0.5,
            "data_pipeline_emphasis": 0.7,
        },
        "preset_recommendations": {
            "default_preset": "general",
            "default_runtime_profile": "24_hours",
        },
    },

    # ---------------------------------------------------------------
    # Ultra stable long run guardian profile
    # Great for proving 90 day continuous stability.
    # ---------------------------------------------------------------
    "guardian_long_run": {
        "label": "Guardian long run stability",
        "description": (
            "Stability first profile meant to maximize the chance of clean, uninterrupted "
            "long runs. Emphasizes equilibrium, stability, and RYE plateaus, with a "
            "smaller but consistent discovery budget."
        ),
        "depth": "deep",
        "exploration": 0.40,
        "verification_intensity": 0.80,
        "discovery_focus": 0.55,
        "tier3_bias": 0.45,
        "swarm_recommended": True,
        "max_parallel_agents_hint": 8,
        "preferred_swarm_size": 4,
        "use_classic_pair": True,
        "use_discovery_engine": True,
        "use_verification_engine": True,
        "use_meta_controller": True,
        "segment_minutes_hint": 120,
        "max_segments_hint": 24,
        "checkpoint_frequency_cycles": 20,
        "checkpoint_frequency_minutes": 15,
        "safety_bias": "high",
        "hallucination_tolerance": 0.10,
        "require_citations_level": "strict",
        "rye_weight_factor": 1.7,
        "equilibrium_focus": 0.80,
        "disruption_budget": 0.25,
        "advanced_rye_expectations": LONG_RUN_ADVANCED_RYE_EXPECTATIONS,
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.10,
                "stability_index_above": 0.55,
                "oscillation_std_below": 0.28,
                "min_cycles": 80,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": -0.02,
                "window": 40,
            },
        },
        "swarm_behavior_hints": {
            "rotation_strategy": "round_robin",
            "consensus_model": "majority",
        },
        "diagnostics_hints": {
            "run_diagnostics": True,
            "window": 10,
            "frequency_cycles": 30,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },
        "tool_bias": {
            "browser_mode": "adaptive",
            "sandbox_emphasis": 0.6,
            "data_pipeline_emphasis": 0.6,
        },
        "preset_recommendations": {
            "default_preset": "general",
            "default_runtime_profile": "90_days",
        },
    },
    # ---------------------------------------------------------------
    # Short-run (2-cycle) pressure-driven longevity discovery funnel
    # ---------------------------------------------------------------
    "longevity_funnel_2cycle": {
        "label": "Longevity discovery funnel (2-cycle)",
        "description": (
            "Two-cycle, pressure-driven discovery funnel: map/cluster then cull/commit. "
            "Designed to raise stability and RYE by enforcing elimination and cross-domain recombination."
        ),
        "depth": 0.7,
        "exploration": 0.55,
        "verification_intensity": 0.75,
        "discovery_focus": 0.85,
        "tier3_bias": 0.35,
        "swarm_recommended": True,
        "preferred_swarm_size": 64,
        "cycle_funnel": {
            "cycle_1": "map_cluster",
            "cycle_2": "cull_stress_commit",
        },
        "constraints": {
            "kill_quota": {"min_rejections_fraction": 0.5, "ratio_reject_to_create": 0.8},
            "domain_mix_min": 2,
            "novelty_floor": {"penalize_mainstream_overlap": True},
        },
        "event_logging": {
            "emit_agent_output": True,
            "emit_candidate_hypotheses": True,
            "emit_verifications": True,
            "emit_discoveries": True,
            "store_sources": True,
        },
    },

}


# -------------------------------------------------------------------
# Recommended defaults by domain and runtime profile
# These line up with presets.py domains and runtime profiles.
# -------------------------------------------------------------------

DEFAULT_PROFILE_BY_DOMAIN: Dict[str, str] = {
    "general": "baseline",
    "longevity": "longevity_clinical",
    "antiaging": "longevity_clinical",
    "anti_aging": "longevity_clinical",
    "aging": "longevity_clinical",
    "anti_ageing": "longevity_clinical",
    "math": "math_formal",
    "theory": "math_formal",
    "formal": "math_formal",
}

DEFAULT_PROFILE_BY_RUNTIME: Dict[str, str] = {
    "1_hour": "baseline",
    "8_hours": "aggressive_discovery",
    "24_hours": "aggressive_discovery",
    "1_week": "guardian_long_run",
    "1_month": "guardian_long_run",
    "90_days": "tier3_hunter",
    "forever": "guardian_long_run",
}


# Simple aliases so UI or API can refer to shorter names
PROFILE_ALIASES: Dict[str, str] = {
    "tier3": "tier3_hunter",
    "guardian": "guardian_long_run",
    "explore_swarm": "exploration_swarm",
    "clinical_longevity": "longevity_clinical",
    "math_precise": "math_formal",
    "safe": "conservative",
    "default": "baseline",
}


# -------------------------------------------------------------------
# Accessors
# -------------------------------------------------------------------
def get_intelligence_profile(name: str) -> Dict[str, Any]:
    """
    Return a profile config, annotated with profile_name.
    If unknown, fall back to baseline.

    Name resolution order:
      1) direct key in INTELLIGENCE_PROFILES
      2) alias in PROFILE_ALIASES
      3) baseline
    """
    key = (name or "baseline").strip().lower()
    profile_name: str

    if key in INTELLIGENCE_PROFILES:
        profile_name = key
        cfg = INTELLIGENCE_PROFILES[profile_name]
    else:
        alias = PROFILE_ALIASES.get(key)
        if alias and alias in INTELLIGENCE_PROFILES:
            profile_name = alias
            cfg = INTELLIGENCE_PROFILES[profile_name]
        else:
            profile_name = "baseline"
            cfg = INTELLIGENCE_PROFILES[profile_name]

    prof = dict(cfg)
    prof.setdefault("profile_name", profile_name)
    return prof


def resolve_profile_for_domain(domain: Optional[str]) -> Dict[str, Any]:
    """
    Resolve a default profile name for a domain, then return the config.
    Falls back to baseline if nothing matches.
    """
    if not domain:
        return get_intelligence_profile("baseline")

    key = str(domain).strip().lower()
    profile_name = DEFAULT_PROFILE_BY_DOMAIN.get(key, "baseline")
    return get_intelligence_profile(profile_name)


def resolve_profile_for_runtime(runtime_profile: Optional[str]) -> Dict[str, Any]:
    """
    Resolve a default profile based on runtime profile name
    such as "1_hour", "8_hours", "24_hours", "1_week",
    "1_month", "90_days", or "forever".
    If unknown, returns baseline.
    """
    if not runtime_profile:
        return get_intelligence_profile("baseline")

    key = str(runtime_profile).strip()
    profile_name = DEFAULT_PROFILE_BY_RUNTIME.get(key, "baseline")
    return get_intelligence_profile(profile_name)


def resolve_profile(
    *,
    explicit_profile: Optional[str] = None,
    domain: Optional[str] = None,
    runtime_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High level resolver for CoreAgent and engine_worker.

    Priority:
      1) explicit_profile if provided
      2) domain based default
      3) runtime_profile based default
      4) baseline
    """
    if explicit_profile:
        return get_intelligence_profile(explicit_profile)
    if domain:
        return resolve_profile_for_domain(domain)
    if runtime_profile:
        return resolve_profile_for_runtime(runtime_profile)
    return get_intelligence_profile("baseline")


def list_intelligence_profile_names() -> List[str]:
    """
    Return the list of available profile names.
    """
    return sorted(INTELLIGENCE_PROFILES.keys())


def describe_intelligence_profiles() -> List[Dict[str, Any]]:
    """
    Lightweight descriptions for dashboards or logs.
    """
    out: List[Dict[str, Any]] = []
    for name, cfg in INTELLIGENCE_PROFILES.items():
        rec = cfg.get("preset_recommendations", {}) or {}
        out.append(
            {
                "name": name,
                "label": cfg.get("label", name),
                "description": cfg.get("description", ""),
                "depth": cfg.get("depth"),
                "exploration": cfg.get("exploration"),
                "verification_intensity": cfg.get("verification_intensity"),
                "discovery_focus": cfg.get("discovery_focus"),
                "tier3_bias": cfg.get("tier3_bias"),
                "swarm_recommended": cfg.get("swarm_recommended"),
                "preferred_swarm_size": cfg.get("preferred_swarm_size"),
                "default_preset": rec.get("default_preset"),
                "default_runtime_profile": rec.get("default_runtime_profile"),
            }
        )
    return out
