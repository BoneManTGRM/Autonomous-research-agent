"""Domain presets for the Autonomous Research Agent.

Presets act as domain specific profiles that define:
- default goals
- source controls
- domain tags
- query behavior
- RYE weighting hints and advanced RYE expectations
- reporting structure
- runtime profiles for 1h / 8h / 24h / 1 week / 1 month / 90 day / forever runs
- swarm role contracts and intelligence hints for multi agent operation
- diagnostics cadence for advanced RYE metrics and stability tracking
- learning and forgetting behavior for MemoryStore and vector memory

These settings do not break existing code but unlock future power for:
- continuous mode tuning
- energy aware TGRM behavior
- long run reporting and summaries
- swarm aware behavior (many coordinated logical agents)
- cure or treatment extraction pipelines
- stricter verification and smarter hypothesis selection
- repair efficiency signatures per domain and per preset
- adaptive learning based on RYE gradients and stability metrics

Finite-only note:
- Engine and worker are now finite-only. Runtime profiles are treated as
  *hints* for learning speed, reporting cadence, and diagnostics.
- The "forever" profile is legacy and should be hidden in the UI; if used,
  engines still enforce finite limits via max_minutes / cycles / rounds.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import copy

# Simple version tag so the app and UI can display which preset set is loaded.
PRESETS_VERSION: str = "2025-12-14-finite-snapshots-v1"

# ---------------------------------------------------------------------
# Global Runtime Profiles (applies for all presets)
# These are interpreted by CoreAgent and engine_worker.
# Each profile now also carries learning speed hints that can be used
# by the learning_burst layer and MemoryStore learning profiles.
#
# New field: wallclock_minutes
#   Exact intended duration in minutes so timed runs map to real time:
#   1 hour, 8 hours, 24 hours, 1 week, 1 month, 90 days.
#   The forever profile is treated as a legacy hint only in finite mode.
# ---------------------------------------------------------------------
RUNTIME_PROFILES: Dict[str, Dict[str, Any]] = {
    "1_hour": {
        "label": "1 Hour Run",
        "estimated_cycles": 40,
        "rye_stop_threshold": None,
        "energy_scaling": 1.0,
        "report_frequency": 1,
        "description": "Short diagnostic run for fast repair checks and a sanity pass.",
        "use_advanced_rye": True,
        "expected_equilibrium": "none",
        "target_rye_range": [0.02, 0.15],
        # Exact real time target in minutes
        "wallclock_minutes": 60.0,
        # Faster learning hints
        "learning_speed_factor": 2.0,
        "burst_profile_hint": "light",
        "spread_of_learning_strength": 0.4,
    },
    "8_hours": {
        "label": "8 Hour Run",
        "estimated_cycles": 200,
        "rye_stop_threshold": 0.05,
        "energy_scaling": 1.2,
        "report_frequency": 1,
        "description": "Medium autonomous session where early equilibrium patterns can emerge.",
        "use_advanced_rye": True,
        "expected_equilibrium": "transient_or_plateau",
        "target_rye_range": [0.04, 0.18],
        # Exact real time target in minutes
        "wallclock_minutes": 8.0 * 60.0,
        "learning_speed_factor": 4.0,
        "burst_profile_hint": "light",
        "spread_of_learning_strength": 0.6,
    },
    "24_hours": {
        "label": "24 Hour Run",
        "estimated_cycles": 600,
        "rye_stop_threshold": 0.08,
        "energy_scaling": 1.4,
        "report_frequency": 2,
        "description": "Full daily autonomous research loop for equilibrium and deep repairs.",
        "use_advanced_rye": True,
        "expected_equilibrium": "plateau",
        "target_rye_range": [0.06, 0.20],
        # Exact real time target in minutes
        "wallclock_minutes": 24.0 * 60.0,
        "learning_speed_factor": 6.0,
        "burst_profile_hint": "balanced",
        "spread_of_learning_strength": 0.8,
    },
    "1_week": {
        "label": "1 Week Run",
        "estimated_cycles": 7 * 600,
        "rye_stop_threshold": 0.10,
        "energy_scaling": 1.5,
        "report_frequency": 6,
        "description": (
            "Seven day continuous research profile tuned for medium term stability, "
            "repeated repair cycles, and detection of equilibrium windows."
        ),
        "use_advanced_rye": True,
        "expected_equilibrium": "plateau_or_high",
        "target_rye_range": [0.08, 0.22],
        # Exact real time target in minutes
        "wallclock_minutes": 7.0 * 24.0 * 60.0,
        "learning_speed_factor": 8.0,
        "burst_profile_hint": "balanced",
        "spread_of_learning_strength": 1.0,
    },
    "1_month": {
        "label": "1 Month Run",
        "estimated_cycles": 30 * 600,
        "rye_stop_threshold": 0.10,
        "energy_scaling": 1.6,
        "report_frequency": 12,
        "description": (
            "Multi week deep autonomy profile for sustained Reparodynamics experiments, "
            "stack refinement, and long horizon repair efficiency tracking."
        ),
        "use_advanced_rye": True,
        "expected_equilibrium": "plateau_or_high",
        "target_rye_range": [0.09, 0.24],
        # Exact real time target in minutes
        "wallclock_minutes": 30.0 * 24.0 * 60.0,
        "learning_speed_factor": 9.0,
        "burst_profile_hint": "aggressive",
        "spread_of_learning_strength": 1.2,
    },
    # Long horizon profile for Reparodynamics style experiments
    "90_days": {
        "label": "90 Day Run",
        "estimated_cycles": 20_000,
        "rye_stop_threshold": 0.10,
        "energy_scaling": 1.6,
        "report_frequency": 24,
        "description": (
            "Long horizon stability experiment for Reparodynamics. "
            "Optimized for equilibrium, drift control, and repair efficiency over 90 days."
        ),
        "use_advanced_rye": True,
        "expected_equilibrium": "high_or_plateau",
        "target_rye_range": [0.10, 0.28],
        # Exact real time target in minutes
        "wallclock_minutes": 90.0 * 24.0 * 60.0,
        "learning_speed_factor": 10.0,
        "burst_profile_hint": "aggressive",
        "spread_of_learning_strength": 1.5,
    },
    "forever": {
        "label": "Run Until Stopped (legacy)",
        "estimated_cycles": 10_000_000,
        "rye_stop_threshold": None,
        "energy_scaling": 1.0,
        "report_frequency": 5,
        "description": (
            "Legacy unbounded profile. In finite-only mode, engines still enforce "
            "finite limits via max_minutes / cycles / rounds. UI should hide this."
        ),
        "use_advanced_rye": True,
        "expected_equilibrium": "mixed",
        "target_rye_range": [0.04, 0.24],
        # No fixed wallclock_minutes for forever in legacy mode
        "learning_speed_factor": 5.0,
        "burst_profile_hint": "balanced",
        "spread_of_learning_strength": 1.0,
    },
}

# ---------------------------------------------------------------------
# Default RYE threshold hints
# These are soft guidelines for the engine and UI, not hard rules.
# ---------------------------------------------------------------------
DEFAULT_RYE_THRESHOLDS: Dict[str, float] = {
    # Below this, the agent is probably still fixing basic defects
    "low": 0.0,
    # Around this and above, maintenance mode can be considered
    "maintenance": 0.05,
    # Good zone for stable long runs
    "good": 0.10,
    # High efficiency zone, usually reached after strong repair phases
    "excellent": 0.20,
}

# Expectations for advanced RYE metrics where available
DEFAULT_ADVANCED_RYE_EXPECTATIONS: Dict[str, Any] = {
    "rolling_window_short": 10,
    "rolling_window_long": 50,
    "stability_index_target": 0.6,       # 0 to 1 scale
    "recovery_momentum_target": 0.1,     # positive values show recovery after perturbation
    "max_oscillation_std": 0.25,
}

# ---------------------------------------------------------------------
# Global snapshot defaults
#
# This is the simple config block that should always be present in the
# job config. The engine and MemoryStore can read these keys directly.
#
# Presets can override any of these fields, e.g. longevity can turn
# snapshots on by default while others stay off.
# ---------------------------------------------------------------------
SNAPSHOT_CONFIG_TEMPLATE: Dict[str, Any] = {
    # Master toggle for run snapshots
    "enabled": False,
    # Take a snapshot every N outer cycles / rounds
    "interval_cycles": 25,
    # Keep at most this many snapshot files per run
    "max_files": 40,
    # What to include
    "include_memory": True,
    "include_run_state": True,
    "include_discoveries": True,
    "include_rye_metrics": True,
    "include_swarm_state": True,
    # Optional note that can be written into the snapshot metadata
    "note": None,
}

# ---------------------------------------------------------------------
# Global search energy template for presets
# This is used as a conceptual template. Individual presets override
# fields where they need domain specific behavior.
# ---------------------------------------------------------------------
SEARCH_ENERGY_TEMPLATE: Dict[str, Any] = {
    "mode": "dynamic",  # static or dynamic
    # Role level hints are merged with config.search_energy.role_multipliers
    "role_bias": {
        "researcher": 1.0,
        "critic": 0.8,
        "explorer": 1.1,
        "integrator": 0.7,
        "planner": 0.7,
        "synthesizer": 0.7,
    },
    # Phase hints follow config.search_energy phases
    "exploration": {
        "min_cycles": 3,
        "max_cycles": 6,
        "intensity_multiplier": 1.0,
        "novelty_priority": "balanced",  # aggressive, conservative, balanced
    },
    "compression": {
        "enabled": True,
        "intensity_multiplier": 0.5,
        "novelty_drop_threshold": 0.35,
        "min_rye_window": 5,
    },
    "verification": {
        "enabled": True,
        "intensity_multiplier": 0.3,
        "contradiction_density_trigger": 0.25,
        "hypothesis_uncertainty_trigger": 0.6,
    },
    # Tavily budget profile is used by tools and orchestration layers
    "tavily_budget_profile": {
        "hourly_soft_cap": 600,
        "hourly_hard_cap": 1000,
        "daily_soft_cap": 5000,
        "daily_hard_cap": 10000,
    },
}

# ---------------------------------------------------------------------
# Global PDF report defaults
# These are read by the report generator to build summarized PDFs
# including breakthrough hints, RYE charts, and discovery tables.
# ---------------------------------------------------------------------
PDF_REPORT_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "engine_hint": "reportlab",  # or "fpdf" or any concrete implementation
    "include_rye_charts": True,
    "include_discovery_table": True,
    "include_swarm_summary": True,
    "include_runtime_profile": True,
    "file_naming": {
        # Engine can substitute {domain}, {runtime_profile}, {timestamp}, {run_id}
        "pattern": "reports/{domain}_{runtime_profile}_{timestamp}.pdf"
    },
    "breakthrough_hints": {
        # These are hints, not hard rules. The breakthrough engine can combine
        # them with discovery metadata and intelligence profiles.
        "tier1_thresholds": {
            "min_high_value_discoveries": 1,
            "min_peak_rye": 0.12,
            "min_avg_rye": 0.06,
        },
        "tier2_thresholds": {
            "min_high_value_discoveries": 3,
            "min_peak_rye": 0.18,
            "min_avg_rye": 0.10,
        },
        "tier3_thresholds": {
            "min_high_value_discoveries": 5,
            "min_peak_rye": 0.22,
            "min_avg_rye": 0.12,
            "min_stability_index": 0.55,
        },
        "use_domain_specific_overrides": True,
        "label_fields": {
            "include_text_label": True,
            "include_numeric_score": True,
            "include_explanation_block": True,
        },
    },
}

# ---------------------------------------------------------------------
# Continuous mode defaults for single agent and swarm runs
# These are read by CoreAgent and engine_worker to keep behavior aligned.
# In finite-only mode, these are treated as policy and hint values.
# ---------------------------------------------------------------------
CONTINUOUS_MODE_DEFAULTS: Dict[str, Any] = {
    # Heartbeat and watchdog cadence. Engine uses this as a target interval
    # for status updates; actual timing may vary slightly with workload.
    "watchdog_interval_minutes": 5.0,
    # Checkpoint cadence for long runs. This is a policy hint, not a hard cap.
    "checkpoint_interval_cycles": 10,
    # Global hard safety ceiling for cycles in continuous mode.
    # This should stay synchronized with engine_worker WORKER_MAX_* settings.
    "max_cycles_failsafe": 10_000_000,
    "heartbeat_labels": {
        "single": "continuous_single",
        "swarm": "continuous_swarm",
    },
    # Default runtime profile used when none is specified explicitly
    "default_runtime_profile_single": "8_hours",
    "default_runtime_profile_swarm": "24_hours",
    # Hints for exact time stop handling (optional, engine may ignore)
    "exact_time_stop_hints": {
        "single_default": False,
        "swarm_default": True,
        "wrap_up_buffer_minutes": 3.0,
    },
    # Learning based control hooks (core can use these if desired)
    "learning_hooks": {
        "use_advanced_rye_metrics": True,
        "adapt_runtime_profile": True,
        "adapt_maintenance_mode": True,
        "adapt_swarm_size": True,
        "rye_gradient_safety_margin": 0.01,
        "stability_index_floor": 0.4,
        # Ten times faster learning layer
        "enable_learning_bursts": True,
        "learning_speed_factor_default": 10.0,
        "use_goal_leaderboard": True,
        "spread_of_learning_mode": "cross_goal",
        "max_parent_goals_for_copy": 3,
        "min_parent_avg_rye": 0.08,
        "min_parent_stability_index": 0.5,
        "learning_burst_profiles": {
            # Light burst mode for short runs
            "light": {
                "cycles": 5,
                "delta_novelty": 0.05,
                "delta_critic_strength": 0.1,
                "delta_verification_rigidity": 0.05,
            },
            # Balanced bursts for 24 hour to 1 month profiles
            "balanced": {
                "cycles": 10,
                "delta_novelty": 0.08,
                "delta_critic_strength": 0.15,
                "delta_verification_rigidity": 0.08,
            },
            # Aggressive bursts for 90 day stability experiments
            "aggressive": {
                "cycles": 20,
                "delta_novelty": 0.12,
                "delta_critic_strength": 0.20,
                "delta_verification_rigidity": 0.10,
            },
        },
        # Fast path integration with MemoryStore goal_index and learning profiles
        "fast_path_goal_index": True,
        "use_learning_profiles": True,
        "learning_profile_reporting": {
            "enabled": True,
            "report_every_n_cycles": 25,
            "include_goal_leaderboard": True,
            "include_role_leaderboard": True,
        },
    },
}

# ---------------------------------------------------------------------
# Swarm defaults (shared across presets)
# ---------------------------------------------------------------------
# Global, domain agnostic role templates with contracts
SWARM_ROLES: List[Dict[str, Any]] = [
    {
        "name": "researcher",
        "description": "Primary deep literature and web researcher that gathers facts and writes detailed notes.",
        "mission": (
            "Continuously search, ingest, and summarize high value sources that are directly relevant to the goal. "
            "Prioritize primary literature and structured data over blogs or opinion pieces."
        ),
        "expected_inputs": [
            "current_goal",
            "open_questions",
            "source_controls",
        ],
        "expected_outputs": [
            "source_summaries",
            "key_mechanisms",
            "candidate_interventions",
        ],
        "forbidden_behaviors": [
            "rephrasing the prompt without adding new information",
            "inventing fake citations or journals",
        ],
    },
    {
        "name": "critic",
        "description": "Methodology critic and refiner that attacks weak points, gaps, and overclaims.",
        "mission": (
            "Stress test claims, mechanisms, and interventions produced by other roles. "
            "Identify logical gaps, unsupported leaps, confounders, and low quality evidence."
        ),
        "expected_inputs": [
            "draft_hypotheses",
            "mechanism_maps",
            "evidence_summaries",
        ],
        "expected_outputs": [
            "explicit_criticisms",
            "risk_flags",
            "required_repairs",
        ],
        "forbidden_behaviors": [
            "adding new hypotheses instead of critiquing existing ones",
            "ignoring obvious methodological flaws",
        ],
    },
    {
        "name": "planner",
        "description": "Planner that proposes next experiments, queries, and high value repair actions.",
        "mission": (
            "Transform gaps and open questions into concrete next steps: new queries, data needs, "
            "simulation plans, or experimental designs that would most increase RYE."
        ),
        "expected_inputs": [
            "open_questions",
            "critic_feedback",
            "current_state_summary",
        ],
        "expected_outputs": [
            "prioritized_action_list",
            "next_queries",
            "experiment_suggestions",
        ],
        "forbidden_behaviors": [
            "repeating actions that have already been tried with no gain in RYE",
        ],
    },
    {
        "name": "synthesizer",
        "description": "Synthesizer that condenses findings into clear narratives, tables, and summaries.",
        "mission": (
            "Compress the current state of knowledge into human readable summaries that preserve nuance. "
            "Highlight what is known, what is uncertain, and where RYE looks strongest."
        ),
        "expected_inputs": [
            "mechanism_maps",
            "source_summaries",
            "critic_feedback",
        ],
        "expected_outputs": [
            "structured_summary",
            "tables_or_bullets",
            "candidate_master_paths",
        ],
        "forbidden_behaviors": [
            "overstating confidence",
            "dropping important caveats",
        ],
    },
    {
        "name": "explorer",
        "description": "Out of distribution explorer that searches for unusual angles, analogies, and adjacent fields.",
        "mission": (
            "Look for analogies, cross domain patterns, and adjacent literatures that may improve RYE. "
            "Propose unconventional but plausible directions that other roles can then test or critique."
        ),
        "expected_inputs": [
            "current_goal",
            "high_level_summary",
        ],
        "expected_outputs": [
            "cross_domain_links",
            "unusual_hypotheses",
            "adjacent_frameworks",
        ],
        "forbidden_behaviors": [
            "drifting into unrelated topics",
            "repeating obvious mainstream facts",
        ],
    },
    {
        "name": "integrator",
        "description": "Integrator that reconciles conflicting notes and hypotheses into a coherent picture.",
        "mission": (
            "Take outputs from researcher, critic, planner, synthesizer, and explorer and reconcile them into "
            "a coherent state. Resolve contradictions when possible, and document irreducible uncertainties."
        ),
        "expected_inputs": [
            "structured_summary",
            "critic_feedback",
            "candidate_paths",
        ],
        "expected_outputs": [
            "integrated_model",
            "ranked_paths",
            "final_hypothesis_set",
        ],
        "forbidden_behaviors": [
            "silencing disagreement without justification",
            "introducing new claims not supported by any role",
        ],
    },
]

SWARM_GLOBAL_HINTS: Dict[str, Any] = {
    # Hard safety ceiling for platform resources.
    # Increased from 32 to 64 to allow larger swarms.
    "max_agents_safe": 64,
    # Good default for most presets when user clicks swarm.
    "default_agents": 5,
    # How time should be split in continuous mode for swarms.
    "time_split_strategy": "equal",  # equal, weighted, or custom
    # Roles available for the swarm orchestration layer.
    "roles": SWARM_ROLES,
    # Versioning for richer role contracts (future engines can check this)
    "contracts_version": 1,
}

# ---------------------------------------------------------------------
# Domain Presets
# ---------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {

    # ================================================================
    # GENERAL PRESET
    # ================================================================
    "general": {
        "label": "General research",
        "domain": "general",

        # Default TGRM and engine hints
        "default_tgrm_level": 3,
        "supports_single_agent": True,
        "supports_swarm": True,

        "default_goal": (
            "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
            "identify similar frameworks in the literature, and produce a structured comparison table."
        ),

        # These keys are used directly by TGRMLoop._normalise_source_controls
        "source_controls": {
            "web": True,
            "pubmed": False,
            "semantic": False,
            "pdf": True,
            "biomarkers": False,
        },

        "focus_keywords": [
            "reparodynamics",
            "RYE",
            "repair yield per energy",
            "TGRM",
            "self repair",
            "autonomous systems",
            "stability",
            "resilience",
        ],

        "preferred_sources": ["web", "semantic", "pdf"],
        "min_citation_count": 0,

        "note_structure_hint": [
            "High level summary",
            "Key definitions (Reparodynamics, RYE, TGRM)",
            "Similar frameworks",
            "Open questions and next steps",
        ],

        "rye_weights": {
            "issues_resolved": 1.0,
            "hypotheses_quality": 1.0,
            "citations_added": 0.5,
        },

        # Domain level cycle tuning
        "cycle_energy_multiplier": 1.0,
        "cycle_length_hint": "short",
        "repair_depth_bias": "balanced",

        # Domain specific TGRM intelligence hints
        "tgrm_hints": {
            "strict_verify": "medium",
            "allow_prompt_echo": False,
            "min_novelty_fraction": 0.15,
            "max_redundant_cycles": 8,
        },

        # Search energy hints for this preset
        "search_energy": {
            **SEARCH_ENERGY_TEMPLATE,
            "role_bias": {
                "researcher": 1.0,
                "critic": 0.8,
                "explorer": 1.1,
                "integrator": 0.7,
                "planner": 0.7,
                "synthesizer": 0.8,
            },
            "exploration": {
                "min_cycles": 3,
                "max_cycles": 6,
                "intensity_multiplier": 1.0,
                "novelty_priority": "balanced",
            },
            "compression": {
                "enabled": True,
                "intensity_multiplier": 0.5,
                "novelty_drop_threshold": 0.35,
                "min_rye_window": 5,
            },
            "verification": {
                "enabled": True,
                "intensity_multiplier": 0.3,
                "contradiction_density_trigger": 0.25,
                "hypothesis_uncertainty_trigger": 0.6,
            },
            "tavily_budget_profile": {
                "hourly_soft_cap": 600,
                "hourly_hard_cap": 1000,
                "daily_soft_cap": 5000,
                "daily_hard_cap": 10000,
            },
        },

        # Tool intelligence for this preset
        "tool_intelligence": {
            "browser_usage": {
                "mode": "adaptive",  # conservative, normal, aggressive, adaptive
                "max_calls_per_cycle": 5,
                "prefer_primary_sources": True,
                "crawl_depth": 1,
            },
            "sandbox_usage": {
                "enabled": True,
                "max_execs_per_cycle": 3,
                "verify_after_exec": True,
                "allowed_packages": ["math", "statistics", "pandas", "numpy"],
            },
            "data_pipeline_usage": {
                "load_csv": True,
                "load_excel": True,
                "load_sql": True,
                "auto_detect_timeseries": True,
            },
        },

        # Memory intelligence
        "memory_intelligence": {
            "write_frequency": "adaptive",  # low, medium, high, adaptive
            "compression_strategy": "semantic",  # none, simple, semantic
            "auto_summarize_every_n_cycles": 10,
            "max_memory_items": 5_000,
            "forgetting_policy": {
                "enabled": True,
                "threshold_rye_gain": 0.01,
                "drop_low_value_notes": True,
                "reinforce_high_value_notes": True,
            },
        },

        # Snapshot defaults (off by default for general preset)
        # Both the block and the top-level boolean are present so job
        # configs always have a simple, predictable shape.
        "snapshot": {
            **SNAPSHOT_CONFIG_TEMPLATE,
            "enabled": False,
        },
        "snapshot_enabled": False,

        # Learning hints that combine memory and advanced RYE metrics
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "advanced_expectations": DEFAULT_ADVANCED_RYE_EXPECTATIONS,
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
            # Ten times faster learning hints
            "learning_speed_factor": 8.0,
            "enable_learning_bursts": True,
            "burst_modes": {
                "default": {
                    "profile": "balanced",
                    "min_cycles_between_bursts": 15,
                    "max_bursts_per_100_cycles": 5,
                },
                "stagnation_recovery": {
                    "profile": "aggressive",
                    "trigger_if_no_rye_gain_cycles": 20,
                    "trigger_if_efficiency_trend_negative": True,
                },
            },
            "spread_of_learning": {
                "enable_cross_goal_copy": True,
                "max_parent_goals": 3,
                "min_parent_avg_rye": 0.08,
                "min_parent_stability_index": 0.5,
                "reuse_best_equilibrium_labels": True,
                "reuse_best_swarm_contracts": True,
            },
        },

        # Discovery engine hints
        "discovery_engine": {
            "enabled": True,
            "threshold_novelty": 0.20,
            "threshold_support": 0.15,
            "detect_mechanisms": True,
            "detect_interventions": True,
            "detect_patterns": True,
            "multi_agent_voting": True,
            "minimum_role_agreement": 3,
            "classification": [
                "mechanism",
                "intervention",
                "treatment",
                "biomarker_shift",
                "mathematical_structure",
                "prediction",
            ],
        },

        # Swarm orchestration hints
        "swarm_orchestration": {
            "rotation_strategy": "round_robin",  # round_robin, weighted, priority
            "role_weighting": {
                "researcher": 1.0,
                "critic": 1.0,
                "planner": 1.0,
                "synthesizer": 1.0,
                "explorer": 1.0,
                "integrator": 1.0,
            },
            "cycle_sync": "partial",  # none, partial, full
            "integration_frequency": 1,
            "consensus_model": "weighted_vote",  # single_best, majority, weighted_vote
        },

        # Compute tuning
        "compute_tuning": {
            "energy_to_depth_ratio": 1.0,
            "exploration_vs_exploitation": "balanced",
            "max_energy_spend_per_cycle": "adaptive",
            "rye_weighting_mode": "domain_specific",
        },

        # Evidence strictness
        "evidence_modes": {
            "mode": "balanced",  # clinical, mechanistic, mathematical, exploratory, balanced
            "require_citations": True,
            "min_confidence_to_accept": 0.2,
            "reject_fabricated_sources": True,
            "disallow_uncited_claims": True,
        },

        # Biomarker intelligence (generic, mostly disabled in general mode)
        "biomarker_intelligence": {
            "enabled": False,
            "panels": {},
            "strictness": "informational",
            "require_numeric_effect_size": False,
            "link_to_cure_extraction": False,
        },

        # Default runtime and RYE behavior for continuous mode
        "default_runtime_profile": "8_hours",
        "default_rye_stop_threshold": None,
        "rye_thresholds": DEFAULT_RYE_THRESHOLDS,
        "advanced_rye_expectations": DEFAULT_ADVANCED_RYE_EXPECTATIONS,

        # Repair efficiency diagnostics hints
        "run_diagnostics": {
            "enabled": True,
            "window": 10,
            "frequency_cycles": 25,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },

        # Cure or treatment extraction is generic here
        "cure_extraction": {
            "enabled": False,
            "targets": [],
            "notes": "General preset does not focus on treatments by default.",
        },

        # Report style
        "report_sections": [
            "summary",
            "rye_statistics",
            "notes",
            "hypotheses",
            "citations",
        ],
        "report_style": "narrative",
        "report_frequency": 1,

        # PDF report configuration for general preset
        "pdf_report": {
            "enabled": True,
            "base_defaults": PDF_REPORT_DEFAULTS,
            "sections": [
                "summary",
                "rye_statistics",
                "notes",
                "hypotheses",
                "citations",
            ],
            "include_biomarker_panels": False,
            "include_candidate_stacks": False,
            "breakthrough_overrides": {
                # General preset treats strong RYE and novelty as interesting,
                # but breakthrough classification is softer than longevity.
                "tier1_thresholds": {
                    "min_high_value_discoveries": 1,
                    "min_peak_rye": 0.10,
                    "min_avg_rye": 0.05,
                },
                "tier2_thresholds": {
                    "min_high_value_discoveries": 3,
                    "min_peak_rye": 0.16,
                    "min_avg_rye": 0.09,
                },
            },
        },

        # Universal runtime profiles
        "runtime_profiles": RUNTIME_PROFILES,

        # Swarm hints for general research
        "swarm": {
            "enabled_by_default": False,
            "max_agents": SWARM_GLOBAL_HINTS["max_agents_safe"],
            "default_agents": 4,
            "time_split_strategy": "equal",
            "roles": SWARM_ROLES,
            "role_bias": "balanced",
            "notes": (
                "General swarm is balanced and good for exploratory research across many topics. "
                "Roles should behave as broad but disciplined specialists rather than narrow domain experts."
            ),
            # Optional smarter role templates
            "role_templates": [
                {
                    "name": "general_researcher",
                    "inherits_from": "researcher",
                    "domain_focus": "broad",
                },
                {
                    "name": "general_critic",
                    "inherits_from": "critic",
                    "domain_focus": "methods_and_logic",
                },
                {
                    "name": "general_explorer",
                    "inherits_from": "explorer",
                    "domain_focus": "cross_domain",
                },
                {
                    "name": "general_integrator",
                    "inherits_from": "integrator",
                    "domain_focus": "summary_and_equilibrium",
                },
            ],
        },

        # UI hints for the dashboard
        "ui_hints": {
            "show_biomarkers_tab": False,
            "show_cure_treatment_tab": False,
            "default_view": "summary",
        },
    },

    # ================================================================
    # LONGEVITY PRESET
    # ================================================================
    "longevity": {
        "label": "Longevity / Anti aging",
        "domain": "longevity",

        # Default TGRM and engine hints
        "default_tgrm_level": 3,
        "supports_single_agent": True,
        "supports_swarm": True,

        "default_goal": (
            "Identify and summarize interventions, biomarkers, and mechanisms that extend healthspan and longevity, "
            "and relate them to RYE and reparodynamic stability across organs and systems."
        ),

        "source_controls": {
            "web": True,
            "pubmed": True,
            "semantic": True,
            "pdf": True,
            "biomarkers": True,
        },

        "focus_keywords": [
            "longevity",
            "healthspan",
            "aging",
            "senescence",
            "autophagy",
            "mTOR",
            "NAD+",
            "rapamycin",
            "metformin",
            "caloric restriction",
            "clinical trial",
            "biomarker",
        ],

        "preferred_sources": ["pubmed", "semantic", "web", "pdf"],
        "min_citation_count": 20,

        "note_structure_hint": [
            "Interventions",
            "Mechanisms and pathways",
            "Biomarkers affected",
            "Evidence strength",
            "RYE interpretation",
        ],

        "rye_weights": {
            "issues_resolved": 1.2,
            "hypotheses_quality": 1.5,
            "citations_added": 1.0,
        },

        "cycle_energy_multiplier": 1.3,
        "cycle_length_hint": "long",
        "repair_depth_bias": "deep",

        # Domain specific TGRM intelligence hints for longevity
        "tgrm_hints": {
            "strict_verify": "high",
            "allow_prompt_echo": False,
            "min_novelty_fraction": 0.25,
            "max_redundant_cycles": 5,
            "prefer_clinical_evidence": True,
            "penalize_hype_language": True,
        },

        # Search energy hints for longevity (more aggressive exploration,
        # strong verification, but still respecting Tavily budget)
        "search_energy": {
            **SEARCH_ENERGY_TEMPLATE,
            "role_bias": {
                "researcher": 1.2,
                "critic": 1.1,
                "explorer": 1.0,
                "integrator": 1.1,
                "planner": 1.0,
                "synthesizer": 0.9,
            },
            "exploration": {
                "min_cycles": 4,
                "max_cycles": 8,
                "intensity_multiplier": 1.2,
                "novelty_priority": "aggressive",
            },
            "compression": {
                "enabled": True,
                "intensity_multiplier": 0.6,
                "novelty_drop_threshold": 0.30,
                "min_rye_window": 6,
            },
            "verification": {
                "enabled": True,
                "intensity_multiplier": 0.4,
                "contradiction_density_trigger": 0.20,
                "hypothesis_uncertainty_trigger": 0.5,
            },
            "tavily_budget_profile": {
                "hourly_soft_cap": 700,
                "hourly_hard_cap": 1100,
                "daily_soft_cap": 6000,
                "daily_hard_cap": 11000,
            },
        },

        # Tool intelligence
        "tool_intelligence": {
            "browser_usage": {
                "mode": "aggressive",
                "max_calls_per_cycle": 7,
                "prefer_primary_sources": True,
                "crawl_depth": 1,
            },
            "sandbox_usage": {
                "enabled": True,
                "max_execs_per_cycle": 4,
                "verify_after_exec": True,
                "allowed_packages": ["math", "statistics", "pandas", "numpy"],
            },
            "data_pipeline_usage": {
                "load_csv": True,
                "load_excel": True,
                "load_sql": True,
                "auto_detect_timeseries": True,
                # Optional biomarker specific hints for pipelines
                "biomarker_pipeline": {
                    "expect_longitudinal": True,
                    "align_by_patient_id": True,
                    "align_by_visit_or_time": True,
                    "compute_delta_per_biomarker": True,
                    "compute_effect_sizes": True,
                },
            },
        },

        # Memory intelligence
        "memory_intelligence": {
            "write_frequency": "high",
            "compression_strategy": "semantic",
            "auto_summarize_every_n_cycles": 8,
            "max_memory_items": 8_000,
            "forgetting_policy": {
                "enabled": True,
                "threshold_rye_gain": 0.02,
                "drop_low_value_notes": True,
                "reinforce_high_value_notes": True,
            },
        },

        # Snapshot defaults (ON by default for longevity so you do not
        # have to toggle it every run)
        "snapshot": {
            **SNAPSHOT_CONFIG_TEMPLATE,
            "enabled": True,
            # Slightly tighter interval and more files for long health runs
            "interval_cycles": 20,
            "max_files": 80,
            "include_biomarkers": True,
        },
        # Redundant top-level flag for compatibility with job builders
        "snapshot_enabled": True,

        # Biomarker intelligence (maxed out)
        "biomarker_intelligence": {
            "enabled": True,
            "strictness": "clinical",
            "require_numeric_effect_size": True,
            "require_direction_of_change": True,
            "min_effect_size_for_candidate": 0.10,
            "min_sample_size_for_confidence": 20,
            "link_to_cure_extraction": True,
            "panel_priority_order": [
                "core_aging",
                "organ_function",
                "inflammation",
                "metabolic",
                "lipids",
                "hormones",
                "other",
            ],
            "panels": {
                "core_aging": [
                    "epigenetic_age",
                    "grim_age",
                    "pheno_age",
                    "telomere_length",
                    "p16INK4a",
                ],
                "organ_function": [
                    "eGFR",
                    "creatinine",
                    "ALT",
                    "AST",
                    "ALP",
                    "bilirubin",
                ],
                "inflammation": [
                    "CRP",
                    "hsCRP",
                    "IL6",
                    "TNF_alpha",
                    "ferritin",
                ],
                "metabolic": [
                    "fasting_glucose",
                    "HOMA_IR",
                    "HbA1c",
                    "insulin",
                ],
                "lipids": [
                    "LDL",
                    "HDL",
                    "triglycerides",
                    "ApoB",
                ],
                "hormones": [
                    "IGF1",
                    "DHEA_S",
                    "testosterone",
                    "estradiol",
                    "TSH",
                    "free_T4",
                ],
                "other": [
                    "WBC",
                    "RBC",
                    "platelets",
                    "vitamin_D_25OH",
                ],
            },
            "biomarker_to_panel_overrides": {
                "dnAmGrimAge": "core_aging",
                "dnAmPhenoAge": "core_aging",
            },
            "flags": {
                "highlight_biomarkers_with_large_shifts": True,
                "highlight_conflicting_biomarker_movements": True,
                "call_out_unrealistic_changes": True,
            },
        },

        # Learning hints that combine advanced RYE metrics with biomarker and cure focus
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "advanced_expectations": {
                "rolling_window_short": 10,
                "rolling_window_long": 50,
                "stability_index_target": 0.55,
                "recovery_momentum_target": 0.12,
                "max_oscillation_std": 0.30,
            },
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.08,
                "stability_index_above": 0.50,
                "oscillation_std_below": 0.28,
                "min_cycles": 40,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": 0.0,
                "window": 25,
            },
            "learning_speed_factor": 10.0,
            "enable_learning_bursts": True,
            "burst_modes": {
                "default": {
                    "profile": "balanced",
                    "min_cycles_between_bursts": 12,
                    "max_bursts_per_100_cycles": 8,
                },
                "stagnation_recovery": {
                    "profile": "aggressive",
                    "trigger_if_no_rye_gain_cycles": 15,
                    "trigger_if_efficiency_trend_negative": True,
                    "require_stability_index_below": 0.55,
                },
                "biomarker_focus": {
                    "profile": "balanced",
                    "trigger_if_few_positive_biomarker_shifts": 2,
                    "lookback_cycles": 30,
                },
            },
            "spread_of_learning": {
                "enable_cross_goal_copy": True,
                "max_parent_goals": 3,
                "min_parent_avg_rye": 0.09,
                "min_parent_stability_index": 0.55,
                "reuse_best_equilibrium_labels": True,
                "reuse_best_swarm_contracts": True,
                "reuse_best_biomarker_panels": True,
            },
        },

        # Discovery engine (strong focus on interventions and biomarkers)
        "discovery_engine": {
            "enabled": True,
            "threshold_novelty": 0.18,
            "threshold_support": 0.20,
            "detect_mechanisms": True,
            "detect_interventions": True,
            "detect_patterns": True,
            "multi_agent_voting": True,
            "minimum_role_agreement": 4,
            "classification": [
                "mechanism",
                "intervention",
                "treatment",
                "biomarker_shift",
                "prediction",
            ],
        },

        # Swarm orchestration hints
        "swarm_orchestration": {
            "rotation_strategy": "weighted",
            "role_weighting": {
                "researcher": 1.3,
                "critic": 1.2,
                "planner": 1.1,
                "synthesizer": 1.0,
                "explorer": 0.9,
                "integrator": 1.3,
            },
            "cycle_sync": "partial",
            "integration_frequency": 1,
            "consensus_model": "weighted_vote",
        },

        # Compute tuning
        "compute_tuning": {
            "energy_to_depth_ratio": 1.3,
            "exploration_vs_exploitation": "exploit_heavy",
            "max_energy_spend_per_cycle": "adaptive",
            "rye_weighting_mode": "domain_specific",
        },

        # Evidence strictness
        "evidence_modes": {
            "mode": "clinical",
            "require_citations": True,
            "min_confidence_to_accept": 0.3,
            "reject_fabricated_sources": True,
            "disallow_uncited_claims": True,
        },

        # Default runtime and RYE behavior for continuous mode
        "default_runtime_profile": "24_hours",
        "default_rye_stop_threshold": 0.08,
        "rye_thresholds": DEFAULT_RYE_THRESHOLDS,
        "advanced_rye_expectations": {
            "rolling_window_short": 10,
            "rolling_window_long": 100,
            "stability_index_target": 0.55,
            "recovery_momentum_target": 0.15,
            "max_oscillation_std": 0.30,
        },

        # Repair efficiency diagnostics hints
        "run_diagnostics": {
            "enabled": True,
            "window": 10,
            "frequency_cycles": 20,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },

        # Cure or treatment extraction is a primary goal for longevity
        "cure_extraction": {
            "enabled": True,
            "targets": [
                "interventions",
                "treatments",
                "protocols",
                "drug_combinations",
                "lifestyle_stacks",
            ],
            "notes": (
                "Longevity preset enables cure or treatment extraction to track candidate "
                "stacks, doses, and evidence links for healthspan and aging interventions."
            ),
        },

        "report_sections": [
            "summary",
            "biomarkers",
            "mechanisms",
            "hypotheses",
            "citations",
            "rye_statistics",
        ],
        "report_style": "structured",
        "report_frequency": 1,

        # PDF report configuration for longevity preset
        "pdf_report": {
            "enabled": True,
            "base_defaults": PDF_REPORT_DEFAULTS,
            "sections": [
                "summary",
                "biomarkers",
                "mechanisms",
                "hypotheses",
                "citations",
                "rye_statistics",
            ],
            "include_biomarker_panels": True,
            "include_candidate_stacks": True,
            "include_clinical_evidence_table": True,
            "include_swarm_summary": True,
            "breakthrough_overrides": {
                # Longevity breakthroughs lean heavily on biomarkers and RYE
                "tier1_thresholds": {
                    "min_high_value_discoveries": 2,
                    "min_peak_rye": 0.14,
                    "min_avg_rye": 0.08,
                },
                "tier2_thresholds": {
                    "min_high_value_discoveries": 4,
                    "min_peak_rye": 0.20,
                    "min_avg_rye": 0.11,
                    "min_biomarkers_with_positive_shift": 3,
                },
                "tier3_thresholds": {
                    "min_high_value_discoveries": 6,
                    "min_peak_rye": 0.24,
                    "min_avg_rye": 0.13,
                    "min_biomarkers_with_positive_shift": 5,
                    "min_stability_index": 0.55,
                },
            },
        },

        "runtime_profiles": RUNTIME_PROFILES,

        # Swarm hints for longevity research
        "swarm": {
            "enabled_by_default": False,
            "max_agents": SWARM_GLOBAL_HINTS["max_agents_safe"],
            # Longevity benefits from more roles (researcher, critic, planner, synthesizer, integrator).
            "default_agents": 5,
            "time_split_strategy": "equal",
            "roles": SWARM_ROLES,
            "role_bias": "deep",
            "notes": (
                "Longevity swarms are tuned for deep evidence gathering, biomarker interpretation, "
                "and critical review of clinical data. Each role should behave like a specialized scientist."
            ),
            # Specific 5 role longevity swarm template for highly intelligent runs
            "role_templates": [
                {
                    "name": "mitochondria_metabolism_specialist",
                    "inherits_from": "researcher",
                    "domain_focus": "mitochondria_and_energy",
                    "mechanism_focus": [
                        "NAD+",
                        "mitophagy",
                        "OXPHOS",
                        "mitochondrial_biogenesis",
                    ],
                },
                {
                    "name": "dna_epigenetic_repair_specialist",
                    "inherits_from": "researcher",
                    "domain_focus": "dna_and_epigenetics",
                    "mechanism_focus": [
                        "DNA_repair",
                        "sirtuins",
                        "partial_reprogramming",
                        "epigenetic_drift",
                    ],
                },
                {
                    "name": "senescence_inflammation_specialist",
                    "inherits_from": "researcher",
                    "domain_focus": "senescence_and_inflammation",
                    "mechanism_focus": [
                        "senolytics",
                        "SASP",
                        "inflammaging",
                        "immune_clearance",
                    ],
                },
                {
                    "name": "proteostasis_autophagy_specialist",
                    "inherits_from": "researcher",
                    "domain_focus": "proteostasis_and_autophagy",
                    "mechanism_focus": [
                        "protein_quality_control",
                        "chaperones",
                        "autophagy",
                        "UPR",
                    ],
                },
                {
                    "name": "systems_integrator_and_planner",
                    "inherits_from": "integrator",
                    "domain_focus": "stack_design_and_equilibrium",
                    "mechanism_focus": [
                        "stack_design",
                        "RYE_tradeoffs",
                        "equilibrium_windows",
                    ],
                },
            ],
        },

        "ui_hints": {
            "show_biomarkers_tab": True,
            "show_cure_treatment_tab": True,
            "default_view": "biomarkers",
        },
    },

    # ================================================================
    # MATH PRESET
    # ================================================================
    "math": {
        "label": "Math / Theory",
        "domain": "math",

        # Default TGRM and engine hints
        "default_tgrm_level": 3,
        "supports_single_agent": True,
        "supports_swarm": True,

        "default_goal": (
            "Formalize Reparodynamics, RYE, and TGRM mathematically: definitions, axioms, theorems, "
            "proof sketches, and comparisons to control, information, and stability theories."
        ),

        "source_controls": {
            "web": True,
            "pubmed": False,
            "semantic": True,
            "pdf": True,
            "biomarkers": False,
        },

        "focus_keywords": [
            "mathematical",
            "formalization",
            "axiom",
            "theorem",
            "proof",
            "stability theory",
            "Lyapunov",
            "information theory",
            "Markov process",
            "equilibrium",
        ],

        "preferred_sources": ["semantic", "pdf", "web"],
        "min_citation_count": 5,

        "note_structure_hint": [
            "Definitions",
            "Core lemmas",
            "Theorems",
            "Connections to existing theory",
            "Open problems",
        ],

        "rye_weights": {
            "issues_resolved": 1.5,
            "hypotheses_quality": 1.7,
            "citations_added": 0.7,
        },

        "cycle_energy_multiplier": 0.8,
        "cycle_length_hint": "short",
        "repair_depth_bias": "precision",

        # Domain specific TGRM intelligence hints for math
        "tgrm_hints": {
            "strict_verify": "very_high",
            "allow_prompt_echo": False,
            "min_novelty_fraction": 0.2,
            "max_redundant_cycles": 3,
            "require_formal_structures": True,
        },

        # Search energy hints for math (conservative Tavily usage,
        # more reliance on internal reasoning and PDFs)
        "search_energy": {
            **SEARCH_ENERGY_TEMPLATE,
            "role_bias": {
                "researcher": 0.9,
                "critic": 0.9,
                "explorer": 0.7,
                "integrator": 0.9,
                "planner": 0.7,
                "synthesizer": 0.9,
            },
            "exploration": {
                "min_cycles": 2,
                "max_cycles": 4,
                "intensity_multiplier": 0.8,
                "novelty_priority": "conservative",
            },
            "compression": {
                "enabled": True,
                "intensity_multiplier": 0.4,
                "novelty_drop_threshold": 0.30,
                "min_rye_window": 6,
            },
            "verification": {
                "enabled": True,
                "intensity_multiplier": 0.25,
                "contradiction_density_trigger": 0.20,
                "hypothesis_uncertainty_trigger": 0.5,
            },
            "tavily_budget_profile": {
                "hourly_soft_cap": 400,
                "hourly_hard_cap": 800,
                "daily_soft_cap": 4000,
                "daily_hard_cap": 9000,
            },
        },

        # Tool intelligence
        "tool_intelligence": {
            "browser_usage": {
                "mode": "conservative",
                "max_calls_per_cycle": 4,
                "prefer_primary_sources": True,
                "crawl_depth": 1,
            },
            "sandbox_usage": {
                "enabled": True,
                "max_execs_per_cycle": 5,
                "verify_after_exec": True,
                "allowed_packages": ["math", "statistics", "numpy"],
            },
            "data_pipeline_usage": {
                "load_csv": True,
                "load_excel": False,
                "load_sql": True,
                "auto_detect_timeseries": False,
            },
        },

        # Memory intelligence
        "memory_intelligence": {
            "write_frequency": "medium",
            "compression_strategy": "semantic",
            "auto_summarize_every_n_cycles": 12,
            "max_memory_items": 4_000,
            "forgetting_policy": {
                "enabled": True,
                "threshold_rye_gain": 0.015,
                "drop_low_value_notes": True,
                "reinforce_high_value_notes": True,
            },
        },

        # Snapshot defaults (off by default for math)
        "snapshot": {
            **SNAPSHOT_CONFIG_TEMPLATE,
            "enabled": False,
        },
        "snapshot_enabled": False,

        # Biomarker intelligence stays off for math
        "biomarker_intelligence": {
            "enabled": False,
            "panels": {},
            "strictness": "informational",
            "require_numeric_effect_size": False,
            "link_to_cure_extraction": False,
        },

        # Learning hints for mathematical refinement
        "learning_hints": {
            "use_advanced_rye_metrics": True,
            "advanced_expectations": {
                "rolling_window_short": 10,
                "rolling_window_long": 40,
                "stability_index_target": 0.65,
                "recovery_momentum_target": 0.08,
                "max_oscillation_std": 0.20,
            },
            "switch_to_maintenance_if": {
                "avg_rye_above": 0.06,
                "stability_index_above": 0.60,
                "oscillation_std_below": 0.18,
                "min_cycles": 30,
            },
            "trigger_exploration_if": {
                "efficiency_trend_negative": True,
                "recovery_momentum_below": 0.0,
                "window": 20,
            },
            "learning_speed_factor": 9.0,
            "enable_learning_bursts": True,
            "burst_modes": {
                "default": {
                    "profile": "balanced",
                    "min_cycles_between_bursts": 20,
                    "max_bursts_per_100_cycles": 6,
                },
                "stagnation_recovery": {
                    "profile": "aggressive",
                    "trigger_if_no_rye_gain_cycles": 25,
                    "trigger_if_efficiency_trend_negative": True,
                },
            },
            "spread_of_learning": {
                "enable_cross_goal_copy": True,
                "max_parent_goals": 2,
                "min_parent_avg_rye": 0.07,
                "min_parent_stability_index": 0.6,
                "reuse_best_equilibrium_labels": True,
                "reuse_best_swarm_contracts": True,
            },
        },

        # Discovery engine (structures and proof ideas)
        "discovery_engine": {
            "enabled": True,
            "threshold_novelty": 0.22,
            "threshold_support": 0.12,
            "detect_mechanisms": False,
            "detect_interventions": False,
            "detect_patterns": True,
            "multi_agent_voting": True,
            "minimum_role_agreement": 3,
            "classification": [
                "mathematical_structure",
                "framework",
                "theorem",
                "lemma",
                "conjecture",
                "prediction",
            ],
        },

        # Swarm orchestration hints
        "swarm_orchestration": {
            "rotation_strategy": "round_robin",
            "role_weighting": {
                "researcher": 1.1,
                "critic": 1.3,
                "planner": 0.9,
                "synthesizer": 1.0,
                "explorer": 0.8,
                "integrator": 1.2,
            },
            "cycle_sync": "full",
            "integration_frequency": 1,
            "consensus_model": "majority",
        },

        # Compute tuning
        "compute_tuning": {
            "energy_to_depth_ratio": 0.8,
            "exploration_vs_exploitation": "explore_heavy",
            "max_energy_spend_per_cycle": "adaptive",
            "rye_weighting_mode": "domain_specific",
        },

        # Evidence strictness
        "evidence_modes": {
            "mode": "mathematical",
            "require_citations": True,
            "min_confidence_to_accept": 0.25,
            "reject_fabricated_sources": True,
            "disallow_uncited_claims": True,
        },

        # Default runtime and RYE behavior for continuous mode
        "default_runtime_profile": "8_hours",
        "default_rye_stop_threshold": 0.05,
        "rye_thresholds": DEFAULT_RYE_THRESHOLDS,
        "advanced_rye_expectations": {
            "rolling_window_short": 10,
            "rolling_window_long": 40,
            "stability_index_target": 0.65,
            "recovery_momentum_target": 0.08,
            "max_oscillation_std": 0.20,
        },

        # Repair efficiency diagnostics hints
        "run_diagnostics": {
            "enabled": True,
            "window": 10,
            "frequency_cycles": 30,
            "include_stability_index": True,
            "include_recovery_momentum": True,
        },

        # Math preset does not extract cures, but can still extract structures
        "cure_extraction": {
            "enabled": False,
            "targets": ["frameworks", "theorems", "formal_models"],
            "notes": (
                "Math preset focuses on formal structures rather than biomedical treatments. "
                "Extraction is about definitions and theorems, not therapies."
            ),
        },

        "report_sections": [
            "summary",
            "definitions",
            "theorems",
            "proof_sketches",
            "citations",
            "rye_statistics",
        ],
        "report_style": "technical",
        "report_frequency": 1,

        # PDF report configuration for math preset
        "pdf_report": {
            "enabled": True,
            "base_defaults": PDF_REPORT_DEFAULTS,
            "sections": [
                "summary",
                "definitions",
                "theorems",
                "proof_sketches",
                "citations",
                "rye_statistics",
            ],
            "include_biomarker_panels": False,
            "include_candidate_stacks": False,
            "breakthrough_overrides": {
                # Breakthrough here is about formal results and structures
                "tier1_thresholds": {
                    "min_high_value_discoveries": 1,
                    "min_peak_rye": 0.08,
                    "min_avg_rye": 0.05,
                },
                "tier2_thresholds": {
                    "min_high_value_discoveries": 3,
                    "min_peak_rye": 0.12,
                    "min_avg_rye": 0.07,
                },
                "tier3_thresholds": {
                    "min_high_value_discoveries": 5,
                    "min_peak_rye": 0.15,
                    "min_avg_rye": 0.09,
                    "require_new_theorem_or_framework": True,
                },
            },
        },

        "runtime_profiles": RUNTIME_PROFILES,

        # Swarm hints for math and theory
        "swarm": {
            "enabled_by_default": False,
            "max_agents": SWARM_GLOBAL_HINTS["max_agents_safe"],
            # Math work often benefits from a slightly smaller, high precision swarm.
            "default_agents": 3,
            "time_split_strategy": "equal",
            "roles": SWARM_ROLES,
            "role_bias": "precision",
            "notes": (
                "Math swarms are tuned for precision and coherence. "
                "Researcher, critic, and theorist style behaviors are most important here."
            ),
            "role_templates": [
                {
                    "name": "formalizer",
                    "inherits_from": "researcher",
                    "domain_focus": "definitions_and_axioms",
                },
                {
                    "name": "proof_critic",
                    "inherits_from": "critic",
                    "domain_focus": "proof_validity_and_gaps",
                },
                {
                    "name": "theory_integrator",
                    "inherits_from": "integrator",
                    "domain_focus": "unifying_existing_theories",
                },
            ],
        },

        "ui_hints": {
            "show_biomarkers_tab": False,
            "show_cure_treatment_tab": False,
            "default_view": "summary",
        },
    },
}

# ---------------------------------------------------------------------
# Aliases so the app can treat different user labels as the same preset
# ---------------------------------------------------------------------
PRESET_ALIASES: Dict[str, str] = {
    # All preset aliases now map to the longevity preset.
    # The math and general presets have been disabled and removed,
    # so any alias that previously pointed to them will instead
    # fall back to longevity. This ensures that older UI inputs or
    # configuration files referencing aliases like "theory" or
    # "general_research" still select the longevity preset.
    "antiaging": "longevity",
    "anti_aging": "longevity",
    "aging": "longevity",
    "anti_ageing": "longevity",
    "longevity_antiaging": "longevity",
    "theory": "longevity",
    "formal": "longevity",
    "general_research": "longevity",
}

# ---------------------------------------------------------------------
# Prune unsupported presets
# ---------------------------------------------------------------------
# Remove any presets that are not longevity.  While the UI and
# get_preset helper override nonâlongevity selections, disabling the
# unused presets here prevents them from appearing in any list of
# presets or being accidentally referenced.  This loop iterates over
# ``PRESETS`` keys and pops those that are not ``"longevity"``.
for _preset_key in list(PRESETS.keys()):
    if str(_preset_key).lower() != "longevity":
        PRESETS.pop(_preset_key, None)


# ---------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------
def get_preset(name: str) -> Dict[str, Any]:
    """Return the longevity preset regardless of the requested name.

    This helper previously returned a preset based on the provided
    `name`, with fallbacks to aliases and then to the "general"
    preset. In this longevity-only build, the math and general
    presets have been removed. To avoid KeyError exceptions and
    guarantee consistent behavior, this function now always returns
    a deep copy of the longevity preset. If the longevity preset
    is missing entirely from ``PRESETS``, an empty dictionary is
    returned instead.
    """
    # Normalize the key but ignore it for lookups. The default when
    # no name is provided is longevity instead of general.
    key = (name or "longevity").lower()
    # Always return the longevity preset if available. This prevents
    # callers from accidentally selecting disabled presets.
    if "longevity" in PRESETS:
        return copy.deepcopy(PRESETS["longevity"])
    # Fallback: if longevity has been removed, return a blank preset
    # rather than raising an exception.
    return {}


def get_runtime_profile(name: str) -> Dict[str, Any]:
    """Return a runtime profile, falling back to 24_hours if unknown."""
    return RUNTIME_PROFILES.get(name, RUNTIME_PROFILES["24_hours"])


def get_continuous_mode_defaults() -> Dict[str, Any]:
    """Return continuous mode defaults used by CoreAgent and engine_worker."""
    return CONTINUOUS_MODE_DEFAULTS


def get_swarm_global_hints() -> Dict[str, Any]:
    """Return global swarm hints and safety ceilings."""
    return SWARM_GLOBAL_HINTS


def list_preset_names() -> List[str]:
    """Return a sorted list of available preset names."""
    return sorted(PRESETS.keys())


def describe_presets() -> List[Dict[str, Any]]:
    """Lightweight summaries for UI or logging."""
    summaries: List[Dict[str, Any]] = []
    for name, cfg in PRESETS.items():
        summaries.append(
            {
                "name": name,
                "label": cfg.get("label", name),
                "domain": cfg.get("domain"),
                "default_goal": cfg.get("default_goal"),
                "default_runtime_profile": cfg.get("default_runtime_profile"),
                "supports_swarm": cfg.get("supports_swarm", False),
                "supports_single_agent": cfg.get("supports_single_agent", True),
                "pdf_report_enabled": cfg.get("pdf_report", {}).get("enabled", False),
            }
        )
    return summaries


# ---------------------------------------------------------------------
# Preset inspector helpers
# ---------------------------------------------------------------------
def build_preset_inspector_snapshot() -> Dict[str, Any]:
    """
    Compact JSON style snapshot of presets for UI inspector panes.

    Example shape per preset:

    {
        "general": {
            "label": "General research",
            "domain": "general",
            "default_goal": "...",
            "default_runtime_profile_key": "8_hours",
            "default_runtime_profile_label": "8 Hour Run",
            "snapshot_enabled": False,
            "supports_swarm": True,
            "supports_single_agent": True,
            "swarm_default_agents": 4,
            "swarm_max_agents": 32,
            "evidence_mode": "balanced",
            "biomarkers_enabled": False,
        },
        ...
    }
    """
    snapshot: Dict[str, Any] = {}

    for name, cfg in PRESETS.items():
        runtime_key = cfg.get("default_runtime_profile", "24_hours")
        runtime_profile = RUNTIME_PROFILES.get(runtime_key, RUNTIME_PROFILES["24_hours"])

        swarm_cfg = cfg.get("swarm") or {}
        snapshot_enabled = bool(
            cfg.get("snapshot_enabled")
            or (cfg.get("snapshot") or {}).get("enabled")
        )

        snapshot[name] = {
            "label": cfg.get("label", name),
            "domain": cfg.get("domain", "general"),
            "default_goal": cfg.get("default_goal"),
            "default_runtime_profile_key": runtime_key,
            "default_runtime_profile_label": runtime_profile.get("label", runtime_key),
            "snapshot_enabled": snapshot_enabled,
            "supports_swarm": cfg.get("supports_swarm", False),
            "supports_single_agent": cfg.get("supports_single_agent", True),
            "swarm_default_agents": swarm_cfg.get(
                "default_agents",
                SWARM_GLOBAL_HINTS.get("default_agents"),
            ),
            "swarm_max_agents": swarm_cfg.get(
                "max_agents",
                SWARM_GLOBAL_HINTS.get("max_agents_safe"),
            ),
            "evidence_mode": (cfg.get("evidence_modes") or {}).get("mode"),
            "biomarkers_enabled": bool(
                (cfg.get("biomarker_intelligence") or {}).get("enabled")
            ),
        }

    return snapshot


def summarize_preset_for_ui(name: str, swarm_size: Optional[int] = None) -> str:
    """
    Turn a preset and optional swarm size into a short human summary line.

    Example:
        summarize_preset_for_ui("longevity", swarm_size=32)

    Could return something like:
        "Longevity / Anti aging, 24 Hour Run, 32 agent swarm, snapshots on,
         high clinical evidence strictness, biomarkers enabled"
    """
    cfg = get_preset(name)

    label = cfg.get("label", name)
    domain = cfg.get("domain", "general")

    runtime_key = cfg.get("default_runtime_profile", "24_hours")
    runtime_profile = get_runtime_profile(runtime_key)
    runtime_label = runtime_profile.get("label", runtime_key)

    # Snapshot state
    snapshot_enabled = bool(
        cfg.get("snapshot_enabled")
        or (cfg.get("snapshot") or {}).get("enabled")
    )

    # Swarm sizing
    swarm_cfg = cfg.get("swarm") or {}
    max_agents = swarm_cfg.get("max_agents", SWARM_GLOBAL_HINTS.get("max_agents_safe"))
    default_agents = swarm_cfg.get(
        "default_agents",
        SWARM_GLOBAL_HINTS.get("default_agents"),
    )

    chosen_swarm: Optional[int]
    if isinstance(swarm_size, int) and swarm_size > 0:
        chosen_swarm = swarm_size
    else:
        chosen_swarm = default_agents if isinstance(default_agents, int) else None

    if isinstance(chosen_swarm, int) and isinstance(max_agents, int):
        chosen_swarm = min(chosen_swarm, max_agents)

    # Evidence and biomarkers
    evidence_mode = (cfg.get("evidence_modes") or {}).get("mode")
    biomarker_enabled = bool(
        (cfg.get("biomarker_intelligence") or {}).get("enabled")
    )

    phrases: List[str] = []

    # Label and runtime
    phrases.append(label)
    if domain and domain.lower() != label.lower():
        phrases.append(f"domain {domain}")
    phrases.append(runtime_label)

    # Swarm phrase
    if isinstance(chosen_swarm, int) and chosen_swarm > 0:
        agent_word = "agent" if chosen_swarm == 1 else "agents"
        phrases.append(f"{chosen_swarm} {agent_word} swarm")

    # Snapshots
    phrases.append("snapshots on" if snapshot_enabled else "snapshots off")

    # Evidence mode phrase
    if evidence_mode:
        if evidence_mode == "clinical":
            phrases.append("high clinical evidence strictness")
        elif evidence_mode == "mathematical":
            phrases.append("formal proof focused evidence")
        elif evidence_mode == "exploratory":
            phrases.append("exploratory evidence mode")
        elif evidence_mode == "balanced":
            phrases.append("balanced evidence mode")
        else:
            phrases.append(f"{evidence_mode} evidence mode")

    # Biomarkers hint
    if biomarker_enabled:
        phrases.append("biomarkers enabled")

    return ", ".join(phrases)
