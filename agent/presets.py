# agent/presets.py

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
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

# Simple version tag so the app and UI can display which preset set is loaded.
PRESETS_VERSION: str = "2025-11-25-10x"

# ---------------------------------------------------------------------
# Global Runtime Profiles (applies for all presets)
# These are interpreted by CoreAgent and engine_worker.
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
        "learning_speed_factor": 10.0,
        "burst_profile_hint": "aggressive",
        "spread_of_learning_strength": 1.5,
    },
    "forever": {
        "label": "Run Until Stopped",
        "estimated_cycles": 10_000_000,
        "rye_stop_threshold": None,
        "energy_scaling": 1.0,
        "report_frequency": 5,
        "description": "Unbounded autonomous operation until the user or environment stops it.",
        "use_advanced_rye": True,
        "expected_equilibrium": "mixed",
        "target_rye_range": [0.04, 0.24],
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
# ---------------------------------------------------------------------
CONTINUOUS_MODE_DEFAULTS: Dict[str, Any] = {
    "watchdog_interval_minutes": 5.0,
    "checkpoint_interval_cycles": 10,
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
    "max_agents_safe": 32,
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
    "antiaging": "longevity",
    "anti_aging": "longevity",
    "aging": "longevity",
    "anti_ageing": "longevity",
    "longevity_antiaging": "longevity",
    "theory": "math",
    "formal": "math",
    "general_research": "general",
}


# ---------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------
def get_preset(name: str) -> Dict[str, Any]:
    """Return a preset config, falling back to general if unknown.

    Aliases such as 'antiaging' or 'theory' are mapped to core presets.
    """
    key = (name or "general").lower()
    if key in PRESETS:
        return PRESETS[key]
    alias_target = PRESET_ALIASES.get(key)
    if alias_target and alias_target in PRESETS:
        return PRESETS[alias_target]
    return PRESETS["general"]


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
