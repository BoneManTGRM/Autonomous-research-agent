# agent/presets.py

"""Domain presets for the Autonomous Research Agent.

Presets act as domain-specific "profiles" that define:
- default goals
- source controls
- domain tags
- query behavior
- RYE weighting hints
- reporting structure
- runtime profiles for 1h / 8h / 24h / 90-day / forever runs
- swarm role contracts and intelligence hints for multi-agent operation

These settings do NOT break existing code but unlock future power for:
- continuous mode tuning
- energy aware TGRM behavior
- long run reporting and summaries
- swarm aware behavior (many coordinated logical agents)
- cure or treatment extraction pipelines
- stricter verification and smarter hypothesis selection
"""

from __future__ import annotations
from typing import Dict, Any, List

# ---------------------------------------------------------------------
# Global Runtime Profiles (applies for all presets)
# ---------------------------------------------------------------------
RUNTIME_PROFILES: Dict[str, Dict[str, Any]] = {
    "1_hour": {
        "label": "1 Hour Run",
        "estimated_cycles": 40,
        "rye_stop_threshold": None,
        "energy_scaling": 1.0,
        "report_frequency": 1,
        "description": "Short diagnostic run for fast repair checks and a sanity pass.",
    },
    "8_hours": {
        "label": "8 Hour Run",
        "estimated_cycles": 200,
        "rye_stop_threshold": 0.05,
        "energy_scaling": 1.2,
        "report_frequency": 1,
        "description": "Medium autonomous session where stable equilibrium patterns can emerge.",
    },
    "24_hours": {
        "label": "24 Hour Run",
        "estimated_cycles": 600,
        "rye_stop_threshold": 0.08,
        "energy_scaling": 1.4,
        "report_frequency": 2,
        "description": "Full daily autonomous research loop for equilibrium and deep repairs.",
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
    },
    "forever": {
        "label": "Run Until Stopped",
        "estimated_cycles": 10_000_000,
        "rye_stop_threshold": None,
        "energy_scaling": 1.0,
        "report_frequency": 5,
        "description": "Unbounded autonomous operation until the user or environment stops it.",
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
}

# ---------------------------------------------------------------------
# Swarm defaults (shared across presets)
# ---------------------------------------------------------------------
# This describes what a "swarm" means at the preset level.
# The Streamlit app and CoreAgent can read these hints but are not forced to.

# Global, domain-agnostic role templates with contracts
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
    # App and core can still choose a lower local limit.
    "max_agents_safe": 32,
    # Good default for most presets when user clicks "swarm".
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

        "default_goal": (
            "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
            "identify similar frameworks in the literature, and produce a structured comparison table."
        ),

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

        # Default runtime and RYE behavior for continuous mode
        "default_runtime_profile": "8_hours",
        "default_rye_stop_threshold": None,
        "rye_thresholds": DEFAULT_RYE_THRESHOLDS,

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
            # Optional smarter role templates (future engines can use these instead of the generic roles)
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

        # Default runtime and RYE behavior for continuous mode
        "default_runtime_profile": "24_hours",
        "default_rye_stop_threshold": 0.08,
        "rye_thresholds": DEFAULT_RYE_THRESHOLDS,

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
            # Specific 5-role longevity swarm template for highly intelligent runs
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

        # Default runtime and RYE behavior for continuous mode
        "default_runtime_profile": "8_hours",
        "default_rye_stop_threshold": 0.05,
        "rye_thresholds": DEFAULT_RYE_THRESHOLDS,

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
# Accessors
# ---------------------------------------------------------------------
def get_preset(name: str) -> Dict[str, Any]:
    """Return a preset config, falling back to 'general' if unknown."""
    return PRESETS.get(name, PRESETS["general"])


def get_runtime_profile(name: str) -> Dict[str, Any]:
    """Return a runtime profile, falling back to 24_hours if unknown."""
    return RUNTIME_PROFILES.get(name, RUNTIME_PROFILES["24_hours"])
