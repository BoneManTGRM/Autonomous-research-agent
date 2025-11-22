# agent/presets.py

"""Domain presets for the Autonomous Research Agent.

Presets act as domain-specific "profiles" that define:
- default goals
- source controls
- domain tags
- query behavior
- RYE weighting hints
- reporting structure
- runtime profiles for 1h / 8h / 24h / forever runs

These settings do NOT break existing code but unlock future power for:
- continuous mode tuning
- energy-aware TGRM behavior
- long-run reporting and summaries
- swarm-aware behavior (many coordinated logical agents)
"""

from __future__ import annotations
from typing import Dict, Any, List

# ---------------------------------------------------------------------
# Global Runtime Profiles (applies for all presets)
# ---------------------------------------------------------------------
RUNTIME_PROFILES = {
    "1_hour": {
        "label": "1 Hour Run",
        "estimated_cycles": 40,
        "rye_stop_threshold": None,
        "energy_scaling": 1.0,
        "report_frequency": 1,
        "description": "Short diagnostic run: fast repair checks and sanity pass."
    },
    "8_hours": {
        "label": "8 Hour Run",
        "estimated_cycles": 200,
        "rye_stop_threshold": 0.05,
        "energy_scaling": 1.2,
        "report_frequency": 1,
        "description": "Medium autonomous session where stable equilibrium patterns can emerge."
    },
    "24_hours": {
        "label": "24 Hour Run",
        "estimated_cycles": 600,
        "rye_stop_threshold": 0.08,
        "energy_scaling": 1.4,
        "report_frequency": 2,
        "description": "Full daily autonomous research loop for equilibrium and deep repairs."
    },
    "forever": {
        "label": "Run Until Stopped",
        "estimated_cycles": 10_000_000,
        "rye_stop_threshold": None,
        "energy_scaling": 1.0,
        "report_frequency": 5,
        "description": "Unbounded autonomous operation until the user or environment stops it."
    },
}

# ---------------------------------------------------------------------
# Swarm defaults (shared across presets)
# ---------------------------------------------------------------------
# This describes what a "swarm" means at the preset level.
# The Streamlit app and CoreAgent can read these hints but are not forced to.
SWARM_ROLES: List[Dict[str, Any]] = [
    {
        "name": "researcher",
        "description": "Primary deep literature and web researcher that gathers facts and writes detailed notes.",
    },
    {
        "name": "critic",
        "description": "Methodology critic and refiner that attacks weak points, gaps, and overclaims.",
    },
    {
        "name": "planner",
        "description": "Planner that proposes next experiments, queries, and high-value repair actions.",
    },
    {
        "name": "synthesizer",
        "description": "Synthesizer that condenses findings into clear narratives, tables, and summaries.",
    },
    {
        "name": "explorer",
        "description": "Out-of-distribution explorer that searches for unusual angles, analogies, and adjacent fields.",
    },
    {
        "name": "integrator",
        "description": "Integrator that reconciles conflicting notes and hypotheses into a coherent picture.",
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
            "reparodynamics", "RYE", "repair yield per energy",
            "TGRM", "self-repair", "autonomous systems",
            "stability", "resilience",
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

        # Domain-level cycle tuning
        "cycle_energy_multiplier": 1.0,
        "cycle_length_hint": "short",
        "repair_depth_bias": "balanced",

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
            "role_bias": "balanced",  # no domain-specific emphasis
            "notes": "General swarm is balanced and good for exploratory research across many topics.",
        },
    },

    # ================================================================
    # LONGEVITY PRESET
    # ================================================================
    "longevity": {
        "label": "Longevity / Anti-aging",
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
            "longevity", "healthspan", "aging", "senescence",
            "autophagy", "mTOR", "NAD+", "rapamycin",
            "metformin", "caloric restriction",
            "clinical trial", "biomarker",
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
            "role_bias": "deep",  # emphasize deep repair and evidence
            "notes": (
                "Longevity swarms are tuned for deep evidence gathering, "
                "biomarker interpretation, and critical review of clinical data."
            ),
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
            "mathematical", "formalization", "axiom", "theorem",
            "proof", "stability theory", "Lyapunov", "information theory",
            "Markov process", "equilibrium",
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
                "Math swarms are tuned for precision and coherence: "
                "researcher, critic, and theorist style roles are most important here."
            ),
        },
    },
}


# ---------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------
def get_preset(name: str) -> Dict[str, Any]:
    """Return a preset config, falling back to 'general' if unknown."""
    return PRESETS.get(name, PRESETS["general"])
