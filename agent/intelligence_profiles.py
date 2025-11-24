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

Nothing in this file calls tools or APIs. It is pure configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

INTELLIGENCE_VERSION: str = "2025-11-23"


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
#       "exploration": float  # 0 to 1
#       "verification_intensity": float  # 0 to 1
#       "discovery_focus": float  # 0 to 1
#       "tier3_bias": float  # 0 to 1, how much this profile is tuned for major discoveries
#
#       # Autonomy and swarm hints
#       "swarm_recommended": bool,
#       "max_parallel_agents_hint": int,
#       "preferred_swarm_size": int,
#       "use_classic_pair": bool,  # researcher plus critic
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
#       "rye_weight_factor": float,  # multiplier for RYE in decision making
#       "equilibrium_focus": float,  # 0 to 1, fraction of time that should bias toward stability
#       "disruption_budget": float,  # 0 to 1, fraction of cycles allowed for risky exploration
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
            "Balanced intelligence profile, tuned for general research. "
            "Moderate exploration, moderate verification, safe behavior, "
            "and good default for early testing."
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
    },

    # ---------------------------------------------------------------
    # Conservative profile
    # ---------------------------------------------------------------
    "conservative": {
        "label": "Conservative and safe",
        "description": (
            "High safety, strict verification, low hallucination tolerance. "
            "Use when you care about correctness and robustness more than novel discovery."
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
            "Designed for 90 day runs with swarm, discovery engine, verification engine, "
            "and meta controller fully engaged. Very high exploration and discovery focus, "
            "but anchored by strict verification, RYE based sanity checks, and logging."
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
            "Ideal for early phases of a 90 day run before tightening into Tier 3 hunter."
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
    },

    # ---------------------------------------------------------------
    # Ultra stable long run guardian profile
    # Great for proving 90 day continuous stability.
    # ---------------------------------------------------------------
    "guardian_long_run": {
        "label": "Guardian long run stability",
        "description": (
            "Stability first profile meant to maximize the chance of clean, uninterrupted "
            "90 day runs. Emphasizes equilibrium, stability, and RYE plateaus, with a "
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
    },
}


# -------------------------------------------------------------------
# Recommended defaults by domain and runtime profile
# -------------------------------------------------------------------

DEFAULT_PROFILE_BY_DOMAIN: Dict[str, str] = {
    "general": "baseline",
    "longevity": "longevity_clinical",
    "math": "math_formal",
}

DEFAULT_PROFILE_BY_RUNTIME: Dict[str, str] = {
    "1_hour": "baseline",
    "8_hours": "aggressive_discovery",
    "24_hours": "aggressive_discovery",
    "90_days": "tier3_hunter",
    "forever": "guardian_long_run",
}


# -------------------------------------------------------------------
# Accessors
# -------------------------------------------------------------------
def get_intelligence_profile(name: str) -> Dict[str, Any]:
    """
    Return a profile config. If unknown, fall back to baseline.
    """
    return INTELLIGENCE_PROFILES.get(name, INTELLIGENCE_PROFILES["baseline"])


def resolve_profile_for_domain(domain: Optional[str]) -> Dict[str, Any]:
    """
    Resolve a default profile name for a domain, then return the config.
    Falls back to baseline if nothing matches.
    """
    if not domain:
        return get_intelligence_profile("baseline")

    key = str(domain).lower()
    profile_name = DEFAULT_PROFILE_BY_DOMAIN.get(key, "baseline")
    return get_intelligence_profile(profile_name)


def resolve_profile_for_runtime(runtime_profile: Optional[str]) -> Dict[str, Any]:
    """
    Resolve a default profile based on runtime profile name
    such as "1_hour", "8_hours", "24_hours", "90_days", "forever".
    If unknown, returns baseline.
    """
    if not runtime_profile:
        return get_intelligence_profile("baseline")

    key = str(runtime_profile)
    profile_name = DEFAULT_PROFILE_BY_RUNTIME.get(key, "baseline")
    return get_intelligence_profile(profile_name)


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
            }
        )
    return out
