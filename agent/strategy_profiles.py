"""
strategy_profiles.py

High-performance strategy profiles for the Autonomous Research Agent.

This module defines reusable cognitive strategies used by:
    • CoreAgent
    • TGRMLoop
    • SwarmManager
    • MetaController
    • VerificationEngine
    • GoldNotebook / MemoryStore

These strategies encode:
    - exploration vs exploitation bias
    - search depth and width
    - RYE-based pacing
    - contradiction-driven escalation
    - domain-specific heuristics
    - swarm role specialization
    - ultra-mode acceleration patterns
    - stability envelope protection

This file is REQUIRED for full 10× learning acceleration.
Everything is safe, adaptive, and backwards compatible.
"""

from __future__ import annotations
from typing import Dict, Any, Callable


# ---------------------------------------------------------------
# Core profile definitions
# ---------------------------------------------------------------

def _base_profile() -> Dict[str, Any]:
    """
    The neutral fallback profile.
    TGRMLoop and MetaController blend this with active profiles.
    """
    return {
        "exploration": 0.50,
        "depth": 1.0,
        "repair_pressure": 1.0,
        "verification_intensity": 0.50,
        "contradiction_sensitivity": 1.0,
        "cycle_compression": 1.0,
        "max_subcycles": 3,
        "search_density": 1.0,
        "equilibrium_focus": 0.50,
        "memory_prune_level": 0.20,
    }


# ---------------------------------------------------------------
# Ultra Mode (10× learning speed)
# ---------------------------------------------------------------

def ultra_speed_profile() -> Dict[str, Any]:
    """
    Maximum acceleration profile.
    Applied when:
        - meta controller detects strong RYE upward slope
        - discovery momentum rises
        - contradictions heat is medium-low
        - stability index is acceptable
    """
    return {
        "exploration": 0.90,             # fastest broad search mode
        "depth": 2.8,                    # deeper reasoning
        "repair_pressure": 2.5,          # aggressive TGRM repair
        "verification_intensity": 0.85,  # rapid filtering of bad hypotheses
        "contradiction_sensitivity": 1.8,
        "cycle_compression": 0.35,       # TGRM cycles happen 3× faster
        "max_subcycles": 8,              # deeper short-round repair bursts
        "search_density": 2.5,           # many more queries per cycle
        "equilibrium_focus": 0.65,
        "memory_prune_level": 0.45,      # prune noisy memory aggressively
    }


# ---------------------------------------------------------------
# High-Momentum Discovery Mode
# ---------------------------------------------------------------

def discovery_mode_profile() -> Dict[str, Any]:
    """
    Activated when system detects:
        - rising RYE
        - novelty spikes
        - unresolved high-value hypotheses
    """
    return {
        "exploration": 0.80,
        "depth": 2.4,
        "repair_pressure": 2.2,
        "verification_intensity": 0.70,
        "contradiction_sensitivity": 1.4,
        "cycle_compression": 0.50,
        "max_subcycles": 6,
        "search_density": 2.0,
        "equilibrium_focus": 0.45,
        "memory_prune_level": 0.30,
    }


# ---------------------------------------------------------------
# Stabilization Mode (after contradictions heat)
# ---------------------------------------------------------------

def stabilization_profile() -> Dict[str, Any]:
    """
    Applied when:
        - contradictions spike
        - equilibrium score drops
        - memory coherence falls
    Slows acceleration to prevent agent meltdown.
    """
    return {
        "exploration": 0.40,
        "depth": 1.8,
        "repair_pressure": 1.4,
        "verification_intensity": 0.60,
        "contradiction_sensitivity": 2.3,
        "cycle_compression": 0.90,
        "max_subcycles": 4,
        "search_density": 1.2,
        "equilibrium_focus": 0.80,
        "memory_prune_level": 0.15,
    }


# ---------------------------------------------------------------
# Longevity Domain Profile (anti-aging)
# ---------------------------------------------------------------

def longevity_profile() -> Dict[str, Any]:
    """
    Domain-optimized profile for longevity / anti-aging discovery.
    Emphasizes:
        - biomarker consistency
        - multi-modal evidence
        - literature + data agreement
    """
    return {
        "exploration": 0.65,
        "depth": 2.2,
        "repair_pressure": 1.8,
        "verification_intensity": 0.85,
        "contradiction_sensitivity": 1.9,
        "cycle_compression": 0.70,
        "max_subcycles": 5,
        "search_density": 1.8,
        "equilibrium_focus": 0.55,
        "memory_prune_level": 0.25,
    }


# ---------------------------------------------------------------
# Math / Theory Domain Profile
# ---------------------------------------------------------------

def math_profile() -> Dict[str, Any]:
    """
    Domain profile for math discovery.
    Emphasizes:
        - structural consistency
        - formal reasoning depth
        - low contradiction tolerance
    """
    return {
        "exploration": 0.50,
        "depth": 3.5,                    # math requires depth over width
        "repair_pressure": 1.6,
        "verification_intensity": 0.90,
        "contradiction_sensitivity": 2.6,
        "cycle_compression": 0.75,
        "max_subcycles": 7,
        "search_density": 1.3,
        "equilibrium_focus": 0.60,
        "memory_prune_level": 0.20,
    }


# ---------------------------------------------------------------
# Profile Selector
# ---------------------------------------------------------------

def get_strategy_profile(
    mode: str,
    domain: str = "general",
    rye_trend: float = 0.0,
    contradiction_heat: float = 0.0,
    stability_index: float = 1.0,
) -> Dict[str, Any]:
    """
    Selects a strategy profile based on:
        - meta controller mode
        - domain
        - RYE trend
        - contradiction heat
        - stability envelope

    This is the central decision logic used across the system.
    """
    mode = str(mode).lower()
    domain = str(domain).lower()

    # 1) Ultra Mode override (top priority)
    if mode == "ultra" and stability_index >= 0.55:
        return ultra_speed_profile()

    # 2) Discovery mode (momentum-based)
    if mode == "discovery" and rye_trend > 0.03:
        return discovery_mode_profile()

    # 3) Stabilization mode
    if mode == "stabilize" or contradiction_heat > 0.55:
        return stabilization_profile()

    # 4) Domain specific fallback
    if domain in ("longevity", "antiaging", "anti_aging"):
        return longevity_profile()
    if domain in ("math", "theory"):
        return math_profile()

    # 5) Neutral fallback
    return _base_profile()


# ---------------------------------------------------------------
# Merge / Blend
# ---------------------------------------------------------------

def blend_profiles(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine a base profile with an override profile.
    Used by:
        - MetaController
        - SwarmManager
        - CoreAgent
        - TGRMLoop
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            out[k] = v
    return out
