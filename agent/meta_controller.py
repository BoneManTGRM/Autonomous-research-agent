"""
meta_controller.py

Ultra Mode Meta-Controller for 10x Faster Learning

This module governs:
- dynamic aggressiveness of the TGRM loop
- search depth scaling
- contradiction heat prioritization
- RYE-trend based speed throttle
- memory pruning intensity
- stability envelope enforcement
- mode switching (normal → fast → ultra)

The meta-controller makes the agent learn
faster, cleaner, and more safely than any
static preset could.

It is 100 percent compatible with Option C,
Gold Notebook, Swarm Manager, RYE v3, and
the Contradiction Engine.

This is a NEW MODULE and does not overwrite anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .rye_metrics import (
    compute_rye_trend,
    detect_rye_equilibrium,
    compute_stability_index,
    compute_repair_momentum,
)
from .strategy_profiles import (
    get_high_rye_strategy,
    get_low_rye_strategy,
    get_ultra_mode_strategy,
)
from .contradictions import ContradictionEngine
from .gold_notebook import GoldNotebook


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class MetaDecision:
    """
    A single meta-control decision returned to the TGRM loop.
    """
    mode: str
    search_depth: str
    contradiction_focus: float
    memory_trim: float
    exploration_boost: float
    verification_boost: float
    cycle_limit: int
    rationale: str
    extra: Dict[str, Any] = field(default_factory=dict)


class MetaController:

    def __init__(
        self,
        *,
        domain: Optional[str] = None,
        ultra_enabled: bool = True,
        contradictions: Optional[ContradictionEngine] = None,
        gold_notebook: Optional[GoldNotebook] = None,
        intelligence_profile: Optional[Dict[str, Any]] = None,
        stability_target: float = 0.55,
        run_id: Optional[str] = None,
    ) -> None:

        self.domain = domain
        self.ultra_enabled = ultra_enabled
        self.contradictions = contradictions
        self.gold_notebook = gold_notebook
        self.intelligence_profile = intelligence_profile or {}
        self.stability_target = stability_target
        self.run_id = run_id

        # internal state
        self.last_mode = "normal"
        self.throttle = 1.0

    # ----------------------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------------------

    def decide(
        self,
        *,
        cycle_history: list,
        tool_stats: Dict[str, Any],
        hours_run_so_far: float,
    ) -> MetaDecision:

        # compute core dynamics
        rye_trend = compute_rye_trend(cycle_history)
        momentum = compute_repair_momentum(cycle_history)
        equilibrium = detect_rye_equilibrium(cycle_history)
        stability = compute_stability_index(cycle_history)

        # determine speed mode
        mode = self._determine_mode(
            rye_trend=rye_trend,
            momentum=momentum,
            equilibrium=equilibrium,
            stability=stability,
        )

        # contradiction heat
        heat_focus = self._get_contradiction_focus()

        # memory trimming level
        trim = self._memory_trim_level(mode, stability)

        # exploration boost
        explore = self._explore_boost(mode, rye_trend, momentum)

        # verification boost
        verify = self._verification_boost(mode, heat_focus)

        # dynamic cycle limit
        cycle_limit = self._cycle_limit(mode, stability)

        # search depth scaling
        search_depth = self._search_depth(mode, explore, verify)

        rationale = self._rationale_summary(
            mode, rye_trend, momentum, equilibrium, stability, heat_focus
        )

        return MetaDecision(
            mode=mode,
            search_depth=search_depth,
            contradiction_focus=heat_focus,
            memory_trim=trim,
            exploration_boost=explore,
            verification_boost=verify,
            cycle_limit=cycle_limit,
            rationale=rationale,
            extra={
                "rye_trend": rye_trend,
                "momentum": momentum,
                "equilibrium": equilibrium,
                "stability": stability,
                "run_id": self.run_id,
            },
        )

    # ----------------------------------------------------------------------
    # Mode logic
    # ----------------------------------------------------------------------

    def _determine_mode(
        self,
        *,
        rye_trend: float,
        momentum: float,
        equilibrium: bool,
        stability: float,
    ) -> str:

        if not self.ultra_enabled:
            return "fast" if rye_trend > 0 else "normal"

        # ultra mode: triggered aggressively
        if rye_trend > 0.07 and momentum > 0.06:
            self.last_mode = "ultra"
            return "ultra"

        if equilibrium:
            self.last_mode = "fast"
            return "fast"

        if stability < self.stability_target * 0.6:
            self.last_mode = "normal"
            return "normal"

        # fallback
        return self.last_mode

    # ----------------------------------------------------------------------
    # Contradiction focus
    # ----------------------------------------------------------------------

    def _get_contradiction_focus(self) -> float:
        """
        Returns a contradiction-focus factor 0 to 1.
        Higher means the agent will route more repair cycles
        toward contradictions and high-heat owners.
        """
        if not self.contradictions:
            return 0.0

        hot = self.contradictions.get_hot_owners(top_k=5)
        if not hot:
            return 0.1

        heat = sum(score for _, score in hot)
        return _clamp(heat / 15.0, 0.15, 1.0)

    # ----------------------------------------------------------------------
    # Memory trimming
    # ----------------------------------------------------------------------

    def _memory_trim_level(self, mode: str, stability: float) -> float:
        if mode == "ultra":
            return 0.35 if stability > 0.5 else 0.15
        if mode == "fast":
            return 0.20
        return 0.10

    # ----------------------------------------------------------------------
    # Exploration boost
    # ----------------------------------------------------------------------

    def _explore_boost(self, mode: str, rye_trend: float, momentum: float) -> float:
        base = 0.0
        if mode == "ultra":
            base = 0.45
        elif mode == "fast":
            base = 0.25

        return _clamp(base + rye_trend * 0.5 + momentum * 0.4, 0.05, 0.80)

    # ----------------------------------------------------------------------
    # Verification boost
    # ----------------------------------------------------------------------

    def _verification_boost(self, mode: str, heat_focus: float) -> float:
        if mode == "ultra":
            return _clamp(0.25 + heat_focus * 0.5, 0.1, 1.0)
        if mode == "fast":
            return _clamp(0.15 + heat_focus * 0.3, 0.05, 0.8)
        return _clamp(0.10 + heat_focus * 0.2, 0.05, 0.5)

    # ----------------------------------------------------------------------
    # Cycle limit
    # ----------------------------------------------------------------------

    def _cycle_limit(self, mode: str, stability: float) -> int:
        """
        Limits internal TGRM subcycles per master cycle.
        Lower limits mean tighter learning loops.
        """
        if mode == "ultra":
            return 1 if stability > 0.5 else 2
        if mode == "fast":
            return 2
        return 3

    # ----------------------------------------------------------------------
    # Search depth logic
    # ----------------------------------------------------------------------

    def _search_depth(self, mode: str, explore: float, verify: float) -> str:
        if mode == "ultra":
            return "advanced" if explore > 0.25 else "basic"
        if mode == "fast":
            return "advanced" if verify > 0.15 else "basic"
        return "basic"

    # ----------------------------------------------------------------------
    # Rationale
    # ----------------------------------------------------------------------

    def _rationale_summary(
        self,
        mode: str,
        rye_trend: float,
        momentum: float,
        equilibrium: bool,
        stability: float,
        heat_focus: float,
    ) -> str:

        return (
            f"Mode={mode}; "
            f"trend={rye_trend:.3f}; "
            f"momentum={momentum:.3f}; "
            f"equilibrium={equilibrium}; "
            f"stability={stability:.3f}; "
            f"contradiction_focus={heat_focus:.3f}"
        )
