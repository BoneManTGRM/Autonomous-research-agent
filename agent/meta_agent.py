# agent/meta_agent.py

"""Meta agent orchestration layer for the Autonomous Research Agent.

This module defines a MetaAgent that sits on top of:
    - CoreAgent and TGRM cycles
    - Presets and runtime profiles
    - Advanced RYE diagnostics
    - Stability kernel analytics
    - Optional MSIL (Meta Skill Intelligence Layer)

Goals:
    - Plan continuous runs using presets and runtime profiles
    - Keep run planning aligned with real wall clock budgets
    - Aggregate stability, learning, and discovery signals
    - Provide intelligence profiles for UI and MSIL
    - Suggest swarm, runtime, and curriculum configurations without
      changing the underlying run time settings

MetaAgent never overrides the real max_minutes budget used by
CoreAgent or engine_worker. It only reads presets and continuous
mode defaults and returns structured hints.

It is safe to ignore if you want only the basic CoreAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import math
import statistics

from .core_agent import CoreAgent
from .memory_store import MemoryStore
from .presets import (
    get_preset,
    get_runtime_profile,
    get_continuous_mode_defaults,
)
from .rye_metrics import build_run_diagnostics

# Stability kernel is expected to exist. Everything is optional gated.
from . import stability_kernel as stability  # type: ignore[import]

# Optional MSIL support
try:
    from .msil import MetaSkillIntelligenceLayer  # type: ignore[import]
except Exception:  # pragma: no cover
    MetaSkillIntelligenceLayer = None  # type: ignore[assignment]


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------


@dataclass
class RunPlan:
    """Structured run plan for engine_worker or local tests."""

    run_id: str
    preset_name: str
    domain: str
    goal: str
    runtime_profile_name: str
    mode: str  # "single", "multi", "swarm"
    created_utc: str
    max_minutes: Optional[float]
    forever: bool
    swarm_enabled: bool
    swarm_size: int
    multi_agent_pair: bool
    source_controls: Dict[str, Any] = field(default_factory=dict)
    preset_snapshot: Dict[str, Any] = field(default_factory=dict)
    runtime_profile: Dict[str, Any] = field(default_factory=dict)
    continuous_defaults: Dict[str, Any] = field(default_factory=dict)
    learning_hints: Dict[str, Any] = field(default_factory=dict)
    discovery_hints: Dict[str, Any] = field(default_factory=dict)
    biomarker_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StabilitySummary:
    """High level stability and RYE summary for UI and MSIL."""

    diagnostics: Dict[str, Any] = field(default_factory=dict)
    stability_profile: Dict[str, Any] = field(default_factory=dict)
    stability_regime: Dict[str, Any] = field(default_factory=dict)
    ui_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligenceProfile:
    """Combined view of stability, learning, discoveries, and MSIL."""

    preset_name: str
    domain: str
    run_id: Optional[str]
    total_cycles: int
    hours_run: Optional[float]
    stability: StabilitySummary
    msil_profile: Optional[Dict[str, Any]]
    learning_hints: Dict[str, Any]
    swarm_hints: Dict[str, Any]
    breakthrough_profile: Dict[str, Any]

    # MAX++ extensions
    meta_learning: Dict[str, Any] = field(default_factory=dict)
    trajectory_forecast: Dict[str, Any] = field(default_factory=dict)
    curriculum_plan: Dict[str, Any] = field(default_factory=dict)
    swarm_topology: Dict[str, Any] = field(default_factory=dict)
    meta_stability: Dict[str, Any] = field(default_factory=dict)
    breakthrough_probability: Dict[str, Any] = field(default_factory=dict)
    ui_summary: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def _approx_hours_from_history(history: List[Dict[str, Any]]) -> Optional[float]:
    """Approximate total hours between first and last cycle timestamps."""
    if not history:
        return None
    timestamps: List[datetime] = []
    for e in history:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            try:
                timestamps.append(datetime.fromisoformat(ts)
                                  if "T" in ts
                                  else datetime.fromisoformat(ts + "Z"))
            except Exception:
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except Exception:
                    continue
    if len(timestamps) < 2:
        return None
    start = min(timestamps)
    end = max(timestamps)
    delta = end - start
    return max(delta.total_seconds() / 3600.0, 0.0)


def _safe_copy_small(obj: Any, max_keys: int = 64) -> Any:
    """Take a shallow copy of a dict or list suitable for logs or UI."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_keys:
                break
            out[k] = v
        return out
    if isinstance(obj, list):
        return obj[:max_keys]
    return obj


def _safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    try:
        return float(statistics.mean(vals))
    except Exception:
        return None


def _safe_std(values: List[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if len(vals) < 2:
        return None
    try:
        return float(statistics.pstdev(vals))
    except Exception:
        return None


# -------------------------------------------------------------------
# MetaAgent
# -------------------------------------------------------------------


class MetaAgent:
    """High level orchestration and intelligence layer.

    MetaAgent keeps CoreAgent as the only component that actually
    executes TGRM cycles. This layer:

        - Loads presets and runtime profiles
        - Plans runs for engine_worker
        - Pulls diagnostics from RYE and stability kernel
        - Optionally calls MSIL for intelligence scoring
        - Suggests swarm configurations from presets
        - Builds cross run meta learning summaries
        - Predicts simple intelligence trajectories
        - Proposes curricula and training sequences

    It is designed to be safe to import in any environment.
    If a dependency such as MSIL or stability_kernel is missing
    any methods, MetaAgent falls back to partial behavior.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        preset_name: str = "general",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory = memory_store
        self.config = config or {}
        self.core_agent = CoreAgent(memory_store=self.memory, config=self.config)

        self.preset_name = preset_name
        self.preset = get_preset(preset_name)
        self.domain = self.preset.get("domain", "general")

        self.runtime_profile_name = self.preset.get(
            "default_runtime_profile", "24_hours"
        )
        self.runtime_profile = get_runtime_profile(self.runtime_profile_name)
        self.continuous_defaults = get_continuous_mode_defaults()

        # Optional MSIL instance
        self._msil = None
        if MetaSkillIntelligenceLayer is not None:
            try:
                # Preferred constructor
                self._msil = MetaSkillIntelligenceLayer(  # type: ignore[call-arg]
                    memory_store=self.memory,
                    config=self.config,
                    preset=self.preset,
                )
            except TypeError:
                try:
                    self._msil = MetaSkillIntelligenceLayer(  # type: ignore[call-arg]
                        memory_store=self.memory,
                        config=self.config,
                    )
                except Exception:
                    try:
                        self._msil = MetaSkillIntelligenceLayer(self.memory)  # type: ignore[call-arg]
                    except Exception:
                        self._msil = None

    # ------------------------------------------------------------------
    # Preset and profile management
    # ------------------------------------------------------------------

    def reload_preset(self, preset_name: Optional[str] = None) -> None:
        """Reload preset and runtime profile."""
        if preset_name is None:
            preset_name = self.preset_name
        self.preset_name = preset_name
        self.preset = get_preset(preset_name)
        self.domain = self.preset.get("domain", "general")
        self.runtime_profile_name = self.preset.get(
            "default_runtime_profile", self.runtime_profile_name
        )
        self.runtime_profile = get_runtime_profile(self.runtime_profile_name)

    def set_runtime_profile(self, profile_name: str) -> None:
        """Set a specific runtime profile name for planning."""
        self.runtime_profile_name = profile_name
        self.runtime_profile = get_runtime_profile(profile_name)

    # ------------------------------------------------------------------
    # Run planning
    # ------------------------------------------------------------------

    def plan_run(
        self,
        goal: Optional[str] = None,
        max_minutes: Optional[float] = None,
        mode: str = "single",
        swarm_size: Optional[int] = None,
        multi_agent_pair: bool = False,
        forever: bool = False,
        runtime_profile_name: Optional[str] = None,
    ) -> RunPlan:
        """Create a structured run plan without starting any cycles.

        This is intended for engine_worker to consume. It does not
        modify any real time settings on its own.

        Arguments:
            goal:
                Target research goal. If None, uses preset default.
            max_minutes:
                Hard time budget in minutes. If None and forever is False,
                engine_worker can derive it from the selected run mode.
            mode:
                "single", "multi", or "swarm".
            swarm_size:
                Number of swarm agents if mode is "swarm". If None, uses
                preset swarm defaults.
            multi_agent_pair:
                If True, run researcher plus critic in simple multi agent mode.
            forever:
                If True, this plan is interpreted as run until stopped.
            runtime_profile_name:
                Optional override of the preset runtime profile.
        """
        preset = self.preset
        domain = self.domain

        if runtime_profile_name is not None:
            self.set_runtime_profile(runtime_profile_name)

        rp = self.runtime_profile
        cont = self.continuous_defaults

        swarm_cfg = preset.get("swarm", {})
        max_safe = int(swarm_cfg.get("max_agents", 32))
        default_swarm_size = int(swarm_cfg.get("default_agents", 4))

        if mode == "swarm":
            size = swarm_size or default_swarm_size
            size = max(2, min(size, max_safe))
            swarm_enabled = True
        else:
            size = 0
            swarm_enabled = False

        default_goal = preset.get("default_goal") or (
            "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
            "and compare them to related frameworks."
        )
        final_goal = goal or default_goal

        # Build source controls based on preset
        src_ctrl = dict(preset.get("source_controls", {}))

        # Learning, discovery, and biomarker hints
        learning_hints = dict(preset.get("learning_hints", {}))
        discovery_hints = dict(preset.get("discovery_engine", {}))
        biomarker_hints = dict(preset.get("biomarker_intelligence", {}))

        plan = RunPlan(
            run_id=str(uuid.uuid4()),
            preset_name=self.preset_name,
            domain=domain,
            goal=final_goal,
            runtime_profile_name=self.runtime_profile_name,
            mode=mode,
            created_utc=datetime.utcnow().isoformat(),
            max_minutes=max_minutes,
            forever=bool(forever),
            swarm_enabled=swarm_enabled,
            swarm_size=size,
            multi_agent_pair=bool(multi_agent_pair),
            source_controls=_safe_copy_small(src_ctrl, max_keys=32),
            preset_snapshot=_safe_copy_small(preset, max_keys=128),
            runtime_profile=_safe_copy_small(rp, max_keys=64),
            continuous_defaults=_safe_copy_small(cont, max_keys=64),
            learning_hints=_safe_copy_small(learning_hints, max_keys=64),
            discovery_hints=_safe_copy_small(discovery_hints, max_keys=64),
            biomarker_hints=_safe_copy_small(biomarker_hints, max_keys=64),
        )
        return plan

    # ------------------------------------------------------------------
    # Local test runner (short sessions)
    # ------------------------------------------------------------------

    def run_short_session(
        self,
        goal: Optional[str] = None,
        cycles: int = 3,
        mode: str = "single",
        swarm_roles: Optional[List[Tuple[str, str]]] = None,
        source_controls_override: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a short local session (manual cycles) using CoreAgent.

        This is similar to the Streamlit manual run path and is only
        intended for local tests and diagnostics.

        It does not change continuous mode behavior or time budgets.
        """
        preset = self.preset
        domain_tag = preset.get("domain", self.domain)

        default_goal = preset.get("default_goal") or (
            "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
            "and compare them to related frameworks."
        )
        final_goal = goal or default_goal

        base_source_controls = dict(preset.get("source_controls", {}))
        if source_controls_override:
            base_source_controls.update(source_controls_override)

        history = self.memory.get_cycle_history()
        next_index = len(history)
        results: List[Dict[str, Any]] = []

        # Swarm manual mode
        if mode == "swarm" and swarm_roles:
            for i in range(int(cycles)):
                base_index = next_index + i * len(swarm_roles)
                for j, (role_name, _) in enumerate(swarm_roles):
                    ci = base_index + j
                    role_goal = self._role_specific_goal(final_goal, role_name)
                    out = self.core_agent.run_cycle(
                        goal=role_goal,
                        cycle_index=ci,
                        role=role_name,
                        source_controls=base_source_controls,
                        pdf_bytes=None,
                        biomarker_snapshot=None,
                        domain=domain_tag,
                    )
                    results.append(out["summary"])
            return results

        # Simple single agent
        if mode == "single":
            for i in range(int(cycles)):
                ci = next_index + i
                out = self.core_agent.run_cycle(
                    goal=final_goal,
                    cycle_index=ci,
                    role="agent",
                    source_controls=base_source_controls,
                    pdf_bytes=None,
                    biomarker_snapshot=None,
                    domain=domain_tag,
                )
                results.append(out["summary"])
            return results

        # Classic researcher plus critic
        if mode == "multi":
            for i in range(int(cycles)):
                base_idx = next_index + 2 * i

                r = self.core_agent.run_cycle(
                    goal=final_goal,
                    cycle_index=base_idx,
                    role="researcher",
                    source_controls=base_source_controls,
                    pdf_bytes=None,
                    biomarker_snapshot=None,
                    domain=domain_tag,
                )
                results.append(r["summary"])

                critic_goal = f"Critically review and refine notes for: {final_goal}"
                c = self.core_agent.run_cycle(
                    goal=critic_goal,
                    cycle_index=base_idx + 1,
                    role="critic",
                    source_controls=base_source_controls,
                    pdf_bytes=None,
                    biomarker_snapshot=None,
                    domain=domain_tag,
                )
                results.append(c["summary"])

            return results

        # Fallback to single
        return self.run_short_session(
            goal=final_goal,
            cycles=cycles,
            mode="single",
            swarm_roles=None,
            source_controls_override=source_controls_override,
        )

    @staticmethod
    def _role_specific_goal(base_goal: str, role: str) -> str:
        """Specialize the goal text slightly for each role."""
        base_goal = base_goal.strip()
        archetype = role.split("_", 1)[0] if "_" in role else role

        if archetype == "researcher":
            return (
                f"Primary deep research agent for goal: {base_goal}.\n"
                "Focus on high quality sources, detailed notes, and clear summaries."
            )
        if archetype == "critic":
            return (
                f"Critically review and refine all Reparodynamic notes and hypotheses for: {base_goal}.\n"
                "Identify weaknesses, gaps, and overclaims."
            )
        if archetype == "explorer":
            return (
                f"Exploration agent for goal: {base_goal}.\n"
                "Look for unusual angles, analogies, adjacent fields, and surprising connections."
            )
        if archetype == "integrator":
            return (
                f"Integration agent for goal: {base_goal}.\n"
                "Synthesize results into coherent narratives, tables, and distilled insights."
            )
        if archetype == "planner":
            return (
                f"Planning agent for goal: {base_goal}.\n"
                "Propose the highest value next actions that would increase RYE."
            )
        return base_goal

    # ------------------------------------------------------------------
    # Stability and intelligence profiles
    # ------------------------------------------------------------------

    def compute_stability_summary(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> StabilitySummary:
        """Build a StabilitySummary using RYE diagnostics and stability kernel."""
        if history is None:
            history = self.memory.get_cycle_history()

        diagnostics: Dict[str, Any] = {}
        try:
            diagnostics = build_run_diagnostics(history=history, domain=None, window=10)
        except TypeError:
            try:
                diagnostics = build_run_diagnostics(history, None, 10)  # type: ignore[arg-type]
            except Exception:
                diagnostics = {}

        stability_profile: Dict[str, Any] = {}
        if hasattr(stability, "compute_stability_profile"):
            try:
                stability_profile = stability.compute_stability_profile(history)  # type: ignore[attr-defined]
            except Exception:
                stability_profile = {}

        regime: Dict[str, Any] = {}
        if hasattr(stability, "classify_stability_regime"):
            try:
                regime = stability.classify_stability_regime(
                    history=history,
                    diagnostics=diagnostics,
                    profile=stability_profile,
                )  # type: ignore[attr-defined]
            except Exception:
                regime = {}

        ui_snapshot: Dict[str, Any] = {}
        if hasattr(stability, "summarize_for_ui"):
            try:
                ui_snapshot = stability.summarize_for_ui(
                    history=history,
                    diagnostics=diagnostics,
                    profile=stability_profile,
                    regime=regime,
                )  # type: ignore[attr-defined]
            except Exception:
                ui_snapshot = {}

        return StabilitySummary(
            diagnostics=diagnostics,
            stability_profile=stability_profile,
            stability_regime=regime,
            ui_snapshot=ui_snapshot,
        )

    def compute_intelligence_profile(
        self,
        run_id: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> IntelligenceProfile:
        """Compute a combined intelligence profile for dashboards and MSIL."""
        if history is None:
            history = self.memory.get_cycle_history()

        total_cycles = len(history)
        hours_run = _approx_hours_from_history(history)
        stability_summary = self.compute_stability_summary(history=history)

        # Optional MSIL profile
        msil_profile: Optional[Dict[str, Any]] = None
        if self._msil is not None:
            try:
                # Preferred method name
                if hasattr(self._msil, "evaluate_run"):
                    msil_profile = self._msil.evaluate_run(  # type: ignore[attr-defined]
                        history=history,
                        diagnostics=stability_summary.diagnostics,
                        stability_profile=stability_summary.stability_profile,
                    )
                elif hasattr(self._msil, "analyze_run"):
                    msil_profile = self._msil.analyze_run(  # type: ignore[attr-defined]
                        history=history,
                        diagnostics=stability_summary.diagnostics,
                    )
                elif hasattr(self._msil, "to_dict"):
                    msil_profile = self._msil.to_dict()  # type: ignore[attr-defined]
            except Exception:
                msil_profile = None

        learning_hints = dict(self.preset.get("learning_hints", {}))
        swarm_hints = dict(self.preset.get("swarm", {}))

        # Breakthrough tier and thresholds
        tier_hints = self._extract_breakthrough_hints()

        breakthrough_profile: Dict[str, Any] = {
            "tier_hints": _safe_copy_small(tier_hints, max_keys=64),
            "stability_regime": _safe_copy_small(
                stability_summary.stability_regime, max_keys=16
            ),
            "hours_run": hours_run,
            "total_cycles": total_cycles,
        }

        # MAX++ extensions
        meta_learning = self._analyze_cross_run_meta()
        trajectory_forecast = self._forecast_intelligence_trajectory(
            stability_summary=stability_summary,
            msil_profile=msil_profile,
            total_cycles=total_cycles,
            hours_run=hours_run,
        )
        curriculum_plan = self._build_curriculum_plan(
            trajectory_forecast=trajectory_forecast,
            meta_learning=meta_learning,
        )
        swarm_topology = self._map_swarm_topology(
            history=history,
            stability_summary=stability_summary,
        )
        meta_stability = self._classify_meta_stability_strata(
            stability_summary=stability_summary,
            hours_run=hours_run,
            total_cycles=total_cycles,
        )
        breakthrough_probability = self._compute_breakthrough_probability(
            stability_summary=stability_summary,
            msil_profile=msil_profile,
            meta_learning=meta_learning,
            tier_hints=tier_hints,
        )
        ui_summary = self._summarize_for_ui(
            stability_summary=stability_summary,
            trajectory_forecast=trajectory_forecast,
            meta_stability=meta_stability,
            breakthrough_probability=breakthrough_probability,
        )

        return IntelligenceProfile(
            preset_name=self.preset_name,
            domain=self.domain,
            run_id=run_id,
            total_cycles=total_cycles,
            hours_run=hours_run,
            stability=stability_summary,
            msil_profile=msil_profile,
            learning_hints=_safe_copy_small(learning_hints, max_keys=64),
            swarm_hints=_safe_copy_small(swarm_hints, max_keys=64),
            breakthrough_profile=_safe_copy_small(breakthrough_profile, max_keys=64),
            meta_learning=_safe_copy_small(meta_learning, max_keys=64),
            trajectory_forecast=_safe_copy_small(
                trajectory_forecast, max_keys=64
            ),
            curriculum_plan=_safe_copy_small(curriculum_plan, max_keys=64),
            swarm_topology=_safe_copy_small(swarm_topology, max_keys=64),
            meta_stability=_safe_copy_small(meta_stability, max_keys=64),
            breakthrough_probability=_safe_copy_small(
                breakthrough_probability, max_keys=64
            ),
            ui_summary=_safe_copy_small(ui_summary, max_keys=64),
        )

    def _extract_breakthrough_hints(self) -> Dict[str, Any]:
        """Pull tier thresholds from preset pdf_report config if present."""
        pdf_cfg = self.preset.get("pdf_report", {})
        overrides = pdf_cfg.get("breakthrough_overrides", {})
        base_defaults = pdf_cfg.get("base_defaults", {})
        base_bt = base_defaults.get("breakthrough_hints", {})
        merged: Dict[str, Any] = {}
        if base_bt:
            merged.update(_safe_copy_small(base_bt, max_keys=64))
        if overrides:
            merged.update(_safe_copy_small(overrides, max_keys=64))
        return merged

    # ------------------------------------------------------------------
    # Swarm configuration helpers
    # ------------------------------------------------------------------

    def suggest_swarm_size(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        current_size: Optional[int] = None,
    ) -> int:
        """Suggest a swarm size based on preset hints and simple stability signals.

        This does not change any real run settings. It only suggests a
        size that engine_worker or UI can choose to apply.
        """
        swarm_cfg = self.preset.get("swarm", {})
        max_safe = int(swarm_cfg.get("max_agents", 32))
        default_size = int(swarm_cfg.get("default_agents", 4))

        if current_size is None or current_size <= 0:
            base = default_size
        else:
            base = current_size

        if history is None:
            history = self.memory.get_cycle_history()

        if not history:
            return max(2, min(base, max_safe))

        stability_summary = self.compute_stability_summary(history=history)
        regime = stability_summary.stability_regime or {}
        label = str(regime.get("label") or regime.get("regime") or "").lower()
        score = regime.get("score")

        suggested = base

        # If regime says unstable, shrink a bit.
        if "unstable" in label or "fragile" in label:
            suggested = max(2, base - 2)
        # If regime is robust and score looks high, allow a slight bump.
        elif "robust" in label or (
            isinstance(score, (int, float)) and score >= 0.75
        ):
            suggested = min(base + 2, max_safe)

        return max(2, min(suggested, max_safe))

    def suggest_runtime_profile_name(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Suggest a runtime profile based on stability and learning hints."""
        learning = self.preset.get("learning_hints", {})
        switch_cond = learning.get("switch_to_maintenance_if", {})

        if history is None:
            history = self.memory.get_cycle_history()

        diagnostics: Dict[str, Any] = {}
        try:
            diagnostics = build_run_diagnostics(history=history, domain=None, window=10)
        except Exception:
            diagnostics = {}

        avg_rye = diagnostics.get("rolling_rye")
        stability_index = diagnostics.get("stability_index")
        osc_std = diagnostics.get("oscillation_std")

        avg_thresh = switch_cond.get("avg_rye_above")
        stab_thresh = switch_cond.get("stability_index_above")
        osc_thresh = switch_cond.get("oscillation_std_below")
        min_cycles = switch_cond.get("min_cycles", 0)

        cycles = len(history)

        can_switch = True
        if isinstance(avg_thresh, (int, float)) and isinstance(avg_rye, (int, float)):
            if avg_rye < avg_thresh:
                can_switch = False
        if isinstance(stab_thresh, (int, float)) and isinstance(
            stability_index, (int, float)
        ):
            if stability_index < stab_thresh:
                can_switch = False
        if isinstance(osc_thresh, (int, float)) and isinstance(
            osc_std, (int, float)
        ):
            if osc_std > osc_thresh:
                can_switch = False
        if cycles < int(min_cycles):
            can_switch = False

        if can_switch:
            # For now switch to 1_week profile hint if everything looks stable.
            if "1_week" in self.preset.get("runtime_profiles", {}):
                return "1_week"
        return self.runtime_profile_name

    # ------------------------------------------------------------------
    # MAX++ meta learning and trajectory layers
    # ------------------------------------------------------------------

    def _analyze_cross_run_meta(self) -> Dict[str, Any]:
        """Analyze patterns across runs if the MemoryStore exposes them.

        Fully optional. Falls back to a single run summary if cross run
        APIs are not available on the memory store.
        """
        meta: Dict[str, Any] = {
            "supported": False,
            "runs_seen": 0,
            "global_rye_mean": None,
            "global_rye_std": None,
            "best_runs": [],
            "domains": {},
        }

        # Try several possible introspection methods without assuming any one exists.
        runs: List[Dict[str, Any]] = []
        try:
            if hasattr(self.memory, "get_run_summaries"):
                rs = self.memory.get_run_summaries()  # type: ignore[attr-defined]
                if isinstance(rs, list):
                    runs = rs
            elif hasattr(self.memory, "list_runs"):
                rs = self.memory.list_runs()  # type: ignore[attr-defined]
                if isinstance(rs, list):
                    runs = rs
        except Exception:
            runs = []

        if not runs:
            # Degenerate case. Try to at least infer a single run meta view.
            history = self.memory.get_cycle_history()
            if not history:
                return meta
            meta["supported"] = False
            meta["runs_seen"] = 1
            try:
                diagnostics = build_run_diagnostics(
                    history=history, domain=None, window=10
                )
            except Exception:
                diagnostics = {}
            meta["global_rye_mean"] = diagnostics.get("rolling_rye")
            meta["best_runs"] = []
            return meta

        meta["supported"] = True
        meta["runs_seen"] = len(runs)

        # Aggregate RYE and basic signals if present.
        rye_values: List[float] = []
        domains: Dict[str, Dict[str, Any]] = {}

        for r in runs:
            dom = r.get("domain") or "unknown"
            stats = domains.setdefault(
                dom,
                {
                    "count": 0,
                    "rye_values": [],
                    "avg_rye": None,
                    "median_rye": None,
                    "best_run": None,
                },
            )
            stats["count"] += 1
            avg_rye = r.get("avg_rye")
            if isinstance(avg_rye, (int, float)):
                rye_values.append(avg_rye)
                stats["rye_values"].append(avg_rye)

        meta["global_rye_mean"] = _safe_mean(rye_values)
        meta["global_rye_std"] = _safe_std(rye_values)

        # Per domain aggregates and top runs
        best_runs: List[Dict[str, Any]] = []
        for dom, stats in domains.items():
            rv = stats.get("rye_values", [])
            stats["avg_rye"] = _safe_mean(rv)
            try:
                stats["median_rye"] = float(statistics.median(rv)) if rv else None
            except Exception:
                stats["median_rye"] = None

            # Track best run per domain
            best_run: Optional[Dict[str, Any]] = None
            best_val: float = float("-inf")
            for r in runs:
                if (r.get("domain") or "unknown") != dom:
                    continue
                v = r.get("avg_rye")
                if isinstance(v, (int, float)) and v > best_val:
                    best_val = v
                    best_run = r
            stats["best_run"] = best_run
            if best_run:
                best_runs.append(
                    {
                        "run_id": best_run.get("run_id"),
                        "domain": dom,
                        "avg_rye": best_run.get("avg_rye"),
                        "stability_index": best_run.get("stability_index"),
                    }
                )

        meta["domains"] = domains
        # Sort best runs by avg_rye if available
        best_runs_sorted = sorted(
            best_runs,
            key=lambda x: x.get("avg_rye") or 0.0,
            reverse=True,
        )
        meta["best_runs"] = best_runs_sorted[:10]
        return meta

    def _forecast_intelligence_trajectory(
        self,
        stability_summary: StabilitySummary,
        msil_profile: Optional[Dict[str, Any]],
        total_cycles: int,
        hours_run: Optional[float],
    ) -> Dict[str, Any]:
        """Predict simple RYE and stability trajectories.

        This is not an ML model. It is a transparent analytic forecast
        based on current slopes and indices.
        """
        diagnostics = stability_summary.diagnostics or {}
        slope = diagnostics.get("regression_rye_slope")
        current_avg = diagnostics.get("rolling_rye")
        stability_index = diagnostics.get("stability_index")
        recovery_momentum = diagnostics.get("recovery_momentum")

        # Use MSIL score if exposed
        msil_score = None
        if msil_profile and isinstance(msil_profile, dict):
            msil_score = msil_profile.get("msil_score")

        def project(value: Optional[float], step: float, factor: float) -> Optional[float]:
            if not isinstance(value, (int, float)):
                return None
            if not isinstance(step, (int, float)):
                step = 0.0
            return float(max(value + step * factor, -1.0))

        steps = {
            "24h": 6.0,
            "1w": 7.0 * 6.0,
            "1m": 30.0 * 6.0,
            "90d": 90.0 * 6.0,
        }

        forecasts: Dict[str, Any] = {}
        for label, factor in steps.items():
            forecasts[label] = {
                "rye": project(current_avg, slope, factor)
                if isinstance(slope, (int, float))
                else current_avg,
                "stability_index": stability_index,
            }

        # Simple qualitative trajectory tag
        trend = "flat"
        if isinstance(slope, (int, float)):
            if slope > 0.001:
                trend = "up"
            elif slope < -0.001:
                trend = "down"

        stability_trend = "unknown"
        if isinstance(recovery_momentum, (int, float)):
            if recovery_momentum > 0.05:
                stability_trend = "improving"
            elif recovery_momentum < -0.05:
                stability_trend = "degrading"
            else:
                stability_trend = "stable_zone"

        return {
            "supported": True,
            "trend": trend,
            "stability_trend": stability_trend,
            "current_avg_rye": current_avg,
            "current_stability_index": stability_index,
            "recovery_momentum": recovery_momentum,
            "msil_score": msil_score,
            "total_cycles": total_cycles,
            "hours_run": hours_run,
            "forecast": forecasts,
        }

    def _build_curriculum_plan(
        self,
        trajectory_forecast: Dict[str, Any],
        meta_learning: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Propose a curriculum plan for future goals and runs.

        Uses current domain, RYE trajectory, and best runs in meta view.
        """
        plan: Dict[str, Any] = {
            "supported": True,
            "priority": [],
            "recommendations": [],
        }

        domain = self.domain
        forecast = trajectory_forecast or {}
        trend = forecast.get("trend")
        stability_trend = forecast.get("stability_trend")
        msil_score = forecast.get("msil_score")
        current_avg = forecast.get("current_avg_rye")

        # Cross run data if available
        domains_meta = (meta_learning or {}).get("domains", {})
        domain_meta = domains_meta.get(domain, {})

        # Base recommendation skeletons
        recs: List[Dict[str, Any]] = []

        # If RYE is trending up and stability improving, push toward deeper profiles.
        if trend == "up" and stability_trend == "improving":
            recs.append(
                {
                    "type": "deepening",
                    "description": (
                        "Run longer profiles in the same domain to consolidate gains. "
                        "Prefer 24h or 1 week profiles before 90 day experiments."
                    ),
                    "suggested_profiles": ["24_hours", "1_week"],
                }
            )

        # If RYE is flat and stability is decent, propose exploration.
        if trend == "flat" and isinstance(current_avg, (int, float)):
            if current_avg >= 0.06:
                recs.append(
                    {
                        "type": "exploration",
                        "description": (
                            "Current RYE is stable but flat. Add exploratory goals in adjacent domains "
                            "or new subtopics inside the same domain to search for new equilibria."
                        ),
                        "suggested_strategies": [
                            "new_subgoals_same_domain",
                            "adjacent_domain_goals",
                        ],
                    }
                )

        # If meta learning shows other domains with high avg RYE, suggest cross domain copy.
        if domains_meta:
            best_other: Optional[Tuple[str, float]] = None
            for dom, stats in domains_meta.items():
                if dom == domain:
                    continue
                avg = stats.get("avg_rye")
                if isinstance(avg, (int, float)):
                    if best_other is None or avg > best_other[1]:
                        best_other = (dom, avg)
            if best_other is not None:
                recs.append(
                    {
                        "type": "cross_domain_copy",
                        "description": (
                            "Another domain shows stronger RYE performance. Borrow its best goals or "
                            "equilibrium configurations and adapt them into the current domain."
                        ),
                        "source_domain": best_other[0],
                        "source_avg_rye": best_other[1],
                    }
                )

        # Use MSIL if present to adjust difficulty
        difficulty = "normal"
        if isinstance(msil_score, (int, float)):
            if msil_score >= 0.8:
                difficulty = "advanced"
            elif msil_score <= 0.4:
                difficulty = "foundational"

        plan["difficulty"] = difficulty
        plan["recommendations"] = recs

        # Priority tags
        priority: List[str] = []
        if difficulty == "foundational":
            priority.append("stabilize_basics")
        if difficulty == "advanced":
            priority.append("deep_theory_and_long_runs")
        if trend == "down":
            priority.append("repair_and_recovery")
        if trend == "up" and stability_trend == "improving":
            priority.append("lock_in_equilibria")

        plan["priority"] = priority
        return plan

    def _map_swarm_topology(
        self,
        history: List[Dict[str, Any]],
        stability_summary: StabilitySummary,
    ) -> Dict[str, Any]:
        """Infer simple swarm specialization and synergy patterns from history.

        This does not require additional metadata besides role labels if
        they are present in cycle entries.
        """
        topology: Dict[str, Any] = {
            "supported": False,
            "roles": {},
            "synergy_pairs": [],
        }

        if not history:
            return topology

        # Role level stats
        role_stats: Dict[str, Dict[str, Any]] = {}
        for entry in history:
            role = entry.get("role") or "agent"
            rye_val = entry.get("rye_value")
            eff_val = entry.get("efficiency") or entry.get("rye_efficiency")

            rs = role_stats.setdefault(
                role,
                {
                    "count": 0,
                    "rye_values": [],
                    "efficiency_values": [],
                },
            )
            rs["count"] += 1
            if isinstance(rye_val, (int, float)):
                rs["rye_values"].append(rye_val)
            if isinstance(eff_val, (int, float)):
                rs["efficiency_values"].append(eff_val)

        if not role_stats:
            return topology

        topology["supported"] = True
        for role, rs in role_stats.items():
            rs["avg_rye"] = _safe_mean(rs.get("rye_values", []))
            rs["avg_efficiency"] = _safe_mean(rs.get("efficiency_values", []))

        topology["roles"] = role_stats

        # Simple synergy heuristic:
        # if two roles frequently appear in adjacent cycles and both have above median RYE, mark as synergy line.
        sorted_history = sorted(
            history,
            key=lambda e: e.get("cycle_index") or 0,
        )
        pair_counts: Dict[Tuple[str, str], int] = {}

        for i in range(len(sorted_history) - 1):
            r1 = sorted_history[i].get("role") or "agent"
            r2 = sorted_history[i + 1].get("role") or "agent"
            if r1 == r2:
                continue
            pair = tuple(sorted((r1, r2)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Compute median RYE across roles
        all_avg_rye: List[float] = []
        for rs in role_stats.values():
            v = rs.get("avg_rye")
            if isinstance(v, (int, float)):
                all_avg_rye.append(v)
        try:
            median_role_rye = float(statistics.median(all_avg_rye)) if all_avg_rye else None
        except Exception:
            median_role_rye = None

        synergy_pairs: List[Dict[str, Any]] = []
        for (r1, r2), count in pair_counts.items():
            avg1 = role_stats.get(r1, {}).get("avg_rye")
            avg2 = role_stats.get(r2, {}).get("avg_rye")
            if not isinstance(avg1, (int, float)) or not isinstance(avg2, (int, float)):
                continue
            if median_role_rye is not None and avg1 >= median_role_rye and avg2 >= median_role_rye:
                synergy_pairs.append(
                    {
                        "roles": [r1, r2],
                        "adjacency_count": count,
                        "avg_rye": (avg1 + avg2) / 2.0,
                    }
                )

        synergy_pairs_sorted = sorted(
            synergy_pairs,
            key=lambda x: x.get("adjacency_count") or 0,
            reverse=True,
        )
        topology["synergy_pairs"] = synergy_pairs_sorted[:10]
        return topology

    def _classify_meta_stability_strata(
        self,
        stability_summary: StabilitySummary,
        hours_run: Optional[float],
        total_cycles: int,
    ) -> Dict[str, Any]:
        """Assign a coarse meta stability stratum S0 to S5.

        Uses stability index, oscillation, and run length as inputs.
        """
        diagnostics = stability_summary.diagnostics or {}
        stability_index = diagnostics.get("stability_index")
        osc_std = diagnostics.get("oscillation_std")

        # Defaults
        stratum = "S0"
        label = "unknown"
        description = "Not enough data to classify meta stability."
        evidence: Dict[str, Any] = {
            "stability_index": stability_index,
            "oscillation_std": osc_std,
            "hours_run": hours_run,
            "total_cycles": total_cycles,
        }

        if not isinstance(stability_index, (int, float)):
            return {
                "stratum": stratum,
                "label": label,
                "description": description,
                "evidence": evidence,
            }

        if stability_index < 0.3:
            stratum = "S1"
            label = "fragile"
            description = (
                "System shows low stability index. Expect high variance and frequent oscillations."
            )
        elif stability_index < 0.5:
            stratum = "S2"
            label = "volatile"
            description = (
                "System has moderate instability. RYE may fluctuate and equilibria are not yet consolidated."
            )
        elif stability_index < 0.65:
            stratum = "S3"
            label = "working"
            description = (
                "System is in a working stability zone. RYE is likely usable and can support longer runs."
            )
        elif stability_index < 0.8:
            stratum = "S4"
            label = "robust"
            description = (
                "System shows robust stability and is a good candidate for 1 week to 90 day runs."
            )
        else:
            stratum = "S5"
            label = "high_robust"
            description = (
                "System exhibits very strong stability. This is a high quality regime for long horizon experiments."
            )

        # Slight adjustments based on oscillation
        if isinstance(osc_std, (int, float)):
            if osc_std > 0.4 and stratum in {"S3", "S4", "S5"}:
                description += " Oscillation is still high, so monitor for overcorrection patterns."
            elif osc_std < 0.15 and stratum in {"S3", "S4", "S5"}:
                description += " Oscillation is low which suggests a well controlled equilibrium window."

        return {
            "stratum": stratum,
            "label": label,
            "description": description,
            "evidence": evidence,
        }

    def _compute_breakthrough_probability(
        self,
        stability_summary: StabilitySummary,
        msil_profile: Optional[Dict[str, Any]],
        meta_learning: Dict[str, Any],
        tier_hints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute a soft breakthrough probability score between 0 and 1.

        This is an interpretable analytic blend of:
            - RYE level and slope
            - Stability index
            - Recovery momentum
            - MSIL score if present
            - Global performance from meta learning
            - Tier thresholds from presets
        """
        diagnostics = stability_summary.diagnostics or {}
        avg_rye = diagnostics.get("rolling_rye")
        slope = diagnostics.get("regression_rye_slope")
        stability_index = diagnostics.get("stability_index")
        recovery_momentum = diagnostics.get("recovery_momentum")

        msil_score = None
        if msil_profile and isinstance(msil_profile, dict):
            msil_score = msil_profile.get("msil_score")

        global_mean_rye = meta_learning.get("global_rye_mean")

        def clamp01(x: float) -> float:
            return float(max(0.0, min(1.0, x)))

        # Normalize components to [0, 1] where possible
        avg_component = 0.0
        if isinstance(avg_rye, (int, float)):
            # Assume 0.0 to 0.25 is the meaningful slice
            avg_component = clamp01(avg_rye / 0.25)

        slope_component = 0.5
        if isinstance(slope, (int, float)):
            # Treat slopes in [-0.01, 0.01] as main region
            slope_component = clamp01(0.5 + 50.0 * slope)

        stability_component = 0.0
        if isinstance(stability_index, (int, float)):
            stability_component = clamp01(stability_index)

        recovery_component = 0.5
        if isinstance(recovery_momentum, (int, float)):
            recovery_component = clamp01(0.5 + 4.0 * recovery_momentum)

        msil_component = None
        if isinstance(msil_score, (int, float)):
            msil_component = clamp01(msil_score)

        global_component = 0.5
        if isinstance(global_mean_rye, (int, float)):
            global_component = clamp01(global_mean_rye / 0.25)

        # Weighted blend
        weights = {
            "avg": 0.30,
            "slope": 0.15,
            "stability": 0.25,
            "recovery": 0.10,
            "global": 0.10,
            "msil": 0.10 if msil_component is not None else 0.0,
        }
        weight_sum = sum(weights.values()) or 1.0

        base_score = (
            avg_component * weights["avg"]
            + slope_component * weights["slope"]
            + stability_component * weights["stability"]
            + recovery_component * weights["recovery"]
            + global_component * weights["global"]
            + (msil_component or 0.0) * weights["msil"]
        ) / weight_sum

        # Map base score into tier probabilities with soft boundaries
        tier1_thresholds = tier_hints.get("tier1_thresholds", {}) if tier_hints else {}
        tier2_thresholds = tier_hints.get("tier2_thresholds", {}) if tier_hints else {}
        tier3_thresholds = tier_hints.get("tier3_thresholds", {}) if tier_hints else {}

        # Use avg_rye relative to thresholds as an additional scaling factor.
        def avg_to_tier_boost(th: Dict[str, Any]) -> float:
            if not isinstance(avg_rye, (int, float)):
                return 0.0
            min_avg = th.get("min_avg_rye")
            if not isinstance(min_avg, (int, float)):
                return 0.0
            if avg_rye <= 0:
                return 0.0
            return clamp01(avg_rye / (min_avg * 1.5))

        t1_boost = avg_to_tier_boost(tier1_thresholds)
        t2_boost = avg_to_tier_boost(tier2_thresholds)
        t3_boost = avg_to_tier_boost(tier3_thresholds)

        # Final tier probabilities
        base = clamp01(base_score)
        # Heuristic: tier1 is easier, tier3 is rare.
        p_t1 = clamp01(base * 0.7 + t1_boost * 0.3)
        p_t2 = clamp01(base * 0.45 + t2_boost * 0.55)
        p_t3 = clamp01(base * 0.25 + t3_boost * 0.75)

        # Normalize so that p_t1 >= p_t2 >= p_t3 logically.
        p_t1 = max(p_t1, p_t2, p_t3)
        p_t2 = min(p_t1, max(p_t2, p_t3))
        p_t3 = min(p_t2, p_t3)

        qualitative = "low"
        if base >= 0.7:
            qualitative = "high"
        elif base >= 0.4:
            qualitative = "medium"

        return {
            "score": base,
            "qualitative": qualitative,
            "components": {
                "avg_component": avg_component,
                "slope_component": slope_component,
                "stability_component": stability_component,
                "recovery_component": recovery_component,
                "msil_component": msil_component,
                "global_component": global_component,
            },
            "tier_probabilities": {
                "tier1": p_t1,
                "tier2": p_t2,
                "tier3": p_t3,
            },
        }

    def _summarize_for_ui(
        self,
        stability_summary: StabilitySummary,
        trajectory_forecast: Dict[str, Any],
        meta_stability: Dict[str, Any],
        breakthrough_probability: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Lightweight UI summary card for dashboards."""
        diagnostics = stability_summary.diagnostics or {}
        score = breakthrough_probability.get("score")
        tiers = breakthrough_probability.get("tier_probabilities", {})
        tier1 = tiers.get("tier1")
        tier2 = tiers.get("tier2")
        tier3 = tiers.get("tier3")

        return {
            "stability_index": diagnostics.get("stability_index"),
            "rolling_rye": diagnostics.get("rolling_rye"),
            "trend": trajectory_forecast.get("trend"),
            "stability_trend": trajectory_forecast.get("stability_trend"),
            "meta_stability_stratum": meta_stability.get("stratum"),
            "meta_stability_label": meta_stability.get("label"),
            "breakthrough_score": score,
            "breakthrough_tiers": {
                "tier1": tier1,
                "tier2": tier2,
                "tier3": tier3,
            },
        } 
