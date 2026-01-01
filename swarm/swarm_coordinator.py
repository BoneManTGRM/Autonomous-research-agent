"""Swarm coordinator for the Autonomous Research Agent.

This module defines a SwarmCoordinator that:
    * Manages a group of CoreAgent based workers with different roles.
    * Schedules cycles across agents using RYE aware load balancing.
    * Tracks swarm level metrics and stability.
    * Streams short_view and meta signals for dashboards or logs.
    * Feeds a ReplayBuffer for long run learning.
    * Talks to CurriculumController and HallmarkProfiles when available.

Design goals:
    * Safe defaults if replay, curriculum, or hallmark modules are not present.
    * Stateless interface at the top level:
        run_swarm(...) returns a full machine log and a compact summary.
    * Friendly to 1 to 32 agent swarms and 1 hour to 90 day runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
import re
from typing import Any, Callable, Iterable, Dict, List, Optional, Tuple

from agent.core_agent import CoreAgent
from agent.rye_metrics import (
    rolling_rye,
    stability_index,
    recovery_momentum,
    rye_percentiles,
)

# Optional helpers. These are nice to have but not required.
try:
    from agent.replay_buffer import ReplayBuffer  # type: ignore[attr-defined]
except Exception:
    ReplayBuffer = None  # type: ignore[assignment]

try:
    from agent.curriculum import CurriculumController  # type: ignore[attr-defined]
except Exception:
    CurriculumController = None  # type: ignore[assignment]

try:
    from agent.hallmark_profiles import HallmarkProfiles  # type: ignore[attr-defined]
except Exception:
    HallmarkProfiles = None  # type: ignore[assignment]


@dataclass
class SwarmAgentConfig:
    """Configuration for a single swarm agent slot."""
    agent_id: str
    role: str = "agent"
    preset_name: Optional[str] = None
    domain: str = "general"
    weight: float = 1.0
    max_cycles: Optional[int] = None
    source_controls: Optional[Dict[str, bool]] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmRunConfig:
    """Top level configuration for a swarm run."""

    # Required
    goal: str
    total_cycles: int

    # Identity (propagated to per-cycle logs/events when available)
    run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Funnel controls (pressure-driven discovery)
    # ------------------------------------------------------------------
    # None => auto-enable for multi-cycle runs
    funnel_mode: Optional[bool] = None
    # Cull stage target: eliminate this fraction of candidates
    funnel_kill_quota: float = 0.80
    # Minimum distinct domain lenses to activate in MAP/CLUSTER
    funnel_min_domains: int = 3
    # Novelty floor used for directives (0-1). Higher => stricter novelty demands.
    funnel_novelty_floor: float = 0.35
    # Early stop guardrail when the run drifts
    funnel_early_stop: bool = True
    funnel_stability_floor: float = 0.30
    funnel_negative_slope_patience: int = 2
    # Reduce cost/noise: cap agents during COMMIT (0 => no cap)
    funnel_commit_max_agents: int = 8
    # Optional explicit domain lenses (overrides defaults)
    funnel_domain_lenses: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Scheduling + curriculum controls
    # ------------------------------------------------------------------
    # ``max_parallel`` controls how many agents run concurrently.  A value of
    # 0 or a negative number means "unlimited": all agents run in each
    # scheduler tick.  The engine will then derive an appropriate parallelism
    # based on the swarm size.  See run_swarm() for details.
    max_parallel: int = 0
    min_cycles_per_agent: int = 1
    curriculum_profile: Optional[str] = None
    hallmark_targets: Optional[List[str]] = None
    two_stage_mode: bool = False
    idea_fraction: float = 0.6
    replay_enabled: bool = True
    seed: Optional[int] = None


@dataclass
class AgentSwarmStats:
    """Per agent learning speed and stability snapshot."""
    cycles: int = 0
    total_delta_r: float = 0.0
    total_energy: float = 0.0
    last_rye: Optional[float] = None
    rye_history: List[float] = field(default_factory=list)

    def record_cycle(self, delta_r: float, energy: float, rye_value: Optional[float]) -> None:
        self.cycles += 1
        self.total_delta_r += float(delta_r)
        self.total_energy += float(energy)
        if rye_value is not None:
            v = float(rye_value)
            self.last_rye = v
            self.rye_history.append(v)

    def avg_rye(self) -> Optional[float]:
        if not self.rye_history:
            return None
        return sum(self.rye_history) / float(len(self.rye_history))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycles": self.cycles,
            "total_delta_r": self.total_delta_r,
            "total_energy": self.total_energy,
            "last_rye": self.last_rye,
            "avg_rye": self.avg_rye(),
            "rye_history": list(self.rye_history),
        }


class SwarmCoordinator:
    """Coordinate a swarm of CoreAgent instances for a single goal.

    Usage pattern:

        coordinator = SwarmCoordinator(memory_store, base_agent_config)
        coordinator.configure_swarm(agent_configs)
        result = coordinator.run_swarm(run_config, stream_callback=...)

    The coordinator focuses on:
        * Picking which agent runs each cycle.
        * Passing down domain, role, and source_controls.
        * Collecting RYE and stability signals.
        * Feeding replay buffer items when available.
        * Producing a swarm level run_summary and machine log.
    """

    def __init__(
        self,
        memory_store: Any,
        base_agent_config: Optional[Dict[str, Any]] = None,
        *,
        replay_buffer: Optional[Any] = None,
        curriculum: Optional[Any] = None,
        hallmark_profiles: Optional[Any] = None,
        core_agent_factory: Optional[Callable[[Dict[str, Any]], CoreAgent]] = None,
        rng: Optional[Callable[[], float]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.base_agent_config = base_agent_config or {}
        self.replay_buffer = replay_buffer
        self.curriculum = curriculum
        self.hallmark_profiles = hallmark_profiles
        self.core_agent_factory = core_agent_factory
        self.rng = rng or self._default_rng

        self.agent_configs: Dict[str, SwarmAgentConfig] = {}
        self.agents: Dict[str, CoreAgent] = {}
        self.agent_stats: Dict[str, AgentSwarmStats] = {}

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------
    def configure_swarm(self, agent_configs: List[SwarmAgentConfig]) -> None:
        """Register or update the agents that participate in the swarm."""
        self.agent_configs.clear()
        self.agents.clear()
        self.agent_stats.clear()

        for cfg in agent_configs:
            self.agent_configs[cfg.agent_id] = cfg
            self.agent_stats[cfg.agent_id] = AgentSwarmStats()

    # ------------------------------------------------------------------
    # Main swarm loop
    # ------------------------------------------------------------------

    def run_swarm(
        self,
        run_config: SwarmRunConfig,
        *,
        stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run a full swarm session for a single goal.

        Returns:
            {
              "goal": ...,
              "swarm_config": ...,
              "agent_summaries": {agent_id: {...}},
              "swarm_metrics": {...},
              "cycle_logs": [...],
            }
        """

        goal = str(run_config.goal)
        total_cycles = max(1, int(run_config.total_cycles))

        # Auto-enable the funnel for multi-cycle runs unless explicitly disabled.
        funnel_mode = self._auto_funnel_enabled(run_config, total_cycles)

        agent_count = len(self.agent_configs) if self.agent_configs else 0
        if agent_count <= 0:
            return {
                "goal": goal,
                "swarm_config": {"total_cycles": total_cycles, "max_parallel": 0},
                "agent_summaries": {},
                "swarm_metrics": {},
                "cycle_logs": [],
            }

        # Determine the parallel cap.  A value of <= 0 means "no cap" (run all agents).
        try:
            requested_parallel = int(run_config.max_parallel)
        except Exception:
            requested_parallel = 0
        if requested_parallel and requested_parallel > 0:
            max_parallel = requested_parallel
        else:
            max_parallel = agent_count
        max_parallel = max(1, min(max_parallel, agent_count))

        hallmark_targets = self._resolve_hallmark_targets(run_config.hallmark_targets)
        curriculum_profile = run_config.curriculum_profile
        idea_fraction = float(run_config.idea_fraction or 0.6)

        # Run-level logs/metrics
        cycle_logs: List[Dict[str, Any]] = []
        swarm_level_rye: List[float] = []
        swarm_level_delta_r: List[float] = []
        swarm_level_energy: List[float] = []

        # Funnel guardrails
        cycle_rye_means: List[float] = []
        down_streak = 0
        early_stop: Optional[Dict[str, Any]] = None

        # Context helpers used in directives
        is_longevity = self._is_longevity_context(goal, hallmark_targets)
        domain_lenses = self._resolve_domain_lenses(run_config, is_longevity)

        # Funnel state is a light-weight baton passed between stages. It is populated
        # opportunistically from structured agent outputs when available.
        funnel_state: Dict[str, Any] = {
            "candidates": [],
            "clusters": [],
            "survivors": [],
            "stress_survivors": [],
            "filtered": [],
        }


        for global_cycle in range(total_cycles):
            # Determine funnel stage and the legacy idea/verify stage tag.
            funnel_stage: Optional[str] = None
            if funnel_mode:
                funnel_stage = self._funnel_stage_for_cycle(global_cycle, total_cycles)
                stage = "idea" if funnel_stage in ("map", "cluster") else "verify"
            else:
                stage = self._stage_for_cycle(global_cycle, total_cycles, run_config)

            # (1) Select which agents run this global cycle
            eligible_agents = self._eligible_agents(global_cycle, run_config)
            if not eligible_agents:
                continue

            max_parallel_this_cycle = max_parallel
            if funnel_mode and funnel_stage == "commit":
                try:
                    cap = int(run_config.funnel_commit_max_agents or 0)
                except Exception:
                    cap = 0
                if cap and cap > 0:
                    max_parallel_this_cycle = max(1, min(max_parallel_this_cycle, cap, len(eligible_agents)))

            scheduled_agents = self._pick_agents_for_cycle(eligible_agents, max_parallel_this_cycle)

            # Ask curriculum (if present) what to focus on at this stage
            curriculum_state = self._query_curriculum(
                goal=goal,
                global_cycle=global_cycle,
                total_cycles=total_cycles,
                stage=stage,
                hallmark_targets=hallmark_targets,
                curriculum_profile=curriculum_profile,
            )

            # Optional hallmark selection
            hallmark_name, hallmark_subgoal = self._pick_hallmark_and_subgoal(
                hallmark_targets,
                global_cycle,
                curriculum_state,
            )

            funnel_directive: Optional[str] = None
            if funnel_mode and funnel_stage is not None:
                funnel_directive = self._funnel_directive_for_stage(
                    funnel_stage,
                    goal=goal,
                    is_longevity=is_longevity,
                    run_config=run_config,
                    domain_lenses=domain_lenses,
                    state=funnel_state,
                )

            # Merge hallmark subgoal with funnel directive (without changing the base goal).
            subgoal = self._merge_subgoals(hallmark_subgoal, funnel_directive)

            # Per-cycle aggregates for guardrails
            cycle_rye_vals: List[float] = []

            # (2) Run each scheduled agent for one cycle
            for agent_id in scheduled_agents:
                cfg = self.agent_configs[agent_id]
                agent = self._ensure_agent_instance(agent_id, cfg)

                cycle_index = self.agent_stats[agent_id].cycles
                base_domain = cfg.domain or "general"
                source_controls = cfg.source_controls

                role_for_call = cfg.role or "agent"
                domain_for_call = base_domain

                if funnel_mode and funnel_stage is not None:
                    role_for_call = self._role_for_funnel_stage(funnel_stage, role_for_call)
                    if funnel_stage in ("map", "cluster") and domain_lenses:
                        lens_idx = self._stable_index(f"{agent_id}:{global_cycle}:{funnel_stage}", len(domain_lenses))
                        lens = domain_lenses[lens_idx]
                        domain_for_call = self._compose_domain(base_domain, lens)

                extra_kwargs: Dict[str, Any] = {}

                # Always pass stage when funnel is active; otherwise only when two-stage mode is enabled.
                if funnel_mode or run_config.two_stage_mode:
                    extra_kwargs["stage"] = stage
                if hallmark_name is not None:
                    extra_kwargs["hallmark_target"] = hallmark_name
                if subgoal is not None:
                    extra_kwargs["subgoal"] = subgoal
                if curriculum_state is not None:
                    extra_kwargs["curriculum_state"] = curriculum_state

                # Propagate run_id when available (used by event/report queries).
                if run_config.run_id is not None:
                    extra_kwargs["run_id"] = str(run_config.run_id)

                # Call CoreAgent with best-effort compatibility: if it doesn't accept newer
                # kwargs, drop them incrementally (preserving funnel-critical fields when possible).
                base_call_kwargs = dict(
                    goal=goal,
                    cycle_index=cycle_index,
                    role=role_for_call,
                    source_controls=source_controls,
                    domain=domain_for_call,
                )

                try:
                    result = agent.run_cycle(**base_call_kwargs, **extra_kwargs)
                except TypeError:
                    retry_extras = dict(extra_kwargs)
                    for drop_key in ("run_id", "curriculum_state", "hallmark_target", "subgoal", "stage"):
                        if drop_key in retry_extras:
                            retry_extras.pop(drop_key, None)
                            try:
                                result = agent.run_cycle(**base_call_kwargs, **retry_extras)
                                break
                            except TypeError:
                                continue
                    else:
                        result = agent.run_cycle(**base_call_kwargs)

                summary = result.get("summary", {}) or {}
                log_entry = result.get("log", {}) or {}

                # Feed structured outputs (when present) into a baton that the next stage can use.
                if funnel_mode and funnel_stage is not None:
                    try:
                        self._update_funnel_state(
                            funnel_state=funnel_state,
                            stage=funnel_stage,
                            summary=summary,
                            log_entry=log_entry,
                            is_longevity=is_longevity,
                            novelty_floor=float(run_config.funnel_novelty_floor or 0.35),
                        )
                    except Exception:
                        pass


                # Extract metrics
                delta_r = float(summary.get("delta_R", 0.0) or 0.0)
                energy_e = float(summary.get("energy_E", 0.0) or 0.0)
                rye_value = summary.get("RYE")

                self.agent_stats[agent_id].record_cycle(delta_r, energy_e, rye_value)

                if rye_value is not None:
                    try:
                        rye_f = float(rye_value)
                        swarm_level_rye.append(rye_f)
                        cycle_rye_vals.append(rye_f)
                    except Exception:
                        pass
                swarm_level_delta_r.append(delta_r)
                swarm_level_energy.append(energy_e)

                # Tag log entry with swarm metadata
                log_entry = dict(log_entry)
                if run_config.run_id is not None:
                    try:
                        log_entry.setdefault("run_id", str(run_config.run_id))
                    except Exception:
                        pass
                try:
                    log_entry.setdefault("cycle", global_cycle)
                except Exception:
                    pass
                log_entry.setdefault("global_cycle", global_cycle)
                log_entry.setdefault("funnel_stage", funnel_stage)
                log_entry.setdefault("stage", stage)
                log_entry.setdefault("role", role_for_call)
                log_entry.setdefault("domain", domain_for_call)
                log_entry.setdefault("base_role", cfg.role)
                log_entry.setdefault("base_domain", base_domain)

                cycle_logs.append(log_entry)

                # Replay buffer logging
                if run_config.replay_enabled:
                    try:
                        self._log_to_replay(
                            agent_id=agent_id,
                            cfg=cfg,
                            summary=summary,
                            log_entry=log_entry,
                            stage=stage,
                            hallmark=hallmark_name,
                            subgoal=subgoal,
                        )
                    except Exception:
                        pass

                # Stream short_view for dashboards
                if stream_callback is not None:
                    short_view = summary.get("short_view") or {}
                    payload = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "agent_id": agent_id,
                        "role": role_for_call,
                        "base_role": cfg.role,
                        "goal": goal,
                        "stage": stage,
                        "funnel_stage": funnel_stage,
                        "hallmark": hallmark_name,
                        "subgoal": subgoal,
                        "short_view": short_view,
                        "meta_signals": summary.get("meta_signals") or {},
                    }
                    stream_callback(payload)

            # -----------------------------
            # Funnel health guardrails
            # -----------------------------
            if funnel_mode and run_config.funnel_early_stop and cycle_rye_vals:
                mean_rye = sum(cycle_rye_vals) / float(len(cycle_rye_vals))
                cycle_rye_means.append(mean_rye)

                if len(cycle_rye_means) >= 2 and cycle_rye_means[-1] < cycle_rye_means[-2]:
                    down_streak += 1
                else:
                    down_streak = 0

                stab = self._simple_stability_index(cycle_rye_means, window=5)
                patience = max(1, int(run_config.funnel_negative_slope_patience or 2))
                stability_floor = float(run_config.funnel_stability_floor or 0.30)

                if down_streak >= patience:
                    early_stop = {
                        "reason": "rye_declining_consecutively",
                        "global_cycle": global_cycle,
                        "down_streak": down_streak,
                        "cycle_rye_means": list(cycle_rye_means),
                        "stability_index": stab,
                    }
                    break

                if len(cycle_rye_means) >= 3 and stab is not None and stab < stability_floor:
                    early_stop = {
                        "reason": "low_stability_after_cycle_3",
                        "global_cycle": global_cycle,
                        "cycle_rye_means": list(cycle_rye_means),
                        "stability_index": stab,
                        "stability_floor": stability_floor,
                    }
                    break

        # If a funnel guardrail triggered, append a lightweight system log entry so
        # downstream code (reports/UI) can surface the stop reason even if the caller
        # only returns cycle logs.
        if early_stop is not None:
            _cycle = early_stop.get("global_cycle")
            try:
                _cycle_i = int(_cycle) if _cycle is not None else None
            except Exception:
                _cycle_i = None
            cycle_logs.append(
                {
                    "run_id": str(run_config.run_id) if run_config.run_id is not None else None,
                    "cycle": _cycle_i,
                    "global_cycle": _cycle_i,
                    "stage": "verify",
                    "funnel_stage": "guardrail",
                    "role": "system",
                    "domain": "system",
                    "text": f"EARLY STOP: {early_stop.get('reason')} (cycle={_cycle_i}, stability={early_stop.get('stability_index')})",
                    "guardrail": early_stop,
                }
            )

        # Build swarm level metrics and summary
        swarm_metrics = self._build_swarm_metrics(
            goal=goal,
            swarm_level_rye=swarm_level_rye,
            swarm_level_delta_r=swarm_level_delta_r,
            swarm_level_energy=swarm_level_energy,
        )

        if funnel_mode:
            swarm_metrics.setdefault("funnel", {})
            swarm_metrics["funnel"].update(
                {
                    "enabled": True,
                    "is_longevity": is_longevity,
                    "domain_lenses": list(domain_lenses),
                    "cycle_rye_means": list(cycle_rye_means),
                }
            )
            if early_stop is not None:
                swarm_metrics["funnel"]["early_stop"] = early_stop

        agent_summaries = {aid: st.to_dict() for aid, st in self.agent_stats.items()}

        return {
            "goal": goal,
            "swarm_config": {
                "total_cycles": total_cycles,
                "max_parallel": max_parallel,
                "hallmark_targets": hallmark_targets,
                "curriculum_profile": curriculum_profile,
                "two_stage_mode": run_config.two_stage_mode,
                "idea_fraction": idea_fraction,
                "funnel_mode": funnel_mode,
                "funnel_commit_max_agents": int(run_config.funnel_commit_max_agents or 0),
            },
            "agent_summaries": agent_summaries,
            "swarm_metrics": swarm_metrics,
            "cycle_logs": cycle_logs,
        }

    def _eligible_agents(self, global_cycle: int, run_config: SwarmRunConfig) -> List[str]:
        """Return agent ids that can run on this global cycle."""
        eligible: List[str] = []
        for agent_id, cfg in self.agent_configs.items():
            stats = self.agent_stats[agent_id]

            if cfg.max_cycles is not None and stats.cycles >= cfg.max_cycles:
                continue

            if stats.cycles < run_config.min_cycles_per_agent:
                # Force at least min_cycles_per_agent for everyone
                eligible.append(agent_id)
                continue

            eligible.append(agent_id)

        return eligible

    def _pick_agents_for_cycle(self, eligible: List[str], max_parallel: int) -> List[str]:
        """RYE aware agent selection.

        Simple strategy:
            * Compute a priority for each eligible agent.
            * Higher priority gets picked first.
            * Priority is a mix of:
                weight, inverse of avg RYE, and a small random jitter.
        """
        if not eligible:
            return []

        scored: List[Tuple[float, str]] = []
        for agent_id in eligible:
            cfg = self.agent_configs[agent_id]
            st = self.agent_stats[agent_id]
            w = max(0.1, float(cfg.weight or 1.0))

            avg_rye = st.avg_rye()
            if avg_rye is None:
                # Prioritize agents with no history to explore configs
                base_priority = 1.5
            else:
                # If avg RYE is low, raise the priority to let it adapt
                base_priority = 1.0 + max(0.0, 0.8 - avg_rye)

            jitter = 0.05 * (self.rng() - 0.5)
            priority = w * base_priority + jitter
            scored.append((priority, agent_id))

        scored.sort(reverse=True, key=lambda x: x[0])
        selected = [aid for _, aid in scored[:max_parallel]]
        return selected

    # ------------------------------------------------------------------
    # Curriculum and hallmark helpers
    # ------------------------------------------------------------------
    def _resolve_hallmark_targets(self, hallmark_targets: Optional[List[str]]) -> List[str]:
        if not hallmark_targets:
            return []
        cleaned = []
        for h in hallmark_targets:
            if not h:
                continue
            h_norm = str(h).strip()
            if not h_norm:
                continue
            cleaned.append(h_norm)
        return cleaned

    def _stage_for_cycle(self, global_cycle: int, total_cycles: int, run_config: SwarmRunConfig) -> str:
        if not run_config.two_stage_mode or total_cycles <= 1:
            return "idea"
        cutoff = int(run_config.idea_fraction * total_cycles)
        if global_cycle < cutoff:
            return "idea"
        return "verify"


    # ------------------------------------------------------------------
    # Funnel helpers (pressure-driven discovery)
    # ------------------------------------------------------------------
    def _auto_funnel_enabled(self, run_config: SwarmRunConfig, total_cycles: int) -> bool:
        """Return whether the funnel should be enabled for this run."""
        if run_config.funnel_mode is not None:
            return bool(run_config.funnel_mode)
        # Default: enable for multi-cycle runs (2+). Users can disable explicitly.
        return bool(total_cycles >= 2)

    def _funnel_stage_for_cycle(self, global_cycle: int, total_cycles: int) -> str:
        """Map a 0-indexed cycle into a funnel stage."""
        if total_cycles <= 1:
            return "map"
        if total_cycles == 2:
            return "map" if global_cycle == 0 else "commit"
        if total_cycles == 3:
            return ["map", "cull", "commit"][min(global_cycle, 2)]
        if total_cycles == 4:
            return ["map", "cluster", "stress", "commit"][min(global_cycle, 3)]

        # 5+ cycles: map -> cluster -> (cull/stress alternating) -> commit
        if global_cycle == 0:
            return "map"
        if global_cycle == 1:
            return "cluster"
        if global_cycle >= total_cycles - 1:
            return "commit"
        # alternate starting with cull on cycle 2
        return "cull" if (global_cycle % 2 == 0) else "stress"

    def _role_for_funnel_stage(self, stage: str, base_role: str) -> str:
        """Assign a pressure-appropriate role label for the stage."""
        role = (base_role or "agent").strip()
        low = role.lower()

        def has_any(*keys: str) -> bool:
            return any(k in low for k in keys)

        if stage == "map":
            return role if has_any("explor", "research", "mapper") else "explorer"
        if stage == "cluster":
            return role if has_any("synth", "integr", "cluster", "planner", "architect") else "synthesizer"
        if stage == "cull":
            return role if has_any("critic", "skeptic", "verif", "falsif", "review") else "critic"
        if stage == "stress":
            return role if has_any("red", "advers", "critic", "verif", "skeptic") else "adversary"
        if stage == "commit":
            return role if has_any("writer", "integr", "synth", "planner") else "integrator"
        return role or "agent"

    def _is_longevity_context(self, goal: str, hallmark_targets: Optional[List[str]] = None) -> bool:
        text = (goal or "").lower()
        if any(k in text for k in ("longevity", "aging", "ageing", "senescence", "geroscience")):
            return True
        if hallmark_targets:
            # Hallmarks are strongly tied to aging/longevity contexts.
            return True
        return False

    def _resolve_domain_lenses(self, run_config: SwarmRunConfig, is_longevity: bool) -> List[str]:
        # User override
        if run_config.funnel_domain_lenses:
            lenses = [str(x).strip() for x in (run_config.funnel_domain_lenses or []) if str(x).strip()]
            if lenses:
                return lenses

        if is_longevity:
            return [
                "transport_physics",
                "systems_immunology",
                "error_correction_clearance",
                "mechanical_stress_tissue_flow",
                "information_theory_control",
            ]
        # Generic multi-lens set
        return [
            "mechanistic_model",
            "systems_dynamics",
            "measurement_validation",
            "engineering_constraints",
            "failure_modes_safety",
        ]

    def _stable_index(self, key: str, modulo: int) -> int:
        """Stable integer hashing for deterministic domain-lens assignment."""
        if modulo <= 0:
            return 0
        try:
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            return int(h[:8], 16) % modulo
        except Exception:
            return abs(hash(key)) % modulo

    def _compose_domain(self, base_domain: str, lens: str) -> str:
        base = (base_domain or "general").strip()
        lens = (lens or "").strip()
        if not lens:
            return base
        if base and base.lower() not in ("general", "default"):
            if lens in base:
                return base
            return f"{base}:{lens}"
        return lens

    def _merge_subgoals(self, base: Optional[str], extra: Optional[str]) -> Optional[str]:
        b = (base or "").strip()
        e = (extra or "").strip()
        if not b and not e:
            return None
        if b and not e:
            return b
        if e and not b:
            return e
        # Avoid duplicates if the extra is already included
        if e.lower() in b.lower():
            return b
        if b.lower() in e.lower():
            return e
        return b + "\n\n" + e

    def _funnel_directive_for_stage(
        self,
        stage: str,
        *,
        goal: str,
        is_longevity: bool,
        run_config: SwarmRunConfig,
        domain_lenses: List[str],
        state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stage-specific mechanical instructions that drive convergence."""

        kill_q = float(run_config.funnel_kill_quota or 0.80)
        min_domains = max(1, int(run_config.funnel_min_domains or 3))
        novelty_floor = float(run_config.funnel_novelty_floor or 0.35)

        # Longevity-specific stress axes
        stress_axes = [
            "cancer_risk",
            "fibrosis_risk",
            "immune_collapse_risk",
            "metabolic_failure_risk",
        ] if is_longevity else [
            "safety_risk",
            "feasibility_risk",
            "scalability_risk",
            "confounders_and_failure_modes",
        ]

        novelty_bans = []
        if is_longevity:
            novelty_bans = [
                "mTOR/rapamycin",
                "metformin/AMPK",
                "NAD+/sirtuins",
                "senolytics",
                "caloric restriction mimetics",
            ]

        lines: List[str] = []
        lines.append(f"Funnel stage: {stage.upper()} (mechanical, pressure-driven).")
        lines.append("Do not be polite. Prefer elimination + compression over exploration.")
        lines.append("Output MUST be explicit and falsifiable (no vague speculation).")

        if stage == "map":
            lines.append("MAP: Generate 20â30 raw candidate mechanisms/constraints. No conclusions allowed.")
            lines.append("Each candidate must include: (a) mechanism in 1 sentence, (b) quick falsifier, (c) 2-domain mix.")
            if domain_lenses:
                lines.append(f"Activate at least {min_domains} domain lenses across the swarm: {', '.join(domain_lenses)}.")
            if novelty_bans:
                lines.append(
                    "Novelty floor: reject ideas that resemble mainstream longevity pathways "
                    f"(examples to avoid: {', '.join(novelty_bans)})."
                )

        elif stage == "cluster":
            lines.append("CLUSTER: Group candidates into 3â5 mechanism families.")
            lines.append("For each family: name it, list members, and state the single core bottleneck.")
            lines.append("Carry forward only the families with clear bottlenecks and falsifiers.")

        elif stage == "cull":
            lines.append("CULL: Eliminate aggressively.")
            lines.append(f"Requirement: reject at least {int(kill_q * 100)}% of candidates.")
            lines.append("Each rejection must include: (a) why it fails, (b) fastest falsification test, (c) what evidence would revive it.")
            lines.append("Output: top 3 survivors with crisp theses (1â2 sentences each).")

        elif stage == "stress":
            lines.append("STRESS: Red-team survivors until they break.")
            lines.append("Force adversarial checks across:")
            for ax in stress_axes:
                lines.append(f"  - {ax}")
            lines.append("Output: 1â2 survivors still standing + one decisive failure mode per survivor.")

        elif stage == "commit":
            lines.append("COMMIT: Convert the best survivor into a single thesis and an action plan.")
            lines.append("Output must include:")
            lines.append("  - 1 core thesis")
            lines.append("  - 3 falsifiable predictions")
            lines.append("  - 2 minimal experiments (fast + cheap if possible)")
            lines.append("  - 1 clear failure mode / stop condition")
            lines.append("No brainstorming. Commit to the most defensible option.")

        else:
            lines.append("Proceed with high-pressure discovery and explicit falsification.")

        # Context baton (best effort): carry forward prior-stage outputs when available.
        if state:
            try:
                if stage == "cluster":
                    cand = state.get("candidates") or []
                    if cand:
                        lines.append("")
                        lines.append("Context baton (from MAP): candidates to cluster:")
                        for item in cand[:40]:
                            lines.append(f"  - {item}")
                elif stage == "cull":
                    pool = state.get("clusters") or state.get("candidates") or []
                    if pool:
                        lines.append("")
                        lines.append("Context baton: pool to cull (clustered families preferred):")
                        for item in pool[:40]:
                            lines.append(f"  - {item}")
                elif stage == "stress":
                    pool = state.get("survivors") or []
                    if not pool:
                        pool = (state.get("candidates") or [])[:10]
                    if pool:
                        lines.append("")
                        lines.append("Context baton: survivors to stress-test (fallback to top candidates):")
                        for item in pool[:10]:
                            lines.append(f"  - {item}")
                elif stage == "commit":
                    pool = state.get("stress_survivors") or state.get("survivors") or []
                    if pool:
                        lines.append("")
                        lines.append("Context baton: option(s) to commit (pick the most defensible):")
                        for item in pool[:5]:
                            lines.append(f"  - {item}")
            except Exception:
                pass

        lines.append(f"Novelty floor target: >= {novelty_floor:.2f} (qualitative).")
        return "\n".join(lines)

    def _simple_stability_index(self, values: List[float], window: int = 5) -> Optional[float]:
        """A lightweight [0,1] stability proxy based on relative variability."""
        if not values:
            return None
        n = min(len(values), max(2, int(window)))
        recent = values[-n:]
        mean = sum(recent) / float(n)
        var = sum((v - mean) ** 2 for v in recent) / float(n)
        std = math.sqrt(var)
        rsd = std / (abs(mean) + 1e-9)
        stab = 1.0 - rsd
        if stab < 0.0:
            stab = 0.0
        if stab > 1.0:
            stab = 1.0
        return stab


    # ------------------------------------------------------------------
    # Funnel baton extraction (best-effort parsing of structured outputs)
    # ------------------------------------------------------------------
    def _extract_strings(self, value: Any, *, limit: int = 200) -> List[str]:
        """Best-effort extraction of strings from nested JSON-ish values."""
        out: List[str] = []

        def _rec(v: Any) -> None:
            if len(out) >= limit:
                return
            if v is None:
                return
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
                return
            if isinstance(v, (int, float, bool)):
                return
            if isinstance(v, dict):
                # Prefer common "text" fields if present
                for k in ("text", "thesis", "hypothesis", "candidate", "name"):
                    if k in v and isinstance(v.get(k), str):
                        s = str(v.get(k)).strip()
                        if s:
                            out.append(s)
                            if len(out) >= limit:
                                return
                for vv in v.values():
                    _rec(vv)
                return
            if isinstance(v, list):
                for item in v:
                    _rec(item)
                return

        _rec(value)
        return out[:limit]

    def _collect_by_keys(self, obj: Any, keys: Iterable[str], *, max_depth: int = 3) -> List[str]:
        """Collect strings for any matching key in a nested mapping."""
        wanted = {str(k).lower() for k in keys}
        out: List[str] = []

        def _rec(v: Any, depth: int) -> None:
            if depth < 0:
                return
            if isinstance(v, dict):
                for k, vv in v.items():
                    kk = str(k).lower()
                    if kk in wanted:
                        out.extend(self._extract_strings(vv, limit=200))
                    else:
                        _rec(vv, depth - 1)
                return
            if isinstance(v, list):
                for item in v:
                    _rec(item, depth - 1)

        _rec(obj, max_depth)
        return out

    def _novelty_score(self, text: str, *, is_longevity: bool) -> float:
        """Heuristic novelty score in [0,1]. This is intentionally simple."""
        t = (text or "").lower()
        score = 1.0

        if len(t.strip()) < 30:
            score -= 0.20

        # Penalize mainstream longevity pathways
        if is_longevity:
            mainstream = [
                "rapamycin",
                "mtor",
                "metformin",
                "ampk",
                "nad",
                "sirtuin",
                "senolytic",
                "dasatinib",
                "quercetin",
                "caloric restriction",
                "cr mimetic",
            ]
            if any(k in t for k in mainstream):
                score -= 0.70

        # Penalize generic lifestyle advice (low mechanism novelty)
        generic = ["exercise", "diet", "sleep", "meditation", "fasting"]
        if any(k in t for k in generic):
            score -= 0.25

        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return score

    def _clean_item(self, s: str) -> str:
        s2 = (s or "").strip()
        # Strip common list/bullet prefixes
        s2 = re.sub(r"^[-*â¢\s]+", "", s2)
        s2 = re.sub(r"^\(?\d+\)?[\.)]\s+", "", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    def _update_funnel_state(
        self,
        *,
        funnel_state: Dict[str, Any],
        stage: str,
        summary: Dict[str, Any],
        log_entry: Dict[str, Any],
        is_longevity: bool,
        novelty_floor: float,
    ) -> None:
        """Update the baton with any structured candidates/survivors found."""

        # Candidate-like keys we might see in structured outputs
        candidate_keys = [
            "candidates",
            "candidate_hypotheses",
            "hypotheses",
            "ideas",
            "proposals",
            "candidate_interventions",
        ]
        cluster_keys = ["clusters", "families", "groups", "mechanism_families"]
        survivor_keys = ["survivors", "top_survivors", "kept", "selected", "finalists"]

        # Pull candidates opportunistically from summary/log
        candidates = []
        candidates.extend(self._collect_by_keys(summary, candidate_keys, max_depth=3))
        candidates.extend(self._collect_by_keys(log_entry, candidate_keys, max_depth=3))

        clusters = []
        clusters.extend(self._collect_by_keys(summary, cluster_keys, max_depth=3))
        clusters.extend(self._collect_by_keys(log_entry, cluster_keys, max_depth=3))

        survivors = []
        survivors.extend(self._collect_by_keys(summary, survivor_keys, max_depth=3))
        survivors.extend(self._collect_by_keys(log_entry, survivor_keys, max_depth=3))

        def _norm(x: str) -> str:
            return re.sub(r"\s+", " ", (x or "").strip().lower())

        # Ensure internal containers exist
        for k in ("candidates", "clusters", "survivors", "stress_survivors", "filtered"):
            if k not in funnel_state or not isinstance(funnel_state.get(k), list):
                funnel_state[k] = []

        # Update candidates on MAP/CLUSTER primarily
        if stage in ("map", "cluster") and candidates:
            seen = {_norm(x) for x in funnel_state["candidates"] if isinstance(x, str)}
            for raw in candidates:
                item = self._clean_item(raw)
                if not item:
                    continue
                key = _norm(item)
                if key in seen:
                    continue
                score = self._novelty_score(item, is_longevity=is_longevity)
                if score < float(novelty_floor):
                    # keep a breadcrumb for debugging
                    if len(funnel_state["filtered"]) < 120:
                        funnel_state["filtered"].append({"text": item, "score": score})
                    continue
                funnel_state["candidates"].append(item)
                seen.add(key)
                if len(funnel_state["candidates"]) >= 220:
                    break

        # Update clusters on CLUSTER
        if stage == "cluster" and clusters:
            seen = {_norm(x) for x in funnel_state["clusters"] if isinstance(x, str)}
            for raw in clusters:
                item = self._clean_item(raw)
                if not item:
                    continue
                key = _norm(item)
                if key in seen:
                    continue
                funnel_state["clusters"].append(item)
                seen.add(key)
                if len(funnel_state["clusters"]) >= 120:
                    break

        # Update survivors on CULL
        if stage == "cull" and survivors:
            seen = {_norm(x) for x in funnel_state["survivors"] if isinstance(x, str)}
            for raw in survivors:
                item = self._clean_item(raw)
                if not item:
                    continue
                key = _norm(item)
                if key in seen:
                    continue
                funnel_state["survivors"].append(item)
                seen.add(key)
                if len(funnel_state["survivors"]) >= 30:
                    break

        # Update stress survivors on STRESS
        if stage == "stress" and survivors:
            seen = {_norm(x) for x in funnel_state["stress_survivors"] if isinstance(x, str)}
            for raw in survivors:
                item = self._clean_item(raw)
                if not item:
                    continue
                key = _norm(item)
                if key in seen:
                    continue
                funnel_state["stress_survivors"].append(item)
                seen.add(key)
                if len(funnel_state["stress_survivors"]) >= 15:
                    break

    def _query_curriculum(
        self,
        *,
        goal: str,
        global_cycle: int,
        total_cycles: int,
        stage: str,
        hallmark_targets: List[str],
        curriculum_profile: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Ask CurriculumController (if present) how to focus this cycle."""
        if self.curriculum is None or CurriculumController is None:
            return None

        controller = self.curriculum
        if isinstance(controller, type):
            # Caller passed the class instead of an instance
            controller = controller()

        try:
            if hasattr(controller, "select_segment"):
                return controller.select_segment(
                    goal=goal,
                    global_cycle=global_cycle,
                    total_cycles=total_cycles,
                    stage=stage,
                    hallmark_targets=hallmark_targets,
                    curriculum_profile=curriculum_profile,
                )
            if hasattr(controller, "next_state"):
                return controller.next_state(
                    goal=goal,
                    cycle_index=global_cycle,
                    stage=stage,
                    hallmark_targets=hallmark_targets,
                    profile=curriculum_profile,
                )
        except Exception:
            return None

        return None

    def _pick_hallmark_and_subgoal(
        self,
        hallmark_targets: List[str],
        global_cycle: int,
        curriculum_state: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Decide which hallmark and subgoal this cycle should focus on."""
        if not hallmark_targets:
            return None, None

        # Curriculum can override the hallmark choice
        if curriculum_state is not None:
            h = curriculum_state.get("hallmark")
            sub = curriculum_state.get("subgoal")
            if h:
                return str(h), str(sub) if sub else None

        # Simple round robin across targets
        idx = global_cycle % len(hallmark_targets)
        hallmark = hallmark_targets[idx]

        subgoal = None
        if self.hallmark_profiles is not None and HallmarkProfiles is not None:
            try:
                hp = self.hallmark_profiles
                if isinstance(hp, type):
                    hp = hp()
                if hasattr(hp, "pick_subgoal_for_cycle"):
                    subgoal_info = hp.pick_subgoal_for_cycle(
                        hallmark,
                        cycle_index=global_cycle,
                    )
                    if isinstance(subgoal_info, dict):
                        subgoal = subgoal_info.get("subgoal_name") or subgoal_info.get("subgoal")
            except Exception:
                subgoal = None

        return hallmark, subgoal

    # ------------------------------------------------------------------
    # Replay logging
    # ------------------------------------------------------------------
    def _log_to_replay(
        self,
        *,
        agent_id: str,
        cfg: SwarmAgentConfig,
        summary: Dict[str, Any],
        log_entry: Dict[str, Any],
        stage: str,
        hallmark: Optional[str],
        subgoal: Optional[str],
    ) -> None:
        """Record high RYE moves into the replay buffer when available."""
        if self.replay_buffer is None or ReplayBuffer is None:
            return

        rye_value = summary.get("RYE")
        if rye_value is None:
            return

        try:
            rye_float = float(rye_value)
        except Exception:
            return

        # Only log cycles that reach a minimum efficiency
        if rye_float < 0.3:
            return

        hypotheses = summary.get("hypotheses") or []
        candidate_interventions = summary.get("candidate_interventions") or []
        citations = summary.get("citations") or []
        breakthrough = summary.get("breakthrough") or {}

        payload = {
            "agent_id": agent_id,
            "role": cfg.role,
            "hallmark": hallmark,
            "stage": stage,
            "subgoal": subgoal,
            "goal": summary.get("goal"),
            "cycle": summary.get("cycle"),
            "rye": rye_float,
            "delta_R": summary.get("delta_R"),
            "energy_E": summary.get("energy_E"),
            "hypotheses": hypotheses,
            "candidate_interventions": candidate_interventions,
            "citations": citations,
            "breakthrough_score": breakthrough.get("breakthrough_score"),
            "log_short_view": summary.get("short_view"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        rb = self.replay_buffer
        try:
            if hasattr(rb, "add_from_cycle"):
                rb.add_from_cycle(payload, raw_log=log_entry)
            elif hasattr(rb, "add_item"):
                rb.add_item(payload)
            elif hasattr(rb, "log_item"):
                rb.log_item(payload)
        except Exception:
            # Replay failures cannot break the run
            return

    # ------------------------------------------------------------------
    # Swarm metrics
    # ------------------------------------------------------------------
    def _build_swarm_metrics(
        self,
        *,
        goal: str,
        swarm_level_rye: List[float],
        swarm_level_delta_r: List[float],
        swarm_level_energy: List[float],
    ) -> Dict[str, Any]:
        if swarm_level_rye:
            rye_series = swarm_level_rye
        else:
            rye_series = []

        diagnostics: Dict[str, Any] = {}
        if rye_series:
            try:
                roll = rolling_rye(rye_series, window=10)
                diagnostics["rolling_rye"] = roll
            except Exception:
                diagnostics["rolling_rye"] = None

            try:
                diagnostics["stability_index"] = stability_index(rye_series)
            except Exception:
                diagnostics["stability_index"] = None

            try:
                diagnostics["recovery_momentum"] = recovery_momentum(rye_series)
            except Exception:
                diagnostics["recovery_momentum"] = None

            try:
                pcts = rye_percentiles(rye_series)
                diagnostics["rye_percentiles"] = pcts
            except Exception:
                diagnostics["rye_percentiles"] = None
        else:
            diagnostics["rolling_rye"] = None
            diagnostics["stability_index"] = None
            diagnostics["recovery_momentum"] = None
            diagnostics["rye_percentiles"] = None

        total_delta_r = float(sum(swarm_level_delta_r)) if swarm_level_delta_r else 0.0
        total_energy = float(sum(swarm_level_energy)) if swarm_level_energy else 0.0
        avg_rye = sum(rye_series) / float(len(rye_series)) if rye_series else None

        return {
            "goal": goal,
            "total_delta_R": total_delta_r,
            "total_energy_E": total_energy,
            "avg_rye": avg_rye,
            "cycle_count": len(swarm_level_delta_r),
            "diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------
    # Agent construction and RNG
    # ------------------------------------------------------------------
    def _ensure_agent_instance(self, agent_id: str, cfg: SwarmAgentConfig) -> CoreAgent:
        if agent_id in self.agents:
            return self.agents[agent_id]

        config = dict(self.base_agent_config)
        config.update(cfg.extra_config or {})
        config.setdefault("domain", cfg.domain)
        config.setdefault("role", cfg.role)

        if self.core_agent_factory is not None:
            agent = self.core_agent_factory(config)
        else:
            agent = CoreAgent(memory_store=self.memory_store, config=config)

        self.agents[agent_id] = agent
        return agent

    def _default_rng(self) -> float:
        """Very small pseudo random source that does not require numpy."""
        # Linear congruential style toy RNG; state lives on the object.
        state = getattr(self, "_rng_state", 1234567)
        state = (1103515245 * state + 12345) % 2_147_483_647
        self._rng_state = state
        return state / 2_147_483_647.0
