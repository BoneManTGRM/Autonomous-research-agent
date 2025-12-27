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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    goal: str
    total_cycles: int
    max_parallel: int = 4
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
        goal = run_config.goal
        # Interpret run_config.total_cycles as the number of rounds each agent
        # should complete rather than the number of global scheduler ticks.
        # To produce the expected number of microâcycles (rounds Ã agents),
        # compute how many batches are needed per round based on max_parallel.
        total_rounds = max(1, int(run_config.total_cycles))
        max_parallel = max(1, int(run_config.max_parallel))
        agent_count = len(self.agent_configs) if self.agent_configs else 0
        # Number of groups required to run all agents once given the parallel cap.
        groups_per_round = 1
        if agent_count > 0 and max_parallel > 0:
            # Ceiling division
            groups_per_round = (agent_count + max_parallel - 1) // max_parallel
        # Effective number of scheduler ticks to run the requested rounds across all agents.
        total_cycles = total_rounds * groups_per_round
        idea_fraction = min(1.0, max(0.0, run_config.idea_fraction))

        cycle_logs: List[Dict[str, Any]] = []
        swarm_level_rye: List[float] = []
        swarm_level_delta_r: List[float] = []
        swarm_level_energy: List[float] = []

        # Curriculum and hallmark state
        hallmark_targets = self._resolve_hallmark_targets(run_config.hallmark_targets)
        curriculum_profile = run_config.curriculum_profile

        for global_cycle in range(total_cycles):
            # (1) Decide which agents run on this step
            eligible = self._eligible_agents(global_cycle, run_config)
            scheduled_agents = self._pick_agents_for_cycle(eligible, max_parallel)

            if not scheduled_agents:
                # No agent eligible. Stop early.
                break

            # Estimate idea vs verify stage for this cycle
            stage = self._stage_for_cycle(global_cycle, total_cycles, run_config)

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
            hallmark_name, subgoal = self._pick_hallmark_and_subgoal(
                hallmark_targets,
                global_cycle,
                curriculum_state,
            )

            # (2) Run each scheduled agent for one cycle
            for agent_id in scheduled_agents:
                cfg = self.agent_configs[agent_id]
                agent = self._ensure_agent_instance(agent_id, cfg)

                cycle_index = self.agent_stats[agent_id].cycles
                domain = cfg.domain or "general"
                source_controls = cfg.source_controls

                # Note: CoreAgent is already wired to use TGRM and internal
                # curriculum. We pass down domain and role. Stage and hallmark
                # are added via extra_config if CoreAgent supports them.
                extra_kwargs: Dict[str, Any] = {}

                if run_config.two_stage_mode:
                    extra_kwargs["stage"] = stage
                if hallmark_name is not None:
                    extra_kwargs["hallmark_target"] = hallmark_name
                if subgoal is not None:
                    extra_kwargs["subgoal"] = subgoal
                if curriculum_state is not None:
                    extra_kwargs["curriculum_state"] = curriculum_state

                try:
                    result = agent.run_cycle(  # type: ignore[arg-type]
                        goal=goal,
                        cycle_index=cycle_index,
                        role=cfg.role,
                        source_controls=source_controls,
                        domain=domain,
                        **extra_kwargs,
                    )
                except TypeError:
                    # Backward compatibility if CoreAgent does not accept extras
                    result = agent.run_cycle(  # type: ignore[arg-type]
                        goal=goal,
                        cycle_index=cycle_index,
                        role=cfg.role,
                        source_controls=source_controls,
                        domain=domain,
                    )

                summary = result.get("summary", {})
                log_entry = result.get("log", {})
                stats = summary.get("tool_usage", {}) or {}

                # Extract metrics
                delta_r = float(summary.get("delta_R", 0.0) or 0.0)
                energy_e = float(summary.get("energy_E", 0.0) or 0.0)
                rye_value = summary.get("RYE")

                self.agent_stats[agent_id].record_cycle(delta_r, energy_e, rye_value)

                if rye_value is not None:
                    swarm_level_rye.append(float(rye_value))
                swarm_level_delta_r.append(delta_r)
                swarm_level_energy.append(energy_e)

                # Tag log entry with swarm metadata
                log_entry = dict(log_entry)
                log_entry["swarm"] = {
                    "agent_id": agent_id,
                    "role": cfg.role,
                    "stage": stage,
                    "hallmark": hallmark_name,
                    "subgoal": subgoal,
                    "curriculum_profile": curriculum_profile,
                    "global_cycle_index": global_cycle,
                }
                log_entry["tool_usage"] = stats
                cycle_logs.append(log_entry)

                # Feed replay buffer if available
                self._log_to_replay(
                    agent_id=agent_id,
                    cfg=cfg,
                    summary=summary,
                    log_entry=log_entry,
                    stage=stage,
                    hallmark=hallmark_name,
                    subgoal=subgoal,
                )

                # Stream short_view for dashboards
                if stream_callback is not None:
                    short_view = summary.get("short_view") or {}
                    payload = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "agent_id": agent_id,
                        "role": cfg.role,
                        "goal": goal,
                        "stage": stage,
                        "hallmark": hallmark_name,
                        "subgoal": subgoal,
                        "short_view": short_view,
                        "meta_signals": summary.get("meta_signals") or {},
                    }
                    stream_callback(payload)

        # Build swarm level metrics and summary
        swarm_metrics = self._build_swarm_metrics(
            goal=goal,
            swarm_level_rye=swarm_level_rye,
            swarm_level_delta_r=swarm_level_delta_r,
            swarm_level_energy=swarm_level_energy,
        )
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
            },
            "agent_summaries": agent_summaries,
            "swarm_metrics": swarm_metrics,
            "cycle_logs": cycle_logs,
        }

    # ------------------------------------------------------------------
    # Scheduling and selection
    # ------------------------------------------------------------------
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
