# agent/run_manager.py

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core_agent import CoreAgent


RunProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class StageConfig:
    """Configuration for idea and verify stages in two stage mode."""

    idea_cycles: int = 1
    verify_cycles: int = 1
    enable_two_stage: bool = False


@dataclass
class SwarmConfig:
    """Configuration for swarm runs."""

    swarm_size: int = 1
    roles: List[str] = field(default_factory=lambda: ["agent"])
    max_cycles_per_agent: int = 1
    stagger_start: bool = False
    max_agents_per_tick: int = 0  # 0 means no throttling


@dataclass
class LongevityConfig:
    """Longevity specific configuration used for curriculum and hallmarks."""

    hallmark_targets: List[str] = field(default_factory=list)
    curriculum_profile: Optional[str] = None
    subgoal: Optional[str] = None


@dataclass
class RunConfig:
    """Top level configuration for a run.

    This config is deliberately rich so you can drive many patterns
    from the Streamlit UI or from scripts without changing code.
    """

    goal: str
    mode: str = "single"  # single, swarm, two_stage
    domain: str = "general"

    # Cycles and time
    total_cycles: int = 1
    max_seconds: Optional[int] = None  # optional hard time cap (wall clock)

    # RYE and safety controls
    rye_stop_threshold: Optional[float] = None
    equilibrium_stop_label: Optional[str] = None
    min_cycles_before_stop: int = 3

    # Source controls
    source_controls: Dict[str, bool] = field(
        default_factory=lambda: {
            "web": True,
            "pubmed": False,
            "semantic": False,
            "pdf": False,
            "biomarkers": False,
        }
    )

    # Two stage and swarm
    stage_config: StageConfig = field(default_factory=StageConfig)
    swarm_config: SwarmConfig = field(default_factory=SwarmConfig)
    longevity_config: LongevityConfig = field(default_factory=LongevityConfig)

    # Misc
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seed: Optional[int] = None
    notes: Optional[str] = None  # free text description or experiment label


class RunManager:
    """Orchestrate CoreAgent runs for single, two stage, and swarm modes.

    This class is the central conductor for speed of learning experiments.
    It gives you:

    - Single agent runs with RYE aware early stop.
    - Two stage idea plus verify runs.
    - Swarm runs with role based staging (explorer vs critic).
    - Run level learning speed summaries that the UI can show as trends.
    """

    def __init__(
        self,
        memory_store: Any,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.base_config = base_config or {}

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------
    def _build_agent(
        self,
        role: str,
        run_config: RunConfig,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> CoreAgent:
        """Create a CoreAgent configured for a given role and run."""
        cfg = dict(self.base_config)
        cfg.update(extra_config or {})
        cfg.setdefault("role", role)
        cfg.setdefault("domain", run_config.domain)
        cfg.setdefault("ultra_speed", True)

        # Small default so tests do not break if tgrm_level is not set
        cfg.setdefault("tgrm_level", 3)

        return CoreAgent(
            memory_store=self.memory_store,
            config=cfg,
            tools=None,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        run_config: RunConfig,
        progress_cb: Optional[RunProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Run according to the given configuration.

        The return value is a run level summary that can be logged
        or rendered in the UI.
        """
        start_ts = datetime.utcnow().isoformat() + "Z"
        start_time = time.time()

        # Hard wall clock deadline (if configured)
        if run_config.max_seconds is not None:
            deadline_ts: Optional[float] = start_time + float(run_config.max_seconds)
        else:
            deadline_ts = None

        if run_config.mode == "single":
            summary = self._run_single(run_config, progress_cb, start_time, deadline_ts)
        elif run_config.mode == "two_stage":
            summary = self._run_two_stage(run_config, progress_cb, start_time, deadline_ts)
        elif run_config.mode == "swarm":
            summary = self._run_swarm(run_config, progress_cb, start_time, deadline_ts)
        else:
            raise ValueError(f"Unknown run mode: {run_config.mode}")

        end_time = time.time()
        end_ts = datetime.utcnow().isoformat() + "Z"

        summary["run_id"] = run_config.run_id
        summary["goal"] = run_config.goal
        summary["domain"] = run_config.domain
        summary["mode"] = run_config.mode
        summary["start_timestamp"] = start_ts
        summary["end_timestamp"] = end_ts
        summary["elapsed_seconds"] = end_time - start_time
        summary["notes"] = run_config.notes

        # Attach the configured max_seconds and an explicit stop_reason if missing
        summary.setdefault("max_seconds", run_config.max_seconds)
        summary.setdefault("stop_reason", "completed")

        return summary

    # ------------------------------------------------------------------
    # Single agent mode
    # ------------------------------------------------------------------
    def _run_single(
        self,
        run_config: RunConfig,
        progress_cb: Optional[RunProgressCallback],
        start_time: float,
        deadline_ts: Optional[float],
    ) -> Dict[str, Any]:
        agent = self._build_agent(role="agent", run_config=run_config)
        cycles: List[Dict[str, Any]] = []
        best_rye: Optional[float] = None
        last_cycle_meta: Dict[str, Any] = {}
        stop_reason: str = "completed"

        for cycle_index in range(1, run_config.total_cycles + 1):
            if self._time_exceeded(deadline_ts):
                stop_reason = "time_limit"
                break

            cycle_kwargs = self._build_cycle_kwargs(
                run_config=run_config,
                cycle_index=cycle_index,
                role="agent",
                stage="idea",
            )
            result = agent.run_cycle(**cycle_kwargs)

            cycle_log = result.get("log", {})
            cycles.append(cycle_log)

            short_view = cycle_log.get("short_view", {})
            rye_val = short_view.get("RYE")
            eq_label = short_view.get("equilibrium_label")

            best_rye = self._update_best(best_rye, rye_val)
            last_cycle_meta = short_view or {}

            if progress_cb is not None:
                progress_cb(
                    {
                        "run_id": run_config.run_id,
                        "mode": "single",
                        "cycle_index": cycle_index,
                        "short_view": short_view,
                    }
                )

            if self._should_stop_early(
                run_config=run_config,
                cycle_index=cycle_index,
                rye=rye_val,
                equilibrium_label=eq_label,
            ):
                stop_reason = "early_stop"
                break

        return self._build_run_summary(
            run_config=run_config,
            cycles=cycles,
            best_rye=best_rye,
            last_cycle_meta=last_cycle_meta,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Two stage mode
    # ------------------------------------------------------------------
    def _run_two_stage(
        self,
        run_config: RunConfig,
        progress_cb: Optional[RunProgressCallback],
        start_time: float,
        deadline_ts: Optional[float],
    ) -> Dict[str, Any]:
        stage_cfg = run_config.stage_config
        if not stage_cfg.enable_two_stage:
            # Fallback to single if two stage is not enabled
            return self._run_single(run_config, progress_cb, start_time, deadline_ts)

        idea_agent = self._build_agent(role="researcher", run_config=run_config)
        critic_agent = self._build_agent(role="critic", run_config=run_config)

        cycles: List[Dict[str, Any]] = []
        best_rye: Optional[float] = None
        last_cycle_meta: Dict[str, Any] = {}
        stop_reason: str = "completed"

        cycle_index = 1
        while cycle_index <= run_config.total_cycles:
            if self._time_exceeded(deadline_ts):
                stop_reason = "time_limit"
                break

            # Idea stage block
            for _ in range(stage_cfg.idea_cycles):
                if cycle_index > run_config.total_cycles:
                    break
                if self._time_exceeded(deadline_ts):
                    stop_reason = "time_limit"
                    break

                cycle_kwargs = self._build_cycle_kwargs(
                    run_config=run_config,
                    cycle_index=cycle_index,
                    role="researcher",
                    stage="idea",
                )
                idea_result = idea_agent.run_cycle(**cycle_kwargs)
                idea_log = idea_result.get("log", {})
                cycles.append(idea_log)

                short_view = idea_log.get("short_view", {})
                rye_val = short_view.get("RYE")
                eq_label = short_view.get("equilibrium_label")
                best_rye = self._update_best(best_rye, rye_val)
                last_cycle_meta = short_view or {}

                if progress_cb is not None:
                    progress_cb(
                        {
                            "run_id": run_config.run_id,
                            "mode": "two_stage",
                            "stage": "idea",
                            "cycle_index": cycle_index,
                            "short_view": short_view,
                        }
                    )

                if self._should_stop_early(
                    run_config=run_config,
                    cycle_index=cycle_index,
                    rye=rye_val,
                    equilibrium_label=eq_label,
                ):
                    stop_reason = "early_stop"
                    cycle_index += 1
                    break

                cycle_index += 1

            if self._time_exceeded(deadline_ts):
                stop_reason = "time_limit"
                break
            if cycle_index > run_config.total_cycles:
                break
            if stop_reason == "early_stop":
                break

            # Verify stage block
            for _ in range(stage_cfg.verify_cycles):
                if cycle_index > run_config.total_cycles:
                    break
                if self._time_exceeded(deadline_ts):
                    stop_reason = "time_limit"
                    break

                cycle_kwargs = self._build_cycle_kwargs(
                    run_config=run_config,
                    cycle_index=cycle_index,
                    role="critic",
                    stage="verify",
                )
                verify_result = critic_agent.run_cycle(**cycle_kwargs)
                verify_log = verify_result.get("log", {})
                cycles.append(verify_log)

                short_view = verify_log.get("short_view", {})
                rye_val = short_view.get("RYE")
                eq_label = short_view.get("equilibrium_label")
                best_rye = self._update_best(best_rye, rye_val)
                last_cycle_meta = short_view or {}

                if progress_cb is not None:
                    progress_cb(
                        {
                            "run_id": run_config.run_id,
                            "mode": "two_stage",
                            "stage": "verify",
                            "cycle_index": cycle_index,
                            "short_view": short_view,
                        }
                    )

                if self._should_stop_early(
                    run_config=run_config,
                    cycle_index=cycle_index,
                    rye=rye_val,
                    equilibrium_label=eq_label,
                ):
                    stop_reason = "early_stop"
                    cycle_index += 1
                    break

                cycle_index += 1

            if self._time_exceeded(deadline_ts):
                stop_reason = "time_limit"
                break
            if cycle_index > run_config.total_cycles:
                break
            if stop_reason == "early_stop":
                break

        return self._build_run_summary(
            run_config=run_config,
            cycles=cycles,
            best_rye=best_rye,
            last_cycle_meta=last_cycle_meta,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Swarm mode
    # ------------------------------------------------------------------
    def _run_swarm(
        self,
        run_config: RunConfig,
        progress_cb: Optional[RunProgressCallback],
        start_time: float,
        deadline_ts: Optional[float],
    ) -> Dict[str, Any]:
        swarm_cfg = run_config.swarm_config
        if swarm_cfg.swarm_size <= 1:
            return self._run_single(run_config, progress_cb, start_time, deadline_ts)

        agents: List[CoreAgent] = []
        agent_roles: List[str] = []

        # Build swarm with roles cycling over the list
        for i in range(swarm_cfg.swarm_size):
            role = self._pick_role_for_index(i, swarm_cfg.roles)
            agent_roles.append(role)
            agents.append(self._build_agent(role=role, run_config=run_config))

        cycles: List[Dict[str, Any]] = []
        best_rye: Optional[float] = None
        last_cycle_meta: Dict[str, Any] = {}
        stop_reason: str = "completed"

        total_cycles_budget = run_config.total_cycles * swarm_cfg.swarm_size
        cycle_index_global = 0

        while cycle_index_global < total_cycles_budget:
            if self._time_exceeded(deadline_ts):
                stop_reason = "time_limit"
                break

            for agent_idx, agent in enumerate(agents):
                if self._time_exceeded(deadline_ts):
                    stop_reason = "time_limit"
                    break
                if cycle_index_global >= total_cycles_budget:
                    break

                role = agent_roles[agent_idx]
                stage = self._infer_stage_from_role(role)

                cycle_index_global += 1
                local_cycle_index = cycle_index_global

                cycle_kwargs = self._build_cycle_kwargs(
                    run_config=run_config,
                    cycle_index=local_cycle_index,
                    role=role,
                    stage=stage,
                )
                result = agent.run_cycle(**cycle_kwargs)
                cycle_log = result.get("log", {})
                cycles.append(cycle_log)

                short_view = cycle_log.get("short_view", {})
                rye_val = short_view.get("RYE")
                eq_label = short_view.get("equilibrium_label")
                best_rye = self._update_best(best_rye, rye_val)
                last_cycle_meta = short_view or {}

                if progress_cb is not None:
                    progress_cb(
                        {
                            "run_id": run_config.run_id,
                            "mode": "swarm",
                            "agent_index": agent_idx,
                            "role": role,
                            "cycle_index_global": cycle_index_global,
                            "short_view": short_view,
                        }
                    )

                if self._should_stop_early(
                    run_config=run_config,
                    cycle_index=cycle_index_global,
                    rye=rye_val,
                    equilibrium_label=eq_label,
                ):
                    stop_reason = "early_stop"
                    break

            if self._time_exceeded(deadline_ts):
                stop_reason = "time_limit"
                break
            if stop_reason == "early_stop":
                break

        return self._build_run_summary(
            run_config=run_config,
            cycles=cycles,
            best_rye=best_rye,
            last_cycle_meta=last_cycle_meta,
            is_swarm=True,
            swarm_roles=agent_roles,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _time_exceeded(self, deadline_ts: Optional[float]) -> bool:
        """Return True if the wall-clock deadline has passed."""
        if deadline_ts is None:
            return False
        return time.time() >= deadline_ts

    def _update_best(
        self,
        best: Optional[float],
        current: Optional[float],
    ) -> Optional[float]:
        if current is None:
            return best
        if best is None:
            return current
        return current if current > best else best

    def _should_stop_early(
        self,
        run_config: RunConfig,
        cycle_index: int,
        rye: Optional[float],
        equilibrium_label: Optional[str],
    ) -> bool:
        if cycle_index < run_config.min_cycles_before_stop:
            return False

        if run_config.rye_stop_threshold is not None and rye is not None:
            if rye < run_config.rye_stop_threshold:
                return True

        if run_config.equilibrium_stop_label and equilibrium_label:
            if equilibrium_label == run_config.equilibrium_stop_label:
                return True

        return False

    def _pick_role_for_index(self, idx: int, roles: List[str]) -> str:
        if not roles:
            return "agent"
        return roles[idx % len(roles)]

    def _infer_stage_from_role(self, role: str) -> str:
        r = (role or "agent").lower()
        if r in {"researcher", "explorer"}:
            return "idea"
        if r in {"critic", "planner", "synthesizer", "integrator"}:
            return "verify"
        return "idea"

    def _build_cycle_kwargs(
        self,
        run_config: RunConfig,
        cycle_index: int,
        role: str,
        stage: Optional[str],
    ) -> Dict[str, Any]:
        """Assemble kwargs for CoreAgent.run_cycle.

        This is the single place that knows about longevity config,
        hallmarks, curriculum, and any extra fields you add later.
        """
        kwargs: Dict[str, Any] = {
            "goal": run_config.goal,
            "cycle_index": cycle_index,
            "role": role,
            "source_controls": run_config.source_controls,
            "pdf_bytes": None,
            "biomarker_snapshot": None,
            "domain": run_config.domain,
        }

        if stage is not None:
            kwargs["stage"] = stage

        lon_cfg = run_config.longevity_config
        if lon_cfg.hallmark_targets:
            kwargs["hallmark"] = lon_cfg.hallmark_targets[0]
        if lon_cfg.subgoal:
            kwargs["subgoal"] = lon_cfg.subgoal
        if lon_cfg.curriculum_profile:
            kwargs["curriculum_state"] = {"profile": lon_cfg.curriculum_profile}

        return kwargs

    def _build_run_summary(
        self,
        run_config: RunConfig,
        cycles: List[Dict[str, Any]],
        best_rye: Optional[float],
        last_cycle_meta: Dict[str, Any],
        is_swarm: bool = False,
        swarm_roles: Optional[List[str]] = None,
        stop_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a compact run level summary from cycle logs."""

        rye_values: List[float] = []
        for c in cycles:
            sv = c.get("short_view") or {}
            val = sv.get("RYE")
            if isinstance(val, (int, float)):
                rye_values.append(float(val))

        avg_rye = sum(rye_values) / len(rye_values) if rye_values else None

        learning_speed_summary = {
            "avg_rye": avg_rye,
            "best_rye": best_rye,
            "num_cycles": len(cycles),
        }

        if last_cycle_meta:
            learning_speed_summary["last_cycle"] = last_cycle_meta

        summary: Dict[str, Any] = {
            "cycles": cycles,
            "learning_speed": learning_speed_summary,
            "is_swarm": is_swarm,
        }

        if is_swarm and swarm_roles is not None:
            summary["swarm_roles"] = swarm_roles

        if stop_reason is not None:
            summary["stop_reason"] = stop_reason

        return summary
