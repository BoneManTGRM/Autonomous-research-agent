# agent/swarm_manager.py

"""Swarm management layer for the Autonomous Research Agent.

This module provides a SwarmManager that coordinates multiple CoreAgent
instances as a logical swarm. It is designed to:

- Read swarm contracts and hints from presets
- Respect global swarm safety ceilings
- Track per agent and swarm level RYE and stability metrics
- Expose a simple API for:
    - single step swarm updates
    - short diagnostic swarm runs
    - long horizon continuous swarm runs
- Surface effective learning speed factors using 10x learning hints

The design is intentionally conservative:
    - It does not assume any particular CoreAgent API beyond a generic
      `run_step_fn` callback you pass in.
    - If you prefer, you can plug in CoreAgent.run_continuous or any
      other engine function as the `step_fn`.

Typical usage pattern (pseudo code):

    from agent.core_agent import CoreAgent
    from agent.swarm_manager import SwarmManager, SwarmRunConfig

    swarm = SwarmManager(
        preset_name="longevity",
        base_goal="Discover high value longevity interventions.",
        runtime_profile_name="24_hours",
    )

    # Build CoreAgent instances however your app does it
    agents = [
        CoreAgent(memory_store=..., tools=...),
        CoreAgent(memory_store=..., tools=...),
        ...
    ]

    swarm.attach_agents(agents)

    def step_fn(agent, agent_state, global_context):
        # For example, one step of your engine
        # return agent.run_single_cycle(..., **global_context)
        ...

    summary = swarm.run_short_swarm(
        steps=20,
        step_fn=step_fn,
    )

The manager focuses on orchestration, metrics aggregation, and learning
hints. It does not own network, UI, or storage concerns.
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# Concurrency utilities
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
try:  # PythonÂ 3.14 introduced InterpreterPoolExecutor
    from concurrent.futures import InterpreterPoolExecutor  # type: ignore
except Exception:
    InterpreterPoolExecutor = None  # type: ignore

try:
    # Optional import. SwarmManager does not strictly require CoreAgent
    from .core_agent import CoreAgent
except Exception:  # pragma: no cover
    CoreAgent = Any  # type: ignore

try:
    from .presets import (
        get_preset,
        get_runtime_profile,
        get_continuous_mode_defaults,
        get_swarm_global_hints,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "swarm_manager.py requires presets.py with get_preset, "
        "get_runtime_profile, get_continuous_mode_defaults, "
        "and get_swarm_global_hints."
    ) from e


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SwarmRunConfig:
    """Configuration for a swarm run.

    This is mostly a user facing configuration object that the UI or
    orchestration code can construct and pass into SwarmManager.
    """

    preset_name: str = "general"
    runtime_profile_name: str = "24_hours"
    # If None, manager uses preset["swarm"]["default_agents"]
    swarm_size: Optional[int] = None
    # Optional soft ceiling on cycles for a given run
    max_cycles: Optional[int] = None
    # Optional wall clock limit in seconds (manager enforces)
    max_seconds: Optional[float] = None
    # Optional RYE based stop; if provided, manager can early stop
    target_avg_rye: Optional[float] = None
    # Arbitrary user tags for analytics, logging, or exports
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmAgentState:
    """Per agent runtime state tracked by the manager."""

    agent_id: int
    role_name: str
    domain_focus: Optional[str] = None
    mechanism_focus: Optional[List[str]] = None

    cycles_completed: int = 0
    last_cycle_rye: Optional[float] = None
    last_cycle_delta_r: Optional[float] = None
    last_cycle_energy: Optional[float] = None
    cumulative_rye: float = 0.0
    cumulative_energy: float = 0.0

    # You can store arbitrary per agent metadata here
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmStepResult:
    """Result of a single swarm step across all agents."""

    step_index: int
    timestamp: float
    agent_results: List[Dict[str, Any]]
    swarm_metrics: Dict[str, Any]


@dataclass
class SwarmHistoryEntry:
    """Compact history entry for logging or report export."""

    step_index: int
    timestamp: float
    swarm_avg_rye: Optional[float]
    swarm_peak_rye: Optional[float]
    swarm_stability_index: Optional[float]
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmRunSummary:
    """Summary of a completed swarm run."""

    preset_name: str
    runtime_profile_name: str
    base_goal: str
    total_steps: int
    total_seconds: float
    avg_rye: Optional[float]
    peak_rye: Optional[float]
    learning_speed_factor: float
    agents: List[SwarmAgentState]
    history: List[SwarmHistoryEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type alias for the step function used by the manager
StepFn = Callable[
    [CoreAgent, SwarmAgentState, Dict[str, Any]],
    Dict[str, Any],
]


# ---------------------------------------------------------------------------
# Swarm Manager
# ---------------------------------------------------------------------------

class SwarmManager:
    """Coordinator for multi agent swarm runs.

    The SwarmManager is intentionally lightweight. It is a pure Python
    orchestration layer on top of your existing CoreAgent engine.

    Key responsibilities:

    - Build role aware agent states from presets["swarm"]
    - Track per agent and swarm level metrics (RYE, cycles, energy)
    - Expose simple run APIs:
        - run_swarm_step: one logical step for all agents
        - run_short_swarm: finite number of steps
        - run_continuous_swarm: until stop conditions
    - Compute effective learning speed factor from:
        - runtime_profile["learning_speed_factor"]
        - preset["learning_hints"]["learning_speed_factor"]
        - continuous_mode_defaults["learning_hooks"]["learning_speed_factor_default"]
    """

    def __init__(
        self,
        preset_name: str,
        base_goal: str,
        runtime_profile_name: str = "24_hours",
        logger: Optional[Callable[[str], None]] = None,
        *,
        max_workers: int = 1,
        executor_type: str = "thread",
    ) -> None:
        self.preset_name = preset_name
        self.base_goal = base_goal
        self.runtime_profile_name = runtime_profile_name

        self._preset = get_preset(preset_name)
        self._runtime_profile = get_runtime_profile(runtime_profile_name)
        self._continuous_defaults = get_continuous_mode_defaults()
        self._swarm_global_hints = get_swarm_global_hints()

        self._swarm_cfg = self._preset.get("swarm", {})
        self._swarm_orchestration = self._preset.get("swarm_orchestration", {})
        self._learning_hints = self._preset.get("learning_hints", {})

        self._agents: List[CoreAgent] = []
        self._agent_states: List[SwarmAgentState] = []
        self._history: List[SwarmHistoryEntry] = []

        self._logger = logger or (lambda msg: None)
        self._start_time: Optional[float] = None
        self._step_index: int = 0

        # Concurrency options
        # ``max_workers`` controls the degree of parallelism when executing
        # agent steps.  A value of 1 preserves sequential behaviour.  If
        # greater than 1, agents will be run concurrently using the
        # executor specified by ``executor_type``.
        self.max_workers = max_workers
        self.executor_type = executor_type.lower()

        self._log(
            f"[SwarmManager] Initialized for preset='{preset_name}', "
            f"runtime_profile='{runtime_profile_name}'"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def agents(self) -> List[CoreAgent]:
        return self._agents

    @property
    def agent_states(self) -> List[SwarmAgentState]:
        return self._agent_states

    @property
    def history(self) -> List[SwarmHistoryEntry]:
        return self._history

    def attach_agents(self, agents: List[CoreAgent]) -> None:
        """Attach CoreAgent instances and build role aware states.

        This method inspects the preset's swarm configuration to assign
        roles and domain focuses to each agent.
        """
        if not agents:
            raise ValueError("attach_agents called with empty agent list.")

        max_safe = int(self._swarm_global_hints.get("max_agents_safe", 64))
        if len(agents) > max_safe:
            raise ValueError(
                f"Requested swarm size {len(agents)} exceeds safety ceiling {max_safe}."
            )

        self._agents = list(agents)
        self._agent_states = self._build_agent_states(len(agents))
        self._log(
            f"[SwarmManager] Attached {len(agents)} agents with roles "
            f"{[s.role_name for s in self._agent_states]}"
        )

    # ------------------------------------------------------------------
    # Run entry points
    # ------------------------------------------------------------------

    def run_swarm_step(
        self,
        step_fn: StepFn,
        global_context: Optional[Dict[str, Any]] = None,
    ) -> SwarmStepResult:
        """Run one swarm step.

        Parameters
        ----------
        step_fn:
            Callable that receives (agent, agent_state, global_context)
            and returns a result dict. This is where you plug in your
            CoreAgent logic, such as one cycle of TGRM/RYE.
        global_context:
            Dict of values shared by all agents for this step, such as
            goal text, time budget hints, or UI choices.

        Returns
        -------
        SwarmStepResult with per agent results and aggregated metrics.
        """
        if not self._agents or not self._agent_states:
            raise RuntimeError("No agents attached. Call attach_agents first.")

        if global_context is None:
            global_context = {}

        step_start = time.time()
        self._step_index += 1
        now = time.time()
        agent_results: List[Dict[str, Any]] = []

        # Execute agent steps, optionally in parallel
        if self.max_workers > 1:
            # Determine executor class based on requested type
            exec_type = self.executor_type
            if exec_type == "process":
                executor_cls = ProcessPoolExecutor  # type: ignore
            elif exec_type == "interpreter" and InterpreterPoolExecutor is not None:
                executor_cls = InterpreterPoolExecutor  # type: ignore
            else:
                executor_cls = ThreadPoolExecutor  # type: ignore
            # Soft timeout for a single swarm step.
            #
            # Without this, a single hung tool call (e.g. web search) can
            # cause the whole step to block forever because the executor
            # context manager waits for all futures.
            step_timeout_s: Optional[float] = None
            try:
                v = global_context.get("soft_timeout_s")
                if isinstance(v, (int, float)) and float(v) > 0:
                    step_timeout_s = float(v)
            except Exception:
                step_timeout_s = None
            if step_timeout_s is None:
                try:
                    env_v = os.getenv("SWARM_STEP_TIMEOUT_SECONDS")
                    if env_v:
                        step_timeout_s = float(env_v)
                except Exception:
                    step_timeout_s = None

            deadline: Optional[float] = (time.time() + step_timeout_s) if step_timeout_s else None

            executor: Any = executor_cls(max_workers=min(self.max_workers, len(self._agents)))
            future_to_index: Dict[Any, int] = {}
            temp_results: List[Optional[Dict[str, Any]]] = [None] * len(self._agents)

            try:
                for idx, (agent, state) in enumerate(zip(self._agents, self._agent_states)):
                    # Wrap step_fn call to catch exceptions
                    def _call_step(agent=agent, state=state):  # default args bind current values
                        try:
                            return step_fn(agent, state, global_context)
                        except Exception as exc:  # pragma: no cover
                            return {"status": "error", "error": str(exc)}

                    fut = executor.submit(_call_step)
                    future_to_index[fut] = idx

                pending = set(future_to_index.keys())

                while pending:
                    if deadline is not None:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        poll_timeout = min(1.0, max(0.05, remaining))
                    else:
                        poll_timeout = None

                    done, pending = wait(pending, timeout=poll_timeout, return_when=FIRST_COMPLETED)
                    if not done:
                        continue

                    for fut in done:
                        idx = future_to_index.get(fut)
                        if idx is None:
                            continue
                        try:
                            result = fut.result()
                        except Exception as exc:  # pragma: no cover
                            result = {"status": "error", "error": str(exc)}

                        temp_results[idx] = result
                        if isinstance(result, dict):
                            try:
                                self._update_agent_state_from_result(self._agent_states[idx], result)
                            except Exception:
                                pass

                # Timeout path: mark unfinished agents.
                if pending:
                    for fut in list(pending):
                        idx = future_to_index.get(fut)
                        if idx is None:
                            continue
                        if temp_results[idx] is None:
                            temp_results[idx] = {"status": "timeout"}
                        try:
                            fut.cancel()
                        except Exception:
                            pass

            finally:
                # Do not block forever waiting for hung tasks.
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    try:
                        executor.shutdown(wait=False)
                    except Exception:
                        pass
                except Exception:
                    pass

            # Append results in original order
            for r in temp_results:
                if r is not None:
                    agent_results.append(r)
        else:
            # Sequential execution
            for agent, state in zip(self._agents, self._agent_states):
                try:
                    result = step_fn(agent, state, global_context)
                except Exception as exc:  # pragma: no cover
                    result = {
                        "status": "error",
                        "error": str(exc),
                    }
                agent_results.append(result)
                if isinstance(result, dict):
                    self._update_agent_state_from_result(state, result)

        step_seconds = max(0.0, time.time() - step_start)
        swarm_metrics = self._compute_swarm_metrics(
            agent_results,
            step_seconds=step_seconds,
        )
        self._history.append(
            SwarmHistoryEntry(
                step_index=self._step_index,
                timestamp=now,
                swarm_avg_rye=swarm_metrics.get("avg_rye"),
                swarm_peak_rye=swarm_metrics.get("peak_rye"),
                swarm_stability_index=swarm_metrics.get("stability_index"),
                notes={
                    "step_seconds": swarm_metrics.get("step_seconds"),
                    "agent_count": len(self._agents),
                    "learning_speed_factor": swarm_metrics.get(
                        "learning_speed_factor"
                    ),
                    "burst_profile_hint": swarm_metrics.get("burst_profile_hint"),
                },
            )
        )

        return SwarmStepResult(
            step_index=self._step_index,
            timestamp=now,
            agent_results=agent_results,
            swarm_metrics=swarm_metrics,
        )

    def run_short_swarm(
        self,
        steps: int,
        step_fn: StepFn,
        run_config: Optional[SwarmRunConfig] = None,
        global_context: Optional[Dict[str, Any]] = None,
    ) -> SwarmRunSummary:
        """Run a finite number of swarm steps.

        This is ideal for diagnostics and short experiments.
        """
        if steps <= 0:
            raise ValueError("steps must be positive.")

        if run_config is None:
            run_config = SwarmRunConfig(
                preset_name=self.preset_name,
                runtime_profile_name=self.runtime_profile_name,
            )

        if global_context is None:
            global_context = {}

        self._start_run(run_config)
        for _ in range(steps):
            self.run_swarm_step(step_fn, global_context)

        return self._build_summary(run_config)

    def run_continuous_swarm(
        self,
        step_fn: StepFn,
        run_config: Optional[SwarmRunConfig] = None,
        global_context: Optional[Dict[str, Any]] = None,
    ) -> SwarmRunSummary:
        """Run until stop conditions from run_config are satisfied.

        Stop conditions can include:
            - max_cycles (soft ceiling)
            - max_seconds (wall clock limit)
            - target_avg_rye (early stop once achieved)
        """
        if run_config is None:
            run_config = SwarmRunConfig(
                preset_name=self.preset_name,
                runtime_profile_name=self.runtime_profile_name,
            )

        if global_context is None:
            global_context = {}

        self._start_run(run_config)

        while True:
            if self._check_stop_conditions(run_config):
                break
            self.run_swarm_step(step_fn, global_context)

        return self._build_summary(run_config)

    # ------------------------------------------------------------------
    # Learning speed and hints
    # ------------------------------------------------------------------

    def get_effective_learning_speed_factor(self) -> float:
        """Compute the effective learning speed factor.

        The manager multiplies:
            runtime_profile.learning_speed_factor
            preset.learning_hints.learning_speed_factor
            continuous_defaults.learning_hooks.learning_speed_factor_default

        Missing values default to 1.0.
        """
        rp_factor = float(self._runtime_profile.get("learning_speed_factor", 1.0))
        preset_factor = float(self._learning_hints.get("learning_speed_factor", 1.0))

        hooks = self._continuous_defaults.get("learning_hooks", {})
        default_factor = float(hooks.get("learning_speed_factor_default", 1.0))

        return rp_factor * preset_factor * default_factor

    def get_learning_burst_profile_hint(self) -> str:
        """Return the burst profile hint for the current runtime profile.

        One of: "light", "balanced", "aggressive" or custom strings.
        """
        return str(self._runtime_profile.get("burst_profile_hint", "balanced"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_run(self, run_config: SwarmRunConfig) -> None:
        self._start_time = time.time()
        self._step_index = 0
        self._history.clear()
        for s in self._agent_states:
            s.cycles_completed = 0
            s.last_cycle_rye = None
            s.last_cycle_delta_r = None
            s.last_cycle_energy = None
            s.cumulative_rye = 0.0
            s.cumulative_energy = 0.0

        self._log(
            f"[SwarmManager] Starting swarm run with config: {asdict(run_config)}. "
            f"effective_learning_speed_factor={self.get_effective_learning_speed_factor():.3f}"
        )

    def _build_agent_states(self, count: int) -> List[SwarmAgentState]:
        """Assign roles and domains to each agent.

        Role assignment order:
            1) Use preset["swarm"]["role_templates"] if present
            2) Fall back to global SWARM_ROLES
        """
        role_templates = self._swarm_cfg.get("role_templates") or []
        global_roles = self._swarm_global_hints.get("roles") or []

        states: List[SwarmAgentState] = []

        def pick_role(idx: int) -> Tuple[str, Optional[str], Optional[List[str]]]:
            if role_templates:
                tpl = role_templates[idx % len(role_templates)]
                name = str(tpl.get("name", f"agent_{idx}"))
                domain_focus = tpl.get("domain_focus")
                mechanism_focus = tpl.get("mechanism_focus")
                return name, domain_focus, mechanism_focus
            if global_roles:
                base = global_roles[idx % len(global_roles)]
                return (
                    str(base.get("name", f"agent_{idx}")),
                    None,
                    None,
                )
            return f"agent_{idx}", None, None

        for i in range(count):
            rn, df, mf = pick_role(i)
            states.append(
                SwarmAgentState(
                    agent_id=i,
                    role_name=rn,
                    domain_focus=df,
                    mechanism_focus=mf,
                )
            )

        return states

    def _update_agent_state_from_result(
        self,
        state: SwarmAgentState,
        result: Dict[str, Any],
    ) -> None:
        """Update agent state from a step result.

        Expected but optional keys inside result:
            - "rye"
            - "delta_r"
            - "energy"
            - "cycles_completed" (per agent)
        """
        state.cycles_completed += int(result.get("cycles_completed_increment", 1))

        rye = result.get("rye")
        delta_r = result.get("delta_r")
        energy = result.get("energy")

        if isinstance(rye, (int, float)):
            state.last_cycle_rye = float(rye)
            state.cumulative_rye += float(rye)

        if isinstance(delta_r, (int, float)):
            state.last_cycle_delta_r = float(delta_r)

        if isinstance(energy, (int, float)):
            state.last_cycle_energy = float(energy)
            state.cumulative_energy += float(energy)

        # Preserve any other fields for inspection
        extra_fields = {
            k: v
            for k, v in result.items()
            if k
            not in {
                "rye",
                "delta_r",
                "energy",
                "cycles_completed_increment",
            }
        }
        if extra_fields:
            state.extra.update(extra_fields)

    def _compute_swarm_metrics(
        self,
        agent_results: List[Dict[str, Any]],
        step_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Aggregate swarm level metrics from agent results and states."""
        now = time.time()
        step_ryes: List[float] = []
        total_rye = 0.0
        total_energy = 0.0

        for s in self._agent_states:
            if s.last_cycle_rye is not None:
                step_ryes.append(s.last_cycle_rye)
            total_rye += s.cumulative_rye
            total_energy += s.cumulative_energy

        avg_step_rye = sum(step_ryes) / len(step_ryes) if step_ryes else None
        peak_step_rye = max(step_ryes) if step_ryes else None

        avg_rye_overall = None
        if total_energy > 0:
            avg_rye_overall = total_rye / total_energy

        stability_index = self._estimate_stability_index()
        learning_speed_factor = self.get_effective_learning_speed_factor()
        burst_profile_hint = self.get_learning_burst_profile_hint()

        metrics: Dict[str, Any] = {
            "timestamp": now,
            "avg_rye_step": avg_step_rye,
            "peak_rye_step": peak_step_rye,
            "avg_rye": avg_rye_overall,
            "peak_rye": self._max_rye_seen(),
            "stability_index": stability_index,
            "step_seconds": step_seconds,
            "agent_count": len(self._agent_states),
            "learning_speed_factor": learning_speed_factor,
            "burst_profile_hint": burst_profile_hint,
        }
        return metrics

    def _max_rye_seen(self) -> Optional[float]:
        """Compute global peak RYE across agents."""
        peak: Optional[float] = None
        for s in self._agent_states:
            if s.last_cycle_rye is None:
                continue
            if peak is None or s.last_cycle_rye > peak:
                peak = s.last_cycle_rye
        return peak

    def _estimate_stability_index(self) -> Optional[float]:
        """Very lightweight stability index estimate.

        This intentionally does not duplicate RYE metrics module logic.
        It simply inspects recent history of swarm_avg_rye and returns
        a value between 0 and 1 indicating how stable the swarm is.

        - 0 means chaotic or no data
        - 1 means very stable plateau
        """
        if len(self._history) < 3:
            return None

        window = self._learning_hints.get("advanced_expectations", {}).get(
            "rolling_window_short",
            10,
        )
        recent = self._history[-min(window, len(self._history)) :]
        values = [h.swarm_avg_rye for h in recent if h.swarm_avg_rye is not None]

        if len(values) < 3:
            return None

        vmin = min(values)
        vmax = max(values)
        spread = vmax - vmin
        if spread <= 0:
            return 1.0

        # Simple heuristic: smaller spread relative to max value is more stable
        max_abs = max(abs(v) for v in values) or 1.0
        ratio = spread / max_abs
        # Compress to 0 to 1, lower ratio means higher stability
        stability = max(0.0, min(1.0, 1.0 - ratio))
        return stability

    def _check_stop_conditions(self, run_config: SwarmRunConfig) -> bool:
        """Check whether continuous swarm should stop."""
        if self._start_time is None:
            return False

        # max_cycles is interpreted over swarm steps, not agent cycles
        if run_config.max_cycles is not None and self._step_index >= run_config.max_cycles:
            self._log(
                f"[SwarmManager] Stop: reached max swarm steps {run_config.max_cycles}."
            )
            return True

        if run_config.max_seconds is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= run_config.max_seconds:
                self._log(
                    f"[SwarmManager] Stop: reached wall clock limit "
                    f"{run_config.max_seconds} seconds."
                )
                return True

        if run_config.target_avg_rye is not None:
            current_avg_rye = None
            if self._history:
                current_avg_rye = self._history[-1].swarm_avg_rye
            if current_avg_rye is not None and current_avg_rye >= run_config.target_avg_rye:
                self._log(
                    f"[SwarmManager] Stop: target_avg_rye {run_config.target_avg_rye} "
                    f"reached with current_avg_rye={current_avg_rye:.4f}."
                )
                return True

        return False

    def _build_summary(self, run_config: SwarmRunConfig) -> SwarmRunSummary:
        if self._start_time is None:
            total_seconds = 0.0
        else:
            total_seconds = time.time() - self._start_time

        avg_rye = None
        peak_rye = None
        stability = None

        if self._history:
            avg_values = [h.swarm_avg_rye for h in self._history if h.swarm_avg_rye is not None]
            peak_values = [h.swarm_peak_rye for h in self._history if h.swarm_peak_rye is not None]
            stability_values = [
                h.swarm_stability_index
                for h in self._history
                if h.swarm_stability_index is not None
            ]

            if avg_values:
                avg_rye = sum(avg_values) / len(avg_values)
            if peak_values:
                peak_rye = max(peak_values)
            if stability_values:
                stability = sum(stability_values) / len(stability_values)

        learning_speed_factor = self.get_effective_learning_speed_factor()
        burst_profile_hint = self.get_learning_burst_profile_hint()

        return SwarmRunSummary(
            preset_name=self.preset_name,
            runtime_profile_name=self.runtime_profile_name,
            base_goal=self.base_goal,
            total_steps=self._step_index,
            total_seconds=total_seconds,
            avg_rye=avg_rye,
            peak_rye=peak_rye,
            learning_speed_factor=learning_speed_factor,
            agents=list(self._agent_states),
            history=list(self._history),
            metadata={
                "runtime_profile": self._runtime_profile,
                "swarm_config": self._swarm_cfg,
                "swarm_orchestration": self._swarm_orchestration,
                "stability_index_avg": stability,
                "agent_count": len(self._agent_states),
                "learning_speed_factor": learning_speed_factor,
                "burst_profile_hint": burst_profile_hint,
                "run_config": asdict(run_config),
            },
        )

    def _log(self, msg: str) -> None:
        try:
            self._logger(msg)
        except Exception:
            # Never crash from logging
            pass
