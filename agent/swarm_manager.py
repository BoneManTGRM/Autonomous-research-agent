"""Swarm orchestrator for the Autonomous Research Agent.

This module coordinates multi agent swarms on top of the core
Reparodynamics engine. It is designed to be:

- Flexible: pluggable single agent runner, curriculum, and profiles.
- Fast: thread based parallelism with bounded workers and soft timeouts.
- Safe: optional stability, RYE, and discovery diagnostics.
- Future friendly: clean hooks for MSIL v2 and protocol synthesis.

The orchestrator itself never talks directly to the LLM. It only knows
how to:

1) Build payloads for single agent runs.
2) Schedule many agents in parallel or sequential modes.
3) Aggregate all outputs into a single swarm summary.
4) Route diagnostics to optional modules when available.
5) Evolve role weights over time using lightweight heuristics.
"""

from __future__ import annotations

import time
import uuid
import copy
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,
)

# InterpreterPoolExecutor was added in PythonÂ 3.14; import lazily to
# gracefully handle older versions where it is unavailable.
try:  # pragma: no cover
    from concurrent.futures import InterpreterPoolExecutor  # type: ignore
except Exception:  # pragma: no cover
    InterpreterPoolExecutor = None  # type: ignore
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# Optional YAML for external swarm profiles and curricula
try:  # pragma: no cover - optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore

# Optional advanced diagnostics
try:  # pragma: no cover - optional
    from .rye_metrics import (
        rolling_rye,
        stability_index,
        recovery_momentum,
        build_run_diagnostics,
    )
except Exception:  # pragma: no cover - optional
    rolling_rye = None  # type: ignore
    stability_index = None  # type: ignore
    recovery_momentum = None  # type: ignore
    build_run_diagnostics = None  # type: ignore

try:  # pragma: no cover - optional
    from .msil_v2 import MetaSkillIntelligenceV2
except Exception:  # pragma: no cover - optional
    MetaSkillIntelligenceV2 = None  # type: ignore

try:  # pragma: no cover - optional
    from .stability_kernel import StabilityKernel
except Exception:  # pragma: no cover - optional
    StabilityKernel = None  # type: ignore

try:  # pragma: no cover - optional
    from .discovery_manager import DiscoveryManager
except Exception:  # pragma: no cover - optional
    DiscoveryManager = None  # type: ignore

try:  # pragma: no cover - optional
    from .protocol_synthesizer import ProtocolSynthesizer
except Exception:  # pragma: no cover - optional
    ProtocolSynthesizer = None  # type: ignore


# ---------------------------------------------------------------------------
# Role specification and simple evolution engine
# ---------------------------------------------------------------------------


@dataclass
class RoleSpec:
    """Specification for a swarm role.

    Attributes
    ----------
    name:
        Human readable role name, for example "Explorer".
    domain:
        Domain label, for example "longevity", "math", or "general".
    system_hint:
        Optional extra hint string for the agent system prompt.
    weight:
        Relative importance used to bias selection in future swarms.
    """

    name: str
    domain: str = "general"
    system_hint: Optional[str] = None
    weight: float = 1.0


class RoleEvolutionEngine:
    """Small heuristic engine that nudges role weights over time.

    This is deliberately simple. It looks at the last swarm summary,
    aggregates a score per role, and slightly increases or decreases
    the role weight for the next swarm.

    Intended input signals:
    - RYE metrics per agent if available.
    - Discovery flags per agent if DiscoveryManager is enabled.
    """

    def evolve(
        self,
        roles: Sequence[RoleSpec],
        swarm_summary: Dict[str, Any],
    ) -> List[RoleSpec]:
        if not roles:
            return list(roles)

        agent_summaries = swarm_summary.get("agent_summaries", [])
        if not agent_summaries:
            return list(roles)

        # Aggregate scores per role name
        scores: Dict[str, float] = {}
        for agent in agent_summaries:
            role_name = agent.get("role", "")
            if not role_name:
                continue

            diagnostics = agent.get("diagnostics", {})
            rye_diag = diagnostics.get("rye", {})
            discoveries = diagnostics.get("discoveries", {})

            rye_score = rye_diag.get("median")
            if rye_score is None:
                rye_score = agent.get("rye_score")

            if rye_score is None:
                rye_score = 1.0

            disc_bonus = 0.0
            if isinstance(discoveries, dict):
                disc_bonus += float(discoveries.get("major_count", 0)) * 2.0
                disc_bonus += float(discoveries.get("minor_count", 0)) * 0.5

            score = float(rye_score) + disc_bonus
            scores[role_name] = scores.get(role_name, 0.0) + score

        if not scores:
            return list(roles)

        max_score = max(scores.values())
        if max_score <= 0:
            return list(roles)

        updated: List[RoleSpec] = []
        for role in roles:
            raw = scores.get(role.name, max_score * 0.5)
            boost = raw / max_score  # 0 to 1 scale
            target_weight = 0.5 + 0.5 * boost
            new_weight = 0.8 * role.weight + 0.2 * target_weight
            updated.append(
                RoleSpec(
                    name=role.name,
                    domain=role.domain,
                    system_hint=role.system_hint,
                    weight=new_weight,
                )
            )

        return updated


class CrossDomainFederation:
    """Simple cross domain bridge.

    Right now this just buffers swarm headers so future runs can use
    cross domain hints or curricula if desired.
    """

    def __init__(self) -> None:
        self._buffer: List[Dict[str, Any]] = []

    def exchange(self, swarm_summary: Dict[str, Any]) -> Dict[str, Any]:
        self._buffer.append(
            {
                "run_id": swarm_summary.get("run_id"),
                "goal": swarm_summary.get("goal"),
                "timestamp": swarm_summary.get("timestamp"),
                "domains": sorted(
                    {
                        agent.get("domain", "general")
                        for agent in swarm_summary.get(
                            "agent_summaries", []
                        )
                    }
                ),
            }
        )
        return {
            "buffer_size": len(self._buffer),
            "last_goal": swarm_summary.get("goal"),
        }


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------


def _load_yaml(path_str: str) -> Optional[Dict[str, Any]]:
    if yaml is None:
        return None
    try:  # pragma: no cover - optional
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / path_str
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _load_swarm_profiles_from_yaml() -> Optional[List[RoleSpec]]:
    cfg = _load_yaml("config/swarm_profiles.yaml")
    if not cfg:
        return None

    roles_cfg = cfg.get("roles", [])
    roles: List[RoleSpec] = []
    for item in roles_cfg:
        if not isinstance(item, dict):
            continue
        roles.append(
            RoleSpec(
                name=str(item.get("name", "Agent")),
                domain=str(item.get("domain", "general")),
                system_hint=item.get("system_hint"),
                weight=float(item.get("weight", 1.0)),
            )
        )
    return roles or None


def _default_roles() -> List[RoleSpec]:
    """Fallback role mix that works for most research goals."""
    return [
        RoleSpec(
            name="Explorer",
            domain="general",
            system_hint="Aggressively search for new signals, mechanisms, and hypotheses.",
        ),
        RoleSpec(
            name="Critic",
            domain="general",
            system_hint="Attack weak points, contradictions, and overclaims.",
        ),
        RoleSpec(
            name="Planner",
            domain="general",
            system_hint="Design experiments, roadmaps, and protocols.",
        ),
        RoleSpec(
            name="Integrator",
            domain="general",
            system_hint="Fuse all findings into coherent, verified narratives.",
        ),
    ]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

AgentFn = Callable[[Dict[str, Any]], Dict[str, Any]]
TraceCallback = Callable[[Dict[str, Any]], None]


@dataclass
class SwarmOrchestrator:
    """High level swarm coordinator.

    Main public entrypoint is `run_swarm`.

    Key options
    -----------
    - agent_fn: default single agent runner (can be overridden per call).
    - max_workers: max parallel threads for burst and pulse modes.
    - max_swarm_size: hard cap on agents per swarm.
    - default_mode: burst, pulse, sequential, or hybrid.
    - default_domain: domain label when none is provided.
    - enable_cross_domain_bridge: plug into CrossDomainFederation.
    - enable_diagnostics: build full diagnostics where possible.
    - enable_msil: call MSIL v2 if available.
    - enable_stability_kernel: call StabilityKernel if available.
    - enable_discovery_manager: call DiscoveryManager if available.
    - enable_protocol_synthesizer: call ProtocolSynthesizer if available.
    - record_history: append each swarm summary to self.history.
    - trace_callbacks: optional list of functions called with the final
      swarm_summary for logging or external monitoring.
    """

    agent_fn: Optional[AgentFn] = None
    max_workers: int = 32
    max_swarm_size: int = 64
    default_mode: str = "burst"
    default_domain: str = "general"

    enable_cross_domain_bridge: bool = True
    enable_diagnostics: bool = True
    enable_msil: bool = True
    enable_stability_kernel: bool = True
    enable_discovery_manager: bool = True
    enable_protocol_synthesizer: bool = True
    record_history: bool = True

    roles: List[RoleSpec] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    trace_callbacks: List[TraceCallback] = field(default_factory=list)

    # Optional advanced modules
    msil: Any = None
    stability_kernel: Any = None
    discovery_manager: Any = None
    protocol_synthesizer: Any = None

    # ------------------------------------------------------------------
    # Extended configuration options
    #
    # ``executor_type`` controls which executor implementation to use for
    # parallel agent execution. The default of ``"thread"`` preserves the
    # existing behaviour of using a ``ThreadPoolExecutor``. Other supported
    # values are ``"process"`` (uses ``ProcessPoolExecutor`` to leverage
    # multiple CPU cores for CPUâbound workloads) and ``"interpreter"``
    # (uses ``InterpreterPoolExecutor`` when running on PythonÂ 3.14+).  If
    # the requested executor is unavailable (for example, on older Python
    # versions), the orchestrator will automatically fall back to the
    # threaded executor to maintain compatibility.
    executor_type: str = "thread"

    # ``randomize_roles`` optionally randomizes role selection for each
    # swarm based on the role weights.  When set to ``True``, roles are
    # sampled without replacement using their weights as probabilities.
    # This can encourage greater diversity in role ordering across runs.
    randomize_roles: bool = False

    def __post_init__(self) -> None:
        # Load roles from YAML or defaults
        if not self.roles:
            self.roles = _load_swarm_profiles_from_yaml() or _default_roles()

        # Attach optional modules only if enabled
        if self.enable_msil and MetaSkillIntelligenceV2 is not None and self.msil is None:
            try:
                self.msil = MetaSkillIntelligenceV2()
            except Exception:
                self.msil = None

        if self.enable_stability_kernel and StabilityKernel is not None and self.stability_kernel is None:
            try:
                self.stability_kernel = StabilityKernel()
            except Exception:
                self.stability_kernel = None

        if self.enable_discovery_manager and DiscoveryManager is not None and self.discovery_manager is None:
            try:
                self.discovery_manager = DiscoveryManager()
            except Exception:
                self.discovery_manager = None

        if self.enable_protocol_synthesizer and ProtocolSynthesizer is not None and self.protocol_synthesizer is None:
            try:
                self.protocol_synthesizer = ProtocolSynthesizer()
            except Exception:
                self.protocol_synthesizer = None

        # Always available helpers
        self.role_evo = RoleEvolutionEngine()
        self.federation = CrossDomainFederation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_swarm(
        self,
        goal: str,
        base_preset: Dict[str, Any],
        swarm_size: int = 4,
        max_cycles: int = 6,
        mode: Optional[str] = None,
        agent_fn: Optional[AgentFn] = None,
        run_id: Optional[str] = None,
        domain: Optional[str] = None,
        curriculum_id: Optional[str] = None,
        profile_name: Optional[str] = None,
        seed: Optional[int] = None,
        global_context: Optional[Dict[str, Any]] = None,
        per_agent_overrides: Optional[Sequence[Dict[str, Any]]] = None,
        timeout_s: Optional[float] = None,
        return_agent_payloads: bool = False,
        extra_tags: Optional[Dict[str, Any]] = None,
        **agent_extra: Any,
    ) -> Dict[str, Any]:
        """Run a full swarm on a single goal.

        Parameters
        ----------
        goal:
            High level research or discovery goal.
        base_preset:
            Base preset used for all agents. Cloned per role.
        swarm_size:
            Number of agents requested. Will be capped by max_swarm_size.
        max_cycles:
            Max cycles per agent. Interpreted by the single agent runner.
        mode:
            "burst"       - maximize parallelism.
            "pulse"       - half sized bursts, good for longer runs.
            "sequential"  - one by one, lowest resource mode.
            "hybrid"      - start burst, then switch to pulse.
            If None, default_mode is used.
        agent_fn:
            Optional override runner for this call.
        run_id:
            Optional external run id. If None a UUID is generated.
        domain:
            Domain label. If None default_domain is used.
        curriculum_id:
            Optional curriculum identifier for long running programs.
        profile_name:
            Optional profile name from swarm_profiles.yaml, if used.
        seed:
            Optional random seed for downstream components.
        global_context:
            Arbitrary info that is broadcast into every agent payload.
        per_agent_overrides:
            Optional list of dicts where index i provides extra keys
            for agent i only.
        timeout_s:
            Soft timeout for the whole swarm. Agents that do not finish
            before this time are marked as timed out.
        return_agent_payloads:
            If True, the swarm summary will include a debug field
            "agent_payloads" containing the full payloads used.
        extra_tags:
            Optional dict of extra metadata that will be attached under
            swarm_summary["tags"].
        agent_extra:
            Passed through to every agent payload.
        """
        if agent_fn is None:
            agent_fn = self.agent_fn
        if agent_fn is None:
            raise ValueError(
                "SwarmOrchestrator requires an agent_fn at init time or in run_swarm."
            )

        if mode is None:
            mode = self.default_mode

        if swarm_size <= 0:
            raise ValueError("swarm_size must be at least 1")
        swarm_size = min(swarm_size, self.max_swarm_size)

        domain = domain or self.default_domain

        run_id = run_id or str(uuid.uuid4())
        timestamp_start = time.time()

        chosen_roles = self._select_roles_for_swarm(
            swarm_size=swarm_size,
            domain=domain,
        )

        agent_payloads: List[Dict[str, Any]] = []
        for idx, role in enumerate(chosen_roles):
            payload = self._build_agent_payload(
                index=idx,
                role=role,
                goal=goal,
                base_preset=base_preset,
                max_cycles=max_cycles,
                run_id=run_id,
                swarm_mode=mode,
                curriculum_id=curriculum_id,
                profile_name=profile_name,
                seed=seed,
                global_context=global_context,
                **agent_extra,
            )
            agent_payloads.append(payload)

        if per_agent_overrides:
            agent_payloads = self._apply_per_agent_overrides(
                agent_payloads, per_agent_overrides
            )

        if mode == "sequential":
            agent_summaries = [agent_fn(p) for p in agent_payloads]
        elif mode == "hybrid":
            # First half burst, second half pulse
            half = max(1, len(agent_payloads) // 2)
            first_half = agent_payloads[:half]
            second_half = agent_payloads[half:]
            agent_summaries = []
            agent_summaries.extend(
                self._run_parallel(
                    agent_fn, first_half, mode="burst", soft_timeout_s=timeout_s
                )
            )
            if second_half:
                agent_summaries.extend(
                    self._run_parallel(
                        agent_fn, second_half, mode="pulse", soft_timeout_s=timeout_s
                    )
                )
        else:
            agent_summaries = self._run_parallel(
                agent_fn, agent_payloads, mode=mode, soft_timeout_s=timeout_s
            )

        elapsed = time.time() - timestamp_start

        diagnostics = self._build_swarm_diagnostics(
            goal=goal,
            run_id=run_id,
            agent_summaries=agent_summaries,
            mode=mode,
            elapsed_seconds=elapsed,
            curriculum_id=curriculum_id,
            profile_name=profile_name,
        )

        swarm_summary: Dict[str, Any] = {
            "run_id": run_id,
            "goal": goal,
            "mode": mode,
            "swarm_size": len(agent_summaries),
            "requested_swarm_size": swarm_size,
            "roles": [r.name for r in chosen_roles],
            "agent_summaries": agent_summaries,
            "diagnostics": diagnostics,
            "timestamp": timestamp_start,
            "elapsed_seconds": elapsed,
            "domain": domain,
        }

        if curriculum_id is not None:
            swarm_summary["curriculum_id"] = curriculum_id
        if profile_name is not None:
            swarm_summary["profile_name"] = profile_name
        if seed is not None:
            swarm_summary["seed"] = seed
        if extra_tags:
            swarm_summary["tags"] = dict(extra_tags)

        if return_agent_payloads:
            swarm_summary["agent_payloads"] = agent_payloads

        # Final reparodynamic hooks

        # Add full diagnostics
        if self.record_history:
            self.history.append(swarm_summary)

        # Evolve roles dynamically
        self.roles = self.role_evo.evolve(self.roles, swarm_summary)

        # Cross domain bridge optional layer
        if self.enable_cross_domain_bridge:
            _ = self.federation.exchange(swarm_summary)

        # External tracing hooks
        for cb in self.trace_callbacks:
            try:
                cb(swarm_summary)
            except Exception:
                # Tracing must never break the swarm
                continue

        return swarm_summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_roles_for_swarm(
        self,
        swarm_size: int,
        domain: str,
    ) -> List[RoleSpec]:
        """Pick a list of roles for this swarm, biased by weight and domain.

        By default the orchestrator repeats roles proportionally to their
        weights in a deterministic roundârobin manner.  If
        ``self.randomize_roles`` is true, roles are selected randomly based
        on their weights using ``random.choices``.  This provides a
        lightweight mechanism to introduce variety across swarms while
        respecting weight biases.
        """
        # Ensure we have at least some roles defined
        if not self.roles:
            self.roles = _default_roles()

        # Filter roles by domain when possible
        domain_roles = [r for r in self.roles if r.domain == domain]
        pool = domain_roles or self.roles

        # Build a simple weight vector proportional to each role's weight.
        weights = [max(0.0, r.weight) for r in pool]
        # If all weights are zero, give each role equal probability.
        if all(w == 0.0 for w in weights):
            weights = [1.0 for _ in pool]

        if self.randomize_roles:
            # Sample without replacement if possible.  When swarm_size
            # exceeds the number of available roles, sample with
            # replacement.  random.sample does not accept weights prior
            # to PythonÂ 3.11; therefore we use random.choices when
            # replacement is needed and fallback to sampling by ranks.
            import random

            chosen: List[RoleSpec] = []
            if swarm_size <= len(pool):
                # Normalize weights for sample without replacement
                # Approach: compute cumulative distribution and draw
                # sequentially while updating weights to avoid
                # reâselecting roles.  This avoids external
                # dependencies such as numpy.
                available = list(pool)
                current_weights = list(weights)
                for _ in range(swarm_size):
                    total = sum(current_weights)
                    # Fall back to uniform probabilities if total is zero
                    probs = [w / total if total > 0 else 1.0 / len(available) for w in current_weights]
                    # Draw one role based on probabilities
                    r = random.choices(available, weights=probs, k=1)[0]
                    chosen.append(r)
                    # Remove selected role from available list and weights
                    idx = available.index(r)
                    del available[idx]
                    del current_weights[idx]
            else:
                # Need to sample with replacement when swarm_size > pool
                import random
                chosen = random.choices(pool, weights=weights, k=swarm_size)
            return chosen

        # Deterministic roundârobin selection using weight repetition
        weighted: List[RoleSpec] = []
        for role in pool:
            reps = max(1, int(round(role.weight * 2)))
            weighted.extend([role] * reps)

        if not weighted:
            weighted = list(pool)

        # Fill the selection by cycling through weighted list
        chosen: List[RoleSpec] = []
        idx = 0
        while len(chosen) < swarm_size:
            chosen.append(weighted[idx % len(weighted)])
            idx += 1
        return chosen

    def _build_agent_payload(
        self,
        index: int,
        role: RoleSpec,
        goal: str,
        base_preset: Dict[str, Any],
        max_cycles: int,
        run_id: str,
        swarm_mode: str,
        curriculum_id: Optional[str],
        profile_name: Optional[str],
        seed: Optional[int],
        global_context: Optional[Dict[str, Any]],
        **agent_extra: Any,
    ) -> Dict[str, Any]:
        """Clone and decorate the base preset for a single agent."""
        # Deep copy so each agent can mutate its preset safely
        preset = copy.deepcopy(base_preset or {})
        preset["role_name"] = role.name
        preset["role_domain"] = role.domain
        if role.system_hint:
            preset["role_system_hint"] = role.system_hint

        payload: Dict[str, Any] = {
            "agent_index": index,
            "agent_id": f"agent_{index + 1}_{role.name}",
            "role": role.name,
            "domain": role.domain,
            "goal": goal,
            "preset": preset,
            "max_cycles": max_cycles,
            "run_id": run_id,
            "swarm_mode": swarm_mode,
        }

        if curriculum_id is not None:
            payload["curriculum_id"] = curriculum_id
        if profile_name is not None:
            payload["profile_name"] = profile_name
        if seed is not None:
            payload["seed"] = seed
        if global_context is not None:
            payload["global_context"] = dict(global_context)

        payload.update(agent_extra)
        return payload

    def _apply_per_agent_overrides(
        self,
        payloads: Sequence[Dict[str, Any]],
        overrides: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply per agent overrides by index."""
        result: List[Dict[str, Any]] = []
        for idx, payload in enumerate(payloads):
            if idx < len(overrides) and overrides[idx]:
                merged = dict(payload)
                merged.update(overrides[idx])
                result.append(merged)
            else:
                result.append(payload)
        return result

    def _run_parallel(
        self,
        agent_fn: AgentFn,
        payloads: Sequence[Dict[str, Any]],
        mode: str,
        soft_timeout_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run agents in parallel using a bounded thread pool."""
        if not payloads:
            return []

        # Determine worker count based on mode
        if mode == "pulse":
            workers = min(len(payloads), max(1, self.max_workers // 2))
        else:
            workers = min(len(payloads), self.max_workers)

        results: List[Optional[Dict[str, Any]]] = [None] * len(payloads)
        start_time = time.time()
        deadline = start_time + soft_timeout_s if soft_timeout_s else None

        # Select an executor implementation based on executor_type
        executor_cls: Any
        exec_type = (self.executor_type or "thread").lower()
        if exec_type == "process":
            executor_cls = ProcessPoolExecutor
        elif exec_type == "interpreter" and InterpreterPoolExecutor is not None:
            executor_cls = InterpreterPoolExecutor
        else:
            # Default fallback to threads
            executor_cls = ThreadPoolExecutor

        # Create executor and execute payloads. NOTE: we intentionally do NOT
        # use a `with` context manager here. When a soft timeout triggers, a
        # context manager would still wait for all futures to finish on exit,
        # which defeats the point of a soft timeout and can make the run look
        # "stuck" in the UI.
        executor = executor_cls(max_workers=workers)
        future_to_index: Dict[Any, int] = {}
        try:
            # Submit all tasks immediately to preserve ordering
            future_to_index = {
                executor.submit(agent_fn, payload): idx
                for idx, payload in enumerate(payloads)
            }

            # If we have a deadline, iterate with an overall timeout.
            if deadline is not None:
                remaining = max(0.0, float(deadline - time.time()))
                try:
                    for future in as_completed(future_to_index, timeout=remaining):
                        idx = future_to_index[future]
                        try:
                            results[idx] = future.result()
                        except Exception as exc:  # pragma: no cover - robustness
                            results[idx] = {
                                "agent_index": idx,
                                "role": payloads[idx].get("role", "Unknown"),
                                "domain": payloads[idx].get("domain", "general"),
                                "status": "error",
                                "error": str(exc),
                            }
                        # Recompute remaining budget for as_completed
                        remaining = max(0.0, float(deadline - time.time()))
                        if remaining <= 0.0:
                            break
                except FuturesTimeoutError:
                    # Soft timeout hit; we'll mark unfinished tasks below.
                    pass
            else:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:  # pragma: no cover - robustness
                        results[idx] = {
                            "agent_index": idx,
                            "role": payloads[idx].get("role", "Unknown"),
                            "domain": payloads[idx].get("domain", "general"),
                            "status": "error",
                            "error": str(exc),
                        }
        finally:
            # Mark any unfinished tasks as timed out when a deadline is set.
            if deadline is not None:
                for f, j in future_to_index.items():
                    if results[j] is None:
                        try:
                            f.cancel()
                        except Exception:
                            pass
                        results[j] = {
                            "agent_index": j,
                            "role": payloads[j].get("role", "Unknown"),
                            "domain": payloads[j].get("domain", "general"),
                            "status": "timeout",
                        }

            # Do not wait for hung tasks; attempt to cancel futures.
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # pragma: no cover
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass

        return [r or {} for r in results]

    def _build_swarm_diagnostics(
        self,
        goal: str,
        run_id: str,
        agent_summaries: Sequence[Dict[str, Any]],
        mode: str,
        elapsed_seconds: float,
        curriculum_id: Optional[str],
        profile_name: Optional[str],
    ) -> Dict[str, Any]:
        """Aggregate high level diagnostics for the swarm."""
        if not self.enable_diagnostics:
            return {}

        diagnostics: Dict[str, Any] = {
            "goal": goal,
            "run_id": run_id,
            "mode": mode,
            "elapsed_seconds": elapsed_seconds,
            "agent_count": len(agent_summaries),
            "error_count": 0,
            "timeout_count": 0,
        }

        if curriculum_id is not None:
            diagnostics["curriculum_id"] = curriculum_id
        if profile_name is not None:
            diagnostics["profile_name"] = profile_name

        total_energy = 0.0
        total_tokens = 0.0

        for agent in agent_summaries:
            status = agent.get("status")
            if status == "error":
                diagnostics["error_count"] += 1
            if status == "timeout":
                diagnostics["timeout_count"] += 1

            energy = agent.get("energy") or agent.get("cost_energy")
            if isinstance(energy, (int, float)):
                total_energy += float(energy)

            tokens = agent.get("tokens_used") or agent.get("total_tokens")
            if isinstance(tokens, (int, float)):
                total_tokens += float(tokens)

        if total_energy > 0:
            diagnostics["total_energy"] = total_energy
        if total_tokens > 0:
            diagnostics["total_tokens"] = total_tokens

        # RYE style diagnostics (Option C friendly)
        if build_run_diagnostics is not None:
            try:
                traces: List[Dict[str, Any]] = []
                for agent in agent_summaries:
                    trace = agent.get("rye_trace") or agent.get(
                        "diagnostics", {}
                    ).get("rye_trace")
                    if isinstance(trace, list):
                        traces.extend(trace)

                if traces:
                    # Try to infer a dominant domain for this swarm
                    domains = {
                        agent.get("domain", "general") for agent in agent_summaries
                    }
                    if len(domains) == 1:
                        domain_for_rye = next(iter(domains))
                    else:
                        domain_for_rye = "mixed"

                    diagnostics["rye"] = build_run_diagnostics(
                        traces,
                        domain=domain_for_rye,
                    )
            except Exception:
                # RYE diagnostics are optional; never break the swarm
                pass

        # Stability kernel
        if self.enable_stability_kernel and self.stability_kernel is not None:
            try:
                diagnostics["stability_kernel"] = self.stability_kernel.analyze(
                    agent_summaries
                )
            except Exception:
                pass

        # Discovery manager
        if self.enable_discovery_manager and self.discovery_manager is not None:
            try:
                diagnostics["discoveries"] = self.discovery_manager.scan(
                    agent_summaries
                )
            except Exception:
                pass

        # MSIL v2
        if self.enable_msil and self.msil is not None:
            try:
                diagnostics["msil"] = self.msil.analyze_swarm(
                    goal=goal,
                    run_id=run_id,
                    agent_summaries=agent_summaries,
                )
            except Exception:
                pass

        # Protocol synthesizer
        if self.enable_protocol_synthesizer and self.protocol_synthesizer is not None:
            try:
                diagnostics["protocols"] = self.protocol_synthesizer.summarize_swarm(
                    goal=goal,
                    agent_summaries=agent_summaries,
                )
            except Exception:
                pass

        return diagnostics


# ---------------------------------------------------------------------------
# Global helper and shortcut
# ---------------------------------------------------------------------------

_global_orchestrator: Optional[SwarmOrchestrator] = None


def get_orchestrator(agent_fn: Optional[AgentFn] = None) -> SwarmOrchestrator:
    """Return a process wide SwarmOrchestrator instance.

    If agent_fn is provided it will be set on the orchestrator.
    """
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = SwarmOrchestrator(agent_fn=agent_fn)
    elif agent_fn is not None:
        _global_orchestrator.agent_fn = agent_fn
    return _global_orchestrator


def run_swarm(
    goal: str,
    base_preset: Dict[str, Any],
    swarm_size: int = 4,
    max_cycles: int = 6,
    mode: Optional[str] = None,
    agent_fn: Optional[AgentFn] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Module level shortcut compatible with older code paths."""
    orchestrator = get_orchestrator(agent_fn=agent_fn)
    return orchestrator.run_swarm(
        goal=goal,
        base_preset=base_preset,
        swarm_size=swarm_size,
        max_cycles=max_cycles,
        mode=mode,
        **kwargs,
    )
