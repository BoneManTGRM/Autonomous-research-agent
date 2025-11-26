from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import inspect
from datetime import datetime

try:
    from .tgrm_loop import TGRMLoop
except Exception:  # pragma: no cover
    TGRMLoop = None  # type: ignore


try:
    from .replay_buffer import ReplayBuffer  # type: ignore
except Exception:  # pragma: no cover
    ReplayBuffer = None  # type: ignore


@dataclass
class TwoStageParams:
    """Configuration for idea and verify stages."""

    idea_cycles: int = 3
    verify_cycles: int = 2

    idea_role: str = "researcher"
    critic_role: str = "critic"

    min_rye_for_verify: Optional[float] = 0.4
    min_hypotheses_for_verify: int = 1

    # Optional labels for future extensions
    idea_label: str = "idea"
    verify_label: str = "verify"


@dataclass
class AgentCriticConfig:
    """Config container for the AgentCritic controller."""

    mode: str = "two_stage"  # "single", "two_stage", "critic_only"
    domain: str = "general"

    hallmark_targets: List[str] = field(default_factory=list)
    curriculum_profile: Optional[str] = None

    two_stage: TwoStageParams = field(default_factory=TwoStageParams)

    # Optional fields used by CoreAgent or meta controller
    run_id: Optional[str] = None
    tgrm_config: Dict[str, Any] = field(default_factory=dict)


class AgentCriticController:
    """High level controller that runs paired agent and critic passes.

    This is a lightweight orchestrator that sits beside CoreAgent.

    Responsibilities:
        - Run idea stage cycles with a researcher style role.
        - Run verify stage cycles with a critic style role.
        - Pass through domain, hallmark, and curriculum hints into TGRMLoop
          when supported by the TGRMLoop.run_cycle signature.
        - Optionally record replay items in a ReplayBuffer when available.

    The controller is careful to work with both:
        - older TGRMLoop versions that do not know about stage or hallmark
        - newer longevity aware TGRMLoop versions that accept:
            stage, hallmark, subgoal, replay_buffer, curriculum_state
    """

    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[Any] = None,
        replay_buffer: Optional[Any] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = self._build_config(config or {})
        self.replay_buffer = replay_buffer

        tgrm_cfg = dict(self.config.tgrm_config or {})
        # Make sure ultra_speed and strict_pipeline defaults are enabled here.
        tgrm_cfg.setdefault("ultra_speed", True)
        tgrm_cfg.setdefault("strict_pipeline", True)

        if TGRMLoop is None:
            raise RuntimeError("TGRMLoop could not be imported in agent/critic.py")

        self.tgrm = TGRMLoop(memory_store=memory_store, config=tgrm_cfg, tools=tools)
        self._run_cycle_sig = inspect.signature(self.tgrm.run_cycle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_two_stage_episode(
        self,
        goal: str,
        start_cycle_index: int,
        *,
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        hallmark: Optional[str] = None,
        subgoal: Optional[str] = None,
        curriculum_state: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a full idea plus verify episode on a single goal.

        Returns a dict with:
            {
              "goal": ...,
              "domain": ...,
              "idea_stage": {...},
              "verify_stage": {...},
              "episode_summary": {...}
            }
        """
        cfg = self.config
        domain = cfg.domain

        hallmark_name = hallmark or self._select_hallmark(context=context)
        two_stage = cfg.two_stage

        # Stage 1: idea
        idea_result = self._run_stage(
            stage_label=two_stage.idea_label,
            role=two_stage.idea_role,
            goal=goal,
            domain=domain,
            hallmark=hallmark_name,
            subgoal=subgoal,
            source_controls=source_controls,
            pdf_bytes=pdf_bytes,
            biomarker_snapshot=biomarker_snapshot,
            curriculum_state=curriculum_state,
            start_cycle_index=start_cycle_index,
            max_cycles=two_stage.idea_cycles,
        )

        # Decide whether to run verify stage
        run_verify = self._should_run_verify(two_stage, idea_result)

        verify_result: Optional[Dict[str, Any]] = None
        if run_verify:
            verify_result = self._run_stage(
                stage_label=two_stage.verify_label,
                role=two_stage.critic_role,
                goal=goal,
                domain=domain,
                hallmark=hallmark_name,
                subgoal=subgoal,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                curriculum_state=curriculum_state,
                start_cycle_index=start_cycle_index + len(idea_result["cycles"]),
                max_cycles=two_stage.verify_cycles,
            )

        episode_summary = self._summarise_episode(goal, domain, hallmark_name, idea_result, verify_result)

        return {
            "goal": goal,
            "domain": domain,
            "hallmark": hallmark_name,
            "subgoal": subgoal,
            "run_id": cfg.run_id,
            "idea_stage": idea_result,
            "verify_stage": verify_result,
            "episode_summary": episode_summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_config(self, raw: Dict[str, Any]) -> AgentCriticConfig:
        """Build a strongly typed config object from a dict."""
        two_stage_raw = raw.get("two_stage") or {}
        two_stage = TwoStageParams(
            idea_cycles=int(two_stage_raw.get("idea_cycles", 3)),
            verify_cycles=int(two_stage_raw.get("verify_cycles", 2)),
            idea_role=str(two_stage_raw.get("idea_role", "researcher")),
            critic_role=str(two_stage_raw.get("critic_role", "critic")),
            min_rye_for_verify=two_stage_raw.get("min_rye_for_verify", 0.4),
            min_hypotheses_for_verify=int(two_stage_raw.get("min_hypotheses_for_verify", 1)),
            idea_label=str(two_stage_raw.get("idea_label", "idea")),
            verify_label=str(two_stage_raw.get("verify_label", "verify")),
        )

        cfg = AgentCriticConfig(
            mode=str(raw.get("mode", "two_stage")),
            domain=str(raw.get("domain", "general")),
            hallmark_targets=list(raw.get("hallmark_targets") or []),
            curriculum_profile=raw.get("curriculum_profile"),
            two_stage=two_stage,
            run_id=raw.get("run_id"),
            tgrm_config=dict(raw.get("tgrm_config") or {}),
        )
        return cfg

    def _select_hallmark(self, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Pick a hallmark target from config or context."""
        if context and context.get("hallmark"):
            return str(context["hallmark"])

        cfg_targets = self.config.hallmark_targets
        if cfg_targets:
            return str(cfg_targets[0])

        return None

    def _run_stage(
        self,
        stage_label: str,
        role: str,
        goal: str,
        domain: str,
        hallmark: Optional[str],
        subgoal: Optional[str],
        source_controls: Optional[Dict[str, bool]],
        pdf_bytes: Optional[bytes],
        biomarker_snapshot: Optional[Dict[str, Any]],
        curriculum_state: Optional[Dict[str, Any]],
        start_cycle_index: int,
        max_cycles: int,
    ) -> Dict[str, Any]:
        """Run a sequence of cycles for a given stage."""
        cycles: List[Dict[str, Any]] = []
        rye_values: List[float] = []
        best_cycle: Optional[Dict[str, Any]] = None
        best_rye: Optional[float] = None

        for offset in range(max_cycles):
            cycle_index = start_cycle_index + offset

            cycle_output = self._run_single_cycle(
                stage_label=stage_label,
                role=role,
                goal=goal,
                domain=domain,
                hallmark=hallmark,
                subgoal=subgoal,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                curriculum_state=curriculum_state,
                cycle_index=cycle_index,
            )

            summary = cycle_output.get("summary", {})
            cycles.append(cycle_output)

            rye = summary.get("RYE")
            if isinstance(rye, (int, float)):
                rye_float = float(rye)
                rye_values.append(rye_float)
                if best_rye is None or rye_float > best_rye:
                    best_rye = rye_float
                    best_cycle = cycle_output

            # Optional replay item logging
            self._log_replay_from_cycle(
                stage_label=stage_label,
                goal=goal,
                domain=domain,
                hallmark=hallmark,
                subgoal=subgoal,
                cycle_output=cycle_output,
            )

        avg_rye = float(sum(rye_values) / len(rye_values)) if rye_values else None

        return {
            "stage": stage_label,
            "role": role,
            "cycles": cycles,
            "avg_rye": avg_rye,
            "best_rye": best_rye,
            "best_cycle": best_cycle,
        }

    def _run_single_cycle(
        self,
        stage_label: str,
        role: str,
        goal: str,
        domain: str,
        hallmark: Optional[str],
        subgoal: Optional[str],
        source_controls: Optional[Dict[str, bool]],
        pdf_bytes: Optional[bytes],
        biomarker_snapshot: Optional[Dict[str, Any]],
        curriculum_state: Optional[Dict[str, Any]],
        cycle_index: int,
    ) -> Dict[str, Any]:
        """Single call wrapper around TGRMLoop.run_cycle with stage hints."""
        call_kwargs: Dict[str, Any] = {
            "goal": goal,
            "cycle_index": cycle_index,
            "role": role,
            "source_controls": source_controls,
            "pdf_bytes": pdf_bytes,
            "biomarker_snapshot": biomarker_snapshot,
            "domain": domain,
        }

        # Inject advanced args only if the TGRM implementation supports them.
        if "stage" in self._run_cycle_sig.parameters:
            call_kwargs["stage"] = stage_label
        if "hallmark" in self._run_cycle_sig.parameters:
            call_kwargs["hallmark"] = hallmark
        if "subgoal" in self._run_cycle_sig.parameters:
            call_kwargs["subgoal"] = subgoal
        if "replay_buffer" in self._run_cycle_sig.parameters:
            call_kwargs["replay_buffer"] = self.replay_buffer
        if "curriculum_state" in self._run_cycle_sig.parameters:
            call_kwargs["curriculum_state"] = curriculum_state

        result = self.tgrm.run_cycle(**call_kwargs)
        return result

    def _should_run_verify(self, two_stage: TwoStageParams, idea_result: Dict[str, Any]) -> bool:
        """Decide whether to run the critic stage."""
        if two_stage.verify_cycles <= 0:
            return False

        cycles = idea_result.get("cycles", [])
        if not cycles:
            return True

        # Collect RYE values and hypothesis counts across idea cycles
        rye_values: List[float] = []
        hypothesis_count = 0

        for cycle_output in cycles:
            summary = cycle_output.get("summary", {})
            rye = summary.get("RYE")
            if isinstance(rye, (int, float)):
                rye_values.append(float(rye))

            hypotheses = summary.get("hypotheses") or []
            if isinstance(hypotheses, list):
                hypothesis_count += len(hypotheses)

        avg_rye = float(sum(rye_values) / len(rye_values)) if rye_values else None

        # Check thresholds
        if avg_rye is not None and two_stage.min_rye_for_verify is not None:
            if avg_rye < two_stage.min_rye_for_verify:
                return False

        if hypothesis_count < two_stage.min_hypotheses_for_verify:
            return False

        return True

    def _log_replay_from_cycle(
        self,
        stage_label: str,
        goal: str,
        domain: str,
        hallmark: Optional[str],
        subgoal: Optional[str],
        cycle_output: Dict[str, Any],
    ) -> None:
        """Optional helper that forwards high level items into a replay buffer.

        This does not require a specific ReplayBuffer implementation.
        It will call add_item if available or add if that is present.
        """
        if self.replay_buffer is None:
            return

        summary = cycle_output.get("summary", {})
        log = cycle_output.get("log", {})

        rye = summary.get("RYE")
        delta_r = summary.get("delta_R")
        energy_e = summary.get("energy_E")

        hypotheses = summary.get("hypotheses") or []
        candidate_interventions = summary.get("candidate_interventions") or []
        citations = summary.get("citations") or []

        cycle_index = summary.get("cycle") or log.get("cycle")

        created_at = datetime.utcnow().isoformat() + "Z"

        # Use the top hypothesis as the main replay text if available.
        main_hypothesis_text: Optional[str] = None
        if hypotheses:
            first = hypotheses[0]
            main_hypothesis_text = (
                first.get("description")
                or first.get("text")
                or first.get("title")
            )

        item: Dict[str, Any] = {
            "item_id": f"{goal[:32]}::{cycle_index}::{stage_label}",
            "stage": stage_label,
            "goal": goal,
            "domain": domain,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "rye_score": rye,
            "delta_r": delta_r,
            "energy_cost": energy_e,
            "hypothesis_text": main_hypothesis_text,
            "all_hypotheses": hypotheses,
            "candidate_interventions": candidate_interventions,
            "source_citations": citations,
            "tags": [t for t in {domain, hallmark or "", stage_label} if t],
            "run_id": self.config.run_id,
            "cycle_index": cycle_index,
            "created_at": created_at,
            "equilibrium": summary.get("equilibrium"),
            "breakthrough": summary.get("breakthrough"),
        }

        try:
            if hasattr(self.replay_buffer, "add_item"):
                self.replay_buffer.add_item(item)
            elif hasattr(self.replay_buffer, "add"):
                self.replay_buffer.add(item)
        except Exception:
            # Replay logging must never break a cycle.
            return

    def _summarise_episode(
        self,
        goal: str,
        domain: str,
        hallmark: Optional[str],
        idea_result: Dict[str, Any],
        verify_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a compact summary for dashboards and reports."""
        idea_avg = idea_result.get("avg_rye")
        idea_best = idea_result.get("best_rye")

        verify_avg = verify_result.get("avg_rye") if verify_result else None
        verify_best = verify_result.get("best_rye") if verify_result else None

        # Final hypothesis set is taken from the best available cycle,
        # preferring critic cycles if they exist.
        final_cycle = None
        if verify_result and verify_result.get("best_cycle"):
            final_cycle = verify_result["best_cycle"]
        elif idea_result.get("best_cycle"):
            final_cycle = idea_result["best_cycle"]

        final_summary = (final_cycle or {}).get("summary", {}) if final_cycle else {}
        final_hypotheses = final_summary.get("hypotheses") or []
        final_breakthrough = final_summary.get("breakthrough") or {}

        return {
            "goal": goal,
            "domain": domain,
            "hallmark": hallmark,
            "idea_avg_rye": idea_avg,
            "idea_best_rye": idea_best,
            "verify_avg_rye": verify_avg,
            "verify_best_rye": verify_best,
            "final_hypotheses": final_hypotheses,
            "final_breakthrough": final_breakthrough,
        }
