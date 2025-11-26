"""SynergyAgent

Specialist for building and ranking intervention stacks and mechanism synergies.

Reparodynamics view:
    Biomarkers and mechanisms are the repair surface.
    Synergy stacks are candidate "moves" that potentially give higher RYE
    than any single mechanism alone.

Goals:
    - Take mechanisms and biomarkers and assemble small stacks.
    - Score stacks using RYE, stability, coverage, and diversity.
    - Emit high yield stacks into ReplayBuffer for faster learning.
    - Log synergy insights into MemoryStore without breaking anything.

This file is self contained and defensive. It only assumes that MemoryStore
may provide some of the following:
    - get_mechanisms_by_hallmark(hallmark)
    - get_best_mechanisms(hallmark, top_k)
    - add_synergy_insight(payload)
    - add_note(goal, text, role)

ReplayBuffer is optional and is used if available. It is expected to support
one of:
    - add_item(payload)
    - add_replay_item(payload)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import itertools
import math
import statistics


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Mechanism:
    """Normalized mechanism structure used for synergy search."""

    mech_id: Optional[str]
    name: str
    hallmark: Optional[str]
    rye_score: Optional[float]
    stability_index: Optional[float]
    domain: Optional[str]
    pathways: List[str]
    biomarkers: List[str]
    risk_flag: Optional[str]
    source_count: Optional[int]
    raw: Dict[str, Any]


@dataclass
class SynergyStack:
    """Candidate intervention stack made of several mechanisms."""

    stack_id: Optional[str]
    hallmark: Optional[str]
    mechanisms: List[Mechanism]
    stack_size: int
    avg_rye: Optional[float]
    min_rye: Optional[float]
    coverage_score: float
    diversity_score: float
    stability_score: float
    risk_level: str
    priority_score: float


class SynergyAgent:
    """Synergy specialist for longevity and multi mechanism runs.

    Typical use:
        agent = SynergyAgent(memory_store, config)
        result = agent.propose_synergies(
            goal=goal,
            hallmark="mitochondria",
            mechanisms=None,  # let agent pull from MemoryStore
            biomarker_focus=["LDL", "HDL"],
            run_id=current_run_id,
            cycle_index=cycle_index,
            replay_buffer=replay_buffer,
        )
    """

    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # How many mechanisms to consider per hallmark before stacking.
        self.max_mechanisms: int = int(self.config.get("synergy_max_mechanisms", 24))

        # Maximum size of a synergy stack.
        self.max_stack_size: int = int(self.config.get("synergy_max_stack_size", 3))

        # Number of stacks to emit per call.
        self.max_stacks_per_cycle: int = int(self.config.get("synergy_max_stacks_per_cycle", 10))

        # Minimum average RYE required before a stack is considered high priority.
        self.min_avg_rye_for_top: float = float(self.config.get("synergy_min_avg_rye_for_top", 0.5))

        # Risk penalty weight used when computing priority_score.
        self.risk_penalty_weight: float = float(self.config.get("synergy_risk_penalty_weight", 0.3))

        # Optional bonus for stacks that seem biomarker aligned.
        self.biomarker_alignment_bonus: float = float(
            self.config.get("synergy_biomarker_alignment_bonus", 0.1)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def propose_synergies(
        self,
        goal: str,
        hallmark: Optional[str],
        mechanisms: Optional[List[Dict[str, Any]]] = None,
        biomarker_focus: Optional[List[str]] = None,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        cycle_index: Optional[int] = None,
        replay_buffer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build and rank synergy stacks for the current goal and hallmark.

        Args:
            goal:
                Research goal associated with this synergy search.
            hallmark:
                Target hallmark, for example "mitochondria" or "senescence".
            mechanisms:
                Optional raw mechanism dicts. If None, the agent tries to
                pull them from MemoryStore.
            biomarker_focus:
                Optional list of biomarker names that are most relevant for
                the current run or snapshot.
            domain:
                Optional domain tag. Often "longevity".
            run_id:
                Optional run identifier.
            cycle_index:
                Optional cycle index in the run.
            replay_buffer:
                Optional ReplayBuffer compatible object.

        Returns:
            Dict with:
                - hallmark
                - stacks: list of SynergyStack as dicts
                - top_stacks: small list of highest priority stacks
                - summary: compact description for UI and reports
        """
        hallmark_name = hallmark or "unspecified"
        dom = (domain or "general").lower()

        raw_mechanisms = mechanisms or self._load_mechanisms_from_memory(hallmark_name)
        norm_mechs = self._normalize_mechanisms(raw_mechanisms, hallmark=hallmark_name, domain=dom)
        if not norm_mechs:
            summary = {
                "goal": goal,
                "hallmark": hallmark_name,
                "status": "no_mechanisms",
                "message": "No mechanisms available for synergy search in this cycle.",
            }
            return {
                "hallmark": hallmark_name,
                "stacks": [],
                "top_stacks": [],
                "summary": summary,
            }

        stacks = self._build_and_score_stacks(
            mechanisms=norm_mechs,
            biomarker_focus=biomarker_focus,
            hallmark=hallmark_name,
        )

        # Sort stacks by priority and take the top slice.
        stacks.sort(key=lambda s: s.priority_score, reverse=True)
        top_stacks = stacks[: self.max_stacks_per_cycle]

        timestamp = datetime.utcnow().isoformat() + "Z"
        summary = self._build_summary(
            goal=goal,
            hallmark=hallmark_name,
            stacks=stacks,
            top_stacks=top_stacks,
            biomarker_focus=biomarker_focus,
        )

        # Log synergy insight into MemoryStore if supported.
        self._log_synergy_insight(
            goal=goal,
            hallmark=hallmark_name,
            stacks=stacks,
            top_stacks=top_stacks,
            biomarker_focus=biomarker_focus,
            domain=dom,
            run_id=run_id,
            cycle_index=cycle_index,
            timestamp=timestamp,
            summary=summary,
        )

        # Emit replay items so that ReplayBuffer can learn to prioritize similar stacks.
        self._emit_replay_items(
            goal=goal,
            hallmark=hallmark_name,
            top_stacks=top_stacks,
            biomarker_focus=biomarker_focus,
            run_id=run_id,
            cycle_index=cycle_index,
            replay_buffer=replay_buffer,
        )

        return {
            "hallmark": hallmark_name,
            "stacks": [self._stack_to_dict(s) for s in stacks],
            "top_stacks": [self._stack_to_dict(s) for s in top_stacks],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Mechanism loading and normalization
    # ------------------------------------------------------------------
    def _load_mechanisms_from_memory(self, hallmark: str) -> List[Dict[str, Any]]:
        """Fetch mechanisms for this hallmark from MemoryStore if supported."""
        mechanisms: List[Dict[str, Any]] = []

        try:
            if hasattr(self.memory_store, "get_best_mechanisms"):
                items = self.memory_store.get_best_mechanisms(hallmark, top_k=self.max_mechanisms)  # type: ignore[attr-defined]
                if isinstance(items, list):
                    mechanisms.extend(items)
                    return mechanisms
        except Exception:
            mechanisms = []

        try:
            if hasattr(self.memory_store, "get_mechanisms_by_hallmark"):
                items = self.memory_store.get_mechanisms_by_hallmark(hallmark)  # type: ignore[attr-defined]
                if isinstance(items, list):
                    if len(items) > self.max_mechanisms:
                        mechanisms.extend(items[: self.max_mechanisms])
                    else:
                        mechanisms.extend(items)
        except Exception:
            pass

        return mechanisms

    def _normalize_mechanisms(
        self,
        mechanisms: List[Dict[str, Any]],
        hallmark: Optional[str],
        domain: Optional[str],
    ) -> List[Mechanism]:
        """Convert raw mechanism dicts into Mechanism objects."""
        norm: List[Mechanism] = []
        dom = domain or "general"

        for idx, m in enumerate(mechanisms):
            if not isinstance(m, dict):
                continue

            mech_id = m.get("id") or m.get("mechanism_id") or m.get("uid") or f"mech_{idx}"
            name = str(m.get("name") or m.get("title") or mech_id)

            hm = (
                m.get("hallmark")
                or m.get("hallmark_name")
                or m.get("hallmark_tag")
                or hallmark
            )

            rye_val = m.get("rye") or m.get("RYE") or m.get("rye_score")
            try:
                rye_score = float(rye_val) if rye_val is not None else None
            except Exception:
                rye_score = None

            stability_val = m.get("stability_index") or m.get("stability")
            try:
                stability_index = float(stability_val) if stability_val is not None else None
            except Exception:
                stability_index = None

            source_count_val = m.get("source_count") or m.get("citations") or m.get("citation_count")
            try:
                source_count = int(source_count_val) if source_count_val is not None else None
            except Exception:
                source_count = None

            pathways = self._ensure_str_list(
                m.get("pathways")
                or m.get("pathway_tags")
                or m.get("mechanism_tags")
                or []
            )

            biomarkers = self._ensure_str_list(
                m.get("biomarkers")
                or m.get("biomarker_tags")
                or m.get("marker_tags")
                or []
            )

            risk_flag = (
                m.get("risk_flag")
                or m.get("risk_level")
                or m.get("safety_flag")
            )

            norm.append(
                Mechanism(
                    mech_id=str(mech_id),
                    name=name,
                    hallmark=str(hm) if hm is not None else None,
                    rye_score=rye_score,
                    stability_index=stability_index,
                    domain=dom,
                    pathways=pathways,
                    biomarkers=biomarkers,
                    risk_flag=str(risk_flag) if risk_flag is not None else None,
                    source_count=source_count,
                    raw=m,
                )
            )

        return norm

    def _ensure_str_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    # ------------------------------------------------------------------
    # Stack construction and scoring
    # ------------------------------------------------------------------
    def _build_and_score_stacks(
        self,
        mechanisms: List[Mechanism],
        biomarker_focus: Optional[List[str]],
        hallmark: str,
    ) -> List[SynergyStack]:
        """Create and score stacks of size 2..max_stack_size."""
        if len(mechanisms) < 2:
            return []

        stacks: List[SynergyStack] = []
        focus_set = set(biomarker_focus or [])

        # Precompute base weights so we can reuse them for each combination.
        mech_weights = {
            m.mech_id: self._mechanism_base_weight(m)
            for m in mechanisms
        }

        # Build combinations of mechanisms.
        for size in range(2, min(self.max_stack_size, len(mechanisms)) + 1):
            for combo in itertools.combinations(mechanisms, size):
                mechs = list(combo)

                avg_rye, min_rye = self._stack_rye_stats(mechs)
                coverage, diversity = self._stack_coverage_and_diversity(mechs, focus_set)
                stability_score = self._stack_stability(mechs)
                risk_level = self._stack_risk_level(mechs)

                base_weight = sum(mech_weights.get(m.mech_id, 0.5) for m in mechs) / float(size or 1)

                priority = self._stack_priority(
                    avg_rye=avg_rye,
                    min_rye=min_rye,
                    coverage=coverage,
                    diversity=diversity,
                    stability=stability_score,
                    risk_level=risk_level,
                    base_weight=base_weight,
                )

                stack = SynergyStack(
                    stack_id=None,
                    hallmark=hallmark,
                    mechanisms=mechs,
                    stack_size=size,
                    avg_rye=avg_rye,
                    min_rye=min_rye,
                    coverage_score=coverage,
                    diversity_score=diversity,
                    stability_score=stability_score,
                    risk_level=risk_level,
                    priority_score=priority,
                )
                stacks.append(stack)

        return stacks

    def _mechanism_base_weight(self, mech: Mechanism) -> float:
        """Base utility weight from RYE, stability, and citations."""
        score = 0.5

        if mech.rye_score is not None:
            # Map RYE range [0,1] into [0.2, 1.1] to avoid zero.
            score += 0.7 * max(0.0, min(1.0, mech.rye_score))

        if mech.stability_index is not None:
            score += 0.3 * max(0.0, min(1.0, mech.stability_index))

        if mech.source_count is not None and mech.source_count > 0:
            bonus = min(0.2, 0.02 * math.log(mech.source_count + 1.0))
            score += bonus

        return float(max(0.1, score))

    def _stack_rye_stats(self, mechanisms: List[Mechanism]) -> Tuple[Optional[float], Optional[float]]:
        rye_vals: List[float] = []
        for m in mechanisms:
            if m.rye_score is not None:
                rye_vals.append(float(m.rye_score))

        if not rye_vals:
            return None, None

        avg_rye = sum(rye_vals) / len(rye_vals)
        min_rye = min(rye_vals)
        return float(avg_rye), float(min_rye)

    def _stack_coverage_and_diversity(
        self,
        mechanisms: List[Mechanism],
        biomarker_focus: set,
    ) -> Tuple[float, float]:
        """Compute coverage over biomarkers and diversity over pathways."""
        all_biomarkers: List[str] = []
        all_pathways: List[str] = []

        for m in mechanisms:
            all_biomarkers.extend(m.biomarkers)
            all_pathways.extend(m.pathways)

        bio_set = set(all_biomarkers)
        path_set = set(all_pathways)

        # Coverage score: how many unique biomarkers are touched, plus small
        # bonus for overlap with biomarker_focus.
        coverage = float(len(bio_set))
        if biomarker_focus:
            overlap = len(bio_set & biomarker_focus)
            if coverage > 0:
                coverage += self.biomarker_alignment_bonus * (overlap / coverage)

        # Diversity score: more distinct pathways gives higher diversity.
        diversity = float(len(path_set))

        # Normalize both scores to roughly [0, 1] so they can plug into
        # a combined priority formula without dominating everything.
        coverage_norm = min(1.0, coverage / 8.0)
        diversity_norm = min(1.0, diversity / 8.0)

        return coverage_norm, diversity_norm

    def _stack_stability(self, mechanisms: List[Mechanism]) -> float:
        vals: List[float] = []
        for m in mechanisms:
            if m.stability_index is not None:
                vals.append(float(m.stability_index))
        if not vals:
            return 0.5
        return float(max(0.0, min(1.0, statistics.mean(vals))))

    def _stack_risk_level(self, mechanisms: List[Mechanism]) -> str:
        risk_flags = [m.risk_flag for m in mechanisms if m.risk_flag]

        if not risk_flags:
            return "low_risk"

        # Simple heuristic:
        risk_text = " ".join(risk_flags).lower()
        if any("black box" in r or "unknown" in r for r in risk_flags):
            return "unknown_risk"
        if any("serious" in r or "severe" in r or "high" in r for r in risk_flags):
            return "high_risk"
        if any("moderate" in r or "medium" in r for r in risk_flags):
            return "moderate_risk"
        if "contraindication" in risk_text or "unsafe" in risk_text:
            return "high_risk"

        return "moderate_risk"

    def _stack_priority(
        self,
        avg_rye: Optional[float],
        min_rye: Optional[float],
        coverage: float,
        diversity: float,
        stability: float,
        risk_level: str,
        base_weight: float,
    ) -> float:
        """Combine signals into a single priority score on a rough 0..1 scale."""
        score = base_weight

        if avg_rye is not None:
            score += 0.8 * max(0.0, min(1.0, avg_rye))

        if min_rye is not None:
            score += 0.3 * max(0.0, min(1.0, min_rye))

        score += 0.4 * coverage
        score += 0.4 * diversity
        score += 0.3 * max(0.0, min(1.0, stability))

        if risk_level == "high_risk":
            score -= self.risk_penalty_weight
        elif risk_level == "unknown_risk":
            score -= 0.15 * self.risk_penalty_weight

        score = max(0.0, min(2.5, score))
        return float(score / 2.5)

    # ------------------------------------------------------------------
    # Logging and replay
    # ------------------------------------------------------------------
    def _build_summary(
        self,
        goal: str,
        hallmark: str,
        stacks: List[SynergyStack],
        top_stacks: List[SynergyStack],
        biomarker_focus: Optional[List[str]],
    ) -> Dict[str, Any]:
        if not stacks:
            return {
                "goal": goal,
                "hallmark": hallmark,
                "status": "no_stacks",
                "message": "No synergy stacks could be constructed for the current mechanisms.",
            }

        avg_priority = statistics.mean(s.priority_score for s in stacks)
        best_priority = max(s.priority_score for s in stacks)

        high_quality = [s for s in stacks if s.avg_rye is not None and s.avg_rye >= self.min_avg_rye_for_top]
        high_quality_count = len(high_quality)

        summary = {
            "goal": goal,
            "hallmark": hallmark,
            "status": "ok",
            "stacks_considered": len(stacks),
            "top_stacks": len(top_stacks),
            "high_quality_stacks": high_quality_count,
            "avg_priority_score": avg_priority,
            "best_priority_score": best_priority,
            "biomarker_focus": list(biomarker_focus or []),
        }

        return summary

    def _log_synergy_insight(
        self,
        goal: str,
        hallmark: str,
        stacks: List[SynergyStack],
        top_stacks: List[SynergyStack],
        biomarker_focus: Optional[List[str]],
        domain: str,
        run_id: Optional[str],
        cycle_index: Optional[int],
        timestamp: str,
        summary: Dict[str, Any],
    ) -> None:
        """Record synergy insight into MemoryStore without hard dependency."""
        insight_payload = {
            "goal": goal,
            "hallmark": hallmark,
            "domain": domain,
            "run_id": run_id,
            "cycle_index": cycle_index,
            "timestamp": timestamp,
            "biomarker_focus": list(biomarker_focus or []),
            "summary": summary,
            "top_stacks": [self._stack_to_dict(s) for s in top_stacks],
        }

        # Try dedicated method first.
        try:
            if hasattr(self.memory_store, "add_synergy_insight"):
                self.memory_store.add_synergy_insight(insight_payload)  # type: ignore[attr-defined]
                return
        except Exception:
            pass

        # Fallback: store a note so insight is not lost.
        try:
            text = (
                f"[SynergyAgent] Considered {summary.get('stacks_considered', 0)} stacks "
                f"for hallmark '{hallmark}'. Top stacks: {summary.get('top_stacks', 0)}. "
                f"High quality stacks: {summary.get('high_quality_stacks', 0)}."
            )
            if hasattr(self.memory_store, "add_note"):
                self.memory_store.add_note(goal, text, role="synergy_agent")  # type: ignore[attr-defined]
        except Exception:
            pass

    def _emit_replay_items(
        self,
        goal: str,
        hallmark: str,
        top_stacks: List[SynergyStack],
        biomarker_focus: Optional[List[str]],
        run_id: Optional[str],
        cycle_index: Optional[int],
        replay_buffer: Optional[Any],
    ) -> None:
        """Push high priority stacks into ReplayBuffer for faster learning."""
        if replay_buffer is None or not top_stacks:
            return

        focus_set = set(biomarker_focus or [])
        timestamp = datetime.utcnow().isoformat() + "Z"

        for idx, stack in enumerate(top_stacks):
            mech_names = [m.name for m in stack.mechanisms]
            mech_ids = [m.mech_id for m in stack.mechanisms]
            all_biomarkers: List[str] = []
            for m in stack.mechanisms:
                all_biomarkers.extend(m.biomarkers)
            all_markers = sorted(set(all_biomarkers))
            overlap = sorted(set(all_markers) & focus_set)

            hypothesis_text = (
                f"Synergy stack for goal '{goal}', hallmark '{hallmark}' combining mechanisms: "
                + ", ".join(mech_names)
                + ". Coverage score "
                + f"{stack.coverage_score:.2f}, diversity score {stack.diversity_score:.2f}, "
                + f"stability score {stack.stability_score:.2f}, risk_level={stack.risk_level}."
            )

            replay_payload = {
                "item_id": None,
                "hallmark": hallmark,
                "stage": "idea",
                "mechanism_chain": mech_ids,
                "biomarker_pattern": {
                    "markers": all_markers,
                    "focus_overlap": overlap,
                },
                "hypothesis_text": hypothesis_text,
                "rye_score": stack.avg_rye,
                "energy_cost": None,
                "decision": "pending",
                "reason": "high_priority_synergy_stack",
                "source_citations": [],
                "tags": [
                    "synergy_stack",
                    "longevity",
                    hallmark,
                ],
                "created_at": timestamp,
                "run_id": run_id,
                "cycle_index": cycle_index,
            }

            try:
                if hasattr(replay_buffer, "add_item"):
                    replay_buffer.add_item(replay_payload)  # type: ignore[attr-defined]
                elif hasattr(replay_buffer, "add_replay_item"):
                    replay_buffer.add_replay_item(replay_payload)  # type: ignore[attr-defined]
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _stack_to_dict(self, stack: SynergyStack) -> Dict[str, Any]:
        return {
            "stack_id": stack.stack_id,
            "hallmark": stack.hallmark,
            "stack_size": stack.stack_size,
            "avg_rye": stack.avg_rye,
            "min_rye": stack.min_rye,
            "coverage_score": stack.coverage_score,
            "diversity_score": stack.diversity_score,
            "stability_score": stack.stability_score,
            "risk_level": stack.risk_level,
            "priority_score": stack.priority_score,
            "mechanisms": [
                {
                    "mech_id": m.mech_id,
                    "name": m.name,
                    "hallmark": m.hallmark,
                    "rye_score": m.rye_score,
                    "stability_index": m.stability_index,
                    "domain": m.domain,
                    "pathways": m.pathways,
                    "biomarkers": m.biomarkers,
                    "risk_flag": m.risk_flag,
                    "source_count": m.source_count,
                }
                for m in stack.mechanisms
            ],
        }
