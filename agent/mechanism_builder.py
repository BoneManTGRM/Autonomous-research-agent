"""
Mechanism builder for the Autonomous Research Agent.

Purpose
-------
Turn raw cycle outputs into compact, scored "mechanisms" that are easy to:
    - rank
    - promote to gold memory
    - replay for future cycles

This module does not call external tools.
It only consumes what TGRMLoop and the verification pipeline already produced:
    - hypotheses
    - citations
    - candidate interventions
    - biomarker snapshots
    - RYE and breakthrough signals

Design
------
Mechanism: a small dict that captures:
    - id
    - goal, domain
    - hallmark and subgoal (if available)
    - stage ("idea" or "verify" or None)
    - core_chain
        text chain summarizing hypothesis -> biomarkers -> interventions
    - hypothesis_refs
    - intervention_refs
    - citation_refs
    - scores:
        - mechanism_score
        - support_score
        - novelty_hint
        - rye_value
        - breakthrough_hint

This file is optional and tolerant:
    - If memory_store lacks methods, promotion is skipped.
    - If replay_buffer is None, replay routing is skipped.
    - If hallmark_profiles is provided it is used to tag mechanisms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import uuid


class MechanismBuilder:
    """
    Build and promote mechanisms from a single cycle log.

    Parameters
    ----------
    memory_store:
        Shared MemoryStore used by CoreAgent.
        Optional methods used if available:
            - promote_to_gold(hallmark, mechanism_obj)
            - save_scratch(hallmark, note_bundle)
            - attach_replay_pointer(mechanism_id, replay_item_id)
            - add_mechanism(goal, mechanism_obj)
    replay_buffer:
        Optional ReplayBuffer instance.
        Expected minimal API:
            - add_item(item_dict) -> replay_id
            - register_chain(chain_dict) [optional]
    hallmark_profiles:
        Optional HallmarkProfiles instance used to normalize hallmark names
        and attach tags. Expected methods if present:
            - normalize_name(hallmark) -> str
            - describe(hallmark) -> dict
    config:
        Optional dict for tuning:
            - min_rye_for_promotion: float
            - min_mechanism_score: float
            - max_mechanisms_per_cycle: int
            - prefer_verify_stage: bool
    """

    def __init__(
        self,
        memory_store: Any,
        replay_buffer: Optional[Any] = None,
        hallmark_profiles: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.replay_buffer = replay_buffer
        self.hallmark_profiles = hallmark_profiles
        self.config = config or {}

        self.min_rye_for_promotion: float = float(
            self.config.get("min_rye_for_promotion", 0.55)
        )
        self.min_mechanism_score: float = float(
            self.config.get("min_mechanism_score", 0.6)
        )
        self.max_mechanisms_per_cycle: int = int(
            self.config.get("max_mechanisms_per_cycle", 5)
        )
        self.prefer_verify_stage: bool = bool(
            self.config.get("prefer_verify_stage", True)
        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def build_mechanisms_for_cycle(
        self,
        cycle_log: Dict[str, Any],
        *,
        hallmark: Optional[str] = None,
        subgoal: Optional[str] = None,
        stage: Optional[str] = None,
        verification_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Inputs
        ------
        cycle_log:
            Machine facing log from TGRMLoop.run_cycle.
        hallmark:
            Optional hallmark focus passed from CoreAgent.
        subgoal:
            Optional sub target, for example "mito_membrane_potential".
        stage:
            Optional stage tag, for example "idea" or "verify".
        verification_summary:
            Optional dict from VerificationPipeline.verify_cycle with:
                - verification_score
                - novelty_score
                - disco_confidence
                - flags
        Returns
        -------
        Dict with:
            {
              "mechanisms": [ ... ],
              "promoted": [ids],
              "scratch_only": [ids],
              "skipped_reason": str or None,
            }
        """
        goal = cycle_log.get("goal", "")
        domain = cycle_log.get("domain", "general")
        rye_value = float(cycle_log.get("RYE") or 0.0)

        hallmark_norm = self._normalize_hallmark(hallmark)
        stage_tag = stage or cycle_log.get("stage")

        # Extract raw parts
        hypotheses = cycle_log.get("hypotheses", []) or []
        candidate_interventions = cycle_log.get("candidate_interventions", []) or []
        citations = cycle_log.get("citations", []) or []
        biomarker_snapshot = cycle_log.get("biomarker_snapshot")

        # Short circuit if nothing to work with
        if not hypotheses and not candidate_interventions and not biomarker_snapshot:
            return {
                "mechanisms": [],
                "promoted": [],
                "scratch_only": [],
                "skipped_reason": "no_mechanism_material",
            }

        # Build mechanism candidates
        mechanisms = self._build_candidates(
            goal=goal,
            domain=domain,
            hallmark=hallmark_norm,
            subgoal=subgoal,
            stage=stage_tag,
            rye_value=rye_value,
            hypotheses=hypotheses,
            candidate_interventions=candidate_interventions,
            citations=citations,
            biomarker_snapshot=biomarker_snapshot,
            verification_summary=verification_summary,
            cycle_index=cycle_log.get("cycle"),
        )

        if not mechanisms:
            return {
                "mechanisms": [],
                "promoted": [],
                "scratch_only": [],
                "skipped_reason": "no_valid_mechanisms",
            }

        # Sort mechanisms by score
        mechanisms.sort(
            key=lambda m: float(m.get("scores", {}).get("mechanism_score", 0.0)),
            reverse=True,
        )
        mechanisms = mechanisms[: self.max_mechanisms_per_cycle]

        # Promote and route to replay
        promoted_ids, scratch_ids = self._promote_and_route(
            goal=goal,
            mechanisms=mechanisms,
            hallmark=hallmark_norm,
        )

        return {
            "mechanisms": mechanisms,
            "promoted": promoted_ids,
            "scratch_only": scratch_ids,
            "skipped_reason": None,
        }

    # ------------------------------------------------------------------
    # Candidate builder
    # ------------------------------------------------------------------
    def _build_candidates(
        self,
        goal: str,
        domain: str,
        hallmark: Optional[str],
        subgoal: Optional[str],
        stage: Optional[str],
        rye_value: float,
        hypotheses: List[Dict[str, Any]],
        candidate_interventions: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        biomarker_snapshot: Optional[Dict[str, Any]],
        verification_summary: Optional[Dict[str, Any]],
        cycle_index: Optional[int],
    ) -> List[Dict[str, Any]]:
        mechanisms: List[Dict[str, Any]] = []

        # Basic verification hints
        verification_score = 0.0
        novelty_score = 0.0
        disco_confidence = 0.0
        if verification_summary:
            verification_score = float(
                verification_summary.get("verification_score") or 0.0
            )
            novelty_score = float(verification_summary.get("novelty_score") or 0.0)
            disco_confidence = float(
                verification_summary.get("disco_confidence") or 0.0
            )

        # Map hypotheses to interventions and biomarkers
        # We use a light pairing: each hypothesis gets the strongest looking
        # interventions and whatever biomarkers are present.
        for idx, hyp in enumerate(hypotheses):
            hyp_text = (hyp.get("description") or hyp.get("title") or "").strip()
            if not hyp_text:
                continue

            hyp_conf = hyp.get("confidence")
            hyp_tags = list(hyp.get("tags") or [])

            # Pair a small set of interventions
            linked_interventions = self._select_linked_interventions(
                hyp=hyp,
                interventions=candidate_interventions,
                max_count=3,
            )

            # Biomarker snapshot is shared across mechanisms for this cycle
            biomarkers = self._extract_biomarkers(biomarker_snapshot)

            mechanism_id = self._make_mechanism_id(
                goal=goal,
                cycle_index=cycle_index,
                hyp_index=idx,
            )

            chain_text = self._build_chain_text(
                hypothesis_text=hyp_text,
                interventions=linked_interventions,
                biomarkers=biomarkers,
                hallmark=hallmark,
            )

            support_score = self._compute_support_score(
                interventions=linked_interventions,
                citations=citations,
                biomarkers=biomarkers,
            )

            mechanism_score = self._score_mechanism(
                rye_value=rye_value,
                verification_score=verification_score,
                novelty_score=novelty_score,
                disco_confidence=disco_confidence,
                support_score=support_score,
                hypothesis_confidence=hyp_conf,
                stage=stage,
                domain=domain,
            )

            mechanism = {
                "id": mechanism_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "goal": goal,
                "domain": domain or "general",
                "hallmark": hallmark,
                "subgoal": subgoal,
                "stage": stage,
                "core_chain": chain_text,
                "hypothesis": {
                    "id": hyp.get("id"),
                    "text": hyp_text,
                    "confidence": hyp_conf,
                    "tags": hyp_tags,
                },
                "interventions": linked_interventions,
                "biomarkers": biomarkers,
                "citations_summary": self._compact_citation_view(citations),
                "scores": {
                    "mechanism_score": mechanism_score,
                    "support_score": support_score,
                    "novelty_hint": novelty_score,
                    "rye_value": rye_value,
                    "verification_score": verification_score,
                    "disco_confidence": disco_confidence,
                },
            }

            # Attach hallmark metadata if profiles are provided
            if hallmark:
                mechanism["hallmark_meta"] = self._describe_hallmark(hallmark)

            mechanisms.append(mechanism)

        # If there are no hypotheses but we have interventions plus biomarkers,
        # we still try to create at least one mechanism
        if not mechanisms and (candidate_interventions or biomarker_snapshot):
            mech_id = self._make_mechanism_id(goal=goal, cycle_index=cycle_index, hyp_index=None)
            chain_text = self._build_chain_text(
                hypothesis_text="Implicit mechanism from interventions and biomarkers",
                interventions=candidate_interventions[:3],
                biomarkers=self._extract_biomarkers(biomarker_snapshot),
                hallmark=hallmark,
            )
            support_score = self._compute_support_score(
                interventions=candidate_interventions,
                citations=citations,
                biomarkers=self._extract_biomarkers(biomarker_snapshot),
            )
            mechanism_score = self._score_mechanism(
                rye_value=rye_value,
                verification_score=verification_score,
                novelty_score=novelty_score,
                disco_confidence=disco_confidence,
                support_score=support_score,
                hypothesis_confidence=None,
                stage=stage,
                domain=domain,
            )
            mechanisms.append(
                {
                    "id": mech_id,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "goal": goal,
                    "domain": domain or "general",
                    "hallmark": hallmark,
                    "subgoal": subgoal,
                    "stage": stage,
                    "core_chain": chain_text,
                    "hypothesis": None,
                    "interventions": candidate_interventions[:3],
                    "biomarkers": self._extract_biomarkers(biomarker_snapshot),
                    "citations_summary": self._compact_citation_view(citations),
                    "scores": {
                        "mechanism_score": mechanism_score,
                        "support_score": support_score,
                        "novelty_hint": novelty_score,
                        "rye_value": rye_value,
                        "verification_score": verification_score,
                        "disco_confidence": disco_confidence,
                    },
                }
            )

        return mechanisms

    # ------------------------------------------------------------------
    # Promotion and replay
    # ------------------------------------------------------------------
    def _promote_and_route(
        self,
        goal: str,
        mechanisms: List[Dict[str, Any]],
        hallmark: Optional[str],
    ) -> Tuple[List[str], List[str]]:
        promoted_ids: List[str] = []
        scratch_ids: List[str] = []

        for mech in mechanisms:
            mech_id = mech.get("id") or str(uuid.uuid4())
            mech["id"] = mech_id

            scores = mech.get("scores", {}) or {}
            mech_score = float(scores.get("mechanism_score") or 0.0)
            rye_value = float(scores.get("rye_value") or 0.0)

            # Route into replay first for fast reuse
            replay_id = self._send_to_replay(mech)

            # Decide promotion vs scratch
            if mech_score >= self.min_mechanism_score and rye_value >= self.min_rye_for_promotion:
                self._promote_to_gold(goal=goal, hallmark=hallmark, mechanism=mech)
                promoted_ids.append(mech_id)
                if replay_id is not None:
                    self._attach_replay_pointer(mechanism_id=mech_id, replay_item_id=replay_id)
            else:
                self._save_scratch(goal=goal, hallmark=hallmark, mechanism=mech)
                scratch_ids.append(mech_id)

            # Optional generic "add_mechanism" hook
            self._generic_add_mechanism(goal=goal, mechanism=mech)

        return promoted_ids, scratch_ids

    def _send_to_replay(self, mechanism: Dict[str, Any]) -> Optional[str]:
        if self.replay_buffer is None:
            return None
        try:
            item = {
                "kind": "mechanism",
                "mechanism_id": mechanism.get("id"),
                "goal": mechanism.get("goal"),
                "domain": mechanism.get("domain"),
                "hallmark": mechanism.get("hallmark"),
                "stage": mechanism.get("stage"),
                "core_chain": mechanism.get("core_chain"),
                "scores": mechanism.get("scores"),
                "tags": ["mechanism", mechanism.get("domain")],
                "created_at": mechanism.get("created_at"),
            }
            replay_id = None
            if hasattr(self.replay_buffer, "add_item"):
                replay_id = self.replay_buffer.add_item(item)  # type: ignore[attr-defined]
            elif hasattr(self.replay_buffer, "register_chain"):
                replay_id = self.replay_buffer.register_chain(item)  # type: ignore[attr-defined]
            return replay_id
        except Exception:
            return None

    def _promote_to_gold(
        self,
        goal: str,
        hallmark: Optional[str],
        mechanism: Dict[str, Any],
    ) -> None:
        try:
            if hallmark and hasattr(self.memory_store, "promote_to_gold"):
                self.memory_store.promote_to_gold(hallmark, mechanism)  # type: ignore[attr-defined]
            elif hasattr(self.memory_store, "promote_to_gold"):
                self.memory_store.promote_to_gold("general", mechanism)  # type: ignore[attr-defined]
        except Exception:
            # If promotion fails the mechanism is still returned to caller
            pass

    def _save_scratch(
        self,
        goal: str,
        hallmark: Optional[str],
        mechanism: Dict[str, Any],
    ) -> None:
        try:
            bundle = {
                "goal": goal,
                "hallmark": hallmark or "general",
                "mechanism": mechanism,
            }
            if hasattr(self.memory_store, "save_scratch"):
                self.memory_store.save_scratch(hallmark or "general", bundle)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _attach_replay_pointer(
        self,
        mechanism_id: str,
        replay_item_id: Optional[str],
    ) -> None:
        if not replay_item_id:
            return
        try:
            if hasattr(self.memory_store, "attach_replay_pointer"):
                self.memory_store.attach_replay_pointer(mechanism_id, replay_item_id)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _generic_add_mechanism(self, goal: str, mechanism: Dict[str, Any]) -> None:
        try:
            if hasattr(self.memory_store, "add_mechanism"):
                self.memory_store.add_mechanism(goal, mechanism)  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_mechanism(
        self,
        rye_value: float,
        verification_score: float,
        novelty_score: float,
        disco_confidence: float,
        support_score: float,
        hypothesis_confidence: Optional[float],
        stage: Optional[str],
        domain: str,
    ) -> float:
        base = 0.0

        base += max(0.0, min(1.0, rye_value)) * 0.25
        base += max(0.0, min(1.0, verification_score)) * 0.25
        base += max(0.0, min(1.0, novelty_score)) * 0.20
        base += max(0.0, min(1.0, disco_confidence)) * 0.15
        base += max(0.0, min(1.0, support_score)) * 0.10

        if isinstance(hypothesis_confidence, (int, float)):
            base += max(0.0, min(1.0, float(hypothesis_confidence))) * 0.05

        stage_lower = (stage or "").lower()
        domain_lower = (domain or "general").lower()

        if self.prefer_verify_stage and stage_lower == "verify":
            base *= 1.08
        elif stage_lower == "idea":
            base *= 0.95

        if domain_lower == "longevity":
            base *= 1.03
        elif domain_lower == "math":
            base *= 1.05

        return max(0.0, min(1.0, base))

    def _compute_support_score(
        self,
        interventions: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        biomarkers: List[Dict[str, Any]],
    ) -> float:
        if not interventions and not citations and not biomarkers:
            return 0.0

        iv_term = min(1.0, len(interventions) / 3.0)
        cite_term = min(1.0, len({c.get("url") for c in citations if c.get("url")}) / 10.0)
        biom_term = min(1.0, len(biomarkers) / 4.0)

        score = iv_term * 0.4 + cite_term * 0.35 + biom_term * 0.25
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _normalize_hallmark(self, hallmark: Optional[str]) -> Optional[str]:
        if not hallmark:
            return None
        if self.hallmark_profiles is not None and hasattr(
            self.hallmark_profiles, "normalize_name"
        ):
            try:
                return self.hallmark_profiles.normalize_name(hallmark)  # type: ignore[attr-defined]
            except Exception:
                return hallmark
        return hallmark

    def _describe_hallmark(self, hallmark: str) -> Dict[str, Any]:
        if self.hallmark_profiles is not None and hasattr(
            self.hallmark_profiles, "describe"
        ):
            try:
                desc = self.hallmark_profiles.describe(hallmark)  # type: ignore[attr-defined]
                if isinstance(desc, dict):
                    return desc
            except Exception:
                return {}
        return {}

    def _select_linked_interventions(
        self,
        hyp: Dict[str, Any],
        interventions: List[Dict[str, Any]],
        max_count: int,
    ) -> List[Dict[str, Any]]:
        if not interventions:
            return []

        hyp_text = (hyp.get("description") or hyp.get("title") or "").lower()
        if not hyp_text:
            return interventions[:max_count]

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for iv in interventions:
            label = (iv.get("label") or iv.get("title") or "").lower()
            if not label:
                scored.append((0, iv))
                continue
            score = 0
            for token in ["mitochond", "senesc", "nad", "mtor", "autophagy", "inflamm"]:
                if token in hyp_text and token in label:
                    score += 2
            if any(word in label for word in ["trial", "study", "randomized"]):
                score += 1
            scored.append((score, iv))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [iv for _, iv in scored[:max_count]]

    def _extract_biomarkers(
        self,
        biomarker_snapshot: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not biomarker_snapshot or not isinstance(biomarker_snapshot, dict):
            return []

        biomarkers: List[Dict[str, Any]] = []
        for key, value in biomarker_snapshot.items():
            biomarkers.append(
                {
                    "name": str(key),
                    "value": value,
                }
            )
        return biomarkers

    def _compact_citation_view(
        self,
        citations: List[Dict[str, Any]],
        max_count: int = 10,
    ) -> List[Dict[str, Any]]:
        view: List[Dict[str, Any]] = []
        seen = set()
        for c in citations:
            title = (c.get("title") or "").strip()
            url = c.get("url")
            key = (title, url)
            if key in seen:
                continue
            seen.add(key)
            view.append(
                {
                    "title": title,
                    "url": url,
                    "source": c.get("source"),
                }
            )
            if len(view) >= max_count:
                break
        return view

    def _build_chain_text(
        self,
        hypothesis_text: str,
        interventions: List[Dict[str, Any]],
        biomarkers: List[Dict[str, Any]],
        hallmark: Optional[str],
    ) -> str:
        parts: List[str] = []

        if hallmark:
            parts.append(f"Hallmark: {hallmark}")

        parts.append(f"Hypothesis: {hypothesis_text}")

        if interventions:
            labels = [
                (iv.get("label") or iv.get("title") or "").strip()
                for iv in interventions
                if (iv.get("label") or iv.get("title"))
            ]
            labels = [l for l in labels if l]
            if labels:
                parts.append("Interventions: " + "; ".join(labels))

        if biomarkers:
            names = [b.get("name") for b in biomarkers if b.get("name")]
            if names:
                parts.append("Biomarkers involved: " + ", ".join(sorted(set(names))))

        return " | ".join(parts)

    def _make_mechanism_id(
        self,
        goal: str,
        cycle_index: Optional[int],
        hyp_index: Optional[int],
    ) -> str:
        base = f"mech_{uuid.uuid4().hex[:8]}"
        if cycle_index is not None:
            base += f"_c{cycle_index}"
        if hyp_index is not None:
            base += f"_h{hyp_index}"
        return base
