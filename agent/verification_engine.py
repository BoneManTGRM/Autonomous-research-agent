"""
verification_engine.py

Verification engine for the Autonomous Research Agent.

Purpose
    Take promising hypotheses and push them through a deeper
    verification process, including:
      - extra literature checks
      - consistency checks against memory
      - cross agent critique (if swarm is available)
      - optional code or data based checks
      - RYE and energy aware scoring
      - long run Tier classification hints (Tier 1 / 2 / 3 candidates)

This module does not call tools directly. Instead it expects you
to give it callable hooks so it stays decoupled from specific APIs.

Typical usage:

    from agent.verification_engine import VerificationEngine
    from agent.hypothesis_manager import HypothesisManager
    from agent.discovery_log import DiscoveryLogger

    hm = HypothesisManager(run_id="run_001")
    dl = DiscoveryLogger.default(run_id="run_001")

    ve = VerificationEngine(
        hypothesis_manager=hm,
        discovery_logger=dl,
        literature_check_fn=do_literature_check,
        critique_fn=do_swarm_critique,
        data_check_fn=run_data_checks,
    )

    ve.verify_pending(limit=3)

You can call verify_pending on a schedule from engine_worker.py,
for example once per week.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hypothesis_manager import HypothesisManager, HypothesisRecord  # type: ignore
from .discovery_log import DiscoveryLogger  # type: ignore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class VerificationResult:
    hypothesis_id: str
    status: str  # "validated", "rejected", or "inconclusive"
    score: float
    reasons: List[str]
    extra: Dict[str, Any]


class VerificationEngine:
    """
    Orchestrates deeper verification passes over hypotheses.

    You plug in three core optional functions:

        literature_check_fn(h: HypothesisRecord) -> Dict
        critique_fn(h: HypothesisRecord) -> Dict
        data_check_fn(h: HypothesisRecord) -> Dict

    And you may plug in advanced hooks (all optional, fully backwards compatible):

        structural_check_fn(h: HypothesisRecord) -> Dict
        plausibility_check_fn(h: HypothesisRecord) -> Dict
        consensus_check_fn(h: HypothesisRecord) -> Dict

    Each function should return a dict like:

        {
            "score": float between 0 and 1,
            "reasons": [list of short text reasons],
            "extra": {...optional debug info...}
        }

    The engine combines these into one final score using:
      - weighted aggregation
      - RYE and energy aware scoring
      - internal consistency heuristics

    Then it:
      - validates the hypothesis if score >= validate_threshold
      - rejects it if score <= reject_threshold
      - leaves it pending if in between (inconclusive)

    This module is deliberately "maxed out" in logic, but optional hooks
    mean you can start simple and grow into the full power over time.
    """

    def __init__(
        self,
        hypothesis_manager: HypothesisManager,
        discovery_logger: DiscoveryLogger,
        literature_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        critique_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        data_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        # Advanced optional hooks (all default to None, so existing code keeps working)
        structural_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        plausibility_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        consensus_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        # Thresholds
        validate_threshold: float = 0.7,
        reject_threshold: float = 0.3,
        run_id: Optional[str] = None,
    ) -> None:
        self.hypothesis_manager = hypothesis_manager
        self.discovery_logger = discovery_logger

        # Core hooks
        self.literature_check_fn = literature_check_fn
        self.critique_fn = critique_fn
        self.data_check_fn = data_check_fn

        # Advanced hooks
        self.structural_check_fn = structural_check_fn
        self.plausibility_check_fn = plausibility_check_fn
        self.consensus_check_fn = consensus_check_fn

        # Score thresholds
        self.validate_threshold = validate_threshold
        self.reject_threshold = reject_threshold
        self.run_id = run_id

    # ------------------------------------------------------------------
    # main public entry point
    # ------------------------------------------------------------------
    def verify_pending(self, limit: Optional[int] = None) -> List[VerificationResult]:
        """
        Run verification on pending hypotheses.

        If limit is provided, verify only up to that many items.
        Returns a list of VerificationResult objects.
        """
        pending = self.hypothesis_manager.list_hypotheses("pending")
        if limit is not None:
            pending = pending[:limit]

        results: List[VerificationResult] = []
        for hyp in pending:
            result = self._verify_single(hyp)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # internal workers
    # ------------------------------------------------------------------
    def _verify_single(self, hyp: HypothesisRecord) -> VerificationResult:
        """
        Run all available checks on a single hypothesis and decide outcome.

        This now includes:
          - user-provided checks (literature / critique / data / structural / plausibility / consensus)
          - internal RYE and energy heuristic scoring
        """
        pieces: List[Dict[str, Any]] = []
        reasons: List[str] = []
        extra_all: Dict[str, Any] = {}

        # 1) literature check
        if self.literature_check_fn is not None:
            lit_res = self._safe_call("literature_check", self.literature_check_fn, hyp)
            pieces.append(self._tagged_piece("literature", lit_res))
            reasons.extend(lit_res.get("reasons", []))
            extra_all["literature_check"] = lit_res

        # 2) critique check (swarm or specialist critic role)
        if self.critique_fn is not None:
            crit_res = self._safe_call("critique", self.critique_fn, hyp)
            pieces.append(self._tagged_piece("critique", crit_res))
            reasons.extend(crit_res.get("reasons", []))
            extra_all["critique"] = crit_res

        # 3) data / code check
        if self.data_check_fn is not None:
            data_res = self._safe_call("data_check", self.data_check_fn, hyp)
            pieces.append(self._tagged_piece("data", data_res))
            reasons.extend(data_res.get("reasons", []))
            extra_all["data_check"] = data_res

        # 4) structural / formal check (optional)
        if self.structural_check_fn is not None:
            struct_res = self._safe_call("structural", self.structural_check_fn, hyp)
            pieces.append(self._tagged_piece("structural", struct_res))
            reasons.extend(struct_res.get("reasons", []))
            extra_all["structural_check"] = struct_res

        # 5) plausibility check (optional: domain-aware, safety-aware)
        if self.plausibility_check_fn is not None:
            plaus_res = self._safe_call("plausibility", self.plausibility_check_fn, hyp)
            pieces.append(self._tagged_piece("plausibility", plaus_res))
            reasons.extend(plaus_res.get("reasons", []))
            extra_all["plausibility_check"] = plaus_res

        # 6) consensus/alignment check (optional: multi-agent agreement)
        if self.consensus_check_fn is not None:
            cons_res = self._safe_call("consensus", self.consensus_check_fn, hyp)
            pieces.append(self._tagged_piece("consensus", cons_res))
            reasons.extend(cons_res.get("reasons", []))
            extra_all["consensus_check"] = cons_res

        # 7) internal RYE + energy consistency scoring
        rye_piece = self._compute_rye_energy_piece(hyp)
        pieces.append(rye_piece)
        reasons.extend(rye_piece.get("reasons", []))
        extra_all["rye_energy"] = rye_piece

        # Combine to final score
        final_score, component_scores = self._combine_scores(pieces, hyp=hyp)
        status = "inconclusive"

        # Decide and propagate back into hypothesis manager + discovery log
        if final_score >= self.validate_threshold:
            status = "validated"
            note = (
                f"Validated by verification engine at {_utc_iso()} "
                f"with score {final_score:.3f}"
            )
            updated = self.hypothesis_manager.validate_hypothesis(
                hyp.hypothesis_id,
                note=note,
            )
            self._log_validated(updated or hyp, final_score, reasons, extra_all, component_scores=component_scores)

        elif final_score <= self.reject_threshold:
            status = "rejected"
            note = (
                f"Rejected by verification engine at {_utc_iso()} "
                f"with score {final_score:.3f}"
            )
            updated = self.hypothesis_manager.reject_hypothesis(
                hyp.hypothesis_id,
                note=note,
            )
            self._log_rejected(updated or hyp, final_score, reasons, extra_all, component_scores=component_scores)
        else:
            # still pending, but we log the attempt
            self._log_inconclusive(hyp, final_score, reasons, extra_all, component_scores=component_scores)

        extra_all["component_scores"] = component_scores

        return VerificationResult(
            hypothesis_id=hyp.hypothesis_id,
            status=status,
            score=final_score,
            reasons=reasons,
            extra=extra_all,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _safe_call(
        self,
        name: str,
        fn: Callable[[HypothesisRecord], Dict[str, Any]],
        hyp: HypothesisRecord,
    ) -> Dict[str, Any]:
        """
        Call a check function and handle any errors.
        Ensures we always get a dict with score and reasons.
        """
        try:
            res = fn(hyp) or {}
        except Exception as e:
            return {
                "score": 0.0,
                "reasons": [f"{name} check failed with error: {e}"],
                "extra": {"error": str(e)},
            }

        score = float(res.get("score", 0.0))
        reasons = res.get("reasons", [])
        extra = res.get("extra", {})

        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        return {
            "score": score,
            "reasons": reasons,
            "extra": extra,
        }

    @staticmethod
    def _tagged_piece(kind: str, res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attach a 'kind' label to each score piece so the combiner
        can weight different verification streams.
        """
        out = dict(res)
        out.setdefault("kind", kind)
        return out

    def _compute_rye_energy_piece(self, hyp: HypothesisRecord) -> Dict[str, Any]:
        """
        Build an internal score piece based on RYE and energy metadata
        on the hypothesis itself.

        Uses:
          - delta_R (improvement)
          - energy
          - RYE (delta_R / energy)
          - rye_before / rye_after when present

        This is where Reparodynamics is explicitly wired into verification.
        High RYE, sustained RYE, and positive delta_R give strong boosts
        toward Tier 2 / Tier 3 style discoveries.
        """
        reasons: List[str] = []
        score = 0.5  # neutral default

        # Extract fields defensively
        delta_r = getattr(hyp, "delta_r", None)
        energy = getattr(hyp, "energy", None)
        rye_before = getattr(hyp, "rye_before", None)
        rye_after = getattr(hyp, "rye_after", None)

        # Compute RYE if possible
        rye_val: Optional[float] = None
        if delta_r is not None and energy not in (None, 0):
            try:
                rye_val = float(delta_r) / float(energy)
            except Exception:
                rye_val = None

        # 1) Improvement based scoring
        if delta_r is not None:
            try:
                dr = float(delta_r)
            except Exception:
                dr = 0.0

            if dr <= 0:
                score = 0.10
                reasons.append("RYE: hypothesis associated with non-positive delta_R, penalized.")
            elif dr < 0.01:
                score = 0.45
                reasons.append("RYE: very small positive delta_R, slight boost.")
            elif dr < 0.05:
                score = 0.65
                reasons.append("RYE: modest positive delta_R, moderate boost.")
            elif dr < 0.20:
                score = 0.80
                reasons.append("RYE: strong positive delta_R, high boost.")
            else:
                score = 0.92
                reasons.append("RYE: very strong delta_R, Tier 3 candidate pattern.")

        # 2) Energy efficiency refinement (if RYE is available)
        if rye_val is not None:
            try:
                rv = float(rye_val)
            except Exception:
                rv = 0.0

            # For a research agent, RYE in ~0.02–0.20 is realistic high-efficiency zone.
            if rv <= 0:
                score = min(score, 0.15)
                reasons.append("Energy: non-positive RYE, downgraded.")
            elif rv < 0.01:
                reasons.append("Energy: low but positive RYE, weak efficiency.")
            elif rv < 0.05:
                score = max(score, 0.70)
                reasons.append("Energy: moderate RYE, consistent with useful repair.")
            elif rv < 0.20:
                score = max(score, 0.88)
                reasons.append("Energy: high RYE, strong efficiency pattern.")
            else:
                # extremely high RYE looks suspicious; clip a bit
                score = min(max(score, 0.80), 0.93)
                reasons.append("Energy: extremely high RYE, clipped to avoid overconfidence.")

        # 3) RYE before/after direction check
        if rye_before is not None and rye_after is not None:
            try:
                rb = float(rye_before)
                ra = float(rye_after)
                d_rye = ra - rb
            except Exception:
                d_rye = 0.0

            if d_rye > 0:
                score = max(score, 0.75)
                reasons.append("RYE trend: rye_after > rye_before, consistent improvement.")
            elif d_rye < 0:
                score = min(score, 0.40)
                reasons.append("RYE trend: rye_after < rye_before, possible overfitting or regression.")
            else:
                reasons.append("RYE trend: no measurable change, neutral effect.")

        extra = {
            "delta_r": delta_r,
            "energy": energy,
            "rye_computed": rye_val,
            "rye_before": rye_before,
            "rye_after": rye_after,
        }

        return {
            "kind": "rye_energy",
            "score": float(score),
            "reasons": reasons,
            "extra": extra,
        }

    # ------------------------------------------------------------------
    # score combiner
    # ------------------------------------------------------------------
    def _combine_scores(
        self,
        pieces: List[Dict[str, Any]],
        hyp: HypothesisRecord,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Combine scores from the different checks.

        Strategy:
          - attach a weight by kind
          - normalize by sum of used weights
          - fall back to neutral if no scores

        Current weights are tuned to favor:
          - literature + critique + data for core validity
          - rye_energy for reparodynamic strength
          - structural / plausibility / consensus as strong modifiers
        """
        if not pieces:
            return 0.0, {}

        # Base weight map (can be adjusted over time)
        base_weights: Dict[str, float] = {
            "literature": 0.22,
            "critique": 0.18,
            "data": 0.18,
            "structural": 0.10,
            "plausibility": 0.08,
            "consensus": 0.10,
            "rye_energy": 0.14,
        }

        # If this is explicitly a math/theory hypothesis, increase structural weight
        domain = getattr(hyp, "domain", None) or getattr(hyp, "domain_tag", "general")
        if str(domain).lower() in ("math", "theory"):
            base_weights["structural"] = 0.18

        scores_by_kind: Dict[str, float] = {}
        used_weights: Dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for p in pieces:
            raw_score = float(p.get("score", 0.0))
            kind = str(p.get("kind", "generic"))
            weight = base_weights.get(kind, 0.05)

            scores_by_kind[kind] = raw_score

            weighted_sum += raw_score * weight
            total_weight += weight
            used_weights[kind] = used_weights.get(kind, 0.0) + weight

        if total_weight <= 0:
            final_score = 0.0
        else:
            final_score = weighted_sum / total_weight

        # Tiny safety clamp
        final_score = max(0.0, min(1.0, final_score))

        return final_score, scores_by_kind

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def _log_validated(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
        component_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a validated hypothesis in the discovery log.

        High score hypotheses can be tagged as Tier 2/Tier 3 candidates.
        """
        component_scores = component_scores or {}
        tier_tags: List[str] = []

        if score >= 0.9:
            tier_tags.append("tier3_candidate")
        elif score >= 0.8:
            tier_tags.append("tier2_candidate")

        description_lines = [
            f"Validated hypothesis with final verification score {score:.3f}.",
            "",
            "Component scores:",
        ]
        for k, v in sorted(component_scores.items()):
            description_lines.append(f"- {k}: {v:.3f}")
        description_lines.append("")
        description_lines.append("Reasons:")
        description_lines.extend(f"- {r}" for r in reasons)

        description = "\n".join(description_lines)

        existing_tags = list(hyp.tags or [])
        combined_tags = existing_tags + ["verified"] + tier_tags

        self.discovery_logger.log_validated_hypothesis(
            title=hyp.title,
            description=description,
            cycle_index=hyp.cycle_index,
            agent_role=hyp.agent_role,
            rye_before=hyp.rye_before,
            rye_after=hyp.rye_after,
            delta_r=hyp.delta_r,
            energy=hyp.energy,
            tags=combined_tags,
            extra={
                "hypothesis_id": hyp.hypothesis_id,
                "verification_score": score,
                "component_scores": component_scores,
                "checks": extra,
            },
        )

    def _log_rejected(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
        component_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a rejected hypothesis in the discovery log.
        """
        component_scores = component_scores or {}

        description_lines = [
            f"Rejected hypothesis with final verification score {score:.3f}.",
            "",
            "Component scores:",
        ]
        for k, v in sorted(component_scores.items()):
            description_lines.append(f"- {k}: {v:.3f}")
        description_lines.append("")
        description_lines.append("Reasons:")
        description_lines.extend(f"- {r}" for r in reasons)

        description = "\n".join(description_lines)

        existing_tags = list(hyp.tags or [])
        combined_tags = existing_tags + ["verification_rejected"]

        self.discovery_logger.log_rejected_hypothesis(
            title=hyp.title,
            description=description,
            cycle_index=hyp.cycle_index,
            agent_role=hyp.agent_role,
            tags=combined_tags,
            extra={
                "hypothesis_id": hyp.hypothesis_id,
                "verification_score": score,
                "component_scores": component_scores,
                "checks": extra,
            },
        )

    def _log_inconclusive(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
        component_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record an inconclusive verification attempt.
        The hypothesis stays pending but we keep a trail.
        """
        component_scores = component_scores or {}

        description_lines = [
            f"Inconclusive verification with score {score:.3f}. Hypothesis remains pending.",
            "",
            "Component scores:",
        ]
        for k, v in sorted(component_scores.items()):
            description_lines.append(f"- {k}: {v:.3f}")
        description_lines.append("")
        description_lines.append("Reasons:")
        description_lines.extend(f"- {r}" for r in reasons)

        description = "\n".join(description_lines)

        existing_tags = list(hyp.tags or [])
        combined_tags = existing_tags + ["verification_inconclusive"]

        self.discovery_logger.log_event(
            kind="verification_inconclusive",
            title=hyp.title,
            description=description,
            cycle_index=hyp.cycle_index,
            agent_role=hyp.agent_role,
            tags=combined_tags,
            extra={
                "hypothesis_id": hyp.hypothesis_id,
                "verification_score": score,
                "component_scores": component_scores,
                "checks": extra,
            },
        )
