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
from typing import Any, Callable, Dict, List, Optional

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

    You plug in three optional functions:

        literature_check_fn(h: HypothesisRecord) -> Dict
        critique_fn(h: HypothesisRecord) -> Dict
        data_check_fn(h: HypothesisRecord) -> Dict

    Each function should return a dict like:

        {
            "score": float between 0 and 1,
            "reasons": [list of short text reasons],
            "extra": {...optional debug info...}
        }

    The engine combines these into one final score and then:

      - validates the hypothesis if score >= validate_threshold
      - rejects it if score <= reject_threshold
      - leaves it pending if in between (inconclusive)
    """

    def __init__(
        self,
        hypothesis_manager: HypothesisManager,
        discovery_logger: DiscoveryLogger,
        literature_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        critique_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        data_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        validate_threshold: float = 0.7,
        reject_threshold: float = 0.3,
        run_id: Optional[str] = None,
    ) -> None:
        self.hypothesis_manager = hypothesis_manager
        self.discovery_logger = discovery_logger

        self.literature_check_fn = literature_check_fn
        self.critique_fn = critique_fn
        self.data_check_fn = data_check_fn

        self.validate_threshold = validate_threshold
        self.reject_threshold = reject_threshold
        self.run_id = run_id

    # main public entry point

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

    # internal workers

    def _verify_single(self, hyp: HypothesisRecord) -> VerificationResult:
        """
        Run all available checks on a single hypothesis and decide outcome.
        """
        pieces: List[Dict[str, Any]] = []
        reasons: List[str] = []
        extra_all: Dict[str, Any] = {}

        # literature check
        if self.literature_check_fn is not None:
            lit_res = self._safe_call("literature_check", self.literature_check_fn, hyp)
            pieces.append(lit_res)
            reasons.extend(lit_res.get("reasons", []))
            extra_all["literature_check"] = lit_res

        # critique check
        if self.critique_fn is not None:
            crit_res = self._safe_call("critique", self.critique_fn, hyp)
            pieces.append(crit_res)
            reasons.extend(crit_res.get("reasons", []))
            extra_all["critique"] = crit_res

        # data check
        if self.data_check_fn is not None:
            data_res = self._safe_call("data_check", self.data_check_fn, hyp)
            pieces.append(data_res)
            reasons.extend(data_res.get("reasons", []))
            extra_all["data_check"] = data_res

        final_score = self._combine_scores(pieces)
        status = "inconclusive"

        if final_score >= self.validate_threshold:
            status = "validated"
            updated = self.hypothesis_manager.validate_hypothesis(
                hyp.hypothesis_id,
                note=f"Validated by verification engine at { _utc_iso() } with score { final_score:.3f }",
            )
            self._log_validated(updated or hyp, final_score, reasons, extra_all)

        elif final_score <= self.reject_threshold:
            status = "rejected"
            updated = self.hypothesis_manager.reject_hypothesis(
                hyp.hypothesis_id,
                note=f"Rejected by verification engine at { _utc_iso() } with score { final_score:.3f }",
            )
            self._log_rejected(updated or hyp, final_score, reasons, extra_all)
        else:
            # still pending, but we log the attempt
            self._log_inconclusive(hyp, final_score, reasons, extra_all)

        return VerificationResult(
            hypothesis_id=hyp.hypothesis_id,
            status=status,
            score=final_score,
            reasons=reasons,
            extra=extra_all,
        )

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

    def _combine_scores(self, pieces: List[Dict[str, Any]]) -> float:
        """
        Combine scores from the different checks.

        Current strategy:
          - simple average of available scores.
          - if no checks are available, return 0.0.
        """
        scores = [float(p.get("score", 0.0)) for p in pieces if "score" in p]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    # logging helpers

    def _log_validated(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
    ) -> None:
        """
        Record a validated hypothesis in the discovery log.
        """
        description = (
            f"Validated hypothesis with final verification score {score:.3f}.\n\n"
            f"Reasons:\n"
            + "\n".join(f"- {r}" for r in reasons)
        )

        self.discovery_logger.log_validated_hypothesis(
            title=hyp.title,
            description=description,
            cycle_index=hyp.cycle_index,
            agent_role=hyp.agent_role,
            rye_before=hyp.rye_before,
            rye_after=hyp.rye_after,
            delta_r=hyp.delta_r,
            energy=hyp.energy,
            tags=(hyp.tags or []) + ["verified"],
            extra={
                "hypothesis_id": hyp.hypothesis_id,
                "verification_score": score,
                "checks": extra,
            },
        )

    def _log_rejected(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
    ) -> None:
        """
        Record a rejected hypothesis in the discovery log.
        """
        description = (
            f"Rejected hypothesis with final verification score {score:.3f}.\n\n"
            f"Reasons:\n"
            + "\n".join(f"- {r}" for r in reasons)
        )

        self.discovery_logger.log_rejected_hypothesis(
            title=hyp.title,
            description=description,
            cycle_index=hyp.cycle_index,
            agent_role=hyp.agent_role,
            tags=(hyp.tags or []) + ["verification_rejected"],
            extra={
                "hypothesis_id": hyp.hypothesis_id,
                "verification_score": score,
                "checks": extra,
            },
        )

    def _log_inconclusive(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
    ) -> None:
        """
        Record an inconclusive verification attempt.
        The hypothesis stays pending but we keep a trail.
        """
        description = (
            f"Inconclusive verification with score {score:.3f}. "
            f"Hypothesis remains pending.\n\n"
            f"Reasons:\n"
            + "\n".join(f"- {r}" for r in reasons)
        )

        self.discovery_logger.log_event(
            kind="verification_inconclusive",
            title=hyp.title,
            description=description,
            cycle_index=hyp.cycle_index,
            agent_role=hyp.agent_role,
            tags=(hyp.tags or []) + ["verification_inconclusive"],
            extra={
                "hypothesis_id": hyp.hypothesis_id,
                "verification_score": score,
                "checks": extra,
            },
        )
