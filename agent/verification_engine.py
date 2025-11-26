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
      - intelligence profile aware weighting
      - history based learning over repeated verification attempts
      - equilibrium and stability aware scoring

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
        # optional extra hooks:
        # structural_check_fn=do_structural_check,
        # plausibility_check_fn=do_plausibility_check,
        # consensus_check_fn=do_consensus_check,
        # memory_consistency_fn=do_memory_consistency_check,
        # equilibrium_check_fn=do_equilibrium_check,
        # intelligence_profile=intelligence_profile_dict,
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
        memory_consistency_fn(h: HypothesisRecord) -> Dict
        equilibrium_check_fn(h: HypothesisRecord) -> Dict

    Each function should return a dict like:

        {
            "score": float between 0 and 1,
            "reasons": [list of short text reasons],
            "extra": {...optional debug info...}
        }

    The engine combines these into one final score using:
      - weighted aggregation
      - RYE and energy aware scoring
      - intelligence-profile aware weighting
      - history based learning and trend analysis
      - internal consistency heuristics

    Then it:
      - validates the hypothesis if score >= validate_threshold (adaptive)
      - rejects it if score <= reject_threshold (adaptive)
      - leaves it pending if in between (inconclusive)

    This module is deliberately maxed out in logic, but all hooks
    and intelligence integration are optional, so you can start
    simple and grow into the full power over time.
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
        # New optional hooks for learning and stability
        memory_consistency_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        equilibrium_check_fn: Optional[Callable[[HypothesisRecord], Dict[str, Any]]] = None,
        # Thresholds (base, before adaptive tuning)
        validate_threshold: float = 0.7,
        reject_threshold: float = 0.3,
        # Intelligence profile (from intelligence_profiles.py) is optional
        intelligence_profile: Optional[Dict[str, Any]] = None,
        # Optional domain and runtime hints
        domain: Optional[str] = None,
        runtime_profile: Optional[str] = None,
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

        # New learning and stability hooks
        self.memory_consistency_fn = memory_consistency_fn
        self.equilibrium_check_fn = equilibrium_check_fn

        # Score thresholds (base values)
        self.validate_threshold = validate_threshold
        self.reject_threshold = reject_threshold

        # Intelligence and context metadata
        self.intelligence_profile = intelligence_profile or {}
        self.domain = domain
        self.runtime_profile = runtime_profile
        self.run_id = run_id

        # In process verification history for learning and trend analysis
        # Maps hypothesis_id -> list of past scores (most recent at end)
        self._history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # main public entry points
    # ------------------------------------------------------------------
    def verify_pending(self, limit: Optional[int] = None) -> List[VerificationResult]:
        """
        Run verification on pending hypotheses.

        If limit is provided, verify only up to that many items.
        The pending list is first prioritized for learning speed,
        so high impact and high information hypotheses are handled first.
        Returns a list of VerificationResult objects.
        """
        pending = self.hypothesis_manager.list_hypotheses("pending")

        # Prioritize for learning speed: high impact, strong RYE, recent items first
        pending = self._prioritize_pending(pending)

        if limit is not None:
            pending = pending[:limit]

        results: List[VerificationResult] = []
        for hyp in pending:
            result = self._verify_single(hyp)
            results.append(result)

        return results

    def verify_one(self, hypothesis_id: str) -> Optional[VerificationResult]:
        """
        Convenience helper to verify a single hypothesis by id.

        If the hypothesis is not pending or does not exist, returns None.
        """
        record = self.hypothesis_manager.get_hypothesis(hypothesis_id)
        if record is None:
            return None
        if getattr(record, "status", "pending") != "pending":
            return None
        return self._verify_single(record)

    # ------------------------------------------------------------------
    # internal workers
    # ------------------------------------------------------------------
    def _verify_single(self, hyp: HypothesisRecord) -> VerificationResult:
        """
        Run all available checks on a single hypothesis and decide outcome.

        Learning speed upgrades:
          - compute fast internal RYE and history score first
          - use those to attempt a low cost early decision
          - only call expensive external hooks if needed

        The full flow includes:
          - internal RYE and energy heuristic scoring
          - history based trend scoring
          - user-provided checks (literature, critique, data, structural, plausibility, consensus)
          - memory consistency and equilibrium checks (optional)
          - intelligence-profile guided weighting and thresholds
        """
        reasons: List[str] = []
        extra_all: Dict[str, Any] = {}
        pieces: List[Dict[str, Any]] = []

        # 0) Resolve adaptive thresholds for this hypothesis
        local_validate, local_reject = self._adapt_thresholds(hyp)
        extra_all["thresholds"] = {
            "base_validate": self.validate_threshold,
            "base_reject": self.reject_threshold,
            "local_validate": local_validate,
            "local_reject": local_reject,
        }

        # 1) Always compute internal RYE and history pieces first
        rye_piece = self._compute_rye_energy_piece(hyp)
        pieces.append(rye_piece)
        reasons.extend(rye_piece.get("reasons", []))
        extra_all["rye_energy"] = rye_piece

        history_piece = self._compute_history_piece(hyp)
        if history_piece is not None:
            pieces.append(history_piece)
            reasons.extend(history_piece.get("reasons", []))
            extra_all["history"] = history_piece

        # 2) Try a fast path decision using only internal pieces
        pre_score, pre_component_scores = self._combine_scores(pieces, hyp=hyp)
        extra_all["fast_path_score"] = pre_score
        extra_all["fast_path_components"] = pre_component_scores

        fast_margin = 0.15  # how far beyond threshold we require for early decision
        fast_status: Optional[str] = None

        if pre_score >= (local_validate + fast_margin):
            fast_status = "validated"
        elif pre_score <= (local_reject - fast_margin):
            fast_status = "rejected"

        if fast_status is not None:
            # Fast path: skip external costly checks, but still update history and logs
            self._update_history(hyp, pre_score)
            tier_label = self._infer_tier_label(pre_score)
            extra_all["tier_label"] = tier_label
            extra_all["fast_path_used"] = True

            if fast_status == "validated":
                note = (
                    f"Fast path validation by verification engine at {_utc_iso()} "
                    f"with score {pre_score:.3f}"
                )
                updated = self.hypothesis_manager.validate_hypothesis(
                    hyp.hypothesis_id,
                    note=note,
                )
                self._log_validated(
                    updated or hyp,
                    pre_score,
                    reasons,
                    extra_all,
                    component_scores=pre_component_scores,
                    thresholds={"validate": local_validate, "reject": local_reject},
                )
            else:
                note = (
                    f"Fast path rejection by verification engine at {_utc_iso()} "
                    f"with score {pre_score:.3f}"
                )
                updated = self.hypothesis_manager.reject_hypothesis(
                    hyp.hypothesis_id,
                    note=note,
                )
                self._log_rejected(
                    updated or hyp,
                    pre_score,
                    reasons,
                    extra_all,
                    component_scores=pre_component_scores,
                    thresholds={"validate": local_validate, "reject": local_reject},
                )

            extra_all["component_scores"] = pre_component_scores
            extra_all["final_score"] = pre_score
            extra_all["status"] = fast_status

            return VerificationResult(
                hypothesis_id=hyp.hypothesis_id,
                status=fast_status,
                score=pre_score,
                reasons=reasons,
                extra=extra_all,
            )

        # 3) Full verification path - external checks only if fast path did not decide
        extra_all["fast_path_used"] = False

        # literature check
        if self.literature_check_fn is not None:
            lit_res = self._safe_call("literature_check", self.literature_check_fn, hyp)
            pieces.append(self._tagged_piece("literature", lit_res))
            reasons.extend(lit_res.get("reasons", []))
            extra_all["literature_check"] = lit_res

        # critique check
        if self.critique_fn is not None:
            crit_res = self._safe_call("critique", self.critique_fn, hyp)
            pieces.append(self._tagged_piece("critique", crit_res))
            reasons.extend(crit_res.get("reasons", []))
            extra_all["critique"] = crit_res

        # data / code check
        if self.data_check_fn is not None:
            data_res = self._safe_call("data_check", self.data_check_fn, hyp)
            pieces.append(self._tagged_piece("data", data_res))
            reasons.extend(data_res.get("reasons", []))
            extra_all["data_check"] = data_res

        # structural check
        if self.structural_check_fn is not None:
            struct_res = self._safe_call("structural_check", self.structural_check_fn, hyp)
            pieces.append(self._tagged_piece("structural", struct_res))
            reasons.extend(struct_res.get("reasons", []))
            extra_all["structural_check"] = struct_res

        # plausibility check
        if self.plausibility_check_fn is not None:
            plaus_res = self._safe_call("plausibility_check", self.plausibility_check_fn, hyp)
            pieces.append(self._tagged_piece("plausibility", plaus_res))
            reasons.extend(plaus_res.get("reasons", []))
            extra_all["plausibility_check"] = plaus_res

        # consensus check
        if self.consensus_check_fn is not None:
            cons_res = self._safe_call("consensus_check", self.consensus_check_fn, hyp)
            pieces.append(self._tagged_piece("consensus", cons_res))
            reasons.extend(cons_res.get("reasons", []))
            extra_all["consensus_check"] = cons_res

        # memory consistency check
        if self.memory_consistency_fn is not None:
            mem_res = self._safe_call("memory_consistency_check", self.memory_consistency_fn, hyp)
            pieces.append(self._tagged_piece("memory", mem_res))
            reasons.extend(mem_res.get("reasons", []))
            extra_all["memory_consistency_check"] = mem_res

        # equilibrium / stability check
        if self.equilibrium_check_fn is not None:
            eq_res = self._safe_call("equilibrium_check", self.equilibrium_check_fn, hyp)
            pieces.append(self._tagged_piece("equilibrium", eq_res))
            reasons.extend(eq_res.get("reasons", []))
            extra_all["equilibrium_check"] = eq_res

        # 4) Combine all scores
        final_score, component_scores = self._combine_scores(pieces, hyp=hyp)

        # Update learning history
        self._update_history(hyp, final_score)

        # Tier label from verification score
        tier_label = self._infer_tier_label(final_score)
        extra_all["tier_label"] = tier_label

        status = "inconclusive"

        # Decide and propagate back into hypothesis manager and discovery log
        if final_score >= local_validate:
            status = "validated"
            note = (
                f"Validated by verification engine at {_utc_iso()} "
                f"with score {final_score:.3f}"
            )
            updated = self.hypothesis_manager.validate_hypothesis(
                hyp.hypothesis_id,
                note=note,
            )
            self._log_validated(
                updated or hyp,
                final_score,
                reasons,
                extra_all,
                component_scores=component_scores,
                thresholds={"validate": local_validate, "reject": local_reject},
            )
        elif final_score <= local_reject:
            status = "rejected"
            note = (
                f"Rejected by verification engine at {_utc_iso()} "
                f"with score {final_score:.3f}"
            )
            updated = self.hypothesis_manager.reject_hypothesis(
                hyp.hypothesis_id,
                note=note,
            )
            self._log_rejected(
                updated or hyp,
                final_score,
                reasons,
                extra_all,
                component_scores=component_scores,
                thresholds={"validate": local_validate, "reject": local_reject},
            )
        else:
            self._log_inconclusive(
                hyp,
                final_score,
                reasons,
                extra_all,
                component_scores=component_scores,
                thresholds={"validate": local_validate, "reject": local_reject},
            )

        extra_all["component_scores"] = component_scores
        extra_all["final_score"] = final_score
        extra_all["status"] = status

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
                "reasons": [f"{name} failed with error: {e}"],
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

    # ------------------------------------------------------------------
    # pending prioritization for learning speed
    # ------------------------------------------------------------------
    def _prioritize_pending(
        self,
        hypotheses: List[HypothesisRecord],
    ) -> List[HypothesisRecord]:
        """
        Sort pending hypotheses so the engine learns faster.

        Priority rules (all best effort and defensive):
            - higher estimated impact first
            - higher RYE or delta_R first
            - more recent hypotheses first
        """
        def _impact_score(h: HypothesisRecord) -> float:
            val = getattr(h, "impact", None)
            if isinstance(val, (int, float)):
                return float(val)
            val = getattr(h, "estimated_impact", None)
            if isinstance(val, (int, float)):
                return float(val)
            return 0.0

        def _rye_score(h: HypothesisRecord) -> float:
            try:
                rye = getattr(h, "rye_after", None)
                if rye is None:
                    rye = getattr(h, "RYE", None)
                if rye is not None:
                    return float(rye)
            except Exception:
                pass
            try:
                dr = getattr(h, "delta_r", None)
                if dr is not None:
                    return float(dr)
            except Exception:
                pass
            return 0.0

        def _timestamp_score(h: HypothesisRecord) -> float:
            ts = getattr(h, "created_at", None) or getattr(h, "timestamp", None)
            if isinstance(ts, (int, float)):
                return float(ts)
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts).timestamp()
                except Exception:
                    return 0.0
            return 0.0

        scored = []
        for h in hypotheses:
            s_imp = _impact_score(h)
            s_rye = _rye_score(h)
            s_ts = _timestamp_score(h)
            scored.append((s_imp, s_rye, s_ts, h))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [h for _, _, _, h in scored]

    # ------------------------------------------------------------------
    # RYE / energy piece
    # ------------------------------------------------------------------
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
        toward Tier 2 and Tier 3 style discoveries.
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
                reasons.append("RYE: hypothesis associated with non positive delta_R, penalized.")
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

            # For a research agent, RYE in about 0.02 to 0.20 is realistic high efficiency zone.
            if rv <= 0:
                score = min(score, 0.15)
                reasons.append("Energy: non positive RYE, downgraded.")
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
    # History based learning piece
    # ------------------------------------------------------------------
    def _compute_history_piece(self, hyp: HypothesisRecord) -> Optional[Dict[str, Any]]:
        """
        Build a score piece based on prior verification attempts for
        this hypothesis.

        Behavior:
          - First time: neutral score with explanatory reason
          - Stable high scores over multiple attempts: boost
          - Repeated low scores: penalize
          - Improving trend: boost
          - Deteriorating trend: penalize

        History is in-process only. If you want persistent history
        across runs, plug this engine into a higher level controller
        that reloads and feeds historical scores.
        """
        hist = self._history.get(hyp.hypothesis_id, [])
        attempts = len(hist)

        reasons: List[str] = []
        score = 0.5  # neutral default

        if attempts == 0:
            reasons.append("History: first verification attempt for this hypothesis.")
            extra = {"attempts": 0, "trend": None, "avg_score": None}
            return {
                "kind": "history",
                "score": score,
                "reasons": reasons,
                "extra": extra,
            }

        avg_score = sum(hist) / attempts
        last_score = hist[-1]
        first_score = hist[0]
        trend = last_score - first_score

        # Base on average
        if avg_score >= 0.8 and attempts >= 2:
            score = 0.80
            reasons.append("History: consistently high verification scores across attempts.")
        elif avg_score <= 0.35 and attempts >= 2:
            score = 0.30
            reasons.append("History: consistently low verification scores across attempts.")
        elif 0.35 < avg_score < 0.8:
            score = 0.55
            reasons.append("History: mixed verification scores, slight positive weight.")

        # Trend adjustment
        if trend > 0.15:
            score = max(score, 0.75)
            reasons.append("History trend: verification scores improving over time.")
        elif trend < -0.15:
            score = min(score, 0.35)
            reasons.append("History trend: verification scores worsening over time.")
        else:
            reasons.append("History trend: no strong trend detected.")

        extra = {
            "attempts": attempts,
            "avg_score": avg_score,
            "last_score": last_score,
            "first_score": first_score,
            "trend": trend,
        }

        return {
            "kind": "history",
            "score": float(score),
            "reasons": reasons,
            "extra": extra,
        }

    def _update_history(self, hyp: HypothesisRecord, score: float) -> None:
        """
        Update in-process history for this hypothesis.
        Keeps last N entries to avoid unbounded growth.
        """
        hist = self._history.setdefault(hyp.hypothesis_id, [])
        hist.append(float(score))
        # keep only last 12 attempts for this hypothesis
        if len(hist) > 12:
            del hist[:-12]

    # ------------------------------------------------------------------
    # Adaptive thresholds
    # ------------------------------------------------------------------
    def _adapt_thresholds(self, hyp: HypothesisRecord) -> Tuple[float, float]:
        """
        Adapt validation and rejection thresholds based on:
          - intelligence_profile
          - domain (for example, longevity, math)
          - safety bias and required citations level
        """
        v = float(self.validate_threshold)
        r = float(self.reject_threshold)

        profile = self.intelligence_profile or {}
        safety_bias = str(profile.get("safety_bias", "medium")).lower()
        require_citations_level = str(profile.get("require_citations_level", "normal")).lower()
        tier3_bias = float(profile.get("tier3_bias", 0.0))
        discovery_focus = float(profile.get("discovery_focus", 0.0))

        # Domain hint from engine plus hypothesis metadata
        domain = self.domain or getattr(hyp, "domain", None) or getattr(hyp, "domain_tag", None)
        domain_str = str(domain or "general").lower()

        # Safety bias tuning
        if safety_bias == "high":
            v += 0.05  # require slightly higher score to validate
            r -= 0.05  # reject a bit more aggressively
        elif safety_bias == "low":
            v -= 0.05
            r += 0.05

        # Citation strictness tuning
        if require_citations_level in ("strict", "clinical"):
            v += 0.03

        # Domain specific tuning
        if domain_str in ("longevity", "antiaging", "anti_aging"):
            v += 0.03
            r -= 0.03
        elif domain_str in ("math", "theory"):
            v += 0.02

        # Tier3 and discovery focus can loosen validation slightly for breakthrough hunting
        if tier3_bias > 0.7 and discovery_focus > 0.7:
            v -= 0.02  # allow slightly more adventurous validation

        # Clamp to sane range
        v = max(0.55, min(0.95, v))
        r = max(0.05, min(0.45, r))

        # Ensure reject < validate
        if r >= v:
            mid = (v + r) / 2
            r = max(0.05, mid - 0.10)
            v = min(0.95, mid + 0.10)

        return v, r

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
          - adjust by intelligence_profile
          - normalize by sum of used weights
          - fall back to neutral if no scores

        Current weights are tuned to favor:
          - literature, critique and data for core validity
          - memory and equilibrium for long-run stability
          - rye_energy for reparodynamic strength
          - structural, plausibility and consensus as strong modifiers
          - history as a learning stabilizer
        """
        if not pieces:
            return 0.0, {}

        # Base weight map
        base_weights: Dict[str, float] = {
            "literature": 0.20,
            "critique": 0.16,
            "data": 0.16,
            "structural": 0.10,
            "plausibility": 0.08,
            "consensus": 0.08,
            "memory": 0.10,
            "equilibrium": 0.08,
            "rye_energy": 0.12,
            "history": 0.08,
        }

        # If this is explicitly a math or theory hypothesis, increase structural weight
        domain = getattr(hyp, "domain", None) or getattr(hyp, "domain_tag", "general")
        domain_str = str(domain).lower()
        if domain_str in ("math", "theory"):
            base_weights["structural"] = 0.18

        # Domain specific nudge for longevity
        if domain_str in ("longevity", "antiaging", "anti_aging"):
            base_weights["literature"] += 0.03
            base_weights["data"] += 0.03
            base_weights["rye_energy"] += 0.02

        # Adjust weights using intelligence profile (if present)
        base_weights = self._adjust_weights_with_intelligence(base_weights)

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

        final_score = max(0.0, min(1.0, final_score))

        return final_score, scores_by_kind

    def _adjust_weights_with_intelligence(
        self,
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Modify base weights based on intelligence_profile hints:

            depth
            exploration
            verification_intensity
            discovery_focus
            tier3_bias
            equilibrium_focus
            safety_bias
            hallucination_tolerance
            require_citations_level

        Effect is soft: relative structure remains similar.
        """
        profile = self.intelligence_profile or {}
        depth = str(profile.get("depth", "balanced"))
        exploration = float(profile.get("exploration", 0.5))
        verification_intensity = float(profile.get("verification_intensity", 0.5))
        discovery_focus = float(profile.get("discovery_focus", 0.5))
        equilibrium_focus = float(profile.get("equilibrium_focus", 0.5))
        tier3_bias = float(profile.get("tier3_bias", 0.0))

        # Deep or ultra_deep should favor structural, data and memory
        if depth in ("deep", "ultra_deep"):
            base_weights["structural"] += 0.02
            base_weights["data"] += 0.02
            base_weights["memory"] += 0.02

        # High verification intensity: push critique, plausibility, consensus
        if verification_intensity > 0.7:
            base_weights["critique"] += 0.03
            base_weights["plausibility"] += 0.02
            base_weights["consensus"] += 0.02

        # High discovery focus: push literature, data, rye_energy
        if discovery_focus > 0.7:
            base_weights["literature"] += 0.02
            base_weights["data"] += 0.02
            base_weights["rye_energy"] += 0.02

        # Equilibrium focus: reward equilibrium and history more
        if equilibrium_focus > 0.6:
            base_weights["equilibrium"] += 0.03
            base_weights["history"] += 0.02

        # Tier 3 bias: emphasize rye_energy and consensus
        if tier3_bias > 0.6:
            base_weights["rye_energy"] += 0.02
            base_weights["consensus"] += 0.02

        # Exploration knob could slightly push literature and data higher
        if exploration > 0.7:
            base_weights["literature"] += 0.01
            base_weights["data"] += 0.01

        return base_weights

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def _infer_tier_label(self, score: float) -> Optional[str]:
        """
        Convert verification score into a Tier style label.
        These are soft hints for later classification and papers.
        """
        if score >= 0.9:
            return "tier3_candidate"
        if score >= 0.8:
            return "tier2_candidate"
        if score >= 0.65:
            return "tier1_candidate"
        return None

    def _log_validated(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
        component_scores: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a validated hypothesis in the discovery log.

        High score hypotheses can be tagged as Tier 1, Tier 2 or Tier 3 candidates.
        """
        component_scores = component_scores or {}
        thresholds = thresholds or {}
        tier_label = self._infer_tier_label(score)

        tier_tags: List[str] = []
        if tier_label:
            tier_tags.append(tier_label)

        description_lines = [
            f"Validated hypothesis with final verification score {score:.3f}.",
        ]
        if thresholds:
            description_lines.append(
                f"Thresholds used - validate >= {thresholds.get('validate', self.validate_threshold):.3f}, "
                f"reject <= {thresholds.get('reject', self.reject_threshold):.3f}."
            )
        description_lines.append("")
        description_lines.append("Component scores:")
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
                "thresholds": thresholds,
                "tier_label": tier_label,
                "run_id": self.run_id,
                "logged_at": _utc_iso(),
            },
        )

    def _log_rejected(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
        component_scores: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a rejected hypothesis in the discovery log.
        """
        component_scores = component_scores or {}
        thresholds = thresholds or {}

        description_lines = [
            f"Rejected hypothesis with final verification score {score:.3f}.",
        ]
        if thresholds:
            description_lines.append(
                f"Thresholds used - validate >= {thresholds.get('validate', self.validate_threshold):.3f}, "
                f"reject <= {thresholds.get('reject', self.reject_threshold):.3f}."
            )
        description_lines.append("")
        description_lines.append("Component scores:")
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
                "thresholds": thresholds,
                "run_id": self.run_id,
                "logged_at": _utc_iso(),
            },
        )

    def _log_inconclusive(
        self,
        hyp: HypothesisRecord,
        score: float,
        reasons: List[str],
        extra: Dict[str, Any],
        component_scores: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record an inconclusive verification attempt.
        The hypothesis stays pending but we keep a trail.
        """
        component_scores = component_scores or {}
        thresholds = thresholds or {}

        description_lines = [
            f"Inconclusive verification with score {score:.3f}. Hypothesis remains pending.",
        ]
        if thresholds:
            description_lines.append(
                f"Thresholds used - validate >= {thresholds.get('validate', self.validate_threshold):.3f}, "
                f"reject <= {thresholds.get('reject', self.reject_threshold):.3f}."
            )
        description_lines.append("")
        description_lines.append("Component scores:")
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
                "thresholds": thresholds,
                "run_id": self.run_id,
                "logged_at": _utc_iso(),
            },
        )
