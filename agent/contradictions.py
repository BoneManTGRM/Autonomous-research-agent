"""
contradictions.py

Contradiction engine for the Autonomous Research Agent.

Purpose
    Help the system learn faster by detecting and tracking contradictions
    across hypotheses and long term memory.

    Core ideas
      - treat contradictions as high value learning signals
      - prioritize high impact contradictions first
      - log everything into the DiscoveryLogger so the gold notebook
        and reports can surface them
      - keep in process history so repeated contradictions get extra weight
      - be fully hook based so detection can be done by LLM tools,
        embedding similarity, or external checkers

This module does not talk to tools directly.
Instead, you pass it hooks for:
    conflict_fn(a_claim, b_claim) -> Dict
    severity_fn(contradiction_dict) -> float
    repair_hint_fn(contradiction_dict) -> Dict

It also provides utilities so other parts of the agent can query
which hypotheses or memory items are currently the hottest sources
of contradictions for 10x faster learning focus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .discovery_log import DiscoveryLogger  # type: ignore
from .hypothesis_manager import HypothesisRecord  # type: ignore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """
    Minimal representation of a factual claim.

    Fields are intentionally generic so you can build these from:
        - hypotheses
        - memory entries
        - notes, snapshots, or external documents
    """

    claim_id: str
    text: str
    source: str  # example "hypothesis", "memory", "paper"
    owner_id: Optional[str] = None  # for example hypothesis_id or memory_id
    domain: Optional[str] = None
    rye_after: Optional[float] = None
    delta_r: Optional[float] = None
    energy: Optional[float] = None
    timestamp: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContradictionRecord:
    """
    Canonical record for a contradiction between two claims.
    """

    contradiction_id: str
    claim_a: Claim
    claim_b: Claim
    severity: float
    confidence: float
    reasons: List[str]
    created_at: str
    status: str = "open"  # "open", "resolved", "dismissed"
    resolution_notes: Optional[str] = None
    last_updated_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contradiction engine
# ---------------------------------------------------------------------------


class ContradictionEngine:
    """
    Detects and tracks contradictions across claims.

    Learning speed design
        - fast, cheap scoring uses RYE and simple heuristics to filter
          out low value contradictions
        - only the top ranked items are logged and returned
        - maintains per owner heat so the agent can focus the TGRM loop
          on the most contradictory parts of its knowledge

    Hooks
        conflict_fn(a: Claim, b: Claim) -> Dict
            Required for auto detection in scan_claims.
            Expected return format:
                {
                    "is_conflict": bool,
                    "confidence": float between 0 and 1,
                    "reasons": [list of strings],
                    "extra": {...optional...}
                }

        severity_fn(c: ContradictionRecord) -> float
            Optional. If not provided a default severity heuristic is used.

        repair_hint_fn(c: ContradictionRecord) -> Dict
            Optional. Can provide suggestions for what the TGRM loop
            should do in order to repair the contradiction.
    """

    def __init__(
        self,
        discovery_logger: DiscoveryLogger,
        conflict_fn: Optional[Callable[[Claim, Claim], Dict[str, Any]]] = None,
        severity_fn: Optional[Callable[[ContradictionRecord], float]] = None,
        repair_hint_fn: Optional[Callable[[ContradictionRecord], Dict[str, Any]]] = None,
        intelligence_profile: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.discovery_logger = discovery_logger
        self.conflict_fn = conflict_fn
        self.severity_fn = severity_fn
        self.repair_hint_fn = repair_hint_fn
        self.intelligence_profile = intelligence_profile or {}
        self.domain = domain
        self.run_id = run_id

        # In process store for contradiction records by id
        self._contradictions: Dict[str, ContradictionRecord] = {}

        # Heat maps for faster learning focus
        # owner_id -> cumulative severity score from open contradictions
        self._owner_heat: Dict[str, float] = {}

        # History store so repeated contradictions on the same pair
        # can be tracked and weighted
        # key is (claim_a_id, claim_b_id) sorted tuple
        self._pair_history: Dict[Tuple[str, str], List[float]] = {}

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------

    def scan_claims(
        self,
        claims: Iterable[Claim],
        max_pairs: Optional[int] = None,
        min_confidence: float = 0.55,
        min_severity: float = 0.35,
        top_k: Optional[int] = 25,
    ) -> List[ContradictionRecord]:
        """
        Auto detect contradictions across a batch of claims.

        Steps
            - build all unique pairs (or early stop at max_pairs)
            - ask conflict_fn about each pair
            - build ContradictionRecord for those with is_conflict True
              and confidence >= min_confidence
            - compute severity for each
            - keep only items with severity >= min_severity
            - rank by severity and RYE priority
            - keep top_k and log them

        Returns the list of created ContradictionRecord objects.
        """
        if self.conflict_fn is None:
            return []

        claims_list = list(claims)
        n = len(claims_list)
        records: List[ContradictionRecord] = []

        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if max_pairs is not None and pair_count >= max_pairs:
                    break
                pair_count += 1

                a = claims_list[i]
                b = claims_list[j]
                res = self._safe_conflict_call(a, b)
                if not res.get("is_conflict", False):
                    continue

                confidence = float(res.get("confidence", 0.0))
                if confidence < min_confidence:
                    continue

                reasons = res.get("reasons", []) or []
                extra = res.get("extra", {}) or {}

                contradiction = self._make_record(
                    claim_a=a,
                    claim_b=b,
                    confidence=confidence,
                    reasons=reasons,
                    extra=extra,
                )

                # severity step
                severity = self._compute_severity(contradiction)
                if severity < min_severity:
                    continue

                contradiction.severity = severity
                self._register_contradiction(contradiction)
                records.append(contradiction)

            if max_pairs is not None and pair_count >= max_pairs:
                break

        # Rank and clip
        if not records:
            return []

        ranked = sorted(
            records,
            key=lambda c: (
                c.severity,
                self._rye_priority(c.claim_a),
                self._rye_priority(c.claim_b),
            ),
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        # Log ranked contradictions into the discovery log
        for rec in ranked:
            self._log_contradiction(rec)

        return ranked

    def record_manual_contradiction(
        self,
        claim_a: Claim,
        claim_b: Claim,
        confidence: float,
        reasons: Optional[List[str]] = None,
        severity: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ContradictionRecord:
        """
        Allow other parts of the agent to manually register a contradiction.

        This is useful when an LLM tool, a human, or another engine
        already knows that two claims conflict.
        """
        rec = self._make_record(
            claim_a=claim_a,
            claim_b=claim_b,
            confidence=confidence,
            reasons=reasons or [],
            extra=extra or {},
        )
        rec.severity = self._compute_severity(rec) if severity is None else float(severity)
        self._register_contradiction(rec)
        self._log_contradiction(rec)
        return rec

    def resolve_contradiction(
        self,
        contradiction_id: str,
        status: str,
        notes: Optional[str] = None,
    ) -> Optional[ContradictionRecord]:
        """
        Mark a contradiction as resolved or dismissed.

        Status should be "resolved" or "dismissed".
        This will update the heat map, record history,
        and log a resolution event.
        """
        rec = self._contradictions.get(contradiction_id)
        if rec is None:
            return None

        if status not in ("resolved", "dismissed"):
            raise ValueError("status must be 'resolved' or 'dismissed'")

        rec.status = status
        rec.resolution_notes = notes
        rec.last_updated_at = _utc_iso()

        self._update_heat(rec, remove=True)
        self._log_resolution(rec)
        return rec

    # ------------------------------------------------------------------
    # learning focus utilities
    # ------------------------------------------------------------------

    def get_hot_owners(
        self,
        top_k: int = 20,
        min_heat: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Return the owners that currently have the most contradiction heat.

        Typical use
            - feed this into the TGRM loop as a priority list
            - focus web search and verification passes on these owners
        """
        items = [
            (owner, heat)
            for owner, heat in self._owner_heat.items()
            if heat >= min_heat
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def get_open_contradictions_for_owner(
        self,
        owner_id: str,
    ) -> List[ContradictionRecord]:
        """
        Convenience helper.
        """
        out: List[ContradictionRecord] = []
        for rec in self._contradictions.values():
            if rec.status != "open":
                continue
            if rec.claim_a.owner_id == owner_id or rec.claim_b.owner_id == owner_id:
                out.append(rec)
        out.sort(key=lambda c: c.severity, reverse=True)
        return out

    # ------------------------------------------------------------------
    # internal constructors and helpers
    # ------------------------------------------------------------------

    def _safe_conflict_call(self, a: Claim, b: Claim) -> Dict[str, Any]:
        """
        Wrapper around conflict_fn that always returns a dict
        with minimal fields present.
        """
        if self.conflict_fn is None:
            return {"is_conflict": False, "confidence": 0.0, "reasons": [], "extra": {}}

        try:
            res = self.conflict_fn(a, b) or {}
        except Exception as e:
            return {
                "is_conflict": False,
                "confidence": 0.0,
                "reasons": [f"conflict_fn failed with error: {e}"],
                "extra": {"error": str(e)},
            }

        is_conflict = bool(res.get("is_conflict", False))
        confidence = float(res.get("confidence", 0.0))
        reasons = res.get("reasons", []) or []
        extra = res.get("extra", {}) or {}

        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        return {
            "is_conflict": is_conflict,
            "confidence": confidence,
            "reasons": reasons,
            "extra": extra,
        }

    def _make_record(
        self,
        claim_a: Claim,
        claim_b: Claim,
        confidence: float,
        reasons: List[str],
        extra: Dict[str, Any],
    ) -> ContradictionRecord:
        """
        Build a new ContradictionRecord with a stable key based on the pair.
        """
        pair_key = tuple(sorted([claim_a.claim_id, claim_b.claim_id]))
        pair_str = f"{pair_key[0]}::{pair_key[1]}"

        now = _utc_iso()
        contradiction_id = f"ctr_{pair_str}_{now}"

        rec = ContradictionRecord(
            contradiction_id=contradiction_id,
            claim_a=claim_a,
            claim_b=claim_b,
            severity=0.5,  # will be updated later
            confidence=float(confidence),
            reasons=list(reasons),
            created_at=now,
            status="open",
            resolution_notes=None,
            last_updated_at=None,
            extra=dict(extra),
        )

        # update pair history for learning
        scores = self._pair_history.setdefault(pair_key, [])
        scores.append(rec.confidence)
        if len(scores) > 12:
            del scores[:-12]

        return rec

    def _register_contradiction(self, rec: ContradictionRecord) -> None:
        """
        Add contradiction to local store and heat maps.
        """
        self._contradictions[rec.contradiction_id] = rec
        self._update_heat(rec, remove=False)

    def _update_heat(self, rec: ContradictionRecord, remove: bool) -> None:
        """
        Update heat maps when contradictions are added or resolved.
        """
        sign = -1.0 if remove else 1.0
        magnitude = rec.severity * rec.confidence

        for claim in (rec.claim_a, rec.claim_b):
            if not claim.owner_id:
                continue
            current = self._owner_heat.get(claim.owner_id, 0.0)
            self._owner_heat[claim.owner_id] = max(
                0.0,
                current + sign * magnitude,
            )

    def _rye_priority(self, claim: Claim) -> float:
        """
        Small helper that turns RYE metrics into a priority score.

        Higher RYE and delta_R give higher priority.
        """
        score = 0.0
        try:
            if claim.rye_after is not None:
                score += float(claim.rye_after)
            if claim.delta_r is not None:
                score += 0.5 * float(claim.delta_r)
        except Exception:
            pass
        return score

    def _compute_severity(self, rec: ContradictionRecord) -> float:
        """
        Compute a severity score for a contradiction in the range 0 to 1.

        Inputs
            - conflict confidence
            - domain risk profile
            - RYE priority
            - history pattern on this pair
        """
        if self.severity_fn is not None:
            try:
                val = float(self.severity_fn(rec))
                return max(0.0, min(1.0, val))
            except Exception:
                pass

        # Default heuristic severity
        base = rec.confidence

        domain = (
            rec.claim_a.domain
            or rec.claim_b.domain
            or self.domain
            or "general"
        )
        domain_str = str(domain).lower()

        # domain risk
        if domain_str in ("longevity", "antiaging", "anti_aging", "clinical"):
            base += 0.15
        elif domain_str in ("math", "theory"):
            base += 0.05

        # RYE driven boost
        rye_boost = 0.0
        rye_boost += self._rye_priority(rec.claim_a)
        rye_boost += self._rye_priority(rec.claim_b)
        if rye_boost > 0:
            rye_boost = min(0.25, 0.1 + 0.25 * min(1.0, rye_boost))
        base += rye_boost

        # pair history influence
        pair_key = tuple(sorted([rec.claim_a.claim_id, rec.claim_b.claim_id]))
        hist = self._pair_history.get(pair_key, [])
        if len(hist) >= 3:
            avg_conf = sum(hist) / len(hist)
            if avg_conf >= 0.7:
                base += 0.10
            elif avg_conf <= 0.35:
                base -= 0.05

        # intelligence profile tuning
        profile = self.intelligence_profile or {}
        equilibrium_focus = float(profile.get("equilibrium_focus", 0.5))
        verification_intensity = float(profile.get("verification_intensity", 0.5))

        base += 0.05 * (equilibrium_focus - 0.5)
        base += 0.05 * (verification_intensity - 0.5)

        return max(0.0, min(1.0, base))

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def _log_contradiction(self, rec: ContradictionRecord) -> None:
        """
        Send a contradiction event into the discovery log.
        """
        repair_hints: Dict[str, Any] = {}
        if self.repair_hint_fn is not None:
            try:
                repair_hints = self.repair_hint_fn(rec) or {}
            except Exception as e:
                repair_hints = {"error": str(e)}

        description_lines: List[str] = []
        description_lines.append(
            f"Detected contradiction with severity {rec.severity:.3f} and confidence {rec.confidence:.3f}."
        )
        description_lines.append("")
        description_lines.append("Claim A:")
        description_lines.append(f"  id: {rec.claim_a.claim_id}")
        description_lines.append(f"  owner_id: {rec.claim_a.owner_id}")
        description_lines.append(f"  source: {rec.claim_a.source}")
        description_lines.append(f"  text: {rec.claim_a.text}")
        description_lines.append("")
        description_lines.append("Claim B:")
        description_lines.append(f"  id: {rec.claim_b.claim_id}")
        description_lines.append(f"  owner_id: {rec.claim_b.owner_id}")
        description_lines.append(f"  source: {rec.claim_b.source}")
        description_lines.append(f"  text: {rec.claim_b.text}")
        description_lines.append("")
        if rec.reasons:
            description_lines.append("Reasons:")
            for r in rec.reasons:
                description_lines.append(f"- {r}")
            description_lines.append("")
        if repair_hints:
            description_lines.append("Repair hints:")
            for k, v in repair_hints.items():
                description_lines.append(f"- {k}: {v}")

        description = "\n".join(description_lines)

        tags: List[str] = [
            "contradiction",
            "open" if rec.status == "open" else rec.status,
        ]
        if self.domain:
            tags.append(str(self.domain))

        self.discovery_logger.log_event(
            kind="contradiction_detected",
            title="Contradiction detected",
            description=description,
            cycle_index=None,
            agent_role=None,
            tags=tags,
            extra={
                "contradiction_id": rec.contradiction_id,
                "severity": rec.severity,
                "confidence": rec.confidence,
                "claim_a": rec.claim_a.__dict__,
                "claim_b": rec.claim_b.__dict__,
                "reasons": rec.reasons,
                "repair_hints": repair_hints,
                "status": rec.status,
                "created_at": rec.created_at,
                "run_id": self.run_id,
                "logged_at": _utc_iso(),
            },
        )

    def _log_resolution(self, rec: ContradictionRecord) -> None:
        """
        Log resolution or dismissal of a contradiction.
        """
        description_lines: List[str] = []
        description_lines.append(
            f"Contradiction {rec.contradiction_id} marked as {rec.status}."
        )
        description_lines.append(f"Severity was {rec.severity:.3f}, confidence {rec.confidence:.3f}.")
        if rec.resolution_notes:
            description_lines.append("")
            description_lines.append("Resolution notes:")
            description_lines.append(rec.resolution_notes)

        description = "\n".join(description_lines)

        self.discovery_logger.log_event(
            kind="contradiction_resolved",
            title="Contradiction resolved",
            description=description,
            cycle_index=None,
            agent_role=None,
            tags=["contradiction", rec.status],
            extra={
                "contradiction_id": rec.contradiction_id,
                "status": rec.status,
                "resolution_notes": rec.resolution_notes,
                "severity": rec.severity,
                "confidence": rec.confidence,
                "run_id": self.run_id,
                "logged_at": _utc_iso(),
            },
        )
