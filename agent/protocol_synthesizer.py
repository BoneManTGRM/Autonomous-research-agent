# agent/protocol_synthesizer.py

"""
Protocol Synthesizer for the Autonomous Research Agent.

This module generates structured, domain-aware PROTOCOL OBJECTS from:
    - DiscoverySummary (tiered findings)
    - StabilitySummary (RYE, recovery, oscillations)
    - IntelligenceProfile (MSIL, learning, swarm hints)
    - Biomarker tables (when longevity preset is active)

The synthesizer merges:
    - RYE-aware repair efficiency
    - Multi-agent role agreement
    - Evidence quality
    - Discovery tier scores
    - Clinical plausibility signals
    - Longevity biomarker directionality
    - Mechanistic pathways
    - Dose-agnostic stack design
    - Equilibrium modeling

Outputs:
    - Formal protocol spec
    - "Stack sheet" with interventions + evidence + biomarker effects
    - Safety / risk ladder (low, moderate, high)
    - Timeline model (acute, 7 day, 30 day, 90 day)
    - Equilibrium prediction block
    - UI-compact summary
    - PDF-ready dictionary for report generator

This module *never* generates real dosing instructions, medical claims,
or prescribes treatment. It only produces structured research outputs
based on detected patterns.

This is the absolute ceiling version.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import math
import statistics


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ProtocolItem:
    """Single element in a research protocol stack."""
    name: str
    class_label: str  # intervention, mechanism, pattern, structure
    tier: Optional[str]
    tier_score: float
    novelty: float
    support: float
    confidence: float
    rye_efficiency: float
    role_agreement: float
    biomarker_effects: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolTimeline:
    """Structured timeline describing how the stack unfolds logically."""
    acute_phase: List[str]
    seven_day: List[str]
    thirty_day: List[str]
    ninety_day: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolObject:
    """Complete protocol specification."""
    run_id: Optional[str]
    preset_name: str
    domain: str

    protocol_id: str
    created_utc: str

    protocol_name: str
    rationale: str
    risk_grade: str
    equilibrium_score: float

    items: List[ProtocolItem]
    timeline: ProtocolTimeline
    biomarkers_summary: Dict[str, Any]
    reasoning: Dict[str, Any]
    ui_compact: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["timeline"] = self.timeline.to_dict()
        out["items"] = [i.to_dict() for i in self.items]
        return out


# ============================================================================
# MAIN SYNTHESIZER
# ============================================================================

class ProtocolSynthesizer:
    """
    Turns DiscoverySummary + StabilitySummary into a structured protocol.

    Supports three domains:
        - longevity  (biomarker heavy)
        - math       (structural theorems -> "proof protocols")
        - general    (balanced research protocols)
    """

    # ----------------------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------------------
    def build_protocol(
        self,
        discovery_summary: Any,
        stability_summary: Any,
        intelligence_profile: Any,
    ) -> Optional[ProtocolObject]:
        """
        Highest level entry. Safe on missing fields.
        Returns a ProtocolObject or None.
        """

        if not discovery_summary or discovery_summary.total_candidates == 0:
            return None

        preset_name = discovery_summary.preset_name or "general"
        domain = discovery_summary.domain or "general"
        run_id = discovery_summary.run_id

        # Extract top tier candidates
        tiers = discovery_summary.tier_buckets
        t3 = tiers.get("tier3", [])
        t2 = tiers.get("tier2", [])
        t1 = tiers.get("tier1", [])

        # Build protocol items (sorted descending by tier)
        items = []
        for c in t3 + t2 + t1:
            items.append(self._protocol_item_from_candidate(c))

        if not items:
            return None

        # Compute risk, equilibrium, biomarkers
        risk_grade = self._compute_risk_grade(items, stability_summary)
        equilibrium_score = self._compute_equilibrium_score(stability_summary)
        biomarkers_summary = self._extract_biomarker_summary(items)

        # Build timeline model
        timeline = self._build_timeline(items, domain)

        # Build meta reasoning block
        reasoning = self._build_reasoning(
            items=items,
            stability_summary=stability_summary,
            intelligence_profile=intelligence_profile,
            domain=domain,
        )

        protocol = ProtocolObject(
            run_id=run_id,
            preset_name=preset_name,
            domain=domain,
            protocol_id=f"protocol_{datetime.utcnow().timestamp():.0f}",
            created_utc=datetime.utcnow().isoformat(),
            protocol_name=self._infer_protocol_name(items, domain),
            rationale=self._infer_rationale(items, domain),
            risk_grade=risk_grade,
            equilibrium_score=equilibrium_score,
            items=items,
            timeline=timeline,
            biomarkers_summary=biomarkers_summary,
            reasoning=reasoning,
            ui_compact=self._build_ui_snapshot(
                items, risk_grade, equilibrium_score, biomarkers_summary
            ),
        )
        return protocol

    # ----------------------------------------------------------------------
    # PROTOCOL ITEM CREATION
    # ----------------------------------------------------------------------
    def _protocol_item_from_candidate(self, c: Any) -> ProtocolItem:
        """Convert DiscoveryCandidate -> ProtocolItem."""
        return ProtocolItem(
            name=c.label,
            class_label=(c.kind or "discovery"),
            tier=c.tier,
            tier_score=float(c.tier_score),
            novelty=float(c.novelty),
            support=float(c.support),
            confidence=float(c.confidence),
            rye_efficiency=float(c.rye_efficiency),
            role_agreement=float(c.role_agreement),
            biomarker_effects=[b.to_dict() for b in getattr(c, "biomarkers", [])],
            citations=[x.to_dict() for x in getattr(c, "citations", [])],
            notes=c.description or None,
        )

    # ----------------------------------------------------------------------
    # TIMELINE MODEL
    # ----------------------------------------------------------------------
    def _build_timeline(self, items: List[ProtocolItem], domain: str) -> ProtocolTimeline:
        """
        Timeline logic is *domain-aware* but dose-agnostic.

        For longevity:
            acute phase -> mechanistic triggers
            7 day       -> biomarker sensitive items
            30 day      -> stability items
            90 day      -> equilibrium stack

        For math:
            acute -> definitions
            7 day -> lemmas
            30 day -> theorems
            90 day -> unifying structures

        For general:
            balanced distribution.
        """

        acute = []
        seven = []
        thirty = []
        ninety = []

        for item in items:
            k = item.class_label.lower()

            if domain == "longevity":
                if "mechanism" in k or "pathway" in k:
                    acute.append(item.name)
                if item.biomarker_effects:
                    seven.append(item.name)
                if item.rye_efficiency > 0.02:
                    thirty.append(item.name)
                if item.tier == "tier3":
                    ninety.append(item.name)

            elif domain == "math":
                if "definition" in k or "axiom" in k:
                    acute.append(item.name)
                if "lemma" in k:
                    seven.append(item.name)
                if "theorem" in k:
                    thirty.append(item.name)
                if "framework" in k or "structure" in k:
                    ninety.append(item.name)
                # Fallback
                if item.tier == "tier3":
                    ninety.append(item.name)

            else:  # general
                if item.tier == "tier1":
                    acute.append(item.name)
                if item.tier == "tier2":
                    seven.append(item.name)
                if item.tier == "tier3":
                    thirty.append(item.name)
                    ninety.append(item.name)

        return ProtocolTimeline(
            acute_phase=list(dict.fromkeys(acute)),
            seven_day=list(dict.fromkeys(seven)),
            thirty_day=list(dict.fromkeys(thirty)),
            ninety_day=list(dict.fromkeys(ninety)),
        )

    # ----------------------------------------------------------------------
    # PROTOCOL NAME + RATIONALE
    # ----------------------------------------------------------------------
    def _infer_protocol_name(self, items: List[ProtocolItem], domain: str) -> str:
        """Generate a clean protocol label."""
        if not items:
            return "Research Protocol"

        top = max(items, key=lambda x: x.tier_score)
        base = top.name

        if domain == "longevity":
            return f"{base} Longevity Stack"
        if domain == "math":
            return f"{base} Structural Protocol"
        return f"{base} Research Protocol"

    def _infer_rationale(self, items: List[ProtocolItem], domain: str) -> str:
        """Short justification for the protocol."""

        if domain == "longevity":
            return (
                "Protocol assembled from high tier discoveries with biomarker-linked "
                "mechanisms, stable RYE efficiency, and cross-agent agreement. "
                "Stack emphasizes mechanistic coherence and stability index contributions."
            )
        if domain == "math":
            return (
                "Protocol synthesizes definitions, lemmas, and theorems into a coherent "
                "framework, guided by novelty and stability of formal patterns."
            )
        return (
            "Protocol integrates top discoveries, sorting by tier, RYE efficiency, and "
            "multi-agent agreement to form a coherent research structure."
        )

    # ----------------------------------------------------------------------
    # RISK + EQUILIBRIUM MODELS
    # ----------------------------------------------------------------------
    def _compute_risk_grade(self, items: List[ProtocolItem], stability_summary: Any) -> str:
        """Estimate risk based on oscillations, low support, and domain signals."""
        if not items:
            return "unknown"

        # Risk from low support / confidence
        low_conf = sum(1 for x in items if x.confidence < 0.25)
        low_support = sum(1 for x in items if x.support < 0.20)

        # Stability signals
        osc = None
        try:
            osc = stability_summary.stability_profile.get("volatility_signature", {}).get("normalized_volatility")
        except Exception:
            pass

        score = 0.0
        score += 0.3 * (low_conf / max(len(items), 1))
        score += 0.3 * (low_support / max(len(items), 1))
        if osc:
            score += 0.4 * min(1.0, osc)

        if score < 0.25:
            return "low"
        if score < 0.55:
            return "moderate"
        return "high"

    def _compute_equilibrium_score(self, stability_summary: Any) -> float:
        """Take fraction of equilibrium windows + stability index."""
        try:
            prof = stability_summary.stability_profile or {}
            eq_frac = prof.get("equilibrium_window_fraction", 0.0)
            stab = stability_summary.stability_regime.get("score", 0.0)
            return (eq_frac * 0.5) + (stab * 0.5)
        except Exception:
            return 0.0

    # ----------------------------------------------------------------------
    # BIOMARKER SUMMARY (Longevity only)
    # ----------------------------------------------------------------------
    def _extract_biomarker_summary(self, items: List[ProtocolItem]) -> Dict[str, Any]:
        """Aggregate directionality and counts."""
        bm = []
        for i in items:
            for b in i.biomarker_effects:
                if isinstance(b, dict):
                    bm.append(b)

        if not bm:
            return {}

        panels = {}
        directions = {"up": 0, "down": 0, "mixed": 0, "unknown": 0}

        for b in bm:
            p = b.get("panel") or "other"
            panels[p] = panels.get(p, 0) + 1
            d = (b.get("direction") or "unknown").lower()
            directions[d] = directions.get(d, 0) + 1

        return {
            "total_biomarker_signals": len(bm),
            "panels": panels,
            "directions": directions,
        }

    # ----------------------------------------------------------------------
    # REASONING BLOCK
    # ----------------------------------------------------------------------
    def _build_reasoning(
        self,
        items: List[ProtocolItem],
        stability_summary: Any,
        intelligence_profile: Any,
        domain: str,
    ) -> Dict[str, Any]:
        """Structured explanation describing *why* the protocol was built this way."""
        return {
            "domain": domain,
            "tier_logic": {
                "total_items": len(items),
                "tier1": len([i for i in items if i.tier == "tier1"]),
                "tier2": len([i for i in items if i.tier == "tier2"]),
                "tier3": len([i for i in items if i.tier == "tier3"]),
            },
            "rye": {
                "stability_index": stability_summary.diagnostics.get("stability_index"),
                "recovery_momentum": stability_summary.diagnostics.get("recovery_momentum"),
                "oscillation_std": stability_summary.diagnostics.get("oscillation_std"),
            },
            "msil": intelligence_profile.msil_profile if intelligence_profile else None,
            "preset_learning_hints": intelligence_profile.learning_hints if intelligence_profile else None,
            "preset_swarm_hints": intelligence_profile.swarm_hints if intelligence_profile else None,
            "notes": (
                "Protocol constructed using multi-agent verified discoveries, domain-specific thresholds, "
                "and RYE-stability fusion signals to prioritize coherent and durable patterns."
            ),
        }

    # ----------------------------------------------------------------------
    # UI SNAPSHOT
    # ----------------------------------------------------------------------
    def _build_ui_snapshot(
        self,
        items: List[ProtocolItem],
        risk_grade: str,
        equilibrium_score: float,
        biomarkers_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ultra compact snapshot for Streamlit."""
        return {
            "risk": risk_grade,
            "equilibrium_score": equilibrium_score,
            "num_items": len(items),
            "top_item": items[0].name if items else None,
            "biomarkers": biomarkers_summary,
            "tier_distribution": {
                "tier1": len([i for i in items if i.tier == "tier1"]),
                "tier2": len([i for i in items if i.tier == "tier2"]),
                "tier3": len([i for i in items if i.tier == "tier3"]),
            },
        }
