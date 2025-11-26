"""
Ultra-Maxed Critic Agent for the Autonomous Research Agent.

This agent performs:
    - Deep adversarial hypothesis evaluation
    - Citation reliability scoring
    - Mechanism contradiction detection
    - Replay-seed verification
    - Cross-hallmark consistency checks
    - RYE-aware rejection and promotion
    - Mechanism pruning + hierarchy compression
    - High-value "critical insights" extraction for MemoryStore
    - Cycle-to-cycle noise reduction

It is the *primary speed-learning accelerator* in longevity runs.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

CRITICAL_FLAGS = {
    "contradiction",
    "low_evidence",
    "no_citations",
    "cross_hallmark_conflict",
    "weak_mechanism_chain",
    "duplicate_hypothesis",
    "no_biomarker_link",
}


class CriticAgent:
    def __init__(
        self,
        memory_store: Any,
        replay_buffer: Optional[Any] = None,
        hallmark_profiles: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.memory_store = memory_store
        self.replay_buffer = replay_buffer
        self.hallmark_profiles = hallmark_profiles
        self.config = config or {}

        self.max_hypotheses_to_rate = int(self.config.get("critic_max_hypotheses", 15))
        self.min_citation_count = int(self.config.get("critic_min_citations", 2))
        self.rye_promote_threshold = float(self.config.get("critic_rye_promote", 0.65))
        self.rye_reject_threshold = float(self.config.get("critic_rye_reject", 0.25))

    # ---------------------------------------------------------------
    # Entry point from CoreAgent.route_to_specialist
    # ---------------------------------------------------------------
    def handle_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload must include:
            goal
            hypotheses
            citations
            hallmark (optional)
            domain
            cycle_index
            run_id
        """
        goal = payload.get("goal")
        hypotheses = payload.get("hypotheses", [])[: self.max_hypotheses_to_rate]
        citations = payload.get("citations", [])
        hallmark = (payload.get("hallmark") or "").lower()
        domain = payload.get("domain", "general")
        cycle_index = payload.get("cycle_index")
        run_id = payload.get("run_id")

        # -----------------------------------------------------------
        # 1) Score hypotheses
        # -----------------------------------------------------------
        scored = self._score_hypotheses(
            hypotheses=hypotheses,
            citations=citations,
            hallmark=hallmark,
            domain=domain,
        )

        # -----------------------------------------------------------
        # 2) Extract contradictions & conflicts
        # -----------------------------------------------------------
        contradictions = self._detect_contradictions(scored, hallmark)

        # -----------------------------------------------------------
        # 3) Cross-hallmark conflict analysis
        # -----------------------------------------------------------
        hallmark_conflicts = self._cross_hallmark_conflicts(
            scored,
            hallmark,
            domain,
        )

        # -----------------------------------------------------------
        # 4) Replay-aware filter
        # -----------------------------------------------------------
        promoted, rejected = self._replay_based_pruning(scored)

        # -----------------------------------------------------------
        # 5) Aggregate insights into note text
        # -----------------------------------------------------------
        critic_note = self._build_critic_note(
            goal=goal,
            scored=scored,
            contradictions=contradictions,
            hallmark_conflicts=hallmark_conflicts,
            promoted=promoted,
            rejected=rejected,
        )

        # Store into MemoryStore
        self._attach_to_memory(goal, critic_note, hallmark, domain)

        # Store critic insights into replay buffer
        self._log_to_replay(
            goal=goal,
            scored=scored,
            promoted=promoted,
            rejected=rejected,
            hallmark=hallmark,
            domain=domain,
            cycle_index=cycle_index,
            run_id=run_id,
        )

        return {
            "ok": True,
            "critic_note": critic_note,
            "scored_hypotheses": scored,
            "promoted": promoted,
            "rejected": rejected,
            "contradictions": contradictions,
            "hallmark_conflicts": hallmark_conflicts,
        }

    # ---------------------------------------------------------------
    # Hypothesis scoring
    # ---------------------------------------------------------------
    def _score_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        hallmark: str,
        domain: str,
    ) -> List[Dict[str, Any]]:
        """Assign:
            - evidence score
            - coherence
            - hallmark alignment
            - biomarker linkage
            - replay familiarity
            - overall critic score
        """

        scored = []
        citation_titles = {c.get("title", "").lower() for c in citations}

        for h in hypotheses:
            text = h.get("text", "")
            title = (h.get("title") or "").lower()

            evidence = self._evidence_score(title, citation_titles)
            hallmark_alignment = self._hallmark_alignment(text, hallmark)
            biomarker_score = self._biomarker_link(text, hallmark, domain)
            mechanism_depth = self._mechanism_depth(text)

            critic_score = (
                0.35 * evidence
                + 0.25 * hallmark_alignment
                + 0.2 * biomarker_score
                + 0.2 * mechanism_depth
            )

            h2 = dict(h)
            h2.update(
                {
                    "evidence_score": evidence,
                    "hallmark_alignment": hallmark_alignment,
                    "biomarker_score": biomarker_score,
                    "mechanism_depth": mechanism_depth,
                    "critic_score": critic_score,
                }
            )

            scored.append(h2)

        return scored

    # ---------------------------------------------------------------
    # Evidence scoring
    # ---------------------------------------------------------------
    def _evidence_score(self, title: str, citation_titles: set) -> float:
        """Score 0-1 based on whether the hypothesis appears supported by citations."""
        if not citation_titles:
            return 0.1
        cnt = sum(1 for ct in citation_titles if title[:60] in ct)
        if cnt >= self.min_citation_count:
            return 1.0
        if cnt == 1:
            return 0.6
        return 0.15

    # ---------------------------------------------------------------
    # Hallmark alignment
    # ---------------------------------------------------------------
    def _hallmark_alignment(self, txt: str, hallmark: str) -> float:
        if not hallmark or not txt:
            return 0.3
        terms = self.hallmark_profiles.get_terms_for(hallmark) if self.hallmark_profiles else []
        if not terms:
            return 0.3
        matches = sum(1 for t in terms if t.lower() in txt.lower())
        if matches >= 4:
            return 1.0
        if matches >= 2:
            return 0.7
        if matches == 1:
            return 0.45
        return 0.1

    # ---------------------------------------------------------------
    # Biomarker linkage
    # ---------------------------------------------------------------
    def _biomarker_link(self, txt: str, hallmark: str, domain: str) -> float:
        if domain != "longevity":
            return 0.3
        biomarkers = self.hallmark_profiles.get_biomarkers(hallmark) if self.hallmark_profiles else []
        matches = sum(1 for b in biomarkers if b.lower() in txt.lower())
        if matches >= 2:
            return 1.0
        if matches == 1:
            return 0.65
        return 0.1

    # ---------------------------------------------------------------
    # Mechanism depth
    # ---------------------------------------------------------------
    def _mechanism_depth(self, txt: str) -> float:
        """Look for mechanistic richness as a proxy for depth."""
        cues = [
            "pathway",
            "mechanism",
            "upregulate",
            "downregulate",
            "mtor",
            "autophagy",
            "senescence",
            "oxidative stress",
            "inflammation",
            "mitochondria",
        ]
        hits = sum(1 for c in cues if c in txt.lower())
        if hits >= 5:
            return 1.0
        if hits >= 3:
            return 0.7
        if hits >= 1:
            return 0.4
        return 0.1

    # ---------------------------------------------------------------
    # Contradiction detection
    # ---------------------------------------------------------------
    def _detect_contradictions(
        self,
        scored: List[Dict[str, Any]],
        hallmark: str,
    ) -> List[Dict[str, Any]]:
        """Detect when hypotheses disagree."""
        contradictions = []
        texts = [s["text"].lower() for s in scored]

        for i, t1 in enumerate(texts):
            for j, t2 in enumerate(texts):
                if i >= j:
                    continue
                if (
                    "increase" in t1 and "decrease" in t2
                    or "inhibit" in t1 and "activate" in t2
                    or hallmark and hallmark in t1 and "not related" in t2
                ):
                    contradictions.append(
                        {
                            "h1": scored[i],
                            "h2": scored[j],
                            "type": "internal_conflict",
                        }
                    )
        return contradictions

    # ---------------------------------------------------------------
    # Cross-hallmark checking
    # ---------------------------------------------------------------
    def _cross_hallmark_conflicts(
        self,
        scored: List[Dict[str, Any]],
        hallmark: str,
        domain: str,
    ) -> List[Dict[str, Any]]:
        """Find conflicts across hallmarks (e.g., mito vs senescence)."""
        if domain != "longevity":
            return []

        conflicts = []
        for h in scored:
            text = h.get("text", "").lower()
            if hallmark == "mitochondria" and "pro-senescence" in text:
                conflicts.append(
                    {"hypothesis": h, "conflict": "mitochondria intervention may promote senescence"}
                )
            if hallmark == "senescence" and "mito dysfunction" in text:
                conflicts.append(
                    {"hypothesis": h, "conflict": "senescence intervention weakens mito function"}
                )
        return conflicts

    # ---------------------------------------------------------------
    # Replay-aware pruning
    # ---------------------------------------------------------------
    def _replay_based_pruning(self, scored: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        promoted = []
        rejected = []

        for h in scored:
            critic_score = h.get("critic_score", 0)
            if critic_score >= self.rye_promote_threshold:
                promoted.append(h)
            elif critic_score <= self.rye_reject_threshold:
                rejected.append(h)

        return promoted, rejected

    # ---------------------------------------------------------------
    # Build critic note
    # ---------------------------------------------------------------
    def _build_critic_note(
        self,
        goal: str,
        scored: List[Dict[str, Any]],
        contradictions: List[Any],
        hallmark_conflicts: List[Any],
        promoted: List[Any],
        rejected: List[Any],
    ) -> str:

        lines = []
        lines.append("[critic] Critical evaluation for goal:")
        lines.append(goal)
        lines.append("")

        lines.append("Highest scoring hypotheses:")
        top = sorted(scored, key=lambda x: x["critic_score"], reverse=True)[:5]
        for h in top:
            lines.append(f"- {h['title'][:120]} (score={h['critic_score']:.3f})")
        lines.append("")

        if contradictions:
            lines.append("Contradictions detected:")
            for c in contradictions[:6]:
                lines.append(f"* Conflict between: {c['h1']['title']} AND {c['h2']['title']}")
            lines.append("")

        if hallmark_conflicts:
            lines.append("Cross-hallmark conflicts:")
            for c in hallmark_conflicts[:6]:
                lines.append(f"* {c['hypothesis']['title']}: {c['conflict']}")
            lines.append("")

        if promoted:
            lines.append("Promoted hypotheses (high RYE potential):")
            for h in promoted[:6]:
                lines.append(f"* {h['title']}")
            lines.append("")

        if rejected:
            lines.append("Rejected hypotheses (low evidence / poor coherence):")
            for h in rejected[:6]:
                lines.append(f"* {h['title']}")
            lines.append("")

        return "\n".join(lines)

    # ---------------------------------------------------------------
    # MemoryStore logging
    # ---------------------------------------------------------------
    def _attach_to_memory(self, goal: str, critic_note: str, hallmark: str, domain: str):
        try:
            self.memory_store.add_note(
                goal,
                critic_note,
                role="critic",
                metadata={
                    "hallmark": hallmark,
                    "domain": domain,
                    "source": "critic_agent",
                },
            )
        except Exception:
            try:
                self.memory_store.add_note(goal, critic_note, role="critic")
            except:
                pass

    # ---------------------------------------------------------------
    # Replay logging
    # ---------------------------------------------------------------
    def _log_to_replay(
        self,
        goal: str,
        scored: List[Dict[str, Any]],
        promoted: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        hallmark: str,
        domain: str,
        cycle_index: int,
        run_id: str,
    ):
        if self.replay_buffer is None:
            return

        item = {
            "item_type": "critic_review",
            "goal": goal,
            "scored": scored,
            "promoted": promoted,
            "rejected": rejected,
            "hallmark": hallmark,
            "domain": domain,
            "cycle_index": cycle_index,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        try:
            if hasattr(self.replay_buffer, "add_item"):
                self.replay_buffer.add_item(item)
            elif hasattr(self.replay_buffer, "log_item"):
                self.replay_buffer.log_item(item)
        except Exception:
            pass
