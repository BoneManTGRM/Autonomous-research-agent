"""
Ultra verification pipeline for the Autonomous Research Agent.

Goal:
    Maximize learning speed and discovery yield by turning every cycle output
    into a verified, scored, and replay friendly artifact.

This module does not call tools. It consumes what TGRMLoop already produced
(notes, hypotheses, citations, candidate interventions, RYE, etc) and converts
it into:
    - verification scores
    - novelty and coherence scores
    - citation quality metrics
    - contradiction flags
    - replay ready motifs
    - breakthrough reinforcement hints

High level:
    1. Validate citations and compute a citation quality profile.
    2. Check hypotheses against prior cycles for novelty and contradictions.
    3. Build mechanism and intervention motifs for replay.
    4. Compute verification_score, novelty_score, disco_confidence.
    5. Persist a compact verification summary in MemoryStore and ReplayBuffer.

This file is written to be optional:
    - If memory_store lacks certain methods, calls are silently skipped.
    - If replay_buffer is None, motif routing is skipped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Optional event stream integration (append-only JSONL via agent/event_log.py).
try:  # pragma: no cover
    from .event_log import log_event as _log_event  # type: ignore
except Exception:  # pragma: no cover
    try:
        from event_log import log_event as _log_event  # type: ignore
    except Exception:
        _log_event = None  # type: ignore
import statistics
from datetime import datetime


class VerificationPipeline:
    """
    Ultra verification and scoring module.

    Parameters
    ----------
    memory_store:
        The same MemoryStore used by CoreAgent and TGRMLoop, ideally supporting:
            - log_verification(...)
            - get_cycle_history_for_goal(...)
            - get_cycle_history(...)
    replay_buffer:
        Optional ReplayBuffer used for high RYE motifs and cross hallmark chains.
        Expected minimal API:
            - add_item(...)
            - register_motif(...)
            - count_item_hits(...)
    config:
        Optional dict with tuning parameters. All keys are optional:
            - verification_window: int
            - novelty_window: int
            - min_rye_for_motif: float
            - min_verification_for_motif: float
    """

    def __init__(
        self,
        memory_store: Any,
        replay_buffer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.replay_buffer = replay_buffer
        self.config = config or {}

        self.verification_window: int = int(self.config.get("verification_window", 50))
        self.novelty_window: int = int(self.config.get("novelty_window", 100))
        self.min_rye_for_motif: float = float(self.config.get("min_rye_for_motif", 0.65))
        self.min_verification_for_motif: float = float(
            self.config.get("min_verification_for_motif", 0.7)
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def verify_cycle(
        self,
        cycle_log: Dict[str, Any],
        *,
        hallmark: Optional[str] = None,
        subgoal: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full verification pipeline on a single cycle log.

        Inputs:
            cycle_log:
                The machine facing log from TGRMLoop.run_cycle.
            hallmark:
                Optional hallmark focus for this cycle.
            subgoal:
                Optional sub target such as "mito_membrane_potential".
            stage:
                Optional "idea" or "verify" for two stage loops.

        Returns:
            Dict with:
                {
                  "verification_score": float,
                  "novelty_score": float,
                  "disco_confidence": float,
                  "flags": [...],
                  "citation_profile": {...},
                  "hypothesis_profile": {...},
                  "motifs": [...],
                }
        """
        goal = cycle_log.get("goal", "")
        domain = cycle_log.get("domain", "general")

        
        # Preserve run isolation when possible (prevents blended runs with same goal)
        run_id = cycle_log.get("run_id")
        if run_id is None:
            rm = cycle_log.get("run_metadata") or {}
            if isinstance(rm, dict):
                run_id = rm.get("run_id")
        try:
            run_id = str(run_id) if run_id is not None else None
        except Exception:
            run_id = None
# Flexible RYE extraction to match other modules (DiscoveryManager etc)
        rye_raw = cycle_log.get("RYE")
        if rye_raw is None:
            rye_raw = (
                cycle_log.get("rye_value")
                or cycle_log.get("rye_after")
                or cycle_log.get("rye")
            )
        try:
            rye_value = float(rye_raw) if rye_raw is not None else 0.0
        except Exception:
            rye_value = 0.0

        hypotheses = cycle_log.get("hypotheses", []) or []
        citations = cycle_log.get("citations", []) or []
        candidate_interventions = cycle_log.get("candidate_interventions", []) or []

        # 1) Fetch history slice for this goal
        history = self._get_recent_history_for_goal(goal, limit=self.novelty_window)

        # 2) Build citation profile
        citation_profile = self._analyze_citations(citations, history)

        # 3) Hypothesis profile, including conflict detection
        hypothesis_profile = self._analyze_hypotheses(
            hypotheses=hypotheses,
            history=history,
            domain=domain,
        )

        # 4) Motifs from hypotheses, citations, and candidate interventions
        motifs = self._build_motifs(
            goal=goal,
            domain=domain,
            hallmark=hallmark,
            subgoal=subgoal,
            stage=stage,
            rye_value=rye_value,
            hypotheses=hypotheses,
            candidate_interventions=candidate_interventions,
        )

        # 5) Core scores
        verification_score, flags = self._compute_verification_score(
            rye_value=rye_value,
            citation_profile=citation_profile,
            hypothesis_profile=hypothesis_profile,
            domain=domain,
            stage=stage,
        )

        novelty_score = self._compute_novelty_score(
            hypotheses=hypotheses,
            citations=citations,
            history=history,
        )

        disco_confidence = self._compute_disco_confidence(
            verification_score=verification_score,
            novelty_score=novelty_score,
            rye_value=rye_value,
            hypothesis_profile=hypothesis_profile,
        )

        # 6) Route motifs into replay buffer if strong enough
        self._route_motifs_to_replay(
            motifs=motifs,
            goal=goal,
            domain=domain,
            hallmark=hallmark,
            rye_value=rye_value,
            verification_score=verification_score,
        )

        verification_summary = {
            "cycle": cycle_log.get("cycle"),
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "domain": domain,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "stage": stage,
            "rye_value": rye_value,
            "verification_score": verification_score,
            "novelty_score": novelty_score,
            "disco_confidence": disco_confidence,
            "flags": flags,
            "citation_profile": citation_profile,
            "hypothesis_profile": hypothesis_profile,
            "motifs": motifs,
        }

        # 7) Log into memory store if possible
        self._log_verification_summary(verification_summary)

        return verification_summary

    # ------------------------------------------------------------------
    # History fetch
    # ------------------------------------------------------------------
    def _get_recent_history_for_goal(
        self,
        goal: str,
        limit: int,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Try to fetch recent history rows for this goal."""
        try:
            if hasattr(self.memory_store, "get_cycle_history_for_goal"):
                try:
                    rows = self.memory_store.get_cycle_history_for_goal(goal, limit=limit, run_id=run_id)  # type: ignore[attr-defined]
                except TypeError:
                    rows = self.memory_store.get_cycle_history_for_goal(goal, limit=limit)  # type: ignore[attr-defined]
                if isinstance(rows, list):
                    return rows
        except Exception:
            pass

        try:
            if hasattr(self.memory_store, "get_cycle_history"):
                try:
                    full = self.memory_store.get_cycle_history(run_id=run_id)  # type: ignore[attr-defined]
                except TypeError:
                    full = self.memory_store.get_cycle_history()  # type: ignore[attr-defined]
                if isinstance(full, list):
                    filtered = [r for r in full if r.get("goal") == goal]
                    return filtered[-limit:]
        except Exception:
            pass

        return []

    # ------------------------------------------------------------------
    # Citation analysis
    # ------------------------------------------------------------------
    def _normalize_citation_list(self, raw: Any) -> List[Dict[str, Any]]:
        """Normalize citation structures from mixed types into list of dicts."""
        if raw is None:
            return []
        if isinstance(raw, list):
            out: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, str):
                    out.append({"raw": item})
            return out
        if isinstance(raw, dict):
            return [raw]
        return [{"raw": str(raw)}]

    def _analyze_citations(
        self,
        citations: Any,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute citation quality and redundancy profile."""
        citations_list = self._normalize_citation_list(citations)
        total = len(citations_list)
        if total == 0:
            return {
                "total": 0,
                "unique_sources": 0,
                "unique_urls": 0,
                "missing_urls": 0,
                "mean_age_days": None,
                "redundancy_ratio": None,
                "history_overlap_ratio": None,
                "peer_reviewed_ratio": None,
            }

        # Unique sources and urls
        source_pairs = set()
        urls = set()
        missing_urls = 0
        for c in citations_list:
            src = c.get("source")
            url = c.get("url")
            source_pairs.add((src, url))
            if url:
                urls.add(url)
            else:
                missing_urls += 1

        unique_sources = len(source_pairs)
        unique_urls = len(urls)

        # Determine peer-reviewed ratio based on citation source and metadata
        def _is_peer_reviewed_cite(c: Dict[str, Any]) -> bool:
            """
            Heuristic to infer if a citation likely refers to a peer-reviewed source.

            Returns True for citations from PubMed, Semantic Scholar, or other
            structured paper sources with a DOI or non-empty venue/year.
            """
            try:
                src = str(c.get("source") or "").lower()
                # Common peer-reviewed tool identifiers
                if any(key in src for key in ("pubmed", "semantic", "semanticscholar", "paper")):
                    return True
                # Presence of a DOI often implies a published paper
                doi = c.get("doi") or c.get("DOI")
                if doi and "10." in str(doi):
                    return True
                # Venue and year hint at journal or conference publication
                venue = c.get("venue") or ""
                year = c.get("year") or ""
                if venue and year:
                    venue_l = str(venue).lower()
                    # Exclude known preprint venues
                    if any(pre in venue_l for pre in ("arxiv", "biorxiv", "medrxiv", "preprint")):
                        return False
                    return True
            except Exception:
                pass
            return False

        peer_reviewed_count = sum(1 for cite in citations_list if _is_peer_reviewed_cite(cite))
        peer_ratio: Optional[float] = None
        if total > 0:
            peer_ratio = peer_reviewed_count / float(total)

        # Redundancy ratio
        redundancy_ratio = 0.0
        if unique_urls > 0:
            redundancy_ratio = max(0.0, 1.0 - (unique_urls / float(total)))

        # Overlap with historical urls
        hist_urls = set()
        for row in history:
            row_cites = self._normalize_citation_list(row.get("citations", []))
            for c in row_cites:
                url = c.get("url")
                if url:
                    hist_urls.add(url)

        overlap = len(urls.intersection(hist_urls)) if hist_urls else 0
        history_overlap_ratio = (overlap / float(unique_urls)) if unique_urls > 0 else None

        # Publication age is optional and depends on fields, so we do not compute
        mean_age_days = None

        return {
            "total": total,
            "unique_sources": unique_sources,
            "unique_urls": unique_urls,
            "missing_urls": missing_urls,
            "mean_age_days": mean_age_days,
            "redundancy_ratio": redundancy_ratio,
            "history_overlap_ratio": history_overlap_ratio,
            "peer_reviewed_ratio": peer_ratio,
        }

    # ------------------------------------------------------------------
    # Hypothesis analysis
    # ------------------------------------------------------------------
    def _analyze_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        domain: str,
    ) -> Dict[str, Any]:
        """Check hypotheses for focus, conflict, and historical coverage."""
        count = len(hypotheses)
        if count == 0:
            return {
                "count": 0,
                "mean_confidence": None,
                "focused": False,
                "conflict_flags": [],
                "history_overlap_ratio": None,
                "short_text_set": [],
            }

        confs: List[float] = []
        short_texts: List[str] = []
        for h in hypotheses:
            c = h.get("confidence")
            if isinstance(c, (int, float)):
                confs.append(float(c))
            txt = (h.get("title") or h.get("text") or "").strip()
            if txt:
                if len(txt) > 160:
                    txt = txt[:160]
                short_texts.append(txt)

        mean_conf = statistics.mean(confs) if confs else None

        # Focused if we have a small set of distinct hypotheses
        unique_short = list({t for t in short_texts})
        focused = 1 <= len(unique_short) <= 6

        # Simple historical overlap
        hist_titles = set()
        for row in history:
            for h in row.get("hypotheses", []) or []:
                if not isinstance(h, dict):
                    continue
                title = (h.get("title") or h.get("text") or "").strip()
                if title:
                    if len(title) > 160:
                        title = title[:160]
                    hist_titles.add(title)

        overlap = len(hist_titles.intersection(unique_short)) if hist_titles else 0
        history_overlap_ratio = (
            overlap / float(len(unique_short)) if unique_short else None
        )

        # Conflict flags
        conflict_flags = self._detect_conflicts(hypotheses, domain=domain)

        return {
            "count": count,
            "mean_confidence": mean_conf,
            "focused": focused,
            "conflict_flags": conflict_flags,
            "history_overlap_ratio": history_overlap_ratio,
            "short_text_set": unique_short,
        }

    def _detect_conflicts(
        self,
        hypotheses: List[Dict[str, Any]],
        domain: str,
    ) -> List[str]:
        """Lightweight conflict detector between hypotheses."""
        flags: List[str] = []
        if len(hypotheses) < 2:
            return flags

        texts = []
        for h in hypotheses:
            if isinstance(h, dict):
                texts.append((h.get("title") or h.get("text") or "").lower())
            else:
                texts.append(str(h).lower())

        neg_words = ["reduces", "decreases", "suppresses"]
        pos_words = ["increases", "boosts", "enhances"]

        # Extremely simple contradiction check:
        # if the same target is described with opposite trends.
        for i, ti in enumerate(texts):
            for j, tj in enumerate(texts):
                if j <= i:
                    continue
                if any(w in ti for w in neg_words) and any(w in tj for w in pos_words):
                    if self._same_target(ti, tj):
                        flags.append("trend_conflict_pair")

        if domain.lower() == "longevity" and flags:
            flags.append("longevity_conflict_requires_verify_stage")

        return flags

    def _same_target(self, a: str, b: str) -> bool:
        """Heuristic target matching."""
        a_low = a.lower()
        b_low = b.lower()

        if "mtor" in a_low and "mtor" in b_low:
            return True
        if "nad" in a_low and "nad" in b_low:
            return True
        if "senescence" in a_low and "senescence" in b_low:
            return True
        return False

    # ------------------------------------------------------------------
    # Motif builder
    # ------------------------------------------------------------------
    def _build_motifs(
        self,
        goal: str,
        domain: str,
        hallmark: Optional[str],
        subgoal: Optional[str],
        stage: Optional[str],
        rye_value: float,
        hypotheses: List[Dict[str, Any]],
        candidate_interventions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build compact motifs from hypotheses and interventions."""
        motifs: List[Dict[str, Any]] = []

        for h in hypotheses:
            if not isinstance(h, dict):
                continue
            text = (h.get("title") or h.get("text") or "").strip()
            if not text:
                continue
            motifs.append(
                {
                    "kind": "hypothesis",
                    "text": text,
                    "goal": goal,
                    "domain": domain,
                    "hallmark": hallmark,
                    "subgoal": subgoal,
                    "stage": stage,
                    "rye_value": rye_value,
                    "confidence": h.get("confidence"),
                }
            )

        for iv in candidate_interventions:
            if not isinstance(iv, dict):
                continue
            label = iv.get("label") or iv.get("title")
            if not label:
                continue
            motifs.append(
                {
                    "kind": "intervention",
                    "label": label,
                    "source": iv.get("source"),
                    "url": iv.get("url"),
                    "goal": goal,
                    "domain": iv.get("domain") or domain,
                    "hallmark": hallmark,
                    "subgoal": subgoal,
                    "stage": stage,
                    "rye_value": rye_value,
                }
            )

        return motifs

    # ------------------------------------------------------------------
    # Verification score
    # ------------------------------------------------------------------
    def _compute_verification_score(
        self,
        rye_value: float,
        citation_profile: Dict[str, Any],
        hypothesis_profile: Dict[str, Any],
        domain: str,
        stage: Optional[str],
    ) -> Tuple[float, List[str]]:
        """
        Combine RYE, citation quality, and hypothesis structure into a single
        verification score in [0, 1].
        """
        flags: List[str] = []
        domain_lower = (domain or "general").lower()
        stage_lower = (stage or "").lower()

        # RYE base term
        rye_term = max(0.0, min(1.0, rye_value))

        # Citation strength
        total_cites = citation_profile.get("total") or 0
        unique_sources = citation_profile.get("unique_sources") or 0
        redundancy = citation_profile.get("redundancy_ratio")
        overlap = citation_profile.get("history_overlap_ratio")
        peer_ratio = citation_profile.get("peer_reviewed_ratio")

        if total_cites == 0:
            cite_term = 0.1
            flags.append("no_citations")
        else:
            diversity = (unique_sources / float(total_cites)) if total_cites > 0 else 0.0
            cite_term = 0.2 + 0.6 * max(0.0, min(1.0, diversity))
            if redundancy is not None and redundancy > 0.6:
                cite_term *= 0.8
                flags.append("high_citation_redundancy")
            if overlap is not None and overlap > 0.8:
                cite_term *= 0.85
                flags.append("historical_citation_reuse")

            # Adjust citation term by quality of sources (peer review ratio)
            try:
                if peer_ratio is not None:
                    # If less than half of citations are peer-reviewed, penalize
                    if peer_ratio < 0.5:
                        cite_term *= 0.7
                        flags.append("low_peer_reviewed_ratio")
                    # If majority are peer-reviewed but not overwhelming
                    elif peer_ratio < 0.8:
                        cite_term *= 0.9
                    # Else, minimal adjustment for high-quality citations
            except Exception:
                pass

        cite_term = max(0.0, min(1.0, cite_term))

        # Hypothesis structure
        hyp_count = hypothesis_profile.get("count") or 0
        focused = bool(hypothesis_profile.get("focused"))
        conflict_flags = hypothesis_profile.get("conflict_flags") or []
        hyp_term = 0.0

        if hyp_count == 0:
            hyp_term = 0.1
            flags.append("no_hypotheses")
        else:
            if focused:
                hyp_term = 0.7
                flags.append("focused_hypotheses")
            else:
                hyp_term = 0.4
                flags.append("diffuse_hypotheses")

            if conflict_flags:
                hyp_term *= 0.7
                flags.extend(conflict_flags)

        hyp_term = max(0.0, min(1.0, hyp_term))

        # Stage adjustment
        stage_factor = 1.0
        if stage_lower == "idea":
            stage_factor = 0.85
        elif stage_lower == "verify":
            stage_factor = 1.05

        # Domain adjustment
        domain_factor = 1.0
        if domain_lower == "longevity":
            domain_factor = 1.02
        elif domain_lower == "math":
            domain_factor = 1.05

        base_score = (
            rye_term * 0.45
            + cite_term * 0.25
            + hyp_term * 0.30
        )

        verification_score = max(0.0, min(1.0, base_score * stage_factor * domain_factor))
        return verification_score, flags

    # ------------------------------------------------------------------
    # Novelty and disco confidence
    # ------------------------------------------------------------------
    def _compute_novelty_score(
        self,
        hypotheses: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> float:
        """Estimate novelty in [0, 1] relative to historical hypotheses and citations."""
        if not hypotheses and not citations:
            return 0.0

        # Hypothesis novelty
        current_hyp = set()
        for h in hypotheses:
            if not isinstance(h, dict):
                continue
            txt = (h.get("title") or h.get("text") or "").strip()
            if txt:
                if len(txt) > 160:
                    txt = txt[:160]
                current_hyp.add(txt)

        hist_hyp = set()
        for row in history:
            for h in row.get("hypotheses", []) or []:
                if not isinstance(h, dict):
                    continue
                txt = (h.get("title") or h.get("text") or "").strip()
                if txt:
                    if len(txt) > 160:
                        txt = txt[:160]
                    hist_hyp.add(txt)

        hyp_novel = 0.0
        if current_hyp:
            overlap = len(current_hyp.intersection(hist_hyp))
            hyp_novel = 1.0 - (overlap / float(len(current_hyp)))

        # Citation novelty
        citations_list = self._normalize_citation_list(citations)
        current_urls = set(c.get("url") for c in citations_list if c.get("url"))
        hist_urls = set()
        for row in history:
            row_cites = self._normalize_citation_list(row.get("citations", []))
            for c in row_cites:
                url = c.get("url")
                if url:
                    hist_urls.add(url)

        cite_novel = 0.0
        if current_urls:
            overlap_c = len(current_urls.intersection(hist_urls))
            cite_novel = 1.0 - (overlap_c / float(len(current_urls)))

        # Combine
        novelty = hyp_novel * 0.6 + cite_novel * 0.4
        return max(0.0, min(1.0, novelty))

    def _compute_disco_confidence(
        self,
        verification_score: float,
        novelty_score: float,
        rye_value: float,
        hypothesis_profile: Dict[str, Any],
    ) -> float:
        """
        Disco confidence is the probability like signal that this cycle
        contains at least one discovery candidate worth long run replay.
        """
        focused = bool(hypothesis_profile.get("focused"))
        hyp_count = hypothesis_profile.get("count") or 0

        base = (
            verification_score * 0.45
            + novelty_score * 0.35
            + max(0.0, min(1.0, rye_value)) * 0.20
        )

        if focused and 1 <= hyp_count <= 5:
            base *= 1.1
        elif hyp_count > 10:
            base *= 0.9

        return max(0.0, min(1.0, base))

    # ------------------------------------------------------------------
    # Replay routing
    # ------------------------------------------------------------------
    def _route_motifs_to_replay(
        self,
        motifs: List[Dict[str, Any]],
        goal: str,
        domain: str,
        hallmark: Optional[str],
        rye_value: float,
        verification_score: float,
    ) -> None:
        """Send high value motifs to replay buffer if available."""
        if self.replay_buffer is None:
            return

        if rye_value < self.min_rye_for_motif:
            return
        if verification_score < self.min_verification_for_motif:
            return

        for m in motifs:
            try:
                text = m.get("text") or m.get("label")
                if not text:
                    continue
                item = {
                    "goal": goal,
                    "domain": domain,
                    "hallmark": hallmark,
                    "kind": m.get("kind"),
                    "text": text,
                    "rye_score": rye_value,
                    "tags": [
                        "verified_motif",
                        m.get("kind", "unknown"),
                        domain,
                    ],
                }
                self.replay_buffer.add_item(item)  # type: ignore[attr-defined]
            except Exception:
                # Motif routing must never break the main pipeline
                continue

    # ------------------------------------------------------------------
    # Memory store logging
    # ------------------------------------------------------------------
    def _log_verification_summary(self, summary: Dict[str, Any]) -> None:
        """Persist verification summary in the MemoryStore if supported."""
        try:
            if hasattr(self.memory_store, "log_verification"):
                self.memory_store.log_verification(summary)  # type: ignore[attr-defined]
            else:
                # Fallback: append into cycle history if a generic logger exists
                if hasattr(self.memory_store, "log_cycle_annotation"):
                    self.memory_store.log_cycle_annotation(summary)  # type: ignore[attr-defined]
            # Mirror into the unified event stream (events.jsonl) so UI/report can tail verification
            if _log_event is not None and isinstance(summary, dict):
                try:
                    _log_event(
                        run_id=str(summary.get("run_id")) if summary.get("run_id") is not None else None,
                        kind="verification",
                        message="verification_summary",
                        level="info",
                        data=summary,
                        role="verification_pipeline",
                        domain=summary.get("domain"),
                        cycle=summary.get("cycle"),
                    )
                except Exception:
                    pass
        except Exception:
            # Logging failures must not break verification
            pass
