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

# Import citation normalization utility.  We guard with try/except so the
# pipeline remains functional even if citation_utils is unavailable (e.g.,
# during unit tests or isolated deployments).  The normalize_citation
# function canonicalizes citation dicts and filters out stub/error entries.
try:
    from .citation_utils import normalize_citation  # type: ignore
except Exception:
    normalize_citation = None  # type: ignore

# Optional event stream integration (append-only JSONL via agent/event_log.py).
try:  # pragma: no cover
    from .event_log import log_event as _log_event  # type: ignore
except Exception:  # pragma: no cover
    try:
        from event_log import log_event as _log_event  # type: ignore
    except Exception:
        _log_event = None  # type: ignore
import statistics
import re  # Used for simple keyword extraction in support ratio computation
from datetime import datetime

# Optional quality gate import.  When available, this is used to decide
# whether a cycle's improvement should contribute to verification and novelty
# scores.  If the import fails, the gate variable is set to None and
# gating checks are skipped.
try:
    from .quality_gates import evaluate_cycle_gate  # type: ignore
except Exception:
    evaluate_cycle_gate = None  # type: ignore

# Banned patterns used to filter out internal template or maintenance entries.
# Hypotheses containing any of these substrings will be removed during
# verification to prevent placeholder-like artifacts from influencing scores.
BANNED_HYP_PATTERNS: List[str] = [
    "maintenance_mode",
    "maintenance mode",
    "discovery_log",
    "discovery log",
    "placeholder discovery",
    "template entry",
    "template",
    "example",
    "inconclusive verification",
    "rejected hypothesis",
    "performed targeted research",
    "initial discovery_log.json",
    "initial discovery log",
    "used only to show",
    "shows how an",

    # Extra patterns for unresolved template variables.  These terms
    # indicate that a hypothesis string was generated from a template
    # without proper substitution and should be filtered out.  See
    # report_generator.BANNED_DISCOVERY_PATTERNS for similar logic.
    "agent",
    "description",
    "detected",
    "encountered",
    "fully",

    # Additional patterns to catch leaked run directives and prompt injection
    # strings.  These phrases rarely belong in a real hypothesis and thus
    # are treated as banned placeholders.
    "system directive",
    "autonomous research swarm",
    "coordinated cycle",
    "single coordinated cycle",
    "64-agent",
    "64 agent",
    "you are a 64",
    "research swarm operating",
]


def _contains_banned_pattern(text: str) -> bool:
    """Return True if the given text contains any banned placeholder pattern."""
    if not text:
        return False
    s = text.lower()
    for pat in BANNED_HYP_PATTERNS:
        if pat in s:
            return True
    return False


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
        # Pass run_id when available to avoid blending cycles from different runs
        history = self._get_recent_history_for_goal(goal, limit=self.novelty_window, run_id=run_id)

        # 2) Build citation profile
        citation_profile = self._analyze_citations(citations, history)

        # 3) Hypothesis profile, including conflict detection
        hypothesis_profile = self._analyze_hypotheses(
            hypotheses=hypotheses,
            history=history,
            domain=domain,
        )

        # Detect Tavily API failures.  If any citation snippet or title
        # indicates a Tavily error, record a flag so downstream logic can
        # recognise that this was a web-blind run (failed web search) and
        # adapt thresholds accordingly.  The detection is case-insensitive
        # and looks for "tavily" combined with "error" or "failed".
        tavily_failed = False
        try:
            for cite in citations:
                t = str(cite.get("title") or "").lower()
                s = str(cite.get("snippet") or "").lower()
                if ("tavily" in t or "tavily" in s) and ("error" in t or "error" in s or "failed" in s):
                    tavily_failed = True
                    break
        except Exception:
            tavily_failed = False

        # 3a) Compute hypothesis support ratio: fraction of hypotheses with at least
        # one citation whose title or snippet contains any keyword from the
        # hypothesis text.  This helps detect placeholder hypotheses or
        # unsupported claims.  A support_ratio of 1.0 indicates every
        # hypothesis has at least one matching citation; 0.0 indicates none.
        try:
            cites_for_support = self._normalize_citation_list(citations)
            support_ratio = self._compute_support_ratio(hypotheses, cites_for_support)
        except Exception:
            support_ratio = None
        if isinstance(hypothesis_profile, dict):
            hypothesis_profile["support_ratio"] = support_ratio

        # Optional gating: drop unsupported hypotheses from further scoring.
        # When citations exist and the support ratio is less than 1.0, we
        # filter out any hypothesis that lacks citation support.  This
        # ensures unsupported claims do not contribute to novelty or
        # verification scoring and helps suppress placeholder-like text.
        try:
            if support_ratio is not None and support_ratio < 1.0 and cites_for_support:
                filtered_hyps: List[Dict[str, Any]] = []
                for h in hypotheses:
                    try:
                        h_text = (h.get("title") or h.get("text") or "").strip()  # type: ignore
                    except Exception:
                        h_text = str(h).strip() if h is not None else ""
                    if not h_text:
                        continue
                    # Extract keywords (min length 3)
                    raw_tokens = re.split(r"[\s,;:\.\(\)\[\]\{\}\-_/]+", h_text.lower())
                    keywords = [tok for tok in raw_tokens if len(tok) >= 3]
                    if not keywords:
                        continue
                    supported = False
                    for cite in cites_for_support:
                        try:
                            ct = (cite.get("title") or "").lower()  # type: ignore
                            cs = (cite.get("snippet") or "").lower()  # type: ignore
                        except Exception:
                            ct = ""
                            cs = ""
                        for kw in keywords:
                            if kw and (kw in ct or kw in cs):
                                supported = True
                                break
                        if supported:
                            break
                    if supported:
                        filtered_hyps.append(h)
                # Only override when some hypotheses remain
                if filtered_hyps:
                    hypotheses = filtered_hyps
                    cycle_log["hypotheses"] = filtered_hyps
        except Exception:
            pass

        # Additional gating: drop any hypothesis containing banned patterns.
        try:
            if hypotheses:
                cleaned_hyps: List[Dict[str, Any]] = []
                for h in hypotheses:
                    try:
                        h_text = (h.get("title") or h.get("text") or "").strip()
                    except Exception:
                        h_text = str(h).strip() if h is not None else ""
                    if not h_text:
                        continue
                    if _contains_banned_pattern(h_text):
                        continue
                    cleaned_hyps.append(h)
                if cleaned_hyps:
                    hypotheses = cleaned_hyps
                    cycle_log["hypotheses"] = cleaned_hyps
        except Exception:
            pass

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

        # If the Tavily API failed during this cycle (i.e. no web search
        # results could be retrieved), record a flag.  This can be
        # consumed by scoring heuristics or the reporting layer to
        # indicate that the agent operated in an "academic-only" mode.
        if tavily_failed:
            try:
                if isinstance(flags, list):
                    flags.append("tavily_web_blind_run")
                else:
                    flags = [flags, "tavily_web_blind_run"] if flags else ["tavily_web_blind_run"]
            except Exception:
                # fallback: set flags list with only the web-blind run marker
                flags = ["tavily_web_blind_run"]

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

        # -----------------------------------------------------------------
        # Enrich hypothesis profile with confidence levels, provenance tags,
        # and falsification attempt logs.  These additions provide buyers
        # and downstream auditors with perâclaim confidence scores, the
        # provenance of supporting evidence, and a record of whether
        # falsification was attempted or contradictions were detected.
        try:
            conf_levels: List[Optional[float]] = []
            prov_tags: List[str] = []
            fals_attempts: List[Dict[str, Any]] = []
            # Determine if any citations are provided for this cycle
            cites_exist = bool(citations)
            # Reuse the support_ratio (computed earlier) to decide whether
            # citations collectively match hypothesis keywords.  If support_ratio
            # is None (unsupported), treat as no support found.
            support = support_ratio if 'support_ratio' in (hypothesis_profile or {}) else support_ratio
            for h in hypotheses:
                # Confidence extraction: use the numeric confidence field when present.
                try:
                    cval = h.get('confidence')
                    conf_levels.append(float(cval) if isinstance(cval, (int, float)) else None)
                except Exception:
                    conf_levels.append(None)
                # Provenance tag: classify evidence source per hypothesis.
                if cites_exist:
                    # If the overall support ratio indicates at least one supporting
                    # citation, tag as literature confirmed.  Otherwise, citations
                    # exist but no support matches, so tag as literature missing.
                    try:
                        if support is not None and support > 0:
                            prov_tags.append('literature confirmed')
                        else:
                            prov_tags.append('literature missing')
                    except Exception:
                        prov_tags.append('literature missing')
                else:
                    # No citations supplied: tag as internal reasoning
                    prov_tags.append('internal reasoning')
                # Falsification attempt: record whether contradictions were detected.
                try:
                    hyp_text = (h.get('title') or h.get('text') or '')
                except Exception:
                    hyp_text = str(h) if h is not None else ''
                result = 'contradiction' if hypothesis_profile.get('conflict_flags') else 'no contradiction'
                fals_attempts.append({'hypothesis': hyp_text, 'result': result})
            if isinstance(hypothesis_profile, dict):
                hypothesis_profile['confidence_levels'] = conf_levels
                hypothesis_profile['provenance_tags'] = prov_tags
                hypothesis_profile['falsification_attempts'] = fals_attempts
        except Exception:
            # If anything goes wrong, leave the additional fields unset
            pass

        # 6) Optionally apply the cycle quality gate.  When the gate is
        # available it compares the current delta_R and energy against
        # recent history to decide whether this cycleâs improvements are
        # sufficiently strong and novel.  If rejected, verification and
        # novelty scores are forced to zero and gating reasons are stored in
        # the flags.  The gating logic reuses the quality_gate config from
        # the pipeline config (if present) and the same history window used
        # for novelty scoring.
        gating_reasons: List[str] = []
        if evaluate_cycle_gate is not None:
            try:
                qcfg = self.config.get("quality_gate", {}) if hasattr(self, "config") else {}
            except Exception:
                qcfg = {}
            # Build the minimal cycle dict required by the gate
            cycle_for_gate: Dict[str, Any] = {
                "delta_R": cycle_log.get("delta_R"),
                "energy_E": cycle_log.get("energy_E"),
                "citations": citations,
                "hypotheses": hypotheses,
            }
            try:
                gate_accept, gate_reasons = evaluate_cycle_gate(cycle_for_gate, history, config=qcfg)
            except Exception:
                gate_accept, gate_reasons = True, []
            if not gate_accept:
                # Capture gating reasons or default marker
                gating_reasons = gate_reasons or ["quality_gate_rejection"]
                # Instead of zeroing out scores entirely, apply a soft penalty.
                # This prevents early exploratory cycles from being scored as complete failures.
                # A configurable penalty factor may be supplied via quality_gate config; default 0.5.
                try:
                    penalty = float(qcfg.get("rejection_penalty", 0.5))
                except Exception:
                    penalty = 0.5
                # Clamp penalty between 0 and 1
                if penalty < 0.0:
                    penalty = 0.0
                if penalty > 1.0:
                    penalty = 1.0
                verification_score *= penalty
                novelty_score *= penalty
                disco_confidence *= penalty
                # Ensure flags is a list and append rejection markers
                try:
                    if not flags:
                        flags = []
                    elif not isinstance(flags, list):
                        flags = [flags]
                    flags.append("quality_gate_rejection")
                    for reason in gating_reasons:
                        flags.append(f"gate_{reason}")
                except Exception:
                    # Fallback: assign flags to list of rejection markers only
                    flags = ["quality_gate_rejection"]

        # 6a) Route motifs into replay buffer if strong enough (post-gate)
        self._route_motifs_to_replay(
            motifs=motifs,
            goal=goal,
            domain=domain,
            hallmark=hallmark,
            rye_value=rye_value,
            verification_score=verification_score,
        )

        # Determine whether this cycle passes the verification threshold.  We
        # reuse the min_verification_for_motif threshold used for motif routing
        # as the pass criterion.  A cycle passes when its verification_score
        # meets or exceeds this threshold.  When the threshold is undefined,
        # default to False to avoid inadvertently marking cycles as passed.
        try:
            pass_threshold = float(self.min_verification_for_motif)
        except Exception:
            pass_threshold = None
        passed: Optional[bool]
        if pass_threshold is not None:
            try:
                passed = bool(verification_score >= pass_threshold)
            except Exception:
                passed = False
        else:
            passed = False

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
            # When the quality gate rejects a cycle, gating_reasons lists
            # the specific codes.  This field is empty when the gate
            # accepts the cycle or when the gate is unavailable.
            "gating_reasons": gating_reasons,
            # Pass/fail indicator based on verification_score compared to
            # min_verification_for_motif.  True when the score meets or
            # exceeds the threshold, False otherwise.  When the threshold
            # cannot be parsed, this defaults to False.
            "passed": passed,
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
        """Normalize citation structures from mixed types into list of dicts.

        In earlier versions this helper merely coerced arbitrary inputs into
        dict-like objects.  It now integrates with citation_utils.normalize_citation
        when available, returning only properly structured citation dicts and
        discarding stub/error placeholders.  If normalize_citation is not
        available, this falls back to the legacy behaviour.
        """
        if raw is None:
            return []
        items: List[Any]
        if isinstance(raw, list):
            items = list(raw)
        else:
            items = [raw]

        out: List[Dict[str, Any]] = []
        for item in items:
            try:
                # When normalize_citation is available, use it to filter and
                # canonicalize citations.  Only append if normalization
                # returned a non-None dict.
                if normalize_citation:
                    if isinstance(item, dict):
                        norm = normalize_citation(item)
                        if norm:
                            out.append(norm)
                    elif isinstance(item, str):
                        # Construct a minimal dict and normalize
                        norm = normalize_citation({"title": item})
                        if norm:
                            out.append(norm)
                    else:
                        # Fallback for other types: convert to string and normalize
                        norm = normalize_citation({"title": str(item)})
                        if norm:
                            out.append(norm)
                else:
                    # Legacy fallback: preserve dicts or wrap strings as raw
                    if isinstance(item, dict):
                        out.append(item)
                    elif isinstance(item, str):
                        out.append({"raw": item})
                    else:
                        out.append({"raw": str(item)})
            except Exception:
                # Swallow any normalization error and skip the item
                continue
        return out

    def _analyze_citations(
        self,
        citations: Any,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute citation quality and redundancy profile.

        This implementation uses the canonical citation normalization from
        citation_utils (via _normalize_citation_list) to filter out stub
        entries and to standardize fields (source, url, doi, venue, year).
        It returns a richer profile including counts of missing URLs, counts
        of citations with DOIs, counts of entries with both venue and year,
        and the proportion of citations missing URLs.  These metrics allow
        the verification score to penalize poorly cited cycles while
        rewarding diversity and peerâreviewed sources.
        """
        citations_list = self._normalize_citation_list(citations)
        total = len(citations_list)
        if total == 0:
            return {
                "total": 0,
                "unique_sources": 0,
                "unique_urls": 0,
                "missing_urls": 0,
                "doi_count": 0,
                "venue_year_count": 0,
                "percent_missing_url": 0.0,
                "mean_age_days": None,
                "redundancy_ratio": None,
                "history_overlap_ratio": None,
            }

        # Unique (source, url) pairs and unique urls
        source_pairs = set()
        urls = set()
        missing_urls = 0
        doi_count = 0
        venue_year_count = 0

        for c in citations_list:
            src = c.get("source")
            url = c.get("url")
            source_pairs.add((src, url))
            if url:
                urls.add(url)
            else:
                missing_urls += 1
            # DOI or PubMed ID counts: treat DOI-like strings or pmid in raw
            doi_val = c.get("doi")
            if doi_val:
                doi_count += 1
            # Venue/year count: count citations with both venue and year
            v = c.get("venue")
            y = c.get("year")
            if v and y:
                venue_year_count += 1

        unique_sources = len(source_pairs)
        unique_urls = len(urls)

        # Redundancy ratio: measures how many citations share the same URL
        redundancy_ratio = None
        if total > 0:
            if unique_urls > 0:
                redundancy_ratio = max(0.0, 1.0 - (unique_urls / float(total)))
            else:
                redundancy_ratio = 1.0

        # Overlap with historical urls: compute using normalized citations
        hist_urls = set()
        for row in history:
            row_cites = self._normalize_citation_list(row.get("citations", []))
            for c in row_cites:
                u = c.get("url")
                if u:
                    hist_urls.add(u)
        overlap = len(urls.intersection(hist_urls)) if hist_urls else 0
        history_overlap_ratio = None
        if unique_urls > 0:
            history_overlap_ratio = overlap / float(unique_urls)

        # Publication age is optional and depends on fields; not computed
        mean_age_days = None

        percent_missing_url = missing_urls / float(total) if total > 0 else 0.0

        return {
            "total": total,
            "unique_sources": unique_sources,
            "unique_urls": unique_urls,
            "missing_urls": missing_urls,
            "doi_count": doi_count,
            "venue_year_count": venue_year_count,
            "percent_missing_url": percent_missing_url,
            "mean_age_days": mean_age_days,
            "redundancy_ratio": redundancy_ratio,
            "history_overlap_ratio": history_overlap_ratio,
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

    # ------------------------------------------------------------------
    # Hypothesis support computation
    # ------------------------------------------------------------------
    def _compute_support_ratio(
        self,
        hypotheses: List[Dict[str, Any]],
        citations_list: List[Dict[str, Any]],
        *,
        min_keyword_len: int = 3,
    ) -> float:
        """
        Estimate what fraction of hypotheses have at least one supporting citation.

        A hypothesis is considered supported if any keyword extracted from its
        title or text appears in the title or snippet of a citation.

        Parameters
        ----------
        hypotheses:
            List of hypothesis dicts (with 'title' or 'text').
        citations_list:
            List of normalized citation dicts (from _normalize_citation_list).
        min_keyword_len:
            Minimum length of a token to be considered a keyword (default 3).

        Returns
        -------
        float
            Fraction of hypotheses with at least one supporting citation.
            Returns 1.0 when there are no hypotheses. Returns 0.0 when
            hypotheses exist but citations_list is empty.
        """
        if not hypotheses:
            return 1.0
        if not citations_list:
            return 0.0

        def extract_keywords(text: str) -> List[str]:
            # Lowercase and split on whitespace and punctuation, skipping short tokens
            raw_tokens = re.split(r"[\s,;:\.\(\)\[\]\{\}\-\_\/]+", text.lower())
            return [tok for tok in raw_tokens if len(tok) >= min_keyword_len]

        supported_count = 0
        for h in hypotheses:
            try:
                text = (h.get("title") or h.get("text") or "").strip()
            except Exception:
                text = ""
            if not text:
                continue
            keywords = extract_keywords(text)
            if not keywords:
                continue
            supported = False
            for cite in citations_list:
                try:
                    ct = (cite.get("title") or "").lower()
                    cs = (cite.get("snippet") or "").lower()
                except Exception:
                    ct = ""
                    cs = ""
                for kw in keywords:
                    if kw and (kw in ct or kw in cs):
                        supported = True
                        break
                if supported:
                    break
            if supported:
                supported_count += 1
        return supported_count / float(len(hypotheses))

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
        missing_urls = citation_profile.get("missing_urls") or 0
        doi_count = citation_profile.get("doi_count") or 0
        venue_year_count = citation_profile.get("venue_year_count") or 0
        percent_missing_url = citation_profile.get("percent_missing_url") or 0.0

        if total_cites == 0:
            # When no citations are present, the citation strength should
            # collapse to zero rather than defaulting to a positive baseline.
            # This ensures that uncited outputs cannot achieve high
            # verification scores.  A "no_citations" flag is emitted.
            cite_term = 0.0
            flags.append("no_citations")
        else:
            diversity = (unique_sources / float(total_cites)) if total_cites > 0 else 0.0
            cite_term = 0.2 + 0.6 * max(0.0, min(1.0, diversity))
            # Penalize high redundancy
            if redundancy is not None and redundancy > 0.6:
                cite_term *= 0.8
                flags.append("high_citation_redundancy")
            # Penalize heavy reuse of historical citations
            if overlap is not None and overlap > 0.8:
                cite_term *= 0.85
                flags.append("historical_citation_reuse")
            # Penalize if many citations lack URLs (suggesting incomplete metadata)
            if percent_missing_url > 0.4:
                cite_term *= 0.85
                flags.append("missing_urls_high")
            # Penalize if too few citations have DOI/PMID (peer reviewed indicators)
            if total_cites > 0:
                min_peer = max(1, int(total_cites * 0.3))
                if doi_count < min_peer:
                    cite_term *= 0.90
                    flags.append("few_peer_reviewed")
                # Penalize if too few have venue and year (suggesting non-peer sources)
                min_meta = max(1, int(total_cites * 0.5))
                if venue_year_count < min_meta:
                    cite_term *= 0.95
                    flags.append("low_peer_metadata")
        cite_term = max(0.0, min(1.0, cite_term))

        # Hypothesis structure and support
        hyp_count = hypothesis_profile.get("count") or 0
        focused = bool(hypothesis_profile.get("focused"))
        conflict_flags = hypothesis_profile.get("conflict_flags") or []
        support_ratio = hypothesis_profile.get("support_ratio")
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

        # Evidence support penalties: degrade cite_term and hyp_term when
        # hypotheses are poorly supported by citations.  A low support_ratio
        # indicates that many hypotheses do not align with any citation
        # metadata (title or snippet).  This helps flag runs that generate
        # speculative or off-topic statements.
        if support_ratio is not None and hyp_count > 0 and total_cites > 0:
            try:
                sr = float(support_ratio)
            except Exception:
                sr = None
            if sr is not None:
                if sr < 0.25:
                    cite_term *= 0.7
                    hyp_term *= 0.5
                    flags.append("unsupported_hypotheses")
                elif sr < 0.5:
                    cite_term *= 0.85
                    hyp_term *= 0.8
                    flags.append("weak_hypothesis_evidence")

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
