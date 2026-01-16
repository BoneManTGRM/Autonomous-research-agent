"""Simple hypothesis engine for the autonomous research agent (90-day safe).

This module is deliberately:
- LLM-free (pure Python, deterministic).
- Stateless (no global counters that can drift over 24-90 day runs).
- Swarm-safe (no shared mutable state across agents).
- Domain-aware (optional hooks for longevity / math / general).
- Role-aware (uses swarm role hints to shape hypothesis style).

It generates plausible, testable hypotheses from:
- the current goal,
- existing notes,
- citations (web, PubMed, Semantic Scholar).

TGRM uses this in the REPAIR / VERIFY phases to propose new directions.

Tier and RYE awareness:
- Each hypothesis carries:
  - novelty, rye_relevance, priority, delta_r_hint
  - a simple tier_label (tier1_candidate, tier2_candidate, tier3_candidate, or None)
- Optional intelligence_profile lets you tune how strict the engine is:
  - safety_bias: "low" | "medium" | "high"
  - discovery_focus: 0.0-1.0 (more or fewer hypotheses)
  - tier3_bias: 0.0-1.0 (push toward only strongest hypotheses)

This base version is fully self-contained and safe for very long autonomous runs.
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

# ---------------------------------------------------------------------
# Small domain keyword hints (mirrors presets but kept local and static)
# ---------------------------------------------------------------------
LONGEVITY_DOMAIN_KEYWORDS = {
    "longevity",
    "healthspan",
    "aging",
    "ageing",
    "senescence",
    "autophagy",
    "mtor",
    "nad+",
    "mitochondria",
    "mitochondrial",
    "rapamycin",
    "metformin",
    "biomarker",
    "telomere",
    "telomeres",
    "inflammaging",
}

MATH_DOMAIN_KEYWORDS = {
    "theorem",
    "axiom",
    "lemma",
    "proof",
    "stability",
    "equilibrium",
    "lyapunov",
    "markov",
    "martingale",
    "measure",
    "functional",
    "operator",
    "entropy",
    "information",
}

# Framework terms we usually do not want as main drivers in hypotheses
FRAMEWORK_STOP_WORDS = {
    "reparodynamics",
    "rye",
    "tgrm",
    "repair",
    "repairs",
    "stability",
    "stable",
    "system",
    "systems",
    "autonomous",
}


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def _normalize_token(t: str) -> str:
    return t.strip(".,;:!?()[]{}\"'").lower()


def _shorten_goal_for_topic(goal: str, max_len: int = 120) -> str:
    """Shorten a very long goal to a compact topic string.

    This avoids echoing giant prompt-like goals inside hypotheses.
    """
    g = " ".join(str(goal).strip().split())
    if len(g) <= max_len:
        return g or "the current research topic"
    # Prefer to cut at a sentence or phrase boundary
    for sep in [". ", "; ", ", "]:
        if sep in g and len(g.split(sep)[0]) >= 40:
            head = g.split(sep)[0]
            return head[:max_len].rstrip() + "..."
    return g[:max_len].rstrip() + "..."


def _shares_too_much_with_goal(h_text: str, goal: str, threshold: float = 0.75) -> bool:
    """Heuristic to avoid hypotheses that mostly rephrase the goal.

    We compare token overlap between hypothesis and goal.
    If overlap ratio is too high, treat it as low-novelty echo.
    """
    if not goal:
        return False
    h_tokens = {t for t in _normalize_token(h_text).split() if len(t) >= 4}
    g_tokens = {t for t in _normalize_token(goal).split() if len(t) >= 4}
    if not h_tokens or not g_tokens:
        return False
    inter = h_tokens.intersection(g_tokens)
    ratio = len(inter) / float(max(1, len(h_tokens)))
    return ratio >= threshold


def _novelty_score(
    h_text: str,
    goal: str,
    core_terms: List[str],
) -> float:
    """Crude novelty score in [0, 1].

    High score means:
    - uses some domain/keyword structure
    - is not just repeating goal language
    """
    if not h_text:
        return 0.0

    h_tokens = {t for t in _normalize_token(h_text).split() if len(t) >= 4}
    g_tokens = {t for t in _normalize_token(goal).split() if len(t) >= 4}
    core_set = {t for t in core_terms if len(t) >= 4}

    if not h_tokens:
        return 0.0

    # Overlap penalty with goal
    overlap = h_tokens.intersection(g_tokens)
    overlap_ratio = len(overlap) / float(max(1, len(h_tokens)))

    # Reward for using core keywords that are not just framework buzzwords
    useful_core = h_tokens.intersection(core_set)
    core_ratio = len(useful_core) / float(max(1, len(h_tokens)))

    # Map to [0, 1]
    # Prefer: low overlap, high core_ratio
    novelty = max(0.0, min(1.0, (0.6 * (1.0 - overlap_ratio) + 0.4 * core_ratio)))
    return novelty


def _estimate_rye_relevance(text: str) -> float:
    """Estimate how directly a hypothesis speaks to RYE style ideas.

    This is a small heuristic in [0, 1] based on presence of terms
    related to repair efficiency, energy, tradeoffs, and equilibrium.
    """
    if not text:
        return 0.0

    tokens = [_normalize_token(t) for t in text.split() if t]
    if not tokens:
        return 0.0

    rye_terms = {
        "repair",
        "repairs",
        "yield",
        "efficiency",
        "efficient",
        "energy",
        "cost",
        "costs",
        "tradeoff",
        "tradeoffs",
        "equilibrium",
        "equilibria",
        "stable",
        "stability",
        "resilience",
        "drift",
    }

    hits = 0
    for t in tokens:
        if t in rye_terms:
            hits += 1

    # Simple normalized count
    return max(0.0, min(1.0, hits / float(max(1, len(tokens)))))


def _estimate_delta_r_hint(
    novelty: float,
    rye_relevance: float,
    domain: Optional[str],
    role: Optional[str],
) -> float:
    """Deterministic hint for how much delta R this hypothesis might deliver if confirmed.

    This is NOT delta R itself, just a small scalar that can feed into
    compute_delta_r(...) as an extra signal or be logged with the hypothesis.
    """
    d = (domain or "").lower()
    r = (role or "").lower()

    # Base potential from novelty and RYE relevance
    base = 0.4 * novelty + 0.6 * rye_relevance  # tilt toward direct RYE language
    # Domain scaling: longevity/math are harder, so each good hypothesis is more valuable
    if d in {"longevity", "math"}:
        base *= 1.2

    # Role scaling: explorer = more speculative, integrator = more actionable
    if r == "explorer":
        base *= 0.9
    elif r == "integrator":
        base *= 1.1
    elif r == "critic":
        base *= 1.0
    elif r == "researcher":
        base *= 1.0

    # Map to a gentle delta R hint range, for example [0.0, 2.0]
    return max(0.0, min(2.0, 2.0 * base))


def _has_contradiction_signals(notes: List[Dict[str, Any]]) -> bool:
    """Detect whether the current notes contain contradiction markers.

    This lets us bias hypothesis style toward integrator or critic tones
    in cases where the literature is already internally inconsistent.
    """
    for n in notes:
        content = str(n.get("content", "")).lower()
        if "contradiction" in content or "inconsistent" in content or "conflict" in content:
            return True
    return False


# ---------------------------------------------------------------------
# Keyword extraction helpers (notes + citations)
# ---------------------------------------------------------------------
def _extract_keywords_from_notes(
    notes: List[Dict[str, Any]],
    max_keywords: int = 8,
) -> List[str]:
    """Naive keyword extractor based on frequency and length.

    Heuristics:
        - ignore very short tokens (< 5 characters)
        - lower-case + strip punctuation
        - count frequencies across all notes
        - skip framework-level generic words
        - return the most common terms

    This is intentionally simple and deterministic so it remains stable
    across 24-90 day runs and across multiple swarm agents.
    """
    freq: Dict[str, int] = {}
    for n in notes:
        content = str(n.get("content", ""))
        tokens = content.replace("\n", " ").split()
        for t in tokens:
            cleaned = _normalize_token(t)
            if len(cleaned) < 5:
                continue
            # Skip obvious filler / stop-ish words (very small hand list)
            if cleaned in {
                "which",
                "their",
                "there",
                "where",
                "about",
                "these",
                "those",
                "within",
                "between",
                "could",
                "should",
                "would",
                "based",
                "using",
                "study",
                "effect",
                "effects",
                "model",
                "models",
                "analysis",
            }:
                continue
            if cleaned in FRAMEWORK_STOP_WORDS:
                continue
            freq[cleaned] = freq.get(cleaned, 0) + 1

    # Sort by frequency (and then alphabetically for determinism)
    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for (t, _) in sorted_terms[:max_keywords]]


def _extract_keywords_from_citations(
    citations: List[Dict[str, Any]],
    max_keywords: int = 8,
) -> List[str]:
    """Extract candidate keywords from citation titles and snippets.

    This complements note-based extraction so that even very short notes
    can still yield usable hypotheses if citations are rich.
    """
    freq: Dict[str, int] = {}
    for c in citations:
        title = str(c.get("title", ""))
        snippet = str(c.get("snippet", ""))
        text = f"{title} {snippet}"
        tokens = text.replace("\n", " ").split()
        for t in tokens:
            cleaned = _normalize_token(t)
            if len(cleaned) < 5:
                continue
            if cleaned in {
                "study",
                "effect",
                "effects",
                "based",
                "using",
                "analysis",
                "model",
                "models",
                "repair",
                "stability",
                "system",
                "systems",
                "results",
                "methods",
            }:
                continue
            if cleaned in FRAMEWORK_STOP_WORDS:
                continue
            freq[cleaned] = freq.get(cleaned, 0) + 1

    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for (t, _) in sorted_terms[:max_keywords]]


def _boost_domain_terms(
    keywords: List[str],
    domain: Optional[str],
    limit: int,
) -> List[str]:
    """Boost domain-relevant terms toward the front of the keyword list.

    For longevity/math, if domain-specific markers appear, we want them
    favored as k1/k2 seeds in hypothesis templates.
    """
    d = (domain or "").lower()
    if not keywords:
        return keywords

    if d == "longevity":
        domain_set = LONGEVITY_DOMAIN_KEYWORDS
    elif d == "math":
        domain_set = MATH_DOMAIN_KEYWORDS
    else:
        return keywords[:limit]

    domain_terms: List[str] = []
    other_terms: List[str] = []

    for k in keywords:
        if _normalize_token(k) in domain_set:
            domain_terms.append(k)
        else:
            other_terms.append(k)

    merged = domain_terms + other_terms
    # Preserve original relative order within each group
    dedup_seen = set()
    out: List[str] = []
    for k in merged:
        if k in dedup_seen:
            continue
        dedup_seen.add(k)
        out.append(k)
        if len(out) >= limit:
            break
    return out


def _merge_keywords(
    note_terms: List[str],
    cit_terms: List[str],
    domain: Optional[str],
    limit: int = 10,
) -> List[str]:
    """Merge keywords from notes and citations, preserving order, uniqueness,
    and optionally boosting domain-specific markers."""
    seen = set()
    merged: List[str] = []

    for term in note_terms + cit_terms:
        if term in seen:
            continue
        seen.add(term)
        merged.append(term)
        if len(merged) >= limit:
            break

    # Apply domain-aware boosting if relevant
    if not merged:
        return merged
    return _boost_domain_terms(merged, domain=domain, limit=limit)


# ---------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------
def _infer_domain_from_text(
    goal: str,
    notes: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
) -> Optional[str]:
    """Infer a likely domain if none is explicitly provided.

    This is deterministic and uses simple keyword counts over
    goal, notes, and citation titles/snippets.
    """
    text_parts: List[str] = [str(goal or "")]
    for n in notes:
        text_parts.append(str(n.get("content", "")))
    for c in citations:
        text_parts.append(str(c.get("title", "")))
        text_parts.append(str(c.get("snippet", "")))

    full_text = " ".join(text_parts).lower()
    if not full_text.strip():
        return None

    lon_count = 0
    for kw in LONGEVITY_DOMAIN_KEYWORDS:
        if kw in full_text:
            lon_count += 1

    math_count = 0
    for kw in MATH_DOMAIN_KEYWORDS:
        if kw in full_text:
            math_count += 1

    # Simple decision rule: whichever count is larger, if it is clearly non-zero
    if lon_count == 0 and math_count == 0:
        return None
    if lon_count > math_count:
        return "longevity"
    if math_count > lon_count:
        return "math"
    # Tie case: do not force, let caller treat as general
    return None


# ---------------------------------------------------------------------
# Domain and role-aware templates
# ---------------------------------------------------------------------
def _get_templates_for_domain_and_role(
    domain: Optional[str],
    role: Optional[str],
) -> List[str]:
    """Return a pool of hypothesis templates based on domain and role.

    Domain-specific templates encourage more grounded, testable ideas.
    Role-specific variants adjust tone:
      - researcher: generative mechanisms
      - critic: tension / uncertainty oriented
      - explorer: cross-domain / analogy
      - integrator: combining and ranking
    """
    d = (domain or "").lower()
    r = (role or "").lower()

    # Longevity domain
    if d == "longevity":
        if r == "critic":
            return [
                "Existing claims that changes in {k1} drive {topic} are overstated when {k2} is not controlled for.",
                "Heterogeneous outcomes in {topic} are explained by unmeasured interactions between {k1} and {k2}.",
                "If {k1}-focused interventions fail when {k2} is dysregulated, then {k1} is a secondary rather than primary driver in {topic}.",
                "Reported longevity benefits linked to {k1} disappear after adjusting for {k2}-related confounders in {topic}.",
            ]
        if r == "explorer":
            return [
                "Mechanisms involving {k1} in non-aging fields transfer to {topic} via shared {k2}-linked pathways.",
                "Interventions that modulate {k1} in other chronic conditions improve {topic} when {k2} is also affected.",
                "The role of {k1} in stress adaptation suggests cross-domain analogies that reorganize how {topic} is targeted when {k2} shifts.",
            ]
        if r == "integrator":
            return [
                "A combined model where {k1} tracks upstream stress and {k2} tracks repair response explains multi-organ outcomes in {topic}.",
                "Prioritizing interventions that improve both {k1} and {k2} defines a high RYE intervention stack for {topic}.",
                "Differences in {topic} trajectories across cohorts are captured by a two-axis frame based on {k1} and {k2}.",
            ]
        # default longevity templates (researcher, planner, synthesizer, etc.)
        return [
            # Assertive templates avoid vague qualifiers like "may", "might", or "could"
            "Changes in {k1} correlate with improvements in {topic} via modulation of {k2}-linked pathways.",
            "In {topic}, tracking {k1} as a biomarker alongside {k2} improves prediction of repair or resilience outcomes.",
            "Interventions that reduce adverse trends in {k1} while stabilizing {k2} yield higher RYE in {topic}.",
            "The interaction between {k1} and {k2} explains heterogeneous responses to longevity interventions in {topic}.",
            "Combining {k1}-focused protocols with monitoring of {k2} identifies individuals with superior self-repair capacity in {topic}.",
        ]

    # Math domain
    if d == "math":
        if r == "critic":
            return [
            "If {topic} cannot ensure monotonic change in a functional built from {k1} and {k2}, its claimed stability properties fail.",
            "Apparent equivalence between {topic} and a {k1}/{k2}-based model breaks when boundary conditions are made explicit.",
            "Any theorem for {topic} that ignores the coupling between {k1} and {k2} risks missing a critical instability mode.",
        ]
        if r == "integrator":
            return [
            "A unified formalism where {k1} encodes state and {k2} encodes repair effort places {topic} within an existing stability theory.",
            "Expressing {topic} as the evolution of a functional in {k1} and {k2} reveals hidden conservation or dissipation laws.",
        ]
        # default math templates
        return [
            "A formal definition that links {k1} and {k2} yields a stability criterion for {topic}.",
            "Constructing a Lyapunov-like functional using {k1} and {k2} proves convergence for {topic}.",
            "Recasting {topic} in terms of {k1} and {k2} reveals equivalence to an existing stability framework.",
            "Non-linear coupling between {k1} and {k2} explains phase transitions or repair plateaus in {topic}.",
            "An information-theoretic view where {k1} encodes state and {k2} encodes repair signal formalizes RYE in {topic}.",
        ]

    # General / fallback domain
    if r == "critic":
        return [
            "Claims that {k1} is a primary driver of {topic} are confounded by unmeasured changes in {k2}.",
            "Inconsistent findings about {k1} in {topic} suggest that {k2} is a hidden moderator.",
            "If interventions on {k1} do not consistently improve {topic} when {k2} is unfavorable, then {k1} is not a robust repair lever.",
        ]
    if r == "explorer":
        return [
            "Patterns involving {k1} in other domains transfer to {topic} when similar {k2}-related constraints are present.",
            "Unexpected correlations between {k1} and {k2} in adjacent fields inspire alternative models for {topic}.",
        ]
    if r == "integrator":
        return [
            "A compact model where {k1} summarizes stress and {k2} summarizes repair clarifies tradeoffs in {topic}.",
            "Integrating evidence about {k1} and {k2} from multiple sources resolves contradictions in {topic}.",
        ]

    # Default general templates
    return [
        # Assertive general templates without weak qualifiers
        "A key driver of {topic} is the interaction between {k1} and {k2}.",
        "{topic} is strongly influenced by changes in {k1} under conditions affecting {k2}.",
        "Interventions that modify {k1} increase stability or repair efficiency in {topic}.",
        "Contradictions in the literature about {k1} and {k2} indicate a non-linear relationship affecting {topic}.",
        "Combined evidence from multiple sources shows that {k1} is an upstream regulator in {topic}.",
    ]


# ---------------------------------------------------------------------
# Hypothesis classification helpers
# ---------------------------------------------------------------------
def _classify_hypothesis(
    text: str,
    domain: Optional[str],
    role: Optional[str],
    k1: str,
    k2: str,
) -> Dict[str, Any]:
    """Assign a simple kind/focus label to a hypothesis.

    This does not change behavior of the engine but gives richer
    metadata for reports and future agents.
    """
    d = (domain or "").lower()
    r = (role or "").lower()
    t = text.lower()

    kind = "general"
    focus = "unspecified"

    if d == "longevity":
        if "biomarker" in t or "marker" in t:
            kind = "biomarker"
        elif "intervention" in t or "protocol" in t or "stack" in t:
            kind = "intervention"
        elif "pathway" in t or "mechanism" in t:
            kind = "mechanism"
        else:
            kind = "longevity_pattern"

    elif d == "math":
        if "definition" in t or "axiom" in t:
            kind = "formal_definition"
        elif "theorem" in t or "proof" in t or "convergence" in t:
            kind = "theorem_or_proof"
        elif "functional" in t or "lyapunov" in t:
            kind = "stability_functional"
        else:
            kind = "math_structure"

    else:
        if "model" in t or "framework" in t:
            kind = "model"
        elif "intervention" in t or "protocol" in t:
            kind = "intervention"
        else:
            kind = "relationship"

    if r == "critic":
        focus = "tension_or_gap"
    elif r == "explorer":
        focus = "cross_domain"
    elif r == "integrator":
        focus = "integration"
    elif r == "researcher":
        focus = "mechanism_or_pattern"
    else:
        focus = "unspecified"

    return {
        "kind": kind,
        "focus": focus,
        "k1": k1,
        "k2": k2,
    }


def _build_tags(
    priority: float,
    rye_relevance: float,
    novelty: float,
    contradiction_sensitive: bool,
    domain: Optional[str],
    role: Optional[str],
    swarm_size: Optional[int] = None,
    run_id: Optional[str] = None,
    cycle_index: Optional[int] = None,
    emit_events: bool = False,
    tier_label: Optional[str] = None,
) -> List[str]:
    """Lightweight tags for downstream filtering or visualization."""
    tags: List[str] = []

    if priority >= 0.7:
        tags.append("high_priority")
    elif priority <= 0.3:
        tags.append("low_priority")

    if rye_relevance >= 0.4:
        tags.append("rye_core")
    elif rye_relevance <= 0.1:
        tags.append("weak_rye_link")

    if novelty >= 0.5:
        tags.append("high_novelty")
    elif novelty <= 0.15:
        tags.append("low_novelty")

    if contradiction_sensitive:
        tags.append("contradiction_sensitive")

    d = (domain or "").lower()
    r = (role or "").lower()
    if d:
        tags.append(f"domain:{d}")
    if r:
        tags.append(f"role:{r}")

    if swarm_size is not None:
        tags.append(f"swarm_size:{swarm_size}")

    if tier_label:
        tags.append(tier_label)

    return tags


def _infer_tier_from_scores(
    priority: float,
    delta_r_hint: float,
) -> Optional[str]:
    """Rough tier style hint from priority and delta_r_hint.

    This is deterministic and only uses local hypothesis metrics.
    """
    # Normalize delta_r_hint into an approximate [0,1] band
    dr_norm = max(0.0, min(1.0, delta_r_hint / 2.0))

    # Emphasize high priority and reasonable delta_r_hint
    combined = 0.6 * priority + 0.4 * dr_norm

    if combined >= 0.9:
        return "tier3_candidate"
    if combined >= 0.75:
        return "tier2_candidate"
    if combined >= 0.6:
        return "tier1_candidate"
    return None


def _adapt_from_intelligence_profile(
    max_hypotheses: int,
    domain: Optional[str],
    intelligence_profile: Optional[Dict[str, Any]],
) -> Tuple[int, float]:
    """Adjust max_hypotheses and novelty threshold based on intelligence profile.

    Returns:
        (adjusted_max_hypotheses, novelty_threshold)
    """
    max_h = int(max_hypotheses)
    novelty_threshold = 0.10

    if domain and domain.lower() in {"longevity", "math"}:
        novelty_threshold = 0.18

    if not intelligence_profile:
        return max_h, novelty_threshold

    safety_bias = str(intelligence_profile.get("safety_bias", "medium")).lower()
    discovery_focus = float(intelligence_profile.get("discovery_focus", 0.5))
    tier3_bias = float(intelligence_profile.get("tier3_bias", 0.0))

    # Safety bias raises novelty requirement and reduces count
    if safety_bias == "high":
        novelty_threshold += 0.03
        max_h = max(1, int(max_h * 0.7))
    elif safety_bias == "low":
        novelty_threshold = max(0.05, novelty_threshold - 0.03)
        max_h = min(20, int(max_h * 1.2))

    # Discovery focus widens or narrows beam
    if discovery_focus > 0.75:
        max_h = min(20, int(max_h * 1.4))
        novelty_threshold = max(0.05, novelty_threshold - 0.02)
    elif discovery_focus < 0.3:
        max_h = max(1, int(max_h * 0.6))
        novelty_threshold += 0.02

    # Tier3 bias nudges toward only the very strongest
    if tier3_bias > 0.6:
        novelty_threshold += 0.03

    # Clamp
    max_h = max(1, min(32, max_h))
    novelty_threshold = max(0.05, min(0.35, novelty_threshold))

    return max_h, novelty_threshold


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
def generate_hypotheses(
    goal: str,
    notes: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    max_hypotheses: int = 5,
    domain: Optional[str] = None,
    role: Optional[str] = None,
    *,
    intelligence_profile: Optional[Dict[str, Any]] = None,
    swarm_size: Optional[int] = None,
    run_id: Optional[str] = None,
    cycle_index: Optional[int] = None,
    emit_events: bool = False,
) -> List[Dict[str, Any]]:
    """Generate simple, structured hypotheses.

    Args:
        goal:
            Human readable research goal.
        notes:
            List of note dicts (from MemoryStore).
        citations:
            List of citation dicts (from web / PubMed / Semantic Scholar).
        max_hypotheses:
            Upper bound on number of hypotheses to generate.
        domain:
            Optional domain tag: "general", "longevity", "math", etc.
        role:
            Optional swarm role label (researcher, critic, explorer, integrator, etc.).
            Used to mildly adjust confidence ranges and template style.
        intelligence_profile:
            Optional dict controlling strictness and breadth, e.g.:
                {
                  "safety_bias": "low" | "medium" | "high",
                  "discovery_focus": 0.0-1.0,
                  "tier3_bias": 0.0-1.0
                }
        swarm_size:
            Optional swarm size hint for tagging and analysis.

    Returns:
        List of:
            {
              "text": "...",
              "confidence": 0.30-0.90,
              "domain": "longevity" | "math" | "general" | None,
              "role": "researcher" | "critic" | ... | None,
              "keywords": [k1, k2],
              "novelty": 0.0-1.0,
              "rye_relevance": 0.0-1.0,
              "priority": 0.0-1.0,
              "score": 0.0-1.0,              # alias of priority for reports/sorting
              "delta_r_hint": float,         # estimated delta R contribution if confirmed
              "tier_label": "tier1_candidate" | "tier2_candidate" | "tier3_candidate" | None,
              "classification": {
                  "kind": "...",
                  "focus": "...",
                  "k1": "...",
                  "k2": "..."
              },
              "contradiction_sensitive": bool,
              "tags": [ ... ],
              "swarm_size": int | None,
              "intelligence_profile_used": { ... } | None,
            }

    Reparodynamics angle:
        These hypotheses are potential "repair targets" for future cycles.
        If later cycles confirm or refute them, delta_R reflects that, and
        long run RYE (delta_R / E) can track which directions were fruitful.
    """
    if not goal and not notes and not citations:
        return []

    # If domain not given, attempt a deterministic inference
    inferred_domain = _infer_domain_from_text(goal, notes, citations)
    if domain is None:
        domain = inferred_domain
    if domain is None:
        domain = "general"

    # If role not given, bias toward critic when contradictions appear,
    # otherwise default to researcher behavior.
    if role is None:
        if _has_contradiction_signals(notes):
            role = "critic"
        else:
            role = "researcher"

    # Adapt max_hypotheses and novelty threshold from intelligence profile
    max_hypotheses, base_novelty_threshold = _adapt_from_intelligence_profile(
        max_hypotheses,
        domain,
        intelligence_profile,
    )

    # 1) Build keyword pool from notes + citations
    note_terms = _extract_keywords_from_notes(notes, max_keywords=8)
    cit_terms = _extract_keywords_from_citations(citations, max_keywords=8)
    keywords = _merge_keywords(note_terms, cit_terms, domain=domain, limit=10)

    # Enforce evidence-driven keyword selection: prioritize terms found in citations
    # When citation terms exist, bias keywords toward them and ensure pairs include
    # at least one citation-derived term. This encourages hypotheses to be
    # anchored to actual evidence rather than speculative combinations.
    citation_keywords = set(cit_terms)

    # Fallback if everything is empty
    if not keywords:
        keywords = ["resilience", "autophagy", "equilibrium", "signal"]

    # 2) Choose template pool based on domain and role
    templates = _get_templates_for_domain_and_role(domain, role)

    # 3) Generate keyword pairs
    pairs: List[tuple[str, str]] = []
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            if keywords[i] == keywords[j]:
                continue
            # Only create pairs where at least one term is in citation_keywords
            if citation_keywords:
                if keywords[i] not in citation_keywords and keywords[j] not in citation_keywords:
                    continue
            pairs.append((keywords[i], keywords[j]))
    if not pairs:
        pairs = [(keywords[0], keywords[0])]

    # Single-axis focus: select top pairs to avoid topic drift
    # When multiple pairs remain, keep at most max_hypotheses pairs to
    # concentrate the hypothesis space. Prioritize pairs where both terms
    # appear in citations (citation driven) and then by lexical order.
    if pairs and len(pairs) > max_hypotheses:
        # Rank pairs: two citation terms highest, one citation term next
        def _pair_rank(pair: Tuple[str, str]) -> Tuple[int, str, str]:
            a, b = pair
            both_in = int(a in citation_keywords and b in citation_keywords)
            one_in = int((a in citation_keywords) or (b in citation_keywords))
            # Sorting descending by both_in, then one_in, then alphabetical for stability
            return (-both_in, -one_in, a + b)

        pairs = sorted(pairs, key=_pair_rank)[: max_hypotheses]

    # 4) Generate hypotheses with basic novelty and echo filtering
    base_goal = _shorten_goal_for_topic(goal) if goal else "the current research topic"
    hypo_list: List[Dict[str, Any]] = []
    used_texts = set()

    # Role-based confidence band (deterministic)
    base_low, base_high = 0.35, 0.85
    r = (role or "").lower()
    if r == "critic":
        base_low, base_high = 0.30, 0.75
    elif r == "explorer":
        base_low, base_high = 0.40, 0.90
    elif r == "integrator":
        base_low, base_high = 0.40, 0.85
    elif r == "researcher":
        base_low, base_high = 0.45, 0.85

    step = (base_high - base_low) / max(1, len(templates))

    contradiction_sensitive = _has_contradiction_signals(notes)

    idx = 0
    for (k1, k2) in pairs:
        tpl = templates[idx % len(templates)]
        text = tpl.format(topic=base_goal, k1=k1, k2=k2)

        # Avoid exact duplicates
        if text in used_texts:
            idx += 1
            continue

        # Novelty and echo checks
        novelty = _novelty_score(text, goal or "", keywords)

        # Raise or lower novelty threshold per intelligence profile and domain
        novelty_threshold = base_novelty_threshold

        if _shares_too_much_with_goal(text, goal or "", threshold=0.75):
            idx += 1
            continue
        if novelty < novelty_threshold:
            idx += 1
            continue

        # Deterministic confidence progression across hypotheses
        band_index = idx % len(templates)
        confidence = base_low + step * band_index
        confidence = max(0.0, min(1.0, confidence))

        rye_rel = _estimate_rye_relevance(text)

        # Simple composite priority for downstream sorting or filtering
        # Weight novelty slightly higher than direct RYE wording
        priority = max(
            0.0,
            min(1.0, 0.6 * novelty + 0.4 * rye_rel),
        )

        delta_r_hint = _estimate_delta_r_hint(
            novelty=novelty,
            rye_relevance=rye_rel,
            domain=domain,
            role=role,
        )

        tier_label = _infer_tier_from_scores(
            priority=priority,
            delta_r_hint=delta_r_hint,
        )

        classification = _classify_hypothesis(
            text=text,
            domain=domain,
            role=role,
            k1=k1,
            k2=k2,
        )

        tags = _build_tags(
            priority=priority,
            rye_relevance=rye_rel,
            novelty=novelty,
            contradiction_sensitive=contradiction_sensitive,
            domain=domain,
            role=role,
            swarm_size=swarm_size,
            tier_label=tier_label,
        )

        used_texts.add(text)
        hypo_list.append(
            {
                "text": text,
                "confidence": round(confidence, 2),
                "domain": (domain or None),
                "role": (role or None),
                "keywords": [k1, k2],
                "novelty": round(novelty, 3),
                "rye_relevance": round(rye_rel, 3),
                "priority": round(priority, 3),
                "score": round(priority, 3),  # alias for compatibility with report sorters
                "delta_r_hint": round(delta_r_hint, 3),
                "tier_label": tier_label,
                "classification": classification,
                "contradiction_sensitive": bool(contradiction_sensitive),
                "tags": tags,
                "swarm_size": swarm_size,
                "intelligence_profile_used": intelligence_profile or None,
            }
        )

        idx += 1
        if len(hypo_list) >= max_hypotheses:
            break


    # Optional: mirror hypotheses into the unified event stream for UI/report consumption.
    # This is a best-effort helper; core workers may also emit richer structured events.
    if emit_events and _log_event is not None and run_id is not None:
        try:
            for h in hypo_list:
                title = h.get("title")
                if not title:
                    cls = h.get("classification") or {}
                    k1 = cls.get("k1") if isinstance(cls, dict) else None
                    k2 = cls.get("k2") if isinstance(cls, dict) else None
                    if k1 and k2:
                        # Use ASCII arrow for compatibility
                        title = f"{k1} <-> {k2} constraint"
                    else:
                        title = "candidate hypothesis"
                _log_event(
                    run_id=str(run_id),
                    kind="candidate_hypothesis",
                    message=str(title),
                    level="info",
                    data={
                        "title": title,
                        "text": h.get("text"),
                        "goal": goal,
                        "domain": domain,
                        "cycle": cycle_index,
                        "novelty": h.get("novelty"),
                        "priority": h.get("priority"),
                        "score": h.get("score"),
                        "delta_r_hint": h.get("delta_r_hint"),
                        "tier_label": h.get("tier_label"),
                        "classification": h.get("classification"),
                        "tags": h.get("tags"),
                        "swarm_size": swarm_size,
                        "citations": citations,
                    },
                    role=role,
                    domain=domain,
                    cycle=cycle_index,
                )
        except Exception:
            pass

    return hypo_list
