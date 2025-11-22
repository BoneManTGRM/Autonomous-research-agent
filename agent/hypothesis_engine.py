"""Simple hypothesis engine for the autonomous research agent (90-day safe).

This module is deliberately:
- LLM-free (pure Python, deterministic).
- Stateless (no global counters that can drift over 24–90 day runs).
- Swarm-safe (no shared mutable state across agents).
- Domain-aware (optional hooks for longevity / math / general).
- Role-aware (uses swarm role hints to shape hypothesis style).

It generates plausible, testable hypotheses from:
- the current goal,
- existing notes,
- citations (web, PubMed, Semantic Scholar).

TGRM uses this in the REPAIR / VERIFY phases to propose new directions.
Later, you can upgrade this to call a language model, but this base
version is fully self-contained and safe for very long autonomous runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    across 24–90 day runs and across multiple swarm agents.
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
                "Existing claims that changes in {k1} drive {topic} may be overstated if {k2} is not controlled for.",
                "Heterogeneous outcomes in {topic} might be explained by unmeasured interactions between {k1} and {k2}.",
                "If {k1}-focused interventions fail when {k2} is dysregulated, then {k1} may be a secondary rather than primary driver in {topic}.",
                "Reported longevity benefits linked to {k1} might disappear after adjusting for {k2}-related confounders in {topic}.",
            ]
        if r == "explorer":
            return [
                "Mechanisms involving {k1} in non-aging fields could transfer to {topic} via shared {k2}-linked pathways.",
                "Interventions that modulate {k1} in other chronic conditions may unexpectedly improve {topic} when {k2} is also affected.",
                "The role of {k1} in stress adaptation suggests cross-domain analogies that could reorganize how {topic} is targeted when {k2} shifts.",
            ]
        if r == "integrator":
            return [
                "A combined model where {k1} tracks upstream stress and {k2} tracks repair response may explain multi-organ outcomes in {topic}.",
                "Prioritizing interventions that improve both {k1} and {k2} may define a high RYE intervention stack for {topic}.",
                "Differences in {topic} trajectories across cohorts might be captured by a two-axis frame based on {k1} and {k2}.",
            ]
        # default longevity templates (researcher, planner, synthesizer, etc.)
        return [
            "Changes in {k1} may correlate with improvements in {topic} via modulation of {k2}-linked pathways.",
            "In {topic}, tracking {k1} as a biomarker alongside {k2} could better predict repair or resilience outcomes.",
            "Interventions that reduce adverse trends in {k1} while stabilizing {k2} might yield higher RYE in {topic}.",
            "The interaction between {k1} and {k2} may explain heterogeneous responses to longevity interventions in {topic}.",
            "Combining {k1}-focused protocols with monitoring of {k2} might identify individuals with superior self-repair capacity in {topic}.",
        ]

    # Math domain
    if d == "math":
        if r == "critic":
            return [
                "If {topic} cannot ensure monotonic change in a functional built from {k1} and {k2}, its claimed stability properties may fail.",
                "Apparent equivalence between {topic} and a {k1}/{k2}-based model may break when boundary conditions are made explicit.",
                "Any theorem for {topic} that ignores the coupling between {k1} and {k2} risks missing a critical instability mode.",
            ]
        if r == "integrator":
            return [
                "A unified formalism where {k1} encodes state and {k2} encodes repair effort could place {topic} within an existing stability theory.",
                "Expressing {topic} as the evolution of a functional in {k1} and {k2} may reveal hidden conservation or dissipation laws.",
            ]
        # default math templates
        return [
            "A formal definition that links {k1} and {k2} could yield a stability criterion for {topic}.",
            "Constructing a Lyapunov-like functional using {k1} and {k2} may prove convergence for {topic}.",
            "Recasting {topic} in terms of {k1} and {k2} could reveal equivalence to an existing stability framework.",
            "Non-linear coupling between {k1} and {k2} might explain phase transitions or repair plateaus in {topic}.",
            "An information-theoretic view where {k1} encodes state and {k2} encodes repair signal may formalize RYE in {topic}.",
        ]

    # General / fallback domain
    if r == "critic":
        return [
            "Claims that {k1} is a primary driver of {topic} may be confounded by unmeasured changes in {k2}.",
            "Inconsistent findings about {k1} in {topic} suggest that {k2} might be a hidden moderator.",
            "If interventions on {k1} do not consistently improve {topic} when {k2} is unfavorable, then {k1} may not be a robust repair lever.",
        ]
    if r == "explorer":
        return [
            "Patterns involving {k1} in other domains might transfer to {topic} when similar {k2}-related constraints are present.",
            "Unexpected correlations between {k1} and {k2} in adjacent fields could inspire alternative models for {topic}.",
        ]
    if r == "integrator":
        return [
            "A compact model where {k1} summarizes stress and {k2} summarizes repair could clarify tradeoffs in {topic}.",
            "Integrating evidence about {k1} and {k2} from multiple sources may resolve contradictions in {topic}.",
        ]

    # Default general templates
    return [
        "A key driver of {topic} could be the interaction between {k1} and {k2}.",
        "{topic} may be strongly influenced by changes in {k1} under conditions affecting {k2}.",
        "Interventions that modify {k1} might increase stability or repair efficiency in {topic}.",
        "Contradictions in the literature about {k1} and {k2} suggest a non-linear relationship affecting {topic}.",
        "Combining evidence from multiple sources suggests that {k1} could be an upstream regulator in {topic}.",
    ]


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
) -> List[Dict[str, Any]]:
    """Generate simple, structured hypotheses.

    Args:
        goal:
            Human-readable research goal.
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

    Returns:
        List of:
            {
              "text": "...",
              "confidence": 0.30–0.90,
              "domain": "longevity" | "math" | "general" | None,
              "role": "researcher" | "critic" | ... | None,
              "keywords": [k1, k2],
              "novelty": 0.0–1.0,
            }

    Reparodynamics angle:
        These hypotheses are potential "repair targets" for future cycles.
        If later cycles confirm or refute them, ΔR reflects that, and
        long-run RYE (ΔR / E) can track which directions were fruitful.
    """
    if not goal and not notes and not citations:
        return []

    # 1) Build keyword pool from notes + citations
    note_terms = _extract_keywords_from_notes(notes, max_keywords=8)
    cit_terms = _extract_keywords_from_citations(citations, max_keywords=8)
    keywords = _merge_keywords(note_terms, cit_terms, domain=domain, limit=10)

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
            pairs.append((keywords[i], keywords[j]))
    if not pairs:
        pairs = [(keywords[0], keywords[0])]

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
        # For higher scrutiny domains, raise novelty requirement a bit
        novelty_threshold = 0.10
        d = (domain or "").lower()
        if d in {"longevity", "math"}:
            novelty_threshold = 0.18

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

        used_texts.add(text)
        hypo_list.append(
            {
                "text": text,
                "confidence": round(confidence, 2),
                "domain": (domain or None),
                "role": (role or None),
                "keywords": [k1, k2],
                "novelty": round(novelty, 3),
            }
        )

        idx += 1
        if len(hypo_list) >= max_hypotheses:
            break

    return hypo_list
