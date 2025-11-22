"""Simple hypothesis engine for the autonomous research agent (90-day safe).

This module is deliberately:
- LLM-free (pure Python, deterministic).
- Stateless (no global counters that can drift over 24–90 day runs).
- Swarm-safe (no shared mutable state across agents).
- Domain-aware (optional hooks for longevity / math / general).

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
        - return the most common terms

    This is intentionally simple and deterministic so it remains stable
    across 24–90 day runs and across multiple swarm agents.
    """
    freq: Dict[str, int] = {}
    for n in notes:
        content = str(n.get("content", ""))
        tokens = content.replace("\n", " ").split()
        for t in tokens:
            cleaned = t.strip(".,;:!?()[]{}\"'").lower()
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
            }:
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
            cleaned = t.strip(".,;:!?()[]{}\"'").lower()
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
            }:
                continue
            freq[cleaned] = freq.get(cleaned, 0) + 1

    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for (t, _) in sorted_terms[:max_keywords]]


def _merge_keywords(note_terms: List[str], cit_terms: List[str], limit: int = 10) -> List[str]:
    """Merge keywords from notes and citations, preserving order and uniqueness."""
    seen = set()
    merged: List[str] = []

    for term in note_terms + cit_terms:
        if term in seen:
            continue
        seen.add(term)
        merged.append(term)
        if len(merged) >= limit:
            break

    return merged


# ---------------------------------------------------------------------
# Domain-aware templates
# ---------------------------------------------------------------------
def _get_templates_for_domain(domain: Optional[str]) -> List[str]:
    """Return a pool of hypothesis templates based on domain.

    Domain-specific templates encourage more grounded, testable ideas
    (e.g., biomarkers for longevity, formal definitions for math).
    """
    d = (domain or "").lower()

    if d == "longevity":
        return [
            "Changes in {k1} may correlate with improvements in {topic} via modulation of {k2}-linked pathways.",
            "In {topic}, tracking {k1} as a biomarker alongside {k2} could better predict repair or resilience outcomes.",
            "Interventions that reduce adverse trends in {k1} while stabilizing {k2} might yield higher RYE in {topic}.",
            "The interaction between {k1} and {k2} may explain heterogeneous responses to longevity interventions in {topic}.",
            "Combining {k1}-focused protocols with monitoring of {k2} might identify individuals with superior self-repair capacity in {topic}.",
        ]

    if d == "math":
        return [
            "A formal definition that links {k1} and {k2} could yield a stability criterion for {topic}.",
            "Constructing a Lyapunov-like functional using {k1} and {k2} may prove convergence for {topic}.",
            "Recasting {topic} in terms of {k1} and {k2} could reveal equivalence to an existing stability framework.",
            "Non-linear coupling between {k1} and {k2} might explain phase transitions or repair plateaus in {topic}.",
            "An information-theoretic view where {k1} encodes state and {k2} encodes repair signal may formalize RYE in {topic}.",
        ]

    # General fallback templates
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
            Optional swarm role label (researcher, critic, etc.).
            Currently only used to mildly adjust confidence ranges.

    Returns:
        List of:
            {
              "text": "...",
              "confidence": 0.30–0.85,
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
    keywords = _merge_keywords(note_terms, cit_terms, limit=10)

    # Fallback if everything is empty
    if not keywords:
        keywords = ["repair", "resilience", "stability", "rye"]

    # 2) Choose template pool based on domain
    templates = _get_templates_for_domain(domain)

    # 3) Generate keyword pairs
    pairs = []
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            pairs.append((keywords[i], keywords[j]))
    if not pairs:
        pairs = [(keywords[0], keywords[0])]

    # 4) Generate hypotheses
    base_goal = goal if goal else "the current research topic"
    hypo_list: List[Dict[str, Any]] = []

    # Role-based confidence band (very light touch, fully deterministic)
    base_low, base_high = 0.35, 0.8
    r = (role or "").lower()
    if r in {"critic"}:
        base_low, base_high = 0.30, 0.7
    elif r in {"researcher", "explorer"}:
        base_low, base_high = 0.40, 0.85

    step = (base_high - base_low) / max(1, len(templates))

    idx = 0
    for (k1, k2) in pairs:
        tpl = templates[idx % len(templates)]
        text = tpl.format(topic=base_goal, k1=k1, k2=k2)

        # Deterministic confidence progression across hypotheses
        band_index = idx % len(templates)
        confidence = base_low + step * band_index
        confidence = max(0.0, min(1.0, confidence))

        hypo_list.append(
            {
                "text": text,
                "confidence": round(confidence, 2),
            }
        )

        idx += 1
        if len(hypo_list) >= max_hypotheses:
            break

    return hypo_list
