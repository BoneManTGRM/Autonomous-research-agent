"""Simple hypothesis engine for the autonomous research agent.

This does NOT use an LLM. It is a structured template engine that
generates plausible, testable hypotheses from:

- the current goal,
- existing notes,
- citations (web, PubMed, Semantic Scholar).

TGRM uses this in the REPAIR / VERIFY phases to propose new directions.
Later, you can upgrade this to call a language model, but this base
version is fully self-contained.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _extract_keywords_from_notes(notes: List[Dict[str, Any]], max_keywords: int = 5) -> List[str]:
    """Very naive keyword extractor based on simple heuristics.

    We look for words that:
    - appear in uppercase, or
    - look like domain terms (contain digits or hyphens), or
    - are long and repeated.
    """
    freq: Dict[str, int] = {}
    for n in notes:
        content = str(n.get("content", ""))
        tokens = content.replace("\n", " ").split()
        for t in tokens:
            cleaned = t.strip(".,;:!?()[]{}\"'").lower()
            if len(cleaned) < 6:
                continue
            freq[cleaned] = freq.get(cleaned, 0) + 1

    # Sort by frequency
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [t for (t, _) in sorted_terms[:max_keywords]]


def generate_hypotheses(
    goal: str,
    notes: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    max_hypotheses: int = 5,
) -> List[Dict[str, Any]]:
    """Generate simple, structured hypotheses.

    Returns a list of:
        {
          "text": "...",
          "confidence": 0.3-0.8,
        }

    Reparodynamics angle:
        These hypotheses are potential "repair targets" for future cycles.
        If later cycles confirm or refute them, ΔR reflects that.
    """
    if not goal and not notes and not citations:
        return []

    keywords = _extract_keywords_from_notes(notes)
    hypo_list: List[Dict[str, Any]] = []

    base_goal = goal if goal else "the current research topic"

    # Simple templates
    template_pool = [
        "A key driver of {topic} could be the interaction between {k1} and {k2}.",
        "{topic} may be strongly influenced by changes in {k1} under conditions affecting {k2}.",
        "Interventions that modify {k1} might increase stability or repair efficiency in {topic}.",
        "Contradictions in the literature about {k1} and {k2} suggest a non-linear relationship affecting {topic}.",
        "Combining evidence from multiple sources suggests that {k1} could be an upstream regulator in {topic}.",
    ]

    # Fallback keywords if none detected
    if not keywords:
        keywords = ["repair", "resilience", "stability"]

    # Use pairs of keywords
    pairs = []
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            pairs.append((keywords[i], keywords[j]))
    if not pairs:
        pairs = [(keywords[0], keywords[0])]

    idx = 0
    for (k1, k2) in pairs:
        tpl = template_pool[idx % len(template_pool)]
        text = tpl.format(topic=base_goal, k1=k1, k2=k2)
        # Very rough confidence: later citations and contradictions could adjust this
        confidence = 0.4 + 0.05 * (idx % 5)
        hypo_list.append({"text": text, "confidence": round(confidence, 2)})
        idx += 1
        if len(hypo_list) >= max_hypotheses:
            break

    return hypo_list
