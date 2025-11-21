# agent/presets.py

"""Domain presets for the Autonomous Research Agent.

These presets are like "profiles" for different domains. Each preset is a
small configuration bundle that can be used by the UI (Streamlit) and by
the core agent / TGRM loop.

Nothing in here is required by the current code except:
    - label
    - default_goal
    - source_controls
    - domain

Everything else (focus_keywords, min_citation_count, etc.) is "future
power" that you can plug into TGRM and RYE scoring later without breaking
the app today.
"""

from __future__ import annotations
from typing import Dict, Any

# Top-level dictionary of presets. Keys are internal IDs
# like "general", "longevity", "math".
PRESETS: Dict[str, Dict[str, Any]] = {
    "general": {
        "label": "General research",
        "default_goal": (
            "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
            "identify similar frameworks in the literature, and produce a structured comparison table."
        ),
        # Which sources/tools should be enabled by default for this preset.
        "source_controls": {
            "web": True,
            "pubmed": False,
            "semantic": False,
            "pdf": True,
            "biomarkers": False,
        },
        # Logical domain tag for later use in TGRM / RYE logic.
        "domain": "general",

        # ---------- Extra metadata (not yet required by other files) ----------
        # Keywords that can be appended/boosted in queries for this domain.
        "focus_keywords": [
            "reparodynamics",
            "RYE",
            "repair yield per energy",
            "TGRM",
            "self-repair",
            "autonomous systems",
            "stability",
            "resilience",
        ],
        # Minimum citation count when filtering Semantic Scholar results (if you decide to use this).
        "min_citation_count": 0,
        # Preferred source order for this preset.
        "preferred_sources": ["web", "semantic", "pdf"],
        # Text structuring hint – how notes should ideally be organized.
        "note_structure_hint": [
            "High level summary",
            "Key definitions (Reparodynamics, RYE, TGRM)",
            "Similar frameworks",
            "Open questions / next steps",
        ],
        # Optional weighting hints for future RYE enhancements.
        "rye_weights": {
            "issues_resolved": 1.0,
            "hypotheses_quality": 1.0,
            "citations_added": 0.5,
        },
    },

    "longevity": {
        "label": "Longevity / Anti-aging",
        "default_goal": (
            "Identify and summarize interventions, biomarkers, and mechanisms that extend healthspan and longevity, "
            "and relate them to reparodynamic efficiency (RYE) and self-repair stability across organs and systems."
        ),
        "source_controls": {
            "web": True,
            "pubmed": True,
            "semantic": True,
            "pdf": True,
            "biomarkers": True,
        },
        "domain": "longevity",

        # For building richer domain-aware queries.
        "focus_keywords": [
            "longevity",
            "healthspan",
            "aging",
            "senescence",
            "autophagy",
            "mTOR",
            "NAD+",
            "rapamycin",
            "metformin",
            "caloric restriction",
            "clinical trial",
            "biomarker",
        ],
        # For longevity you often want at least decently cited papers.
        "min_citation_count": 20,
        "preferred_sources": ["pubmed", "semantic", "web", "pdf"],
        "note_structure_hint": [
            "High level summary",
            "Interventions (drug / lifestyle / protocol)",
            "Mechanisms and pathways",
            "Biomarkers affected",
            "Evidence strength (animal vs human, trial size)",
            "Potential RYE interpretation (repair yield per energy / side effect cost)",
        ],
        "rye_weights": {
            # Reward resolving contradictions in the literature.
            "issues_resolved": 1.2,
            # Reward generating good hypotheses about mechanisms / biomarkers.
            "hypotheses_quality": 1.5,
            # Reward adding well-cited sources.
            "citations_added": 1.0,
        },
    },

    "math": {
        "label": "Math / Theory",
        "default_goal": (
            "Formalize Reparodynamics, RYE, and TGRM as a mathematical framework, including precise definitions, "
            "axioms, theorems, and sketches of proofs, and compare them to existing control, information, and "
            "stability theories in the literature."
        ),
        "source_controls": {
            "web": True,
            "pubmed": False,   # rarely needed for pure theory
            "semantic": True,  # good for theoretical CS / math papers
            "pdf": True,
            "biomarkers": False,
        },
        "domain": "math",

        "focus_keywords": [
            "mathematical",
            "formalization",
            "axiom",
            "theorem",
            "proof",
            "stability theory",
            "Lyapunov",
            "control theory",
            "information theory",
            "Markov process",
            "equilibrium",
        ],
        # You might want stronger filtering here too.
        "min_citation_count": 5,
        "preferred_sources": ["semantic", "pdf", "web"],
        "note_structure_hint": [
            "Informal overview",
            "Definitions (Reparodynamics, RYE, TGRM, equilibrium, damage, repair operator)",
            "Core lemmas",
            "Candidate theorems",
            "Connections to existing theories",
            "Open problems / conjectures",
        ],
        "rye_weights": {
            # In math, resolving contradictions and sharpening definitions is huge.
            "issues_resolved": 1.5,
            # Hypotheses here are conjectures / theorems → high weight.
            "hypotheses_quality": 1.7,
            # Citations still matter but less than the structure itself.
            "citations_added": 0.7,
        },
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """Return a preset config, falling back to 'general' if unknown.

    This keeps the rest of the app safe even if an invalid preset key is used.
    """
    return PRESETS.get(name, PRESETS["general"])
