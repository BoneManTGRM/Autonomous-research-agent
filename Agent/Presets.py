# agent/presets.py

from __future__ import annotations
from typing import Dict, Any

PRESETS: Dict[str, Dict[str, Any]] = {
    "general": {
        "label": "General research",
        "default_goal": (
            "Research and summarize the concept of Reparodynamics, define RYE and TGRM, "
            "identify similar frameworks in the literature, and produce a structured comparison table."
        ),
        "source_controls": {
            "web": True,
            "pubmed": False,
            "semantic": False,
            "pdf": True,
            "biomarkers": False,
        },
        "domain": "general",
    },
    "longevity": {
        "label": "Longevity / Anti-aging",
        "default_goal": (
            "Identify and summarize interventions, biomarkers, and mechanisms that extend healthspan and longevity, "
            "relating them to reparodynamic efficiency (RYE) and self-repair stability."
        ),
        "source_controls": {
            "web": True,
            "pubmed": True,
            "semantic": True,
            "pdf": True,
            "biomarkers": True,
        },
        "domain": "longevity",
    },
    "math": {
        "label": "Math / Theory",
        "default_goal": (
            "Formalize Reparodynamics, RYE, and TGRM as a mathematical framework, including definitions, "
            "axioms, theorems, and proofs, and compare to existing control and information theories."
        ),
        "source_controls": {
            "web": True,
            "pubmed": False,
            "semantic": True,
            "pdf": True,
            "biomarkers": False,
        },
        "domain": "math",
    },
}

def get_preset(name: str) -> Dict[str, Any]:
    """Return a preset config; fall back to 'general'."""
    return PRESETS.get(name, PRESETS["general"])
