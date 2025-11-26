"""Hallmark profiles for longevity focused runs.

This module provides structured metadata for aging and longevity hallmarks
so the agent can:

    - Keep each hallmark in its own clean namespace
    - Attach biologically meaningful subgoals and biomarkers to cycles
    - Suggest mechanism pathways and intervention stacks
    - Rotate deterministic subgoals for curriculum and replay

The goal is to give the CoreAgent, CurriculumController, and TGRMLoop a
shared, stable source of truth about what "mitochondria", "senescence",
or "mito_sen_hybrid" actually mean in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HallmarkProfile:
    """Single hallmark configuration.

    All fields are intentionally simple Python types so they are easy to
    log, serialize, and surface in the UI.
    """

    key: str
    label: str
    namespace: str
    description: str

    canonical_tags: List[str] = field(default_factory=list)
    core_pathways: List[str] = field(default_factory=list)

    # Subgoals are concrete things a cycle can focus on, for example
    # "improve mitochondrial membrane potential" rather than just "mitochondria".
    default_subgoals: List[str] = field(default_factory=list)

    # Biomarkers grouped into rough categories.
    biomarkers_primary: List[str] = field(default_factory=list)
    biomarkers_secondary: List[str] = field(default_factory=list)
    biomarkers_experimental: List[str] = field(default_factory=list)

    # Example interventions or levers. These are not recommendations.
    example_interventions: List[str] = field(default_factory=list)

    # High level risk or caveat tags to keep the agent cautious.
    risk_flags: List[str] = field(default_factory=list)

    # Synergy stacks: tuples like (name, [elements])
    synergy_templates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict view of the profile."""
        return {
            "key": self.key,
            "label": self.label,
            "namespace": self.namespace,
            "description": self.description,
            "canonical_tags": list(self.canonical_tags),
            "core_pathways": list(self.core_pathways),
            "default_subgoals": list(self.default_subgoals),
            "biomarkers_primary": list(self.biomarkers_primary),
            "biomarkers_secondary": list(self.biomarkers_secondary),
            "biomarkers_experimental": list(self.biomarkers_experimental),
            "example_interventions": list(self.example_interventions),
            "risk_flags": list(self.risk_flags),
            "synergy_templates": list(self.synergy_templates),
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class HallmarkProfiles:
    """Registry of hallmark profiles with convenience helpers.

    This is intended to be instantiated once and shared through CoreAgent
    so that CurriculumController and TGRMLoop can:

        - Resolve user chosen hallmarks into stable keys
        - Pull subgoals and biomarker sets for each cycle
        - Build replay and reporting summaries per hallmark
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, HallmarkProfile] = {}
        self._build_defaults()

    # ------------------------------------------------------------------
    # Core build
    # ------------------------------------------------------------------
    def _build_defaults(self) -> None:
        """Populate default longevity hallmarks."""
        mito = HallmarkProfile(
            key="mitochondria",
            label="Mitochondrial function and energetics",
            namespace="hallmark.mitochondria",
            description=(
                "Mitochondrial function, bioenergetics, ROS handling, and quality control, "
                "including mitophagy and biogenesis."
            ),
            canonical_tags=[
                "mitochondria",
                "mitochondrial dysfunction",
                "bioenergetics",
                "mitophagy",
                "ROS",
                "oxidative stress",
            ],
            core_pathways=[
                "OXPHOS",
                "mitochondrial biogenesis",
                "mitophagy",
                "ROS detoxification",
                "NAD+ metabolism",
                "AMPK",
                "PGC1alpha",
                "SIRT1",
            ],
            default_subgoals=[
                "map key biomarkers of mitochondrial health in humans",
                "optimize mitochondrial membrane potential and ATP output",
                "reduce maladaptive ROS signaling while preserving hormesis",
                "improve mitophagy and mitochondrial quality control",
                "identify safe NAD+ and mitochondrial metabolism interventions",
            ],
            biomarkers_primary=[
                "ATP production rate",
                "mitochondrial membrane potential",
                "VO2 max",
                "lactate threshold",
                "resting energy expenditure",
            ],
            biomarkers_secondary=[
                "NAD+/NADH ratio",
                "mitochondrial DNA copy number",
                "oxidative phosphorylation capacity",
                "cardiorespiratory fitness scores",
                "skeletal muscle mitochondrial function assays",
            ],
            biomarkers_experimental=[
                "single cell mitochondrial functional readouts",
                "mitochondrial network morphology metrics",
                "live cell ROS dynamics under stress",
            ],
            example_interventions=[
                "graded endurance training",
                "high intensity interval training",
                "mitochondrial targeted antioxidants research overview",
                "NAD+ precursor protocols research overview",
                "mitochondrial biogenesis stimulators research overview",
            ],
            risk_flags=[
                "caution with untested mitochondrial targeted compounds",
                "risk of ROS over suppression and blunted hormesis",
                "potential off target mitochondrial toxicity",
            ],
            synergy_templates=[
                {
                    "name": "Mito cardiorespiratory stack",
                    "elements": [
                        "endurance training protocol",
                        "HIIT protocol",
                        "sleep and circadian optimization",
                        "mitochondrial biomarker panel",
                    ],
                },
                {
                    "name": "NAD+ focused mito stack",
                    "elements": [
                        "NAD+ precursor protocol research",
                        "exercise program",
                        "mitochondrial function lab follow up",
                    ],
                },
            ],
        )

        sen = HallmarkProfile(
            key="senescence",
            label="Cellular senescence",
            namespace="hallmark.senescence",
            description=(
                "Senescent cell burden, SASP signaling, tissue microenvironment effects, "
                "and senolytic or senomorphic interventions."
            ),
            canonical_tags=[
                "senescence",
                "senescent cells",
                "SASP",
                "senolytics",
                "senomorphics",
            ],
            core_pathways=[
                "p16INK4a",
                "p21",
                "p53",
                "DNA damage response",
                "NFkB mediated SASP",
                "immune clearance of senescent cells",
            ],
            default_subgoals=[
                "map safe biomarkers of senescent cell burden in humans",
                "clarify SASP driven local and systemic effects",
                "identify human relevant senolytic and senomorphic strategies",
                "study synergy between senescence clearance and regeneration",
            ],
            biomarkers_primary=[
                "p16INK4a expression in accessible tissues",
                "senescence associated beta gal markers in research contexts",
                "inflammatory cytokine panels linked to SASP patterns",
            ],
            biomarkers_secondary=[
                "immune profiling related to senescent cell clearance",
                "tissue specific imaging markers of fibrosis or senescence",
            ],
            biomarkers_experimental=[
                "single cell SASP profiling",
                "spatial transcriptomics of senescent niches",
            ],
            example_interventions=[
                "intermittent senolytic protocols under trial",
                "senomorphic agents that dampen SASP",
                "lifestyle and metabolic levers that affect senescence burden",
            ],
            risk_flags=[
                "high potential for toxicity of senolytics",
                "unknown long run effects of aggressive senescent cell clearance",
                "context specific tradeoffs between senescence and cancer risk",
            ],
            synergy_templates=[
                {
                    "name": "Senescence and regeneration stack",
                    "elements": [
                        "intermittent senolytic or senomorphic strategy research",
                        "regenerative or pro repair interventions",
                        "immune support and surveillance optimization",
                    ],
                }
            ],
        )

        hybrid = HallmarkProfile(
            key="mito_sen_hybrid",
            label="Mitochondria x senescence hybrid",
            namespace="hallmark.mito_sen_hybrid",
            description=(
                "Hybrid view that tracks how mitochondrial dysfunction and senescent cell "
                "burden interact and reinforce each other."
            ),
            canonical_tags=[
                "mitochondria",
                "senescence",
                "hybrid hallmark",
                "mitochondrial ROS and SASP",
            ],
            core_pathways=[
                "mitochondrial ROS driven DNA damage and senescence entry",
                "senescence induced metabolic and mitochondrial remodeling",
                "feedback loops between SASP and mitochondrial stress",
            ],
            default_subgoals=[
                "identify mechanism chains from mitochondrial dysfunction to senescence",
                "identify mechanism chains from senescence to mitochondrial decline",
                "design multi hallmark interventions that target mitochondria and senescence together",
            ],
            biomarkers_primary=[
                "combined mitochondrial function panel",
                "combined inflammation and SASP linked cytokine panel",
            ],
            biomarkers_secondary=[
                "composite risk scores combining mitochondrial and inflammatory markers",
            ],
            biomarkers_experimental=[
                "multiomic signatures that couple mitochondrial stress and senescence states",
            ],
            example_interventions=[
                "paired exercise plus senescence modulating strategies",
                "stacked protocols that coordinate mitochondrial support and senescence load management",
            ],
            risk_flags=[
                "added complexity of multi hallmark interventions",
                "risk of drawing strong conclusions from early stage biomarkers",
            ],
            synergy_templates=[
                {
                    "name": "Mito x sen multi hallmark stack",
                    "elements": [
                        "mitochondrial function training and protocols",
                        "carefully designed senolytic or senomorphic strategies",
                        "inflammation and mitochondrial biomarker panel",
                    ],
                }
            ],
        )

        general = HallmarkProfile(
            key="general",
            label="General health and aging",
            namespace="hallmark.general",
            description=(
                "Fallback hallmark for general healthspan, lifespan, and systems level aging work "
                "when no specific hallmark is selected."
            ),
            canonical_tags=[
                "longevity",
                "healthspan",
                "aging",
                "reparodynamics",
            ],
            core_pathways=[
                "metabolic health",
                "inflammation and immune balance",
                "tissue repair and regeneration",
                "stress response and hormesis",
            ],
            default_subgoals=[
                "map cross hallmark interventions that appear robust in human data",
                "prioritize interventions with strong evidence and clear biomarkers",
            ],
            biomarkers_primary=[
                "standard clinical panel relevant to aging risk",
                "cardiometabolic biomarkers",
                "basic immune and inflammatory markers",
            ],
            biomarkers_secondary=[],
            biomarkers_experimental=[],
            example_interventions=[
                "lifestyle and nutritional interventions with strong human outcome data",
            ],
            risk_flags=[
                "risk of overly broad conclusions from heterogeneous studies",
            ],
            synergy_templates=[],
        )

        for profile in (mito, sen, hybrid, general):
            self._profiles[profile.key] = profile

        # Alias maps for sensible user facing labels
        self._aliases: Dict[str, str] = {
            "mitochondrial": "mitochondria",
            "mito": "mitochondria",
            "cellular_senescence": "senescence",
            "senescent": "senescence",
            "mito_x_sen": "mito_sen_hybrid",
            "mito sen hybrid": "mito_sen_hybrid",
            "hybrid": "mito_sen_hybrid",
        }

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def _normalise_key(self, name: str) -> str:
        """Map incoming label to a canonical key."""
        raw = (name or "").strip().lower()
        if not raw:
            return "general"
        if raw in self._profiles:
            return raw
        if raw in self._aliases:
            return self._aliases[raw]
        # Simple normalization passes
        raw_clean = raw.replace(" ", "_")
        if raw_clean in self._profiles:
            return raw_clean
        return "general"

    def get_profile(self, name: str) -> HallmarkProfile:
        """Return the HallmarkProfile for a given label or key."""
        key = self._normalise_key(name)
        return self._profiles.get(key, self._profiles["general"])

    def get_profile_dict(self, name: str) -> Dict[str, Any]:
        """Return a plain dict view of a hallmark profile."""
        return self.get_profile(name).to_dict()

    def get_hallmark_names(self) -> List[str]:
        """List canonical hallmark keys that the system knows about."""
        return sorted(self._profiles.keys())

    def resolve_targets(self, targets: Optional[Any]) -> List[str]:
        """Resolve a hallmark_target config field into canonical keys.

        Accepts:
            - None -> ["general"]
            - "mitochondria" or "Mitochondria" -> ["mitochondria"]
            - list of labels -> canonicalized unique list
        """
        if targets is None:
            return ["general"]

        if isinstance(targets, str):
            return [self._normalise_key(targets)]

        keys: List[str] = []
        for t in targets:
            key = self._normalise_key(str(t))
            if key not in keys:
                keys.append(key)
        if not keys:
            keys.append("general")
        return keys

    # ------------------------------------------------------------------
    # Subgoals and biomarker helpers
    # ------------------------------------------------------------------
    def get_default_subgoals(self, hallmark: str) -> List[str]:
        profile = self.get_profile(hallmark)
        return list(profile.default_subgoals)

    def pick_subgoal_for_cycle(self, hallmark: str, cycle_index: int) -> Optional[str]:
        """Deterministic rotation of subgoals across cycles."""
        profile = self.get_profile(hallmark)
        subgoals = profile.default_subgoals
        if not subgoals:
            return None
        if cycle_index < 0:
            cycle_index = 0
        idx = cycle_index % len(subgoals)
        return subgoals[idx]

    def get_biomarker_panel(
        self,
        hallmark: str,
        *,
        include_secondary: bool = True,
        include_experimental: bool = False,
    ) -> List[str]:
        """Return a biomarker list tuned for the given hallmark."""
        profile = self.get_profile(hallmark)
        panel: List[str] = []
        panel.extend(profile.biomarkers_primary)
        if include_secondary:
            panel.extend(profile.biomarkers_secondary)
        if include_experimental:
            panel.extend(profile.biomarkers_experimental)

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: List[str] = []
        for b in panel:
            if b not in seen:
                seen.add(b)
                result.append(b)
        return result

    def get_synergy_templates(self, hallmark: str) -> List[Dict[str, Any]]:
        profile = self.get_profile(hallmark)
        return list(profile.synergy_templates)

    def get_namespace(self, hallmark: str) -> str:
        profile = self.get_profile(hallmark)
        return profile.namespace

    # ------------------------------------------------------------------
    # Summaries for notes and reports
    # ------------------------------------------------------------------
    def build_note_header(
        self,
        hallmark: str,
        subgoal: Optional[str],
        biomarker_focus: Optional[List[str]] = None,
    ) -> str:
        """Create a compact header for a cycle note."""
        profile = self.get_profile(hallmark)
        lines: List[str] = []
        lines.append(f"[Hallmark] {profile.label}")
        lines.append(f"[Namespace] {profile.namespace}")
        if subgoal:
            lines.append(f"[Subgoal] {subgoal}")
        if biomarker_focus:
            short_panel = ", ".join(biomarker_focus[:8])
            lines.append(f"[Biomarker focus] {short_panel}")
        return "\n".join(lines)

    def summarize_for_report(self, hallmark: str) -> Dict[str, Any]:
        """Lightweight summary useful for reporting views."""
        profile = self.get_profile(hallmark)
        return {
            "key": profile.key,
            "label": profile.label,
            "namespace": profile.namespace,
            "description": profile.description,
            "core_pathways": list(profile.core_pathways),
            "primary_biomarkers": list(profile.biomarkers_primary),
            "risk_flags": list(profile.risk_flags),
            "example_interventions": list(profile.example_interventions),
        }

    # ------------------------------------------------------------------
    # Multi hallmark helpers
    # ------------------------------------------------------------------
    def merged_biomarker_panel(
        self,
        hallmarks: List[str],
        *,
        include_secondary: bool = True,
        include_experimental: bool = False,
    ) -> List[str]:
        """Build a merged biomarker panel across multiple hallmarks."""
        panel: List[str] = []
        for h in self.resolve_targets(hallmarks):
            panel.extend(
                self.get_biomarker_panel(
                    h,
                    include_secondary=include_secondary,
                    include_experimental=include_experimental,
                )
            )
        seen: set[str] = set()
        result: List[str] = []
        for b in panel:
            if b not in seen:
                seen.add(b)
                result.append(b)
        return result

    def merged_synergy_templates(self, hallmarks: List[str]) -> List[Dict[str, Any]]:
        """Merge synergy templates while keeping the hallmark context."""
        merged: List[Dict[str, Any]] = []
        for h in self.resolve_targets(hallmarks):
            profile = self.get_profile(h)
            for tpl in profile.synergy_templates:
                tpl_copy = dict(tpl)
                tpl_copy.setdefault("hallmark", profile.key)
                merged.append(tpl_copy)
        return merged
