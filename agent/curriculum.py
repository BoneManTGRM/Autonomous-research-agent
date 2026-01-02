# agent/curriculum.py

"""
Curriculum and hallmark controllers for the Autonomous Research Agent.

This module provides:
    - HallmarkProfiles
        Canonical definitions and helper utilities for aging hallmarks
        and sub pathways such as mitochondrial function or senescence.

    - CurriculumPhase
        Configuration container for a single curriculum phase
        (cycle window, hallmarks, subgoals, exploration vs verification).

    - CurriculumState
        Lightweight description of the active curriculum segment for a
        given point in a run. This is what CoreAgent and TGRMLoop use.

    - CurriculumController
        Main API used by CoreAgent:
            select_segment(run_progress: dict) -> dict

Reparodynamics view:
    Curriculum is the long range steering layer for TGRM and RYE.
    It shapes what the agent studies at different stages of a run:
        - early broad exploration
        - focused hallmark depth
        - cross hallmark synergy building
        - exploit and verify high RYE mechanisms from replay
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Hallmark profiles
# ---------------------------------------------------------------------------


@dataclass
class HallmarkDefinition:
    """Canonical information about a single hallmark of aging."""

    key: str
    label: str
    default_subgoals: List[str] = field(default_factory=list)
    default_biomarkers: List[str] = field(default_factory=list)
    notes: str = ""


class HallmarkProfiles:
    """Central registry for longevity hallmarks and related metadata.

    This class is designed to be read only during a run. CoreAgent can
    keep a single instance at self.hallmark_profiles and reuse it for
    curriculum, memory namespaces, and reporting.
    """

    def __init__(self) -> None:
        self._hallmarks: Dict[str, HallmarkDefinition] = {}
        self._build_defaults()

    def _build_defaults(self) -> None:
        """Populate a default set of hallmarks and sub pathways."""
        self._register(
            HallmarkDefinition(
                key="mitochondria",
                label="Mitochondrial function and bioenergetics",
                default_subgoals=[
                    "mito_membrane_potential",
                    "mito_ros_balance",
                    "mito_dynamics_fusion_fission",
                    "mito_biogenesis",
                ],
                default_biomarkers=[
                    "ATP",
                    "lactate",
                    "mito_membrane_potential_assays",
                    "oxidative_phosphorylation_markers",
                ],
                notes="Central energy and ROS control lever for aging.",
            )
        )
        self._register(
            HallmarkDefinition(
                key="senescence",
                label="Cellular senescence and SASP",
                default_subgoals=[
                    "senescent_cell_burden",
                    "sasp_profile",
                    "senolytic_interventions",
                    "immune_clearance_of_senescent_cells",
                ],
                default_biomarkers=[
                    "p16_expression",
                    "SA_beta_gal",
                    "inflammatory_cytokines_IL6_TNF",
                ],
                notes="Senescent cells, their secretory phenotype, and removal.",
            )
        )
        self._register(
            HallmarkDefinition(
                key="mito_sen_hybrid",
                label="Mitochondria x Senescence hybrid axis",
                default_subgoals=[
                    "mito_driven_senescence",
                    "senescence_mito_cross_talk",
                    "mito_sen_therapeutic_stacks",
                ],
                default_biomarkers=[
                    "mito_derived_ros",
                    "senescence_markers",
                    "inflammation_markers",
                ],
                notes="Interactions spanning mitochondrial stress and senescence.",
            )
        )

    def _register(self, definition: HallmarkDefinition) -> None:
        self._hallmarks[definition.key] = definition

    # Public API ---------------------------------------------------------

    def keys(self) -> List[str]:
        return sorted(self._hallmarks.keys())

    def get(self, key: str) -> Optional[HallmarkDefinition]:
        return self._hallmarks.get(self._normalise_key(key))

    def _normalise_key(self, key: str) -> str:
        """Normalise a hallmark string to a canonical key."""
        raw = (key or "").strip().lower()
        if raw in self._hallmarks:
            return raw

        # Flexible mapping for common variants
        mapping = {
            "mito": "mitochondria",
            "mitochondrial": "mitochondria",
            "senescent": "senescence",
            "senolytics": "senescence",
            "mito_sen": "mito_sen_hybrid",
            "mito_senescence": "mito_sen_hybrid",
        }
        return mapping.get(raw, raw)

    def normalise_targets(self, targets: Optional[Sequence[str]]) -> List[str]:
        """Normalise a user supplied hallmark list to known keys."""
        if not targets:
            return list(self.keys())

        out: List[str] = []
        for t in targets:
            key = self._normalise_key(t)
            if key in self._hallmarks and key not in out:
                out.append(key)
        if not out:
            out = list(self.keys())
        return out

    def default_subgoals_for(self, hallmark_key: str) -> List[str]:
        h = self.get(hallmark_key)
        return list(h.default_subgoals) if h else []

    def default_biomarkers_for(self, hallmark_key: str) -> List[str]:
        h = self.get(hallmark_key)
        return list(h.default_biomarkers) if h else []


# ---------------------------------------------------------------------------
# Curriculum data containers
# ---------------------------------------------------------------------------


@dataclass
class CurriculumPhase:
    """Configuration for one curriculum phase within a profile."""

    name: str
    min_cycle: int
    max_cycle: Optional[int]
    focus_hallmarks: List[str]
    subgoals: List[str] = field(default_factory=list)
    exploration_bias: float = 0.5
    verify_bias: float = 0.5
    replay_intensity: float = 0.0
    notes: str = ""


@dataclass
class CurriculumState:
    """Result of curriculum selection for a specific point in a run."""

    profile: str
    phase_name: str
    phase_index: int
    total_phases: int

    hallmark: str
    hallmark_list: List[str]
    subgoal: Optional[str]

    curriculum_phase_label: str
    exploration_bias: float
    verify_bias: float
    replay_intensity: float

    stage_hint: str
    replay_focus_hint: str
    biomarker_focus: List[str] = field(default_factory=list)
    # Optional funnel step hint for short / structured runs (e.g. 2-cycle or 5-cycle discovery funnel)
    funnel_step: Optional[str] = None


# ---------------------------------------------------------------------------
# Curriculum controller
# ---------------------------------------------------------------------------


class CurriculumController:
    """High level curriculum logic for long runs.

    CoreAgent uses this controller to steer each cycle:
        - which hallmark to focus on
        - which subgoal or pathway
        - whether to tilt toward idea stage or verify stage
        - how strongly to lean on the replay buffer

    Input:
        run_progress: dict with fields such as:
            cycles_completed: int
            hours_elapsed: float (optional)
            avg_rye: float (optional)
            equilibrium_label: str (optional)
            breakthrough_score: float (optional)
            domain: str (optional)

    Output:
        CurriculumState, which can be converted to a plain dict and
        passed into:
            - CoreAgent.run_two_stage_cycle(...)
            - TGRMLoop.run_cycle(stage=..., hallmark=..., subgoal=..., curriculum_state=...)
    """

    def __init__(
        self,
        profile_name: str = "longevity_basic",
        target_hallmarks: Optional[Sequence[str]] = None,
        domain: str = "longevity",
        hallmark_profiles: Optional[HallmarkProfiles] = None,
    ) -> None:
        self.profile_name = profile_name or "longevity_basic"
        self.domain = (domain or "general").lower()
        self.hallmark_profiles = hallmark_profiles or HallmarkProfiles()

        normalised_targets = self.hallmark_profiles.normalise_targets(target_hallmarks)
        self.target_hallmarks: List[str] = normalised_targets

        self.phases: List[CurriculumPhase] = self._build_profile(
            self.profile_name,
            self.target_hallmarks,
        )

    # Profile construction ------------------------------------------------

    def _build_profile(
        self,
        profile_name: str,
        target_hallmarks: List[str],
    ) -> List[CurriculumPhase]:
        """Create curriculum phases for a given profile name."""

        if self.domain != "longevity":
            # For non longevity domains we keep a simple exploratory profile
            return [
                CurriculumPhase(
                    name="explore_general",
                    min_cycle=0,
                    max_cycle=None,
                    focus_hallmarks=target_hallmarks,
                    subgoals=[],
                    exploration_bias=0.7,
                    verify_bias=0.3,
                    replay_intensity=0.1,
                    notes="Generic exploratory curriculum for non longevity domains.",
                )
            ]

        # Longevity specific profiles
        pname = profile_name.lower()

        if pname == "mito_depth":
            return self._build_mito_depth_profile(target_hallmarks)
        if pname == "cross_hallmark":
            return self._build_cross_hallmark_profile(target_hallmarks)

        # Default
        return self._build_longevity_basic_profile(target_hallmarks)

    def _build_longevity_basic_profile(
        self,
        targets: List[str],
    ) -> List[CurriculumPhase]:
        """Three phase curriculum for broad but focused longevity work."""

        if not targets:
            targets = ["mitochondria", "senescence", "mito_sen_hybrid"]

        # Early broad exploration over chosen hallmarks
        phase0 = CurriculumPhase(
            name="longevity_bootstrap",
            min_cycle=0,
            max_cycle=60,
            focus_hallmarks=targets,
            subgoals=[],
            exploration_bias=0.75,
            verify_bias=0.25,
            replay_intensity=0.05,
            notes="Wide sweep over selected hallmarks and biomarkers.",
        )

        # Middle depth phase, tighter to key hallmarks
        mid_targets = self._pick_priority_hallmarks(targets, top_k=2)
        subgoals_mid = self._collect_subgoals(mid_targets)

        phase1 = CurriculumPhase(
            name="hallmark_depth",
            min_cycle=60,
            max_cycle=200,
            focus_hallmarks=mid_targets,
            subgoals=subgoals_mid,
            exploration_bias=0.6,
            verify_bias=0.4,
            replay_intensity=0.2,
            notes="Depth on 2 main hallmarks, building stable mechanisms.",
        )

        # Late stage exploit and cross hallmark synthesis
        phase2 = CurriculumPhase(
            name="exploit_and_synergy",
            min_cycle=200,
            max_cycle=None,
            focus_hallmarks=targets,
            subgoals=self._collect_subgoals(targets),
            exploration_bias=0.4,
            verify_bias=0.6,
            replay_intensity=0.4,
            notes="Exploit high RYE motifs, build cross hallmark stacks.",
        )

        return [phase0, phase1, phase2]

    def _build_mito_depth_profile(
        self,
        targets: List[str],
    ) -> List[CurriculumPhase]:
        """Curriculum tuned for mitochondrial heavy runs."""

        mito_key = self.hallmark_profiles._normalise_key("mitochondria")
        mito_subgoals = self.hallmark_profiles.default_subgoals_for(mito_key)

        phase0 = CurriculumPhase(
            name="mito_bootstrap",
            min_cycle=0,
            max_cycle=80,
            focus_hallmarks=[mito_key],
            subgoals=mito_subgoals,
            exploration_bias=0.7,
            verify_bias=0.3,
            replay_intensity=0.1,
            notes="Initial survey of mitochondrial interventions and biomarkers.",
        )
        phase1 = CurriculumPhase(
            name="mito_mechanism_depth",
            min_cycle=80,
            max_cycle=220,
            focus_hallmarks=[mito_key],
            subgoals=mito_subgoals,
            exploration_bias=0.5,
            verify_bias=0.5,
            replay_intensity=0.25,
            notes="Map mechanisms and intervention stacks around mito function.",
        )
        phase2 = CurriculumPhase(
            name="mito_exploit_and_synergy",
            min_cycle=220,
            max_cycle=None,
            focus_hallmarks=[mito_key],
            subgoals=mito_subgoals,
            exploration_bias=0.35,
            verify_bias=0.65,
            replay_intensity=0.45,
            notes="Exploit high RYE mito stacks and verify against biomarkers.",
        )

        return [phase0, phase1, phase2]

    def _build_cross_hallmark_profile(
        self,
        targets: List[str],
    ) -> List[CurriculumPhase]:
        """Curriculum focused on interactions between hallmarks."""

        if not targets:
            targets = ["mitochondria", "senescence", "mito_sen_hybrid"]

        phase0 = CurriculumPhase(
            name="crosshallmark_bootstrap",
            min_cycle=0,
            max_cycle=80,
            focus_hallmarks=targets,
            subgoals=[],
            exploration_bias=0.75,
            verify_bias=0.25,
            replay_intensity=0.1,
            notes="Broad coverage of multiple hallmarks and their biomarkers.",
        )
        phase1 = CurriculumPhase(
            name="interaction_depth",
            min_cycle=80,
            max_cycle=240,
            focus_hallmarks=targets,
            subgoals=self._collect_subgoals(targets),
            exploration_bias=0.55,
            verify_bias=0.45,
            replay_intensity=0.3,
            notes="Focus on cross hallmark pathways and mechanism chains.",
        )
        phase2 = CurriculumPhase(
            name="stack_and_exploit",
            min_cycle=240,
            max_cycle=None,
            focus_hallmarks=targets,
            subgoals=self._collect_subgoals(targets),
            exploration_bias=0.4,
            verify_bias=0.6,
            replay_intensity=0.5,
            notes="Exploit and stress test multi hallmark intervention stacks.",
        )

        return [phase0, phase1, phase2]

    # Phase selection helpers --------------------------------------------

    def _pick_priority_hallmarks(self, targets: List[str], top_k: int) -> List[str]:
        """Select top_k hallmarks from target list. Simple and stable."""
        if len(targets) <= top_k:
            return list(targets)
        return list(targets[:top_k])

    def _collect_subgoals(self, hallmarks: List[str]) -> List[str]:
        """Collect default subgoals for a list of hallmarks."""
        out: List[str] = []
        for h in hallmarks:
            for sg in self.hallmark_profiles.default_subgoals_for(h):
                if sg not in out:
                    out.append(sg)
        return out

    def _choose_phase_index(
        self,
        cycles_completed: int,
        equilibrium_label: Optional[str],
        breakthrough_score: Optional[float],
    ) -> int:
        """Select a curriculum phase index with some adaptive behavior."""

        if not self.phases:
            return 0

        # Base on cycle windows
        idx = 0
        for i, phase in enumerate(self.phases):
            if phase.max_cycle is None:
                if cycles_completed >= phase.min_cycle:
                    idx = i
            else:
                if phase.min_cycle <= cycles_completed < phase.max_cycle:
                    idx = i
                    break

        # Adaptive bump conditions:
        #   - if high equilibrium and not in last phase, move forward
        #   - if breakthrough score is high, move to exploit phase
        eq = (equilibrium_label or "").lower()
        brk = breakthrough_score or 0.0

        if eq in {"high_equilibrium", "plateau_equilibrium"} and idx < len(self.phases) - 1:
            idx += 1

        if brk >= 0.8:
            idx = len(self.phases) - 1

        return max(0, min(idx, len(self.phases) - 1))

    def _choose_hallmark_for_phase(
        self,
        phase: CurriculumPhase,
        cycles_completed: int,
    ) -> str:
        """Pick a hallmark from the phase focus set in a stable way."""
        hallmarks = phase.focus_hallmarks or self.target_hallmarks
        if not hallmarks:
            hallmarks = self.hallmark_profiles.keys()
        if not hallmarks:
            return "general"

        index = max(0, cycles_completed) % len(hallmarks)
        return hallmarks[index]

    def _choose_subgoal_for_phase(
        self,
        phase: CurriculumPhase,
        hallmark: str,
        cycles_completed: int,
    ) -> Optional[str]:
        """Pick a subgoal, falling back to hallmark defaults if needed."""
        pool: List[str] = []
        if phase.subgoals:
            pool.extend(phase.subgoals)
        if not pool:
            pool.extend(self.hallmark_profiles.default_subgoals_for(hallmark))

        if not pool:
            return None

        index = max(0, cycles_completed // 5) % len(pool)
        return pool[index]

    def _derive_stage_hint(
        self,
        phase: CurriculumPhase,
        equilibrium_label: Optional[str],
        avg_rye: Optional[float],
    ) -> str:
        """Recommend an idea or verify tilt based on current context."""
        eq = (equilibrium_label or "").lower()
        avg_val = avg_rye if avg_rye is not None else 0.0

        if eq in {"high_equilibrium", "plateau_equilibrium"} and avg_val >= 0.7:
            return "verify"
        if phase.exploration_bias > phase.verify_bias:
            return "idea"
        return "mixed"

    def _derive_replay_focus_hint(
        self,
        phase: CurriculumPhase,
        breakthrough_score: Optional[float],
    ) -> str:
        """Compress replay intent into a simple hint string."""
        brk = breakthrough_score or 0.0
        if brk >= 0.8:
            return "exploit_top_motifs"
        if phase.replay_intensity >= 0.4:
            return "heavy"
        if phase.replay_intensity >= 0.2:
            return "medium"
        if phase.replay_intensity > 0:
            return "light"
        return "off"

    # Public selection API -----------------------------------------------

    def select_segment(self, run_progress: Dict[str, Any]) -> CurriculumState:
        """Return a CurriculumState for a given run progress snapshot.

        Expected keys in run_progress (all optional except cycles_completed):
            cycles_completed: int
            avg_rye: float
            equilibrium_label: str
            breakthrough_score: float
            domain: str

        The output is stable even if some fields are missing.
        """
        cycles_completed = int(run_progress.get("cycles_completed", 0))
        avg_rye = run_progress.get("avg_rye")
        equilibrium_label = run_progress.get("equilibrium_label")
        breakthrough_score = run_progress.get("breakthrough_score")
        domain_override = run_progress.get("domain")
        if domain_override:
            self.domain = str(domain_override).lower()

        phase_index = self._choose_phase_index(
            cycles_completed=cycles_completed,
            equilibrium_label=equilibrium_label,
            breakthrough_score=breakthrough_score,
        )
        phase = self.phases[phase_index]

        hallmark = self._choose_hallmark_for_phase(phase, cycles_completed)
        subgoal = self._choose_subgoal_for_phase(phase, hallmark, cycles_completed)
        biomarker_focus = self.hallmark_profiles.default_biomarkers_for(hallmark)

        stage_hint = self._derive_stage_hint(
            phase=phase,
            equilibrium_label=equilibrium_label,
            avg_rye=avg_rye,
        )
        replay_focus_hint = self._derive_replay_focus_hint(
            phase=phase,
            breakthrough_score=breakthrough_score,
        )

        # Optional funnel hint: for short runs, guide the system through a pressure-driven structure.
        funnel_step: Optional[str] = None
        try:
            total_cycles_req = run_progress.get("total_cycles_requested") or run_progress.get("total_cycles") or run_progress.get("cycle_target")
            if total_cycles_req is not None:
                total_cycles_req_int = int(total_cycles_req)
            else:
                total_cycles_req_int = None
        except Exception:
            total_cycles_req_int = None

        if total_cycles_req_int is not None and total_cycles_req_int > 0:
            try:
                idx0 = max(0, int(cycles_completed))
            except Exception:
                idx0 = 0
            if total_cycles_req_int <= 2:
                funnel_step = "cycle1_map_cluster" if idx0 == 0 else "cycle2_cull_commit"
            elif total_cycles_req_int <= 5:
                steps = ["map", "cluster", "cull", "stress", "commit"]
                if 0 <= idx0 < len(steps):
                    funnel_step = f"cycle{idx0+1}_{steps[idx0]}"

        state = CurriculumState(
            profile=self.profile_name,
            phase_name=phase.name,
            phase_index=phase_index,
            total_phases=len(self.phases),
            hallmark=hallmark,
            hallmark_list=list(phase.focus_hallmarks),
            subgoal=subgoal,
            curriculum_phase_label=phase.notes or phase.name,
            exploration_bias=phase.exploration_bias,
            verify_bias=phase.verify_bias,
            replay_intensity=phase.replay_intensity,
            stage_hint=stage_hint,
            replay_focus_hint=replay_focus_hint,
            biomarker_focus=biomarker_focus,
            funnel_step=funnel_step,
        )
        return state

    # Convenience helper for CoreAgent -----------------------------------

    def select_segment_dict(self, run_progress: Dict[str, Any]) -> Dict[str, Any]:
        """Return CurriculumState as a plain dict for easy logging or JSON."""
        state = self.select_segment(run_progress)
        return {
            "profile": state.profile,
            "phase_name": state.phase_name,
            "phase_index": state.phase_index,
            "total_phases": state.total_phases,
            "hallmark": state.hallmark,
            "hallmark_list": state.hallmark_list,
            "subgoal": state.subgoal,
            "curriculum_phase_label": state.curriculum_phase_label,
            "exploration_bias": state.exploration_bias,
            "verify_bias": state.verify_bias,
            "replay_intensity": state.replay_intensity,
            "stage_hint": state.stage_hint,
            "replay_focus_hint": state.replay_focus_hint,
            "biomarker_focus": state.biomarker_focus,
        }
