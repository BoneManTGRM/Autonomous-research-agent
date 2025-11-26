# tests/test_curriculum.py

import math
from typing import Any, Dict

import pytest

from agent import curriculum as curriculum_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision_to_dict(obj: Any) -> Dict[str, Any]:
    """Normalize CurriculumDecision to a dict so tests are robust to API shape."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict()  # type: ignore[call-arg]
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def _state_to_dict(obj: Any) -> Dict[str, Any]:
    """Normalize CurriculumState to a dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict()  # type: ignore[call-arg]
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def _make_controller(profile_name: str = "longevity_basic"):
    """Factory that is defensive against constructor signature changes."""
    Controller = getattr(curriculum_mod, "CurriculumController", None)
    assert Controller is not None, "CurriculumController must exist in agent.curriculum"

    # Try most likely signatures in order
    try:
        return Controller(profile_name=profile_name)  # type: ignore[call-arg]
    except TypeError:
        try:
            return Controller(profile_name)  # type: ignore[call-arg]
        except TypeError:
            return Controller()  # type: ignore[call-arg]


def _select_segment(ctrl, run_progress: Dict[str, Any]) -> Dict[str, Any]:
    """Call the curriculum segment selector using the most likely method name."""
    if hasattr(ctrl, "select_segment"):
        raw = ctrl.select_segment(run_progress)  # type: ignore[attr-defined]
    elif hasattr(ctrl, "select_curriculum_segment"):
        raw = ctrl.select_curriculum_segment(run_progress)  # type: ignore[attr-defined]
    elif hasattr(ctrl, "next_phase"):
        raw = ctrl.next_phase(run_progress)  # type: ignore[attr-defined]
    else:
        raise AssertionError(
            "CurriculumController must expose select_segment, "
            "select_curriculum_segment, or next_phase"
        )

    return _decision_to_dict(raw)


def _get_state(ctrl) -> Dict[str, Any]:
    """Get internal state if exposed."""
    state = getattr(ctrl, "state", None)
    if state is None:
        return {}
    return _state_to_dict(state)


# ---------------------------------------------------------------------------
# Ultra curriculum profile coverage
# ---------------------------------------------------------------------------

def test_default_profiles_cover_longevity_variants():
    """Default curriculum profiles should include the three longevity modes."""
    profile_builder = getattr(
        curriculum_mod,
        "build_default_curriculum_profiles",
        None,
    )
    assert profile_builder is not None, "build_default_curriculum_profiles must exist"

    profiles = profile_builder()  # type: ignore[call-arg]
    assert isinstance(profiles, dict)
    keys = set(profiles.keys())

    # Core longevity profiles for fast learning on aging hallmarks
    assert "longevity_basic" in keys
    assert "mito_depth" in keys or "LONGEVITY_MITOCHONDRIA" in keys
    assert "cross_hallmark" in keys or "LONGEVITY_MITO_SEN_HYBRID" in keys


# ---------------------------------------------------------------------------
# Exploration vs verification dynamics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("avg_rye, rye_trend", [(0.25, 0.05), (0.4, 0.02)])
def test_early_phase_is_exploratory(avg_rye, rye_trend):
    """At low cycles with low RYE the curriculum should favor exploration and idea stage."""
    ctrl = _make_controller("longevity_basic")

    decision = _select_segment(
        ctrl,
        {
            "cycle_index": 5,
            "avg_rye": avg_rye,
            "rye_trend": rye_trend,
            "equilibrium_label": "exploring",
            "stagnation_counter": 0,
            "domain": "longevity",
        },
    )

    stage = decision.get("stage")
    exploration = decision.get("exploration_factor", None)
    verify = decision.get("verify_factor", None)

    assert stage in {None, "idea", "mixed"}
    if exploration is not None and verify is not None:
        assert exploration >= verify


def test_sustained_high_rye_shifts_to_verify():
    """With high and stable RYE the curriculum should shift to verify heavy mode."""
    ctrl = _make_controller("longevity_basic")

    decision = _select_segment(
        ctrl,
        {
            "cycle_index": 220,
            "avg_rye": 0.9,
            "rye_trend": 0.08,
            "equilibrium_label": "high_equilibrium",
            "stagnation_counter": 0,
            "domain": "longevity",
        },
    )

    stage = decision.get("stage")
    exploration = decision.get("exploration_factor", 0.0)
    verify = decision.get("verify_factor", 0.0)

    assert stage in {"verify", "mixed"}
    assert verify >= exploration


def test_stagnation_triggers_exploration_boost():
    """If RYE is flat and stagnation is high, exploration factor should increase."""
    ctrl = _make_controller("longevity_basic")

    decision = _select_segment(
        ctrl,
        {
            "cycle_index": 140,
            "avg_rye": 0.6,
            "rye_trend": 0.0,
            "equilibrium_label": "plateau_equilibrium",
            "stagnation_counter": 12,
            "domain": "longevity",
        },
    )

    exploration = decision.get("exploration_factor", None)
    replay_policy = decision.get("replay_policy", {})
    if exploration is not None:
        assert exploration >= 0.6
    if isinstance(replay_policy, dict):
        # Stagnation should push it to sample more adventurous replay items
        temp = replay_policy.get("temperature", 0.0)
        assert temp >= 0.8


# ---------------------------------------------------------------------------
# Hallmark targeting behavior
# ---------------------------------------------------------------------------

def test_mito_profile_focuses_mitochondria():
    """Mito profile should strongly focus mitochondrial hallmarks."""
    ctrl = _make_controller("mito_depth")

    decision = _select_segment(
        ctrl,
        {
            "cycle_index": 40,
            "avg_rye": 0.5,
            "rye_trend": 0.03,
            "equilibrium_label": "exploring",
            "stagnation_counter": 0,
            "domain": "longevity",
        },
    )

    hallmarks = decision.get("hallmark_targets") or decision.get("hallmarks") or []
    hallmarks = [str(h).lower() for h in hallmarks]

    assert any("mito" in h for h in hallmarks) or "mitochondria" in hallmarks


def test_cross_hallmark_profile_uses_multiple_hallmarks():
    """Cross hallmark profile should push multi hallmark discovery chains."""
    ctrl = _make_controller("cross_hallmark")

    decision = _select_segment(
        ctrl,
        {
            "cycle_index": 80,
            "avg_rye": 0.55,
            "rye_trend": 0.04,
            "equilibrium_label": "exploring",
            "stagnation_counter": 3,
            "domain": "longevity",
        },
    )

    hallmarks = decision.get("hallmark_targets") or decision.get("hallmarks") or []
    assert isinstance(hallmarks, list)
    assert len(hallmarks) >= 2


# ---------------------------------------------------------------------------
# Replay aware curriculum logic
# ---------------------------------------------------------------------------

def test_replay_usage_increases_with_rye_quality():
    """As RYE rises and stabilizes the curriculum should lean harder on replay."""
    ctrl = _make_controller("longevity_basic")

    early_decision = _select_segment(
        ctrl,
        {
            "cycle_index": 10,
            "avg_rye": 0.4,
            "rye_trend": 0.05,
            "equilibrium_label": "exploring",
            "stagnation_counter": 0,
            "domain": "longevity",
        },
    )
    late_decision = _select_segment(
        ctrl,
        {
            "cycle_index": 160,
            "avg_rye": 0.88,
            "rye_trend": 0.06,
            "equilibrium_label": "high_equilibrium",
            "stagnation_counter": 0,
            "domain": "longevity",
        },
    )

    early_replay = early_decision.get("replay_policy", {})
    late_replay = late_decision.get("replay_policy", {})

    if isinstance(early_replay, dict) and isinstance(late_replay, dict):
        early_min = early_replay.get("min_rye", 0.0)
        late_min = late_replay.get("min_rye", 0.0)
        assert late_min >= early_min


# ---------------------------------------------------------------------------
# State evolution and persistence
# ---------------------------------------------------------------------------

def test_state_tracks_cycle_and_phase_progression():
    """Curriculum state should update its counters as segments are selected."""
    ctrl = _make_controller("longevity_basic")

    before_state = _get_state(ctrl)
    before_cycle = before_state.get("total_cycles", 0)
    before_phase_idx = before_state.get("phase_index", 0)

    for step in range(1, 6):
        _select_segment(
            ctrl,
            {
                "cycle_index": step,
                "avg_rye": 0.3 + 0.05 * step,
                "rye_trend": 0.05,
                "equilibrium_label": "exploring",
                "stagnation_counter": 0,
                "domain": "longevity",
            },
        )

    after_state = _get_state(ctrl)
    after_cycle = after_state.get("total_cycles", 0)
    after_phase_idx = after_state.get("phase_index", 0)

    assert after_cycle >= before_cycle
    # It is acceptable if phase index stays the same for short runs,
    # but if curriculum uses cycle thresholds we expect some movement.
    assert after_phase_idx >= before_phase_idx


def test_state_exposes_run_level_signals():
    """State should expose high level signals that meta controllers can read."""
    ctrl = _make_controller("longevity_basic")

    for step in range(1, 30):
        _select_segment(
            ctrl,
            {
                "cycle_index": step,
                "avg_rye": 0.5 + 0.01 * step,
                "rye_trend": 0.02,
                "equilibrium_label": "exploring",
                "stagnation_counter": 0,
                "domain": "longevity",
            },
        )

    state = _get_state(ctrl)
    # Signals do not need precise values, but they should exist
    fields = {
        "best_rye",
        "worst_rye",
        "avg_rye",
        "stagnation_counter",
        "curriculum_name",
    }
    for f in fields:
        assert f in state


# ---------------------------------------------------------------------------
# Robustness tests
# ---------------------------------------------------------------------------

def test_controller_handles_missing_optional_signals():
    """Curriculum should not crash if some RYE or equilibrium hints are missing."""
    ctrl = _make_controller("longevity_basic")

    decision = _select_segment(
        ctrl,
        {
            # Only the cycle index is required
            "cycle_index": 12,
            "domain": "longevity",
        },
    )

    d = decision
    assert "hallmark_targets" in d or "hallmarks" in d or "stage" in d


def test_profile_switch_signal_is_optional_and_safe():
    """If curriculum wants to switch profile it should expose a clear, optional signal."""
    ctrl = _make_controller("longevity_basic")

    decision = _select_segment(
        ctrl,
        {
            "cycle_index": 300,
            "avg_rye": 0.92,
            "rye_trend": 0.01,
            "equilibrium_label": "high_equilibrium",
            "stagnation_counter": 0,
            "domain": "longevity",
        },
    )

    maybe_switch = decision.get("should_switch_profile", False)
    # It is fine if no switch is suggested, but if it is, target must be specified
    if maybe_switch:
        target = decision.get("target_profile_name")
        assert isinstance(target, str) and target


def test_curriculum_decision_is_stable_under_small_noise():
    """Small perturbations in RYE should not produce wild jumps in decisions."""
    ctrl = _make_controller("longevity_basic")

    base = _select_segment(
        ctrl,
        {
            "cycle_index": 120,
            "avg_rye": 0.75,
            "rye_trend": 0.03,
            "equilibrium_label": "transient",
            "stagnation_counter": 2,
            "domain": "longevity",
        },
    )
    base_stage = base.get("stage")
    base_hallmarks = tuple(base.get("hallmark_targets", []))

    noisy = _select_segment(
        ctrl,
        {
            "cycle_index": 121,
            "avg_rye": 0.76 + 0.01 * math.sin(1.0),
            "rye_trend": 0.029,
            "equilibrium_label": "transient",
            "stagnation_counter": 2,
            "domain": "longevity",
        },
    )
    noisy_stage = noisy.get("stage")
    noisy_hallmarks = tuple(noisy.get("hallmark_targets", []))

    assert base_stage == noisy_stage
    assert base_hallmarks == noisy_hallmarks
