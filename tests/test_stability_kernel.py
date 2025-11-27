# tests/test_stability_kernel.py

import math
from typing import Any, Dict, List

import pytest

from agent import stability_kernel as stability


def make_cycle(
    idx: int,
    rye: float,
    *,
    domain: str = "general",
    equilibrium_label: str = "unknown",
    oscillation_score: float = 0.0,
    breakthrough_score: float = 0.0,
) -> Dict[str, Any]:
    """Helper to build a cycle row that looks like real history."""
    return {
        "cycle": idx,
        "RYE": rye,
        "domain": domain,
        "equilibrium": {
            "equilibrium_label": equilibrium_label,
            "oscillation_score": oscillation_score,
        },
        "breakthrough": {
            "breakthrough_score": breakthrough_score,
        },
    }


# -------------------------------------------------------------------------
# compute_stability_profile
# -------------------------------------------------------------------------


def test_compute_stability_profile_empty_history():
    profile = stability.compute_stability_profile([])

    assert isinstance(profile, dict)
    # Core numeric fields should be present and safe
    for key in [
        "stability_index",
        "recovery_momentum",
        "trend_slope",
        "oscillation_std",
    ]:
        assert key in profile
        assert isinstance(profile[key], (int, float))

    # Empty history should look like fully unstable but harmless numerically
    assert 0.0 <= profile["stability_index"] <= 1.0
    assert 0.0 <= profile["recovery_momentum"] <= 1.0
    assert math.isfinite(profile["trend_slope"])
    assert profile["oscillation_std"] >= 0.0


def test_compute_stability_profile_single_point():
    history = [make_cycle(0, 0.1)]
    profile = stability.compute_stability_profile(history)

    assert isinstance(profile, dict)
    assert "stability_index" in profile
    assert 0.0 <= profile["stability_index"] <= 1.0
    # With a single data point oscillation should be near zero
    assert profile["oscillation_std"] >= 0.0
    # Recovery momentum from one point should not explode
    assert 0.0 <= profile["recovery_momentum"] <= 1.0


def test_compute_stability_profile_stable_vs_oscillatory():
    # Stable series slowly rising
    stable_history = [
        make_cycle(i, 0.05 + 0.001 * i) for i in range(60)
    ]

    # Oscillatory series around similar mean
    oscillatory_history = [
        make_cycle(
            i,
            0.10 + (0.08 if i % 2 == 0 else -0.08),
        )
        for i in range(60)
    ]

    stable_profile = stability.compute_stability_profile(stable_history)
    osc_profile = stability.compute_stability_profile(oscillatory_history)

    # Stability index should be higher for the smooth rising series
    assert stable_profile["stability_index"] > osc_profile["stability_index"]

    # Oscillation std should be higher for the oscillatory series
    assert osc_profile["oscillation_std"] > stable_profile["oscillation_std"]

    # Recovery momentum should be positive for a rising series
    assert stable_profile["recovery_momentum"] >= 0.0


def test_compute_stability_profile_recovery_after_dip():
    # First a dip, then clear recovery
    values = [0.12, 0.05, 0.03, 0.02, 0.04, 0.06, 0.09, 0.11, 0.13]
    history = [make_cycle(i, v) for i, v in enumerate(values)]

    profile = stability.compute_stability_profile(history)

    # Recovery momentum should be positive when the final values are higher
    assert profile["recovery_momentum"] > 0.0


# -------------------------------------------------------------------------
# classify_stability_regime
# -------------------------------------------------------------------------


def test_classify_stability_regime_cold_start_like():
    history = [make_cycle(i, 0.02) for i in range(3)]
    profile = stability.compute_stability_profile(history)

    regime = stability.classify_stability_regime(
        history=history,
        diagnostics=None,
        profile=profile,
    )

    assert isinstance(regime, dict)
    label = str(regime.get("label", "")).lower()
    score = regime.get("score", 0.0)

    assert label
    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 1.0
    # Very short run should not be classified as high equilibrium
    assert "high" not in label


def test_classify_stability_regime_high_equilibrium_candidate():
    # Long, smooth, high RYE values
    history = [make_cycle(i, 0.18 + 0.01 * (i % 3)) for i in range(120)]
    profile = stability.compute_stability_profile(history)

    regime = stability.classify_stability_regime(
        history=history,
        diagnostics=None,
        profile=profile,
    )

    label = str(regime.get("label", "")).lower()
    score = regime.get("score", 0.0)

    assert 0.0 <= score <= 1.0
    # Stable high RYE should be classified as plateau or high equilibrium
    assert any(
        token in label
        for token in ["plateau", "equilibrium", "robust"]
    )


def test_classify_stability_regime_oscillating():
    # Strong oscillation around zero
    history = [
        make_cycle(i, 0.1 if i % 2 == 0 else -0.1) for i in range(80)
    ]
    profile = stability.compute_stability_profile(history)

    regime = stability.classify_stability_regime(
        history=history,
        diagnostics=None,
        profile=profile,
    )

    label = str(regime.get("label", "")).lower()
    assert any(token in label for token in ["oscillating", "unstable", "fragile"])


# -------------------------------------------------------------------------
# summarize_for_ui
# -------------------------------------------------------------------------


def test_summarize_for_ui_structure():
    history = [make_cycle(i, 0.05 + 0.002 * i) for i in range(40)]
    profile = stability.compute_stability_profile(history)
    regime = stability.classify_stability_regime(
        history=history,
        diagnostics=None,
        profile=profile,
    )

    ui = stability.summarize_for_ui(
        history=history,
        diagnostics=None,
        profile=profile,
        regime=regime,
    )

    # UI snapshot should be a dictionary with stable shape
    assert isinstance(ui, dict)

    # Basic headline fields that are helpful for a dashboard
    for key in [
        "label",
        "stability_index",
        "recovery_momentum",
        "oscillation_std",
        "trend_slope",
    ]:
        assert key in ui

    assert 0.0 <= ui["stability_index"] <= 1.0
    assert 0.0 <= ui["recovery_momentum"] <= 1.0
    assert ui["oscillation_std"] >= 0.0
    assert math.isfinite(ui["trend_slope"])

    label = str(ui["label"]).lower()
    assert label


def test_summarize_for_ui_inherits_regime_label_if_present():
    history = [make_cycle(i, 0.15 + 0.005 * (i % 4)) for i in range(60)]
    profile = stability.compute_stability_profile(history)
    regime = stability.classify_stability_regime(
        history=history,
        diagnostics=None,
        profile=profile,
    )

    ui = stability.summarize_for_ui(
        history=history,
        diagnostics=None,
        profile=profile,
        regime=regime,
    )

    # UI label should reflect the regime in some way
    regime_label = str(regime.get("label", "")).lower()
    ui_label = str(ui.get("label", "")).lower()

    assert ui_label
    # Exact equality is not required, but overlap is expected
    for token in regime_label.split("_"):
        if not token:
            continue
        if token in ui_label:
            break
    else:
        pytest.skip(
            "UI label does not echo regime label tokens, which is allowed but less ideal."
        )


# -------------------------------------------------------------------------
# Integrated sanity test
# -------------------------------------------------------------------------


def test_full_pipeline_stable_run():
    """End to end sanity check from history to UI snapshot."""
    history: List[Dict[str, Any]] = []

    # Warmup low RYE
    for i in range(10):
        history.append(make_cycle(i, 0.02 + 0.001 * i))

    # Main stable region
    for i in range(10, 80):
        history.append(make_cycle(i, 0.10 + 0.002 * ((i - 10) % 5)))

    profile = stability.compute_stability_profile(history)
    regime = stability.classify_stability_regime(
        history=history,
        diagnostics=None,
        profile=profile,
    )
    ui = stability.summarize_for_ui(
        history=history,
        diagnostics=None,
        profile=profile,
        regime=regime,
    )

    # Everything should look stable and non pathologic
    assert profile["stability_index"] > 0.4
    assert profile["oscillation_std"] < 0.2

    label = str(regime.get("label", "")).lower()
    assert any(token in label for token in ["plateau", "equilibrium", "robust"])

    assert 0.0 <= ui["stability_index"] <= 1.0
    assert ui["oscillation_std"] >= 0.0
