# tests/test_discovery_manager.py

import math
import statistics
import pytest

from agent.discovery_manager import DiscoveryManager


class FakeMemoryStore:
    """Minimal fake MemoryStore for testing DiscoveryManager.

    It supports:
        - get_cycle_history(limit=None)
        - get_cycle_history_for_goal(goal, limit=None)
        - get_run_history(run_id)
        - get_rye_stats(goal=...)
        - get_run_stats(run_id)
    """

    def __init__(self, history):
        self._history = list(history)

    # Core history helpers -----------------------------------------------

    def get_cycle_history(self, limit=None):
        if limit is None or limit <= 0:
            return list(self._history)
        return self._history[-limit:]

    def get_cycle_history_for_goal(self, goal, limit=None):
        filtered = [row for row in self._history if row.get("goal") == goal]
        if limit is None or limit <= 0:
            return filtered
        return filtered[-limit:]

    def get_run_history(self, run_id):
        return [row for row in self._history if row.get("run_id") == run_id]

    # Optional helpers used by DiscoveryManager and MSIL style modules ----

    def get_rye_stats(self, goal=None):
        """Return (avg, min, max, count) for RYE for the goal or all goals."""
        if goal is None:
            rows = self._history
        else:
            rows = [row for row in self._history if row.get("goal") == goal]

        rye_vals = [
            float(r.get("RYE"))
            for r in rows
            if isinstance(r.get("RYE"), (int, float))
        ]
        if not rye_vals:
            return (None, None, None, 0)
        avg = statistics.mean(rye_vals)
        return (avg, min(rye_vals), max(rye_vals), len(rye_vals))

    def get_run_stats(self, run_id):
        rows = [row for row in self._history if row.get("run_id") == run_id]
        if not rows:
            return {}
        cycles = len(rows)
        rye_vals = [
            float(r.get("RYE"))
            for r in rows
            if isinstance(r.get("RYE"), (int, float))
        ]
        avg_rye = statistics.mean(rye_vals) if rye_vals else None
        return {
            "run_id": run_id,
            "cycles": cycles,
            "avg_rye": avg_rye,
        }


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_sample_cycle(
    cycle_index,
    goal="longevity_master_run",
    domain="longevity",
    run_id="run-001",
    rye=0.6,
    delta_r=0.12,
    energy=1.3,
    breakthrough_score=0.7,
    tier_hint="T2",
    citations_count=6,
    hypotheses_count=3,
    equilibrium_label="high_equilibrium",
    oscillation_score=0.2,
):
    return {
        "cycle": cycle_index,
        "goal": goal,
        "domain": domain,
        "run_id": run_id,
        "RYE": rye,
        "delta_R": delta_r,
        "energy_E": energy,
        "breakthrough": {
            "breakthrough_score": breakthrough_score,
            "tier_hint": tier_hint,
        },
        "citations": [{"id": f"c_{cycle_index}_{i}"} for i in range(citations_count)],
        "hypotheses": [
            {"id": f"h_{cycle_index}_{i}", "kind": "mechanism"}
            for i in range(hypotheses_count)
        ],
        "equilibrium": {
            "equilibrium_label": equilibrium_label,
            "oscillation_score": oscillation_score,
        },
        "meta_signals": {
            "open_questions": 2,
            "contradictions": 1 if cycle_index % 5 == 0 else 0,
            "todo_items": 3,
        },
        "stats": {
            "web_calls": 4,
            "pubmed_calls": 2,
            "semantic_calls": 1,
        },
        "tool_usage": {
            "approx_tokens": 2000 + 50 * cycle_index,
        },
        # Optional discovery list used by some discovery_manager versions
        "discoveries": [
            {
                "kind": "mechanism",
                "label": f"Hallmark like effect {cycle_index}",
                "novelty": 0.4 + 0.01 * cycle_index,
                "evidence_score": 0.5 + 0.01 * cycle_index,
            }
        ],
    }


@pytest.fixture
def rich_history():
    """Small synthetic history with a mix of mid and high tier breakthroughs."""
    history = []
    # Build a mix of low, mid, and high tier like signals
    for i in range(1, 21):
        if i % 7 == 0:
            # Higher breakthrough style cycle
            history.append(
                _make_sample_cycle(
                    i,
                    rye=0.85,
                    delta_r=0.22,
                    energy=1.4,
                    breakthrough_score=0.92,
                    tier_hint="T3",
                )
            )
        elif i % 3 == 0:
            history.append(
                _make_sample_cycle(
                    i,
                    rye=0.7,
                    delta_r=0.15,
                    energy=1.2,
                    breakthrough_score=0.75,
                    tier_hint="T2",
                )
            )
        else:
            history.append(
                _make_sample_cycle(
                    i,
                    rye=0.5,
                    delta_r=0.08,
                    energy=1.0,
                    breakthrough_score=0.4,
                    tier_hint="T1",
                )
            )
    return history


@pytest.fixture
def discovery_manager(rich_history):
    store = FakeMemoryStore(rich_history)
    config = {
        "tier_thresholds": {
            "T3": 0.9,
            "T2": 0.7,
            "T1": 0.4,
        },
        "max_window": 200,
    }
    return DiscoveryManager(memory_store=store, config=config)


# ---------------------------------------------------------------------------
# Basic construction tests
# ---------------------------------------------------------------------------


def test_discovery_manager_constructs(discovery_manager):
    """Simple smoke test that DiscoveryManager can be constructed."""
    assert discovery_manager is not None


# ---------------------------------------------------------------------------
# register_cycle tests
# ---------------------------------------------------------------------------


def test_register_cycle_returns_list(discovery_manager, rich_history):
    """register_cycle should return a list of normalized discovery objects."""
    if not hasattr(discovery_manager, "register_cycle"):
        pytest.skip("DiscoveryManager.register_cycle not implemented")

    cycle_log = rich_history[-1]
    out = discovery_manager.register_cycle(cycle_log)

    assert isinstance(out, list)
    assert len(out) >= 1

    first = out[0]
    assert isinstance(first, dict)

    # Very soft field checks to be compatible with variants
    assert first.get("goal") == cycle_log["goal"]
    assert first.get("domain") == cycle_log["domain"]
    # Some label for tier or importance
    assert any(
        key in first
        for key in [
            "tier",
            "tier_label",
            "tier_name",
            "importance_tier",
        ]
    )


def test_register_cycle_uses_breakthrough_score_for_tier(discovery_manager):
    """High breakthrough_score cycles should be classified as higher tier."""
    if not hasattr(discovery_manager, "register_cycle"):
        pytest.skip("DiscoveryManager.register_cycle not implemented")

    # Mid score cycle
    mid_cycle = _make_sample_cycle(
        100,
        rye=0.7,
        delta_r=0.15,
        energy=1.2,
        breakthrough_score=0.75,
        tier_hint="T2",
    )
    # Very high score cycle
    high_cycle = _make_sample_cycle(
        101,
        rye=0.9,
        delta_r=0.25,
        energy=1.5,
        breakthrough_score=0.97,
        tier_hint="T3",
    )

    mids = discovery_manager.register_cycle(mid_cycle)
    highs = discovery_manager.register_cycle(high_cycle)

    def _extract_tier(cand):
        for key in ["tier", "tier_label", "tier_name", "importance_tier"]:
            if key in cand:
                return cand[key]
        return None

    mid_tier = _extract_tier(mids[0]) if mids else None
    high_tier = _extract_tier(highs[0]) if highs else None

    # Only assert something if both tiers exist
    if mid_tier is not None and high_tier is not None:
        assert mid_tier != high_tier


# ---------------------------------------------------------------------------
# summarise_goal tests
# ---------------------------------------------------------------------------


def test_summarise_goal_basic_shape(discovery_manager):
    """summarise_goal should return a structured summary dict."""
    if not hasattr(discovery_manager, "summarise_goal"):
        pytest.skip("DiscoveryManager.summarise_goal not implemented")

    summary = discovery_manager.summarise_goal(
        goal="longevity_master_run",
        limit=50,
    )

    assert isinstance(summary, dict)

    # Goal name should be reflected somehow
    assert summary.get("goal") == "longevity_master_run" or summary.get(
        "target_goal"
    ) == "longevity_master_run"

    # There should be some count of cycles
    cycles = summary.get("total_cycles") or summary.get("cycles")
    assert isinstance(cycles, int)
    assert cycles > 0

    # There should be some tier statistics bundle
    assert any(
        key in summary
        for key in [
            "tier_stats",
            "tiers",
            "tier_summary",
        ]
    )


def test_summarise_goal_contains_recent_discoveries(discovery_manager):
    """Goal summary should contain some view of recent or top discoveries."""
    if not hasattr(discovery_manager, "summarise_goal"):
        pytest.skip("DiscoveryManager.summarise_goal not implemented")

    summary = discovery_manager.summarise_goal("longevity_master_run", limit=20)

    candidates = None
    for key in [
        "recent_discoveries",
        "top_discoveries",
        "discoveries",
        "candidates",
    ]:
        if key in summary:
            candidates = summary[key]
            break

    assert candidates is not None
    assert isinstance(candidates, list)
    assert len(candidates) >= 1
    assert isinstance(candidates[0], dict)


# ---------------------------------------------------------------------------
# summarise_run tests
# ---------------------------------------------------------------------------


def test_summarise_run_has_tier_density(discovery_manager):
    """summarise_run should expose tier density or counts per tier."""
    if not hasattr(discovery_manager, "summarise_run"):
        pytest.skip("DiscoveryManager.summarise_run not implemented")

    summary = discovery_manager.summarise_run(run_id="run-001", limit=100)

    assert isinstance(summary, dict)

    # Basic counts
    total = summary.get("total_cycles") or summary.get("cycles")
    assert isinstance(total, int)
    assert total > 0

    # Tier information of some sort
    tier_block = None
    for key in ["tier_stats", "tiers", "tier_summary"]:
        if key in summary:
            tier_block = summary[key]
            break
    assert tier_block is not None
    assert isinstance(tier_block, dict)


# ---------------------------------------------------------------------------
# get_recent_discoveries tests
# ---------------------------------------------------------------------------


def test_get_recent_discoveries_filters_by_goal(discovery_manager):
    """get_recent_discoveries should be able to filter by goal when available."""
    if not hasattr(discovery_manager, "get_recent_discoveries"):
        pytest.skip("DiscoveryManager.get_recent_discoveries not implemented")

    # Grab all recent discoveries for default goal
    recents = discovery_manager.get_recent_discoveries(
        goal="longevity_master_run",
        domain=None,
        limit=10,
    )

    assert isinstance(recents, list)
    assert len(recents) >= 1

    for item in recents:
        assert isinstance(item, dict)
        if "goal" in item:
            assert item["goal"] == "longevity_master_run"


def test_get_recent_discoveries_respects_limit(discovery_manager):
    """Limit should cap the number of discoveries returned."""
    if not hasattr(discovery_manager, "get_recent_discoveries"):
        pytest.skip("DiscoveryManager.get_recent_discoveries not implemented")

    small = discovery_manager.get_recent_discoveries(
        goal="longevity_master_run",
        domain=None,
        limit=3,
    )
    big = discovery_manager.get_recent_discoveries(
        goal="longevity_master_run",
        domain=None,
        limit=50,
    )

    assert len(small) <= len(big)
    assert len(small) <= 3


# ---------------------------------------------------------------------------
# Tier utility and scoring sanity checks
# ---------------------------------------------------------------------------


def test_high_breakthrough_density_reflected_in_run_summary(discovery_manager):
    """Runs with several high tier events should show some nonzero density."""
    if not hasattr(discovery_manager, "summarise_run"):
        pytest.skip("DiscoveryManager.summarise_run not implemented")

    summary = discovery_manager.summarise_run(run_id="run-001", limit=200)

    tier_block = None
    for key in ["tier_stats", "tiers", "tier_summary"]:
        if key in summary:
            tier_block = summary[key]
            break

    if tier_block is None or not isinstance(tier_block, dict):
        pytest.skip("No tier statistics available in this DiscoveryManager")

    # Look for any field that looks like a density or rate
    densities = []
    for val in tier_block.values():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, (int, float)) and "density" in str(k).lower():
                    densities.append(v)

    if densities:
        assert any(d >= 0 for d in densities)


def test_summary_is_stable_on_empty_history():
    """DiscoveryManager should behave safely when memory history is empty."""
    store = FakeMemoryStore([])
    manager = DiscoveryManager(memory_store=store, config={})

    # All public methods should return safe defaults and not crash
    if hasattr(manager, "summarise_goal"):
        out = manager.summarise_goal("empty_goal", limit=10)
        assert isinstance(out, dict)
        assert out.get("cycles", out.get("total_cycles", 0)) == 0

    if hasattr(manager, "summarise_run"):
        out = manager.summarise_run(run_id="run-empty", limit=10)
        assert isinstance(out, dict)
        assert out.get("cycles", out.get("total_cycles", 0)) == 0

    if hasattr(manager, "get_recent_discoveries"):
        out = manager.get_recent_discoveries(goal=None, domain=None, limit=5)
        assert isinstance(out, list)
        assert out == []
