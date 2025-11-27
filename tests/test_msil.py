# tests/test_msil.py

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from agent.msil import (
    MetaSkillIntelligenceLayer,
    SkillDimensionScores,
    DomainProfile,
    MSILSnapshot,
)


class MemoryStoreStub:
    """Stub MemoryStore for MSIL tests.

    This stub implements all optional methods that MSIL tries to call so
    tests can cover the richer code paths.
    """

    def __init__(self, history: Optional[List[Dict[str, Any]]] = None) -> None:
        self._history: List[Dict[str, Any]] = history or []
        self._run_history: Dict[str, List[Dict[str, Any]]] = {}
        self._run_stats: Dict[str, Dict[str, Any]] = {}
        self._replay_items: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def add_cycle(self, row: Dict[str, Any]) -> None:
        self._history.append(row)

    def set_run_history(self, run_id: str, rows: List[Dict[str, Any]]) -> None:
        self._run_history[run_id] = rows

    def set_run_stats(self, run_id: str, stats: Dict[str, Any]) -> None:
        self._run_stats[run_id] = stats

    def set_replay_items_for_goal(
        self, goal: str, items: List[Dict[str, Any]]
    ) -> None:
        self._replay_items[goal] = items

    # Methods used by MSIL (soft assumptions)
    def get_cycle_history_for_goal(
        self,
        goal: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = [r for r in self._history if r.get("goal") == goal]
        if limit is None:
            return rows
        return rows[-limit:]

    def get_cycle_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit is None:
            return list(self._history)
        return self._history[-limit:]

    def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        return self._run_history.get(run_id, [])

    def get_run_stats(self, run_id: str) -> Dict[str, Any]:
        return self._run_stats.get(run_id, {})

    # ------------------------------------------------------------------
    # RYE and replay helpers
    # ------------------------------------------------------------------
    def get_rye_stats(
        self,
        goal: Optional[str] = None,
    ) -> Optional[tuple]:
        rows = self._history
        if goal is not None:
            rows = [r for r in rows if r.get("goal") == goal]

        rye_vals: List[float] = []
        for r in rows:
            v = r.get("RYE")
            if isinstance(v, (int, float)):
                rye_vals.append(float(v))

        if not rye_vals:
            return None

        avg = sum(rye_vals) / len(rye_vals)
        mn = min(rye_vals)
        mx = max(rye_vals)
        return avg, mn, mx, len(rye_vals)

    def get_replay_items_for_goal(
        self,
        goal: str,
    ) -> List[Dict[str, Any]]:
        return list(self._replay_items.get(goal, []))


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_dummy_cycle(
    cycle_index: int,
    goal: str = "test_goal",
    domain: str = "general",
    rye: float = 0.1,
    breakthrough_score: float = 0.0,
    ts: Optional[datetime] = None,
) -> Dict[str, Any]:
    if ts is None:
        ts = datetime(2025, 1, 1) + timedelta(minutes=cycle_index)

    return {
        "cycle": cycle_index,
        "goal": goal,
        "domain": domain,
        "timestamp": ts.isoformat(),
        "RYE": rye,
        "delta_R": rye * 0.5,
        "energy_E": 1.0 + rye,
        "breakthrough": {
            "breakthrough_score": breakthrough_score,
        },
        "citations": ["c1", "c2"] if rye > 0 else [],
        "hypotheses": [{"id": 1}] if rye > 0 else [],
        "equilibrium": {
            "equilibrium_label": "plateau_equilibrium" if rye >= 0.1 else "low_efficiency",
            "oscillation_score": 0.2,
        },
        "meta_signals": {
            "open_questions": 2,
            "contradictions": 0,
            "todo_items": 1,
        },
        "stats": {
            "web_calls": 2,
            "pubmed_calls": 2 if domain == "longevity" else 0,
            "semantic_calls": 1,
        },
        "tool_usage": {
            "approx_tokens": 1000,
        },
        "stage": "verify",
        "role": "researcher",
        "tags": ["curriculum:phase1"],
    }


@pytest.fixture
def simple_history() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(30):
        rye = 0.05 + 0.01 * i
        br = 0.3 if i < 20 else 0.9
        rows.append(_make_dummy_cycle(i, goal="test_goal", domain="longevity", rye=rye, breakthrough_score=br))
    return rows


@pytest.fixture
def memory_store(simple_history: List[Dict[str, Any]]) -> MemoryStoreStub:
    store = MemoryStoreStub(simple_history)
    # Add a run id mapping
    store.set_run_history("run_1", list(simple_history))
    store.set_run_stats("run_1", {"run_id": "run_1", "notes": "stub run"})
    # Add replay items for the goal
    store.set_replay_items_for_goal(
        "test_goal",
        [
            {"rye_score": 0.7},
            {"rye_score": 0.9},
        ],
    )
    return store


# ----------------------------------------------------------------------
# Basic construction and disabled mode
# ----------------------------------------------------------------------


def test_msil_constructs_and_uses_defaults(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    assert msil.enabled is True
    assert msil.window > 0
    assert msil.long_window >= msil.window
    assert msil.min_cycles > 0


def test_msil_disabled_snapshot_on_observe_cycle(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={"msil_enabled": False})
    cycle_log = _make_dummy_cycle(0)
    snapshot = msil.observe_cycle(cycle_log)
    assert isinstance(snapshot, MSILSnapshot)
    assert snapshot.intelligence_stage == "disabled"
    assert snapshot.msil_score == 0.0
    # Priority message should reflect disabled state
    assert "MSIL is disabled" in snapshot.actions["priority"][0]


def test_msil_summarise_run_disabled(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={"msil_enabled": False})
    summary = msil.summarise_run(goal="test_goal")
    assert summary["enabled"] is False
    assert "reason" in summary


# ----------------------------------------------------------------------
# observe_cycle and snapshots
# ----------------------------------------------------------------------


def test_msil_observe_cycle_builds_snapshot(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={"msil_window": 20})

    latest_cycle = memory_store.get_cycle_history_for_goal("test_goal", limit=1)[-1]
    snapshot = msil.observe_cycle(latest_cycle)

    assert isinstance(snapshot, MSILSnapshot)
    assert snapshot.goal == "test_goal"
    assert 0.0 <= snapshot.msil_score <= 1.0
    assert snapshot.total_cycles_for_goal >= 1
    assert snapshot.recent_window > 0

    # Skills and domain profiles should be populated
    skills_dict = snapshot.skills.to_dict()
    assert set(skills_dict.keys()) == {
        "reasoning",
        "literature_navigation",
        "hypothesis_generation",
        "planning_and_curriculum",
        "stability_and_safety",
    }
    for v in skills_dict.values():
        assert 0.0 <= v <= 1.0

    assert snapshot.per_domain_profiles
    assert all(isinstance(p, DomainProfile) for p in snapshot.per_domain_profiles)

    # Extras should have RYE stats and replay stats
    extras = snapshot.extras
    assert "avg_rye_for_goal" in extras
    assert "replay_stats" in extras
    replay_stats = extras["replay_stats"]
    assert replay_stats["items_for_goal"] == 2
    assert replay_stats["high_rye_items"] == 2
    assert replay_stats["replay_density"] > 0.0


def test_msil_observe_cycle_intelligence_stage_progression(memory_store: MemoryStoreStub) -> None:
    # Use low msil_min_cycles so stage moves beyond warmup
    msil = MetaSkillIntelligenceLayer(
        memory_store,
        config={
            "msil_min_cycles": 10,
            "msil_window": 20,
        },
    )
    latest_cycle = memory_store.get_cycle_history_for_goal("test_goal", limit=1)[-1]
    snapshot = msil.observe_cycle(latest_cycle)

    assert snapshot.total_cycles_for_goal >= 10
    assert snapshot.intelligence_stage in {
        "learning_basic",
        "learning_capable",
        "specialised_expert",
        "broad_expert",
        "frontier_candidate",
    }


# ----------------------------------------------------------------------
# summarise_run
# ----------------------------------------------------------------------


def test_msil_summarise_run_with_goal(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    summary = msil.summarise_run(goal="test_goal", limit=100)

    assert summary["enabled"] is True
    assert summary["cycles"] > 0
    assert 0.0 <= summary["msil_score"] <= 1.0
    assert summary["intelligence_stage"]
    assert isinstance(summary["skills"], dict)
    assert summary["domain_profiles"]
    assert isinstance(summary["diagnostics"], dict)


def test_msil_summarise_run_with_run_id(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    summary = msil.summarise_run(run_id="run_1", limit=500)

    assert summary["enabled"] is True
    assert summary["cycles"] == len(memory_store.get_run_history("run_1"))
    assert summary["run_stats"].get("run_id") == "run_1"


# ----------------------------------------------------------------------
# Internal scoring behavior (white box but stable)
# ----------------------------------------------------------------------


def test_skill_dimension_scores_clamped_between_zero_and_one(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    history = memory_store.get_cycle_history(limit=200)
    skills = msil._compute_skill_dimensions(history)  # type: ignore[attr-defined]

    assert isinstance(skills, SkillDimensionScores)
    for v in skills.to_dict().values():
        assert 0.0 <= v <= 1.0


def test_domain_profiles_sorted_by_score(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    history = memory_store.get_cycle_history(limit=200)
    profiles = msil._compute_domain_profiles(history)  # type: ignore[attr-defined]

    if len(profiles) > 1:
        scores = [p.msil_score for p in profiles]
        assert scores == sorted(scores, reverse=True)


def test_aggregate_msil_score_responds_to_skill_boost(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    history = memory_store.get_cycle_history(limit=200)
    profiles = msil._compute_domain_profiles(history)  # type: ignore[attr-defined]
    skills = msil._compute_skill_dimensions(history)  # type: ignore[attr-defined]

    base_score = msil._aggregate_msil_score(skills, profiles)  # type: ignore[attr-defined]
    # Boost reasoning skill and check that score increases
    boosted = SkillDimensionScores(
        reasoning=min(1.0, skills.reasoning + 0.3),
        literature_navigation=skills.literature_navigation,
        hypothesis_generation=skills.hypothesis_generation,
        planning_and_curriculum=skills.planning_and_curriculum,
        stability_and_safety=skills.stability_and_safety,
    )
    boosted_score = msil._aggregate_msil_score(boosted, profiles)  # type: ignore[attr-defined]

    assert boosted_score >= base_score


def test_build_run_diagnostics_includes_energy_and_tools(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    history = memory_store.get_cycle_history(limit=100)
    diag = msil._build_run_diagnostics(history)  # type: ignore[attr-defined]

    assert "series" in diag
    assert "rye" in diag["series"]
    assert "energy_and_tools" in diag
    e = diag["energy_and_tools"]
    assert "avg_energy_per_cycle" in e
    assert "avg_web_calls_per_cycle" in e
    assert "avg_tokens_per_cycle" in e


def test_recommend_actions_returns_risk_flag_when_oscillatory(memory_store: MemoryStoreStub) -> None:
    msil = MetaSkillIntelligenceLayer(memory_store, config={})
    history = memory_store.get_cycle_history(limit=30)

    # Force recent history to look oscillatory
    recent = []
    for i, row in enumerate(history[-10:]):
        r = dict(row)
        eq = dict(r.get("equilibrium") or {})
        eq["equilibrium_label"] = "oscillating"
        eq["oscillation_score"] = 0.9
        r["equilibrium"] = eq
        recent.append(r)

    skills = msil._compute_skill_dimensions(history)  # type: ignore[attr-defined]
    profiles = msil._compute_domain_profiles(history)  # type: ignore[attr-defined]
    score = msil._aggregate_msil_score(skills, profiles)  # type: ignore[attr-defined]

    actions = msil._recommend_actions(  # type: ignore[attr-defined]
        msil_score=score,
        skills=skills,
        domain_profiles=profiles,
        total_cycles=len(history),
        recent_history=recent,
    )

    monitoring = actions.get("monitoring", {})
    assert monitoring.get("risk_flag") == "oscillation_risk"
