"""Meta Skill Intelligence Layer (MSIL).

This module defines a MetaSkillIntelligenceLayer that sits on top of
TGRM cycles, RYE metrics, and swarm history. It is designed to be an
optional ceiling for the Autonomous Research Agent that tracks:

    - Multi dimensional "agent intelligence" over time
    - Domain and role specific skill profiles
    - Stability and learning dynamics from RYE and delta_R
    - Discovery potential and breakthrough density
    - Run level diagnostics for 90 day and multi run experiments
    - Optional meta signals from TGRM (open questions, contradictions)
    - Optional tool usage and energy style patterns
    - Optional curriculum and replay hints

MSIL is purely analytic and advisory:
    - It never executes tools directly.
    - It reads cycle history from MemoryStore and produces:
        * msil_score on a 0 to 1 scale
        * skill dimensions (reasoning, literature, hypothesis, planning)
        * domain profiles (longevity, math, general, others)
        * curriculum and swarm configuration suggestions
        * monitoring and safety hints

It is safe to ignore:
    - CoreAgent and TGRMLoop can run without importing this file.
    - If RYE helper functions are missing, internal fallbacks are used.
    - If MemoryStore lacks extended methods, MSIL falls back safely.

Design goals
------------
    - No circular imports with core_agent or tgrm_loop.
    - Soft dependency on rye_metrics and MemoryStore extras.
    - Lightweight enough for single agent runs, but rich enough
      for full 90 day swarm experiments.

Typical usage (optional)
------------------------
    from msil import MetaSkillIntelligenceLayer

    msil = MetaSkillIntelligenceLayer(memory_store, config={"msil_window": 200})

    # After each cycle (from CoreAgent or engine worker):
    snapshot = msil.observe_cycle(cycle_log)
    # snapshot.to_dict() for logging or UI

    # For full run summaries:
    run_view = msil.summarise_run(goal="anti aging longevity master run")

You can also use the light wrapper:

    from msil import analyze_run
    profile = analyze_run(history, goal="longevity")

which does not require a full MemoryStore instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import statistics


# Optional imports from rye_metrics. Fallbacks are defined if not present.
try:
    # Try absolute import first (top-level file layout), then relative.
    try:
        from rye_metrics import (  # type: ignore[attr-defined]
            rolling_rye,
            stability_index,
            recovery_momentum,
            regression_rye_slope,
            rye_percentiles,
            build_run_diagnostics,
        )
    except Exception:  # pragma: no cover
        from .rye_metrics import (  # type: ignore[attr-defined]
            rolling_rye,
            stability_index,
            recovery_momentum,
            regression_rye_slope,
            rye_percentiles,
            build_run_diagnostics,
        )
except Exception:  # pragma: no cover
    def rolling_rye(values: List[float], window: int = 20) -> List[float]:
        if not values:
            return []
        out: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            win = values[start : i + 1]
            out.append(sum(win) / len(win))
        return out

    def stability_index(values: List[float]) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return 1.0
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            return 0.0
        stdev = statistics.pstdev(values)
        ratio = stdev / abs(mean_val)
        return max(0.0, min(1.0, 1.0 - ratio))

    def recovery_momentum(values: List[float]) -> float:
        if not values:
            return 0.0
        if len(values) < 2:
            return 0.0
        return max(
            0.0,
            min(1.0, (values[-1] - values[0]) / max(1e-6, abs(values[0]) + 1e-6)),
        )

    def regression_rye_slope(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(values) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        den = sum((x - mean_x) ** 2 for x in xs) or 1.0
        return num / den

    def rye_percentiles(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"p10": None, "p50": None, "p90": None}
        sorted_vals = sorted(values)

        def pct(p: float) -> float:
            if not sorted_vals:
                return 0.0
            idx = min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1)))
            return sorted_vals[idx]

        return {
            "p10": pct(0.10),
            "p50": pct(0.50),
            "p90": pct(0.90),
        }

    def build_run_diagnostics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        rye_vals = [
            float(row.get("RYE"))
            for row in history
            if isinstance(row.get("RYE"), (int, float))
        ]
        return {
            "series": {
                "rye": rye_vals,
            },
            "stability_index": stability_index(rye_vals),
            "recovery_momentum": recovery_momentum(rye_vals),
            "trend_slope": regression_rye_slope(rye_vals),
            "rye_percentiles": rye_percentiles(rye_vals),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


@dataclass
class SkillDimensionScores:
    """Core skill dimensions tracked by MSIL on a 0 to 1 scale."""

    reasoning: float = 0.0
    literature_navigation: float = 0.0
    hypothesis_generation: float = 0.0
    planning_and_curriculum: float = 0.0
    stability_and_safety: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reasoning": self.reasoning,
            "literature_navigation": self.literature_navigation,
            "hypothesis_generation": self.hypothesis_generation,
            "planning_and_curriculum": self.planning_and_curriculum,
            "stability_and_safety": self.stability_and_safety,
        }


@dataclass
class DomainProfile:
    """Per domain MSIL view for a goal."""

    domain: str
    cycles: int
    median_rye: Optional[float]
    trend_slope: float
    stability_index: float
    recovery_momentum: float
    breakthrough_density: float
    msil_score: float
    avg_energy: Optional[float] = None
    avg_delta_r: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "cycles": self.cycles,
            "median_rye": self.median_rye,
            "trend_slope": self.trend_slope,
            "stability_index": self.stability_index,
            "recovery_momentum": self.recovery_momentum,
            "breakthrough_density": self.breakthrough_density,
            "msil_score": self.msil_score,
            "avg_energy": self.avg_energy,
            "avg_delta_r": self.avg_delta_r,
        }


@dataclass
class MSILSnapshot:
    """Full MSIL snapshot after observing a new cycle."""

    timestamp: str
    goal: str
    msil_score: float
    intelligence_stage: str
    total_cycles_for_goal: int
    recent_window: int
    skills: SkillDimensionScores
    per_domain_profiles: List[DomainProfile] = field(default_factory=list)
    actions: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "goal": self.goal,
            "msil_score": self.msil_score,
            "intelligence_stage": self.intelligence_stage,
            "total_cycles_for_goal": self.total_cycles_for_goal,
            "recent_window": self.recent_window,
            "skills": self.skills.to_dict(),
            "per_domain_profiles": [p.to_dict() for p in self.per_domain_profiles],
            "actions": self.actions,
            "extras": self.extras,
        }


class MetaSkillIntelligenceLayer:
    """Meta Skill Intelligence Layer for the Autonomous Research Agent.

    This class reads cycle history from the MemoryStore and produces
    multi dimensional intelligence views for a goal or full run.

    MemoryStore interface assumptions (soft)
    ----------------------------------------
    MSIL tries to use the following methods when present:

        - get_cycle_history_for_goal(goal, limit)
        - get_cycle_history(limit=None)
        - get_run_history(run_id)
        - get_rye_stats(goal=...)
        - get_replay_items_for_goal(goal)   (optional)
        - get_run_stats(run_id)             (optional)

    If methods are missing, MSIL falls back to more generic paths.

    Config options (all optional)
    -----------------------------
        msil_enabled: bool
            Hard toggle. Defaults to True.

        msil_window: int
            Sliding window size in cycles for recent intelligence view.
            Defaults to 200.

        msil_long_window: int
            Window size for long horizon trends when available.
            Defaults to 1000.

        msil_min_cycles: int
            Minimum number of cycles before MSIL score is considered
            stable. Defaults to 20.

        msil_breakthrough_high: float
            Threshold for breakthrough_score above which cycles are
            treated as high value discoveries. Defaults to 0.8.

        msil_breakthrough_mid: float
            Mid threshold for partial breakthrough. Defaults to 0.6.

        msil_use_replay_density: bool
            If true and replay buffer stats exist, fold replay
            density into hypothesis skills. Defaults to True.

        msil_frontier_threshold: float
            Threshold above which MSIL stage becomes frontier_candidate.
            Defaults to 0.9.
    """

    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        self.enabled: bool = bool(self.config.get("msil_enabled", True))
        self.window: int = int(self.config.get("msil_window", 200))
        self.long_window: int = int(self.config.get("msil_long_window", 1000))
        self.min_cycles: int = int(self.config.get("msil_min_cycles", 20))
        self.breakthrough_high: float = float(
            self.config.get("msil_breakthrough_high", 0.8)
        )
        self.breakthrough_mid: float = float(
            self.config.get("msil_breakthrough_mid", 0.6)
        )
        self.use_replay_density: bool = bool(
            self.config.get("msil_use_replay_density", True)
        )
        self.frontier_threshold: float = float(
            self.config.get("msil_frontier_threshold", 0.9)
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def observe_cycle(self, cycle_log: Dict[str, Any]) -> MSILSnapshot:
        """Observe a single cycle and return an updated MSIL snapshot.

        This is the main entry point for CoreAgent or engine workers.
        """
        goal = cycle_log.get("goal") or "unknown_goal"

        if not self.enabled:
            return self._build_disabled_snapshot(goal)

        history = self._get_history_for_goal(goal, limit=self.long_window)
        total_cycles = len(history)

        # If MemoryStore has not yet recorded this cycle, append it
        if history:
            last_idx = history[-1].get("cycle_index", history[-1].get("cycle"))
            current_idx = cycle_log.get("cycle_index", cycle_log.get("cycle"))
            if last_idx != current_idx:
                history.append(cycle_log)
                total_cycles = len(history)
        else:
            history.append(cycle_log)
            total_cycles = len(history)

        recent = history[-self.window :]
        skills = self._compute_skill_dimensions(recent)
        domain_profiles = self._compute_domain_profiles(recent)
        msil_score = self._aggregate_msil_score(skills, domain_profiles)

        # Optional goal level RYE stats from MemoryStore
        avg_rye_goal: Optional[float] = None
        min_rye_goal: Optional[float] = None
        max_rye_goal: Optional[float] = None
        try:
            if hasattr(self.memory_store, "get_rye_stats"):
                (
                    avg_rye_goal,
                    min_rye_goal,
                    max_rye_goal,
                    _count,
                ) = self.memory_store.get_rye_stats(  # type: ignore[attr-defined]
                    goal=goal
                )
        except Exception:
            avg_rye_goal = None
            min_rye_goal = None
            max_rye_goal = None

        intelligence_stage = self._infer_stage(msil_score, total_cycles)
        actions = self._recommend_actions(
            msil_score=msil_score,
            skills=skills,
            domain_profiles=domain_profiles,
            total_cycles=total_cycles,
            recent_history=recent,
        )

        # Optional replay density and curriculum hints
        replay_stats = self._get_replay_stats_for_goal(goal)

        # Small extras bundle for UI or logs
        extras: Dict[str, Any] = {
            "avg_rye_for_goal": avg_rye_goal,
            "min_rye_for_goal": min_rye_goal,
            "max_rye_for_goal": max_rye_goal,
            "replay_stats": replay_stats,
            "last_cycle_index": cycle_log.get("cycle_index", cycle_log.get("cycle")),
            "last_cycle_rye": cycle_log.get("RYE"),
            "last_cycle_breakthrough_score": (
                cycle_log.get("breakthrough") or {}
            ).get("breakthrough_score"),
        }

        snapshot = MSILSnapshot(
            timestamp=datetime.utcnow().isoformat() + "Z",
            goal=goal,
            msil_score=msil_score,
            intelligence_stage=intelligence_stage,
            total_cycles_for_goal=total_cycles,
            recent_window=len(recent),
            skills=skills,
            per_domain_profiles=domain_profiles,
            actions=actions,
            extras=extras,
        )
        return snapshot

    def summarise_run(
        self,
        goal: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Produce a full MSIL view for a goal or run id."""
        if not self.enabled:
            return {
                "enabled": False,
                "reason": "msil_enabled is False in config",
            }

        history = self._get_history(
            goal=goal, run_id=run_id, limit=limit or self.long_window
        )
        if not history:
            return {
                "enabled": True,
                "cycles": 0,
                "msil_score": 0.0,
                "intelligence_stage": "cold_start",
                "skills": SkillDimensionScores().to_dict(),
                "domain_profiles": [],
                "diagnostics": {},
                "run_stats": {},
            }

        skills = self._compute_skill_dimensions(history)
        domain_profiles = self._compute_domain_profiles(history)
        msil_score = self._aggregate_msil_score(skills, domain_profiles)

        diagnostics = self._build_run_diagnostics(history)
        intelligence_stage = self._infer_stage(msil_score, len(history))
        run_stats = self._get_run_stats(run_id)

        return {
            "enabled": True,
            "cycles": len(history),
            "msil_score": msil_score,
            "intelligence_stage": intelligence_stage,
            "skills": skills.to_dict(),
            "domain_profiles": [p.to_dict() for p in domain_profiles],
            "diagnostics": diagnostics,
            "run_stats": run_stats,
        }

    def compute_profile(
        self,
        cycle_history: List[Dict[str, Any]],
        goal: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a compact MSIL profile from an in-memory cycle history.

        This is a lightweight helper intended for:
            - Streamlit UI panels that already hold cycle history
            - Diagnostics code that wants a single profile dict

        Returns:
            dict with msil_score, stage, skills, domain_profiles, diagnostics
            or None if there is not enough data.
        """
        if not cycle_history:
            return None

        # Optional goal filter
        if goal is not None:
            history = [row for row in cycle_history if row.get("goal") == goal]
        else:
            history = list(cycle_history)

        if not history:
            return None

        skills = self._compute_skill_dimensions(history)
        domain_profiles = self._compute_domain_profiles(history)
        msil_score = self._aggregate_msil_score(skills, domain_profiles)
        diagnostics = self._build_run_diagnostics(history)
        intelligence_stage = self._infer_stage(msil_score, len(history))

        profile: Dict[str, Any] = {
            "cycles": len(history),
            "msil_score": msil_score,
            "intelligence_stage": intelligence_stage,
            "skills": skills.to_dict(),
            "domain_profiles": [p.to_dict() for p in domain_profiles],
            "diagnostics": diagnostics,
        }
        return profile

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def _get_history_for_goal(self, goal: str, limit: int) -> List[Dict[str, Any]]:
        """Get cycle history for a single goal with soft fallbacks."""
        try:
            if hasattr(self.memory_store, "get_cycle_history_for_goal"):
                rows = self.memory_store.get_cycle_history_for_goal(  # type: ignore[attr-defined]
                    goal, limit=limit
                )
                if isinstance(rows, list):
                    return rows
        except Exception:
            pass

        try:
            if hasattr(self.memory_store, "get_cycle_history"):
                full = self.memory_store.get_cycle_history(limit=limit)  # type: ignore[attr-defined]
            else:
                full = self.memory_store.get_cycle_history()  # type: ignore[attr-defined]
        except Exception:
            return []

        if isinstance(full, list):
            filtered = [r for r in full if r.get("goal") == goal]
            return filtered[-limit:]
        return []

    def _get_history(
        self,
        goal: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """More general history fetch with optional goal and run filters."""
        history: List[Dict[str, Any]] = []

        # Try dedicated run history if available
        if run_id is not None and hasattr(self.memory_store, "get_run_history"):
            try:
                history = self.memory_store.get_run_history(run_id)  # type: ignore[attr-defined]
            except Exception:
                history = []

        if not history:
            try:
                if hasattr(self.memory_store, "get_cycle_history"):
                    history = self.memory_store.get_cycle_history(limit=limit)  # type: ignore[attr-defined]
                else:
                    history = self.memory_store.get_cycle_history()  # type: ignore[attr-defined]
            except Exception:
                history = []

        if not isinstance(history, list):
            return []

        if goal:
            history = [row for row in history if row.get("goal") == goal]

        if run_id:
            history = [row for row in history if row.get("run_id") == run_id]

        if limit and len(history) > limit:
            history = history[-limit:]

        return history

    # ------------------------------------------------------------------
    # Optional replay and run stats
    # ------------------------------------------------------------------
    def _get_replay_stats_for_goal(self, goal: str) -> Dict[str, Any]:
        """Optional read of replay buffer stats for a goal."""
        stats: Dict[str, Any] = {
            "items_for_goal": None,
            "high_rye_items": None,
            "replay_density": None,
        }
        try:
            if hasattr(self.memory_store, "get_replay_items_for_goal"):
                items = self.memory_store.get_replay_items_for_goal(  # type: ignore[attr-defined]
                    goal
                )
            else:
                return stats
        except Exception:
            return stats

        if not isinstance(items, list) or not items:
            return stats

        total_items = len(items)
        high_rye = 0
        for it in items:
            rye_score = _safe_float(it.get("rye_score"), default=0.0)
            if rye_score >= self.breakthrough_mid:
                high_rye += 1

        replay_density = high_rye / max(1, total_items)
        stats["items_for_goal"] = total_items
        stats["high_rye_items"] = high_rye
        stats["replay_density"] = replay_density
        return stats

    def _get_run_stats(self, run_id: Optional[str]) -> Dict[str, Any]:
        """Optional wrapper around MemoryStore run level stats."""
        if run_id is None:
            return {}
        try:
            if hasattr(self.memory_store, "get_run_stats"):
                rs = self.memory_store.get_run_stats(run_id)  # type: ignore[attr-defined]
                if isinstance(rs, dict):
                    return rs
        except Exception:
            return {}
        return {}

    # ------------------------------------------------------------------
    # Skill and domain scoring
    # ------------------------------------------------------------------
    def _compute_skill_dimensions(
        self, history: List[Dict[str, Any]]
    ) -> SkillDimensionScores:
        """Compute core skill dimensions from cycle history."""
        if not history:
            return SkillDimensionScores()

        rye_vals: List[float] = []
        delta_vals: List[float] = []
        breakthrough_vals: List[float] = []
        citation_counts: List[int] = []
        hypothesis_counts: List[int] = []
        equilibrium_labels: List[str] = []
        oscillation_scores: List[float] = []
        open_questions_counts: List[int] = []
        contradictions_counts: List[int] = []
        todo_counts: List[int] = []
        energy_vals: List[float] = []

        # Tool usage aggregates
        web_calls_all: List[int] = []
        pubmed_calls_all: List[int] = []
        semantic_calls_all: List[int] = []

        # Domain flags aggregates
        longevity_issue_cycles = 0
        math_issue_cycles = 0

        for row in history:
            rye_vals.append(_safe_float(row.get("RYE")))
            delta_vals.append(
                _safe_float(
                    row.get("delta_r") or row.get("delta_R")
                )
            )
            energy_vals.append(
                _safe_float(
                    row.get("energy_e")
                    or row.get("energy_E")
                    or row.get("Energy")
                )
            )
            br = row.get("breakthrough") or {}
            breakthrough_vals.append(_safe_float(br.get("breakthrough_score")))

            citations_list = row.get("citations") or []
            citation_counts.append(len(citations_list))

            hyp_list = row.get("hypotheses") or []
            hypothesis_counts.append(len(hyp_list))

            eq = row.get("equilibrium") or {}
            equilibrium_labels.append(str(eq.get("equilibrium_label") or "unknown"))
            oscillation_scores.append(
                _safe_float(eq.get("oscillation_score"), default=0.0)
            )

            ms = row.get("meta_signals") or {}
            open_questions_counts.append(_safe_int(ms.get("open_questions")))
            contradictions_counts.append(_safe_int(ms.get("contradictions")))
            todo_counts.append(_safe_int(ms.get("todo_items")))

            stats_block = row.get("stats") or {}
            web_calls_all.append(_safe_int(stats_block.get("web_calls")))
            pubmed_calls_all.append(_safe_int(stats_block.get("pubmed_calls")))
            semantic_calls_all.append(_safe_int(stats_block.get("semantic_calls")))

            # Domain issue flags (longevity and math specific)
            dom_flags_before = row.get("domain_issue_flags_before") or {}
            dom_flags_after = row.get("domain_issue_flags_after") or {}
            if any(
                dom_flags_before.get(k) or dom_flags_after.get(k)
                for k in ["missing_biomarkers", "missing_mechanisms"]
            ):
                longevity_issue_cycles += 1
            if any(
                dom_flags_before.get(k) or dom_flags_after.get(k)
                for k in ["missing_formalism", "missing_connections"]
            ):
                math_issue_cycles += 1

        # Reasoning: based on RYE level, trend slope, and stability
        rye_series = [v for v in rye_vals if v > 0]
        if rye_series:
            rye_med = statistics.median(rye_series)
            rye_trend = regression_rye_slope(rye_series)
            rye_stability = stability_index(rye_series)
        else:
            rye_med = 0.0
            rye_trend = 0.0
            rye_stability = 0.0

        reasoning_score = 0.0
        reasoning_score += 0.5 * max(0.0, min(1.0, rye_med))
        reasoning_score += 0.3 * max(0.0, min(1.0, rye_trend + 0.5))
        reasoning_score += 0.2 * rye_stability

        # Literature navigation: citations and science heavy tool usage
        avg_citations = statistics.mean(citation_counts) if citation_counts else 0.0
        rich_citation_cycles = sum(1 for c in citation_counts if c >= 5)
        richness_ratio = rich_citation_cycles / max(1, len(citation_counts))

        avg_web_calls = statistics.mean(web_calls_all) if web_calls_all else 0.0
        avg_pubmed_calls = (
            statistics.mean(pubmed_calls_all) if pubmed_calls_all else 0.0
        )
        avg_sem_calls = (
            statistics.mean(semantic_calls_all) if semantic_calls_all else 0.0
        )

        sci_calls = avg_pubmed_calls + avg_sem_calls
        sci_factor = min(1.0, sci_calls / 4.0)
        web_factor = min(1.0, avg_web_calls / 6.0)

        literature_nav = 0.4 * min(1.0, avg_citations / 10.0)
        literature_nav += 0.3 * richness_ratio
        literature_nav += 0.2 * sci_factor
        literature_nav += 0.1 * web_factor

        # Hypothesis generation: count, focus, breakthroughs, replay density if available
        avg_hyp = statistics.mean(hypothesis_counts) if hypothesis_counts else 0.0
        focused_cycles = sum(1 for h in hypothesis_counts if 1 <= h <= 5)
        focused_ratio = focused_cycles / max(1, len(hypothesis_counts))
        avg_breakthrough = (
            statistics.mean(breakthrough_vals) if breakthrough_vals else 0.0
        )

        hypothesis_skill = 0.35 * min(1.0, avg_hyp / 10.0)
        hypothesis_skill += 0.35 * focused_ratio
        hypothesis_skill += 0.30 * avg_breakthrough

        # Optional replay density hook if available in MemoryStore
        replay_boost = 0.0
        if self.use_replay_density and history:
            some_goal = history[-1].get("goal")
            if some_goal:
                rs = self._get_replay_stats_for_goal(str(some_goal))
                replay_density = _safe_float(rs.get("replay_density"), default=0.0)
                replay_boost = 0.2 * min(1.0, replay_density / 0.5)
        hypothesis_skill = max(0.0, min(1.0, hypothesis_skill + replay_boost))

        # Planning and curriculum: stages, roles, curriculum tags
        stages = [str(row.get("stage") or "idea") for row in history]
        unique_stages = len({s for s in stages if s})
        roles = [str(row.get("role") or "agent") for row in history]
        unique_roles = len(set(roles))

        curriculum_tags: List[str] = []
        for row in history:
            tags = row.get("tags") or []
            for t in tags:
                if isinstance(t, str) and t.startswith("curriculum:"):
                    curriculum_tags.append(t)

        unique_curriculum_phases = len(set(curriculum_tags)) if curriculum_tags else 0
        stage_balance = min(1.0, unique_stages / 3.0)
        role_balance = min(1.0, unique_roles / 6.0)
        curriculum_balance = min(1.0, unique_curriculum_phases / 4.0)

        planning_skill = 0.4 * stage_balance
        planning_skill += 0.4 * role_balance
        planning_skill += 0.2 * curriculum_balance

        # Stability and safety: equilibrium labels, oscillation, open questions and contradictions
        stable_cycles = sum(
            1
            for label in equilibrium_labels
            if label in {"high_equilibrium", "plateau_equilibrium"}
        )
        unstable_cycles = sum(
            1
            for label in equilibrium_labels
            if label in {"oscillating", "low_efficiency"}
        )
        total = max(1, len(equilibrium_labels))
        stability_fraction = stable_cycles / total
        instability_fraction = unstable_cycles / total
        avg_osc = statistics.mean(oscillation_scores) if oscillation_scores else 0.0

        avg_open_questions = (
            statistics.mean(open_questions_counts) if open_questions_counts else 0.0
        )
        avg_todos = statistics.mean(todo_counts) if todo_counts else 0.0
        avg_contradictions = (
            statistics.mean(contradictions_counts) if contradictions_counts else 0.0
        )

        # Penalize if contradictions never drop
        unresolved_pressure = min(
            1.0, (avg_open_questions + avg_todos + avg_contradictions) / 10.0
        )
        longevity_penalty = 0.0
        math_penalty = 0.0

        if history:
            longevity_ratio = longevity_issue_cycles / max(1, len(history))
            math_ratio = math_issue_cycles / max(1, len(history))
            longevity_penalty = min(0.2, longevity_ratio * 0.3)
            math_penalty = min(0.2, math_ratio * 0.3)

        stability_skill = 0.6 * stability_fraction
        stability_skill += 0.2 * max(0.0, 1.0 - avg_osc)
        stability_skill += 0.2 * max(0.0, 1.0 - instability_fraction)
        stability_skill -= 0.15 * unresolved_pressure
        stability_skill -= longevity_penalty
        stability_skill -= math_penalty
        stability_skill = max(0.0, min(1.0, stability_skill))

        return SkillDimensionScores(
            reasoning=max(0.0, min(1.0, reasoning_score)),
            literature_navigation=max(0.0, min(1.0, literature_nav)),
            hypothesis_generation=hypothesis_skill,
            planning_and_curriculum=max(0.0, min(1.0, planning_skill)),
            stability_and_safety=stability_skill,
        )

    def _compute_domain_profiles(
        self, history: List[Dict[str, Any]]
    ) -> List[DomainProfile]:
        """Compute per domain MSIL profiles."""
        if not history:
            return []

        by_domain: Dict[str, List[Dict[str, Any]]] = {}
        for row in history:
            dom = (row.get("domain") or "general").lower()
            by_domain.setdefault(dom, []).append(row)

        profiles: List[DomainProfile] = []

        for dom, rows in by_domain.items():
            if not rows:
                continue

            rye_vals: List[float] = []
            breakthrough_vals: List[float] = []
            energy_vals: List[float] = []
            delta_vals: List[float] = []

            for r in rows:
                rye_vals.append(_safe_float(r.get("RYE")))
                br = r.get("breakthrough") or {}
                breakthrough_vals.append(_safe_float(br.get("breakthrough_score")))
                energy_vals.append(
                    _safe_float(
                        r.get("energy_e")
                        or r.get("energy_E")
                        or r.get("Energy")
                    )
                )
                delta_vals.append(
                    _safe_float(
                        r.get("delta_r") or r.get("delta_R")
                    )
                )

            rye_series = [v for v in rye_vals if v > 0]
            median_rye = statistics.median(rye_series) if rye_series else None
            trend_slope = regression_rye_slope(rye_series) if rye_series else 0.0
            stab = stability_index(rye_series) if rye_series else 0.0
            rec = recovery_momentum(rye_series) if rye_series else 0.0

            high_br_cycles = sum(
                1 for v in breakthrough_vals if v >= self.breakthrough_high
            )
            breakthrough_density = high_br_cycles / max(1, len(rows))

            # Combine into a small domain specific MSIL score
            score = 0.35 * (median_rye or 0.0)
            score += 0.25 * stab
            score += 0.20 * max(0.0, min(1.0, trend_slope + 0.5))
            score += 0.20 * breakthrough_density
            score = max(0.0, min(1.0, score))

            avg_energy = statistics.mean(energy_vals) if energy_vals else None
            avg_delta_r = statistics.mean(delta_vals) if delta_vals else None

            profiles.append(
                DomainProfile(
                    domain=dom,
                    cycles=len(rows),
                    median_rye=median_rye,
                    trend_slope=trend_slope,
                    stability_index=stab,
                    recovery_momentum=rec,
                    breakthrough_density=breakthrough_density,
                    msil_score=score,
                    avg_energy=avg_energy,
                    avg_delta_r=avg_delta_r,
                )
            )

        profiles.sort(key=lambda p: p.msil_score, reverse=True)
        return profiles

    def _aggregate_msil_score(
        self,
        skills: SkillDimensionScores,
        domain_profiles: List[DomainProfile],
    ) -> float:
        """Combine skills and domain scores into a single MSIL score."""
        skill_vals = list(skills.to_dict().values())
        skill_mean = statistics.mean(skill_vals) if skill_vals else 0.0

        if domain_profiles:
            dom_scores = [p.msil_score for p in domain_profiles]
            dom_mean = statistics.mean(dom_scores)
            dom_top = max(dom_scores)
        else:
            dom_mean = 0.0
            dom_top = 0.0

        score = 0.55 * skill_mean
        score += 0.30 * dom_mean
        score += 0.15 * dom_top

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Diagnostics and stages
    # ------------------------------------------------------------------
    def _build_run_diagnostics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Wrap build_run_diagnostics with safe fallbacks plus tool stats."""
        try:
            diag = build_run_diagnostics(history)
            if isinstance(diag, dict):
                base_diag = diag
            else:
                base_diag = {}
        except Exception:
            rye_vals = [
                _safe_float(row.get("RYE"))
                for row in history
                if isinstance(row.get("RYE"), (int, float))
            ]
            base_diag = {
                "series": {"rye": rye_vals},
                "stability_index": stability_index(rye_vals),
                "recovery_momentum": recovery_momentum(rye_vals),
                "trend_slope": regression_rye_slope(rye_vals),
                "rye_percentiles": rye_percentiles(rye_vals),
            }

        # Aggregate optional tool stats and energy diagnostics
        energy_vals: List[float] = []
        web_calls_all: List[int] = []
        pubmed_calls_all: List[int] = []
        semantic_calls_all: List[int] = []
        tokens_all: List[int] = []

        for row in history:
            energy_vals.append(
                _safe_float(
                    row.get("energy_e")
                    or row.get("energy_E")
                    or row.get("Energy")
                )
            )
            stats_block = row.get("stats") or {}
            web_calls_all.append(_safe_int(stats_block.get("web_calls")))
            pubmed_calls_all.append(_safe_int(stats_block.get("pubmed_calls")))
            semantic_calls_all.append(_safe_int(stats_block.get("semantic_calls")))
            tu = row.get("tool_usage") or {}
            tokens_all.append(_safe_int(tu.get("approx_tokens")))

        extra_diag: Dict[str, Any] = {}
        if energy_vals:
            extra_diag["avg_energy_per_cycle"] = statistics.mean(energy_vals)
        if web_calls_all:
            extra_diag["avg_web_calls_per_cycle"] = statistics.mean(web_calls_all)
        if pubmed_calls_all:
            extra_diag["avg_pubmed_calls_per_cycle"] = statistics.mean(pubmed_calls_all)
        if semantic_calls_all:
            extra_diag[
                "avg_semantic_calls_per_cycle"
            ] = statistics.mean(semantic_calls_all)
        if tokens_all:
            extra_diag["avg_tokens_per_cycle"] = statistics.mean(tokens_all)

        base_diag["energy_and_tools"] = extra_diag
        return base_diag

    def _infer_stage(self, msil_score: float, cycles: int) -> str:
        """Map MSIL score and cycles to an interpretive intelligence stage."""
        if cycles < max(5, self.min_cycles // 2):
            return "cold_start"

        if cycles < self.min_cycles:
            if msil_score < 0.3:
                return "warmup_low"
            elif msil_score < 0.6:
                return "warmup_mid"
            else:
                return "warmup_high"

        if msil_score < 0.3:
            return "learning_basic"
        if msil_score < 0.55:
            return "learning_capable"
        if msil_score < 0.75:
            return "specialised_expert"
        if msil_score < self.frontier_threshold:
            return "broad_expert"
        return "frontier_candidate"

    # ------------------------------------------------------------------
    # Action and curriculum suggestions
    # ------------------------------------------------------------------
    def _recommend_actions(
        self,
        msil_score: float,
        skills: SkillDimensionScores,
        domain_profiles: List[DomainProfile],
        total_cycles: int,
        recent_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Produce advisory actions for swarm, curriculum, and config."""
        actions: Dict[str, Any] = {
            "priority": [],
            "swarm_config": {},
            "curriculum": {},
            "monitoring": {},
        }

        skill_dict = skills.to_dict()

        # Priority hints based on weakest dimensions
        weakest_skill_name = min(skill_dict, key=skill_dict.get)
        weakest_skill_value = skill_dict[weakest_skill_name]

        if weakest_skill_value < 0.45:
            if weakest_skill_name == "stability_and_safety":
                actions["priority"].append(
                    "Stability and safety appear to be the weakest dimension. Increase maintenance_mode frequency, add more critic cycles, and tighten source controls for high risk domains."
                )
            elif weakest_skill_name == "literature_navigation":
                actions["priority"].append(
                    "Literature navigation is the weakest dimension. Enable pubmed and semantic sources on key goals and run citation strengthening cycles."
                )
            elif weakest_skill_name == "hypothesis_generation":
                actions["priority"].append(
                    "Hypothesis generation is the weakest dimension. Add idea stage runs with explicit hypothesis prompts and ensure replay logging is enabled."
                )
            elif weakest_skill_name == "planning_and_curriculum":
                actions["priority"].append(
                    "Planning and curriculum look weak. Use explicit stages (idea and verify), add planner and integrator roles, and tag runs with curriculum phases."
                )
            else:
                actions["priority"].append(
                    "Reasoning quality is the weakest dimension. Focus on fewer goals, longer runs per goal, and monitor RYE trend slope carefully."
                )
        else:
            actions["priority"].append(
                "Skills appear balanced. Focus on sustained long runs and monitoring MSIL and breakthrough density over weeks."
            )

        # Swarm config suggestions
        best_domain = domain_profiles[0] if domain_profiles else None
        if best_domain and best_domain.domain in {"longevity", "math"} and msil_score >= 0.6:
            actions["swarm_config"]["suggested_mode"] = "8_to_16_agent_swarm"
        elif msil_score >= 0.4:
            actions["swarm_config"]["suggested_mode"] = "4_to_8_agent_swarm"
        else:
            actions["swarm_config"]["suggested_mode"] = "single_or_small_swarm"

        actions["swarm_config"]["role_mix_hint"] = (
            "Use at least one critic, one planner, one explorer, and one synthesizer for high stakes swarms."
        )

        # Curriculum suggestions
        if best_domain:
            actions["curriculum"]["primary_domain"] = best_domain.domain
            actions["curriculum"]["hint"] = (
                "Use two stage curriculum. Idea runs focus on exploration, verify runs focus on top hypotheses and replay items."
            )
        else:
            actions["curriculum"]["primary_domain"] = "general"
            actions["curriculum"]["hint"] = (
                "Lock onto one clear goal and domain until RYE stabilises before branching outward."
            )

        # Optional hallmarks and subgoals summary
        hallmarks: List[str] = []
        subgoals: List[str] = []
        for row in recent_history:
            if row.get("hallmark"):
                hallmarks.append(str(row["hallmark"]))
            if row.get("subgoal"):
                subgoals.append(str(row["subgoal"]))
        if hallmarks:
            actions["curriculum"]["observed_hallmarks"] = sorted(set(hallmarks))
        if subgoals:
            actions["curriculum"]["observed_subgoals"] = sorted(set(subgoals))

        # Monitoring suggestions
        if total_cycles < 100:
            actions["monitoring"]["note"] = (
                "MSIL metrics are early. Re evaluate after passing 100 cycles on the same goal."
            )
        else:
            actions["monitoring"]["note"] = (
                "Track MSIL score, in stability_index, and breakthrough_density weekly for long runs."
            )

        actions["monitoring"]["recommended_metrics"] = [
            "msil_score",
            "stability_index",
            "trend_slope",
            "breakthrough_density",
        ]

        # Simple risk flag from recent contradictions and oscillation
        recent_eq = [row.get("equilibrium") or {} for row in recent_history]
        recent_osc = [
            _safe_float(eq.get("oscillation_score"), default=0.0) for eq in recent_eq
        ]
        recent_labels = [
            str(eq.get("equilibrium_label") or "unknown") for eq in recent_eq
        ]
        osc_high = any(v >= 0.7 for v in recent_osc)
        oscillating_recent = any(lab == "oscillating" for lab in recent_labels)

        if osc_high or oscillating_recent:
            actions["monitoring"]["risk_flag"] = "oscillation_risk"
            actions["monitoring"]["risk_hint"] = (
                "Recent RYE behaviour looks oscillatory. Consider lowering swarm size and increasing maintenance_mode."
            )

        return actions

    # ------------------------------------------------------------------
    # Disabled snapshot helper
    # ------------------------------------------------------------------
    def _build_disabled_snapshot(self, goal: str) -> MSILSnapshot:
        """Return a neutral snapshot when msil_enabled is false."""
        return MSILSnapshot(
            timestamp=datetime.utcnow().isoformat() + "Z",
            goal=goal,
            msil_score=0.0,
            intelligence_stage="disabled",
            total_cycles_for_goal=0,
            recent_window=0,
            skills=SkillDimensionScores(),
            per_domain_profiles=[],
            actions={
                "priority": [
                    "MSIL is disabled in config. Set msil_enabled to True to activate."
                ],
                "swarm_config": {},
                "curriculum": {},
                "monitoring": {},
            },
            extras={},
        )


# ----------------------------------------------------------------------
# Lightweight adapter so the UI can call analyze_run(history, ...)
# without needing a real MemoryStore. This helps fix the
# "MSIL module not detected or no MSIL profile available" state
# when only cycle history is available in app_streamlit.
# ----------------------------------------------------------------------


class _HistoryBackedMemoryStore:
    """Minimal MemoryStore like wrapper around an in memory history list.

    It supports the subset of methods that MSIL expects and is only used
    when callers have a plain history list instead of a real store.
    """

    def __init__(self, history: List[Dict[str, Any]]) -> None:
        # Make a shallow copy to avoid accidental mutation from outside
        self._history: List[Dict[str, Any]] = list(history or [])

    def get_cycle_history_for_goal(
        self, goal: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        rows = [row for row in self._history if row.get("goal") == goal]
        if limit is not None and len(rows) > limit:
            rows = rows[-limit:]
        return rows

    def get_cycle_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        rows = list(self._history)
        if limit is not None and len(rows) > limit:
            rows = rows[-limit:]
        return rows


def analyze_run(
    history: List[Dict[str, Any]],
    goal: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience helper for UIs.

    Args:
        history:
            List of cycle logs, as usually returned by MemoryStore.
        goal:
            Optional goal filter. If None, uses the goal from the last cycle.
        config:
            Optional MSIL config dict.

    Returns:
        A summarise_run style dict with MSIL metrics.
    """
    if not history:
        return {
            "enabled": True,
            "cycles": 0,
            "msil_score": 0.0,
            "intelligence_stage": "cold_start",
            "skills": SkillDimensionScores().to_dict(),
            "domain_profiles": [],
            "diagnostics": {},
            "run_stats": {},
        }

    if goal is None:
        goal = str(history[-1].get("goal") or "unknown_goal")

    store = _HistoryBackedMemoryStore(history)
    msil = MetaSkillIntelligenceLayer(store, config=config or {})
    return msil.summarise_run(goal=goal, limit=len(history))


# Backward compatible alias in case some code calls analyze_history
analyze_history = analyze_run
