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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
import math
import statistics
import re


__all__ = [
    "SkillDimensionScores",
    "DomainProfile",
    "MSILSnapshot",
    "MetaSkillIntelligenceLayer",
    "analyze_run",
    "analyze_history",
]

__version__ = "2025.12.18"

HistoryRow = Dict[str, Any]


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
        w = max(1, int(window))
        out: List[float] = []
        for i in range(len(values)):
            start = max(0, i - w + 1)
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
        ratio = stdev / (abs(mean_val) + 1e-9)
        return _clip01(1.0 - ratio)

    def recovery_momentum(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        base = abs(values[0]) + 1e-9
        return _clip01((values[-1] - values[0]) / base)

    def regression_rye_slope(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = (n - 1) / 2.0
        mean_y = sum(values) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        den = sum((x - mean_x) ** 2 for x in xs) or 1.0
        return float(num / den)

    def rye_percentiles(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"p10": None, "p50": None, "p90": None}
        sorted_vals = sorted(values)

        def pct(p: float) -> float:
            idx = int(round(p * (len(sorted_vals) - 1)))
            idx = max(0, min(len(sorted_vals) - 1, idx))
            return float(sorted_vals[idx])

        return {"p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90)}

    def build_run_diagnostics(history: List[HistoryRow]) -> Dict[str, Any]:
        rye_vals = [
            _safe_float(row.get("RYE", row.get("rye")), default=math.nan)
            for row in history
        ]
        rye_vals = [
            v
            for v in rye_vals
            if isinstance(v, (int, float)) and not math.isnan(v)
        ]
        return {
            "series": {"rye": rye_vals},
            "stability_index": stability_index(rye_vals),
            "recovery_momentum": recovery_momentum(rye_vals),
            "trend_slope": regression_rye_slope(rye_vals),
            "rye_percentiles": rye_percentiles(rye_vals),
        }


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Float coercion that tolerates numeric strings and None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        v = float(value)
    except Exception:
        return default
    if math.isnan(v):
        return default
    return v


def _safe_int(value: Any, default: int = 0) -> int:
    """Int coercion that tolerates numeric strings and None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))  # handles "3.0"
    except Exception:
        return default


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return []


def _norm_domain(value: Any) -> str:
    s = str(value or "general").strip().lower()
    return s or "general"



def _extract_domains_from_row(row: 'HistoryRow') -> List[str]:
    """Extract one or more normalized domain labels from a cycle/history row.

    MSIL historically used a single ``row['domain']`` value, which often stays
    constant (e.g. 'general') even when the underlying hypotheses mix multiple
    domains. To better reflect cross-domain synthesis, we also look for
    multi-domain fields (lists/strings) commonly produced by cycle logs or
    discovery payloads.
    """
    if not isinstance(row, dict):
        return ['general']

    candidates: List[str] = []

    def _ingest(val: Any) -> None:
        if val is None:
            return
        if isinstance(val, str):
            # Split on common separators for compact domain strings
            parts = [p.strip() for p in re.split(r"[;,|]", val) if p.strip()]
            candidates.extend(parts or [val.strip()])
            return
        if isinstance(val, (list, tuple, set)):
            for x in val:
                if x is None:
                    continue
                candidates.append(str(x))

    # Common keys used by engines/promoters
    for key in (
        'domains_involved',
        'domains',
        'domain_lenses',
        'domain_tags',
        'cross_domain',
        'tags',
    ):
        _ingest(row.get(key))

    # Nested discovery payloads may carry domains
    disc = row.get('discovery')
    if isinstance(disc, dict):
        for key in ('domains', 'domains_involved', 'domain_lenses', 'cross_domain', 'tags'):
            _ingest(disc.get(key))

    # Fallback to the single-domain field
    if not candidates:
        candidates = [str(row.get('domain') or 'general')]

    # Normalize and unique-preserve
    out: List[str] = []
    for d in candidates:
        nd = _norm_domain(d)
        if nd and nd not in out:
            out.append(nd)
    return out or ['general']
def _get_cycle_index(row: HistoryRow) -> Optional[int]:
    for k in ("cycle_index", "cycle", "index", "i"):
        if k in row and row.get(k) is not None:
            v = _safe_int(row.get(k), default=-1)
            return v if v >= 0 else None
    return None


def _sorted_history(history: Sequence[HistoryRow]) -> List[HistoryRow]:
    """Sort by cycle_index when available; otherwise preserve order."""
    rows = list(history or [])
    if not rows:
        return []
    idxs = [_get_cycle_index(r) for r in rows]
    if sum(1 for i in idxs if i is not None) >= max(2, len(rows) // 4):
        try:
            return sorted(
                rows,
                key=lambda r: (_get_cycle_index(r) is None, _get_cycle_index(r) or 0),
            )
        except Exception:
            return rows
    return rows


def _get_rye(row: HistoryRow) -> float:
    return _safe_float(
        row.get("RYE", row.get("rye", row.get("rye_score"))), default=0.0
    )


def _get_delta_r(row: HistoryRow) -> float:
    return _safe_float(row.get("delta_r", row.get("delta_R")), default=0.0)


def _get_energy(row: HistoryRow) -> float:
    return _safe_float(
        row.get("energy_e", row.get("energy_E", row.get("Energy", row.get("energy")))),
        default=0.0,
    )


def _get_breakthrough_score(row: HistoryRow) -> float:
    if "breakthrough_score" in row:
        return _safe_float(row.get("breakthrough_score"), default=0.0)
    br = _as_dict(row.get("breakthrough"))
    return _safe_float(br.get("breakthrough_score"), default=0.0)


def _len_citations(row: HistoryRow) -> int:
    c = row.get("citations")
    if isinstance(c, list):
        return len(c)
    if isinstance(c, dict):
        return len(c)
    return 0


def _len_hypotheses(row: HistoryRow) -> int:
    h = row.get("hypotheses")
    if isinstance(h, list):
        return len(h)
    if isinstance(h, dict):
        return len(h)
    return 0


def _eq_label(row: HistoryRow) -> str:
    eq = _as_dict(row.get("equilibrium"))
    lab = eq.get("equilibrium_label", row.get("equilibrium_label", "unknown"))
    return str(lab or "unknown")


def _oscillation(row: HistoryRow) -> float:
    eq = _as_dict(row.get("equilibrium"))
    return _safe_float(
        eq.get("oscillation_score", row.get("oscillation_score")), default=0.0
    )


def _meta_signals(row: HistoryRow) -> Dict[str, Any]:
    return _as_dict(row.get("meta_signals", row.get("meta", {})))


def _count_like(value: Any) -> int:
    """If value is a list/dict, return len; if scalar numeric, return int; else 0."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value)
    return _safe_int(value, default=0)


def _stats_block(row: HistoryRow) -> Dict[str, Any]:
    stats = _as_dict(row.get("stats"))
    if stats:
        return stats
    return _as_dict(row.get("tool_usage"))


def _mean_or_none(values: List[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None
    try:
        return float(statistics.mean(vals))
    except Exception:
        return None


def _median_or_none(values: List[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None
    try:
        return float(statistics.median(vals))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


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
            "reasoning": _clip01(self.reasoning),
            "literature_navigation": _clip01(self.literature_navigation),
            "hypothesis_generation": _clip01(self.hypothesis_generation),
            "planning_and_curriculum": _clip01(self.planning_and_curriculum),
            "stability_and_safety": _clip01(self.stability_and_safety),
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
            "cycles": int(self.cycles),
            "median_rye": self.median_rye,
            "trend_slope": float(self.trend_slope),
            "stability_index": _clip01(self.stability_index),
            "recovery_momentum": _clip01(self.recovery_momentum),
            "breakthrough_density": _clip01(self.breakthrough_density),
            "msil_score": _clip01(self.msil_score),
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
            "msil_score": _clip01(self.msil_score),
            "intelligence_stage": self.intelligence_stage,
            "total_cycles_for_goal": int(self.total_cycles_for_goal),
            "recent_window": int(self.recent_window),
            "skills": self.skills.to_dict(),
            "per_domain_profiles": [p.to_dict() for p in self.per_domain_profiles],
            "actions": self.actions,
            "extras": self.extras,
            "version": __version__,
        }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class MetaSkillIntelligenceLayer:
    """Meta Skill Intelligence Layer for the Autonomous Research Agent."""

    def __init__(self, memory_store: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.config: Dict[str, Any] = dict(config or {})

        self.enabled: bool = bool(self.config.get("msil_enabled", True))
        self.window: int = max(1, _safe_int(self.config.get("msil_window", 200), default=200))
        self.long_window: int = max(
            self.window, _safe_int(self.config.get("msil_long_window", 1000), default=1000)
        )
        self.min_cycles: int = max(1, _safe_int(self.config.get("msil_min_cycles", 20), default=20))

        self.breakthrough_high: float = _clip01(self.config.get("msil_breakthrough_high", 0.8))
        self.breakthrough_mid: float = _clip01(self.config.get("msil_breakthrough_mid", 0.6))

        self.use_replay_density: bool = bool(self.config.get("msil_use_replay_density", True))
        self.frontier_threshold: float = _clip01(self.config.get("msil_frontier_threshold", 0.9))

        # Optional: allow callers to override which equilibrium labels are "stable"/"unstable".
        self._stable_labels = set(
            self.config.get("msil_stable_labels") or ["high_equilibrium", "plateau_equilibrium"]
        )
        self._unstable_labels = set(
            self.config.get("msil_unstable_labels") or ["oscillating", "low_efficiency"]
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def observe_cycle(self, cycle_log: HistoryRow) -> MSILSnapshot:
        """Observe a single cycle and return an updated MSIL snapshot."""
        goal = str(cycle_log.get("goal") or "unknown_goal")

        if not self.enabled:
            return self._build_disabled_snapshot(goal)

        run_id = cycle_log.get("run_id")
        history = self._get_history_for_goal(goal, run_id=str(run_id) if run_id else None, limit=self.long_window)
        history = _sorted_history(history)

        # Ensure the current cycle is present (avoid duplicates by cycle_index if possible).
        current_idx = _get_cycle_index(cycle_log)
        if not history:
            history = [cycle_log]
        else:
            if current_idx is not None:
                replaced = False
                for i, row in enumerate(history):
                    if _get_cycle_index(row) == current_idx:
                        history[i] = cycle_log
                        replaced = True
                        break
                if not replaced:
                    history.append(cycle_log)
            else:
                if history[-1] is not cycle_log:
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
                res = self.memory_store.get_rye_stats(goal=goal)  # type: ignore[attr-defined]
                if isinstance(res, tuple) and len(res) >= 3:
                    avg_rye_goal = _safe_float(res[0], default=0.0)
                    min_rye_goal = _safe_float(res[1], default=0.0)
                    max_rye_goal = _safe_float(res[2], default=0.0)
        except Exception:
            pass

        intelligence_stage = self._infer_stage(msil_score, total_cycles)
        actions = self._recommend_actions(
            msil_score=msil_score,
            skills=skills,
            domain_profiles=domain_profiles,
            total_cycles=total_cycles,
            recent_history=recent,
        )

        replay_stats = self._get_replay_stats_for_goal(goal)

        extras: Dict[str, Any] = {
            "version": __version__,
            "avg_rye_for_goal": avg_rye_goal,
            "min_rye_for_goal": min_rye_goal,
            "max_rye_for_goal": max_rye_goal,
            "replay_stats": replay_stats,
            "last_cycle_index": current_idx,
            "last_cycle_rye": _get_rye(cycle_log),
            "last_cycle_breakthrough_score": _get_breakthrough_score(cycle_log),
        }

        return MSILSnapshot(
            timestamp=_now_utc_iso(),
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
                "version": __version__,
            }

        effective_limit = max(1, int(limit or self.long_window))
        history = self._get_history(goal=goal, run_id=run_id, limit=effective_limit)
        history = _sorted_history(history)

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
                "version": __version__,
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
            "version": __version__,
        }

    def compute_profile(
        self,
        cycle_history: List[HistoryRow],
        goal: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a compact MSIL profile from an in-memory cycle history."""
        if not cycle_history:
            return None

        if goal is not None:
            history = [row for row in cycle_history if str(row.get("goal") or "") == str(goal)]
        else:
            history = list(cycle_history)

        history = _sorted_history(history)
        if not history:
            return None

        skills = self._compute_skill_dimensions(history)
        domain_profiles = self._compute_domain_profiles(history)
        msil_score = self._aggregate_msil_score(skills, domain_profiles)
        diagnostics = self._build_run_diagnostics(history)
        intelligence_stage = self._infer_stage(msil_score, len(history))

        return {
            "cycles": len(history),
            "msil_score": msil_score,
            "intelligence_stage": intelligence_stage,
            "skills": skills.to_dict(),
            "domain_profiles": [p.to_dict() for p in domain_profiles],
            "diagnostics": diagnostics,
            "version": __version__,
        }

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def _get_history_for_goal(self, goal: str, limit: int, run_id: Optional[str] = None) -> List[HistoryRow]:
        """Get cycle history for a single goal with soft fallbacks.

        If run_id is provided, history is filtered so repeated runs with the same
        goal do not blend together (this can otherwise inflate cycle counts and
        smear MSIL/MSIL-trend signals across runs).
        """
        goal_s = str(goal or "unknown_goal")
        lim = max(1, int(limit))
        history: List[HistoryRow] = []

        # Prefer the goal-specific accessor when available (best signal for TGRM)
        try:
            if hasattr(self.memory_store, "get_cycle_history_for_goal"):
                try:
                    if run_id is not None:
                        rows = self.memory_store.get_cycle_history_for_goal(goal_s, limit=lim, run_id=run_id)  # type: ignore[attr-defined]
                    else:
                        rows = self.memory_store.get_cycle_history_for_goal(goal_s, limit=lim)  # type: ignore[attr-defined]
                except TypeError:
                    # Older MemoryStore implementations may not accept run_id
                    rows = self.memory_store.get_cycle_history_for_goal(goal_s, limit=lim)  # type: ignore[attr-defined]
                if isinstance(rows, list):
                    history = rows
        except Exception:
            pass

        # Fallback: general history (optionally filtered by run_id)
        if not history:
            try:
                if hasattr(self.memory_store, "get_cycle_history"):
                    try:
                        history = self.memory_store.get_cycle_history(limit=lim, run_id=run_id)  # type: ignore[attr-defined]
                    except TypeError:
                        history = self.memory_store.get_cycle_history(limit=lim)  # type: ignore[attr-defined]
                else:
                    history = self.memory_store.get_cycle_history()  # type: ignore[attr-defined]
            except Exception:
                return []

        if not isinstance(history, list):
            return []

        filtered = [r for r in history if isinstance(r, dict) and str(r.get("goal") or "") == goal_s]
        if run_id is not None:
            rid = str(run_id)
            filtered = [r for r in filtered if str(r.get("run_id") or "") == rid]

        return filtered[-lim:]
    def _get_history(
        self,
        goal: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[HistoryRow]:
        """More general history fetch with optional goal and run filters."""
        lim = max(1, int(limit))
        history: List[HistoryRow] = []

        if run_id is not None and hasattr(self.memory_store, "get_run_history"):
            try:
                rows = self.memory_store.get_run_history(run_id)  # type: ignore[attr-defined]
                if isinstance(rows, list):
                    history = rows
            except Exception:
                history = []

        if not history:
            try:
                if hasattr(self.memory_store, "get_cycle_history"):
                    history = self.memory_store.get_cycle_history(limit=lim)  # type: ignore[attr-defined]
                else:
                    history = self.memory_store.get_cycle_history()  # type: ignore[attr-defined]
            except Exception:
                history = []

        if not isinstance(history, list):
            return []

        if goal is not None:
            g = str(goal)
            history = [row for row in history if str(row.get("goal") or "") == g]

        if run_id is not None:
            rid = str(run_id)
            history = [row for row in history if str(row.get("run_id") or "") == rid]

        if lim and len(history) > lim:
            history = history[-lim:]

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
            if not hasattr(self.memory_store, "get_replay_items_for_goal"):
                return stats
            items = self.memory_store.get_replay_items_for_goal(goal)  # type: ignore[attr-defined]
        except Exception:
            return stats

        if not isinstance(items, list) or not items:
            return stats

        total_items = len(items)
        high_rye = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            rye_score = _safe_float(it.get("rye_score", it.get("RYE", it.get("rye"))), default=0.0)
            if rye_score >= self.breakthrough_mid:
                high_rye += 1

        stats["items_for_goal"] = total_items
        stats["high_rye_items"] = high_rye
        stats["replay_density"] = high_rye / max(1, total_items)
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
    def _compute_skill_dimensions(self, history: List[HistoryRow]) -> SkillDimensionScores:
        """Compute core skill dimensions from cycle history."""
        if not history:
            return SkillDimensionScores()

        history = _sorted_history(history)

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
            rye_vals.append(_get_rye(row))
            delta_vals.append(_get_delta_r(row))
            energy_vals.append(_get_energy(row))
            breakthrough_vals.append(_get_breakthrough_score(row))

            citation_counts.append(_len_citations(row))
            hypothesis_counts.append(_len_hypotheses(row))

            equilibrium_labels.append(_eq_label(row))
            oscillation_scores.append(_oscillation(row))

            ms = _meta_signals(row)
            open_questions_counts.append(_count_like(ms.get("open_questions")))
            contradictions_counts.append(_count_like(ms.get("contradictions")))
            todo_counts.append(_count_like(ms.get("todo_items", ms.get("todos"))))

            stats_block = _stats_block(row)
            web_calls_all.append(_safe_int(stats_block.get("web_calls")))
            pubmed_calls_all.append(_safe_int(stats_block.get("pubmed_calls")))
            semantic_calls_all.append(_safe_int(stats_block.get("semantic_calls")))

            dom_flags_before = _as_dict(row.get("domain_issue_flags_before"))
            dom_flags_after = _as_dict(row.get("domain_issue_flags_after"))
            if any(dom_flags_before.get(k) or dom_flags_after.get(k) for k in ("missing_biomarkers", "missing_mechanisms")):
                longevity_issue_cycles += 1
            if any(dom_flags_before.get(k) or dom_flags_after.get(k) for k in ("missing_formalism", "missing_connections")):
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
        reasoning_score += 0.5 * _clip01(rye_med)
        reasoning_score += 0.3 * _clip01(rye_trend + 0.5)
        reasoning_score += 0.2 * _clip01(rye_stability)

        # Literature navigation: citations and science heavy tool usage
        avg_citations = statistics.mean(citation_counts) if citation_counts else 0.0
        rich_citation_cycles = sum(1 for c in citation_counts if c >= 5)
        richness_ratio = rich_citation_cycles / max(1, len(citation_counts))

        avg_web_calls = statistics.mean(web_calls_all) if web_calls_all else 0.0
        avg_pubmed_calls = statistics.mean(pubmed_calls_all) if pubmed_calls_all else 0.0
        avg_sem_calls = statistics.mean(semantic_calls_all) if semantic_calls_all else 0.0

        sci_calls = avg_pubmed_calls + avg_sem_calls
        sci_factor = _clip01(sci_calls / 4.0)
        web_factor = _clip01(avg_web_calls / 6.0)

        literature_nav = 0.4 * _clip01(avg_citations / 10.0)
        literature_nav += 0.3 * _clip01(richness_ratio)
        literature_nav += 0.2 * sci_factor
        literature_nav += 0.1 * web_factor

        # Hypothesis generation: count, focus, breakthroughs, replay density if available
        avg_hyp = statistics.mean(hypothesis_counts) if hypothesis_counts else 0.0
        focused_cycles = sum(1 for h in hypothesis_counts if 1 <= h <= 5)
        focused_ratio = focused_cycles / max(1, len(hypothesis_counts))
        avg_breakthrough = statistics.mean(breakthrough_vals) if breakthrough_vals else 0.0

        hypothesis_skill = 0.35 * _clip01(avg_hyp / 10.0)
        hypothesis_skill += 0.35 * _clip01(focused_ratio)
        hypothesis_skill += 0.30 * _clip01(avg_breakthrough)

        # Optional replay density hook if available in MemoryStore
        replay_boost = 0.0
        if self.use_replay_density and history:
            some_goal = history[-1].get("goal")
            if some_goal:
                rs = self._get_replay_stats_for_goal(str(some_goal))
                replay_density = _safe_float(rs.get("replay_density"), default=0.0)
                replay_boost = 0.2 * _clip01(replay_density / 0.5)
        hypothesis_skill = _clip01(hypothesis_skill + replay_boost)

        # Planning and curriculum: stages, roles, curriculum tags
        stages = [str(row.get("stage") or "idea") for row in history]
        unique_stages = len({s for s in stages if s})

        roles = [str(row.get("role") or "agent") for row in history]
        unique_roles = len(set(roles))

        curriculum_tags: List[str] = []
        for row in history:
            for t in _as_list(row.get("tags")):
                if isinstance(t, str) and t.startswith("curriculum:"):
                    curriculum_tags.append(t)

        unique_curriculum_phases = len(set(curriculum_tags)) if curriculum_tags else 0
        stage_balance = _clip01(unique_stages / 3.0)
        role_balance = _clip01(unique_roles / 6.0)
        curriculum_balance = _clip01(unique_curriculum_phases / 4.0)

        planning_skill = 0.4 * stage_balance + 0.4 * role_balance + 0.2 * curriculum_balance
        planning_skill = _clip01(planning_skill)

        # Stability and safety: equilibrium labels, oscillation, open questions and contradictions
        stable_cycles = sum(1 for label in equilibrium_labels if label in self._stable_labels)
        unstable_cycles = sum(1 for label in equilibrium_labels if label in self._unstable_labels)
        total = max(1, len(equilibrium_labels))
        stability_fraction = stable_cycles / total
        instability_fraction = unstable_cycles / total
        avg_osc = statistics.mean(oscillation_scores) if oscillation_scores else 0.0

        avg_open_questions = statistics.mean(open_questions_counts) if open_questions_counts else 0.0
        avg_todos = statistics.mean(todo_counts) if todo_counts else 0.0
        avg_contradictions = statistics.mean(contradictions_counts) if contradictions_counts else 0.0

        unresolved_pressure = _clip01((avg_open_questions + avg_todos + avg_contradictions) / 10.0)

        longevity_ratio = longevity_issue_cycles / max(1, len(history))
        math_ratio = math_issue_cycles / max(1, len(history))
        longevity_penalty = min(0.2, longevity_ratio * 0.3)
        math_penalty = min(0.2, math_ratio * 0.3)

        stability_skill = 0.6 * _clip01(stability_fraction)
        stability_skill += 0.2 * _clip01(1.0 - avg_osc)
        stability_skill += 0.2 * _clip01(1.0 - instability_fraction)
        stability_skill -= 0.15 * unresolved_pressure
        stability_skill -= longevity_penalty
        stability_skill -= math_penalty
        stability_skill = _clip01(stability_skill)

        return SkillDimensionScores(
            reasoning=_clip01(reasoning_score),
            literature_navigation=_clip01(literature_nav),
            hypothesis_generation=_clip01(hypothesis_skill),
            planning_and_curriculum=_clip01(planning_skill),
            stability_and_safety=_clip01(stability_skill),
        )

    def _compute_domain_profiles(self, history: List[HistoryRow]) -> List[DomainProfile]:
        """Compute per domain MSIL profiles."""
        if not history:
            return []

        history = _sorted_history(history)

        by_domain: Dict[str, List[HistoryRow]] = {}
        for row in history:
            for dom in _extract_domains_from_row(row):
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
                rye_vals.append(_get_rye(r))
                breakthrough_vals.append(_get_breakthrough_score(r))
                energy_vals.append(_get_energy(r))
                delta_vals.append(_get_delta_r(r))

            rye_series = [v for v in rye_vals if v > 0]
            median_rye = statistics.median(rye_series) if rye_series else None
            trend_slope = regression_rye_slope(rye_series) if rye_series else 0.0
            stab = stability_index(rye_series) if rye_series else 0.0
            rec = recovery_momentum(rye_series) if rye_series else 0.0

            high_br_cycles = sum(1 for v in breakthrough_vals if v >= self.breakthrough_high)
            breakthrough_density = high_br_cycles / max(1, len(rows))

            # Combine into a small domain specific MSIL score
            score = 0.35 * _clip01(median_rye or 0.0)
            score += 0.25 * _clip01(stab)
            score += 0.20 * _clip01(trend_slope + 0.5)
            score += 0.20 * _clip01(breakthrough_density)
            score = _clip01(score)

            profiles.append(
                DomainProfile(
                    domain=dom,
                    cycles=len(rows),
                    median_rye=median_rye,
                    trend_slope=float(trend_slope),
                    stability_index=float(_clip01(stab)),
                    recovery_momentum=float(_clip01(rec)),
                    breakthrough_density=float(_clip01(breakthrough_density)),
                    msil_score=float(_clip01(score)),
                    avg_energy=_mean_or_none(energy_vals),
                    avg_delta_r=_mean_or_none(delta_vals),
                )
            )

        profiles.sort(key=lambda p: p.msil_score, reverse=True)
        return profiles

    def _aggregate_msil_score(self, skills: SkillDimensionScores, domain_profiles: List[DomainProfile]) -> float:
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

        score = 0.55 * _clip01(skill_mean) + 0.30 * _clip01(dom_mean) + 0.15 * _clip01(dom_top)
        return _clip01(score)

    # ------------------------------------------------------------------
    # Diagnostics and stages
    # ------------------------------------------------------------------
    def _build_run_diagnostics(self, history: List[HistoryRow]) -> Dict[str, Any]:
        """Wrap build_run_diagnostics with safe fallbacks plus tool stats."""
        try:
            diag = build_run_diagnostics(history)
            base_diag = diag if isinstance(diag, dict) else {}
        except Exception:
            rye_vals = [_get_rye(row) for row in history]
            rye_vals = [v for v in rye_vals if isinstance(v, (int, float)) and v > 0]
            base_diag = {
                "series": {"rye": rye_vals},
                "stability_index": stability_index(rye_vals),
                "recovery_momentum": recovery_momentum(rye_vals),
                "trend_slope": regression_rye_slope(rye_vals),
                "rye_percentiles": rye_percentiles(rye_vals),
            }

        energy_vals: List[float] = []
        web_calls_all: List[int] = []
        pubmed_calls_all: List[int] = []
        semantic_calls_all: List[int] = []
        tokens_all: List[int] = []

        for row in history:
            energy_vals.append(_get_energy(row))
            stats_block = _stats_block(row)
            web_calls_all.append(_safe_int(stats_block.get("web_calls")))
            pubmed_calls_all.append(_safe_int(stats_block.get("pubmed_calls")))
            semantic_calls_all.append(_safe_int(stats_block.get("semantic_calls")))
            tu = _as_dict(row.get("tool_usage"))
            tokens_all.append(_safe_int(tu.get("approx_tokens", tu.get("tokens"))))

        extra_diag: Dict[str, Any] = {}
        avg_energy = _mean_or_none(energy_vals)
        if avg_energy is not None:
            extra_diag["avg_energy_per_cycle"] = avg_energy
        if web_calls_all:
            extra_diag["avg_web_calls_per_cycle"] = statistics.mean(web_calls_all)
        if pubmed_calls_all:
            extra_diag["avg_pubmed_calls_per_cycle"] = statistics.mean(pubmed_calls_all)
        if semantic_calls_all:
            extra_diag["avg_semantic_calls_per_cycle"] = statistics.mean(semantic_calls_all)
        if tokens_all:
            extra_diag["avg_tokens_per_cycle"] = statistics.mean(tokens_all)

        base_diag["energy_and_tools"] = extra_diag
        base_diag["version"] = __version__
        return base_diag

    def _infer_stage(self, msil_score: float, cycles: int) -> str:
        """Map MSIL score and cycles to an interpretive intelligence stage."""
        cycles_int = int(cycles)
        score = _clip01(msil_score)

        if cycles_int < max(5, self.min_cycles // 2):
            return "cold_start"

        if cycles_int < self.min_cycles:
            if score < 0.3:
                return "warmup_low"
            if score < 0.6:
                return "warmup_mid"
            return "warmup_high"

        if score < 0.3:
            return "learning_basic"
        if score < 0.55:
            return "learning_capable"
        if score < 0.75:
            return "specialised_expert"
        if score < self.frontier_threshold:
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
        recent_history: List[HistoryRow],
    ) -> Dict[str, Any]:
        """Produce advisory actions for swarm, curriculum, and config."""
        actions: Dict[str, Any] = {
            "priority": [],
            "swarm_config": {},
            "curriculum": {},
            "monitoring": {},
            "version": __version__,
        }

        skill_dict = skills.to_dict()
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

        best_domain = domain_profiles[0] if domain_profiles else None
        score = _clip01(msil_score)

        if best_domain and best_domain.domain in {"longevity", "math"} and score >= 0.6:
            actions["swarm_config"]["suggested_mode"] = "8_to_16_agent_swarm"
        elif score >= 0.4:
            actions["swarm_config"]["suggested_mode"] = "4_to_8_agent_swarm"
        else:
            actions["swarm_config"]["suggested_mode"] = "single_or_small_swarm"

        actions["swarm_config"]["role_mix_hint"] = (
            "Use at least one critic, one planner, one explorer, and one synthesizer for high stakes swarms."
        )

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

        hallmarks: List[str] = []
        subgoals: List[str] = []
        for row in recent_history:
            if row.get("hallmark"):
                hallmarks.append(str(row.get("hallmark")))
            if row.get("subgoal"):
                subgoals.append(str(row.get("subgoal")))
        if hallmarks:
            actions["curriculum"]["observed_hallmarks"] = sorted(set(hallmarks))
        if subgoals:
            actions["curriculum"]["observed_subgoals"] = sorted(set(subgoals))

        if total_cycles < 100:
            actions["monitoring"]["note"] = (
                "MSIL metrics are early. Re evaluate after passing 100 cycles on the same goal."
            )
        else:
            actions["monitoring"]["note"] = (
                "Track MSIL score, stability_index, and breakthrough_density weekly for long runs."
            )

        actions["monitoring"]["recommended_metrics"] = [
            "msil_score",
            "stability_index",
            "trend_slope",
            "breakthrough_density",
        ]

        recent_osc = [_oscillation(row) for row in recent_history]
        recent_labels = [_eq_label(row) for row in recent_history]
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
            timestamp=_now_utc_iso(),
            goal=str(goal or "unknown_goal"),
            msil_score=0.0,
            intelligence_stage="disabled",
            total_cycles_for_goal=0,
            recent_window=0,
            skills=SkillDimensionScores(),
            per_domain_profiles=[],
            actions={
                "priority": ["MSIL is disabled in config. Set msil_enabled to True to activate."],
                "swarm_config": {},
                "curriculum": {},
                "monitoring": {},
                "version": __version__,
            },
            extras={"version": __version__},
        )


# ----------------------------------------------------------------------
# Lightweight adapter so the UI can call analyze_run(history, ...)
# without needing a real MemoryStore.
# ----------------------------------------------------------------------


class _HistoryBackedMemoryStore:
    """Minimal MemoryStore-like wrapper around an in-memory history list."""

    def __init__(self, history: List[HistoryRow]) -> None:
        self._history: List[HistoryRow] = list(history or [])

    def get_cycle_history_for_goal(self, goal: str, limit: Optional[int] = None) -> List[HistoryRow]:
        rows = [row for row in self._history if str(row.get("goal") or "") == str(goal)]
        if limit is not None and len(rows) > int(limit):
            rows = rows[-int(limit) :]
        return rows

    def get_cycle_history(self, limit: Optional[int] = None) -> List[HistoryRow]:
        rows = list(self._history)
        if limit is not None and len(rows) > int(limit):
            rows = rows[-int(limit) :]
        return rows


def analyze_run(
    history: List[HistoryRow],
    goal: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience helper for UIs.

    Args:
        history: List of cycle logs, as usually returned by MemoryStore.
        goal: Optional goal filter. If None, uses the goal from the last cycle.
        config: Optional MSIL config dict.

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
            "version": __version__,
        }

    if goal is None:
        goal = str(history[-1].get("goal") or "unknown_goal")

    store = _HistoryBackedMemoryStore(history)
    msil = MetaSkillIntelligenceLayer(store, config=config or {})
    return msil.summarise_run(goal=goal, limit=len(history))


# Backward compatible alias in case some code calls analyze_history
analyze_history = analyze_run
