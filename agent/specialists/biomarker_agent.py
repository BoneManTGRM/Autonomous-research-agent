"""BiomarkerAgent

Specialist for handling biomarker snapshots, trends, and longevity signals.

Reparodynamics view:
    This specialist treats biomarker streams as a repair surface:
        - Each snapshot is a state vector.
        - History of snapshots forms trajectories over time.
        - It estimates stability, drift, and risk bands.
        - It produces replay items for high yield mechanism or stack ideas.

Goals:
    - Turn raw biomarker snapshots into structured, comparable signals.
    - Detect useful trends fast for longevity runs.
    - Surface high RYE candidate moves for other agents (synergy, meta, critic).

This file is intentionally self contained and defensive. It only assumes that
MemoryStore provides:
    - log_cycle(...) or get_cycle_history_for_goal(...)
    - Optional add_biomarker_insight(...)

It also integrates with ReplayBuffer if provided, but does not require it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math
import statistics


@dataclass
class BiomarkerPoint:
    """Single biomarker reading at a point in time."""

    name: str
    value: float
    unit: Optional[str] = None
    timestamp: Optional[str] = None
    run_id: Optional[str] = None
    cycle_index: Optional[int] = None


@dataclass
class BiomarkerTrend:
    """Summary of a biomarker trajectory over recent history."""

    name: str
    mean: Optional[float]
    std: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    latest: Optional[float]
    slope: Optional[float]
    direction: str
    volatility: str
    sample_count: int


class BiomarkerAgent:
    """Biomarker specialist for longevity and health oriented runs.

    This agent does not fetch new data. It organizes and analyzes the
    biomarker_snapshot field already recorded in cycle logs, and the
    snapshot passed in for the current cycle.

    Typical use:
        agent = BiomarkerAgent(memory_store, config)
        result = agent.analyze_snapshot(
            goal=goal,
            snapshot=current_biomarkers,
            hallmark="mitochondria",
            run_id=run_id,
            cycle_index=cycle_index,
            replay_buffer=replay_buffer,
        )
    """

    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # How many past cycles to look at when building history.
        self.history_window: int = int(self.config.get("biomarker_history_window", 60))

        # Minimum points required before trend analysis is considered meaningful.
        self.min_points_for_trend: int = int(self.config.get("biomarker_min_points_for_trend", 5))

        # Thresholds for volatility and direction heuristics.
        self.slope_small: float = float(self.config.get("biomarker_slope_small", 0.01))
        self.slope_large: float = float(self.config.get("biomarker_slope_large", 0.05))
        self.std_low: float = float(self.config.get("biomarker_std_low", 0.05))
        self.std_high: float = float(self.config.get("biomarker_std_high", 0.25))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_snapshot(
        self,
        goal: str,
        snapshot: Optional[Dict[str, Any]],
        hallmark: Optional[str] = None,
        run_id: Optional[str] = None,
        cycle_index: Optional[int] = None,
        replay_buffer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Analyze a single biomarker snapshot in the context of history.

        Args:
            goal:
                High level research goal this snapshot belongs to.
            snapshot:
                Dict mapping biomarker name to numeric value, or a small dict
                with value and unit, for example:
                    {
                        "LDL": 120,
                        "HDL": {"value": 55, "unit": "mg/dL"},
                    }
            hallmark:
                Optional hallmark label such as "mitochondria" or "senescence".
            run_id:
                Optional run identifier.
            cycle_index:
                Optional cycle index within the run.
            replay_buffer:
                Optional ReplayBuffer compatible object with add_item(...) or
                add_replay_item(...).

        Returns:
            Dict with:
                - snapshot_points: per biomarker parsed values
                - trends: per biomarker trajectory stats
                - z_scores: per biomarker z score for latest value
                - risk_bands: heuristic risk or vitality band per biomarker
                - global_summary: compact summary for reports and UI
        """
        hallmark_name = hallmark or "unspecified"
        timestamp = datetime.utcnow().isoformat() + "Z"

        parsed_points = self._parse_snapshot(
            snapshot=snapshot,
            timestamp=timestamp,
            run_id=run_id,
            cycle_index=cycle_index,
        )

        history = self._load_history(goal, limit=self.history_window)
        trends = self._build_trends(history, parsed_points)
        z_scores = self._compute_z_scores(trends, parsed_points)
        risk_bands = self._assign_risk_bands(trends, z_scores, hallmark_name)

        global_summary = self._build_global_summary(
            goal=goal,
            hallmark=hallmark_name,
            trends=trends,
            risk_bands=risk_bands,
        )

        # Persist biomarker insight into MemoryStore if supported.
        insight_payload = {
            "goal": goal,
            "hallmark": hallmark_name,
            "run_id": run_id,
            "cycle_index": cycle_index,
            "timestamp": timestamp,
            "snapshot": {p.name: p.value for p in parsed_points},
            "trends": {t.name: asdict(t) for t in trends.values()},
            "z_scores": z_scores,
            "risk_bands": risk_bands,
            "summary": global_summary,
        }
        self._log_insight_to_memory(insight_payload)

        # Optionally send high value replay items to ReplayBuffer.
        self._maybe_emit_replay_items(
            goal=goal,
            hallmark=hallmark_name,
            run_id=run_id,
            cycle_index=cycle_index,
            trends=trends,
            risk_bands=risk_bands,
            z_scores=z_scores,
            replay_buffer=replay_buffer,
        )

        return {
            "goal": goal,
            "hallmark": hallmark_name,
            "timestamp": timestamp,
            "snapshot_points": [asdict(p) for p in parsed_points],
            "trends": {name: asdict(t) for name, t in trends.items()},
            "z_scores": z_scores,
            "risk_bands": risk_bands,
            "global_summary": global_summary,
        }

    # ------------------------------------------------------------------
    # History loading
    # ------------------------------------------------------------------
    def _load_history(
        self,
        goal: str,
        limit: int,
    ) -> Dict[str, List[BiomarkerPoint]]:
        """Reconstruct biomarker history from MemoryStore cycle logs.

        This tries specialized APIs first, then falls back to scanning
        cycle history logs if needed.
        """
        points_by_name: Dict[str, List[BiomarkerPoint]] = {}

        # Preferred: dedicated history method on MemoryStore.
        try:
            if hasattr(self.memory_store, "get_biomarker_history_for_goal"):
                rows = self.memory_store.get_biomarker_history_for_goal(goal, limit=limit)  # type: ignore[attr-defined]
                for row in rows or []:
                    name = row.get("name")
                    value = row.get("value")
                    if name is None or value is None:
                        continue
                    try:
                        v = float(value)
                    except Exception:
                        continue
                    point = BiomarkerPoint(
                        name=str(name),
                        value=v,
                        unit=row.get("unit"),
                        timestamp=row.get("timestamp"),
                        run_id=row.get("run_id"),
                        cycle_index=row.get("cycle_index"),
                    )
                    points_by_name.setdefault(point.name, []).append(point)
                return points_by_name
        except Exception:
            pass

        # Fallback: scan cycle history for biomarker_snapshot fields.
        rows: List[Dict[str, Any]] = []
        try:
            if hasattr(self.memory_store, "get_cycle_history_for_goal"):
                rows = self.memory_store.get_cycle_history_for_goal(goal, limit=limit)  # type: ignore[attr-defined]
            elif hasattr(self.memory_store, "get_cycle_history"):
                all_rows = self.memory_store.get_cycle_history()  # type: ignore[attr-defined]
                if isinstance(all_rows, list):
                    rows = [r for r in all_rows if r.get("goal") == goal][-limit:]
        except Exception:
            rows = []

        for row in rows or []:
            snap = row.get("biomarker_snapshot")
            if not isinstance(snap, dict):
                continue
            ts = row.get("timestamp") or row.get("time") or row.get("created_at")
            run_id = row.get("run_id")
            cycle_idx = row.get("cycle")

            parsed = self._parse_snapshot(
                snapshot=snap,
                timestamp=ts,
                run_id=run_id,
                cycle_index=cycle_idx,
            )
            for p in parsed:
                points_by_name.setdefault(p.name, []).append(p)

        return points_by_name

    # ------------------------------------------------------------------
    # Parsing and trend building
    # ------------------------------------------------------------------
    def _parse_snapshot(
        self,
        snapshot: Optional[Dict[str, Any]],
        timestamp: Optional[str],
        run_id: Optional[str],
        cycle_index: Optional[int],
    ) -> List[BiomarkerPoint]:
        """Normalize a snapshot dict into a list of BiomarkerPoint objects."""
        if not snapshot:
            return []

        points: List[BiomarkerPoint] = []
        for name, raw in snapshot.items():
            unit: Optional[str] = None
            value: Optional[float] = None

            if isinstance(raw, dict):
                raw_val = raw.get("value")
                unit = raw.get("unit")
            else:
                raw_val = raw

            try:
                value = float(raw_val)
            except Exception:
                continue

            point = BiomarkerPoint(
                name=str(name),
                value=value,
                unit=unit,
                timestamp=timestamp,
                run_id=run_id,
                cycle_index=cycle_index,
            )
            points.append(point)

        return points

    def _build_trends(
        self,
        history: Dict[str, List[BiomarkerPoint]],
        current_points: List[BiomarkerPoint],
    ) -> Dict[str, BiomarkerTrend]:
        """Build trend stats for each biomarker using history plus current point."""
        trends: Dict[str, BiomarkerTrend] = {}

        by_name = {p.name: p for p in current_points}
        names = set(history.keys()) | set(by_name.keys())

        for name in sorted(names):
            series = list(history.get(name, []))

            # Append current value for trend analysis.
            if name in by_name:
                series = series + [by_name[name]]

            if not series:
                continue

            values = [p.value for p in series]
            sample_count = len(values)

            mean_val: Optional[float]
            std_val: Optional[float]
            if sample_count >= 2:
                try:
                    mean_val = statistics.mean(values)
                    std_val = statistics.pstdev(values)
                except Exception:
                    mean_val = None
                    std_val = None
            else:
                mean_val = values[0]
                std_val = None

            min_val = min(values)
            max_val = max(values)
            latest = values[-1]

            slope = self._estimate_slope(values)
            direction = self._direction_label(slope, sample_count)
            volatility = self._volatility_label(std_val, sample_count)

            trend = BiomarkerTrend(
                name=name,
                mean=mean_val,
                std=std_val,
                min_val=min_val,
                max_val=max_val,
                latest=latest,
                slope=slope,
                direction=direction,
                volatility=volatility,
                sample_count=sample_count,
            )
            trends[name] = trend

        return trends

    def _estimate_slope(self, values: List[float]) -> Optional[float]:
        """Approximate trend slope using a simple regression on index vs value."""
        n = len(values)
        if n < self.min_points_for_trend:
            return None

        # Simple least squares fit to y = a + b x, x = 0..n-1
        x_vals = list(range(n))
        mean_x = sum(x_vals) / n
        mean_y = sum(values) / n

        numer = 0.0
        denom = 0.0
        for x, y in zip(x_vals, values):
            dx = x - mean_x
            dy = y - mean_y
            numer += dx * dy
            denom += dx * dx

        if denom == 0:
            return None

        slope = numer / denom
        return float(slope)

    def _direction_label(self, slope: Optional[float], sample_count: int) -> str:
        """Convert slope to a qualitative direction label."""
        if slope is None or sample_count < self.min_points_for_trend:
            return "unknown_trend"

        s = abs(slope)
        if s < self.slope_small:
            return "flat"
        if slope > 0:
            if s >= self.slope_large:
                return "rising_fast"
            return "rising"
        else:
            if s >= self.slope_large:
                return "falling_fast"
            return "falling"

    def _volatility_label(self, std_val: Optional[float], sample_count: int) -> str:
        """Assign a volatility label based on the spread."""
        if std_val is None or sample_count < self.min_points_for_trend:
            return "unknown_volatility"

        s = float(std_val)
        if s <= self.std_low:
            return "low_volatility"
        if s >= self.std_high:
            return "high_volatility"
        return "moderate_volatility"

    # ------------------------------------------------------------------
    # Z scores and risk bands
    # ------------------------------------------------------------------
    def _compute_z_scores(
        self,
        trends: Dict[str, BiomarkerTrend],
        current_points: List[BiomarkerPoint],
    ) -> Dict[str, Optional[float]]:
        """Compute simple z scores for latest value relative to mean and std."""
        z_scores: Dict[str, Optional[float]] = {}
        current_map = {p.name: p for p in current_points}

        for name, trend in trends.items():
            latest_val: Optional[float] = None
            if trend.latest is not None:
                latest_val = trend.latest
            elif name in current_map:
                latest_val = current_map[name].value

            if latest_val is None or trend.mean is None or trend.std is None:
                z_scores[name] = None
                continue

            if trend.std == 0:
                z_scores[name] = None
                continue

            z = (latest_val - trend.mean) / trend.std
            z_scores[name] = float(z)

        return z_scores

    def _assign_risk_bands(
        self,
        trends: Dict[str, BiomarkerTrend],
        z_scores: Dict[str, Optional[float]],
        hallmark: str,
    ) -> Dict[str, str]:
        """Assign rough risk or vitality bands based on z scores and hallmark.

        For now this uses generic bands only:
            z in [-0.5, 0.5]  -> "near_baseline"
            z in [0.5, 1.5]   -> "mild_shift"
            |z| > 2.0         -> "strong_deviation"

        Future versions can plug in domain specific ranges per marker.
        """
        bands: Dict[str, str] = {}
        for name, trend in trends.items():
            z = z_scores.get(name)
            if z is None:
                bands[name] = "unknown_band"
                continue

            az = abs(z)
            if az <= 0.5:
                bands[name] = "near_baseline"
            elif az <= 1.5:
                bands[name] = "mild_shift"
            elif az <= 2.5:
                bands[name] = "moderate_deviation"
            else:
                # Large magnitude deviation. Risk interpretation depends on marker
                # sign, so for now it is labeled as high deviation rather than
                # explicit risk.
                bands[name] = "strong_deviation"

        return bands

    # ------------------------------------------------------------------
    # Global summary and replay
    # ------------------------------------------------------------------
    def _build_global_summary(
        self,
        goal: str,
        hallmark: str,
        trends: Dict[str, BiomarkerTrend],
        risk_bands: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build a compact summary for reports and dashboards."""
        if not trends:
            return {
                "goal": goal,
                "hallmark": hallmark,
                "trend_quality": "no_biomarkers",
                "high_signal_markers": [],
                "stable_markers": [],
                "unstable_markers": [],
            }

        high_signal: List[str] = []
        stable: List[str] = []
        unstable: List[str] = []

        for name, trend in trends.items():
            band = risk_bands.get(name, "unknown_band")
            if band in {"moderate_deviation", "strong_deviation"}:
                high_signal.append(name)

            if trend.volatility == "low_volatility":
                stable.append(name)
            elif trend.volatility == "high_volatility":
                unstable.append(name)

        if high_signal:
            trend_quality = "high_signal"
        elif stable and not unstable:
            trend_quality = "stable"
        else:
            trend_quality = "mixed"

        return {
            "goal": goal,
            "hallmark": hallmark,
            "trend_quality": trend_quality,
            "high_signal_markers": sorted(high_signal),
            "stable_markers": sorted(stable),
            "unstable_markers": sorted(unstable),
            "marker_count": len(trends),
        }

    def _maybe_emit_replay_items(
        self,
        goal: str,
        hallmark: str,
        run_id: Optional[str],
        cycle_index: Optional[int],
        trends: Dict[str, BiomarkerTrend],
        risk_bands: Dict[str, str],
        z_scores: Dict[str, Optional[float]],
        replay_buffer: Optional[Any],
    ) -> None:
        """Emit replay items for markers that look most informative."""
        if replay_buffer is None:
            return

        # Decide which markers are worth replay focus.
        candidates: List[Tuple[str, float]] = []
        for name, trend in trends.items():
            z = z_scores.get(name)
            if z is None:
                continue
            band = risk_bands.get(name, "unknown_band")
            if band in {"moderate_deviation", "strong_deviation"} or trend.volatility == "high_volatility":
                candidates.append((name, abs(z)))

        if not candidates:
            return

        # Sort strongest deviations first and emit replay items for the top few.
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = min(5, len(candidates))

        timestamp = datetime.utcnow().isoformat() + "Z"
        for name, _score in candidates[:top_k]:
            trend = trends[name]
            z = z_scores.get(name)
            band = risk_bands.get(name)

            hypothesis_text = (
                f"Biomarker {name} shows {trend.direction} pattern with volatility "
                f"{trend.volatility} and deviation band '{band}' for goal '{goal}', "
                f"hallmark '{hallmark}'. This may indicate a promising mechanistic "
                f"or intervention target for further longevity analysis."
            )

            replay_payload = {
                "item_id": None,  # allow ReplayBuffer to assign
                "hallmark": hallmark,
                "stage": "idea",
                "mechanism_chain": None,
                "biomarker_pattern": {
                    "name": name,
                    "direction": trend.direction,
                    "volatility": trend.volatility,
                    "band": band,
                    "mean": trend.mean,
                    "latest": trend.latest,
                    "z_score": z,
                },
                "hypothesis_text": hypothesis_text,
                "rye_score": None,
                "energy_cost": None,
                "decision": "pending",
                "reason": "high_signal_biomarker_pattern",
                "source_citations": [],
                "tags": [
                    "biomarker_signal",
                    "longevity",
                    hallmark,
                ],
                "created_at": timestamp,
                "run_id": run_id,
                "cycle_index": cycle_index,
            }

            try:
                if hasattr(replay_buffer, "add_item"):
                    replay_buffer.add_item(replay_payload)  # type: ignore[attr-defined]
                elif hasattr(replay_buffer, "add_replay_item"):
                    replay_buffer.add_replay_item(replay_payload)  # type: ignore[attr-defined]
            except Exception:
                # Replay failures must never break the cycle.
                continue

    # ------------------------------------------------------------------
    # Memory logging
    # ------------------------------------------------------------------
    def _log_insight_to_memory(self, insight_payload: Dict[str, Any]) -> None:
        """Record biomarker insight into MemoryStore if supported."""
        try:
            if hasattr(self.memory_store, "add_biomarker_insight"):
                self.memory_store.add_biomarker_insight(insight_payload)  # type: ignore[attr-defined]
                return
        except Exception:
            pass

        # Fallback: store as a generic note so nothing is lost.
        try:
            goal = str(insight_payload.get("goal", "biomarker_insight"))
            text = (
                f"[BiomarkerAgent] Snapshot analysis for hallmark="
                f"{insight_payload.get('hallmark', 'unspecified')} "
                f"with {insight_payload.get('marker_count', 'unknown')} markers."
            )
            if hasattr(self.memory_store, "add_note"):
                self.memory_store.add_note(goal, text, role="biomarker_agent")  # type: ignore[attr-defined]
        except Exception:
            pass
