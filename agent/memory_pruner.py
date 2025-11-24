"""
memory_pruner.py - TGRM / RYE aligned Memory Optimization Engine
Tier 3 version for long run autonomy and breakthrough discovery odds.

This pruner is optimized for:
    - 90 day and longer runs
    - swarm intelligence
    - TGRM stability zones
    - high RYE preservation
    - major discovery detection
    - anti collapse memory maintenance

Capabilities:
    ✓ Adaptive scoring using RYE × Recency × Access × Discovery Boost × Predictive Access
    ✓ Hard protection for verified hypotheses, discoveries, equilibrium notes
    ✓ Soft clustering of related notes (keeps idea families intact)
    ✓ Predictive access scoring for the next run segment and domain
    ✓ Intelligent compression instead of naive deletion
    ✓ Intelligence profile and diagnostics aware tuning
    ✓ Discovery log integration for pruning events
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRUNE_LOG_PATH = LOG_DIR / "memory_pruning_log.md"

PRUNER_VERSION: str = "2025-11-23"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _soft_clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class ScoredEntry:
    entry_id: str
    score: float

    rye: Optional[float]
    created_at: Optional[datetime]
    last_accessed: Optional[datetime]
    access_count: Optional[int]
    tags: List[str]
    is_protected: bool = False

    # Component scores for diagnostics and learning
    rye_component: float = 0.0
    recency_component: float = 0.0
    access_component: float = 0.0
    discovery_component: float = 0.0
    equilibrium_component: float = 0.0
    predictive_component: float = 0.0

    domain: Optional[str] = None
    goal: Optional[str] = None
    kind: Optional[str] = None


# ------------------------------------------------------------
# Tier 3 Memory Pruner
# ------------------------------------------------------------
class MemoryPruner:
    """
    MemoryPruner implements RYE aligned, stability preserving, discovery maximizing pruning.

    Integrated protections:
        - Protect discoveries, hypotheses, validations
        - Protect equilibrium zone notes (RYE stable windows)
        - Protect high RYE entries
        - Compress stale clusters instead of deleting them
        - Bias toward keeping memory that is predicted to be useful in the next segment

    Expected Memory Store API:
        memory_store.list_entries() -> list of dicts:
            {
                "id": str,
                "content": str,
                "meta": {
                    "tags": [...],
                    "rye": float,
                    "created_at": iso_timestamp,
                    "last_accessed": iso_timestamp,
                    "access_count": int,
                    "domain": str,
                    "goal": str,
                    "kind": str,
                    "predictive_score": float,
                    "cluster_id": str,
                    ...
                }
            }

        memory_store.delete_entries(ids: List[str])

        memory_store.update_entry(id: str, fields: Dict[str, Any])  (optional for compression and tagging)
        memory_store.add_entry(content: str, meta: Dict[str, Any])  (optional for cluster roots)
    """

    # Tags that are always protected from deletion
    PROTECTED_TAGS = {
        "discovery",
        "verified",
        "hypothesis",
        "equilibrium",
        "rye_peak",
        "cluster_root",
        "swarm_key_insight",
        "math_key_theorem",
        "longevity_stack",
        "guardian_core",
    }

    def __init__(
        self,
        memory_store: Any,
        run_id: Optional[str] = None,
        discovery_logger: Optional[Any] = None,
    ) -> None:
        self.memory_store = memory_store
        self.run_id = run_id
        self.discovery_logger = discovery_logger
        self.last_summary: Optional[Dict[str, Any]] = None

        if not PRUNE_LOG_PATH.exists():
            self._write_header()

    # ------------------------------------------------------------
    #   PUBLIC API
    # ------------------------------------------------------------
    def prune(
        self,
        min_keep: int = 1200,
        max_drop_fraction: float = 0.25,
        rye_weight: float = 0.55,
        recency_weight: float = 0.25,
        access_weight: float = 0.20,
        discovery_boost: float = 0.30,
        equilibrium_boost: float = 0.20,
        predictive_weight: float = 0.25,
        *,
        current_domain: Optional[str] = None,
        current_goal: Optional[str] = None,
        intelligence_profile: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a Tier 3 pruning pass with automatic adaptive thresholds.

        Parameters:
            min_keep:
                Minimum number of entries to keep regardless of scores.
            max_drop_fraction:
                Maximum fraction of total entries that can be dropped in this pass.
            rye_weight, recency_weight, access_weight:
                Base weights for scoring components.
            discovery_boost, equilibrium_boost:
                Additive boosts for discovery and equilibrium related notes.
            predictive_weight:
                Weight for predictive access scoring.

            current_domain, current_goal:
                Optional hints about the active preset or run focus to bias
                predictive access scoring and protection.

            intelligence_profile:
                Optional intelligence profile dict (from intelligence_profiles.py).
                If provided, the pruner will:
                    - Raise equilibrium protection when equilibrium_focus is high.
                    - Allow a larger drop fraction when disruption_budget is high.
                    - Increase RYE emphasis when rye_weight_factor is high.

            diagnostics:
                Optional diagnostics snapshot from the engine.
                Expected keys (all optional):
                    - "stability_index": float (0 to 1)
                    - "recent_rye_trend": float (negative means decline)
                    - "equilibrium_fragile": bool

        Returns:
            Summary dict with counts and effective weights used.
        """

        entries = self.memory_store.list_entries()
        total = len(entries)

        # Adjust based on intelligence profile and diagnostics
        (
            rye_weight,
            recency_weight,
            access_weight,
            predictive_weight,
            discovery_boost,
            equilibrium_boost,
            max_drop_fraction,
        ) = self._tune_from_intelligence_and_diagnostics(
            rye_weight=rye_weight,
            recency_weight=recency_weight,
            access_weight=access_weight,
            predictive_weight=predictive_weight,
            discovery_boost=discovery_boost,
            equilibrium_boost=equilibrium_boost,
            max_drop_fraction=max_drop_fraction,
            intelligence_profile=intelligence_profile,
            diagnostics=diagnostics,
        )

        # Adaptive min_keep scaling based on size
        adaptive_min_keep = self._adaptive_min_keep(min_keep, total)

        if total <= adaptive_min_keep:
            summary = {
                "timestamp": _utc_iso(),
                "pruner_version": PRUNER_VERSION,
                "run_id": self.run_id,
                "total_before": total,
                "total_after": total,
                "dropped": 0,
                "min_keep": adaptive_min_keep,
                "max_drop_fraction": max_drop_fraction,
                "reason": "below_minimum",
            }
            self.last_summary = summary
            self._append_log(summary)
            self._log_to_discovery_log(summary)
            return summary

        scored = self._score_entries(
            entries,
            rye_weight=rye_weight,
            recency_weight=recency_weight,
            access_weight=access_weight,
            discovery_boost=discovery_boost,
            equilibrium_boost=equilibrium_boost,
            predictive_weight=predictive_weight,
            current_domain=current_domain,
            current_goal=current_goal,
        )

        # Sort by protection flag and score
        scored.sort(key=lambda x: (x.is_protected, x.score), reverse=True)

        max_drop = int(total * max_drop_fraction)
        keep_count = max(adaptive_min_keep, total - max_drop)

        keep_ids = {s.entry_id for s in scored[:keep_count]}
        drop_candidates = [s for s in scored[keep_count:] if not s.is_protected]

        drop_ids = [s.entry_id for s in drop_candidates]

        # Optional: compression step for stale clusters
        self._compress_stale_clusters(scored, drop_ids)

        # Apply deletions
        if drop_ids:
            self.memory_store.delete_entries(drop_ids)

        summary = {
            "timestamp": _utc_iso(),
            "pruner_version": PRUNER_VERSION,
            "run_id": self.run_id,
            "total_before": total,
            "total_after": total - len(drop_ids),
            "dropped": len(drop_ids),
            "min_keep": adaptive_min_keep,
            "max_drop_fraction": max_drop_fraction,
            "weights": {
                "rye": rye_weight,
                "recency": recency_weight,
                "access": access_weight,
                "predictive": predictive_weight,
                "discovery": discovery_boost,
                "equilibrium": equilibrium_boost,
            },
            "diagnostics": self._compute_prune_diagnostics(scored, keep_ids, drop_ids),
        }

        self.last_summary = summary
        self._append_log(summary)
        self._log_to_discovery_log(summary)
        return summary

    # Convenient helper if you want a hard cap instead of min_keep
    def prune_if_needed(
        self,
        max_entries: int,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Run prune only if memory size is above max_entries.

        Returns pruning summary if pruning occurred, else None.
        """
        total = len(self.memory_store.list_entries())
        if total <= max_entries:
            return None
        # use max_entries as a hard minimum to keep
        return self.prune(min_keep=max_entries, **kwargs)

    # ------------------------------------------------------------
    #   INTELLIGENCE AND DIAGNOSTICS TUNING
    # ------------------------------------------------------------
    def _tune_from_intelligence_and_diagnostics(
        self,
        *,
        rye_weight: float,
        recency_weight: float,
        access_weight: float,
        predictive_weight: float,
        discovery_boost: float,
        equilibrium_boost: float,
        max_drop_fraction: float,
        intelligence_profile: Optional[Dict[str, Any]],
        diagnostics: Optional[Dict[str, Any]],
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Tune weights and drop fraction based on intelligence profile and diagnostics.
        """

        # Intelligence profile adaptation
        if intelligence_profile:
            ryef = float(intelligence_profile.get("rye_weight_factor", 1.0) or 1.0)
            equilibrium_focus = float(intelligence_profile.get("equilibrium_focus", 0.5) or 0.5)
            disruption_budget = float(intelligence_profile.get("disruption_budget", 0.3) or 0.3)

            # Emphasize RYE when profile weight factor is high
            rye_weight *= ryef

            # If equilibrium focus is high, emphasize equilibrium protection and recency
            if equilibrium_focus >= 0.7:
                equilibrium_boost *= 1.5
                recency_weight *= 1.2
                max_drop_fraction *= 0.7  # prune less aggressively

            # If disruption budget is high, allow more aggressive pruning
            if disruption_budget >= 0.45:
                max_drop_fraction *= 1.2
                predictive_weight *= 1.1
                discovery_boost *= 1.1

        # Diagnostics adaptation
        if diagnostics:
            stability_index = diagnostics.get("stability_index")
            recent_rye_trend = diagnostics.get("recent_rye_trend")
            fragile = diagnostics.get("equilibrium_fragile")

            # If stability index is low or equilibrium is fragile, reduce pruning
            if isinstance(stability_index, (int, float)) and stability_index < 0.4:
                max_drop_fraction *= 0.6
                equilibrium_boost *= 1.4
            if fragile:
                max_drop_fraction *= 0.6
                equilibrium_boost *= 1.5

            # If recent RYE trend is strongly positive, we can prune a bit more
            if isinstance(recent_rye_trend, (int, float)) and recent_rye_trend > 0.1:
                max_drop_fraction *= 1.15
            # If recent RYE trend is negative, be more conservative
            if isinstance(recent_rye_trend, (int, float)) and recent_rye_trend < -0.05:
                max_drop_fraction *= 0.8

        max_drop_fraction = _soft_clip(max_drop_fraction, 0.05, 0.5)
        return (
            rye_weight,
            recency_weight,
            access_weight,
            predictive_weight,
            discovery_boost,
            equilibrium_boost,
            max_drop_fraction,
        )

    def _adaptive_min_keep(self, base_min_keep: int, total: int) -> int:
        """
        Scale min_keep based on overall memory size.
        """
        min_keep = base_min_keep
        if total > 10000:
            min_keep = int(min_keep * 1.4)
        if total > 20000:
            min_keep = int(min_keep * 1.8)
        if total > 50000:
            min_keep = int(min_keep * 2.2)
        return min_keep

    # ------------------------------------------------------------
    #   SCORING ENGINE (Tier 3)
    # ------------------------------------------------------------
    def _score_entries(
        self,
        entries: List[Dict[str, Any]],
        rye_weight: float,
        recency_weight: float,
        access_weight: float,
        discovery_boost: float,
        equilibrium_boost: float,
        predictive_weight: float,
        current_domain: Optional[str],
        current_goal: Optional[str],
    ) -> List[ScoredEntry]:

        now = datetime.now(timezone.utc)

        raw_scores: List[float] = []
        objects: List[ScoredEntry] = []

        for e in entries:
            meta = e.get("meta", {}) or {}
            tags: List[str] = list(meta.get("tags", []) or [])

            is_protected = any(t in self.PROTECTED_TAGS for t in tags)

            rye = meta.get("rye", 0.0)
            created_at = _parse_ts(meta.get("created_at"))
            last_accessed = _parse_ts(meta.get("last_accessed"))
            access_count = meta.get("access_count") or 0

            domain = meta.get("domain")
            goal = meta.get("goal")
            kind = meta.get("kind")

            # 1) RYE component
            rye_val = float(rye or 0.0)
            rye_comp = rye_weight * rye_val

            # 2) Recency component using a decay curve
            ref = last_accessed or created_at or now
            age_days = max((now - ref).total_seconds() / 86400.0, 0.0)
            # Newer entries get higher score; exponential decay
            recency_raw = math.exp(-age_days / 30.0)  # half life about 20 to 30 days
            recency_comp = recency_weight * recency_raw

            # 3) Access component
            access_raw = math.log1p(access_count)
            access_comp = access_weight * access_raw

            # 4) Discovery component
            has_discovery = any(t in ("discovery", "verified", "hypothesis") for t in tags)
            d_comp = discovery_boost if has_discovery else 0.0

            # 5) Equilibrium component
            has_equilibrium = "equilibrium" in tags or "rye_plateau" in tags
            e_comp = equilibrium_boost if has_equilibrium else 0.0

            # 6) Predictive access component
            # If the engine provided an explicit prediction, use it; else infer
            pred_raw = meta.get("predictive_score")
            if not isinstance(pred_raw, (int, float)):
                pred_raw = 0.0

            # Bias predictive score when domain and goal match current run
            domain_match = current_domain and domain and str(domain).lower() == str(current_domain).lower()
            goal_match = current_goal and goal and str(goal).lower() == str(current_goal).lower()
            if domain_match:
                pred_raw += 0.15
            if goal_match:
                pred_raw += 0.10

            # Hypotheses and discoveries get a predictive bump, they are often reused later
            if has_discovery:
                pred_raw += 0.10

            # Clip predictive score
            pred_raw = _soft_clip(pred_raw, 0.0, 1.5)
            pred_comp = predictive_weight * pred_raw

            combined = rye_comp + recency_comp + access_comp + d_comp + e_comp + pred_comp

            raw_scores.append(combined)
            objects.append(
                ScoredEntry(
                    entry_id=e["id"],
                    score=combined,
                    rye=rye,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=access_count,
                    tags=tags,
                    is_protected=is_protected,
                    rye_component=rye_comp,
                    recency_component=recency_comp,
                    access_component=access_comp,
                    discovery_component=d_comp,
                    equilibrium_component=e_comp,
                    predictive_component=pred_comp,
                    domain=domain,
                    goal=goal,
                    kind=kind,
                )
            )

        # Normalize scores to 0 to 1
        if raw_scores:
            lo, hi = min(raw_scores), max(raw_scores)
            span = max(hi - lo, 1e-9)
            for obj in objects:
                obj.score = (obj.score - lo) / span

        # Optional tagging of top core entries if memory_store supports it
        self._tag_core_entries(objects)

        return objects

    def _tag_core_entries(self, scored: List[ScoredEntry], top_fraction: float = 0.1) -> None:
        """
        Mark top fraction of entries as guardian_core if memory_store.update_entry is available.
        This lets the system slowly learn what forms the stable core of its memory.
        """
        if not hasattr(self.memory_store, "update_entry"):
            return

        if not scored:
            return

        scored_sorted = sorted(scored, key=lambda s: s.score, reverse=True)
        cutoff_index = max(1, int(len(scored_sorted) * top_fraction))
        core_entries = scored_sorted[:cutoff_index]

        for s in core_entries:
            try:
                self.memory_store.update_entry(
                    s.entry_id,
                    {
                        "meta": {
                            "add_tags": ["guardian_core"],
                            "last_core_marked_at": _utc_iso(),
                        }
                    },
                )
            except Exception:
                # Best effort only
                continue

    # ------------------------------------------------------------
    #   OPTIONAL CLUSTER COMPRESSION
    # ------------------------------------------------------------
    def _compress_stale_clusters(self, scored: List[ScoredEntry], drop_ids: List[str]) -> None:
        """
        Instead of dropping entire stale clusters, compress them:
            - Combine multiple stale notes into one compressed summary root.
            - Preserve high level metadata so cluster continuity is maintained.
        Priority for Tier 3 runs: keep conceptual continuity.

        Requires memory_store.update_entry or memory_store.add_entry to be useful.
        """

        if not drop_ids:
            return

        has_update = hasattr(self.memory_store, "update_entry")
        has_add = hasattr(self.memory_store, "add_entry")

        if not (has_update or has_add):
            return

        stale_group = [s for s in scored if s.entry_id in drop_ids]

        if len(stale_group) < 5:
            return

        merged_tags = list({t for s in stale_group for t in s.tags})
        avg_rye = None
        rye_vals = [float(s.rye) for s in stale_group if isinstance(s.rye, (int, float))]
        if rye_vals:
            avg_rye = sum(rye_vals) / len(rye_vals)

        cluster_note = {
            "compressed": True,
            "count": len(stale_group),
            "merged_tags": merged_tags,
            "avg_rye": avg_rye,
            "timestamp": _utc_iso(),
        }

        cluster_content = f"[Compressed cluster of {len(stale_group)} stale notes]"
        cluster_meta = {
            "tags": list(set(merged_tags + ["cluster_root"])),
            "created_at": _utc_iso(),
            "last_accessed": _utc_iso(),
            "extra": cluster_note,
        }

        if has_add:
            try:
                self.memory_store.add_entry(cluster_content, cluster_meta)
                return
            except Exception:
                pass

        if has_update:
            cluster_id = f"cluster_{_utc_iso().replace(':', '_')}"
            try:
                self.memory_store.update_entry(
                    cluster_id,
                    {
                        "content": cluster_content,
                        "meta": cluster_meta,
                    },
                )
            except Exception:
                pass

    # ------------------------------------------------------------
    #   DIAGNOSTICS
    # ------------------------------------------------------------
    def _compute_prune_diagnostics(
        self,
        scored: List[ScoredEntry],
        keep_ids: set,
        drop_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compute simple diagnostics about what was kept and dropped.
        Useful for dashboards and later analysis.
        """
        kept = [s for s in scored if s.entry_id in keep_ids]
        dropped = [s for s in scored if s.entry_id in drop_ids]

        def _avg_rye(vals: List[ScoredEntry]) -> Optional[float]:
            rye_vals = [float(s.rye) for s in vals if isinstance(s.rye, (int, float))]
            if not rye_vals:
                return None
            return sum(rye_vals) / len(rye_vals)

        def _tag_stats(vals: List[ScoredEntry]) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for s in vals:
                for t in s.tags:
                    counts[t] = counts.get(t, 0) + 1
            # Limit to top 15 tags for brevity
            return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15])

        return {
            "avg_rye_kept": _avg_rye(kept),
            "avg_rye_dropped": _avg_rye(dropped),
            "tag_stats_kept": _tag_stats(kept),
            "tag_stats_dropped": _tag_stats(dropped),
        }

    # ------------------------------------------------------------
    #   LOGGING
    # ------------------------------------------------------------
    def _write_header(self) -> None:
        PRUNE_LOG_PATH.write_text(
            "# Memory Pruning Log\n\n"
            "Tier 3 Autonomous Research Agent memory pruning history.\n\n"
            f"Pruner version: {PRUNER_VERSION}\n\n"
            "---\n\n",
            encoding="utf-8",
        )

    def _append_log(self, summary: Dict[str, Any]) -> None:
        lines = []
        ts = summary.get("timestamp", _utc_iso())
        lines.append(f"## Event {ts}")
        lines.append("")
        if self.run_id:
            lines.append(f"- Run ID: `{self.run_id}`")
        lines.append(f"- Pruner version: `{summary.get('pruner_version', PRUNER_VERSION)}`")
        lines.append(f"- Before: `{summary.get('total_before', summary.get('total', 0))}`")
        lines.append(f"- After: `{summary.get('total_after', summary.get('total', 0))}`")
        lines.append(f"- Dropped: `{summary.get('dropped', 0)}`")
        lines.append(f"- min_keep: `{summary.get('min_keep')}`")
        lines.append(f"- max_drop_fraction: `{summary.get('max_drop_fraction')}`")
        weights = summary.get("weights", {})
        if weights:
            lines.append("- Weights:")
            for k, v in weights.items():
                lines.append(f"  - {k}: `{v}`")
        lines.append("")
        lines.append("---\n")

        with PRUNE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _log_to_discovery_log(self, summary: Dict[str, Any]) -> None:
        """
        Optional integration with discovery_log for high level tracking.
        Uses injected discovery_logger if present, else falls back to get_global_logger().
        """
        logger = self.discovery_logger
        if logger is None:
            try:
                from .discovery_log import get_global_logger  # type: ignore[import]
                logger = get_global_logger(run_id=self.run_id)
            except Exception:
                return

        diag = summary.get("diagnostics", {}) or {}
        description_lines = [
            "Memory pruning pass completed.",
            "",
            f"Total before: {summary.get('total_before')}",
            f"Total after: {summary.get('total_after')}",
            f"Dropped: {summary.get('dropped')}",
            "",
            "Diagnostics:",
            f"- avg_rye_kept: {diag.get('avg_rye_kept')}",
            f"- avg_rye_dropped: {diag.get('avg_rye_dropped')}",
        ]
        description = "\n".join(description_lines)

        logger.log_event(
            kind="memory_prune",
            title="Memory pruning pass",
            description=description,
            cycle_index=None,
            agent_role="MemoryPruner",
            rye_before=diag.get("avg_rye_kept"),
            rye_after=diag.get("avg_rye_dropped"),
            delta_r=None,
            energy=None,
            tags=["memory_prune", "maintenance", "reparodynamics"],
            extra=summary,
        )
