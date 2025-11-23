"""
memory_pruner.py

Memory pruning and compaction for the Autonomous Research Agent.

Goal
    Keep long run memory useful and compact by removing low value,
    rarely used, and stale entries while preserving the highest
    RYE and most recently relevant items.

This module is written to be ADAPTABLE to your existing memory store.

Assumed memory_store interface (you can wrap your own class to match):

    memory_store.list_entries() -> List[Dict[str, Any]]
        Each entry dict should contain at least:
            - "id": unique identifier (str)
            - "content": text or payload (str)
            - "meta": dict with optional fields:
                - "rye": float
                - "created_at": ISO string
                - "last_accessed": ISO string
                - "access_count": int
                - "tags": list of str

    memory_store.delete_entries(ids: List[str]) -> None

You can add a thin adapter around your real MemoryStore or VectorMemory
to satisfy this interface.

Pruning strategy
    1. Read all entries.
    2. Compute a score per entry based on:
           - RYE (higher is better)
           - recency of last access
           - access count
    3. Sort by score descending.
    4. Keep the top N entries (min_keep) and drop some fraction of the rest.
    5. Log the pruning summary into:
           logs/memory_pruning_log.md

Typical usage:

    from agent.memory_pruner import MemoryPruner
    from agent.memory_store import MemoryStore   # your existing class

    store = MemoryStore(...)
    pruner = MemoryPruner(store, run_id="run_001")

    pruner.prune(
        min_keep=1000,
        max_drop_fraction=0.3
    )

Run this periodically from engine_worker.py (for example, once per month
or after a fixed number of cycles).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRUNE_LOG_PATH = LOG_DIR / "memory_pruning_log.md"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Accept both with and without timezone
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass
class ScoredEntry:
    entry_id: str
    score: float
    rye: Optional[float]
    created_at: Optional[datetime]
    last_accessed: Optional[datetime]
    access_count: Optional[int]
    tags: List[str]


class MemoryPruner:
    """
    MemoryPruner orchestrates scoring and pruning of memory entries.

    It does NOT implement storage itself. It relies on a provided
    memory_store with a simple interface:

        list_entries() -> List[Dict[str, Any]]
        delete_entries(ids: List[str]) -> None

    You can wrap your MemoryStore or VectorMemory in a small adapter
    if needed.
    """

    def __init__(
        self,
        memory_store: Any,
        run_id: Optional[str] = None,
    ) -> None:
        self.memory_store = memory_store
        self.run_id = run_id

        if not PRUNE_LOG_PATH.exists():
            self._write_header()

    # --------------- public API ---------------

    def prune(
        self,
        min_keep: int = 1000,
        max_drop_fraction: float = 0.3,
        rye_weight: float = 0.5,
        recency_weight: float = 0.3,
        access_weight: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Perform a pruning pass.

        Args:
            min_keep:
                Minimum number of entries to keep, no matter what.
            max_drop_fraction:
                Maximum fraction of total entries that can be dropped
                in one pruning pass. For example 0.3 means at most
                30 percent of entries are removed.
            rye_weight, recency_weight, access_weight:
                Relative weights for the scoring function.

        Returns:
            A summary dict with counts and thresholds used.
        """
        entries = self.memory_store.list_entries()
        total = len(entries)
        if total <= min_keep:
            summary = {
                "timestamp": _utc_iso(),
                "run_id": self.run_id,
                "total_entries": total,
                "dropped": 0,
                "reason": "below_min_keep",
            }
            self._append_log(summary)
            return summary

        scored = self._score_entries(
            entries,
            rye_weight=rye_weight,
            recency_weight=recency_weight,
            access_weight=access_weight,
        )

        # Entries sorted by score descending (keep best first)
        scored.sort(key=lambda x: x.score, reverse=True)

        # Determine how many we are allowed to drop
        max_drop = int(math.floor(total * max_drop_fraction))
        keep_count = max(min_keep, total - max_drop)
        keep_count = min(keep_count, total)

        keep_ids = {se.entry_id for se in scored[:keep_count]}
        drop_ids = [se.entry_id for se in scored[keep_count:]]

        if drop_ids:
            self.memory_store.delete_entries(drop_ids)

        summary = {
            "timestamp": _utc_iso(),
            "run_id": self.run_id,
            "total_entries_before": total,
            "total_entries_after": total - len(drop_ids),
            "dropped": len(drop_ids),
            "min_keep": min_keep,
            "max_drop_fraction": max_drop_fraction,
            "scoring_weights": {
                "rye": rye_weight,
                "recency": recency_weight,
                "access": access_weight,
            },
        }
        self._append_log(summary)
        return summary

    # --------------- scoring ---------------

    def _score_entries(
        self,
        entries: List[Dict[str, Any]],
        rye_weight: float,
        recency_weight: float,
        access_weight: float,
    ) -> List[ScoredEntry]:
        """
        Compute a score for each entry based on RYE, recency, and access.

        The score is normalized between 0 and 1 for each component and
        then combined using the provided weights.
        """
        scored: List[ScoredEntry] = []

        # Extract raw values
        rye_vals: List[float] = []
        recency_vals: List[float] = []
        access_vals: List[float] = []

        now = datetime.now(timezone.utc)

        for e in entries:
            meta = e.get("meta", {}) or {}
            rye = meta.get("rye")
            created_at = _parse_ts(meta.get("created_at"))
            last_accessed = _parse_ts(meta.get("last_accessed"))
            access_count = meta.get("access_count")

            # Compute recency in days (more recent = smaller delta)
            ref_time = last_accessed or created_at or now
            age_days = max((now - ref_time).total_seconds() / 86400.0, 0.0)
            recency_score = -age_days  # smaller age -> higher score

            rye_vals.append(float(rye) if rye is not None else 0.0)
            recency_vals.append(recency_score)
            access_vals.append(float(access_count) if access_count is not None else 0.0)

            scored.append(
                ScoredEntry(
                    entry_id=str(e.get("id")),
                    score=0.0,  # filled later
                    rye=rye if rye is not None else None,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=access_count if access_count is not None else None,
                    tags=list(meta.get("tags", []) or []),
                )
            )

        # Normalization helpers
        def normalize(values: List[float]) -> List[float]:
            if not values:
                return []
            vmin = min(values)
            vmax = max(values)
            if vmax <= vmin:
                return [0.0 for _ in values]
            return [(v - vmin) / (vmax - vmin) for v in values]

        rye_norm = normalize(rye_vals)
        recency_norm = normalize(recency_vals)
        access_norm = normalize(access_vals)

        for i, se in enumerate(scored):
            score = (
                rye_weight * rye_norm[i]
                + recency_weight * recency_norm[i]
                + access_weight * access_norm[i]
            )
            se.score = float(score)

        return scored

    # --------------- logging ---------------

    def _write_header(self) -> None:
        lines: List[str] = []
        lines.append("# Memory Pruning Log")
        lines.append("")
        lines.append("This file records pruning events for the Autonomous Research Agent.")
        lines.append("Each entry includes counts, thresholds, and weights used.")
        lines.append("")
        lines.append("---")
        lines.append("")
        PRUNE_LOG_PATH.write_text("\n".join(lines), encoding="utf-8")

    def _append_log(self, summary: Dict[str, Any]) -> None:
        """
        Append a pruning summary as a Markdown block.
        """
        lines: List[str] = []
        lines.append(f"## Pruning event at {summary.get('timestamp', _utc_iso())}")
        lines.append("")
        if self.run_id:
            lines.append(f"- Run ID: `{self.run_id}`")
        lines.append(f"- Total before: `{summary.get('total_entries_before', summary.get('total_entries'))}`")
        lines.append(f"- Total after: `{summary.get('total_entries_after', summary.get('total_entries'))}`")
        lines.append(f"- Dropped: `{summary.get('dropped', 0)}`")
        lines.append(f"- Min keep: `{summary.get('min_keep', 'n/a')}`")
        lines.append(f"- Max drop fraction: `{summary.get('max_drop_fraction', 'n/a')}`")

        weights = summary.get("scoring_weights", {})
        if weights:
            lines.append(f"- Scoring weights: rye={weights.get('rye')}, recency={weights.get('recency')}, access={weights.get('access')}")

        reason = summary.get("reason")
        if reason:
            lines.append(f"- Reason: `{reason}`")

        lines.append("")
        lines.append("---")
        lines.append("")

        with PRUNE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))
