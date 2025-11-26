# agent/replay_buffer.py

from __future__ import annotations

import uuid
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ReplayItem:
    """Single replay memory unit for longevity aware learning.

    This carries a compact but rich description of a move the agent made,
    the evidence it used, and the resulting RYE performance.

    Fields:
        item_id:
            Unique identifier for this replay item.
        hallmark:
            High level target such as "mitochondria" or "senescence".
        stage:
            TGRM stage that produced this item such as "idea" or "verify".
        mechanism_chain:
            Text or structured description of the mechanism path.
        biomarker_pattern:
            Optional dict of biomarker or lab signal snapshot.
        hypothesis_text:
            Core hypothesis text that this item represents.
        rye_score:
            RYE value associated with this move or hypothesis.
        energy_cost:
            Energy estimate used to obtain this item.
        decision:
            "accepted", "rejected", or "pending".
        decision_reason:
            Short explanation of why the decision was taken.
        source_citations:
            List of citation dicts that support this item.
        tags:
            Free form tags such as ["longevity", "stack", "synergy"].
        created_at:
            ISO timestamp when the item was created.
        run_id:
            Identifier for the run that produced this item.
        cycle_index:
            Cycle index in that run when the item was logged.
        metadata:
            Open slot for any extra fields from TGRM or CoreAgent.
    """

    item_id: str
    hallmark: Optional[str] = None
    stage: Optional[str] = None
    mechanism_chain: Optional[Any] = None
    biomarker_pattern: Optional[Dict[str, Any]] = None
    hypothesis_text: Optional[str] = None
    rye_score: Optional[float] = None
    energy_cost: Optional[float] = None
    decision: str = "pending"
    decision_reason: Optional[str] = None
    source_citations: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    run_id: Optional[str] = None
    cycle_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON serialisable dict."""
        data = asdict(self)
        return data

    def short_label(self) -> str:
        """Return a short label for UI views."""
        if self.hypothesis_text:
            txt = self.hypothesis_text.strip()
            if len(txt) > 140:
                return txt[:137] + "..."
            return txt
        if self.mechanism_chain:
            txt = str(self.mechanism_chain)
            if len(txt) > 140:
                return txt[:137] + "..."
            return txt
        return f"Replay item {self.item_id}"


class ReplayBuffer:
    """Longevity ready replay memory for fast learning.

    This buffer is designed as a side channel memory that TGRM and
    CoreAgent can use to:

        - store promising hypotheses and mechanism chains
        - reuse high RYE moves in later verify cycles
        - build per hallmark learning stats for reporting
        - expose "learned moves" and motifs to the UI

    Capacity is kept modest so full scans remain cheap even in Python
    while still supporting long runs.
    """

    def __init__(self, max_items: int = 2000) -> None:
        self.max_items = max_items
        self._items: List[ReplayItem] = []
        self._index_by_id: Dict[str, ReplayItem] = {}

    # ------------------------------------------------------------------
    # Core mutation methods
    # ------------------------------------------------------------------
    def add_item(self, item: ReplayItem) -> ReplayItem:
        """Add a fully constructed ReplayItem.

        Capacity is enforced by dropping the oldest items.
        """
        if item.item_id in self._index_by_id:
            # Replace in place if it already exists
            existing = self._index_by_id[item.item_id]
            # Keep created_at and id stable, update all other fields
            preserved_created = existing.created_at
            existing_dict = existing.to_dict()
            new_dict = item.to_dict()
            existing_dict.update(new_dict)
            existing_dict["created_at"] = preserved_created
            updated = ReplayItem(**existing_dict)
            self._index_by_id[item.item_id] = updated
            # Replace in list
            for i, it in enumerate(self._items):
                if it.item_id == item.item_id:
                    self._items[i] = updated
                    break
            return updated

        self._items.append(item)
        self._index_by_id[item.item_id] = item
        self._enforce_capacity()
        return item

    def create_item(
        self,
        hallmark: Optional[str] = None,
        stage: Optional[str] = None,
        mechanism_chain: Optional[Any] = None,
        biomarker_pattern: Optional[Dict[str, Any]] = None,
        hypothesis_text: Optional[str] = None,
        rye_score: Optional[float] = None,
        energy_cost: Optional[float] = None,
        decision: str = "pending",
        decision_reason: Optional[str] = None,
        source_citations: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        cycle_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> ReplayItem:
        """Convenience constructor plus add.

        This is the method TGRM can call directly.
        """
        replay_item = ReplayItem(
            item_id=item_id or uuid.uuid4().hex,
            hallmark=hallmark,
            stage=stage,
            mechanism_chain=mechanism_chain,
            biomarker_pattern=biomarker_pattern,
            hypothesis_text=hypothesis_text,
            rye_score=rye_score,
            energy_cost=energy_cost,
            decision=decision,
            decision_reason=decision_reason,
            source_citations=source_citations or [],
            tags=tags or [],
            run_id=run_id,
            cycle_index=cycle_index,
            metadata=metadata or {},
        )
        return self.add_item(replay_item)

    def record_from_cycle(
        self,
        *,
        run_id: str,
        cycle_index: int,
        hallmark: Optional[str],
        stage: Optional[str],
        hypothesis_text: Optional[str],
        mechanism_chain: Optional[Any],
        biomarker_pattern: Optional[Dict[str, Any]],
        rye_score: Optional[float],
        energy_cost: Optional[float],
        source_citations: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReplayItem:
        """High level helper to log a replay candidate from a cycle.

        Typical use from TGRM:
            buffer.record_from_cycle(
                run_id=current_run_id,
                cycle_index=cycle_index,
                hallmark=hallmark,
                stage=stage,
                hypothesis_text=hyp_text,
                mechanism_chain=mechanism_chain,
                biomarker_pattern=bio_snap,
                rye_score=rye_value,
                energy_cost=energy_e,
                source_citations=citations,
                tags=["candidate"],
                metadata={"goal": goal},
            )
        """
        return self.create_item(
            hallmark=hallmark,
            stage=stage,
            mechanism_chain=mechanism_chain,
            biomarker_pattern=biomarker_pattern,
            hypothesis_text=hypothesis_text,
            rye_score=rye_score,
            energy_cost=energy_cost,
            decision="pending",
            decision_reason=None,
            source_citations=source_citations,
            tags=tags,
            run_id=run_id,
            cycle_index=cycle_index,
            metadata=metadata,
        )

    def update_decision(
        self,
        item_id: str,
        decision: str,
        reason: Optional[str] = None,
    ) -> Optional[ReplayItem]:
        """Update decision status for an item.

        decision in {"accepted", "rejected", "pending"}.
        """
        item = self._index_by_id.get(item_id)
        if item is None:
            return None

        new_decision = decision.strip().lower()
        if new_decision not in {"accepted", "rejected", "pending"}:
            new_decision = "pending"

        updated = ReplayItem(
            item_id=item.item_id,
            hallmark=item.hallmark,
            stage=item.stage,
            mechanism_chain=item.mechanism_chain,
            biomarker_pattern=item.biomarker_pattern,
            hypothesis_text=item.hypothesis_text,
            rye_score=item.rye_score,
            energy_cost=item.energy_cost,
            decision=new_decision,
            decision_reason=reason or item.decision_reason,
            source_citations=list(item.source_citations),
            tags=list(item.tags),
            created_at=item.created_at,
            run_id=item.run_id,
            cycle_index=item.cycle_index,
            metadata=dict(item.metadata),
        )
        self._index_by_id[item_id] = updated
        for i, it in enumerate(self._items):
            if it.item_id == item_id:
                self._items[i] = updated
                break
        return updated

    def attach_tag(self, item_id: str, tag: str) -> Optional[ReplayItem]:
        """Attach a tag to an item if it is not already present."""
        item = self._index_by_id.get(item_id)
        if item is None:
            return None
        if tag in item.tags:
            return item
        new_tags = list(item.tags) + [tag]
        updated = ReplayItem(
            item_id=item.item_id,
            hallmark=item.hallmark,
            stage=item.stage,
            mechanism_chain=item.mechanism_chain,
            biomarker_pattern=item.biomarker_pattern,
            hypothesis_text=item.hypothesis_text,
            rye_score=item.rye_score,
            energy_cost=item.energy_cost,
            decision=item.decision,
            decision_reason=item.decision_reason,
            source_citations=list(item.source_citations),
            tags=new_tags,
            created_at=item.created_at,
            run_id=item.run_id,
            cycle_index=item.cycle_index,
            metadata=dict(item.metadata),
        )
        self._index_by_id[item_id] = updated
        for i, it in enumerate(self._items):
            if it.item_id == item_id:
                self._items[i] = updated
                break
        return updated

    def _enforce_capacity(self) -> None:
        """Keep only the newest max_items in the buffer."""
        extra = len(self._items) - self.max_items
        if extra <= 0:
            return
        # Drop from the front (oldest)
        to_drop = self._items[:extra]
        self._items = self._items[extra:]
        for it in to_drop:
            self._index_by_id.pop(it.item_id, None)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_item(self, item_id: str) -> Optional[ReplayItem]:
        """Return a single item."""
        return self._index_by_id.get(item_id)

    def items(self) -> List[ReplayItem]:
        """Return all items in insertion order (oldest first)."""
        return list(self._items)

    def recent_items(
        self,
        limit: int = 50,
        hallmark: Optional[str] = None,
        stage: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> List[ReplayItem]:
        """Return a list of recent items filtered by hallmark, stage, decision."""
        filtered = self._filter_items(
            items=reversed(self._items),
            hallmark=hallmark,
            stage=stage,
            decision=decision,
        )
        return list(filtered)[:limit]

    def top_by_rye(
        self,
        limit: int = 20,
        hallmark: Optional[str] = None,
        stage: Optional[str] = None,
        decision: Optional[str] = None,
        min_rye: Optional[float] = None,
    ) -> List[ReplayItem]:
        """Return best items by rye_score with optional filters."""
        candidates = list(
            self._filter_items(
                items=self._items,
                hallmark=hallmark,
                stage=stage,
                decision=decision,
            )
        )
        candidates = [c for c in candidates if c.rye_score is not None]
        if min_rye is not None:
            candidates = [c for c in candidates if c.rye_score is not None and c.rye_score >= min_rye]
        candidates.sort(key=lambda x: x.rye_score or 0.0, reverse=True)
        return candidates[:limit]

    def sample_for_verify(
        self,
        hallmark: Optional[str] = None,
        limit: int = 5,
    ) -> List[ReplayItem]:
        """Return pending high RYE items to feed into verify stage.

        Strategy:
            - filter for pending items
            - sort by rye_score descending and energy_cost ascending
        """
        candidates = self.top_by_rye(
            limit=len(self._items),
            hallmark=hallmark,
            stage=None,
            decision="pending",
        )
        candidates.sort(
            key=lambda x: (
                -(x.rye_score or 0.0),
                x.energy_cost if x.energy_cost is not None else float("inf"),
            )
        )
        return candidates[:limit]

    def statistics_by_hallmark(self) -> Dict[str, Dict[str, Any]]:
        """Return RYE and count stats per hallmark."""
        buckets: Dict[str, List[ReplayItem]] = {}
        for it in self._items:
            if not it.hallmark:
                continue
            h = it.hallmark
            buckets.setdefault(h, []).append(it)

        stats: Dict[str, Dict[str, Any]] = {}
        for hallmark, items in buckets.items():
            rye_vals = [i.rye_score for i in items if i.rye_score is not None]
            if rye_vals:
                avg = sum(rye_vals) / len(rye_vals)
                try:
                    med = statistics.median(rye_vals)
                except statistics.StatisticsError:
                    med = rye_vals[0]
            else:
                avg = None
                med = None

            accepted = len([i for i in items if i.decision == "accepted"])
            pending = len([i for i in items if i.decision == "pending"])
            rejected = len([i for i in items if i.decision == "rejected"])

            stats[hallmark] = {
                "count": len(items),
                "avg_rye": avg,
                "median_rye": med,
                "accepted": accepted,
                "pending": pending,
                "rejected": rejected,
                "top_examples": [
                    {
                        "item_id": it.item_id,
                        "label": it.short_label(),
                        "rye": it.rye_score,
                        "decision": it.decision,
                    }
                    for it in self.top_by_rye(hallmark=hallmark, limit=5)
                ],
            }

        return stats

    def statistics_global(self) -> Dict[str, Any]:
        """Return global replay stats suitable for a diagnostics panel."""
        total = len(self._items)
        if total == 0:
            return {
                "total_items": 0,
                "avg_rye": None,
                "median_rye": None,
                "accepted": 0,
                "pending": 0,
                "rejected": 0,
            }

        rye_vals = [i.rye_score for i in self._items if i.rye_score is not None]
        if rye_vals:
            avg = sum(rye_vals) / len(rye_vals)
            try:
                med = statistics.median(rye_vals)
            except statistics.StatisticsError:
                med = rye_vals[0]
        else:
            avg = None
            med = None

        accepted = len([i for i in self._items if i.decision == "accepted"])
        pending = len([i for i in self._items if i.decision == "pending"])
        rejected = len([i for i in self._items if i.decision == "rejected"])

        return {
            "total_items": total,
            "avg_rye": avg,
            "median_rye": med,
            "accepted": accepted,
            "pending": pending,
            "rejected": rejected,
        }

    def motif_summary(
        self,
        hallmark: Optional[str] = None,
        min_count: int = 2,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Extract repeated patterns that look like learned moves.

        Simple heuristic:
            - use lowercased hypothesis_text as key
            - group by exact match
            - keep entries with count >= min_count
            - sort by combined rye_score
        """
        groups: Dict[str, List[ReplayItem]] = {}
        for it in self._items:
            if hallmark and it.hallmark != hallmark:
                continue
            if not it.hypothesis_text:
                continue
            key = it.hypothesis_text.strip().lower()
            groups.setdefault(key, []).append(it)

        motifs: List[Dict[str, Any]] = []
        for key, items in groups.items():
            if len(items) < min_count:
                continue
            rye_vals = [i.rye_score for i in items if i.rye_score is not None]
            total_rye = sum(rye_vals) if rye_vals else 0.0
            avg_rye = total_rye / len(rye_vals) if rye_vals else None
            motifs.append(
                {
                    "pattern": key[:200],
                    "count": len(items),
                    "avg_rye": avg_rye,
                    "example_item_id": items[0].item_id,
                }
            )

        motifs.sort(
            key=lambda m: (
                -(m["avg_rye"] or 0.0),
                -m["count"],
            )
        )
        return motifs[:limit]

    # ------------------------------------------------------------------
    # Export helpers for reports and UI
    # ------------------------------------------------------------------
    def build_report_view(self) -> Dict[str, Any]:
        """Return a structured snapshot for the reporting layer.

        This is intended for report_generator.build_hallmark_summary and
        replay insights sections.
        """
        hallmark_stats = self.statistics_by_hallmark()
        global_stats = self.statistics_global()
        motifs = self.motif_summary()

        top_items_global = [
            {
                "item_id": it.item_id,
                "label": it.short_label(),
                "hallmark": it.hallmark,
                "stage": it.stage,
                "rye": it.rye_score,
                "decision": it.decision,
            }
            for it in self.top_by_rye(limit=15)
        ]

        return {
            "global_stats": global_stats,
            "hallmark_stats": hallmark_stats,
            "motifs": motifs,
            "top_items_global": top_items_global,
        }

    def build_ui_view(
        self,
        hallmark: Optional[str] = None,
        limit: int = 30,
    ) -> Dict[str, Any]:
        """Compact view for Streamlit panels and dashboards."""
        recent = self.recent_items(limit=limit, hallmark=hallmark)
        top = self.top_by_rye(limit=min(10, limit), hallmark=hallmark)
        motifs = self.motif_summary(hallmark=hallmark, limit=10)

        recent_view = [
            {
                "item_id": it.item_id,
                "label": it.short_label(),
                "hallmark": it.hallmark,
                "stage": it.stage,
                "rye": it.rye_score,
                "decision": it.decision,
                "created_at": it.created_at,
            }
            for it in recent
        ]

        top_view = [
            {
                "item_id": it.item_id,
                "label": it.short_label(),
                "hallmark": it.hallmark,
                "stage": it.stage,
                "rye": it.rye_score,
                "decision": it.decision,
            }
            for it in top
        ]

        return {
            "hallmark": hallmark,
            "recent": recent_view,
            "top": top_view,
            "motifs": motifs,
        }

    # ------------------------------------------------------------------
    # Internal filter helper
    # ------------------------------------------------------------------
    def _filter_items(
        self,
        items: Iterable[ReplayItem],
        hallmark: Optional[str],
        stage: Optional[str],
        decision: Optional[str],
    ) -> Iterable[ReplayItem]:
        decision_norm = decision.strip().lower() if decision else None
        for it in items:
            if hallmark and it.hallmark != hallmark:
                continue
            if stage and (it.stage or "").lower() != stage.lower():
                continue
            if decision_norm and (it.decision or "").lower() != decision_norm:
                continue
            yield it
