"""Vector-based long-term memory with optional time decay.

This module implements a small, self-contained vector memory for the
autonomous research agent. It is intentionally lightweight:

- If `sentence-transformers` and `numpy` are available, it uses embeddings.
- If embeddings are not available, it falls back to simple keyword search.
- It applies a simple time-decay weighting so older items slowly lose
  influence compared to recent ones.
- It respects importance and basic metadata filters, making it swarm and
  meta-controller (Option C) aware.

Reparodynamics interpretation:
    VectorMemory is the substrate where "repair" accumulates. High-quality
    notes, hypotheses, citations, and discoveries are stored here so that
    future TGRM cycles can retrieve them efficiently with minimal extra
    energy. Importance, time decay, and metadata filters act as a RYE-aware
    routing surface for long 24 to 90 day missions and multi-phase runs.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

# Optional embedding model
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass
class MemoryItem:
    """Single item stored in vector memory."""

    text: str
    metadata: Dict[str, Any]
    timestamp: _dt.datetime = field(default_factory=lambda: _dt.datetime.utcnow())
    vector: Optional[Any] = None  # numpy array or list, depending on backend


class VectorMemory:
    """Small vector memory with optional embeddings and time decay.

    This class is designed to be safe for long autonomous runs:
        - Bounded memory size (max_items)
        - Importance-aware scoring
        - Time-decayed relevance
        - Simple metadata filters for goal, role, domain, kind/type
        - Optional meta filters for phase and segment_index (Option C)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        decay_half_life_days: float = 30.0,
        max_items: int = 5000,
    ) -> None:
        """Initialise the vector memory.

        Args:
            model_name:
                Name of the sentence-transformers model to use.
            decay_half_life_days:
                Half-life for time decay. After this many days, a memory
                item's weight is reduced by half.
            max_items:
                Maximum number of items to keep in memory. When exceeded,
                the oldest items are trimmed first.
        """
        self.items: List[MemoryItem] = []
        self.decay_half_life_days = float(decay_half_life_days)
        self.max_items = int(max_items)

        # Lazy model loading to avoid heavy startup cost if embeddings are not needed
        self._model_name: Optional[str] = model_name if SentenceTransformer is not None and np is not None else None
        self._model: Optional[Any] = None
        self._model_loaded: bool = False

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _now(self) -> _dt.datetime:
        return _dt.datetime.utcnow()

    def _ensure_model(self) -> None:
        """Lazy load the embedding model if possible."""
        if self._model_loaded:
            return
        self._model_loaded = True
        if self._model_name is None or SentenceTransformer is None or np is None:
            self._model = None
            return
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._model = None

    def _time_decay_weight(self, ts: _dt.datetime) -> float:
        """Compute time-decay weight for a timestamp.

        Weight = 0.5 ** (age_days / half_life)
        """
        age_days = (self._now() - ts).total_seconds() / 86400.0
        if age_days <= 0 or self.decay_half_life_days <= 0:
            return 1.0
        return 0.5 ** (age_days / self.decay_half_life_days)

    def _importance_weight(self, meta: Dict[str, Any]) -> float:
        """Extract importance weight from metadata (default 1.0, clamped)."""
        val = meta.get("importance", 1.0)
        try:
            w = float(val)
        except Exception:
            w = 1.0
        if not math.isfinite(w):
            w = 1.0
        # Clamp to a reasonable range so it cannot explode scores
        return max(0.1, min(w, 5.0))

    def _embed(self, text: str) -> Optional[Any]:
        """Compute embedding for text if model is available."""
        if not text or np is None:
            return None
        self._ensure_model()
        if self._model is None:
            return None
        try:
            vec = self._model.encode([text])[0]
            return np.asarray(vec, dtype=float)
        except Exception:
            return None

    def _match_filters(
        self,
        item: MemoryItem,
        run_id: Optional[str],
        goal: Optional[str],
        role: Optional[str],
        domain: Optional[str],
        kind: Optional[str],
        phase: Optional[str],
        segment_index: Optional[int],
    ) -> bool:
        """Return True if the item passes the metadata filters."""
        meta = item.metadata or {}

        # Run isolation filter (prevents cross-run blending when goals are similar)
        if run_id is not None:
            meta_run_id = meta.get("run_id")
            if meta_run_id is None:
                rm = meta.get("run_metadata") or {}
                if isinstance(rm, dict):
                    meta_run_id = rm.get("run_id")
            try:
                meta_run_id_str = str(meta_run_id) if meta_run_id is not None else None
            except Exception:
                meta_run_id_str = None
            if meta_run_id_str != str(run_id):
                return False


        if goal is not None and meta.get("goal") != goal:
            return False
        if role is not None and meta.get("role") != role:
            return False
        if domain is not None and meta.get("domain") != domain:
            return False

        if kind is not None:
            # Accept either explicit "kind" or "type" matches
            k1 = meta.get("kind")
            k2 = meta.get("type")
            if k1 != kind and k2 != kind:
                return False

        # Meta-controller phase filter (Option C)
        if phase is not None:
            # Allow either direct metadata["phase"] or nested run_metadata["phase"]
            meta_phase = meta.get("phase")
            if meta_phase is None:
                rm = meta.get("run_metadata") or {}
                if isinstance(rm, dict):
                    meta_phase = rm.get("phase")
            if meta_phase != phase:
                return False

        # Meta-controller segment filter
        if segment_index is not None:
            # Accept either top level segment_index or nested
            seg = meta.get("segment_index")
            if seg is None:
                rm = meta.get("run_metadata") or {}
                if isinstance(rm, dict):
                    seg = rm.get("segment_index")
            try:
                if seg is None or int(seg) != int(segment_index):
                    return False
            except Exception:
                return False

        return True

    def _trim_if_needed(self) -> None:
        """Keep total items under max_items by trimming oldest first."""
        if self.max_items <= 0:
            return
        if len(self.items) <= self.max_items:
            return
        # Sort by timestamp ascending and keep the most recent max_items
        self.items.sort(key=lambda it: it.timestamp)
        self.items = self.items[-self.max_items :]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_item(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add a new item to memory, optionally with an embedding."""
        if not text:
            return
        vec = self._embed(text)
        item = MemoryItem(text=text, metadata=metadata or {}, vector=vec)
        self.items.append(item)
        self._trim_if_needed()

    def add_structured_note(
        self,
        text: str,
        *,
        run_id: Optional[str] = None,
        goal: Optional[str] = None,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        kind: Optional[str] = None,
        importance: float = 1.0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Helper to add a note with common fields wired in one call.

        This is a thin wrapper over add_item and does not change behavior
        for existing callers that use add_item directly.
        """
        meta: Dict[str, Any] = dict(extra_metadata or {})
        if run_id is not None:
            meta.setdefault("run_id", run_id)
        if goal is not None:
            meta.setdefault("goal", goal)
        if role is not None:
            meta.setdefault("role", role)
        if domain is not None:
            meta.setdefault("domain", domain)
        if kind is not None:
            meta.setdefault("kind", kind)
        meta.setdefault("importance", importance)
        self.add_item(text=text, metadata=meta)

    def _similarity(self, v1: Any, v2: Any) -> float:
        """Cosine similarity between two vectors, with safety checks."""
        if np is None:
            return 0.0
        try:
            v1 = np.asarray(v1, dtype=float)
            v2 = np.asarray(v2, dtype=float)
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom == 0:
                return 0.0
            return float(np.dot(v1, v2) / denom)
        except Exception:
            return 0.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        run_id: Optional[str] = None,
        goal: Optional[str] = None,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        kind: Optional[str] = None,
        phase: Optional[str] = None,
        segment_index: Optional[int] = None,
    ) -> List[MemoryItem]:
        """Search memory with a query string.

        If embeddings are available, we use cosine similarity + time decay
        + importance weighting. If not, we fall back to simple keyword
        frequency matching with time decay and importance.

        Args:
            query:
                Natural language query.
            top_k:
                Maximum number of results to return.
            goal, role, domain, kind:
                Optional metadata filters. If provided, only items whose
                metadata match these fields are considered.
            phase:
                Optional meta-controller phase filter
                (for example "exploration", "stabilization", "refinement").
            segment_index:
                Optional segment index filter from the meta controller.

        Returns:
            List[MemoryItem]: up to top_k best-matching items.
        """
        if not query or not self.items:
            return []

        # Pre-filter based on metadata so we do not score unrelated items
        candidate_items: List[MemoryItem] = []
        for item in self.items:
            if self._match_filters(
                item,
                run_id=run_id,
                goal=goal,
                role=role,
                domain=domain,
                kind=kind,
                phase=phase,
                segment_index=segment_index,
            ):
                candidate_items.append(item)

        if not candidate_items:
            return []

        # If we can embed, use vector search with decay and importance weighting
        q_vec = self._embed(query)
        if q_vec is not None:
            scored: List[Tuple[float, MemoryItem]] = []
            for item in candidate_items:
                if item.vector is None:
                    continue
                sim = self._similarity(q_vec, item.vector)
                if sim <= 0:
                    continue
                decay = self._time_decay_weight(item.timestamp)
                importance = self._importance_weight(item.metadata or {})
                score = sim * decay * importance
                scored.append((score, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for (score, it) in scored[:top_k] if score > 0]

        # Fallback: simple keyword matching if no embeddings
        q_lower = query.lower()
        scored_kw: List[Tuple[float, MemoryItem]] = []
        for item in candidate_items:
            text_lower = item.text.lower()
            hits = text_lower.count(q_lower)
            if hits <= 0:
                continue
            decay = self._time_decay_weight(item.timestamp)
            importance = self._importance_weight(item.metadata or {})
            score = float(hits) * decay * importance
            scored_kw.append((score, item))

        scored_kw.sort(key=lambda x: x[0], reverse=True)
        return [it for (score, it) in scored_kw[:top_k] if score > 0]

    # ------------------------------------------------------------------
    # Introspection helpers (optional but useful for diagnostics)
    # ------------------------------------------------------------------
    def size(self) -> int:
        """Return current number of items stored."""
        return len(self.items)

    def clear(self) -> None:
        """Remove all items. Use with care in long runs."""
        self.items.clear()

    def clear_for_goal(self, goal: str) -> int:
        """Remove items matching a given goal. Returns number removed."""
        if not goal:
            return 0
        before = len(self.items)
        self.items = [it for it in self.items if (it.metadata or {}).get("goal") != goal]
        return before - len(self.items)

    def dump_metadata_snapshot(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return a lightweight snapshot of recent items' metadata.

        Useful for debugging or UI views. No embeddings are exposed.
        """
        recent = self.items[-limit:]
        out: List[Dict[str, Any]] = []
        for it in recent:
            out.append(
                {
                    "timestamp": it.timestamp.isoformat(),
                    "metadata": dict(it.metadata or {}),
                    "text_preview": (it.text[:200] + "...") if len(it.text) > 200 else it.text,
                }
            )
        return out
