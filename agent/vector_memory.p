"""Vector-based long-term memory with optional time decay.

This module implements a small, self-contained vector memory for the
autonomous research agent. It is intentionally lightweight:

- If `sentence-transformers` and `numpy` are available, it uses embeddings.
- If embeddings are not available, it falls back to simple keyword search.
- It applies a simple time-decay weighting so older items slowly lose
  influence compared to recent ones.

Reparodynamics interpretation:
    VectorMemory is the substrate where "repair" accumulates. High-quality
    notes, hypotheses, and citations are stored here so that future TGRM
    cycles can retrieve them efficiently with minimal extra energy.
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
    """Small vector memory with optional embeddings and time decay."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        decay_half_life_days: float = 30.0,
    ) -> None:
        """Initialise the vector memory.

        Args:
            model_name:
                Name of the sentence-transformers model to use.
            decay_half_life_days:
                Half-life for time decay. After this many days, a memory
                item's weight is reduced by half.
        """
        self.items: List[MemoryItem] = []
        self.decay_half_life_days = float(decay_half_life_days)

        self._model = None
        if SentenceTransformer is not None and np is not None:
            try:
                self._model = SentenceTransformer(model_name)
            except Exception:
                self._model = None

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _now(self) -> _dt.datetime:
        return _dt.datetime.utcnow()

    def _time_decay_weight(self, ts: _dt.datetime) -> float:
        """Compute time-decay weight for a timestamp.

        Weight = 0.5 ** (age_days / half_life)
        """
        age_days = (self._now() - ts).total_seconds() / 86400.0
        if age_days <= 0 or self.decay_half_life_days <= 0:
            return 1.0
        return 0.5 ** (age_days / self.decay_half_life_days)

    def _embed(self, text: str) -> Optional[Any]:
        """Compute embedding for text if model is available."""
        if not text or self._model is None or np is None:
            return None
        try:
            vec = self._model.encode([text])[0]
            return np.asarray(vec, dtype=float)
        except Exception:
            return None

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

    def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Search memory with a query string.

        If embeddings are available, we use cosine similarity + time decay.
        If not, we fall back to simple keyword frequency matching.

        Returns:
            List[MemoryItem]: up to top_k best-matching items.
        """
        if not query or not self.items:
            return []

        # If we can embed, use vector search with decay weighting
        q_vec = self._embed(query)
        if q_vec is not None:
            scored: List[Tuple[float, MemoryItem]] = []
            for item in self.items:
                if item.vector is None:
                    continue
                sim = self._similarity(q_vec, item.vector)
                decay = self._time_decay_weight(item.timestamp)
                score = sim * decay
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for (score, it) in scored[:top_k] if score > 0]

        # Fallback: simple keyword matching if no embeddings
        q_lower = query.lower()
        scored_kw: List[Tuple[float, MemoryItem]] = []
        for item in self.items:
            text_lower = item.text.lower()
            hits = text_lower.count(q_lower)
            if hits <= 0:
                continue
            decay = self._time_decay_weight(item.timestamp)
            score = float(hits) * decay
            scored_kw.append((score, item))

        scored_kw.sort(key=lambda x: x[0], reverse=True)
        return [it for (score, it) in scored_kw[:top_k] if score > 0]
