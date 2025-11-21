"""Memory storage for the autonomous research agent.

This module provides a JSON-based persistent storage layer. It stores:
- notes
- hypotheses
- citations
- biomarker summaries (future use)
- cycle logs

The memory store acts as a lightweight knowledge base for the agent and is
referenced each cycle to retrieve prior context and to persist new findings.

Reparodynamics interpretation:
    MemoryStore is the long-term substrate where repairs accumulate.
    Each TGRM cycle writes its improvements here, and future cycles read
    from it to reduce energy cost. When combined with VectorMemory, this
    becomes a semantic, time-aware repair substrate.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# Optional vector memory integration
try:
    from .vector_memory import VectorMemory  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    VectorMemory = None  # type: ignore[assignment]


class MemoryStore:
    """A lightweight persistent memory store using a JSON file.

    The JSON file is structured with several top-level keys:
        - "notes":       free-form text notes with metadata
        - "cycles":      logs of each research cycle
        - "hypotheses":  generated hypotheses for each goal
        - "citations":   structured citation objects from web/papers
        - "biomarkers":  placeholder for anti-aging / lab data

    In-memory (non-persistent) vector memory may also be attached to
    support semantic search and time-decayed retrieval if the optional
    `VectorMemory` class is available.
    """

    def __init__(self, memory_file: str) -> None:
        self.memory_file = memory_file
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        # Core JSON-backed data
        self._data: Dict[str, Any] = {
            "notes": [],
            "cycles": [],
            "hypotheses": [],
            "citations": [],
            "biomarkers": [],
        }
        self._load()
        self._ensure_keys()

        # Optional vector memory for semantic, time-decayed retrieval
        if VectorMemory is not None:
            try:
                self.vector_memory: Optional[VectorMemory] = VectorMemory()
            except Exception:
                self.vector_memory = None
        else:
            self.vector_memory = None

    # ------------------------------------------------------------------
    # Internal JSON persistence
    # ------------------------------------------------------------------
    def _ensure_keys(self) -> None:
        """Ensure all expected top-level keys exist in _data."""
        for key, default in [
            ("notes", []),
            ("cycles", []),
            ("hypotheses", []),
            ("citations", []),
            ("biomarkers", []),
        ]:
            if key not in self._data or not isinstance(self._data.get(key), list):
                self._data[key] = default

    def _load(self) -> None:
        """Load memory from disk if the file exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # If the file is corrupt or cannot be read, start fresh
                self._data = {
                    "notes": [],
                    "cycles": [],
                    "hypotheses": [],
                    "citations": [],
                    "biomarkers": [],
                }

    def _save(self) -> None:
        """Persist memory to disk."""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    def add_note(
        self,
        goal: str,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        role: str = "agent",
        importance: float = 1.0,
    ) -> None:
        """Add a note associated with a research goal.

        Args:
            goal:
                Research goal or topic this note belongs to.
            content:
                Free-form text of the note.
            tags:
                Optional list of tags (e.g., ["reparodynamics", "hypothesis"]).
            role:
                Logical role that produced the note (agent, researcher,
                critic, etc.).
            importance:
                A rough importance score. This can be used in the future
                for prioritised retrieval.
        """
        note = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "content": content,
            "tags": tags or [],
            "role": role,
            "importance": float(importance),
        }
        self._data.setdefault("notes", []).append(note)
        self._save()

        # Also store in vector memory if available, for semantic retrieval
        if self.vector_memory is not None and content:
            try:
                meta = {
                    "goal": goal,
                    "tags": tags or [],
                    "role": role,
                    "importance": float(importance),
                }
                self.vector_memory.add_item(text=content, metadata=meta)
            except Exception:
                # Vector memory is optional; ignore failures
                pass

    def get_notes(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve notes.

        Args:
            goal:
                If provided, filter notes by this goal. If None, return
                all notes.
        """
        notes = self._data.get("notes", [])
        if goal is None:
            return list(notes)
        return [n for n in notes if n.get("goal") == goal]

    def search_notes(self, query: str, top_k: int = 5, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search notes either via vector memory (if available) or simple keyword search.

        Args:
            query:
                Text query to search for.
            top_k:
                Maximum number of results to return.
            goal:
                Optional goal filter.

        Returns:
            List of note dicts, best-matching first.
        """
        if not query:
            return []

        # If vector memory is present, use it
        if self.vector_memory is not None:
            try:
                items = self.vector_memory.search(query=query, top_k=top_k)
                # Each item.metadata should contain at least "goal" and "content" keys
                results: List[Dict[str, Any]] = []
                for it in items:
                    meta = dict(it.metadata)
                    # Reconstruct minimal note-like structure
                    meta.setdefault("content", it.text)
                    meta.setdefault("timestamp", it.timestamp.isoformat() + "Z")
                    if goal is None or meta.get("goal") == goal:
                        results.append(meta)
                return results
            except Exception:
                # Fall back to keyword search if vector memory fails
                pass

        # Keyword-based fallback
        notes = self.get_notes(goal)
        query_lower = query.lower()
        matched = [n for n in notes if query_lower in str(n.get("content", "")).lower()]
        return matched[:top_k]

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------
    def add_hypothesis(
        self,
        goal: str,
        text: str,
        *,
        score: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a generated hypothesis for a given goal."""
        hyp = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "text": text,
            "score": float(score) if score is not None else None,
            "tags": tags or [],
        }
        self._data.setdefault("hypotheses", []).append(hyp)
        self._save()

        # Hypotheses are also valuable semantic memory
        if self.vector_memory is not None and text:
            try:
                meta = {
                    "goal": goal,
                    "type": "hypothesis",
                    "score": hyp["score"],
                    "tags": hyp["tags"],
                }
                self.vector_memory.add_item(text=text, metadata=meta)
            except Exception:
                pass

    def get_hypotheses(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored hypotheses, optionally filtered by goal."""
        hyps = self._data.get("hypotheses", [])
        if goal is None:
            return list(hyps)
        return [h for h in hyps if h.get("goal") == goal]

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------
    def add_citation(self, goal: str, citation: Dict[str, Any]) -> None:
        """Add a structured citation object."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "citation": citation,
        }
        self._data.setdefault("citations", []).append(entry)
        self._save()

        if self.vector_memory is not None:
            try:
                text_parts = [
                    str(citation.get("title", "")),
                    str(citation.get("snippet", "")),
                    str(citation.get("url", "")),
                ]
                text = " ".join([p for p in text_parts if p])
                meta = {
                    "goal": goal,
                    "type": "citation",
                    "source": citation.get("source", "web"),
                    "url": citation.get("url", ""),
                }
                if text:
                    self.vector_memory.add_item(text=text, metadata=meta)
            except Exception:
                pass

    def get_citations(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored citations, optionally filtered by goal."""
        entries = self._data.get("citations", [])
        if goal is None:
            return list(entries)
        return [e for e in entries if e.get("goal") == goal]

    # ------------------------------------------------------------------
    # Biomarkers (placeholder for anti-aging / lab data)
    # ------------------------------------------------------------------
    def add_biomarker_snapshot(self, goal: str, data: Dict[str, Any]) -> None:
        """Store a biomarker snapshot (anti-aging / health metrics)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "data": data,
        }
        self._data.setdefault("biomarkers", []).append(entry)
        self._save()

    def get_biomarker_history(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve biomarker history, optionally filtered by goal."""
        entries = self._data.get("biomarkers", [])
        if goal is None:
            return list(entries)
        return [e for e in entries if e.get("goal") == goal]

    # ------------------------------------------------------------------
    # Cycle logs
    # ------------------------------------------------------------------
    def log_cycle(self, cycle_data: Dict[str, Any]) -> None:
        """Append cycle data to the cycle log."""
        self._data.setdefault("cycles", []).append(cycle_data)
        self._save()

    def get_cycle_history(self) -> List[Dict[str, Any]]:
        """Return the history of cycles."""
        return self._data.get("cycles", [])
