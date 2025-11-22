"""Memory storage for the autonomous research agent.

This module provides a JSON-based persistent storage layer. It stores:
- notes
- hypotheses
- citations
- biomarker summaries (future use)
- cycle logs
- run_state metadata (for long continuous runs and auto resume)
- watchdog information (heartbeats for crash diagnostics)
- goal_index summaries (lightweight per goal stats for swarms)

The memory store acts as a lightweight knowledge base for the agent and is
referenced each cycle to retrieve prior context and to persist new findings.

Reparodynamics interpretation:
    MemoryStore is the long-term substrate where repairs accumulate.
    Each TGRM cycle writes its improvements here, and future cycles read
    from it to reduce energy cost. When combined with VectorMemory, this
    becomes a semantic, time-aware repair substrate.

    The run_state, watchdog, and goal_index sections act as a meta-layer:
    they record how the system itself is running so that the agent
    can restart and continue repair with minimal extra energy and
    give swarm level analytics (per role and per goal).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Optional vector memory integration
try:
    from .vector_memory import VectorMemory  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    VectorMemory = None  # type: ignore[assignment]


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MemoryStore:
    """A lightweight persistent memory store using a JSON file.

    The JSON file is structured with several top-level keys:
        - "notes":       free-form text notes with metadata
        - "cycles":      logs of each research cycle
        - "hypotheses":  generated hypotheses for each goal
        - "citations":   structured citation objects from web/papers
        - "biomarkers":  placeholder for anti-aging / lab data
        - "run_state":   metadata for long-running autonomous sessions
        - "watchdog":    timestamps and counters for heartbeats
        - "goal_index":  compact per goal stats, including per role counts

    In-memory (non-persistent) vector memory may also be attached to
    support semantic search and time-decayed retrieval if the optional
    VectorMemory class is available.
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
            "run_state": {},   # crash proof run metadata
            "watchdog": {},    # heartbeat and last seen info
            "goal_index": {},  # compact goal wise stats
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
        defaults = {
            "notes": [],
            "cycles": [],
            "hypotheses": [],
            "citations": [],
            "biomarkers": [],
            "run_state": {},
            "watchdog": {},
            "goal_index": {},
        }
        for key, default in defaults.items():
            if key not in self._data:
                self._data[key] = default
            else:
                # Make sure types are reasonable
                if key in ("notes", "cycles", "hypotheses", "citations", "biomarkers"):
                    if not isinstance(self._data.get(key), list):
                        self._data[key] = []
                else:
                    if not isinstance(self._data.get(key), dict):
                        self._data[key] = {}

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
                    "run_state": {},
                    "watchdog": {},
                    "goal_index": {},
                }

    def _save(self) -> None:
        """Persist memory to disk."""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers for goal index
    # ------------------------------------------------------------------
    def _touch_goal_index(
        self,
        goal: str,
        *,
        last_event_type: str,
        role: Optional[str] = None,
        delta_notes: int = 0,
        delta_cycles: int = 0,
        rye_value: Optional[float] = None,
        domain: Optional[str] = None,
    ) -> None:
        """Update compact per goal stats for swarms and analytics."""
        gi = self._data.setdefault("goal_index", {})
        entry = gi.get(goal) or {}
        if not isinstance(entry, dict):
            entry = {}

        now = _utc_now_iso()
        entry.setdefault("created_at", now)
        entry["last_updated"] = now
        entry["last_event_type"] = last_event_type

        if domain is not None:
            entry.setdefault("domain", domain)

        # Note counts
        note_count = int(entry.get("note_count", 0)) + int(delta_notes)
        entry["note_count"] = note_count

        # Cycle counts
        cycle_count = int(entry.get("cycle_count", 0)) + int(delta_cycles)
        entry["cycle_count"] = cycle_count

        # Per role counters for swarms
        roles_dict = entry.get("roles") or {}
        if not isinstance(roles_dict, dict):
            roles_dict = {}
        if role:
            r_stats = roles_dict.get(role) or {}
            if not isinstance(r_stats, dict):
                r_stats = {}
            r_stats["note_count"] = int(r_stats.get("note_count", 0)) + int(delta_notes)
            r_stats["cycle_count"] = int(r_stats.get("cycle_count", 0)) + int(delta_cycles)
            if rye_value is not None:
                # Simple streaming average for RYE per role
                prev_avg = r_stats.get("avg_rye")
                prev_n = int(r_stats.get("rye_count", 0))
                if isinstance(prev_avg, (int, float)) and prev_n > 0:
                    new_avg = (prev_avg * prev_n + float(rye_value)) / float(prev_n + 1)
                    r_stats["avg_rye"] = new_avg
                    r_stats["rye_count"] = prev_n + 1
                else:
                    r_stats["avg_rye"] = float(rye_value)
                    r_stats["rye_count"] = 1
            roles_dict[role] = r_stats
        entry["roles"] = roles_dict

        # Goal level RYE stats (streaming) for fast summary
        if rye_value is not None:
            prev_avg = entry.get("avg_rye")
            prev_n = int(entry.get("rye_count", 0))
            if isinstance(prev_avg, (int, float)) and prev_n > 0:
                new_avg = (prev_avg * prev_n + float(rye_value)) / float(prev_n + 1)
                entry["avg_rye"] = new_avg
                entry["rye_count"] = prev_n + 1
            else:
                entry["avg_rye"] = float(rye_value)
                entry["rye_count"] = 1

        gi[goal] = entry
        self._data["goal_index"] = gi

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
        domain: Optional[str] = None,
    ) -> None:
        """Add a note associated with a research goal."""
        note = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "content": content,
            "tags": tags or [],
            "role": role,
            "importance": float(importance),
        }
        if domain is not None:
            note["domain"] = domain

        self._data.setdefault("notes", []).append(note)
        self._touch_goal_index(
            goal,
            last_event_type="note",
            role=role,
            delta_notes=1,
            delta_cycles=0,
            rye_value=None,
            domain=domain,
        )
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
                if domain is not None:
                    meta["domain"] = domain
                self.vector_memory.add_item(text=content, metadata=meta)
            except Exception:
                # Vector memory is optional; ignore failures
                pass

    def get_notes(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve notes. If goal is None, return all notes."""
        notes = self._data.get("notes", [])
        if goal is None:
            return list(notes)
        return [n for n in notes if n.get("goal") == goal]

    def search_notes(self, query: str, top_k: int = 5, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search notes either via vector memory (if available) or simple keyword search."""
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
                    meta.setdefault("timestamp", _utc_now_iso())
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
            "timestamp": _utc_now_iso(),
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
            "timestamp": _utc_now_iso(),
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
            "timestamp": _utc_now_iso(),
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

        # Update compact goal index for swarms
        goal = str(cycle_data.get("goal", ""))
        role = str(cycle_data.get("role", "agent"))
        domain = cycle_data.get("domain")
        rye_val = cycle_data.get("RYE")
        if goal:
            try:
                rye_float: Optional[float]
                if isinstance(rye_val, (int, float)):
                    rye_float = float(rye_val)
                else:
                    rye_float = None
                self._touch_goal_index(
                    goal,
                    last_event_type="cycle",
                    role=role,
                    delta_notes=0,
                    delta_cycles=1,
                    rye_value=rye_float,
                    domain=domain,
                )
            except Exception:
                # Any goal_index failure must not kill logging
                pass

        self._save()

    def get_cycle_history(self) -> List[Dict[str, Any]]:
        """Return the history of cycles."""
        return self._data.get("cycles", [])

    def get_recent_cycles(
        self,
        goal: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return recent cycles, optionally filtered by goal.

        Args:
            goal:
                If provided, only cycles whose 'goal' matches are returned.
            limit:
                Maximum number of cycles to return (most recent first).
        """
        history = self._data.get("cycles", [])
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]
        # Most recent first
        history_sorted = sorted(
            history,
            key=lambda c: c.get("timestamp", ""),
            reverse=True,
        )
        return history_sorted[:limit]

    # ------------------------------------------------------------------
    # Run state and checkpoint helpers
    # ------------------------------------------------------------------
    def save_run_state(
        self,
        *,
        goal: str,
        mode: str,
        minutes_remaining: Optional[float],
        last_cycle_index: Optional[int],
        domain: Optional[str] = None,
        role: str = "agent",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist high level run state for crash proof continuous mode.

        This is a compact summary so the agent can resume a long run:
            - what it was doing (goal, domain, role)
            - how it was running (mode)
            - how much time remained (approx minutes_remaining)
            - which cycle index was last completed
        """
        state: Dict[str, Any] = {
            "updated_at": _utc_now_iso(),
            "goal": goal,
            "mode": mode,
            "domain": domain,
            "role": role,
            "minutes_remaining": minutes_remaining,
            "last_cycle_index": last_cycle_index,
        }
        if extra:
            state["extra"] = extra

        self._data.setdefault("run_state", {})
        self._data["run_state"] = state
        self._save()

    def load_run_state(self) -> Optional[Dict[str, Any]]:
        """Return the last saved run state, if any."""
        state = self._data.get("run_state") or {}
        if not isinstance(state, dict) or not state:
            return None
        return dict(state)

    def clear_run_state(self) -> None:
        """Clear saved run state metadata."""
        self._data["run_state"] = {}
        self._save()

    # ------------------------------------------------------------------
    # Watchdog heartbeats for long runs
    # ------------------------------------------------------------------
    def heartbeat(self, label: str = "continuous_run") -> None:
        """Record a watchdog heartbeat.

        The agent can call this periodically so that:
            - you can see if it is still alive
            - a supervising process can measure downtime
        """
        wd = self._data.setdefault("watchdog", {})
        now = _utc_now_iso()
        entry = wd.get(label, {})
        if isinstance(entry, dict):
            count = int(entry.get("count", 0)) + 1
        else:
            count = 1
        wd[label] = {
            "last_beat": now,
            "count": count,
        }
        self._save()

    def get_watchdog_info(self, label: str = "continuous_run") -> Dict[str, Any]:
        """Return watchdog data for a label.

        The structure contains:
            - last_beat: ISO timestamp or None
            - count: how many beats were recorded
            - seconds_since_last: float or None
        """
        wd = self._data.get("watchdog") or {}
        if not isinstance(wd, dict):
            wd = {}
        entry = wd.get(label) or {}
        last_beat = entry.get("last_beat")
        count = entry.get("count", 0)

        seconds_since_last: Optional[float] = None
        if isinstance(last_beat, str):
            try:
                dt = datetime.fromisoformat(last_beat.replace("Z", "+00:00"))
                seconds_since_last = (datetime.now(timezone.utc) - dt).total_seconds()
            except Exception:
                seconds_since_last = None

        return {
            "last_beat": last_beat,
            "count": count,
            "seconds_since_last": seconds_since_last,
        }

    # ------------------------------------------------------------------
    # Reporting helpers (for long autonomous runs)
    # ------------------------------------------------------------------
    def get_rye_stats(
        self,
        goal: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        """Compute basic RYE statistics.

        If goal is provided, restrict to that goal.
        If role is also provided, restrict to that logical role.
        Returns:
            (avg_rye, min_rye, max_rye, count)
        """
        history = self._data.get("cycles", [])
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]
        if role is not None:
            history = [c for c in history if c.get("role") == role]

        values: List[float] = []
        for c in history:
            v = c.get("RYE")
            if isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            return None, None, None, 0

        avg_rye = sum(values) / len(values)
        min_rye = min(values)
        max_rye = max(values)
        return avg_rye, min_rye, max_rye, len(values)

    def build_text_report(self, goal: Optional[str] = None) -> str:
        """Build a simple text or markdown report for a goal or all goals.

        This is designed to be used by the UI after a long run:
        - You can show it directly in Streamlit
        - Or offer it as a downloadable .txt or .md file
        """
        # Header
        if goal:
            title = f"Autonomous Research Report\nGoal: {goal}\n"
        else:
            title = "Autonomous Research Report\n(All goals)\n"

        title += "=" * 40 + "\n\n"

        # Basic stats
        history = self.get_cycle_history()
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]

        total_cycles = len(history)
        avg_rye, min_rye, max_rye, rye_count = self.get_rye_stats(goal=goal)

        title += f"Total cycles: {total_cycles}\n"
        if rye_count > 0 and avg_rye is not None:
            title += f"RYE (avg): {avg_rye:.3f}\n"
            title += f"RYE (min): {min_rye:.3f}\n"
            title += f"RYE (max): {max_rye:.3f}\n"
        else:
            title += "RYE: no data available\n"
        title += "\n"

        # Notes
        notes = self.get_notes(goal=goal)
        title += f"Notes collected: {len(notes)}\n"
        if notes:
            title += "\nRecent notes:\n"
            for n in notes[-5:]:
                ts = n.get("timestamp", "")
                content = str(n.get("content", "")).strip()
                if len(content) > 200:
                    content = content[:200] + "..."
                title += f"- [{ts}] {content}\n"
        title += "\n"

        # Hypotheses
        hyps = self.get_hypotheses(goal=goal)
        title += f"Hypotheses generated: {len(hyps)}\n"
        if hyps:
            title += "\nRecent hypotheses:\n"
            for h in hyps[-5:]:
                ts = h.get("timestamp", "")
                text = str(h.get("text", "")).strip()
                score = h.get("score", None)
                if len(text) > 200:
                    text = text[:200] + "..."
                if isinstance(score, (int, float)):
                    title += f"- [{ts}] ({score:.2f}) {text}\n"
                else:
                    title += f"- [{ts}] {text}\n"
        title += "\n"

        # Citations
        cits = self.get_citations(goal=goal)
        title += f"Citations logged: {len(cits)}\n"
        if cits:
            title += "\nSample citations:\n"
            for e in cits[-5:]:
                ts = e.get("timestamp", "")
                c = e.get("citation", {}) or {}
                src = c.get("source", "web")
                c_title = c.get("title", "")
                url = c.get("url", "")
                title += f"- [{ts}] [{src}] {c_title} - {url}\n"

        title += "\nEnd of report.\n"
        return title

    # ------------------------------------------------------------------
    # Convenience getters for report_generator and UI
    # ------------------------------------------------------------------
    def get_all_cycles(self) -> List[Dict[str, Any]]:
        """Alias for getting all cycles (for report generators)."""
        return self._data.get("cycles", [])

    def get_cycles_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Return all cycles for a specific goal."""
        return [c for c in self._data.get("cycles", []) if c.get("goal") == goal]

    def get_all_notes(self) -> List[Dict[str, Any]]:
        """Return all notes regardless of goal."""
        return self._data.get("notes", [])

    def get_notes_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Explicit alias for get_notes(goal=...)."""
        return self.get_notes(goal=goal)

    def get_all_hypotheses(self) -> List[Dict[str, Any]]:
        """Return all hypotheses for all goals."""
        return self._data.get("hypotheses", [])

    def get_hypotheses_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Explicit alias for get_hypotheses(goal=...)."""
        return self.get_hypotheses(goal=goal)

    def get_all_citations(self) -> List[Dict[str, Any]]:
        """Return all citation entries for all goals."""
        return self._data.get("citations", [])

    def get_citations_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Explicit alias for get_citations(goal=...)."""
        return self.get_citations(goal=goal)

    # ------------------------------------------------------------------
    # Goal index accessors for swarm analytics
    # ------------------------------------------------------------------
    def list_goals(self) -> List[str]:
        """Return a sorted list of known goals from goal_index."""
        gi = self._data.get("goal_index") or {}
        if not isinstance(gi, dict):
            return []
        return sorted(gi.keys())

    def get_goal_index(self, goal: Optional[str] = None) -> Dict[str, Any]:
        """Return compact goal index entry or the full index.

        If goal is None, return the entire goal_index dict.
        """
        gi = self._data.get("goal_index") or {}
        if not isinstance(gi, dict):
            return {}
        if goal is None:
            return dict(gi)
        return dict(gi.get(goal, {}))
