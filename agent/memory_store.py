"""Memory storage for the autonomous research agent.

This module provides a simple JSON-based persistent storage layer. It stores
notes, summaries, and cycle logs across sessions. The memory store acts
similar to a knowledge base for the agent and is referenced each cycle to
retrieve prior context and to persist new findings.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List


class MemoryStore:
    """A lightweight persistent memory store using a JSON file.

    The memory is structured with two top-level keys: "notes" and "cycles".
    Notes store free-form text objects annotated with metadata like goal and
    timestamp. Cycles store logs of each research cycle run by the agent.
    """

    def __init__(self, memory_file: str) -> None:
        self.memory_file = memory_file
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self._data = {"notes": [], "cycles": []}
        self._load()

    def _load(self) -> None:
        """Load memory from disk if the file exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # If the file is corrupt or cannot be read, start fresh
                self._data = {"notes": [], "cycles": []}

    def _save(self) -> None:
        """Persist memory to disk."""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # Note-related methods
    def add_note(self, goal: str, content: str) -> None:
        """Add a note associated with a research goal."""
        note = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "content": content,
        }
        self._data.setdefault("notes", []).append(note)
        self._save()

    def get_notes(self, goal: str) -> List[Dict[str, Any]]:
        """Retrieve notes relevant to a particular goal."""
        return [n for n in self._data.get("notes", []) if n.get("goal") == goal]

    # Cycle log-related methods
    def log_cycle(self, cycle_data: Dict[str, Any]) -> None:
        """Append cycle data to the cycle log."""
        self._data.setdefault("cycles", []).append(cycle_data)
        self._save()

    def get_cycle_history(self) -> List[Dict[str, Any]]:
        """Return the history of cycles."""
        return self._data.get("cycles", [])
