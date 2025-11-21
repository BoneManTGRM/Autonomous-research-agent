"""Core agent implementation.

This module defines the `CoreAgent` class which orchestrates the
TGRM loop for running research cycles. It interacts with the memory store
and delegates work to the `TGRMLoop` for each cycle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .memory_store import MemoryStore
from .tgrm_loop import TGRMLoop


class CoreAgent:
    """High-level controller for the autonomous research agent."""

    def __init__(self, memory_store: MemoryStore, config: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.config = config or {}
        self.tgrm_loop = TGRMLoop(memory_store, config)

    def run_cycle(self, goal: str, cycle_index: int) -> Dict[str, Any]:
        """Run a single cycle of research using the TGRM loop.

        Args:
            goal (str): Research goal.
            cycle_index (int): Index of the cycle (starting from 0).

        Returns:
            Dict[str, Any]: Dictionary containing the summary and log of the cycle.
        """
        return self.tgrm_loop.run_cycle(goal, cycle_index)
