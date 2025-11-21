"""Helper for running the CoreAgent in continuous mode from scripts.

This is optional: Streamlit already uses CoreAgent.run_continuous.
You can use this from CLI tools or scheduled jobs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from .core_agent import CoreAgent
from .memory_store import MemoryStore


def run_continuous_session(
    goal: str,
    memory_file: str = "logs/sessions/default_memory.json",
    config: Optional[Dict[str, Any]] = None,
    max_cycles: int = 100,
    stop_rye: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run a continuous session of the research agent.

    Returns a list of human-readable summaries.
    """
    mem = MemoryStore(memory_file)
    agent = CoreAgent(memory_store=mem, config=config or {})
    return agent.run_continuous(goal=goal, max_cycles=max_cycles, stop_rye=stop_rye)
