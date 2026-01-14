"""
phase_manager.py
================

This module defines simple helpers for mapping cycles into phases.  In the
current implementation the agent runs a single phase per cycle; each
iteration of the underlying TGRM loop is considered its own "phase" for
the purposes of progress reporting.  Should you adopt a more complex
workflow (for example, separate "search", "analysis", and "integration"
stages within each cycle) you can modify the functions here to emit
structured phase names and counts accordingly.

The functions are deliberately stateless and deterministic so they can be
called from both the queue worker and the Streamlit UI.
"""

from __future__ import annotations

from typing import List, Optional


def phases_for_cycles(total_cycles: int) -> List[str]:
    """Return a list of phase names for the given number of cycles.

    This implementation alternates between exploration and exploitation
    phases in a 4:2 pattern.  Specifically, cycles 0â3 are labelled
    ``"explore"``, cycles 4â5 are ``"exploit"``, and the pattern repeats
    every six cycles.  This encourages agents to spend the majority of
    their time exploring (research, tool usage) while periodically
    consolidating and exploiting recent discoveries.

    Parameters
    ----------
    total_cycles:
        Number of cycles requested for the run.

    Returns
    -------
    list of str
        A list whose length equals ``total_cycles``.  Each element is
        either ``"explore"`` or ``"exploit"`` according to the
        repeating 4âexplore / 2âexploit pattern.
    """
    if total_cycles < 1:
        return []
    pattern = ["explore", "explore", "explore", "explore", "exploit", "exploit"]
    phases: List[str] = []
    for i in range(total_cycles):
        phases.append(pattern[i % len(pattern)])
    return phases


def phase_name_for_index(index: int, total_cycles: int) -> Optional[str]:
    """Return the phase name for a given index.

    This helper mirrors :func:`phases_for_cycles` without allocating the full
    list.  It computes the phase for a single index using the same 4:2
    explore/exploit pattern.  If ``index`` is out of range, it returns
    ``None``.

    Parameters
    ----------
    index:
        Zeroâbased cycle index.  Must satisfy ``0 <= index < total_cycles``.
    total_cycles:
        Total number of cycles requested for the run.

    Returns
    -------
    str or None
        The phase name (``"explore"`` or ``"exploit"``) or ``None`` if
        ``index`` is invalid.
    """
    if index < 0 or index >= total_cycles:
        return None
    # 6 cycle repeating pattern: explore (0â3), exploit (4â5)
    pos = index % 6
    return "explore" if pos < 4 else "exploit"


__all__ = ["phases_for_cycles", "phase_name_for_index"]
