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

    This implementation uses a simple explore/exploit cadence to expose
    different behaviours across long runs.  The phase naming follows:

    * Every 10th cycle (1-based index) is labelled ``"explore"`` to signal
      a highâexploration burst.  These cycles may allocate more tool budget
      or allow riskier actions.
    * Every 5th cycle (except those already marked as explore) is labelled
      ``"exploit"`` to emphasise consolidation and integration of recent
      findings.
    * All other cycles are labelled ``"run"``.

    Parameters
    ----------
    total_cycles: int
        Number of cycles requested for the run.

    Returns
    -------
    List[str]
        A list of humanâreadable phase names, length equal to ``total_cycles``.
    """
    phases: List[str] = []
    if total_cycles < 1:
        return phases
    for idx in range(total_cycles):
        # Convert to 1-based index for readability
        cycle_number = idx + 1
        if cycle_number % 10 == 0:
            phases.append("explore")
        elif cycle_number % 5 == 0:
            phases.append("exploit")
        else:
            phases.append("run")
    return phases


def phase_name_for_index(index: int, total_cycles: int) -> Optional[str]:
    """Return the phase name for a given index using the explore/exploit cadence.

    Rather than recomputing the full phase list, this convenience wrapper
    applies the same modulus logic used in :func:`phases_for_cycles` to
    determine the phase for a specific index.  See that function for a
    detailed description of the cadence.

    Parameters
    ----------
    index: int
        Zeroâbased index of the cycle.  If out of bounds, ``None`` is
        returned.
    total_cycles: int
        Total number of cycles in the run.

    Returns
    -------
    Optional[str]
        The phase name (``"explore"``, ``"exploit"``, or ``"run"``) for
        the given index, or ``None`` if the index is invalid.
    """
    if index < 0 or index >= total_cycles:
        return None
    cycle_number = index + 1
    if cycle_number % 10 == 0:
        return "explore"
    if cycle_number % 5 == 0:
        return "exploit"
    return "run"


__all__ = ["phases_for_cycles", "phase_name_for_index"]
