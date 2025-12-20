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

    In the simplest finite mode, each cycle corresponds to a single
    phase named ``"run"``.  This helper returns a list of that name
    repeated ``total_cycles`` times.  You can customise this behaviour
    by replacing the body of this function to partition cycles into
    sub‑phases or to use different names depending on the index.

    Parameters
    ----------
    total_cycles:
        Number of cycles requested for the run.

    Returns
    -------
    list of str
        A list whose length equals ``total_cycles``, each element being a
        human‑readable phase name.
    """
    if total_cycles < 1:
        return []
    return ["run" for _ in range(total_cycles)]


def phase_name_for_index(index: int, total_cycles: int) -> Optional[str]:
    """Return the phase name for a given index.

    This is a convenience wrapper around :func:`phases_for_cycles` to
    avoid recomputing the entire list when only one name is needed.

    Parameters
    ----------
    index:
        Zero‑based index of the phase.  If out of bounds, ``None`` is
        returned.
    total_cycles:
        Total number of cycles requested for the run.

    Returns
    -------
    str or None
        The human‑readable name for the phase, or ``None`` if the
        index is invalid.
    """
    if index < 0 or index >= total_cycles:
        return None
    return "run"


__all__ = ["phases_for_cycles", "phase_name_for_index"]