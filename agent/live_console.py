"""
live_console.py
===============

This module defines helpers for the live console component of the
Streamlit UI.  The purpose of these helpers is to translate internal
state into human‑readable summaries for display: autonomy levels,
agent presence chips and a basic narrative timeline derived from event
logs.  They do not depend on Streamlit directly and can therefore be
unit tested in isolation.

The autonomy level heuristic implemented here is simplistic: it
calculates the fraction of phases completed and assigns a label.  You
can customise the thresholds and labels to suit your needs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def compute_autonomy_level(current: Optional[int], total: int) -> Tuple[str, int, int]:
    """Compute the autonomy level label and progress counts.

    Parameters
    ----------
    current:
        The zero‑based index of the current phase.  If ``None`` then
        the run has not started.
    total:
        The total number of phases.

    Returns
    -------
    tuple
        A triple ``(label, numerator, denominator)`` where ``label`` is
        one of ``"not started"``, ``"self‑monitoring"``, ``"cross‑monitoring"`` or
        ``"co‑pilot"``, ``numerator`` equals ``current + 1`` (for display), and
        ``denominator`` is ``total``.
    """
    if total <= 0:
        return ("not started", 0, 0)
    if current is None:
        return ("not started", 0, total)
    fraction = (current + 1) / float(total)
    if fraction < 0.25:
        label = "observing"
    elif fraction < 0.50:
        label = "self‑monitoring"
    elif fraction < 0.75:
        label = "cross‑monitoring"
    else:
        label = "co‑pilot"
    return (label, current + 1, total)


def infer_agent_presence(job_def: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Infer agent presence chips from a job definition.

    The job definition may include a ``roles`` list (each element being
    a tuple of (role_name, description)) or a ``swarm_size`` and domain
    from which we derive a generic set of roles.  If no roles can be
    determined an empty list is returned.
    """
    roles: List[Tuple[str, str]] = []
    # Explicit roles override everything
    if "roles" in job_def and isinstance(job_def["roles"], Iterable):
        for role in job_def["roles"]:
            if isinstance(role, (list, tuple)) and len(role) >= 2:
                roles.append((str(role[0]), str(role[1])))
    else:
        # Use swarm_size to derive a default set of roles
        swarm_size = int(job_def.get("swarm_size", 1))
        domain = str(job_def.get("domain", "general")).lower()
        # Choose a subset of archetypal roles based on swarm size
        archetypes = [
            ("researcher", "Deep literature and web researcher"),
            ("critic", "Methodology critic and refiner"),
            ("explorer", "Out of distribution explorer"),
            ("theorist", "Model builder and unifier"),
            ("integrator", "Synthesizer and summariser"),
        ]
        # Repeat or truncate to match swarm size
        if swarm_size <= len(archetypes):
            roles = archetypes[:swarm_size]
        else:
            roles = archetypes + [archetypes[i % len(archetypes)] for i in range(swarm_size - len(archetypes))]
        # Domain could influence description (not implemented)
    return roles


def build_narrative_timeline(events: Iterable[Dict[str, Any]]) -> List[str]:
    """Construct a human‑readable narrative from event log entries.

    This helper takes an iterable of event dicts (e.g. produced by the
    queue worker) and returns a list of strings summarising the run
    progression.  It expects events to have a ``kind`` key and may
    utilise other fields to build descriptive sentences.

    Currently recognised kinds are ``phase_start`` and ``run_finished``;
    unrecognised kinds are rendered generically.  You can extend this
    function to support additional event types.
    """
    narrative: List[str] = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "phase_start":
            idx = ev.get("phase_index")
            total = ev.get("phase_total")
            name = ev.get("phase_name", "phase")
            narrative.append(f"Started {name} {idx + 1} of {total}.")
        elif kind == "run_finished":
            narrative.append("Run finished.")
        else:
            narrative.append(f"Event: {kind}")
    return narrative


__all__ = [
    "compute_autonomy_level",
    "infer_agent_presence",
    "build_narrative_timeline",
]