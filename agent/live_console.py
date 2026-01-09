"""
live_console.py
===============

This module defines helpers for the live console component of the
Streamlit UI.  The purpose of these helpers is to translate internal
state into human-readable summaries for display: autonomy levels,
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
        The zero-based index of the current phase.  If ``None`` then
        the run has not started.
    total:
        The total number of phases.

    Returns
    -------
    tuple
        A triple ``(label, numerator, denominator)`` where ``label`` is
        one of ``"not started"``, ``"self-monitoring"``, ``"cross-monitoring"`` or
        ``"co-pilot"``, ``numerator`` equals ``current + 1`` (for display), and
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
        label = "self-monitoring"
    elif fraction < 0.75:
        label = "cross-monitoring"
    else:
        label = "co-pilot"
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
    """Construct a human-readable narrative from event log entries.

    This helper takes an iterable of event dicts (e.g. produced by the
    queue worker or event_log JSONL) and returns a list of strings summarising
    run progression.

    Recognised kinds include (best-effort):
      - phase_start / cycle_start
      - run_finished
      - agent_output
      - candidate_hypothesis
      - verification
      - discovery
      - rye_update

    Unrecognised kinds are rendered generically.
    """
    narrative: List[str] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("kind")
        kind_s = str(kind or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}

        # Common index fields
        idx = ev.get("phase_index")
        if idx is None:
            idx = ev.get("cycle")
        if idx is None:
            idx = ev.get("cycle_index")
        try:
            idx_i = int(idx) if idx is not None else None
        except Exception:
            idx_i = None

        # Phase / cycle starts
        if kind_s in ("phase_start", "cycle_start", "cycle_started", "cycle_begin"):
            total = ev.get("phase_total") or ev.get("cycle_total") or ev.get("total_cycles")
            name = ev.get("phase_name") or ev.get("cycle_name") or "cycle"
            if idx_i is not None and total is not None:
                narrative.append(f"Started {name} {idx_i + 1} of {total}.")
            elif idx_i is not None:
                narrative.append(f"Started {name} {idx_i + 1}.")
            else:
                narrative.append(f"Started {name}.")
            continue

        # Agent output
        if kind_s == "agent_output":
            role = ev.get("role") or data.get("role") or "agent"
            text = data.get("text") or data.get("output") or data.get("message") or ""
            text_s = str(text) if text is not None else ""
            snippet = (text_s[:120] + "...") if len(text_s) > 120 else text_s
            if idx_i is not None:
                narrative.append(f"Cycle {idx_i + 1}: {role} produced output: {snippet}")
            else:
                narrative.append(f"{role} produced output: {snippet}")
            continue

        # Candidate hypothesis
        if kind_s in ("candidate_hypothesis", "discovery_candidate"):
            title = data.get("title") or data.get("thesis") or data.get("headline") or "candidate"
            if idx_i is not None:
                narrative.append(f"Cycle {idx_i + 1}: Proposed candidate hypothesis: {title}")
            else:
                narrative.append(f"Proposed candidate hypothesis: {title}")
            continue

        # Verification
        if kind_s == "verification":
            verdict = data.get("verdict") or data.get("pass_fail") or data.get("result")
            title = data.get("title") or data.get("candidate") or ""
            if verdict is None:
                msg = "Verification event."
            else:
                msg = f"Verification: {verdict}"
            if title:
                msg += f" ({title})"
            if idx_i is not None:
                narrative.append(f"Cycle {idx_i + 1}: {msg}")
            else:
                narrative.append(msg)
            continue

        # Final discovery
        if kind_s == "discovery":
            thesis = data.get("thesis") or data.get("title") or "discovery"
            if idx_i is not None:
                narrative.append(f"Cycle {idx_i + 1}: Discovery committed: {thesis}")
            else:
                narrative.append(f"Discovery committed: {thesis}")
            continue

        # RYE update
        if kind_s == "rye_update":
            rye = data.get("RYE") or data.get("rye") or ev.get("rye")
            delta_r = data.get("delta_R") or data.get("delta_r") or ev.get("delta_R") or ev.get("delta_r")
            if idx_i is not None:
                narrative.append(f"Cycle {idx_i + 1}: RYE update (RYE={rye}, delta_R={delta_r})")
            else:
                narrative.append(f"RYE update (RYE={rye}, delta_R={delta_r})")
            continue

        # Run finished
        if kind_s == "run_finished":
            narrative.append("Run finished.")
            continue

        narrative.append(f"Event: {kind_s}")

    return narrative
