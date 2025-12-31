"""
run_state_manager.py
=====================

This module defines a simple run state manager for the Autonomous Research
Agent (ARA).  It encapsulates a run's metadata (such as the number of
cycles, the current phase index and name, and timestamps) and provides
helpers for reading and writing this state to JSON on disk.  Other
components (like the queue worker and Streamlit front‑end) are expected to
use this module to coordinate progress reporting and persistence.

The design deliberately keeps the model free from business logic; it
simply holds data and offers convenience methods.  Should you need to
extend the state with additional fields (for example, to track
equilibrium statistics or RYE metrics), you can add new attributes to
``RunState`` and they will automatically be included in the serialized
output.

Usage example::

    from pathlib import Path
    from run_state_manager import RunState

    run_state = RunState.new(run_id="123", total_cycles=10, goal="ARA test")
    run_state.update_phase(index=0, name="setup")
    run_state.save(Path("runs/logs/run_state.json"))

    # later on
    loaded = RunState.load(Path("runs/logs/run_state.json"))
    print(loaded.current_cycle, loaded.phase_name)

The module does not depend on any of the agent internals, so it can be
reused in both the streamlit UI and the background worker.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    """Return the current UTC time in ISO 8601 format without timezone info."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass
class RunState:
    """Representation of a run's lifecycle and progress.

    Attributes
    ----------
    run_id:
        A unique identifier for this run.  If not provided at construction
        time a new UUID4 string will be generated.
    status:
        A human‑readable status string (e.g. "queued", "running", "finished").
    phase_total:
        The total number of phases expected for this run.  This should
        correspond to the number of cycles passed in when the job is
        created; if you support multi‑phase workflows you can adjust
        accordingly.
    phase_index:
        Zero‑based index of the current phase.  ``None`` until the first
        phase starts.
    phase_name:
        Optional human‑readable name for the current phase.  This is
        useful when displaying progress in the UI.
    current:
        Alias for ``phase_index``; maintained for backwards compatibility
        with existing UIs that expect ``current`` and ``total`` keys.
    total:
        Alias for ``phase_total``.
    current_cycle:
        The current cycle number.  In simple finite mode this is
        identical to ``phase_index``.
    total_cycles:
        Total number of cycles requested for this run.  Usually equal to
        ``phase_total``.
    notes:
        Optional notes or metadata about the run.
    goal:
        Optional string describing the research goal for display.
    created_at:
        Timestamp of when the state was first created.
    updated_at:
        Timestamp of when the state was last persisted.
    """

    run_id: str
    status: str = "queued"
    phase_total: int = 0
    phase_index: Optional[int] = None
    phase_name: Optional[str] = None
    current: Optional[int] = None
    total: int = 0
    current_cycle: Optional[int] = None
    total_cycles: int = 0
    notes: str = ""
    goal: str = ""
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def new(
        cls,
        run_id: Optional[str] = None,
        total_cycles: int = 0,
        goal: str = "",
    ) -> "RunState":
        """Create a new RunState with reasonable defaults.

        Parameters
        ----------
        run_id:
            If provided, use this value; otherwise a new UUID4 string will be
            generated.  Use deterministic IDs if you need reproducibility.
        total_cycles:
            The number of cycles/phases requested for this run.  This
            populates both ``phase_total`` and ``total_cycles``.
        goal:
            Description of the run's objective.
        """
        rid = run_id or str(uuid.uuid4())
        return cls(
            run_id=rid,
            status="queued",
            phase_total=total_cycles,
            total=total_cycles,
            total_cycles=total_cycles,
            goal=goal,
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "RunState":
        """Load a RunState from a JSON file.

        Raises ``FileNotFoundError`` if the file does not exist or
        ``json.JSONDecodeError`` on malformed JSON.  Unknown keys in the
        JSON will be ignored.
        """
        with path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        # Filter keys to those defined on the dataclass
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in field_names}
        return cls(**kwargs)  # type: ignore[arg-type]

    def save(self, path: Path) -> None:
        """Persist the current state to a JSON file.

        The ``updated_at`` timestamp will be refreshed before writing.  The
        directory will be created if it does not already exist.
        """
        self.updated_at = _utc_iso()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write to avoid corrupt/partial JSON on process interruption
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable dictionary representation of this state."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Progress updates
    # ------------------------------------------------------------------
    def update_phase(self, index: int, name: Optional[str] = None) -> None:
        """Update the current phase index and optionally its name.

        The caller is responsible for ensuring that ``index`` does not
        exceed ``phase_total``.  The ``current`` and ``current_cycle``
        aliases will also be updated for compatibility with older UIs.

        Parameters
        ----------
        index:
            Zero‑based index of the phase that just started.
        name:
            Optional human‑readable name for the phase (e.g. "run").
        """
        self.phase_index = index
        self.current = index
        self.current_cycle = index
        if name is not None:
            self.phase_name = name
        # Do not update ``total``; it should remain the original cycle count
        # ``updated_at`` is refreshed on save

    def mark_finished(self) -> None:
        """Mark this run as finished.

        This sets the status to ``finished`` and moves the phase index to the
        last phase (phase_total - 1) if it wasn't already there.
        """
        self.status = "finished"
        if self.phase_total > 0 and (self.phase_index is None or self.phase_index < self.phase_total - 1):
            self.update_phase(self.phase_total - 1, name=self.phase_name)


__all__ = ["RunState"]
