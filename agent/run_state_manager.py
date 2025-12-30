"""
run_state_manager.py

Small utility dataclass for persisting run progress to JSON.

Key idea
- The UI progress bar reads current and total.
- Some parts of ARA historically treated the phase index as 0-based, while
  other parts wrote "completed cycles" as 1-based.
- This module now supports both safely.

What changed
- Added update_cycle(completed, total_cycles, name) as the preferred API.
- update_phase(...) now auto-normalizes when it looks like the caller is
  passing a 1-based completed count (common in swarm/mini-cycle code).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    """Return current UTC time in ISO 8601 format without timezone info."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


@dataclass
class RunState:
    """Serializable run state used by worker and UI."""

    run_id: str
    status: str = "queued"

    # "phase_*" fields are kept for backwards compatibility.
    # In finite mode, a "phase" usually equals a "cycle" for the progress bar,
    # but in some swarm setups phase_total may remain small while total_cycles is large.
    phase_total: int = 0
    phase_index: Optional[int] = None
    phase_name: Optional[str] = None

    # Fields the UI bar commonly uses
    current: Optional[int] = None
    total: int = 0

    # Cycle fields
    current_cycle: Optional[int] = None
    total_cycles: int = 0

    notes: str = ""
    goal: str = ""

    created_at: str = _utc_iso()
    updated_at: str = _utc_iso()

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
        rid = run_id or str(uuid.uuid4())
        return cls(
            run_id=rid,
            status="queued",
            phase_total=total_cycles,
            total=total_cycles,
            total_cycles=total_cycles,
            goal=goal,
        )

    @classmethod
    def load(cls, path: Path) -> "RunState":
        data = json.loads(path.read_text(encoding="utf-8"))
        state = cls(**data)

        # Normalize totals if older files had only one of these
        if state.total_cycles and not state.total:
            state.total = state.total_cycles
        if state.total and not state.total_cycles:
            state.total_cycles = state.total
        if state.phase_total == 0 and state.total_cycles:
            state.phase_total = state.total_cycles

        return state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_iso()
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    # Progress update helpers
    # ------------------------------------------------------------------
    def update_cycle(
        self,
        completed: int,
        total_cycles: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Preferred API for progress.

        Parameters
        - completed: 1-based completed count (1..total)
        - total_cycles: optional override for total cycles
        - name: optional phase name (for UI labels)
        """
        if total_cycles is not None and total_cycles > 0:
            self.total_cycles = int(total_cycles)
            self.total = int(total_cycles)
            if self.phase_total == 0:
                self.phase_total = int(total_cycles)

        total = self.total_cycles or self.total or self.phase_total or 0
        if total > 0:
            completed = _clamp_int(int(completed), 0, total)
            # Keep the UI bar 1..total when running
            self.current = completed
            self.current_cycle = completed
            # Internally store a 0-based phase_index when possible
            self.phase_index = _clamp_int(completed - 1, 0, max(total - 1, 0)) if completed > 0 else 0
        else:
            self.current = int(completed)
            self.current_cycle = int(completed)
            self.phase_index = int(completed)

        if name is not None:
            self.phase_name = name

    def update_phase(self, index: int, name: Optional[str] = None) -> None:
        """
        Backwards compatible API.

        Historically callers used two different conventions:
        - 0-based phase index (0..total-1)
        - 1-based completed count (1..total)

        This method auto-detects the second case and normalizes so the UI bar
        advances one cycle at a time and does not jump to total early.
        """
        idx = int(index)
        total = self.total_cycles or self.total or self.phase_total or 0

        looks_like_completed_count = False
        if total > 0:
            # If idx equals total, it is almost certainly "completed cycles".
            if idx == total:
                looks_like_completed_count = True
            # If idx is in 1..total and phase_index is None or already large,
            # treat as completed count to avoid phase_index becoming total.
            elif 1 <= idx <= total and (self.phase_index is None or (self.phase_index is not None and self.phase_index >= total - 1)):
                looks_like_completed_count = True

        if looks_like_completed_count:
            self.update_cycle(completed=idx, total_cycles=total, name=name)
            return

        # Default 0-based behavior
        self.phase_index = idx
        self.current = idx
        self.current_cycle = idx
        if name is not None:
            self.phase_name = name

    def mark_finished(self) -> None:
        self.status = "finished"
        total = self.total_cycles or self.total or self.phase_total or 0
        if total > 0:
            # Ensure the UI shows completion cleanly
            self.current = total
            self.current_cycle = total
            self.phase_index = total - 1
            if self.phase_total == 0:
                self.phase_total = total
            if self.total == 0:
                self.total = total
        else:
            if self.phase_total > 0 and (self.phase_index is None or self.phase_index < self.phase_total - 1):
                self.update_phase(self.phase_total - 1, name=self.phase_name)


__all__ = ["RunState"]
