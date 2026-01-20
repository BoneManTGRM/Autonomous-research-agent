"""
run_state_manager.py
====================

A small, explicit "run state" model used to coordinate a single engine run across:
  - the queue/worker (producer of progress + state)
  - the Streamlit UI (consumer / viewer)
  - report builders (consumers)

The key fixes in this version:
  1) A run always has a non-empty run_id (UUID4 by default).
  2) Each run has an explicit run_dir (runs_root/<run_id>) that can be created on demand.
  3) The state exposes run_id, run_dir, and cycle index consistently (phase_index/current/current_cycle/cycle_index).

Notes
-----
* This module stays "dumb": it does not implement business logic, it only stores and persists state.
* Unknown keys in JSON are ignored on load to allow forwards/backwards compatibility.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    """Return current UTC timestamp in ISO 8601 (seconds precision)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    s = str(raw).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


# If enabled, persist only {"cycle_index": N} to run_state.json.
# This reduces noise and avoids accidental run mixing.
MINIMAL_RUN_STATE: bool = _env_bool("ARA_MINIMAL_RUN_STATE", False)



def resolve_runs_root() -> Path:
    """Resolve the base runs directory.

    Preference order:
      1) env ARA_RUNS_DIR / ARA_RUNS_ROOT / RUNS_DIR
      2) run_jobs.BASE_DIR if available
      3) repo-local ./runs (next to this file's parent package)
    """
    for k in ("ARA_RUNS_DIR", "ARA_RUNS_ROOT", "RUNS_BASE_DIR", "RUNS_DIR", "RUNS_ROOT"):
        v = os.getenv(k)
        if v:
            try:
                return Path(v).expanduser().resolve()
            except Exception:
                pass

    try:
        # Optional: align with run_jobs.py if present
        from .run_jobs import BASE_DIR as _BASE_DIR  # type: ignore
        if isinstance(_BASE_DIR, Path):
            return _BASE_DIR
        if isinstance(_BASE_DIR, str) and _BASE_DIR.strip():
            return Path(_BASE_DIR).expanduser().resolve()
    except Exception:
        pass

    # Fallback: <repo>/runs
    try:
        here = Path(__file__).resolve()
        # common layout: repo_root/agent/run_state_manager.py -> repo_root/runs
        return (here.parent.parent / "runs").resolve()
    except Exception:
        return Path("./runs").resolve()


def resolve_run_dir(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    root = runs_root if isinstance(runs_root, Path) else resolve_runs_root()
    return root / str(run_id)


def default_state_path(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    """Default location for per-run state."""
    return resolve_run_dir(run_id, runs_root=runs_root) / "run_state.json"


@dataclass
class RunState:
    """Lightweight, JSON-serialisable run state."""

    # Identity + storage
    run_id: str
    run_dir: Optional[str] = None  # stored as a string to keep JSON serialisable

    # Human metadata
    status: str = "queued"  # queued|running|finished|failed|...
    goal: str = ""
    domain: str = "general"
    mode: str = ""  # single|swarm|...

    # Progress tracking
    phase_total: int = 0
    phase_index: Optional[int] = None
    phase_name: Optional[str] = None

    # Back-compat aliases used by older UI code
    current: Optional[int] = None
    total: int = 0
    current_cycle: Optional[int] = None
    total_cycles: int = 0

    # Preferred explicit alias (matches requested wording)
    cycle_index: Optional[int] = None

    notes: str = ""

    # Timestamps
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)

    # ------------------------------------------------------------------
    # Monotonic timing
    #
    # To provide a monotonic measure of uptime that is immune to system
    # clock changes, record the monotonic clock when the RunState is
    # instantiated.  The uptime_monotonic field captures the elapsed
    # monotonic seconds at termination.  These fields are excluded from
    # minimal state mode.
    started_monotonic: float = field(default_factory=time.monotonic)
    uptime_monotonic: Optional[float] = None

    # ------------------------------------------------------------------
    # Extended run termination metadata
    #
    # expected_cycles:
    #     The number of cycles the run was intended to execute (if known).
    # actual_cycles:
    #     The number of cycles actually executed when the run terminated.
    # stop_reason:
    #     A human readable reason for why the run stopped (e.g. "time_limit",
    #     "max_cycles", "repair_error", etc.).  This field is set by the
    #     engine loop or worker when the run concludes.
    # stop_source:
    #     The filename or subsystem that triggered the stop (e.g. "tgrm_loop.py").
    # uptime_seconds:
    #     How long the run executed in seconds.  Useful for diagnostics and
    #     reproducibility.  Not set until the run finishes.
    expected_cycles: Optional[int] = None
    actual_cycles: Optional[int] = None
    stop_reason: Optional[str] = None
    stop_source: Optional[str] = None
    uptime_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def new(
        cls,
        run_id: Optional[str] = None,
        *,
        total_cycles: int = 0,
        goal: str = "",
        domain: str = "general",
        mode: str = "",
        runs_root: Optional[Path] = None,
        ensure_dirs: bool = True,
    ) -> "RunState":
        rid = (str(run_id).strip() if run_id is not None else "") or str(uuid.uuid4())
        run_dir = str(resolve_run_dir(rid, runs_root=runs_root))
        st = cls(
            run_id=rid,
            run_dir=run_dir,
            status="queued",
            goal=goal or "",
            domain=domain or "general",
            mode=mode or "",
            phase_total=int(total_cycles) if total_cycles else 0,
            phase_index=None,
            phase_name=None,
            current=None,
            total=int(total_cycles) if total_cycles else 0,
            current_cycle=None,
            total_cycles=int(total_cycles) if total_cycles else 0,
            cycle_index=None,
            # Initialise extended termination metadata.  expected_cycles is
            # automatically set to the requested total_cycles so that the
            # engine knows how many cycles were planned.  Other fields are
            # left unset until the run finishes.
            expected_cycles=int(total_cycles) if total_cycles else None,
            actual_cycles=None,
            stop_reason=None,
            stop_source=None,
            uptime_seconds=None,
        )
        if ensure_dirs:
            try:
                Path(run_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return st

    @classmethod
    def new_running(
        cls,
        run_id: Optional[str] = None,
        *,
        total_cycles: int = 0,
        goal: str = "",
        domain: str = "general",
        mode: str = "",
        runs_root: Optional[Path] = None,
        ensure_dirs: bool = True,
    ) -> "RunState":
        st = cls.new(
            run_id=run_id,
            total_cycles=total_cycles,
            goal=goal,
            domain=domain,
            mode=mode,
            runs_root=runs_root,
            ensure_dirs=ensure_dirs,
        )
        # Mark the run as running and initialise the cycle index.  expected_cycles
        # is already set by cls.new(); do not override it here.
        st.status = "running"
        st.update_phase(0, name="cycle 0")
        return st

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "RunState":
        """Load RunState from disk.

        If run_state.json is not available yet (or the worker is running in a mode
        that doesn't persist full state continuously), fall back to the live
        progress.json artifacts written by the worker so the UI can display
        up-to-date cycle counts without parsing history logs.
        """

        p = Path(path)

        # 1) Preferred: run_state.json
        try:
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
                filtered = {k: v for k, v in raw.items() if k in allowed}
                st = cls(**filtered)
                st._sync_aliases()
                st._set_updated()
                return st
        except Exception:
            pass

        # 2) Fallback: live progress.json (per-run)
        try:
            run_dir = p.parent
            candidates = [
                run_dir / "progress.json",
                run_dir / "logs" / "progress.json",
            ]
            prog = None
            for cand in candidates:
                if cand.exists():
                    try:
                        prog = json.loads(cand.read_text(encoding="utf-8"))
                    except Exception:
                        prog = None
                    if isinstance(prog, dict):
                        break
                    prog = None

            if isinstance(prog, dict):
                rid = str(prog.get("run_id") or run_dir.name)
                st = cls(run_id=rid)

                # Populate the most important live fields
                cyc = prog.get("cycle_current")
                tot = prog.get("cycle_total")
                if isinstance(cyc, int):
                    st.cycle_index = cyc
                if isinstance(tot, int):
                    st.total_cycles = tot

                dom = prog.get("domain")
                if dom:
                    st.domain = str(dom)

                status = prog.get("status")
                if status:
                    st.status = str(status)

                note = prog.get("note")
                if note:
                    st.note = str(note)

                # Mark provenance
                st.extra = st.extra or {}
                st.extra["from_progress_file"] = True
                st.extra["progress_file"] = str(candidates[0])

                st._sync_aliases()
                st._set_updated()
                return st
        except Exception:
            pass

        raise FileNotFoundError(f"No run_state.json or progress.json found for {p}")


    def save(self, path: Optional[Path] = None, *, runs_root: Optional[Path] = None) -> Path:
        """Persist state to disk (atomic write). Returns the written path."""
        self.updated_at = _utc_iso()
        if path is None:
            # Default to per-run state file
            run_dir = self.ensure_run_dir(runs_root=runs_root)
            path = run_dir / "run_state.json"

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict.

        When MINIMAL_RUN_STATE is enabled, only persist the current cycle index.
        """
        if MINIMAL_RUN_STATE:
            # Prefer the explicit alias if set; otherwise fall back to phase_index.
            idx = self.cycle_index
            if idx is None:
                idx = self.phase_index
            if idx is None:
                idx = self.current_cycle
            if idx is None:
                idx = self.current
            try:
                ci = int(idx) if idx is not None else 0
            except Exception:
                ci = 0
            return {"cycle_index": ci}
        return asdict(self)

    # ------------------------------------------------------------------
    # Progress updates
    # ------------------------------------------------------------------
    def update_phase(self, index: int, *, name: Optional[str] = None) -> None:
        """Update phase/cycle counters consistently."""
        try:
            idx = int(index)
        except Exception:
            idx = 0

        self.phase_index = idx
        if name is not None:
            self.phase_name = str(name)

        # Back-compat aliases
        self.current = idx
        self.current_cycle = idx

        # Preferred explicit alias
        self.cycle_index = idx

    def mark_finished(self) -> None:
        """
        Mark the run as finished and populate termination metadata.  This
        method now records a final cycle count and uptime when they have
        not already been set.  It will not overwrite existing values on
        repeated calls, preserving the "write once" contract for the
        run ledger fields (expected_cycles, actual_cycles, stop_reason,
        stop_source, uptime_seconds).

        When called at the end of a run, it moves the phase index to
        the last cycle (if a total is known) and records the number of
        completed cycles and uptime if they are not already set.  Optionally
        one can provide a stop_reason and stop_source to capture why
        the run ended and which subsystem triggered it.
        """
        self.status = "finished"
        # Move to the last known phase if we have a total
        if self.phase_total > 0:
            last = max(0, int(self.phase_total) - 1)
            if self.phase_index is None or self.phase_index < last:
                self.update_phase(last, name=self.phase_name)

        # Derive actual_cycles if not already set.  Use the most
        # authoritative counter available.  The cycle_index property
        # reflects the zero-based index of the current cycle.  Add 1 to
        # convert to a count of completed cycles.  If cycle_index is
        # unavailable, fall back to phase_index or current.  Do not
        # override a preexisting value.
        if self.actual_cycles is None:
            idx: Optional[int] = None
            for cand in (self.cycle_index, self.phase_index, self.current_cycle, self.current):
                if cand is not None:
                    try:
                        idx = int(cand)
                    except Exception:
                        idx = None
                    if idx is not None:
                        break
            if idx is not None:
                # actual cycles is one more than zero-based index
                self.actual_cycles = max(0, idx + 1)

        # Compute uptime_seconds if unset.  Use created_at as the start
        # time and current UTC as end time.  Parsing errors are silently
        # ignored.
        if self.uptime_seconds is None:
            try:
                # Strip trailing Z for fromisoformat support
                start_str = self.created_at.rstrip("Z")
                start_dt = datetime.fromisoformat(start_str)
                end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
                delta = end_dt - start_dt
                self.uptime_seconds = delta.total_seconds()
            except Exception:
                pass

        # Compute monotonic uptime if unset.  Use the monotonic clock
        # captured at instantiation and the current monotonic reading.
        if self.uptime_monotonic is None:
            try:
                current_mono = time.monotonic()
                # started_monotonic is defined at initialisation; ensure it's
                # numeric and non-negative before subtracting.
                start_mono = float(self.started_monotonic) if self.started_monotonic is not None else 0.0
                if current_mono >= start_mono:
                    self.uptime_monotonic = current_mono - start_mono
                else:
                    # Guard against clock anomalies
                    self.uptime_monotonic = 0.0
            except Exception:
                # If monotonic clock is unavailable, leave unset
                pass

        # Do not modify stop_reason or stop_source here; those should be
        # recorded explicitly by the caller via record_termination() to
        # respect the writeâonce semantics.

    def record_termination(
        self,
        *,
        actual_cycles: Optional[int] = None,
        stop_reason: Optional[str] = None,
        stop_source: Optional[str] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Record termination metadata for the run.  This helper should be
        called once when the engine or worker decides to stop a run.

        Parameters
        ----------
        actual_cycles:
            The total number of cycles executed.  If None and not
            already set, the value will be derived from the run's
            current counters.  The value is ignored when actual_cycles
            is already populated.
        stop_reason:
            A human readable reason why the run stopped (e.g. "time_limit",
            "max_cycles", "repair_error").  Only set when stop_reason
            has not been previously recorded.
        stop_source:
            A string identifying the file or subsystem that triggered
            the stop (e.g. "tgrm_loop.py").  Only set when stop_source
            has not been previously recorded.
        end_time:
            A datetime representing when the run ended.  Defaults to
            current UTC time.  This is used to compute uptime_seconds
            if not already set.

        The writeâonce contract ensures that existing metadata values
        (actual_cycles, stop_reason, stop_source, uptime_seconds) are
        not overwritten by subsequent calls.
        """
        # Set actual_cycles only if unset
        if self.actual_cycles is None:
            ac = actual_cycles
            if ac is None:
                # Derive from current counters
                idx: Optional[int] = None
                for cand in (self.cycle_index, self.phase_index, self.current_cycle, self.current):
                    if cand is not None:
                        try:
                            idx = int(cand)
                        except Exception:
                            idx = None
                        if idx is not None:
                            break
                if idx is not None:
                    ac = max(0, idx + 1)
            if ac is not None:
                try:
                    self.actual_cycles = int(ac)
                except Exception:
                    pass
        # Set stop_reason if unset and provided
        if stop_reason and self.stop_reason is None:
            self.stop_reason = str(stop_reason)
        # Set stop_source if unset and provided
        if stop_source and self.stop_source is None:
            self.stop_source = str(stop_source)
        # Compute uptime_seconds if unset
        if self.uptime_seconds is None:
            try:
                et = end_time
                if et is None:
                    et = datetime.now(timezone.utc).replace(tzinfo=None)
                start_str = self.created_at.rstrip("Z")
                st = datetime.fromisoformat(start_str)
                delta = et - st
                self.uptime_seconds = delta.total_seconds()
            except Exception:
                pass

        # Compute monotonic uptime if unset
        if self.uptime_monotonic is None:
            try:
                current_mono = time.monotonic()
                start_mono = float(self.started_monotonic) if self.started_monotonic is not None else 0.0
                if current_mono >= start_mono:
                    self.uptime_monotonic = current_mono - start_mono
                else:
                    self.uptime_monotonic = 0.0
            except Exception:
                pass


__all__ = ["RunState", "resolve_runs_root", "resolve_run_dir", "default_state_path"]
