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
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    """Return current UTC timestamp in ISO 8601 (seconds precision)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def resolve_runs_root() -> Path:
    """Resolve the base runs directory.

    Preference order:
      1) env ARA_RUNS_DIR / ARA_RUNS_ROOT / RUNS_DIR
      2) run_jobs.BASE_DIR if available
      3) repo-local ./runs (next to this file's parent package)
    """
    for k in ("ARA_RUNS_DIR", "ARA_RUNS_ROOT", "RUNS_DIR"):
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
        st.status = "running"
        st.update_phase(0, name="cycle 0")
        return st

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "RunState":
        """Load a RunState from a JSON file, ignoring unknown keys."""
        with path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in (data or {}).items() if k in field_names}
        st = cls(**kwargs)  # type: ignore[arg-type]
        # Ensure run_id non-empty
        try:
            if not str(st.run_id).strip():
                st.run_id = str(uuid.uuid4())
        except Exception:
            st.run_id = str(uuid.uuid4())
        # Ensure run_dir present
        if not (isinstance(st.run_dir, str) and st.run_dir.strip()):
            st.run_dir = str(resolve_run_dir(st.run_id))
        return st

    def ensure_run_dir(self, *, runs_root: Optional[Path] = None) -> Path:
        """Ensure this run has a directory and return it."""
        rid = str(self.run_id).strip() or str(uuid.uuid4())
        self.run_id = rid
        rd = self.run_dir if isinstance(self.run_dir, str) and self.run_dir.strip() else str(resolve_run_dir(rid, runs_root=runs_root))
        self.run_dir = rd
        p = Path(rd)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

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
        """Return a JSON-serialisable dict."""
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
        self.status = "finished"
        # Move to the last known phase if we have a total
        if self.phase_total > 0:
            last = max(0, int(self.phase_total) - 1)
            if self.phase_index is None or self.phase_index < last:
                self.update_phase(last, name=self.phase_name)


__all__ = ["RunState", "resolve_runs_root", "resolve_run_dir", "default_state_path"]
