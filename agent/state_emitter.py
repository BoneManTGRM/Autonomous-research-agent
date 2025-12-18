# agent/state_emitter.py
# -*- coding: utf-8 -*-
"""
State emitter for the ARA engine worker.

Purpose
-------
This module writes small JSON "state artifacts" to disk so Streamlit can show:
- A sticky heartbeat/status bar (always visible)
- Live progress (1/3 → 2/3 → 3/3) via phase_index/phase_total
- Run diagnostics (worker_state, run_state, heartbeat)
- Optional narrative event feed (event_log.json)

Design goals
------------
- File-based and Render/shared-disk friendly
- Atomic writes (no half-written JSON that breaks Streamlit reads)
- Safe: failures to write state must NEVER crash the worker

Canonical locations written
---------------------------
<runs_root>/logs/watchdog_heartbeat.json
<runs_root>/logs/worker_state.json
<runs_root>/logs/run_state.json
<runs_root>/logs/event_log.json (optional)
<runs_root>/logs/<run_id>_progress.json (optional)

Also writes a few aliases that the UI searches for:
- heartbeat.json, worker_heartbeat.json
- engine_worker_state.json, worker_status.json
- last_run_state.json
- events.json
- per-run variants: <run_id>_worker_state.json, <run_id>_run_state.json, <run_id>_event_log.json, <run_id>_heartbeat.json
- per-run dir variants: <runs_root>/<run_id>/state.json, worker_state.json, run_state.json, heartbeat.json, progress.json, event_log.json

Phase indexing
--------------
The Streamlit UI you shared contains a heuristic that assumes phases might be
0-based (0..phase_total-1) and will display them as 1..phase_total.

So: emit phase_index as 0-based for best results:
- exploration:     phase_index=0, phase_total=3
- stabilization:   phase_index=1, phase_total=3
- refinement:      phase_index=2, phase_total=3
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# -----------------------------
# Path resolution helpers
# -----------------------------
def _repo_root() -> Path:
    # agent/state_emitter.py -> repo root is parent of agent/
    return Path(__file__).resolve().parents[1]


def get_runs_root() -> Path:
    """Resolve runs root directory in the same spirit as the Streamlit UI.

    Priority:
    1) agent.run_jobs.BASE_DIR (if available)
    2) env var ARA_RUNS_DIR
    3) <repo_root>/runs
    """
    # Try to reuse run_jobs constants without creating hard import cycles.
    try:
        from . import run_jobs as _rj  # type: ignore

        base = getattr(_rj, "BASE_DIR", None)
        if isinstance(base, Path):
            return base
        if isinstance(base, str) and base:
            return Path(base)
    except Exception:
        pass

    env = os.getenv("ARA_RUNS_DIR")
    if env:
        return Path(env)

    return _repo_root() / "runs"


def get_queue_root() -> Path:
    """Resolve queue root directory.

    Priority:
    1) agent.run_jobs.QUEUE_ROOT (if available)
    2) env var ARA_QUEUE_ROOT
    3) <runs_root>/queue
    """
    try:
        from . import run_jobs as _rj  # type: ignore

        qr = getattr(_rj, "QUEUE_ROOT", None)
        if isinstance(qr, Path):
            return qr
        if isinstance(qr, str) and qr:
            return Path(qr)

        pending = getattr(_rj, "PENDING_DIR", None)
        if isinstance(pending, Path):
            return pending.parent
    except Exception:
        pass

    env = os.getenv("ARA_QUEUE_ROOT")
    if env:
        return Path(env)

    return get_runs_root() / "queue"


def _utc_iso(timespec: str = "seconds") -> str:
    return datetime.now(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")


def _safe_run_id(run_id: str) -> str:
    # Prevent odd filenames if someone passes a weird run_id.
    rid = str(run_id).strip()
    rid = re.sub(r"[^a-zA-Z0-9_\-]", "_", rid)
    return rid or "run"


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Never crash worker on directory creation issues
        pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so Streamlit never reads half a file."""
    _ensure_dir(path.parent)

    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}_{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        # If something went wrong before replace, try cleanup
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------
# Public API
# -----------------------------
@dataclass
class EmitPaths:
    """Computed output paths used by a StateEmitter instance."""

    logs_dir: Path
    run_dir: Optional[Path]
    queue_root: Optional[Path]
    mirror_to_queue: bool = True

    def _maybe_queue(self, rel: str) -> List[Path]:
        if not self.mirror_to_queue or not self.queue_root:
            return []
        return [
            self.queue_root / rel,
            self.queue_root / "active" / rel,
            self.queue_root / "finished" / rel,
        ]

    def heartbeat_paths(self, run_id: Optional[str]) -> List[Path]:
        paths: List[Path] = [
            self.logs_dir / "watchdog_heartbeat.json",
            self.logs_dir / "heartbeat.json",
            self.logs_dir / "worker_heartbeat.json",
        ]
        paths.extend(self._maybe_queue("watchdog_heartbeat.json"))

        if run_id:
            rid = _safe_run_id(run_id)
            paths.append(self.logs_dir / f"{rid}_heartbeat.json")
            if self.run_dir:
                paths.append(self.run_dir / "heartbeat.json")
        return paths

    def worker_state_paths(self, run_id: Optional[str]) -> List[Path]:
        paths: List[Path] = [
            self.logs_dir / "worker_state.json",
            self.logs_dir / "engine_worker_state.json",
            self.logs_dir / "worker_status.json",
        ]
        paths.extend(self._maybe_queue("worker_state.json"))

        if run_id:
            rid = _safe_run_id(run_id)
            paths.append(self.logs_dir / f"{rid}_worker_state.json")
            paths.append(self.logs_dir / f"{rid}_state.json")
            if self.run_dir:
                paths.append(self.run_dir / "worker_state.json")
                paths.append(self.run_dir / "state.json")
        return paths

    def run_state_paths(self, run_id: Optional[str]) -> List[Path]:
        paths: List[Path] = [
            self.logs_dir / "run_state.json",
            self.logs_dir / "last_run_state.json",
        ]
        paths.extend(self._maybe_queue("run_state.json"))

        if run_id:
            rid = _safe_run_id(run_id)
            paths.append(self.logs_dir / f"{rid}_run_state.json")
            paths.append(self.logs_dir / f"{rid}_runstate.json")
            if self.run_dir:
                paths.append(self.run_dir / "run_state.json")
        return paths

    def progress_paths(self, run_id: Optional[str]) -> List[Path]:
        if not run_id:
            return []
        rid = _safe_run_id(run_id)
        paths: List[Path] = [self.logs_dir / f"{rid}_progress.json"]
        # also mirror to queue root because Streamlit checks there
        if self.mirror_to_queue and self.queue_root:
            paths.extend(
                [
                    self.queue_root / f"{rid}_progress.json",
                    self.queue_root / "active" / f"{rid}_progress.json",
                    self.queue_root / "finished" / f"{rid}_progress.json",
                ]
            )
        if self.run_dir:
            paths.append(self.run_dir / "progress.json")
        return paths

    def event_log_paths(self, run_id: Optional[str]) -> List[Path]:
        paths: List[Path] = [
            self.logs_dir / "event_log.json",
            self.logs_dir / "events.json",
            self.logs_dir / "timeline.json",
        ]
        paths.extend(self._maybe_queue("event_log.json"))

        if run_id:
            rid = _safe_run_id(run_id)
            paths.append(self.logs_dir / f"{rid}_event_log.json")
            paths.append(self.logs_dir / f"{rid}_events.json")
            if self.run_dir:
                paths.append(self.run_dir / "event_log.json")
                paths.append(self.run_dir / "events.json")
        return paths


class StateEmitter:
    """Writes worker heartbeat/state/progress artifacts to disk (safe + atomic)."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        runs_root: Optional[Union[str, Path]] = None,
        queue_root: Optional[Union[str, Path]] = None,
        mirror_to_queue: bool = True,
        mirror_to_run_dir: bool = True,
        raise_on_error: bool = False,
        max_events: int = 500,
    ) -> None:
        self.raise_on_error = bool(raise_on_error)
        self.max_events = int(max_events)

        self.runs_root = Path(runs_root) if runs_root is not None else get_runs_root()
        self.queue_root = Path(queue_root) if queue_root is not None else get_queue_root()

        self.logs_dir = self.runs_root / "logs"
        _ensure_dir(self.logs_dir)

        self._start_ts = time.time()
        self._heartbeat_count = 0

        self.run_id: Optional[str] = str(run_id) if run_id else None
        self.run_dir: Optional[Path] = None
        if self.run_id and mirror_to_run_dir:
            self.run_dir = self.runs_root / _safe_run_id(self.run_id)
            _ensure_dir(self.run_dir)

        self.paths = EmitPaths(
            logs_dir=self.logs_dir,
            run_dir=self.run_dir,
            queue_root=self.queue_root if mirror_to_queue else None,
            mirror_to_queue=mirror_to_queue,
        )

        # Keep a small cache for convenience (doesn't affect writes)
        self._last_worker_state: Dict[str, Any] = {}
        self._last_run_state: Dict[str, Any] = {}

    # --------
    # Internals
    # --------
    def _safe_write_many(self, paths: Sequence[Path], payload: Any) -> None:
        last_err: Optional[Exception] = None
        for p in paths:
            try:
                _atomic_write_json(p, payload)
            except Exception as e:
                last_err = e
                # keep going; state is best-effort
                continue
        if last_err and self.raise_on_error:
            raise last_err

    def _now_meta(self) -> Dict[str, Any]:
        return {
            "timestamp": _utc_iso(),
            "ts": time.time(),
            "pid": os.getpid(),
        }

    # --------
    # Public: heartbeat
    # --------
    def emit_heartbeat(self, status: str = "running", run_id: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        """Write/update watchdog heartbeat JSON.

        Streamlit uses this to compute "seconds since last beat".
        """
        rid = run_id or self.run_id
        self._heartbeat_count += 1

        payload: Dict[str, Any] = {
            "last_beat": _utc_iso(),
            "ts": time.time(),
            "count": int(self._heartbeat_count),
            "run_id": rid,
            "status": status,
            "uptime_seconds": max(0.0, time.time() - self._start_ts),
        }
        payload.update(extra)

        self._safe_write_many(self.paths.heartbeat_paths(rid), payload)
        return payload

    # --------
    # Public: worker state
    # --------
    def emit_worker_state(
        self,
        *,
        status: str,
        run_id: Optional[str] = None,
        mode: Optional[str] = None,
        domain: Optional[str] = None,
        goal: Optional[str] = None,
        role: Optional[str] = None,
        # Phase progress (recommended for 1/3 → 2/3 → 3/3)
        phase_index: Optional[int] = None,
        phase_total: Optional[int] = None,
        phase_name: Optional[str] = None,
        # Cycle progress (optional)
        current: Optional[int] = None,
        total: Optional[int] = None,
        effective_current: Optional[int] = None,
        effective_total: Optional[int] = None,
        message: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Write/update worker_state.json."""
        rid = run_id or self.run_id

        payload: Dict[str, Any] = {}
        payload.update(self._now_meta())
        payload.update(
            {
                "run_id": rid,
                "status": status,
            }
        )

        if mode is not None:
            payload["mode"] = mode
        if domain is not None:
            payload["domain"] = domain
        if goal is not None:
            payload["goal"] = goal
        if role is not None:
            payload["role"] = role

        # Phase progress
        if phase_index is not None:
            payload["phase_index"] = int(phase_index)
        if phase_total is not None:
            payload["phase_total"] = int(phase_total)
        if phase_name is not None:
            payload["phase_name"] = str(phase_name)

        # Cycle progress
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if effective_current is not None:
            payload["effective_current"] = int(effective_current)
        if effective_total is not None:
            payload["effective_total"] = int(effective_total)

        if message is not None:
            payload["message"] = str(message)

        payload.update(extra)

        self._last_worker_state = payload
        self._safe_write_many(self.paths.worker_state_paths(rid), payload)
        return payload

    # --------
    # Public: run state (resume/diagnostics)
    # --------
    def emit_run_state(self, state: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Write/update run_state.json (arbitrary dict; used for resume and diagnostics)."""
        rid = run_id or self.run_id
        payload: Dict[str, Any] = dict(state) if isinstance(state, dict) else {"state": state}

        # Normalize a few common fields
        payload.setdefault("run_id", rid)
        payload.setdefault("timestamp", _utc_iso())
        payload.setdefault("ts", time.time())

        self._last_run_state = payload
        self._safe_write_many(self.paths.run_state_paths(rid), payload)
        return payload

    # --------
    # Public: progress JSON (smooth progress updates)
    # --------
    def emit_progress(
        self,
        *,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        # Phase
        phase_index: Optional[int] = None,
        phase_total: Optional[int] = None,
        phase_name: Optional[str] = None,
        # Cycle
        current: Optional[int] = None,
        total: Optional[int] = None,
        effective_current: Optional[int] = None,
        effective_total: Optional[int] = None,
        message: Optional[str] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Write <run_id>_progress.json.

        This file is intentionally tiny and should be written:
        - at phase start
        - at cycle start/end (optional)
        """
        rid = run_id or self.run_id
        if not rid:
            return None

        payload: Dict[str, Any] = {}
        payload.update(self._now_meta())
        payload["run_id"] = rid

        if status is not None:
            payload["status"] = status

        if phase_index is not None:
            payload["phase_index"] = int(phase_index)
        if phase_total is not None:
            payload["phase_total"] = int(phase_total)
        if phase_name is not None:
            payload["phase_name"] = str(phase_name)

        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if effective_current is not None:
            payload["effective_current"] = int(effective_current)
        if effective_total is not None:
            payload["effective_total"] = int(effective_total)

        if message is not None:
            payload["message"] = str(message)

        payload.update(extra)

        self._safe_write_many(self.paths.progress_paths(rid), payload)
        return payload

    # --------
    # Public: events (narrative timeline)
    # --------
    def emit_event(
        self,
        *,
        kind: str,
        message: str,
        run_id: Optional[str] = None,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        phase_name: Optional[str] = None,
        cycle: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Append an event to event_log.json (bounded to max_events)."""
        rid = run_id or self.run_id

        ev: Dict[str, Any] = {
            "ts": _utc_iso(),
            "kind": str(kind),
            "message": str(message),
        }
        if rid:
            ev["run_id"] = rid
        if role is not None:
            ev["role"] = role
        if domain is not None:
            ev["domain"] = domain
        if phase_name is not None:
            ev["phase_name"] = phase_name
        if cycle is not None:
            ev["cycle"] = int(cycle)
        if data is not None and isinstance(data, dict):
            ev["data"] = data

        ev.update(extra)

        # Load existing events (best-effort), append, keep tail, write atomically
        paths = self.paths.event_log_paths(rid)
        primary = paths[0] if paths else (self.logs_dir / "event_log.json")
        existing = _read_json(primary)

        events: List[Dict[str, Any]] = []
        if isinstance(existing, dict):
            maybe = existing.get("events")
            if isinstance(maybe, list):
                events = [x for x in maybe if isinstance(x, dict)]
        elif isinstance(existing, list):
            events = [x for x in existing if isinstance(x, dict)]

        events.append(ev)
        if self.max_events > 0 and len(events) > self.max_events:
            events = events[-self.max_events :]

        payload: Union[List[Dict[str, Any]], Dict[str, Any]]
        # Keep it simple: just a list
        payload = events

        self._safe_write_many(paths, payload)
        return ev

    # --------
    # Convenience helpers for the worker
    # --------
    def set_run_id(self, run_id: str, mirror_to_run_dir: bool = True) -> None:
        """Update run_id after init (useful if emitter created before job is claimed)."""
        self.run_id = str(run_id)
        if mirror_to_run_dir:
            self.run_dir = self.runs_root / _safe_run_id(self.run_id)
            _ensure_dir(self.run_dir)
            # update paths to include run_dir
            self.paths.run_dir = self.run_dir

    def mark_phase_start(
        self,
        *,
        phase_index: int,
        phase_total: int,
        phase_name: str,
        status: str = "running",
        **extra: Any,
    ) -> None:
        """One call to make the UI show 1/3 → 2/3 → 3/3 smoothly.

        NOTE: Use 0-based phase_index (0..phase_total-1) with your current Streamlit UI.
        """
        self.emit_worker_state(
            status=status,
            run_id=self.run_id,
            phase_index=phase_index,
            phase_total=phase_total,
            phase_name=phase_name,
            **extra,
        )
        self.emit_progress(
            status=status,
            run_id=self.run_id,
            phase_index=phase_index,
            phase_total=phase_total,
            phase_name=phase_name,
            **extra,
        )
        self.emit_event(kind="phase_start", message=f"Phase {phase_index + 1}/{phase_total}: {phase_name}", **extra)

    def mark_cycle(
        self,
        *,
        current: int,
        total: Optional[int] = None,
        status: str = "running",
        cycle_label: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Update cycle progress (optional)."""
        msg = cycle_label or f"Cycle {current}" + (f"/{total}" if isinstance(total, int) and total > 0 else "")
        self.emit_worker_state(
            status=status,
            run_id=self.run_id,
            current=current,
            total=total,
            message=msg,
            **extra,
        )
        self.emit_progress(
            status=status,
            run_id=self.run_id,
            current=current,
            total=total,
            message=msg,
            **extra,
        )
        self.emit_event(kind="cycle", message=msg, cycle=current, **extra)
