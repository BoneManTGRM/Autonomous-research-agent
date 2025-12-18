# agent/state_emitter.py
# -*- coding: utf-8 -*-
"""
Disk-based state emitter used by the worker to communicate with Streamlit.

It writes small JSON snapshots (heartbeat, worker state, run state, progress)
using atomic replace so readers never see partial files. All writes are best-effort:
a write failure must not crash the worker.

Typical outputs (under <runs_root>/logs):
- watchdog_heartbeat.json
- worker_state.json
- run_state.json
- <run_id>_progress.json
- event_log.json (optional narrative feed)

The module also writes several filename aliases because the UI may probe
multiple locations/names.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


try:
    # Optional: keep schema_version aligned with agent/state_schema.py if present.
    from .state_schema import SCHEMA_VERSION as _SCHEMA_VERSION  # type: ignore
except Exception:
    _SCHEMA_VERSION = "v1"


# -----------------------------
# Path resolution helpers
# -----------------------------
def _repo_root() -> Path:
    # agent/state_emitter.py -> repo root is parent of agent/
    return Path(__file__).resolve().parents[1]


def get_runs_root() -> Path:
    """Resolve a default runs directory.

    Priority:
    1) agent.run_jobs.BASE_DIR (if available)
    2) env var ARA_RUNS_DIR
    3) <repo_root>/runs
    """
    try:
        from . import run_jobs as _rj  # type: ignore

        base = getattr(_rj, "BASE_DIR", None)
        if isinstance(base, Path):
            return base
        if isinstance(base, str) and base.strip():
            return Path(base.strip())
    except Exception:
        pass

    env = os.getenv("ARA_RUNS_DIR")
    if env and env.strip():
        return Path(env.strip())

    return _repo_root() / "runs"


def get_queue_root() -> Path:
    """Resolve a default queue directory.

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
        if isinstance(qr, str) and qr.strip():
            return Path(qr.strip())

        pending = getattr(_rj, "PENDING_DIR", None)
        if isinstance(pending, Path):
            return pending.parent
    except Exception:
        pass

    env = os.getenv("ARA_QUEUE_ROOT")
    if env and env.strip():
        return Path(env.strip())

    return get_runs_root() / "queue"


def _utc_iso(timespec: str = "seconds") -> str:
    return datetime.now(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")


def _safe_run_id(run_id: str) -> str:
    rid = str(run_id).strip()
    rid = re.sub(r"[^a-zA-Z0-9_\-]", "_", rid)
    return rid or "run"


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Never crash the worker for directory creation issues.
        pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via atomic replace (best-effort)."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}_{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)  # atomic on POSIX; replace on Windows
    finally:
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


def _json_safe(value: Any, *, _depth: int = 0, _max_depth: int = 6) -> Any:
    """Convert values into something JSON-serializable, preserving structure when possible."""
    if _depth >= _max_depth:
        try:
            return str(value)
        except Exception:
            return repr(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        try:
            return _json_safe(asdict(value), _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            return str(value)

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            try:
                sk = str(k)
            except Exception:
                sk = repr(k)
            out[sk] = _json_safe(v, _depth=_depth + 1, _max_depth=_max_depth)
        return out

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth=_depth + 1, _max_depth=_max_depth) for v in value]

    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return str(value)
        except Exception:
            return repr(value)


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
        # Only mirror the canonical watchdog file into queue roots by default.
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
    """Writes heartbeat/state/progress/event artifacts to disk (safe + atomic)."""

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

        self._start_ts = float(time.time())
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

        # Small in-memory caches (optional convenience; does not affect persistence)
        self._last_worker_state: Dict[str, Any] = {}
        self._last_run_state: Dict[str, Any] = {}

    # -------------------------
    # Internals
    # -------------------------
    def _safe_write_many(self, paths: Sequence[Path], payload: Any) -> None:
        last_err: Optional[Exception] = None
        seen: set = set()
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            try:
                _atomic_write_json(p, payload)
            except Exception as e:
                last_err = e
                continue
        if last_err and self.raise_on_error:
            raise last_err

    def _now_meta(self) -> Dict[str, Any]:
        iso = _utc_iso()
        ts = float(time.time())
        return {
            "schema_version": _SCHEMA_VERSION,
            "timestamp": iso,
            "updated_at": iso,
            "ts": ts,
            "timestamp_unix": ts,
            "pid": int(os.getpid()),
        }

    # -------------------------
    # Heartbeat
    # -------------------------
    def emit_heartbeat(self, status: str = "running", run_id: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        """Update the watchdog heartbeat snapshot."""
        rid = run_id or self.run_id
        self._heartbeat_count += 1

        now_ts = float(time.time())
        iso = _utc_iso()

        payload: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "last_beat": iso,
            "timestamp": iso,  # alias some readers prefer
            "ts": now_ts,
            "heartbeat_ts": now_ts,  # alias
            "count": int(self._heartbeat_count),
            "beats": int(self._heartbeat_count),  # alias
            "status": str(status),
            "uptime_seconds": max(0.0, float(now_ts - self._start_ts)),
        }
        if rid:
            payload["run_id"] = rid
            payload.setdefault("job_id", rid)
            payload.setdefault("id", rid)

        payload.update({k: _json_safe(v) for k, v in extra.items()})

        self._safe_write_many(self.paths.heartbeat_paths(rid), payload)
        return payload

    # -------------------------
    # Worker state
    # -------------------------
    def emit_worker_state(
        self,
        *,
        status: str,
        run_id: Optional[str] = None,
        mode: Optional[str] = None,
        domain: Optional[str] = None,
        goal: Optional[str] = None,
        role: Optional[str] = None,
        phase_index: Optional[int] = None,
        phase_total: Optional[int] = None,
        phase_name: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        effective_current: Optional[int] = None,
        effective_total: Optional[int] = None,
        message: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Write/update the worker_state snapshot."""
        rid = run_id or self.run_id

        payload: Dict[str, Any] = {}
        payload.update(self._now_meta())
        payload["status"] = str(status)
        if rid:
            payload["run_id"] = rid
            payload.setdefault("job_id", rid)
            payload.setdefault("id", rid)

        if mode is not None:
            payload["mode"] = str(mode)
        if domain is not None:
            payload["domain"] = str(domain)
        if goal is not None:
            payload["goal"] = str(goal)
        if role is not None:
            payload["role"] = str(role)

        if phase_index is not None:
            payload["phase_index"] = int(phase_index)
            payload.setdefault("phase_current", int(phase_index))
        if phase_total is not None:
            payload["phase_total"] = int(phase_total)
            payload.setdefault("phase_count", int(phase_total))
            payload.setdefault("total_phases", int(phase_total))
        if phase_name is not None:
            payload["phase_name"] = str(phase_name)

        if current is not None:
            payload["current"] = int(current)
            payload.setdefault("cycle", int(current))
            payload.setdefault("cycle_index", int(current))
            payload.setdefault("current_cycle", int(current))
        if total is not None:
            payload["total"] = int(total)
            payload.setdefault("total_cycles", int(total))
            payload.setdefault("max_cycles", int(total))
        if effective_current is not None:
            payload["effective_current"] = int(effective_current)
        if effective_total is not None:
            payload["effective_total"] = int(effective_total)

        if message is not None:
            payload["message"] = str(message)

        payload.update({k: _json_safe(v) for k, v in extra.items()})

        self._last_worker_state = payload
        self._safe_write_many(self.paths.worker_state_paths(rid), payload)
        return payload

    # -------------------------
    # Run state
    # -------------------------
    def emit_run_state(self, state: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Write/update the run_state snapshot (arbitrary dict)."""
        rid = run_id or self.run_id
        payload: Dict[str, Any] = dict(state) if isinstance(state, dict) else {"state": state}

        payload.setdefault("schema_version", _SCHEMA_VERSION)
        if rid:
            payload.setdefault("run_id", rid)
            payload.setdefault("job_id", rid)
            payload.setdefault("id", rid)

        payload.setdefault("timestamp", _utc_iso())
        payload.setdefault("updated_at", payload.get("timestamp"))
        payload.setdefault("ts", float(time.time()))
        payload.setdefault("timestamp_unix", payload.get("ts"))

        payload = _json_safe(payload)  # recursive sanitize

        self._last_run_state = payload if isinstance(payload, dict) else {"state": payload}
        self._safe_write_many(self.paths.run_state_paths(rid), self._last_run_state)
        return self._last_run_state

    # -------------------------
    # Progress-only snapshot
    # -------------------------
    def emit_progress(
        self,
        *,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        phase_index: Optional[int] = None,
        phase_total: Optional[int] = None,
        phase_name: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        effective_current: Optional[int] = None,
        effective_total: Optional[int] = None,
        message: Optional[str] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Write <run_id>_progress.json (small and frequently updated)."""
        rid = run_id or self.run_id
        if not rid:
            return None

        payload: Dict[str, Any] = {}
        payload.update(self._now_meta())
        payload["run_id"] = rid
        payload.setdefault("job_id", rid)
        payload.setdefault("id", rid)

        if status is not None:
            payload["status"] = str(status)

        if phase_index is not None:
            payload["phase_index"] = int(phase_index)
            payload.setdefault("phase_current", int(phase_index))
        if phase_total is not None:
            payload["phase_total"] = int(phase_total)
            payload.setdefault("phase_count", int(phase_total))
            payload.setdefault("total_phases", int(phase_total))
        if phase_name is not None:
            payload["phase_name"] = str(phase_name)

        if current is not None:
            payload["current"] = int(current)
            payload.setdefault("cycle", int(current))
            payload.setdefault("cycle_index", int(current))
            payload.setdefault("current_cycle", int(current))
        if total is not None:
            payload["total"] = int(total)
            payload.setdefault("total_cycles", int(total))
            payload.setdefault("max_cycles", int(total))

        if effective_current is not None:
            payload["effective_current"] = int(effective_current)
        if effective_total is not None:
            payload["effective_total"] = int(effective_total)

        if message is not None:
            payload["message"] = str(message)

        payload.update({k: _json_safe(v) for k, v in extra.items()})

        self._safe_write_many(self.paths.progress_paths(rid), payload)
        return payload

    # -------------------------
    # Narrative events
    # -------------------------
    def emit_event(
        self,
        *,
        kind: str,
        message: str,
        run_id: Optional[str] = None,
        level: str = "info",
        role: Optional[str] = None,
        domain: Optional[str] = None,
        phase_index: Optional[int] = None,
        phase_total: Optional[int] = None,
        phase_name: Optional[str] = None,
        cycle: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Append one entry into the narrative event log (bounded by max_events)."""
        rid = run_id or self.run_id

        now_unix = float(time.time())
        now_iso = _utc_iso()

        ev: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "id": uuid.uuid4().hex,
            "kind": str(kind),
            "type": str(kind),  # alias for some viewers
            "level": str(level),
            "message": str(message),
            "text": str(message),  # alias
            "ts": now_unix,
            "timestamp": now_iso,
            "ts_iso": now_iso,
            "unix_ts": now_unix,
        }
        if rid:
            ev["run_id"] = rid
            ev.setdefault("job_id", rid)

        if role is not None:
            ev["role"] = str(role)
        if domain is not None:
            ev["domain"] = str(domain)
        if phase_index is not None:
            ev["phase_index"] = int(phase_index)
        if phase_total is not None:
            ev["phase_total"] = int(phase_total)
        if phase_name is not None:
            ev["phase_name"] = str(phase_name)
        if cycle is not None:
            ev["cycle"] = int(cycle)
        if data is not None:
            ev["data"] = _json_safe(data)

        ev.update({k: _json_safe(v) for k, v in extra.items()})

        paths = self.paths.event_log_paths(rid)
        primary = paths[0] if paths else (self.logs_dir / "event_log.json")

        existing = _read_json(primary)
        events: List[Dict[str, Any]] = []

        if isinstance(existing, dict):
            maybe = existing.get("events") or existing.get("timeline") or existing.get("items")
            if isinstance(maybe, list):
                events = [x for x in maybe if isinstance(x, dict)]
        elif isinstance(existing, list):
            events = [x for x in existing if isinstance(x, dict)]

        events.append(ev)
        if self.max_events > 0 and len(events) > self.max_events:
            events = events[-self.max_events :]

        payload: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "updated_at": now_iso,
            "ts": now_unix,
            "run_id": rid,
            "events": events,
            "timeline": events,  # alias
            "items": events,  # alias
            "count": len(events),
        }

        self._safe_write_many(paths, payload)
        return ev

    # -------------------------
    # Convenience helpers
    # -------------------------
    def set_run_id(self, run_id: str, mirror_to_run_dir: bool = True) -> None:
        """Set/replace the run id after initialization."""
        self.run_id = str(run_id)
        if mirror_to_run_dir:
            self.run_dir = self.runs_root / _safe_run_id(self.run_id)
            _ensure_dir(self.run_dir)
            self.paths.run_dir = self.run_dir

    def mark_phase_start(
        self,
        *,
        phase_index: int,
        phase_total: int,
        phase_name: str,
        status: str = "running",
        role: Optional[str] = None,
        domain: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Call at the start of each phase so the UI updates immediately.

        If your UI expects 0-based phases, emit 0..phase_total-1.
        """
        self.emit_worker_state(
            status=status,
            run_id=self.run_id,
            phase_index=phase_index,
            phase_total=phase_total,
            phase_name=phase_name,
            role=role,
            domain=domain,
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
        self.emit_event(
            kind="phase_start",
            message=f"Phase {phase_index + 1}/{phase_total}: {phase_name}",
            run_id=self.run_id,
            role=role,
            domain=domain,
            phase_index=phase_index,
            phase_total=phase_total,
            phase_name=phase_name,
            **extra,
        )

    def mark_cycle(
        self,
        *,
        current: int,
        total: Optional[int] = None,
        status: str = "running",
        cycle_label: Optional[str] = None,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Optional per-cycle progress updates."""
        msg = cycle_label or (
            "Cycle " + str(current) + ("/" + str(total) if isinstance(total, int) and total > 0 else "")
        )

        self.emit_worker_state(
            status=status,
            run_id=self.run_id,
            current=current,
            total=total,
            message=msg,
            role=role,
            domain=domain,
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
        self.emit_event(
            kind="cycle",
            message=msg,
            run_id=self.run_id,
            role=role,
            domain=domain,
            cycle=current,
            **extra,
        )
