# agent/event_log.py
# -*- coding: utf-8 -*-
"""
File-based event log for ARA / Reparodynamics.

Why this exists
---------------
Streamlit reruns frequently and cannot "stream" from the worker process.
So the worker writes small JSON artifacts to a shared disk, and Streamlit
reads them on every rerun.

This module provides a simple, robust event timeline that supports:
- A narrative timeline feed in Streamlit
- Per-run event files + a global event file
- Atomic writes (so Streamlit never reads half-written JSON)
- Optional file locking (best-effort) to reduce concurrent write loss

Files written (default)
-----------------------
<run_jobs.BASE_DIR or ARA_RUNS_DIR>/logs/event_log.json
<run_jobs.BASE_DIR or ARA_RUNS_DIR>/logs/<run_id>_event_log.json

JSON format (dict container)
----------------------------
{
  "updated_at": "2025-01-01T12:34:56Z",
  "run_id": "optional",
  "events": [
    {
      "id": "...",
      "ts": 1734481234.12,
      "timestamp": "2025-01-01T12:34:56Z",
      "run_id": "abc123",
      "level": "info",
      "kind": "phase_start",
      "message": "Phase 2/3: stabilization",
      "role": "researcher",
      "domain": "longevity",
      "phase_index": 2,
      "phase_total": 3,
      "phase_name": "stabilization",
      "cycle": 17,
      "data": {...}
    }
  ]
}
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Best-effort POSIX file locks (Render is Linux; Windows will just skip locking)
try:  # pragma: no cover
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


JsonObj = Dict[str, Any]

DEFAULT_MAX_EVENTS: int = int(os.getenv("ARA_EVENT_LOG_MAX", "1000"))
DEFAULT_MIRROR_GLOBAL: bool = (
    os.getenv("ARA_EVENT_LOG_MIRROR_GLOBAL", "1").strip() not in {"0", "false", "False"}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_repo_root() -> Path:
    # agent/ is one level below repo root in typical layout
    return Path(__file__).resolve().parents[1]


def resolve_runs_root() -> Path:
    """Resolve the shared runs root directory.

    IMPORTANT: this order is chosen to match app_streamlit.py, so the worker and UI
    always agree about where shared artifacts live.

    Preference order:
    1) agent.run_jobs.BASE_DIR (if available)
    2) ARA_RUNS_DIR env var
    3) <repo_root>/runs
    """
    # 1) Prefer run_jobs.BASE_DIR so worker/UI stay in sync
    try:
        from .run_jobs import BASE_DIR as _BASE_DIR  # type: ignore

        if isinstance(_BASE_DIR, Path):
            return _BASE_DIR
        if isinstance(_BASE_DIR, str) and _BASE_DIR:
            return Path(_BASE_DIR)
    except Exception:
        pass

    # 2) Env fallback
    env = os.getenv("ARA_RUNS_DIR")
    if env:
        return Path(env)

    # 3) Repo fallback
    return _resolve_repo_root() / "runs"


def resolve_logs_dir(runs_root: Optional[Path] = None) -> Path:
    """Resolve logs dir. Allows override via ARA_LOGS_DIR."""
    env = os.getenv("ARA_LOGS_DIR")
    if env:
        return Path(env)
    rr = runs_root if isinstance(runs_root, Path) else resolve_runs_root()
    return rr / "logs"


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Non-fatal (e.g., read-only FS); callers will surface errors later if needed.
        pass


def get_event_log_path(run_id: Optional[str] = None, logs_dir: Optional[Path] = None) -> Path:
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir()
    if run_id:
        return ld / f"{run_id}_event_log.json"
    return ld / "event_log.json"


def _safe_json_load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _coerce_container(data: Any) -> JsonObj:
    """Normalize log file into a dict container with an 'events' list."""
    if isinstance(data, dict):
        ev = data.get("events")
        out = dict(data)
        out["events"] = ev if isinstance(ev, list) else []
        return out

    if isinstance(data, list):
        return {"updated_at": None, "run_id": None, "events": data}

    return {"updated_at": None, "run_id": None, "events": []}


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (write temp then replace)."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            # Not fatal; some environments may not support fsync.
            pass
    # Atomic replace on POSIX if same filesystem
    tmp.replace(path)


@contextmanager
def _file_lock(lock_path: Path):
    """Best-effort exclusive lock using fcntl on POSIX."""
    if fcntl is None:
        yield
        return

    _ensure_dir(lock_path.parent)
    fp = None
    try:
        fp = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        except Exception:
            # If locking fails, still proceed rather than crashing.
            pass
        yield
    finally:
        if fp is not None:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                fp.close()
            except Exception:
                pass


def _safe_int(v: Any) -> Optional[int]:
    """Best-effort convert to int. Never raises."""
    if v is None:
        return None
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            return int(float(s))  # handles "1.0"
        return int(v)
    except Exception:
        return None


def _sanitize_jsonable(value: Any) -> Any:
    """Recursively convert non-JSON-serializable objects into strings.

    Keeps structure for dicts/lists instead of collapsing whole containers into strings.
    """
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            try:
                key = str(k)
            except Exception:
                key = repr(k)
            out[key] = _sanitize_jsonable(v)
        return out

    if isinstance(value, (list, tuple)):
        return [_sanitize_jsonable(v) for v in value]

    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return str(value)
        except Exception:
            return repr(value)


def make_event(
    *,
    run_id: Optional[str],
    kind: str,
    message: str,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
    domain: Optional[str] = None,
    phase_index: Optional[Any] = None,
    phase_total: Optional[Any] = None,
    phase_name: Optional[str] = None,
    cycle: Optional[Any] = None,
) -> JsonObj:
    """Create a normalized event dict.

    This must be "never crash" â logging should not bring down the worker.
    """
    ev: JsonObj = {
        "id": uuid.uuid4().hex,
        "ts": float(time.time()),
        "timestamp": _utc_now_iso(),
        "run_id": str(run_id) if run_id else None,
        "level": str(level or "info"),
        "kind": str(kind or "event"),
        "message": str(message or ""),
        "role": str(role) if role is not None else None,
        "domain": str(domain) if domain is not None else None,
        "phase_index": _safe_int(phase_index),
        "phase_total": _safe_int(phase_total),
        "phase_name": str(phase_name) if phase_name is not None else None,
        "cycle": _safe_int(cycle),
        "data": _sanitize_jsonable(data) if data is not None else {},
    }
    return ev


def append_event_to_file(
    path: Path,
    event: JsonObj,
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
    run_id: Optional[str] = None,
) -> None:
    """Append event to a single event log file."""
    _ensure_dir(path.parent)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with _file_lock(lock_path):
        raw = _safe_json_load(path)
        container = _coerce_container(raw)

        events = container.get("events")
        if not isinstance(events, list):
            events = []
        events.append(event)

        if isinstance(max_events, int) and max_events > 0 and len(events) > max_events:
            events = events[-max_events:]

        container["events"] = events
        container["updated_at"] = _utc_now_iso()
        # Explicitly set run_id on container (None for global file, string for per-run file)
        container["run_id"] = str(run_id) if run_id else None

        _atomic_write_json(path, container)


def log_event(
    *,
    run_id: Optional[str],
    kind: str,
    message: str,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
    domain: Optional[str] = None,
    phase_index: Optional[Any] = None,
    phase_total: Optional[Any] = None,
    phase_name: Optional[str] = None,
    cycle: Optional[Any] = None,
    logs_dir: Optional[Path] = None,
    mirror_global: bool = DEFAULT_MIRROR_GLOBAL,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> JsonObj:
    """Create and append an event to per-run and/or global logs."""
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir()
    _ensure_dir(ld)

    event = make_event(
        run_id=run_id,
        kind=kind,
        message=message,
        level=level,
        data=data,
        role=role,
        domain=domain,
        phase_index=phase_index,
        phase_total=phase_total,
        phase_name=phase_name,
        cycle=cycle,
    )

    # Per-run file
    if run_id:
        per_run_path = get_event_log_path(run_id=run_id, logs_dir=ld)
        append_event_to_file(per_run_path, event, max_events=max_events, run_id=run_id)

    # Global mirror file (optional but very helpful for UI)
    if mirror_global or not run_id:
        global_path = get_event_log_path(run_id=None, logs_dir=ld)
        append_event_to_file(global_path, event, max_events=max_events, run_id=None)

    return event


def read_event_log(
    *,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    logs_dir: Optional[Path] = None,
) -> List[JsonObj]:
    """Read events from a run-specific or global log file."""
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir()
    path = get_event_log_path(run_id=run_id, logs_dir=ld)

    raw = _safe_json_load(path)
    container = _coerce_container(raw)
    events = container.get("events")
    if not isinstance(events, list):
        return []

    out: List[JsonObj] = [e for e in events if isinstance(e, dict)]
    if isinstance(limit, int) and limit > 0 and len(out) > limit:
        out = out[-limit:]
    return out


def clear_event_log(
    *,
    run_id: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> None:
    """Clear an event log file."""
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir()
    path = get_event_log_path(run_id=run_id, logs_dir=ld)
    _ensure_dir(path.parent)

    container: JsonObj = {
        "updated_at": _utc_now_iso(),
        "run_id": str(run_id) if run_id else None,
        "events": [],
    }
    _atomic_write_json(path, container)


@dataclass
class EventLogger:
    """Convenience wrapper for worker code (engine_worker/core_agent)."""

    run_id: Optional[str] = None
    logs_dir: Path = field(default_factory=resolve_logs_dir)
    mirror_global: bool = DEFAULT_MIRROR_GLOBAL
    max_events: int = DEFAULT_MAX_EVENTS


    # Compatibility with CoreAgent (which may pass `path=...` / `config=...` and expects .append/.log)
    path: Optional[Path] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Best-effort compatibility shim.

        - If `path` is provided (often ".../logs/event_log.jsonl"), treat its parent as logs_dir.
        - Leave existing behavior unchanged otherwise.
        """
        try:
            if self.path is not None:
                p = Path(self.path)
                # If path looks like a file, use parent dir; if it's already a dir, use it directly.
                self.logs_dir = p.parent if p.suffix else p
        except Exception:
            # Never allow init-time logging config to break the app.
            pass

    def append(self, evt: Any = None, **kwargs: Any) -> Optional[JsonObj]:
        """Append an event provided as a dict (or kwargs) into the legacy JSON log.

        CoreAgent calls logger.append(evt_dict). We normalize the dict into the legacy
        log_event(...) signature to keep event_log.json and <run_id>_event_log.json populated.
        """
        try:
            payload: Dict[str, Any] = {}
            if isinstance(evt, dict):
                payload.update(evt)
            if kwargs:
                payload.update(kwargs)

            kind = payload.get("kind") or payload.get("type") or payload.get("domain") or "event"
            message = payload.get("message") or payload.get("msg") or payload.get("text") or ""
            level = payload.get("level") or "info"

            run_id = payload.get("run_id") or self.run_id
            role = payload.get("role")
            domain = payload.get("domain")

            cycle = payload.get("cycle") if payload.get("cycle") is not None else payload.get("cycle_index")

            data = payload.get("data")
            if data is None and isinstance(payload.get("extra"), dict):
                data = payload.get("extra")

            if not isinstance(data, dict):
                data = None

            return log_event(
                run_id=str(run_id) if run_id is not None else None,
                kind=str(kind),
                message=str(message),
                level=str(level),
                data=data,
                role=str(role) if role is not None else None,
                domain=str(domain) if domain is not None else None,
                cycle=int(cycle) if isinstance(cycle, int) or (isinstance(cycle, str) and cycle.isdigit()) else None,
                logs_dir=self.logs_dir,
                mirror_global=self.mirror_global,
                max_events=self.max_events,
            )
        except Exception:
            return None

    def log(self, evt: Any = None, **kwargs: Any) -> Optional[JsonObj]:
        """Alias for append (some code prefers `.log(...)`)."""
        return self.append(evt, **kwargs)

    def emit(
        self,
        kind: str,
        message: str,
        *,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        phase_index: Optional[Any] = None,
        phase_total: Optional[Any] = None,
        phase_name: Optional[str] = None,
        cycle: Optional[Any] = None,
    ) -> JsonObj:
        return log_event(
            run_id=self.run_id,
            kind=kind,
            message=message,
            level=level,
            data=data,
            role=role,
            domain=domain,
            phase_index=phase_index,
            phase_total=phase_total,
            phase_name=phase_name,
            cycle=cycle,
            logs_dir=self.logs_dir,
            mirror_global=self.mirror_global,
            max_events=self.max_events,
        )

    def read(self, limit: Optional[int] = None) -> List[JsonObj]:
        return read_event_log(run_id=self.run_id, limit=limit, logs_dir=self.logs_dir)

    def clear(self) -> None:
        clear_event_log(run_id=self.run_id, logs_dir=self.logs_dir)
