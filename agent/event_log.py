"""
event_log.py
============

A single, structured event stream for the ARA runtime.

Core guarantees:
  * Every event contains:
      - run_id (non-null when a run exists)
      - cycle (int or None)
      - role (agent role / component)
      - kind (string, queryable)
      - timestamp (UTC ISO string)
      - data (object/dict)
  * Events are written as JSONL in append-only mode with flush, so Streamlit can tail.
  * A small legacy JSON-array mirror can optionally be written for backwards-compatible UIs.

File layout (default)
---------------------
  runs_root/
    <run_id>/
      events.jsonl          # per-run, append-only stream (primary)
    logs/
      events_global.jsonl   # optional global mirror stream
      event_log.json        # legacy global JSON array (optional)
      <run_id>_event_log.json  # legacy per-run JSON array (optional)

Environment variables
---------------------
  ARA_RUNS_DIR / ARA_RUNS_ROOT / RUNS_DIR:
      Override runs_root
  ARA_RUNS_LOGS_DIR / ARA_RUNS_LOG_DIR / ARA_LOGS_DIR:
      Override logs directory
  ARA_WRITE_LEGACY_EVENT_JSON:
      "1" to write legacy JSON-array logs (default: "1")
  ARA_EVENT_LOG_MAX_EVENTS:
      Max events kept in legacy JSON array (default: 1000)
  ARA_EVENT_LOG_MIRROR_GLOBAL:
      "1" to mirror per-run events to global jsonl/json (default: "1")
  ARA_EVENT_LOG_FSYNC:
      "1" to fsync() after each JSONL append (default: "0") Ã¢ÂÂ safer, slower
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


JsonObj = Dict[str, Any]

_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        if v is None:
            return default
        return int(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


WRITE_LEGACY_JSON: bool = _env_bool("ARA_WRITE_LEGACY_EVENT_JSON", False) or _env_bool("ARA_EVENT_LOG_WRITE_LEGACY_JSON", False)
DEFAULT_MAX_EVENTS: int = _env_int("ARA_EVENT_LOG_MAX_EVENTS", 1000)
DEFAULT_MIRROR_GLOBAL: bool = _env_bool("ARA_EVENT_LOG_MIRROR_GLOBAL", True)
_FSYNC: bool = _env_bool("ARA_EVENT_LOG_FSYNC", False)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def resolve_runs_root() -> Path:
    for k in ("ARA_RUNS_DIR", "ARA_RUNS_ROOT", "RUNS_DIR"):
        v = os.getenv(k)
        if v:
            try:
                return Path(v).expanduser().resolve()
            except Exception:
                pass

    try:
        from .run_jobs import BASE_DIR as _BASE_DIR  # type: ignore
        if isinstance(_BASE_DIR, Path):
            return _BASE_DIR
        if isinstance(_BASE_DIR, str) and _BASE_DIR.strip():
            return Path(_BASE_DIR).expanduser().resolve()
    except Exception:
        pass

    try:
        here = Path(__file__).resolve()
        return (here.parent.parent / "runs").resolve()
    except Exception:
        return Path("./runs").resolve()


def resolve_logs_dir(runs_root: Optional[Path] = None) -> Path:
    # Allow explicit override
    for k in ("ARA_RUNS_LOGS_DIR", "ARA_RUNS_LOG_DIR", "ARA_LOGS_DIR"):
        v = os.getenv(k)
        if v:
            try:
                return Path(v).expanduser().resolve()
            except Exception:
                pass
    root = runs_root if isinstance(runs_root, Path) else resolve_runs_root()
    return (root / "logs").resolve()


def resolve_run_dir(run_id: str, runs_root: Optional[Path] = None) -> Path:
    root = runs_root if isinstance(runs_root, Path) else resolve_runs_root()
    return (root / str(run_id)).resolve()


def get_events_jsonl_path(run_id: str, *, runs_root: Optional[Path] = None) -> Path:
    return resolve_run_dir(run_id, runs_root=runs_root) / "events.jsonl"


def get_global_events_jsonl_path(*, logs_dir: Optional[Path] = None, runs_root: Optional[Path] = None) -> Path:
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir(runs_root=runs_root)
    return ld / "events_global.jsonl"


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _safe_run_id(run_id: Optional[str]) -> Optional[str]:
    if run_id is None:
        return None
    try:
        s = str(run_id).strip()
    except Exception:
        return None
    if not s or s.lower() == "null":
        return None
    # prevent path traversal
    s = s.replace("/", "_").replace("\\", "_")
    return s


def _sanitize_jsonable(obj: Any) -> Any:
    # Keep this intentionally permissive; avoid raising inside logging.
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _sanitize_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize_jsonable(v) for v in obj]
        # Fallback: best-effort string
        return str(obj)
    except Exception:
        return None


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
    """Create a structured event object."""
    rid = _safe_run_id(run_id)
    ev: JsonObj = {
        "timestamp": _utc_iso(),
        "run_id": rid,
        "level": str(level or "info"),
        "kind": str(kind or "event"),
        "message": str(message or ""),
        # Optional aliases for compatibility with older UIs
        "msg": str(message or ""),
        "role": str(role) if role is not None else None,
        "domain": str(domain) if domain is not None else None,
        "phase_index": phase_index,
        "phase_total": phase_total,
        "phase_name": phase_name,
        "cycle": cycle,
        "data": _sanitize_jsonable(data or {}),
    }
    return ev


def get_event_log_path(*, run_id: Optional[str], logs_dir: Optional[Path] = None) -> Path:
    """Legacy JSON-array file (kept for backwards compatibility)."""
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir()
    rid = _safe_run_id(run_id)
    if rid:
        return ld / f"{rid}_event_log.json"
    return ld / "event_log.json"


def _append_jsonl(path: Path, event: JsonObj) -> None:
    """Append one JSON object as a line, flush, optional fsync."""
    _ensure_dir(path.parent)
    line = json.dumps(event, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        if _FSYNC:
            try:
                os.fsync(f.fileno())
            except Exception:
                pass


def append_event_to_file(path: Path, event: JsonObj, *, max_events: int = DEFAULT_MAX_EVENTS) -> None:
    """Append to a legacy JSON array file (atomic rewrite)."""
    try:
        existing: List[JsonObj] = []
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    existing = [e for e in raw if isinstance(e, dict)]
            except Exception:
                existing = []
        existing.append(event)
        if max_events and len(existing) > max_events:
            existing = existing[-max_events:]

        tmp = path.with_suffix(path.suffix + ".tmp")
        _ensure_dir(tmp.parent)
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Never crash due to logging.
        return


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
    """Create and append an event to JSONL (primary) and optional legacy JSON."""
    rid = _safe_run_id(run_id)
    runs_root = resolve_runs_root()
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir(runs_root=runs_root)

    event = make_event(
        run_id=rid,
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

    with _LOCK:
        # JSONL per-run (primary)
        if rid:
            try:
                run_dir = resolve_run_dir(rid, runs_root=runs_root)
                _ensure_dir(run_dir)
                _append_jsonl(run_dir / "events.jsonl", event)
            except Exception:
                pass

        # JSONL global mirror
        if mirror_global or not rid:
            try:
                _ensure_dir(ld)
                _append_jsonl(ld / "events_global.jsonl", event)
            except Exception:
                pass

        # Legacy JSON-array mirrors (optional)
        if WRITE_LEGACY_JSON:
            try:
                _ensure_dir(ld)
                if rid:
                    append_event_to_file(get_event_log_path(run_id=rid, logs_dir=ld), event, max_events=max_events)
                if mirror_global or not rid:
                    append_event_to_file(get_event_log_path(run_id=None, logs_dir=ld), event, max_events=max_events)
            except Exception:
                pass

    return event


def read_event_log(
    *,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    logs_dir: Optional[Path] = None,
) -> List[JsonObj]:
    """Read legacy JSON-array logs (for backwards compatibility)."""
    ld = logs_dir if isinstance(logs_dir, Path) else resolve_logs_dir()
    path = get_event_log_path(run_id=run_id, logs_dir=ld)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            return []
        events = [e for e in raw if isinstance(e, dict)]
        if limit is not None and limit > 0:
            return events[-int(limit):]
        return events
    except Exception:
        return []


def read_events_jsonl(
    *,
    run_id: str,
    runs_root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[JsonObj]:
    """Read per-run JSONL stream. Best-effort; ignores malformed lines."""
    rid = _safe_run_id(run_id) or ""
    if not rid:
        return []
    path = get_events_jsonl_path(rid, runs_root=runs_root)
    if not path.exists():
        return []
    out: List[JsonObj] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    if limit is not None and limit > 0 and len(out) > limit:
        out = out[-int(limit):]
    return out


@dataclass
class EventLogger:
    """Convenience wrapper for emitting events bound to a run_id."""

    run_id: Optional[str] = None
    role: Optional[str] = None
    domain: Optional[str] = None
    logs_dir: Optional[Path] = None
    mirror_global: bool = DEFAULT_MIRROR_GLOBAL
    max_events: int = DEFAULT_MAX_EVENTS

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
            role=role or self.role,
            domain=domain or self.domain,
            phase_index=phase_index,
            phase_total=phase_total,
            phase_name=phase_name,
            cycle=cycle,
            logs_dir=self.logs_dir,
            mirror_global=self.mirror_global,
            max_events=self.max_events,
        )


__all__ = [
    "JsonObj",
    "resolve_runs_root",
    "resolve_logs_dir",
    "resolve_run_dir",
    "get_events_jsonl_path",
    "get_global_events_jsonl_path",
    "make_event",
    "log_event",
    "read_event_log",
    "read_events_jsonl",
    "EventLogger",
]
