# -*- coding: utf-8 -*-
"""
agent/state_schema.py

Shared, stable JSON schemas for the ARA worker <-> Streamlit UI "reparodynamic"
telemetry layer.

Goals:
- Be dependency-light (stdlib only).
- Be backward/forward compatible (accept multiple key aliases).
- Provide simple dataclasses + helpers to emit and parse:
    - watchdog heartbeat
    - worker state
    - run state
    - progress state
    - narrative event log entries

This module does NOT write files by itself. File I/O lives in agent/state_emitter.py
and agent/event_log.py.

Recommended canonical filenames (written under <runs_root>/logs/):
- watchdog_heartbeat.json
- worker_state.json
- run_state.json
- event_log.json
- <run_id>_progress.json

The Streamlit UI you posted already checks many of these names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

# -----------------------------
# Types / constants
# -----------------------------
JSONDict = Dict[str, Any]

SCHEMA_VERSION: str = "v1"

# Canonical filenames (state_emitter/event_log will use these)
WATCHDOG_HEARTBEAT_FILENAME: str = "watchdog_heartbeat.json"
WORKER_STATE_FILENAME: str = "worker_state.json"
RUN_STATE_FILENAME: str = "run_state.json"
EVENT_LOG_FILENAME: str = "event_log.json"
PROGRESS_SUFFIX: str = "_progress.json"

# Common status strings (not enforced, but recommended)
STATUS_QUEUED = "queued"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_ACTIVE = "active"
STATUS_WORKING = "working"
STATUS_BACKOFF = "backoff"
STATUS_FINISHED = "finished"
STATUS_DONE = "done"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"
STATUS_FAILED = "failed"
STATUS_IDLE = "idle"
STATUS_UNKNOWN = "unknown"


# -----------------------------
# Time helpers
# -----------------------------
def utc_now_ts() -> float:
    """Unix seconds (float) in UTC."""
    return datetime.now(tz=timezone.utc).timestamp()


def utc_now_iso(timespec: str = "seconds") -> str:
    """ISO 8601 timestamp with trailing Z."""
    return datetime.now(tz=timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")


def iso_from_ts(ts: Union[int, float], timespec: str = "seconds") -> str:
    try:
        return (
            datetime.fromtimestamp(float(ts), tz=timezone.utc)
            .isoformat(timespec=timespec)
            .replace("+00:00", "Z")
        )
    except Exception:
        return utc_now_iso(timespec=timespec)


def parse_iso(ts: Any) -> Optional[datetime]:
    """Best-effort ISO parsing including trailing Z."""
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


# -----------------------------
# Small coercion helpers
# -----------------------------
def _pick(d: Any, keys: Tuple[str, ...], default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d.get(k)
    return default


def safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return default
            return int(float(s))
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return default
            return float(s)
        return float(v)
    except Exception:
        return default


def safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    try:
        s = str(v)
        return s
    except Exception:
        return default


def clamp_text(s: str, max_len: int = 4000) -> str:
    s = safe_str(s, "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# -----------------------------
# Schemas
# -----------------------------
@dataclass
class WatchdogHeartbeat:
    """
    A small always-updated heartbeat state.

    Streamlit's _normalize_watchdog_info() checks keys:
      - last_beat | timestamp | ts
      - count | heartbeat_count | beats
      - seconds_since_last | age_seconds

    We'll emit:
      last_beat (ISO), ts (float), count (int), interval_seconds (optional)
    """

    ts: float = field(default_factory=utc_now_ts)
    last_beat: str = field(default_factory=utc_now_iso)
    count: int = 0
    interval_seconds: Optional[float] = None

    # Optional context
    run_id: Optional[str] = None
    status: Optional[str] = None

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "WatchdogHeartbeat":
        d = raw if isinstance(raw, dict) else {}

        # Support multiple input spellings
        ts_val = _pick(d, ("ts", "heartbeat_ts", "time", "unix_ts"), default=None)
        iso_val = _pick(d, ("last_beat", "lastBeat", "timestamp", "time_iso"), default=None)
        count_val = _pick(d, ("count", "heartbeat_count", "beats"), default=0)

        ts_f = safe_float(ts_val, None)
        if ts_f is None and isinstance(iso_val, str):
            dt = parse_iso(iso_val)
            if dt is not None:
                ts_f = dt.timestamp()
        if ts_f is None:
            ts_f = utc_now_ts()

        iso_s = safe_str(iso_val, "")
        if not iso_s:
            iso_s = iso_from_ts(ts_f)

        return cls(
            ts=ts_f,
            last_beat=iso_s,
            count=safe_int(count_val, 0) or 0,
            interval_seconds=safe_float(_pick(d, ("interval_seconds", "interval", "period_s"), None), None),
            run_id=safe_str(_pick(d, ("run_id", "job_id", "id"), None), None) if _pick(d, ("run_id", "job_id", "id"), None) is not None else None,
            status=safe_str(_pick(d, ("status",), None), None) if _pick(d, ("status",), None) is not None else None,
        )


@dataclass
class ProgressState:
    """
    Optional progress-only file. This is how you get 1/3 -> 2/3 -> 3/3.

    Streamlit compute_progress_view() looks for:
      phase_index, phase_total, phase_name
      current/total (cycles)
      effective_current/effective_total (optional)
      status

    We'll include all relevant fields.
    """

    run_id: Optional[str] = None
    status: str = STATUS_UNKNOWN

    # Phase progress (recommended for "3-phase" pipelines)
    phase_index: Optional[int] = None  # 1-based recommended
    phase_total: Optional[int] = None
    phase_name: Optional[str] = None

    # Cycle progress (optional)
    current: Optional[int] = None
    total: Optional[int] = None

    # If you want the UI to prefer these values
    effective_current: Optional[int] = None
    effective_total: Optional[int] = None

    # timestamps
    ts: float = field(default_factory=utc_now_ts)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        # Add common aliases for compatibility
        if self.phase_index is not None:
            d.setdefault("phase_current", self.phase_index)
        if self.phase_total is not None:
            d.setdefault("phase_count", self.phase_total)
        if self.current is not None:
            d.setdefault("cycle", self.current)
            d.setdefault("cycle_index", self.current)
            d.setdefault("current_cycle", self.current)
        if self.total is not None:
            d.setdefault("total_cycles", self.total)
            d.setdefault("max_cycles", self.total)
        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "ProgressState":
        d = raw if isinstance(raw, dict) else {}
        return cls(
            run_id=safe_str(_pick(d, ("run_id", "job_id", "id"), None), None) if _pick(d, ("run_id", "job_id", "id"), None) is not None else None,
            status=safe_str(_pick(d, ("status",), STATUS_UNKNOWN), STATUS_UNKNOWN),
            phase_index=safe_int(_pick(d, ("phase_index", "phase_current", "phase"), None), None),
            phase_total=safe_int(_pick(d, ("phase_total", "phase_count", "total_phases"), None), None),
            phase_name=safe_str(_pick(d, ("phase_name",), None), None) if _pick(d, ("phase_name",), None) is not None else None,
            current=safe_int(_pick(d, ("current", "current_cycle", "cycle", "cycle_index"), None), None),
            total=safe_int(_pick(d, ("total", "total_cycles", "max_cycles"), None), None),
            effective_current=safe_int(_pick(d, ("effective_current",), None), None),
            effective_total=safe_int(_pick(d, ("effective_total",), None), None),
            ts=safe_float(_pick(d, ("ts", "timestamp_unix", "time"), None), utc_now_ts()) or utc_now_ts(),
            updated_at=safe_str(_pick(d, ("updated_at", "timestamp", "time_iso"), None), utc_now_iso()),
        )


@dataclass
class WorkerState:
    """
    The primary "live state" snapshot for the engine worker.

    The Streamlit UI reads this to render:
    - sticky heartbeat/topbar state
    - progress
    - mode/domain/goal
    - active agent role (optional)

    Keep it small and update frequently.
    """

    run_id: Optional[str] = None
    status: str = STATUS_UNKNOWN

    # Context
    mode: Optional[str] = None
    domain: Optional[str] = None
    goal: Optional[str] = None

    # Role / agent info (optional)
    role: Optional[str] = None
    active_role: Optional[str] = None
    agents: Optional[List[str]] = None  # configured roles/agents (swarm/pair)

    # Phase + cycle progress
    phase_index: Optional[int] = None  # 1-based recommended
    phase_total: Optional[int] = None
    phase_name: Optional[str] = None

    current: Optional[int] = None
    total: Optional[int] = None

    # timestamps
    ts: float = field(default_factory=utc_now_ts)
    updated_at: str = field(default_factory=utc_now_iso)

    # Optional diagnostics / error
    last_error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION

        # Provide compatibility aliases the UI already checks for
        if self.run_id is not None:
            d.setdefault("job_id", self.run_id)
            d.setdefault("id", self.run_id)

        if self.phase_total is not None:
            d.setdefault("total_phases", self.phase_total)
        if self.total is not None:
            d.setdefault("total_cycles", self.total)

        if self.current is not None:
            d.setdefault("cycle", self.current)
            d.setdefault("cycle_index", self.current)
            d.setdefault("current_cycle", self.current)

        # If someone expects these keys:
        if self.active_role and not self.role:
            d["role"] = self.active_role

        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "WorkerState":
        d = raw if isinstance(raw, dict) else {}

        run_id = _pick(d, ("run_id", "job_id", "id"), None)
        phase_total = _pick(d, ("phase_total", "total_phases"), None)
        phase_index = _pick(d, ("phase_index", "phase"), None)

        total = _pick(d, ("total", "total_cycles"), None)
        current = _pick(d, ("current", "current_cycle", "cycle", "cycle_index"), None)

        agents_val = _pick(d, ("agents", "roles", "swarm_roles"), None)
        agents: Optional[List[str]] = None
        if isinstance(agents_val, list):
            agents = [safe_str(x, "").strip() for x in agents_val if safe_str(x, "").strip()]

        meta_val = _pick(d, ("meta",), None)
        meta = meta_val if isinstance(meta_val, dict) else {}

        return cls(
            run_id=safe_str(run_id, None) if run_id is not None else None,
            status=safe_str(_pick(d, ("status",), STATUS_UNKNOWN), STATUS_UNKNOWN),
            mode=safe_str(_pick(d, ("mode", "run_mode"), None), None) if _pick(d, ("mode", "run_mode"), None) is not None else None,
            domain=safe_str(_pick(d, ("domain",), None), None) if _pick(d, ("domain",), None) is not None else None,
            goal=safe_str(_pick(d, ("goal",), None), None) if _pick(d, ("goal",), None) is not None else None,
            role=safe_str(_pick(d, ("role",), None), None) if _pick(d, ("role",), None) is not None else None,
            active_role=safe_str(_pick(d, ("active_role", "current_role"), None), None) if _pick(d, ("active_role", "current_role"), None) is not None else None,
            agents=agents,
            phase_index=safe_int(phase_index, None),
            phase_total=safe_int(phase_total, None),
            phase_name=safe_str(_pick(d, ("phase_name",), None), None) if _pick(d, ("phase_name",), None) is not None else None,
            current=safe_int(current, None),
            total=safe_int(total, None),
            ts=safe_float(_pick(d, ("ts", "timestamp_unix"), None), utc_now_ts()) or utc_now_ts(),
            updated_at=safe_str(_pick(d, ("updated_at", "timestamp"), None), utc_now_iso()),
            last_error=clamp_text(safe_str(_pick(d, ("last_error", "error", "error_message"), None), "")) or None,
            meta=meta,
        )


@dataclass
class RunState:
    """
    "Last saved run state" snapshot. Typically written:
    - at start (queued/starting)
    - before/after each phase
    - on finish
    - on error

    The Streamlit diagnostics tab reads run_state.json when MemoryStore is absent.
    """

    run_id: Optional[str] = None
    status: str = STATUS_UNKNOWN

    domain: Optional[str] = None
    mode: Optional[str] = None
    goal: Optional[str] = None

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # Latest known phase/cycle
    phase_index: Optional[int] = None
    phase_total: Optional[int] = None
    phase_name: Optional[str] = None
    cycle: Optional[int] = None
    total_cycles: Optional[int] = None

    # Optional: summary/diagnostics
    summary: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None

    # Optional: error details
    error: Optional[str] = None

    ts: float = field(default_factory=utc_now_ts)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION

        # Compatibility aliases
        if self.run_id is not None:
            d.setdefault("job_id", self.run_id)
            d.setdefault("id", self.run_id)

        if self.cycle is not None:
            d.setdefault("current_cycle", self.cycle)
            d.setdefault("cycle_index", self.cycle)

        if self.total_cycles is not None:
            d.setdefault("total", self.total_cycles)

        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "RunState":
        d = raw if isinstance(raw, dict) else {}
        run_id = _pick(d, ("run_id", "job_id", "id"), None)

        return cls(
            run_id=safe_str(run_id, None) if run_id is not None else None,
            status=safe_str(_pick(d, ("status",), STATUS_UNKNOWN), STATUS_UNKNOWN),
            domain=safe_str(_pick(d, ("domain",), None), None) if _pick(d, ("domain",), None) is not None else None,
            mode=safe_str(_pick(d, ("mode", "run_mode"), None), None) if _pick(d, ("mode", "run_mode"), None) is not None else None,
            goal=safe_str(_pick(d, ("goal",), None), None) if _pick(d, ("goal",), None) is not None else None,
            started_at=safe_str(_pick(d, ("started_at", "start_time", "start_ts"), None), None) if _pick(d, ("started_at", "start_time", "start_ts"), None) is not None else None,
            finished_at=safe_str(_pick(d, ("finished_at", "end_time", "end_ts"), None), None) if _pick(d, ("finished_at", "end_time", "end_ts"), None) is not None else None,
            phase_index=safe_int(_pick(d, ("phase_index", "phase"), None), None),
            phase_total=safe_int(_pick(d, ("phase_total", "total_phases"), None), None),
            phase_name=safe_str(_pick(d, ("phase_name",), None), None) if _pick(d, ("phase_name",), None) is not None else None,
            cycle=safe_int(_pick(d, ("cycle", "current_cycle", "cycle_index"), None), None),
            total_cycles=safe_int(_pick(d, ("total_cycles", "total"), None), None),
            summary=clamp_text(safe_str(_pick(d, ("summary", "run_summary", "human_summary"), None), ""), 10000) or None,
            diagnostics=_pick(d, ("diagnostics", "debug"), None) if isinstance(_pick(d, ("diagnostics", "debug"), None), dict) else None,
            error=clamp_text(safe_str(_pick(d, ("error", "last_error", "error_message"), None), ""), 12000) or None,
            ts=safe_float(_pick(d, ("ts", "timestamp_unix"), None), utc_now_ts()) or utc_now_ts(),
            updated_at=safe_str(_pick(d, ("updated_at", "timestamp"), None), utc_now_iso()),
        )


@dataclass
class EventEntry:
    """
    Narrative timeline event entry.

    Streamlit render_narrative_feed() accepts events with keys like:
      - ts or timestamp
      - kind or type
      - message or text or summary

    We'll emit:
      ts (ISO), kind, message, plus optional structured data.
    """

    ts: str = field(default_factory=utc_now_iso)
    kind: str = "event"
    message: str = ""

    # Optional context
    run_id: Optional[str] = None
    domain: Optional[str] = None
    role: Optional[str] = None

    # Optional structured payload
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "schema_version": SCHEMA_VERSION,
            "ts": self.ts,
            "timestamp": self.ts,  # alias for viewers
            "kind": self.kind,
            "type": self.kind,  # alias
            "message": clamp_text(self.message, 6000),
            "text": clamp_text(self.message, 6000),  # alias
            "run_id": self.run_id,
            "domain": self.domain,
            "role": self.role,
            "data": self.data or {},
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "EventEntry":
        d = raw if isinstance(raw, dict) else {}
        ts_val = _pick(d, ("ts", "timestamp", "time"), None)
        ts_iso = safe_str(ts_val, "")
        if not ts_iso:
            # Try unix ts
            ts_num = safe_float(_pick(d, ("unix_ts", "ts_unix"), None), None)
            ts_iso = iso_from_ts(ts_num) if ts_num is not None else utc_now_iso()

        kind = safe_str(_pick(d, ("kind", "type"), "event"), "event")
        msg = safe_str(_pick(d, ("message", "text", "summary"), ""), "")

        data_val = _pick(d, ("data", "payload"), None)
        data = data_val if isinstance(data_val, dict) else {}

        rid = _pick(d, ("run_id", "job_id", "id"), None)

        return cls(
            ts=ts_iso,
            kind=kind or "event",
            message=msg,
            run_id=safe_str(rid, None) if rid is not None else None,
            domain=safe_str(_pick(d, ("domain",), None), None) if _pick(d, ("domain",), None) is not None else None,
            role=safe_str(_pick(d, ("role",), None), None) if _pick(d, ("role",), None) is not None else None,
            data=data,
        )


def normalize_event_container(raw: Any) -> Tuple[List[JSONDict], str]:
    """
    Accept either:
      - a list of event dicts
      - a dict with key "events" (or "timeline" / "items")

    Returns: (events_list, container_type_label)
    """
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)], "list"
    if isinstance(raw, dict):
        maybe = raw.get("events") or raw.get("timeline") or raw.get("items") or []
        if isinstance(maybe, list):
            return [e for e in maybe if isinstance(e, dict)], "dict"
        return [], "dict"
    return [], "unknown"
