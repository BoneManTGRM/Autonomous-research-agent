# -*- coding: utf-8 -*-
"""
agent/state_schema.py

Shared JSON schemas + coercion helpers for the ARA worker <-> Streamlit UI
telemetry layer.

Design goals
- Stdlib only.
- Backward/forward compatible: accept multiple key aliases on read.
- Easy to emit: dataclasses with .to_dict().

This module does NOT write files. Disk I/O lives in agent/state_emitter.py
(and optionally agent/event_log.py).
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

# Canonical filenames (state_emitter/event_log should use these)
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
    """ISO 8601 in UTC from a unix timestamp (best-effort)."""
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


def iso_from_any(v: Any, timespec: str = "seconds") -> str:
    """
    Best-effort conversion into a UTC ISO string with trailing Z.
    Accepts ISO strings, unix timestamps (int/float), numeric strings, or datetimes.
    """
    if v is None:
        return utc_now_iso(timespec=timespec)

    if isinstance(v, datetime):
        try:
            dt = v.astimezone(timezone.utc)
            return dt.isoformat(timespec=timespec).replace("+00:00", "Z")
        except Exception:
            return utc_now_iso(timespec=timespec)

    if isinstance(v, (int, float)):
        return iso_from_ts(v, timespec=timespec)

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return utc_now_iso(timespec=timespec)

        dt = parse_iso(s)
        if dt is not None:
            try:
                dt2 = dt.astimezone(timezone.utc)
                return dt2.isoformat(timespec=timespec).replace("+00:00", "Z")
            except Exception:
                pass

        # numeric string?
        try:
            f = float(s)
            return iso_from_ts(f, timespec=timespec)
        except Exception:
            return utc_now_iso(timespec=timespec)

    return utc_now_iso(timespec=timespec)


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
        return str(v)
    except Exception:
        return default


def safe_opt_str(v: Any) -> Optional[str]:
    """Return a trimmed string or None."""
    if v is None:
        return None
    try:
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None


def clamp_text(s: Any, max_len: int = 4000) -> str:
    txt = safe_str(s, "")
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 3] + "..."


# -----------------------------
# Schemas
# -----------------------------
@dataclass
class WatchdogHeartbeat:
    """
    A small always-updated heartbeat snapshot.

    Fields Streamlit commonly checks:
      - last_beat | timestamp | ts
      - count | heartbeat_count | beats

    We use:
      - last_beat (ISO string)
      - ts (unix float)
      - count (int)
    """

    ts: float = field(default_factory=utc_now_ts)
    last_beat: str = field(default_factory=utc_now_iso)
    count: int = 0
    interval_seconds: Optional[float] = None

    # Optional context
    run_id: Optional[str] = None
    status: Optional[str] = None

    def to_dict(self) -> JSONDict:
        d: JSONDict = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        # Compatibility aliases
        d.setdefault("timestamp", self.last_beat)
        d.setdefault("heartbeat_ts", self.ts)
        d.setdefault("beats", self.count)
        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "WatchdogHeartbeat":
        d = raw if isinstance(raw, dict) else {}

        ts_val = _pick(d, ("ts", "heartbeat_ts", "time", "unix_ts", "timestamp_unix"), default=None)
        iso_val = _pick(d, ("last_beat", "lastBeat", "timestamp", "time_iso", "lastBeatTime"), default=None)
        count_val = _pick(d, ("count", "heartbeat_count", "beats"), default=0)

        ts_f = safe_float(ts_val, None)

        # If ts isn't present, attempt to derive it from iso_val
        if ts_f is None:
            if isinstance(iso_val, (int, float)):
                ts_f = float(iso_val)
            elif isinstance(iso_val, str):
                dt = parse_iso(iso_val)
                if dt is not None:
                    ts_f = dt.timestamp()
                else:
                    ts_f = safe_float(iso_val, None)

        if ts_f is None:
            ts_f = utc_now_ts()

        # Normalize last_beat to a real ISO string
        last_beat_iso = None
        if isinstance(iso_val, str):
            dt = parse_iso(iso_val)
            if dt is not None:
                last_beat_iso = iso_from_any(dt)
        if last_beat_iso is None:
            last_beat_iso = iso_from_ts(ts_f)

        return cls(
            ts=ts_f,
            last_beat=last_beat_iso,
            count=safe_int(count_val, 0) or 0,
            interval_seconds=safe_float(_pick(d, ("interval_seconds", "interval", "period_s"), None), None),
            run_id=safe_opt_str(_pick(d, ("run_id", "job_id", "id"), None)),
            status=safe_opt_str(_pick(d, ("status",), None)),
        )


@dataclass
class ProgressState:
    """
    Optional progress-only snapshot.

    Phase indexing:
      Your current Streamlit progress normalizer treats `phase_index` as 0-based
      while running (0..phase_total-1) and displays it as 1..phase_total.

      Emit 0,1,2 for a 3-phase pipeline to see 1/3,2/3,3/3 in the UI.
    """

    run_id: Optional[str] = None
    status: str = STATUS_UNKNOWN

    # Phase progress
    phase_index: Optional[int] = None  # 0-based recommended
    phase_total: Optional[int] = None
    phase_name: Optional[str] = None

    # Cycle progress (optional)
    current: Optional[int] = None
    total: Optional[int] = None

    # Preferred by Streamlit if present
    effective_current: Optional[int] = None
    effective_total: Optional[int] = None

    # timestamps
    ts: float = field(default_factory=utc_now_ts)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> JSONDict:
        d: JSONDict = asdict(self)
        d["schema_version"] = SCHEMA_VERSION

        # Compatibility aliases
        d.setdefault("timestamp", self.updated_at)
        d.setdefault("timestamp_unix", self.ts)

        if self.phase_index is not None:
            d.setdefault("phase_current", self.phase_index)
            d.setdefault("phase", self.phase_index)
        if self.phase_total is not None:
            d.setdefault("phase_count", self.phase_total)
            d.setdefault("total_phases", self.phase_total)

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
        updated_at_raw = _pick(d, ("updated_at", "timestamp", "time_iso"), None)

        return cls(
            run_id=safe_opt_str(_pick(d, ("run_id", "job_id", "id"), None)),
            status=safe_str(_pick(d, ("status",), STATUS_UNKNOWN), STATUS_UNKNOWN),
            phase_index=safe_int(_pick(d, ("phase_index", "phase_current", "phase"), None), None),
            phase_total=safe_int(_pick(d, ("phase_total", "phase_count", "total_phases"), None), None),
            phase_name=safe_opt_str(_pick(d, ("phase_name",), None)),
            current=safe_int(_pick(d, ("current", "current_cycle", "cycle", "cycle_index"), None), None),
            total=safe_int(_pick(d, ("total", "total_cycles", "max_cycles"), None), None),
            effective_current=safe_int(_pick(d, ("effective_current",), None), None),
            effective_total=safe_int(_pick(d, ("effective_total",), None), None),
            ts=safe_float(_pick(d, ("ts", "timestamp_unix", "time", "unix_ts"), None), utc_now_ts()) or utc_now_ts(),
            updated_at=iso_from_any(updated_at_raw),
        )


@dataclass
class WorkerState:
    """
    Primary "live worker snapshot".

    Keep it small; update frequently.
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
    phase_index: Optional[int] = None  # 0-based recommended for your Streamlit UI
    phase_total: Optional[int] = None
    phase_name: Optional[str] = None

    current: Optional[int] = None
    total: Optional[int] = None

    effective_current: Optional[int] = None
    effective_total: Optional[int] = None

    # Optional short status line
    message: Optional[str] = None

    # timestamps
    ts: float = field(default_factory=utc_now_ts)
    updated_at: str = field(default_factory=utc_now_iso)

    # Optional diagnostics / error
    last_error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        d: JSONDict = asdict(self)
        d["schema_version"] = SCHEMA_VERSION

        # Compatibility aliases
        if self.run_id is not None:
            d.setdefault("job_id", self.run_id)
            d.setdefault("id", self.run_id)

        d.setdefault("timestamp", self.updated_at)
        d.setdefault("timestamp_unix", self.ts)

        if self.phase_total is not None:
            d.setdefault("total_phases", self.phase_total)
        if self.total is not None:
            d.setdefault("total_cycles", self.total)

        if self.current is not None:
            d.setdefault("cycle", self.current)
            d.setdefault("cycle_index", self.current)
            d.setdefault("current_cycle", self.current)

        # Some code expects `role` always set.
        if self.active_role and not self.role:
            d["role"] = self.active_role

        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "WorkerState":
        d = raw if isinstance(raw, dict) else {}

        agents_val = _pick(d, ("agents", "roles", "swarm_roles"), None)
        agents: Optional[List[str]] = None
        if isinstance(agents_val, list):
            cleaned = [safe_str(x, "").strip() for x in agents_val]
            agents = [x for x in cleaned if x]

        meta_val = _pick(d, ("meta",), None)
        meta = meta_val if isinstance(meta_val, dict) else {}

        updated_at_raw = _pick(d, ("updated_at", "timestamp", "time_iso"), None)

        return cls(
            run_id=safe_opt_str(_pick(d, ("run_id", "job_id", "id"), None)),
            status=safe_str(_pick(d, ("status",), STATUS_UNKNOWN), STATUS_UNKNOWN),
            mode=safe_opt_str(_pick(d, ("mode", "run_mode"), None)),
            domain=safe_opt_str(_pick(d, ("domain",), None)),
            goal=safe_opt_str(_pick(d, ("goal",), None)),
            role=safe_opt_str(_pick(d, ("role",), None)),
            active_role=safe_opt_str(_pick(d, ("active_role", "current_role"), None)),
            agents=agents,
            phase_index=safe_int(_pick(d, ("phase_index", "phase", "phase_current"), None), None),
            phase_total=safe_int(_pick(d, ("phase_total", "total_phases", "phase_count"), None), None),
            phase_name=safe_opt_str(_pick(d, ("phase_name",), None)),
            current=safe_int(_pick(d, ("current", "current_cycle", "cycle", "cycle_index"), None), None),
            total=safe_int(_pick(d, ("total", "total_cycles", "max_cycles"), None), None),
            effective_current=safe_int(_pick(d, ("effective_current",), None), None),
            effective_total=safe_int(_pick(d, ("effective_total",), None), None),
            message=safe_opt_str(_pick(d, ("message", "status_text", "detail"), None)),
            ts=safe_float(_pick(d, ("ts", "timestamp_unix", "unix_ts"), None), utc_now_ts()) or utc_now_ts(),
            updated_at=iso_from_any(updated_at_raw),
            last_error=(
                clamp_text(_pick(d, ("last_error", "error", "error_message"), None), 12000).strip() or None
            ),
            meta=meta,
        )


@dataclass
class RunState:
    """
    Last saved run snapshot. Often written:
    - at start (queued/starting)
    - before/after each phase
    - on finish
    - on error
    """

    run_id: Optional[str] = None
    status: str = STATUS_UNKNOWN

    domain: Optional[str] = None
    mode: Optional[str] = None
    goal: Optional[str] = None

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # Latest known phase/cycle
    phase_index: Optional[int] = None  # 0-based recommended for your Streamlit UI
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
        d: JSONDict = asdict(self)
        d["schema_version"] = SCHEMA_VERSION

        # Compatibility aliases
        if self.run_id is not None:
            d.setdefault("job_id", self.run_id)
            d.setdefault("id", self.run_id)

        d.setdefault("timestamp", self.updated_at)
        d.setdefault("timestamp_unix", self.ts)

        if self.cycle is not None:
            d.setdefault("current_cycle", self.cycle)
            d.setdefault("cycle_index", self.cycle)

        if self.total_cycles is not None:
            d.setdefault("total", self.total_cycles)

        if self.phase_total is not None:
            d.setdefault("total_phases", self.phase_total)

        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "RunState":
        d = raw if isinstance(raw, dict) else {}
        updated_at_raw = _pick(d, ("updated_at", "timestamp", "time_iso"), None)

        diagnostics_val = _pick(d, ("diagnostics", "debug"), None)
        diagnostics = diagnostics_val if isinstance(diagnostics_val, dict) else None

        return cls(
            run_id=safe_opt_str(_pick(d, ("run_id", "job_id", "id"), None)),
            status=safe_str(_pick(d, ("status",), STATUS_UNKNOWN), STATUS_UNKNOWN),
            domain=safe_opt_str(_pick(d, ("domain",), None)),
            mode=safe_opt_str(_pick(d, ("mode", "run_mode"), None)),
            goal=safe_opt_str(_pick(d, ("goal",), None)),
            started_at=safe_opt_str(_pick(d, ("started_at", "start_time", "start_ts"), None)),
            finished_at=safe_opt_str(_pick(d, ("finished_at", "end_time", "end_ts"), None)),
            phase_index=safe_int(_pick(d, ("phase_index", "phase", "phase_current"), None), None),
            phase_total=safe_int(_pick(d, ("phase_total", "total_phases", "phase_count"), None), None),
            phase_name=safe_opt_str(_pick(d, ("phase_name",), None)),
            cycle=safe_int(_pick(d, ("cycle", "current_cycle", "cycle_index"), None), None),
            total_cycles=safe_int(_pick(d, ("total_cycles", "total"), None), None),
            summary=(clamp_text(_pick(d, ("summary", "run_summary", "human_summary"), None), 10000).strip() or None),
            diagnostics=diagnostics,
            error=(clamp_text(_pick(d, ("error", "last_error", "error_message"), None), 12000).strip() or None),
            ts=safe_float(_pick(d, ("ts", "timestamp_unix", "unix_ts"), None), utc_now_ts()) or utc_now_ts(),
            updated_at=iso_from_any(updated_at_raw),
        )


@dataclass
class EventEntry:
    """
    Narrative timeline event entry.

    Streamlit viewers typically accept:
      - ts or timestamp
      - kind or type
      - message or text or summary
    """

    ts: str = field(default_factory=utc_now_iso)
    kind: str = "event"
    message: str = ""

    # Optional context
    run_id: Optional[str] = None
    domain: Optional[str] = None
    role: Optional[str] = None
    phase_name: Optional[str] = None
    cycle: Optional[int] = None

    # Optional structured payload
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        msg = clamp_text(self.message, 6000)
        return {
            "schema_version": SCHEMA_VERSION,
            "ts": self.ts,
            "timestamp": self.ts,  # alias
            "kind": self.kind,
            "type": self.kind,  # alias
            "message": msg,
            "text": msg,  # alias
            "run_id": self.run_id,
            "domain": self.domain,
            "role": self.role,
            "phase_name": self.phase_name,
            "cycle": self.cycle,
            "data": self.data or {},
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "EventEntry":
        d = raw if isinstance(raw, dict) else {}
        ts_val = _pick(d, ("ts", "timestamp", "time"), None)
        ts_iso = iso_from_any(ts_val)

        kind = safe_str(_pick(d, ("kind", "type"), "event"), "event")
        msg = safe_str(_pick(d, ("message", "text", "summary"), ""), "")

        data_val = _pick(d, ("data", "payload"), None)
        data = data_val if isinstance(data_val, dict) else {}

        return cls(
            ts=ts_iso,
            kind=kind or "event",
            message=msg,
            run_id=safe_opt_str(_pick(d, ("run_id", "job_id", "id"), None)),
            domain=safe_opt_str(_pick(d, ("domain",), None)),
            role=safe_opt_str(_pick(d, ("role",), None)),
            phase_name=safe_opt_str(_pick(d, ("phase_name",), None)),
            cycle=safe_int(_pick(d, ("cycle", "cycle_index", "current_cycle"), None), None),
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
