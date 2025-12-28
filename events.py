"""
Event logging utilities for the Autonomous Research Agent (ARA).

This module standardizes *structured* event emission for the UI.

Why JSONL?
- Streamlit reruns frequently and cannot stream from the worker process.
- A worker can append JSON objects to a file, and Streamlit can "tail" it
  on each refresh.
- JSON Lines (one JSON object per line) is append-friendly and resilient
  to partial reads.

Default location
- Per-run events are written to:  <run_dir>/events.jsonl
  where run_dir is typically:     <runs_root>/<run_id>

Schema
Each event is a dict with at minimum:
- ts: UNIX epoch seconds
- level: "info" | "warning" | "error" | ...
- domain: freeform category (e.g. "progress", "fetch", "analysis")
- msg: human readable message
- extra: optional dict for arbitrary metadata

Optional top-level fields may be included when provided:
- kind, run_id, role, cycle, phase_index, phase_total, phase_name, message

Notes
- This function is best-effort and never raises.
- The file is opened in append binary mode with buffering disabled, then
  flushed + fsync'd so other processes can observe updates quickly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def emit_event(
    run_dir: Path,
    *,
    level: str,
    msg: str,
    domain: str = "general",
    extra: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None,
    run_id: Optional[str] = None,
    role: Optional[str] = None,
    cycle: Optional[int] = None,
    phase_index: Optional[int] = None,
    phase_total: Optional[int] = None,
    phase_name: Optional[str] = None,
    file_name: str = "events.jsonl",
) -> None:
    """Append a single structured event to a JSONL file.

    Args:
        run_dir: Directory that owns the event log file. The file will be
            created at ``run_dir / file_name``.
        level: Severity string (e.g. "info", "warning", "error").
        msg: Human readable message.
        domain: Freeform category for grouping events.
        extra: Optional dict of additional metadata.
        kind: Optional event type (e.g. "phase_start", "milestone:foo").
        run_id: Optional run identifier to store on the event.
        role: Optional agent/worker role for the event.
        cycle: Optional cycle index.
        phase_index: Optional phase/round index.
        phase_total: Optional phase/round total.
        phase_name: Optional phase/round label.
        file_name: Output file name (default "events.jsonl"). This is
            intentionally simple; callers should pass a name, not a path.

    This function is robust by design:
    - Creates the run_dir if needed
    - Never raises on serialization or IO errors
    - Flushes + fsyncs after each write so Streamlit can tail it reliably
    """
    try:
        # Ensure we have a Path object
        if not isinstance(run_dir, Path):
            run_dir = Path(run_dir)

        # Defensive: treat empty/unsafe file_name as default
        if not isinstance(file_name, str) or not file_name.strip():
            file_name = "events.jsonl"
        file_name = file_name.strip().replace("\\", "/").split("/")[-1]

        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / file_name

        event: Dict[str, Any] = {
            "ts": time.time(),
            "level": str(level),
            "domain": str(domain) if domain is not None else "general",
            "msg": str(msg) if msg is not None else "",
            "extra": extra or {},
        }

        # Provide common aliases for mixed consumers
        # - Some UI code expects "message"
        event["message"] = event["msg"]

        # Optional structured fields
        if kind is not None:
            event["kind"] = str(kind)
        if run_id is not None:
            event["run_id"] = str(run_id)
        if role is not None:
            event["role"] = str(role)
        if cycle is not None:
            try:
                event["cycle"] = int(cycle)
            except Exception:
                event["cycle"] = cycle
        if phase_index is not None:
            try:
                event["phase_index"] = int(phase_index)
            except Exception:
                event["phase_index"] = phase_index
        if phase_total is not None:
            try:
                event["phase_total"] = int(phase_total)
            except Exception:
                event["phase_total"] = phase_total
        if phase_name is not None:
            event["phase_name"] = str(phase_name)

        line = json.dumps(event, ensure_ascii=False) + "\n"

        # Append in binary mode without buffering to ensure immediate write.
        with path.open("ab", buffering=0) as f:
            data = line.encode("utf-8", errors="replace")
            f.write(data)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # Some environments may not support fsync; ignore.
                pass
    except Exception:
        # Never propagate errors to the caller; skip on failure.
        return
