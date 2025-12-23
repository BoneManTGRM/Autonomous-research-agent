"""
Event logging utilities for the Autonomous Research Agent.

This module provides a simple helper for emitting JSON‑line events to a
file. Workers can call `emit_event` to append structured events to a
per‑run log. Each invocation appends a single JSON object on its own
line and flushes the file immediately so that the Streamlit UI can tail
the log in near real‑time. Event files live under a run‑specific
directory; callers pass in the directory and the helper creates the
directory (and file) as needed.

An event is a dictionary with at minimum the following fields:

    - ``ts``: UNIX epoch seconds when the event was emitted.
    - ``level``: A short string such as ``"info"`` or ``"error"``.
    - ``domain``: A freeform category for grouping events, e.g.
      ``"progress"`` or ``"citations"``.
    - ``msg``: Human‑readable message describing the event.
    - ``extra``: Optional dictionary for arbitrary metadata.

The JSON is written in one line per event (JSONL). It is encoded as
UTF‑8 with ASCII fallback to tolerate unexpected input. The file is
opened in append mode with line buffering disabled to guarantee that
each call writes immediately to disk.

Example usage in a worker:

    from pathlib import Path
    from events import emit_event

    run_dir = Path("/path/to/runs_root") / run_id
    emit_event(run_dir, level="info", domain="progress",
               msg="Cycle 1/3 complete", extra={"current": 1, "total": 3})

Downstream consumers (such as the Streamlit UI) can tail the
``events.jsonl`` file within the run directory to display a live event
feed.
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
) -> None:
    """Append a single structured event to the run‑scoped event log.

    Args:
        run_dir: The directory for this run. A file named
            ``events.jsonl`` will be created inside this directory.
        level: A short string such as ``"info"``, ``"warning"`` or
            ``"error"`` describing the severity of the event.
        msg: A human‑readable message for the event.
        domain: A freeform category for grouping events (e.g.
            ``"progress"``, ``"fetch"``, ``"analysis"``). Defaults to
            ``"general"``.
        extra: Optional dictionary of additional metadata to attach to
            the event.

    This function attempts to be robust: it will create the run
    directory if it does not exist, ignore any exceptions while
    serializing the event, and flush the file after each append so that
    other processes can see the update immediately. In the rare case
    that writing fails, the error is silently ignored to avoid
    disrupting the caller.
    """
    try:
        # Ensure we have a Path object
        if not isinstance(run_dir, Path):
            run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "events.jsonl"
        event: Dict[str, Any] = {
            "ts": time.time(),
            "level": str(level),
            "domain": str(domain) if domain is not None else "general",
            "msg": str(msg) if msg is not None else "",
            "extra": extra or {},
        }
        line = json.dumps(event, ensure_ascii=False) + "\n"
        # Open in append binary mode without buffering to ensure flush
        with path.open("ab", buffering=0) as f:
            data = line.encode("utf-8", errors="replace")
            f.write(data)
            # fsync to force the OS to flush to disk
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # Never propagate errors to the caller; skip on failure
        pass
