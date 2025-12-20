"""
history_viewer.py
=================

This module provides helper functions for reading historical artefacts
produced by the agent: event logs, run states and snapshots.  These
helpers centralise file system access logic so that the Streamlit UI can
load artefacts without duplicating path arithmetic.  All paths are
relative to the ``runs`` directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RUNS_ROOT = Path("runs")
LOGS_DIR = RUNS_ROOT / "logs"
SNAPSHOT_DIR = RUNS_ROOT / "snapshots"


def load_event_log(run_id: str) -> List[Dict[str, Any]]:
    """Load the event log for a run.

    Returns a list of events in the order they were recorded.  If the
    log does not exist an empty list is returned.
    """
    path = LOGS_DIR / f"{run_id}_events.json"
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def load_run_state(run_id: str) -> Dict[str, Any]:
    """Load the last saved run state for a run.

    The run state is stored in ``runs/logs/run_state_<run_id>.json``.  If
    the file does not exist an empty dict is returned.
    """
    path = LOGS_DIR / f"run_state_{run_id}.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def list_snapshots(run_id: str) -> List[Path]:
    """Return a sorted list of snapshot file paths for a run.

    The snapshots live under ``runs/snapshots/<run_id>``.  Only JSON
    files are returned.  Files are sorted by filename which should
    correspond to chronological order given the timestamp prefix used
    when writing snapshots.
    """
    run_dir = SNAPSHOT_DIR / run_id
    if not run_dir.is_dir():
        return []
    return sorted(run_dir.glob("*.json"))


__all__ = ["load_event_log", "load_run_state", "list_snapshots"]