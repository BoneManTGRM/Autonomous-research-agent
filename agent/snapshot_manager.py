"""
snapshot_manager.py
===================

This module encapsulates snapshot creation during ARA runs.  A snapshot
captures the state of the memory store and optionally additional
diagnostics at a specific point in time.  Snapshots are stored as JSON
files in a ``runs/snapshots/<run_id>`` directory.  Each snapshot file
includes metadata (run ID, phase index, timestamp, goal) and a list of
memory entries.  You can extend the snapshot structure to include
additional fields, for example RYE metrics or equilibrium detection
results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # Import equilibrium detection if available
    from agent.rye_metrics import detect_rye_equilibrium  # type: ignore[import]
except Exception:
    detect_rye_equilibrium = None  # type: ignore[assignment]


SNAPSHOT_ROOT = Path("runs") / "snapshots"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _serialise_memory_entries(memory_store: Any) -> List[Dict[str, Any]]:
    """Serialise memory entries from a memory store.

    The memory store is expected to implement a ``list_entries`` method
    returning an iterable of dictionaries with at least ``id``,
    ``content`` and ``meta`` keys.  If the API differs the caller can
    preformat entries before passing them here.
    """
    try:
        entries = memory_store.list_entries()
    except Exception:
        entries = []  # Fall back to empty list if unsupported
    # Convert datetimes to ISO strings on the fly
    for entry in entries:
        meta = entry.get("meta", {})
        for key in ("created_at", "last_accessed"):
            ts = meta.get(key)
            if isinstance(ts, datetime):
                meta[key] = ts.replace(tzinfo=None).isoformat(timespec="seconds")
    return entries


def take_snapshot(
    run_id: str,
    *,
    phase_index: Optional[int] = None,
    memory_store: Optional[Any] = None,
    goal: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Capture a snapshot of the current run state.

    Parameters
    ----------
    run_id:
        The identifier of the run being snapshotted.
    phase_index:
        Optional index of the phase at which the snapshot is taken.  If
        provided it will be included in the metadata.
    memory_store:
        The memory store whose contents should be serialised.  If
        omitted or if the store does not implement ``list_entries``,
        ``entries`` will be an empty list.
    goal:
        Optional run goal to store in the snapshot.
    extra:
        Optional dictionary of additional metadata to include.

    Returns
    -------
    pathlib.Path
        The path to the snapshot file that was written.
    """
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = SNAPSHOT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot: Dict[str, Any] = {
        "run_id": run_id,
        "phase_index": phase_index,
        "goal": goal,
        "created_at": _utc_iso(),
    }
    if extra:
        snapshot.update(extra)

    # Include memory entries if available
    if memory_store is not None:
        snapshot["entries"] = _serialise_memory_entries(memory_store)
    else:
        snapshot["entries"] = []

    # Optionally detect equilibrium
    if detect_rye_equilibrium is not None and memory_store is not None:
        try:
            eq = detect_rye_equilibrium(memory_store=memory_store)
            snapshot["equilibrium"] = eq
        except Exception:
            snapshot["equilibrium"] = None

    # Write to disk with an incremental filename
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    idx_part = f"_{phase_index}" if phase_index is not None else ""
    filename = f"snapshot_{ts}{idx_part}.json"
    path = run_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return path


__all__ = ["take_snapshot"]