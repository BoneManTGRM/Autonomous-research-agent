"""
queue_worker.py
================

This module implements a simple file‑based queue worker for the Autonomous
Research Agent.  Jobs are JSON files dropped into a ``queue/pending``
directory.  The worker moves each job into an ``active`` directory,
executes the requested number of cycles using placeholder logic and
writes run artefacts (state, logs, snapshots) into the run's folder.  On
success the job file is moved to ``finished``; on failure it lands in
``error``.

The implementation here is intentionally minimal and should be adapted to
your engine's needs.  It demonstrates how to:

* Read a job definition and extract the desired number of cycles.
* Create and persist a :class:`run_state_manager.RunState`.
* Iterate through phases, updating the run state and optionally taking
  snapshots and pruning memory.
* Write a simple event log for consumption by the Streamlit UI.

If you have complex TGRM logic, replace the placeholder loop with calls
into your core engine and emit meaningful events.
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from run_state_manager import RunState
from phase_manager import phases_for_cycles

try:
    from memory_pruner import MemoryPruner  # type: ignore[import]
except Exception:
    MemoryPruner = None  # type: ignore[assignment]

try:
    from snapshot_manager import take_snapshot  # type: ignore[import]
except Exception:
    # Provide a no‑op fallback if snapshot manager is missing
    def take_snapshot(*args: Any, **kwargs: Any) -> None:
        return None


# ----------------------------------------------------------------------
# Queue directory constants
# ----------------------------------------------------------------------
BASE_DIR = Path("runs").resolve()
QUEUE_ROOT = BASE_DIR / "queue"
PENDING_DIR = QUEUE_ROOT / "pending"
ACTIVE_DIR = QUEUE_ROOT / "active"
FINISHED_DIR = BASE_DIR / "finished"
ERROR_DIR = BASE_DIR / "error"
LOGS_DIR = BASE_DIR / "logs"


def _load_job_definition(job_path: Path) -> Dict[str, Any]:
    """Load a job definition JSON from the given path.

    The JSON is expected to contain at least a ``cycles`` key.  Unknown
    keys are preserved for downstream use (e.g. swarm size, domain, goal).
    """
    with job_path.open("r", encoding="utf-8") as f:
        job = json.load(f)
    return job


def _write_event_log(run_id: str, events: List[Dict[str, Any]]) -> None:
    """Write an event log for the run into the logs directory.

    The event log is stored as JSON with one object per event.  This
    is deliberately simple; the Streamlit UI can parse and display
    these events to build a narrative timeline.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{run_id}_events.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


def _default_memory_store() -> Any:
    """Return a dummy memory store if none is provided.

    This stub implements the minimal API used by the pruning logic.
    It stores entries in memory and allows deletion by ID.  You can
    replace this with a real implementation (for example, the
    agent.MemoryStore) when running inside the full system.
    """
    class DummyStore:
        def __init__(self) -> None:
            self.entries: Dict[str, Dict[str, Any]] = {}

        def list_entries(self) -> List[Dict[str, Any]]:
            return [dict(id=k, content=v["content"], meta=v["meta"]) for k, v in self.entries.items()]

        def delete_entries(self, ids: List[str]) -> None:
            for _id in ids:
                self.entries.pop(_id, None)

        def update_entry(self, id: str, fields: Dict[str, Any]) -> None:
            if id in self.entries:
                self.entries[id]["meta"].update(fields)

        def add_entry(self, content: str, meta: Dict[str, Any]) -> None:
            self.entries[str(len(self.entries) + 1)] = {"content": content, "meta": meta}

    return DummyStore()


def process_next_job(memory_store: Optional[Any] = None) -> Optional[str]:
    """Process the next pending job if one exists.

    Parameters
    ----------
    memory_store:
        Optional memory store implementation used for pruning.  If not
        provided a dummy store is created.

    Returns
    -------
    str or None
        The run ID of the processed job, or ``None`` if no job was
        available.
    """
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    FINISHED_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    # Only consider job metadata files ("*_job.json" and "__job.json")
    job_files = sorted(
        [
            p
            for p in PENDING_DIR.glob("*.json")
            if p.name.endswith("_job.json") or p.name.endswith("__job.json")
        ]
    )
    if not job_files:
        return None
    job_path = job_files[0]
    run_id: Optional[str] = None
    events: List[Dict[str, Any]] = []

    try:
        # Move job to active before processing
        active_path = ACTIVE_DIR / job_path.name
        shutil.move(str(job_path), str(active_path))

        job_def = _load_job_definition(active_path)
        cycles = int(job_def.get("cycles", 0))
        goal = str(job_def.get("goal", ""))
        run_id = job_def.get("run_id") or active_path.stem

        # Initialise run state
        state = RunState.new(run_id=run_id, total_cycles=cycles, goal=goal)
        state.status = "running"

        # Determine phase names
        phase_names = phases_for_cycles(cycles)

        # Ensure memory store exists
        mstore = memory_store or _default_memory_store()
        pruner: Optional[MemoryPruner] = None
        if MemoryPruner is not None:
            try:
                pruner = MemoryPruner(memory_store=mstore, run_id=run_id)
            except Exception:
                pruner = None

        # Run through phases
        for idx, pname in enumerate(phase_names):
            state.update_phase(idx, pname)
            # Persist run state after each phase
            state_path = LOGS_DIR / f"run_state_{run_id}.json"
            state.save(state_path)
            # Emit event
            events.append({
                "ts": datetime.utcnow().isoformat() + "Z",  # type: ignore[call-arg]
                "kind": "phase_start",
                "phase_index": idx,
                "phase_total": cycles,
                "phase_name": pname,
                "run_id": run_id,
            })
            # Take snapshot (if available)
            take_snapshot(run_id=run_id, phase_index=idx, memory_store=mstore, goal=goal)
            # Optionally prune memory mid‑run
            if pruner is not None:
                try:
                    pruner.prune()  # type: ignore[func-returns-value]
                except Exception:
                    pass

        # Mark finished
        state.mark_finished()
        state.save(LOGS_DIR / f"run_state_{run_id}.json")
        events.append({
            "ts": datetime.utcnow().isoformat() + "Z",  # type: ignore[call-arg]
            "kind": "run_finished",
            "run_id": run_id,
            "phase_total": cycles,
        })

        # Write event log
        _write_event_log(run_id, events)

        # Persist an aggregated result for convenience.  This result
        # mirrors the structure returned by run_jobs.load_job_result and
        # allows the Streamlit UI to display run history without
        # recomputing it.  We include basic metadata such as goal,
        # domain and cycle count.  If writing fails we silently
        # continue; load_job_result will assemble the result on the fly.
        try:
            from run_jobs import result_path as _result_path  # type: ignore
            result_payload = {
                "config": job_def,
                "result": {
                    "run_id": run_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "goal": goal,
                    "domain": job_def.get("domain"),
                    "cycles": cycles,
                    "phase_total": cycles,
                    "phase_index": state.phase_index,
                    "phase_name": state.phase_name,
                    "status": state.status,
                    "events": events,
                    "snapshots": [],
                },
            }
            # List snapshot filenames if available
            snap_dir = (BASE_DIR / "snapshots" / run_id)
            if snap_dir.is_dir():
                result_payload["result"]["snapshots"] = sorted([p.name for p in snap_dir.glob("*.json")])
            rpath = _result_path(run_id)
            rpath.parent.mkdir(parents=True, exist_ok=True)
            with rpath.open("w", encoding="utf-8") as rf:
                json.dump(result_payload, rf, ensure_ascii=False, indent=2)
        except Exception:
            # Fallback: skip result persistence
            pass

        # Move job file to finished
        finished_path = FINISHED_DIR / active_path.name
        shutil.move(str(active_path), str(finished_path))
        return run_id

    except Exception:
        # Capture failure details
        trace = traceback.format_exc()
        error_path = ERROR_DIR / f"{job_path.stem}_error.txt"
        with error_path.open("w", encoding="utf-8") as f:
            f.write(trace)
        # Move job to error directory
        if active_path.exists():
            shutil.move(str(active_path), str(ERROR_DIR / active_path.name))
        return None


__all__ = ["process_next_job"]