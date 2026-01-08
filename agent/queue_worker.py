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
import os
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from run_state_manager import RunState
from phase_manager import phases_for_cycles

# Preferred structured event logger (append-only JSONL).
try:  # pragma: no cover
    from event_log import log_event  # type: ignore[import]
except Exception:  # pragma: no cover
    log_event = None  # type: ignore[assignment]

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

def _resolve_runs_root() -> Path:
    """Resolve the runs root directory for queue-worker artifacts.

    Uses ARA_RUNS_DIR when set, otherwise falls back to a local ./runs folder.
    """
    env = (
        os.environ.get("ARA_RUNS_DIR")
        or os.environ.get("ARA_RUN_ROOT")
        or os.environ.get("RUNS_DIR")
    )
    if env:
        try:
            return Path(env).expanduser().resolve()
        except Exception:
            return Path(env)
    return Path('runs').resolve()

BASE_DIR = _resolve_runs_root()
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
    """Append events for a run in a unified, tail-friendly format.

    Key goal: **never overwrite** the run timeline.

    The earlier implementation wrote ``<run_dir>/events.jsonl`` in "w" mode.
    If another component (e.g. engine_worker) starts writing to the same run,
    that "w" would wipe the timeline and Streamlit would suddenly show
    "No events".

    We now:
      1) Prefer `event_log.log_event` (structured, append-only JSONL, plus
         optional legacy mirrors).
      2) Fall back to appending JSONL lines directly (also append-only).
      3) Keep a small legacy JSON array snapshot for older tooling.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    rid = str(run_id)

    # Preferred: structured append-only writer.
    if callable(log_event):
        for ev in events or []:
            if not isinstance(ev, dict):
                continue

            kind = str(ev.get("kind") or ev.get("event") or ev.get("type") or ev.get("domain") or "event")
            level = str(ev.get("level") or "info")
            role = ev.get("role")
            domain = ev.get("domain")
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}

            phase_index = ev.get("phase_index")
            phase_total = ev.get("phase_total")
            phase_name = ev.get("phase_name")
            cycle = ev.get("cycle")

            # Best-effort message synthesis.
            msg = ev.get("message") or ev.get("msg") or ev.get("text") or ev.get("summary")
            if not isinstance(msg, str) or not msg.strip():
                try:
                    if kind == "phase_start":
                        pi = phase_index if phase_index is not None else data.get("phase_index")
                        pt = phase_total if phase_total is not None else data.get("phase_total")
                        pn = phase_name if phase_name is not None else data.get("phase_name")
                        # Display phase index as 1-based when possible.
                        pi_disp = None
                        try:
                            if pi is not None:
                                pi_disp = int(pi) + 1
                        except Exception:
                            pi_disp = pi
                        if pi_disp is not None and pt is not None:
                            msg = f"Phase {pi_disp}/{pt}: {pn or ''}".strip()
                        elif pn:
                            msg = f"Phase: {pn}"
                        else:
                            msg = "Phase started"
                    elif kind in {"run_finished", "job_finished", "finished"}:
                        msg = "Run finished"
                    else:
                        msg = kind
                except Exception:
                    msg = kind

            try:
                log_event(
                    run_id=rid,
                    kind=kind,
                    message=str(msg),
                    level=level,
                    data=data,
                    role=str(role) if role is not None else None,
                    domain=str(domain) if domain is not None else None,
                    phase_index=phase_index,
                    phase_total=phase_total,
                    phase_name=str(phase_name) if phase_name is not None else None,
                    cycle=cycle,
                    logs_dir=LOGS_DIR,
                )
            except Exception:
                continue
        return

    # Fallback: append JSONL lines to the per-run file and the global mirror.
    try:
        run_dir = BASE_DIR / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = run_dir / "events.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            for ev in events or []:
                if isinstance(ev, dict):
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception:
        pass

    try:
        global_path = LOGS_DIR / "events_global.jsonl"
        with global_path.open("a", encoding="utf-8") as f:
            for ev in events or []:
                if isinstance(ev, dict):
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception:
        pass

    # Small legacy snapshot (bounded). This is not used for tailing.
    try:
        legacy_path = LOGS_DIR / f"{rid}_events.json"
        # Keep only a bounded tail to avoid massive JSON arrays.
        snap = [e for e in (events or []) if isinstance(e, dict)][-2000:]
        with legacy_path.open("w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
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
        run_id = job_def.get('run_id')
        if not run_id:
            # Derive from filename when job payload omitted run_id (e.g. <run_id>_job.json)
            stem = active_path.stem
            for suf in ('__job', '_job'):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            run_id = stem

        # Ensure per-run directory exists (for events.jsonl, snapshots, etc.)
        run_dir = BASE_DIR / str(run_id)
        snap_dir = run_dir / 'snapshots'
        run_dir.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)
        events_jsonl_path = run_dir / 'events.jsonl'
        try:
            events_jsonl_path.touch(exist_ok=True)
        except Exception:
            pass


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
            state_path = run_dir / 'run_state.json'
            legacy_state_path = LOGS_DIR / f"run_state_{run_id}.json"
            # Optionally persist a minimal run_state (cycle index only) for lightweight dashboards
            if os.environ.get('ARA_MINIMAL_RUN_STATE', '').strip().lower() in ('1','true','yes','on'):
                try:
                    with state_path.open('w', encoding='utf-8') as sf:
                        json.dump({'cycle_index': int(idx)}, sf)
                except Exception:
                    pass
            else:
                state.save(state_path)
            # Legacy mirror for older UI paths
            try:
                if os.environ.get('ARA_MINIMAL_RUN_STATE', '').strip().lower() in ('1','true','yes','on'):
                    with legacy_state_path.open('w', encoding='utf-8') as sf:
                        json.dump({'cycle_index': int(idx)}, sf)
                else:
                    state.save(legacy_state_path)
            except Exception:
                pass
            # Emit event
            events.append({
                "ts": datetime.utcnow().isoformat() + "Z",  # type: ignore[call-arg]
                "kind": "phase_start",
                "role": "system",
                "cycle": idx,
                "data": {"phase_index": idx, "phase_total": cycles, "phase_name": pname},
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
        # Final state persist
        if os.environ.get('ARA_MINIMAL_RUN_STATE', '').strip().lower() in ('1','true','yes','on'):
            try:
                with state_path.open('w', encoding='utf-8') as sf:
                    json.dump({'cycle_index': int(state.phase_index or 0)}, sf)
            except Exception:
                pass
        else:
            state.save(state_path)
        try:
            if os.environ.get('ARA_MINIMAL_RUN_STATE', '').strip().lower() in ('1','true','yes','on'):
                with legacy_state_path.open('w', encoding='utf-8') as sf:
                    json.dump({'cycle_index': int(state.phase_index or 0)}, sf)
            else:
                state.save(legacy_state_path)
        except Exception:
            pass
        events.append({
            "ts": datetime.utcnow().isoformat() + "Z",  # type: ignore[call-arg]
            "kind": "run_finished",
            "role": "system",
            "cycle": int(state.phase_index or 0),
            "data": {"phase_total": cycles},
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
            snap_dir = (run_dir / 'snapshots')
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
