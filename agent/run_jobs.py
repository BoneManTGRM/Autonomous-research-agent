# agent/run_jobs.py

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Base folder for all runs
BASE_DIR = Path("runs")

# New job layout used by the engine worker:
#   - runs/pending/   : file based queue of pending jobs (was "queue")
#   - runs/active/    : in progress metadata + optional progress JSON
#   - runs/finished/  : final result JSON (one file per run_id)
#   - runs/error/     : failed job metadata + optional traceback
PENDING_DIR = BASE_DIR / "pending"
ACTIVE_DIR = BASE_DIR / "active"
FINISHED_DIR = BASE_DIR / "finished"
ERROR_DIR = BASE_DIR / "error"

# Backwards compatible alias for old name "queue"
# Many engine scripts import QUEUE_DIR from this module, so this keeps them working.
QUEUE_DIR = PENDING_DIR

# Optional legacy "queue" folder support (for older jobs)
LEGACY_QUEUE_DIR = BASE_DIR / "queue"

# Make sure directories exist at import time
for folder in [BASE_DIR, PENDING_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR, LEGACY_QUEUE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class RunJob:
    """
    File based representation of a single research run.

    Each job has:
        run_id      - stable identifier used in filenames and UI
        config      - full configuration dict for RunManager or your engine
        status      - queued, active, finished, or error
        created_at  - unix timestamp when job was created
        updated_at  - unix timestamp when job was last updated
        meta        - optional metadata for UI (user prompt, domain, etc)

    STATUS NOTES:

        - "queued" is the legacy and engine friendly name.
        - "pending" is treated as an alias for "queued".
        - "running" is treated as an alias for "active".

    This file is written by the UI and read by your engine worker.
    """

    run_id: str
    config: Dict[str, Any]
    status: str = "queued"  # keep legacy value so existing engine loops that check == "queued" work
    created_at: float = time.time()
    updated_at: float = time.time()
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunJob":
        """
        Robust loader that tolerates extra keys and missing optional fields.

        This allows older job files (or hand-edited JSON) to be loaded
        without crashing due to unexpected keys. Also normalizes status.
        """
        field_names = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in field_names}

        # Ensure required fields exist; raise a clear error if not.
        required = {"run_id", "config"}
        missing_required = [k for k in required if k not in filtered]
        if missing_required:
            raise ValueError(f"RunJob.from_dict missing required fields: {missing_required}")

        # Fill in optional fields if missing
        now = time.time()
        status = str(filtered.get("status") or "queued").lower()
        if status in ("pending", "queued"):
            filtered["status"] = "queued"
        elif status in ("running", "active"):
            filtered["status"] = "active"
        elif status not in ("active", "finished", "error"):
            filtered["status"] = "queued"

        if "created_at" not in filtered:
            filtered["created_at"] = now
        if "updated_at" not in filtered:
            filtered["updated_at"] = now
        if "meta" not in filtered:
            filtered["meta"] = None

        return cls(**filtered)

    def save_to(self, folder: Path) -> Path:
        """
        Save the job JSON inside the given folder using run_id as filename.

        This writes only the job metadata (config, status, meta, timestamps).
        The engine worker should write the final result into result_path(run_id).
        """
        self.updated_at = time.time()
        path = folder / f"{self.run_id}.json"
        with path.open("w", encoding="utf8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def _job_path_for_status(run_id: str, status: str) -> Path:
    """
    Return the expected job metadata path given run_id and status.

    Supported statuses:
        - "queued"   (canonical queue status)
        - "pending"  (alias for queued, for newer code)
        - "running"  (alias for active, for older engines)
        - "active"
        - "finished"
        - "error"
    """
    norm = status.lower()
    if norm in ("queued", "pending"):
        base = PENDING_DIR
    elif norm in ("running", "active"):
        base = ACTIVE_DIR
    elif norm == "finished":
        base = FINISHED_DIR
    elif norm == "error":
        base = ERROR_DIR
    else:
        raise ValueError(f"Unknown job status: {status}")
    return base / f"{run_id}.json"


def create_job(
    config: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Create a new queued job on disk and return its run_id.

    Typical use from a UI:
        run_id = create_job(
            {
                "goal": user_goal,
                "runtime_hints": {...},
                "swarm": {...},
                ...
            },
            meta={"domain": domain, "created_by": "streamlit_ui"},
        )

    Args:
        config: Arbitrary configuration dict for the engine worker.
        meta:   Optional metadata for UI / logging (domain, label, etc).
        run_id: Optional externally generated ID. If omitted, uuid4 is used.

    Returns:
        The run_id string.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    job = RunJob(
        run_id=run_id,
        config=config,
        status="queued",  # use "queued" so old engine loops pick it up
        created_at=time.time(),
        updated_at=time.time(),
        meta=meta or {},
    )
    # Save into the canonical pending/queue folder
    job.save_to(PENDING_DIR)
    # Also mirror into legacy 'queue' folder so any old scripts scanning runs/queue still see it
    legacy_path = LEGACY_QUEUE_DIR / f"{run_id}.json"
    with legacy_path.open("w", encoding="utf8") as f:
        json.dump(job.to_dict(), f, indent=2)

    return run_id


def load_job(path: Path) -> RunJob:
    """
    Load a job from a specific metadata JSON path.
    """
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)
    return RunJob.from_dict(data)


def load_job_by_id(run_id: str) -> Optional[RunJob]:
    """
    Try to load a job by id from any status folder.
    Returns None if not found.
    """
    for folder in [PENDING_DIR, LEGACY_QUEUE_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
        path = folder / f"{run_id}.json"
        if path.exists():
            return load_job(path)
    return None


def move_job(run_id: str, from_status: str, to_status: str) -> Tuple[Optional[RunJob], Optional[Path]]:
    """
    Move job metadata between status folders.

    Returns (job, new_path) or (None, None) if job was not found
    in the expected source folder.
    """
    src_path = _job_path_for_status(run_id, from_status)
    if not src_path.exists():
        # Try legacy queue dir as a backup for from_status == queued/pending
        if from_status.lower() in ("queued", "pending"):
            legacy_path = LEGACY_QUEUE_DIR / f"{run_id}.json"
            if not legacy_path.exists():
                return None, None
            src_path = legacy_path
        else:
            return None, None

    job = load_job(src_path)

    norm_to = to_status.lower()
    if norm_to in ("pending", "queued"):
        job.status = "queued"
    elif norm_to in ("running", "active"):
        job.status = "active"
    else:
        job.status = norm_to

    job.updated_at = time.time()

    dst_path = _job_path_for_status(run_id, job.status)

    # Remove old file(s) and save new one
    src_path.unlink(missing_ok=True)
    # Also remove from legacy queue if it exists
    legacy_src = LEGACY_QUEUE_DIR / f"{run_id}.json"
    legacy_src.unlink(missing_ok=True)
    job.save_to(dst_path.parent)
    return job, dst_path


def update_job_status(run_id: str, new_status: str) -> Optional[RunJob]:
    """
    Convenience helper to update a job status without knowing the old state.

    It searches all folders for run_id, then moves it to the requested status.
    Returns the updated job or None if not found.

    new_status can be:
        - "queued"  or "pending"  -> queue dir, status set to "queued"
        - "running" or "active"   -> active dir, status set to "active"
        - "finished"
        - "error"
    """
    norm_new = new_status.lower()
    if norm_new in ("pending", "queued"):
        norm_new = "queued"
    elif norm_new in ("running", "active"):
        norm_new = "active"

    for status in ["queued", "pending", "running", "active", "finished", "error"]:
        src_path = _job_path_for_status(run_id, status)
        if src_path.exists():
            job = load_job(src_path)
            job.status = norm_new
            job.updated_at = time.time()
            dst_path = _job_path_for_status(run_id, norm_new)
            src_path.unlink(missing_ok=True)
            # remove any stale legacy file in queue dir
            legacy_src = LEGACY_QUEUE_DIR / f"{run_id}.json"
            legacy_src.unlink(missing_ok=True)
            job.save_to(dst_path.parent)
            return job

    # Also check legacy queue dir if not found by status loop above
    legacy_path = LEGACY_QUEUE_DIR / f"{run_id}.json"
    if legacy_path.exists():
        job = load_job(legacy_path)
        job.status = norm_new
        job.updated_at = time.time()
        legacy_path.unlink(missing_ok=True)
        dst_path = _job_path_for_status(run_id, norm_new)
        job.save_to(dst_path.parent)
        return job

    return None


def list_jobs(status: Optional[str] = None, limit: int = 100) -> List[RunJob]:
    """
    List jobs by status for UI or debugging.

    status:
        None        - search all statuses
        queued      - jobs in runs/pending (canonical queue status)
        pending     - alias for queued
        running     - alias for active
        active      - jobs in runs/active
        finished    - jobs in runs/finished
        error       - jobs in runs/error

    NOTE:
        This function does NOT filter based on the internal job.status string,
        it simply reads all *.json files from the appropriate folder(s).
        That means old jobs with status "queued" or "pending" in runs/pending
        are all visible to the engine and the UI.
    """
    jobs: List[RunJob] = []

    def collect(folder: Path) -> None:
        for path in sorted(folder.glob("*.json")):
            try:
                # skip progress and non-metadata files
                if path.name.endswith("_progress.json"):
                    continue
                job = load_job(path)
                jobs.append(job)
            except Exception:
                continue

    if status is None:
        # Search all status folders plus legacy queue
        for folder in [PENDING_DIR, LEGACY_QUEUE_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
            collect(folder)
    else:
        norm = status.lower()
        if norm in ("pending", "queued"):
            # queue/pending jobs from both canonical and legacy folders
            collect(PENDING_DIR)
            collect(LEGACY_QUEUE_DIR)
        elif norm in ("running", "active"):
            collect(ACTIVE_DIR)
        elif norm == "finished":
            collect(FINISHED_DIR)
        elif norm == "error":
            collect(ERROR_DIR)

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


def get_next_queued_job() -> Optional[RunJob]:
    """
    Simple helper for engine workers.

    Returns the oldest queued job (by created_at) from the queue folder(s),
    or None if there are no queued jobs.

    Typical engine usage pattern:

        while True:
            job = get_next_queued_job()
            if job is None:
                time.sleep(2)
                continue

            update_job_status(job.run_id, "active")
            try:
                ... run engine ...
                update_job_status(job.run_id, "finished")
            except Exception:
                update_job_status(job.run_id, "error")
    """
    queued = list_jobs(status="queued", limit=1000)
    if not queued:
        return None
    # list_jobs returns newest first; for FIFO we take the last
    return queued[-1]


def claim_next_job() -> Optional[RunJob]:
    """
    Atomically claim the next queued job by marking it active.

    Engine workers can call this to get the next job and immediately
    move it to the active folder so that other workers dont pick it up.

    Returns:
        RunJob or None if there is no queued job.
    """
    job = get_next_queued_job()
    if job is None:
        return None
    updated = update_job_status(job.run_id, "active")
    return updated or job


def progress_path(run_id: str) -> Path:
    """
    Path where the worker should write live progress JSON for a run.

    Recommended schema (example):
        {
            "run_id": "...",
            "status": "active",
            "current_cycle": 12,
            "total_cycles": 90,
            "last_update_utc": "...",
            "notes": "optional human readable progress"
        }
    """
    return ACTIVE_DIR / f"{run_id}_progress.json"


def result_path(run_id: str) -> Path:
    """
    Path where the worker should write final result JSON for a run.

    IMPORTANT:
        This returns runs/finished/{run_id}.json to match the Streamlit
        UI, which expects finished results in that location and with that
        naming convention.

    Recommended schema (example):
        {
            "job_id": "...",
            "status": "finished",
            "created_at": "...",
            "completed_at": "...",
            "goal": "...",
            "summary": "...",
            "key_findings": [...],
            "cycles": [...],
            "rye_metrics": {...},
            "sources": [...],
            "debug": {...}
        }
    """
    return FINISHED_DIR / f"{run_id}.json"


def error_log_path(run_id: str) -> Path:
    """
    Path where the worker can write an error message or traceback.
    """
    return ERROR_DIR / f"{run_id}_error.txt"
