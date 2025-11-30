# agent/run_jobs.py

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
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
QUEUE_DIR = PENDING_DIR

# Make sure directories exist at import time
for folder in [BASE_DIR, PENDING_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class RunJob:
    """
    File based representation of a single research run.

    Each job has:
        run_id      - stable identifier used in filenames and UI
        config      - full configuration dict for RunManager or your engine
        status      - pending, active, finished, or error
        created_at  - unix timestamp when job was created
        updated_at  - unix timestamp when job was last updated
        meta        - optional metadata for UI (user prompt, domain, etc)

    NOTE:
        - "pending" is the new canonical status (replaces older "queued").
        - For backwards compatibility, helper functions still understand
          "queued" as an alias for "pending".
    """

    run_id: str
    config: Dict[str, Any]
    status: str = "pending"
    created_at: float = time.time()
    updated_at: float = time.time()
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunJob":
        return cls(**data)

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
        - "pending"  (canonical)
        - "queued"   (alias for pending, for older code)
        - "active"
        - "finished"
        - "error"
    """
    norm = status.lower()
    if norm in ("queued", "pending"):
        base = PENDING_DIR
    elif norm == "active":
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
    Create a new pending job on disk and return its run_id.

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
        status="pending",
        created_at=time.time(),
        updated_at=time.time(),
        meta=meta or {},
    )
    job.save_to(PENDING_DIR)
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
    for folder in [PENDING_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
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
        return None, None

    job = load_job(src_path)
    job.status = "pending" if to_status == "queued" else to_status
    job.updated_at = time.time()

    dst_path = _job_path_for_status(run_id, job.status)
    # Remove old file and save new one
    src_path.unlink(missing_ok=True)
    job.save_to(dst_path.parent)
    return job, dst_path


def update_job_status(run_id: str, new_status: str) -> Optional[RunJob]:
    """
    Convenience helper to update a job status without knowing the old state.

    It searches all folders for run_id, then moves it to the requested status.
    Returns the updated job or None if not found.

    new_status can be "pending", "active", "finished", "error" or legacy "queued".
    """
    norm_new = new_status.lower()
    if norm_new == "queued":
        norm_new = "pending"

    for status in ["pending", "queued", "active", "finished", "error"]:
        src_path = _job_path_for_status(run_id, status)
        if src_path.exists():
            job = load_job(src_path)
            job.status = norm_new
            job.updated_at = time.time()
            dst_path = _job_path_for_status(run_id, norm_new)
            src_path.unlink(missing_ok=True)
            job.save_to(dst_path.parent)
            return job
    return None


def list_jobs(status: Optional[str] = None, limit: int = 100) -> List[RunJob]:
    """
    List jobs by status for UI or debugging.

    status:
        None        - search all statuses
        pending     - only pending jobs (new canonical)
        queued      - alias for pending (backwards compatible)
        active      - only active jobs
        finished    - only finished jobs
        error       - only error jobs
    """
    jobs: List[RunJob] = []

    def collect(folder: Path) -> None:
        for path in sorted(folder.glob("*.json")):
            try:
                job = load_job(path)
                jobs.append(job)
            except Exception:
                continue

    if status is None:
        for folder in [PENDING_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
            collect(folder)
    else:
        norm = status.lower()
        if norm == "queued":
            norm = "pending"

        folder_map = {
            "pending": PENDING_DIR,
            "active": ACTIVE_DIR,
            "finished": FINISHED_DIR,
            "error": ERROR_DIR,
        }
        folder = folder_map.get(norm)
        if folder is not None:
            collect(folder)

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


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
        This now returns runs/finished/{run_id}.json to match the Streamlit
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
