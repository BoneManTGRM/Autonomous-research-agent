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

# Subfolders that define a simple file based queue
QUEUE_DIR = BASE_DIR / "queue"
ACTIVE_DIR = BASE_DIR / "active"
FINISHED_DIR = BASE_DIR / "finished"
ERROR_DIR = BASE_DIR / "error"

# Make sure directories exist at import time
for folder in [BASE_DIR, QUEUE_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
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
    """

    run_id: str
    config: Dict[str, Any]
    status: str = "queued"
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
        """
        self.updated_at = time.time()
        path = folder / f"{self.run_id}.json"
        with path.open("w", encoding="utf8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def _job_path_for_status(run_id: str, status: str) -> Path:
    """
    Return the expected job metadata path given run_id and status.
    """
    if status == "queued":
        base = QUEUE_DIR
    elif status == "active":
        base = ACTIVE_DIR
    elif status == "finished":
        base = FINISHED_DIR
    elif status == "error":
        base = ERROR_DIR
    else:
        raise ValueError(f"Unknown job status: {status}")
    return base / f"{run_id}.json"


def create_job(
    config: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a new queued job on disk and return its run_id.

    Typical use from Streamlit:
        run_id = create_job(
            {
                "goal": user_goal,
                "cycles": cycles,
                "swarm_size": swarm_size,
                ...
            },
            meta={"domain": domain, "created_by": "streamlit_ui"},
        )
    """
    run_id = str(uuid.uuid4())
    job = RunJob(
        run_id=run_id,
        config=config,
        status="queued",
        created_at=time.time(),
        updated_at=time.time(),
        meta=meta or {},
    )
    job.save_to(QUEUE_DIR)
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
    for folder in [QUEUE_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
        path = folder / f"{run_id}.json"
        if path.exists():
            return load_job(path)
    return None


def move_job(run_id: str, from_status: str, to_status: str) -> Tuple[Optional[RunJob], Optional[Path]]:
    """
    Move job metadata between status folders.

    Returns (job, new_path) or (None, None) if job was not found.
    """
    src_path = _job_path_for_status(run_id, from_status)
    if not src_path.exists():
        return None, None

    job = load_job(src_path)
    job.status = to_status
    job.updated_at = time.time()

    dst_path = _job_path_for_status(run_id, to_status)
    # Remove old file and save new one
    src_path.unlink(missing_ok=True)
    job.save_to(dst_path.parent)
    return job, dst_path


def update_job_status(run_id: str, new_status: str) -> Optional[RunJob]:
    """
    Convenience helper to update a job status without knowing the old state.

    It searches all folders for run_id, then moves it to the requested status.
    Returns the updated job or None if not found.
    """
    for status in ["queued", "active", "finished", "error"]:
        src_path = _job_path_for_status(run_id, status)
        if src_path.exists():
            job = load_job(src_path)
            job.status = new_status
            job.updated_at = time.time()
            dst_path = _job_path_for_status(run_id, new_status)
            src_path.unlink(missing_ok=True)
            job.save_to(dst_path.parent)
            return job
    return None


def list_jobs(status: Optional[str] = None, limit: int = 100) -> List[RunJob]:
    """
    List jobs by status for UI or debugging.

    status:
        None        - search all statuses
        queued      - only queued jobs
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
        for folder in [QUEUE_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR]:
            collect(folder)
    else:
        folder_map = {
            "queued": QUEUE_DIR,
            "active": ACTIVE_DIR,
            "finished": FINISHED_DIR,
            "error": ERROR_DIR,
        }
        folder = folder_map.get(status)
        if folder is not None:
            collect(folder)

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


def progress_path(run_id: str) -> Path:
    """
    Path where the worker should write live progress JSON for a run.
    """
    return ACTIVE_DIR / f"{run_id}_progress.json"


def result_path(run_id: str) -> Path:
    """
    Path where the worker should write final result JSON for a run.
    """
    return FINISHED_DIR / f"{run_id}_result.json"


def error_log_path(run_id: str) -> Path:
    """
    Path where the worker can write an error message or traceback.
    """
    return ERROR_DIR / f"{run_id}_error.txt"
