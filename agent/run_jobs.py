# agent/run_jobs.py

from __future__ import annotations

import json
import time
import uuid
import os
from dataclasses import dataclass, asdict, fields, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve repository root so the default runs directory is stable
_THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_FILE_DIR.parent

# Base folder for all runs.
# IMPORTANT:
#   On Render your start command (or environment) should set:
#       ARA_RUNS_DIR="/opt/render/project/src/runs"
#   so BASE_DIR will resolve to that exact shared folder for ALL of:
#       - engine_worker.py
#       - app_streamlit.py (or streamlit_app.py)
#       - this run_jobs.py queue layer
_env_runs_raw = os.getenv("ARA_RUNS_DIR")
_env_runs = _env_runs_raw.strip() if isinstance(_env_runs_raw, str) else None

if _env_runs:
    BASE_DIR = Path(_env_runs).resolve()
else:
    BASE_DIR = (REPO_ROOT / "runs").resolve()

# Job layout used by the engine worker:
#   - runs/pending/   : file based queue of pending jobs (canonical queue)
#   - runs/active/    : in progress metadata and optional progress JSON
#   - runs/finished/  : final result JSON and finished job metadata
#   - runs/error/     : failed job metadata and optional traceback
PENDING_DIR = BASE_DIR / "pending"
ACTIVE_DIR = BASE_DIR / "active"
FINISHED_DIR = BASE_DIR / "finished"
ERROR_DIR = BASE_DIR / "error"

# Backwards compatible alias for old name "queue".
# Many legacy scripts import QUEUE_DIR from this module.
# QUEUE_DIR now points at the canonical pending folder.
QUEUE_DIR = PENDING_DIR

# Optional "queue" folder support (older jobs may still live here).
# Workers that watch runs/queue directly will look here.
LEGACY_QUEUE_DIR = BASE_DIR / "queue"

# Make sure directories exist at import time
for folder in [BASE_DIR, PENDING_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR, LEGACY_QUEUE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# Optional helper you can call from Streamlit to debug the layout
def debug_print_layout() -> None:
    """
    Print the resolved BASE_DIR and subfolders.

    Call this once from app_streamlit.py if you want to confirm that the UI
    and the worker really point at the same physical directories.
    """
    env_val = os.environ.get("ARA_RUNS_DIR")
    try:
        base_resolved = BASE_DIR.resolve()
    except Exception:
        base_resolved = BASE_DIR

    print("[run_jobs] ARA_RUNS_DIR env (raw):", repr(env_val))
    print("[run_jobs] ARA_RUNS_DIR env (stripped):", repr(_env_runs))
    print("[run_jobs] BASE_DIR:", base_resolved)
    print("[run_jobs] PENDING_DIR:", PENDING_DIR)
    print("[run_jobs] ACTIVE_DIR:", ACTIVE_DIR)
    print("[run_jobs] FINISHED_DIR:", FINISHED_DIR)
    print("[run_jobs] ERROR_DIR:", ERROR_DIR)
    print("[run_jobs] LEGACY_QUEUE_DIR:", LEGACY_QUEUE_DIR)


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

    STATUS NOTES (normalized):
        - "queued" is the canonical queue status used by engines.
        - "pending" is treated as an alias for "queued".
        - "running" is treated as an alias for "active".

    This file is written by the UI and read by your engine worker.
    """

    run_id: str
    config: Dict[str, Any]
    status: str = "queued"  # keep "queued" so engine loops that check == "queued" work
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunJob":
        """
        Robust loader that tolerates extra keys and missing optional fields.

        This allows older job files (or hand edited JSON) to be loaded
        without crashing due to unexpected keys. Also normalizes status.
        """
        field_names = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in field_names}

        # Ensure required fields exist; raise a clear error if not.
        required = {"run_id", "config"}
        missing_required: List[str] = []
        for req in required:
            if req not in data or data.get(req) is None:
                missing_required.append(req)
        if missing_required:
            raise ValueError(f"RunJob.from_dict missing required fields: {missing_required}")

        # Fill in optional fields if missing
        now = time.time()
        status = str(filtered.get("status") or data.get("status") or "queued").lower()
        if status in ("pending", "queued"):
            filtered["status"] = "queued"
        elif status in ("running", "active"):
            filtered["status"] = "active"
        elif status not in ("active", "finished", "error"):
            filtered["status"] = "queued"

        if "created_at" not in filtered:
            filtered["created_at"] = float(data.get("created_at", now))
        if "updated_at" not in filtered:
            filtered["updated_at"] = float(data.get("updated_at", now))
        if "meta" not in filtered:
            filtered["meta"] = data.get("meta", None)

        # Make sure required fields from original data are present
        filtered["run_id"] = str(data["run_id"])
        filtered["config"] = dict(data["config"])

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
        - "pending"  (alias for queued)
        - "running"  (alias for active)
        - "active"
        - "finished"
        - "error"

    Important:
        For finished jobs we store metadata separately from the result JSON.
        Metadata: runs/finished/{run_id}_job.json
        Result:   runs/finished/{run_id}.json
    """
    norm = status.lower()
    if norm in ("queued", "pending"):
        base = PENDING_DIR
        filename = f"{run_id}.json"
    elif norm in ("running", "active"):
        base = ACTIVE_DIR
        filename = f"{run_id}.json"
    elif norm == "finished":
        base = FINISHED_DIR
        filename = f"{run_id}_job.json"
    elif norm == "error":
        base = ERROR_DIR
        filename = f"{run_id}.json"
    else:
        raise ValueError(f"Unknown job status: {status}")
    return base / filename


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
        meta:   Optional metadata for UI or logging (domain, label, etc).
        run_id: Optional externally generated ID. If omitted, uuid4 is used.

    Returns:
        The run_id string.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    job = RunJob(
        run_id=run_id,
        config=config,
        status="queued",
        created_at=time.time(),
        updated_at=time.time(),
        meta=meta or {},
    )

    # Save into the canonical pending folder
    job.save_to(PENDING_DIR)

    # Shadow copy to queue folder for maximal compatibility.
    # Workers that still watch runs/queue directly will see this.
    try:
        job.save_to(LEGACY_QUEUE_DIR)
    except Exception:
        # Compatibility should never crash job creation.
        pass

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
    # Finished metadata uses "_job.json"
    finished_meta = FINISHED_DIR / f"{run_id}_job.json"
    if finished_meta.exists():
        return load_job(finished_meta)

    for folder in [PENDING_DIR, LEGACY_QUEUE_DIR, ACTIVE_DIR, ERROR_DIR]:
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
        # Try queue dir as a backup for from_status == queued or pending
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
    # Also remove from queue folder if it exists (prevent duplicates)
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

    # First check finished metadata path for this id
    finished_meta = FINISHED_DIR / f"{run_id}_job.json"
    if finished_meta.exists() and norm_new == "finished":
        job = load_job(finished_meta)
        job.status = "finished"
        job.updated_at = time.time()
        dst_path = _job_path_for_status(run_id, "finished")
        finished_meta.unlink(missing_ok=True)
        job.save_to(dst_path.parent)
        return job

    # Check other status folders
    for status in ["queued", "pending", "running", "active", "finished", "error"]:
        src_path = _job_path_for_status(run_id, status)
        if src_path.exists():
            job = load_job(src_path)
            job.status = norm_new
            job.updated_at = time.time()
            dst_path = _job_path_for_status(run_id, norm_new)
            src_path.unlink(missing_ok=True)
            # remove any stale file in queue dir
            legacy_src = LEGACY_QUEUE_DIR / f"{run_id}.json"
            legacy_src.unlink(missing_ok=True)
            job.save_to(dst_path.parent)
            return job

    # Also check queue dir if not found by status loop above
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
        finished    - jobs in runs/finished (metadata files only)
        error       - jobs in runs/error

    NOTE:
        This function does not filter based on the internal job.status string,
        it simply reads appropriate metadata files from the correct folder(s).
        Result JSON files in runs/finished are ignored so they do not collide
        with the job metadata schema.
    """
    jobs_by_id: Dict[str, RunJob] = {}

    def collect(folder: Path, finished: bool = False) -> None:
        if not folder.exists():
            return

        if finished:
            pattern = "*_job.json"  # only metadata files
        else:
            pattern = "*.json"

        for path in sorted(folder.glob(pattern)):
            # skip progress and non metadata files
            if path.name.endswith("_progress.json"):
                continue
            try:
                job = load_job(path)
            except Exception:
                continue

            existing = jobs_by_id.get(job.run_id)
            if existing is None or job.created_at > existing.created_at:
                jobs_by_id[job.run_id] = job

    if status is None:
        # Search all status folders plus queue dir
        collect(PENDING_DIR)
        collect(LEGACY_QUEUE_DIR)
        collect(ACTIVE_DIR)
        collect(FINISHED_DIR, finished=True)
        collect(ERROR_DIR)
    else:
        norm = status.lower()
        if norm in ("pending", "queued"):
            # queue or pending jobs from both canonical and queue folders
            collect(PENDING_DIR)
            collect(LEGACY_QUEUE_DIR)
        elif norm in ("running", "active"):
            collect(ACTIVE_DIR)
        elif norm == "finished":
            collect(FINISHED_DIR, finished=True)
        elif norm == "error":
            collect(ERROR_DIR)

    jobs: List[RunJob] = list(jobs_by_id.values())
    # Sort by created_at descending (newest first)
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


def get_next_queued_job() -> Optional[RunJob]:
    """
    Simple helper for engine workers.

    Returns the oldest queued job (by created_at) from the queue folder(s),
    or None if there are no queued jobs.

    Implementation detail:
        list_jobs(status="queued") returns newest first, so for FIFO we take
        the last element (oldest created_at).
    """
    queued = list_jobs(status="queued", limit=1000)
    if not queued:
        return None
    # list_jobs returns newest first; for FIFO we take the last (oldest)
    return queued[-1]


def claim_next_job() -> Optional[RunJob]:
    """
    Atomically claim the next queued job by marking it active.

    Engine workers can call this to get the next job and immediately
    move it to the active folder so that other workers do not pick it up.

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

    This returns runs/finished/{run_id}.json.

    Finished jobs now have two files:
        - runs/finished/{run_id}_job.json   (job metadata, used by list_jobs)
        - runs/finished/{run_id}.json       (final result payload, used by UI)

    Recommended result schema (example):
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


# ---------------------------------------------------------------------------
# Engine worker specific helpers (used by engine_worker.py)
# ---------------------------------------------------------------------------

def load_next_pending_job() -> Optional[RunJob]:
    """
    Engine worker entry point for queue mode:
    claim and return the next queued job, already moved to ACTIVE status.
    """
    return claim_next_job()


def save_job_result(job: RunJob, result_obj: Dict[str, Any]) -> None:
    """
    Write final result JSON for a job and mark it finished.

    Used by engine_worker queue mode.
    """
    # Ensure job_id is present in the result for UI convenience
    if "job_id" not in result_obj:
        result_obj["job_id"] = job.run_id

    rp = result_path(job.run_id)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with rp.open("w", encoding="utf8") as f:
        json.dump(result_obj, f, indent=2)

    # Update job metadata status and move it into runs/finished/{run_id}_job.json
    update_job_status(job.run_id, "finished")


def mark_job_error(job: RunJob, error_info: Dict[str, Any]) -> None:
    """
    Write error information for a job and mark it as error.

    error_info can be a string or a dict.
    """
    ep = error_log_path(job.run_id)
    ep.parent.mkdir(parents=True, exist_ok=True)
    with ep.open("w", encoding="utf8") as f:
        if isinstance(error_info, str):
            f.write(error_info)
        else:
            json.dump(error_info, f, indent=2)

    update_job_status(job.run_id, "error")
