"""
Queue-based background worker for the Autonomous Research Agent.

This worker runs on Render as a Background Worker. It:

- Reads jobs from a file-based queue in ARA_RUN_DIR / "queue"
- Marks them as completed and writes a small result file
- Loops forever, sleeping when there are no jobs

This version is a SAFE MINIMAL WORKER:
- It proves the queue wiring works.
- It does NOT run your full ARA logic yet.
  (We can wire it to RunManager/CoreAgent after it's confirmed working.)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Where the queue and results live.
# Must match the ARA_RUN_DIR env var you set on Render.
RUN_DIR = Path(os.environ.get("ARA_RUN_DIR", "runs")).resolve()
QUEUE_DIR = RUN_DIR / "queue"
RESULTS_DIR = RUN_DIR / "results"

POLL_SECONDS = 2.0  # how often to look for new jobs


def _ensure_dirs() -> None:
    """Make sure queue and results directories exist."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_job(job_path: Path) -> dict:
    """Load a job JSON file."""
    with job_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_result(job_id: str, payload: dict) -> None:
    """Write a result JSON file for the completed job."""
    result_path = RESULTS_DIR / f"{job_id}.result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _process_job(job_path: Path) -> None:
    """Handle a single job from the queue."""
    job = _load_job(job_path)
    job_id = str(job.get("job_id") or job_path.stem)

    print(f"[worker] 🧵 Starting job {job_id} from {job_path}")

    started = datetime.utcnow().isoformat() + "Z"

    # -----------------------------------------------------------------
    # TODO: This is where we will later call your real ARA engine
    # (RunManager, CoreAgent, etc).
    #
    # For now we just echo the job back so we can prove that:
    #   - the worker sees the job
    #   - it can read/write files
    #   - the Render service stays alive
    # -----------------------------------------------------------------

    finished = datetime.utcnow().isoformat() + "Z"

    result_payload = {
        "job_id": job_id,
        "status": "completed",
        "started_at": started,
        "finished_at": finished,
        "note": (
            "engine_worker_queue.py placeholder ran successfully. "
            "Wire this up to RunManager/CoreAgent to execute real ARA runs."
        ),
        "echo_job": job,
    }

    _write_result(job_id, result_payload)
    print(f"[worker] ✅ Finished job {job_id}, wrote result file.")

    # Remove job from queue
    job_path.unlink(missing_ok=True)


def main() -> None:
    """Main loop: watch the queue directory forever."""
    _ensure_dirs()
    print(f"[worker] Watching queue: {QUEUE_DIR}")
    print(f"[worker] Results will be written to: {RESULTS_DIR}")

    while True:
        jobs = sorted(QUEUE_DIR.glob("*.json"))
        if not jobs:
            time.sleep(POLL_SECONDS)
            continue

        # Pick the oldest job first
        job_path = jobs[0]
        try:
            _process_job(job_path)
        except Exception as exc:
            # Don't crash the worker if a job is bad.
            print(f"[worker] ❌ Error processing {job_path}: {exc!r}")
            # Move the bad job aside
            bad_path = job_path.with_suffix(".error.json")
            job_path.rename(bad_path)
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
