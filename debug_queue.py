# debug_queue.py
"""
Smoke test for the file based job queue (agent/run_jobs.py).

Usage:
    python debug_queue.py

What it does:
    1. Prints ARA_RUNS_DIR and all queue directories.
    2. Creates a single debug job via create_job.
    3. Simulates an engine worker grabbing the next job.
    4. Writes a fake result via save_job_result.
    5. Confirms the finished result JSON exists and is readable.

If this script fails, the problem is inside run_jobs.py or filesystem/paths.
If this script works, the bug is between Streamlit and the real engine worker.
"""

from pathlib import Path
import json
import os

from agent.run_jobs import (
    BASE_DIR,
    PENDING_DIR,
    ACTIVE_DIR,
    FINISHED_DIR,
    ERROR_DIR,
    create_job,
    list_jobs,
    load_next_pending_job,
    save_job_result,
    result_path,
)

def main() -> None:
    print("=== DEBUG QUEUE SMOKE TEST ===")
    print("ARA_RUNS_DIR env:", os.environ.get("ARA_RUNS_DIR", "(not set)"))
    print("BASE_DIR       :", BASE_DIR)
    print("PENDING_DIR    :", PENDING_DIR)
    print("ACTIVE_DIR     :", ACTIVE_DIR)
    print("FINISHED_DIR   :", FINISHED_DIR)
    print("ERROR_DIR      :", ERROR_DIR)
    print()

    # 1) Create a fake job
    config = {
        "goal": "Debug queue smoke test",
        "domain": "general",
        "mode": "single",
        "total_cycles": 1,
    }
    meta = {"run_label": "debug_smoke_test"}

    run_id = create_job(config=config, meta=meta)
    print(f"[1] Created job with run_id = {run_id}")
    print()

    # 2) List queued jobs
    try:
        queued_jobs = list_jobs(status="queued")
    except TypeError:
        # fallback if list_jobs has no status arg
        queued_jobs = list_jobs()  # type: ignore[call-arg]

    print("[2] Pending / queued jobs after create:")
    if not queued_jobs:
        print("    (none)")
    else:
        for j in queued_jobs:
            print(f"    - {j.run_id} (status={j.status})")
    print()

    # 3) Simulate engine picking up the next pending job
    job = load_next_pending_job()
    if job is None:
        print("[3] ERROR: load_next_pending_job() returned None.")
        print("    That means the engine side cannot see any queued jobs.")
        raise SystemExit(1)

    print(f"[3] load_next_pending_job() -> run_id={job.run_id}, status={job.status}")
    print()

    # 4) Simulate engine writing a fake result
    fake_result = {
        "job_id": job.run_id,
        "status": "finished",
        "summary": f"Fake engine summary for goal: {job.config.get('goal')}",
        "key_findings": ["Test finding A", "Test finding B"],
        "cycles": [],
        "rye_metrics": {"avg_rye": 0.123},
        "sources": [],
        "debug": {"note": "debug_queue.py fake run"},
    }

    save_job_result(job, fake_result)
    print("[4] Called save_job_result(job, fake_result).")
    print()

    # 5) Confirm result file exists and is readable
    rp: Path = result_path(job.run_id)
    print("[5] Result path:", rp)
    print("    exists:", rp.exists())

    if not rp.exists():
        print("    ERROR: result_path does not exist after save_job_result.")
        raise SystemExit(1)

    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
    except Exception as e:
        print("    ERROR: Could not load JSON from result file:", e)
        raise SystemExit(1)

    print("    Loaded result summary:", data.get("summary"))
    print()
    print("=== DEBUG QUEUE SMOKE TEST COMPLETE (OK) ===")

if __name__ == "__main__":
    main()
