# engine_worker_debug.py
"""
Minimal debug engine worker for the Autonomous Research Agent.

Usage:
    python engine_worker_debug.py

What it does:
    - Prints ARA_RUNS_DIR and all queue directories on startup.
    - Repeatedly:
        * load_next_pending_job()
        * Build a fake result (no TGRM, no RYE)
        * save_job_result()
    - This proves that:
        * Streamlit is writing jobs into the same BASE_DIR
        * The worker can see them
        * Finished results land where Streamlit expects

Once this is working, you can swap back to your real engine_worker
knowing that the queue wiring is correct.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from agent.run_jobs import (
    BASE_DIR,
    PENDING_DIR,
    ACTIVE_DIR,
    FINISHED_DIR,
    ERROR_DIR,
    list_jobs as list_run_jobs,
    load_next_pending_job,
    save_job_result,
    mark_job_error,
    result_path,
)

POLL_INTERVAL_SECONDS = 2.0

def debug_print_startup() -> None:
    print("=== ENGINE WORKER DEBUG START ===")
    print("ARA_RUNS_DIR env:", os.environ.get("ARA_RUNS_DIR", "(not set)"))
    print("run_jobs.BASE_DIR:", BASE_DIR)
    print("PENDING_DIR      :", PENDING_DIR)
    print("ACTIVE_DIR       :", ACTIVE_DIR)
    print("FINISHED_DIR     :", FINISHED_DIR)
    print("ERROR_DIR        :", ERROR_DIR)

    try:
        queued = list_run_jobs(status="queued")
    except TypeError:
        queued = list_run_jobs()  # type: ignore[call-arg]

    print("Queued jobs at startup:", [j.run_id for j in queued])
    print("=================================")


def fake_engine_run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extremely simple stand in for the real engine.

    It just echoes the goal and some config values into a fake result JSON.
    """
    goal = config.get("goal", "unknown goal")
    domain = config.get("domain", "general")
    mode = config.get("mode", "single")
    total_cycles = config.get("total_cycles", 0)

    return {
        "status": "finished",
        "summary": f"[DEBUG ENGINE] Fake run for goal: {goal}",
        "key_findings": [
            f"Ran in debug mode; no real TGRM cycles executed.",
            f"Domain: {domain}, mode: {mode}, total_cycles: {total_cycles}",
        ],
        "cycles": [],
        "rye_metrics": {},
        "sources": [],
        "debug": {
            "note": "engine_worker_debug.py fake engine",
            "original_config": config,
        },
    }


def main_loop() -> None:
    debug_print_startup()

    while True:
        # Poll queue
        job = load_next_pending_job()
        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        print(f"[ENGINE DEBUG] Picked job {job.run_id}, status={job.status}")
        print(f"  goal: {job.config.get('goal')}")
        print(f"  mode: {job.config.get('mode')} domain: {job.config.get('domain')}")

        try:
            result_obj = fake_engine_run(job.config)
            save_job_result(job, result_obj)
            rp = result_path(job.run_id)
            print(f"[ENGINE DEBUG] Finished job {job.run_id}, wrote result to {rp}")
        except Exception as e:
            print(f"[ENGINE DEBUG] ERROR processing job {job.run_id}: {e}")
            try:
                mark_job_error(job, {"error": str(e)})
            except Exception as inner:
                print(f"[ENGINE DEBUG] ERROR marking job as error: {inner}")

        # Small pause to avoid log spam if loop is very fast
        time.sleep(0.5)


if __name__ == "__main__":
    main_loop()
