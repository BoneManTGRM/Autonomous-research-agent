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


# -------------------------------------------------------------------
# Small helpers to handle both RunJob objects and dicts
# -------------------------------------------------------------------
def _get_run_id(job: Any) -> str:
    """Return run/job id for either RunJob object or dict."""
    rid = getattr(job, "run_id", None)
    if rid is None and isinstance(job, dict):
        rid = job.get("run_id") or job.get("job_id")
    return str(rid or "unknown")


def _get_status(job: Any) -> str:
    status = getattr(job, "status", None)
    if status is None and isinstance(job, dict):
        status = job.get("status")
    return str(status or "unknown")


def _get_config(job: Any) -> Dict[str, Any]:
    cfg = getattr(job, "config", None)
    if isinstance(job, dict):
        cfg = job.get("config", cfg)
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg


# -------------------------------------------------------------------
# Startup debug info
# -------------------------------------------------------------------
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
    except Exception as e:
        print("Error calling list_run_jobs at startup:", repr(e))
        queued = []

    print("Queued jobs at startup:", [_get_run_id(j) for j in queued])
    print("=================================")


# -------------------------------------------------------------------
# Fake engine
# -------------------------------------------------------------------
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
            "Ran in debug mode; no real TGRM cycles executed.",
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


# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------
def main_loop() -> None:
    debug_print_startup()

    while True:
        # Poll queue
        try:
            job = load_next_pending_job()
        except Exception as e:
            print("[ENGINE DEBUG] ERROR calling load_next_pending_job:", repr(e))
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        run_id = _get_run_id(job)
        status = _get_status(job)
        config = _get_config(job)

        print(f"[ENGINE DEBUG] Picked job {run_id}, status={status}")
        print(f"  goal : {config.get('goal')}")
        print(f"  mode : {config.get('mode')}")
        print(f"  domain: {config.get('domain')}")

        try:
            result_obj = fake_engine_run(config)
            save_job_result(job, result_obj)
            rp = result_path(run_id)
            print(f"[ENGINE DEBUG] Finished job {run_id}, wrote result to {rp}")
        except Exception as e:
            print(f"[ENGINE DEBUG] ERROR processing job {run_id}: {repr(e)}")
            try:
                mark_job_error(job, {"error": str(e)})
            except Exception as inner:
                print(f"[ENGINE DEBUG] ERROR marking job as error: {repr(inner)}")

        # Small pause to avoid log spam if loop is very fast
        time.sleep(0.5)


if __name__ == "__main__":
    main_loop()
