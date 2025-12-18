# debug_queue.py
"""
Smoke test for the file based job queue (agent/run_jobs.py).

Usage:
    python debug_queue.py
    python debug_queue.py --force   # WARNING: may process a job that isn't the one it created
    python debug_queue.py --cleanup # remove artifacts created by this script (best effort)

What it does:
    1. Prints env + run_jobs module path and all queue directories.
    2. Lists queue folder contents (pending/active/finished/error).
    3. Creates a single debug job via create_job.
    4. Confirms the job file exists on disk (supports multiple naming conventions).
    5. Lists jobs (tries queued + pending, then falls back to list_jobs()).
    6. Simulates an engine worker grabbing the next pending job (safely).
    7. Writes a fake result via save_job_result (signature-adaptive).
    8. Confirms the finished result JSON exists and is readable.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _as_path(x: Any) -> Optional[Path]:
    if x is None:
        return None
    if isinstance(x, Path):
        return x
    try:
        return Path(str(x))
    except Exception:
        return None


def _short_list_dir(p: Optional[Path], limit: int = 60) -> List[str]:
    if p is None:
        return ["(missing path)"]
    try:
        if not p.exists():
            return ["(does not exist)"]
        if not p.is_dir():
            return ["(not a dir)"]
        items = []
        for fp in sorted(p.iterdir()):
            if fp.is_dir():
                items.append(fp.name + "/")
            else:
                items.append(fp.name)
        if len(items) > limit:
            items = items[:limit] + ["..."]
        return items
    except Exception as e:
        return [f"(error listing dir: {e})"]


def _job_get(job: Any, key: str, default: Any = None) -> Any:
    if hasattr(job, key):
        return getattr(job, key)
    if isinstance(job, dict):
        return job.get(key, default)
    return default


def _print_jobs(label: str, jobs: List[Any], max_rows: int = 20) -> None:
    print(label)
    if not jobs:
        print("    (none)")
        return
    for j in jobs[:max_rows]:
        rid = _job_get(j, "run_id", _job_get(j, "job_id", "unknown"))
        st = _job_get(j, "status", "unknown")
        ca = _job_get(j, "created_at", None)
        print(f"    - {rid} (status={st}, created_at={ca})")
    if len(jobs) > max_rows:
        print(f"    ... and {len(jobs) - max_rows} more")


def _call_list_jobs(run_jobs_mod: Any, status: Optional[str]) -> List[Any]:
    list_jobs = getattr(run_jobs_mod, "list_jobs", None)
    if not callable(list_jobs):
        return []

    if status is None:
        try:
            return list_jobs()
        except Exception:
            return []

    # Try list_jobs(status=...)
    try:
        return list_jobs(status=status)
    except TypeError:
        # Older signature without args
        try:
            return list_jobs()
        except Exception:
            return []
    except Exception:
        return []


def _call_load_next_pending(run_jobs_mod: Any, worker_id: str) -> Any:
    fn = getattr(run_jobs_mod, "load_next_pending_job", None)
    if not callable(fn):
        return None

    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    # Try to adapt to different signatures
    try:
        if sig is None:
            return fn()

        params = list(sig.parameters.keys())
        if not params:
            return fn()

        # Common patterns: worker_id=..., worker=..., lock_id=...
        kwargs: Dict[str, Any] = {}
        if "worker_id" in sig.parameters:
            kwargs["worker_id"] = worker_id
        elif "worker" in sig.parameters:
            kwargs["worker"] = worker_id
        elif "lock_id" in sig.parameters:
            kwargs["lock_id"] = worker_id

        # If it takes (pending_dir=...) or similar, leave it alone.
        return fn(**kwargs)
    except TypeError:
        # Fallback to no-arg call
        try:
            return fn()
        except Exception:
            return None
    except Exception:
        return None


def _call_save_job_result(run_jobs_mod: Any, job: Any, result: Dict[str, Any]) -> None:
    fn = getattr(run_jobs_mod, "save_job_result", None)
    if not callable(fn):
        raise RuntimeError("save_job_result not found in agent.run_jobs")

    # Detect signature differences across versions.
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
    except Exception:
        sig = None
        params = []

    # Common variants:
    #   save_job_result(job, result)
    #   save_job_result(run_id, result)
    #   save_job_result(job, result, status="finished")
    #   save_job_result(run_id=..., result=...)
    run_id = _job_get(job, "run_id", _job_get(job, "job_id", None))

    # Prefer calling by keywords when possible (most robust).
    if sig is not None:
        kw: Dict[str, Any] = {}
        if "job" in sig.parameters:
            kw["job"] = job
        elif "run_id" in sig.parameters:
            kw["run_id"] = run_id

        if "result" in sig.parameters:
            kw["result"] = result
        elif "data" in sig.parameters:
            kw["data"] = result

        if "status" in sig.parameters:
            kw["status"] = "finished"

        if kw:
            return fn(**kw)

    # Fallback positional attempts.
    try:
        return fn(job, result)
    except TypeError:
        if run_id is None:
            raise
        return fn(run_id, result)


def _find_job_file(pending_dir: Optional[Path], run_id: str) -> Optional[Path]:
    if pending_dir is None or not pending_dir.exists():
        return None

    # Canonical name used by many implementations
    cand = pending_dir / f"{run_id}_job.json"
    if cand.exists():
        return cand

    # Legacy name: <uuid>.json
    cand2 = pending_dir / f"{run_id}.json"
    if cand2.exists():
        return cand2

    # Last resort: search any json containing the run_id in filename
    try:
        for fp in pending_dir.glob("*.json"):
            if run_id in fp.name:
                return fp
    except Exception:
        pass
    return None


def _safe_unlink(fp: Optional[Path]) -> bool:
    if fp is None:
        return False
    try:
        if fp.exists():
            fp.unlink()
            return True
    except Exception:
        return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Allow processing a job that isn't the one created by this script.")
    ap.add_argument("--cleanup", action="store_true", help="Best-effort cleanup of artifacts created by this script.")
    args = ap.parse_args()

    print("=== DEBUG QUEUE SMOKE TEST ===")
    print("CWD:", os.getcwd())
    print("ARA_RUNS_DIR env :", os.environ.get("ARA_RUNS_DIR", "(not set)"))
    print("ARA_QUEUE_ROOT env:", os.environ.get("ARA_QUEUE_ROOT", "(not set)"))
    print()

    # Import module (lets us print which file is actually used)
    from agent import run_jobs as rj  # type: ignore

    print("run_jobs module file:", getattr(rj, "__file__", "(unknown)"))
    print()

    BASE_DIR = getattr(rj, "BASE_DIR", None)
    QUEUE_ROOT = getattr(rj, "QUEUE_ROOT", None)
    PENDING_DIR = getattr(rj, "PENDING_DIR", None)
    ACTIVE_DIR = getattr(rj, "ACTIVE_DIR", None)
    FINISHED_DIR = getattr(rj, "FINISHED_DIR", None)
    ERROR_DIR = getattr(rj, "ERROR_DIR", None)

    print("BASE_DIR       :", BASE_DIR)
    if QUEUE_ROOT is not None:
        print("QUEUE_ROOT     :", QUEUE_ROOT)
    print("PENDING_DIR    :", PENDING_DIR)
    print("ACTIVE_DIR     :", ACTIVE_DIR)
    print("FINISHED_DIR   :", FINISHED_DIR)
    print("ERROR_DIR      :", ERROR_DIR)
    print()

    p_pending = _as_path(PENDING_DIR)
    p_active = _as_path(ACTIVE_DIR)
    p_finished = _as_path(FINISHED_DIR)
    p_error = _as_path(ERROR_DIR)

    print("[0] Directory contents BEFORE create_job")
    print("    pending :", _short_list_dir(p_pending))
    print("    active  :", _short_list_dir(p_active))
    print("    finished:", _short_list_dir(p_finished))
    print("    error   :", _short_list_dir(p_error))
    print()

    create_job = getattr(rj, "create_job", None)
    if not callable(create_job):
        raise SystemExit("ERROR: create_job not found in agent.run_jobs")

    # 1) Create a fake job
    config = {
        "goal": "Debug queue smoke test",
        "domain": "general",
        "mode": "single",
        "total_cycles": 1,
    }
    meta = {"run_label": "debug_smoke_test"}

    run_id = create_job(config=config, meta=meta)
    run_id = str(run_id)
    print(f"[1] Created job with run_id = {run_id}")
    print()

    # 1b) Confirm job file exists
    job_fp = _find_job_file(p_pending, run_id)
    print("[1b] Expected job file in pending dir:")
    if job_fp is None:
        print("    NOT FOUND (this is a strong signal of a path mismatch or create_job writing elsewhere)")
    else:
        print("   ", job_fp, "(exists:", job_fp.exists(), ")")
    print()

    # 2) List queued/pending jobs via list_jobs()
    queued = _call_list_jobs(rj, status="queued")
    pending = _call_list_jobs(rj, status="pending")
    all_jobs = _call_list_jobs(rj, status=None)

    _print_jobs("[2] list_jobs(status='queued'):", queued)
    print()
    _print_jobs("[2] list_jobs(status='pending'):", pending)
    print()
    _print_jobs("[2] list_jobs() (no status):", all_jobs)
    print()

    # 3) Simulate engine picking up the next pending job
    worker_id = "debug_queue.py"
    job = _call_load_next_pending(rj, worker_id=worker_id)

    if job is None:
        print("[3] ERROR: load_next_pending_job() returned None.")
        print("    That means the engine-side loader cannot see any pending jobs in its configured PENDING_DIR.")
        raise SystemExit(1)

    got_run_id = str(_job_get(job, "run_id", _job_get(job, "job_id", "unknown")))
    got_status = _job_get(job, "status", "unknown")
    print(f"[3] load_next_pending_job(...) -> run_id={got_run_id}, status={got_status}")

    if got_run_id != run_id and not args.force:
        print()
        print("    SAFETY STOP: The next pending job is NOT the job created by this script.")
        print("    This means the queue is not empty and processing would modify someone else's job.")
        print("    Re-run after clearing the queue OR run with --force if you accept that risk.")
        raise SystemExit(2)

    print()

    # Show directory contents after pickup
    print("[3b] Directory contents AFTER load_next_pending_job")
    print("    pending :", _short_list_dir(p_pending))
    print("    active  :", _short_list_dir(p_active))
    print("    finished:", _short_list_dir(p_finished))
    print("    error   :", _short_list_dir(p_error))
    print()

    # 4) Simulate engine writing a fake result
    fake_result = {
        "job_id": got_run_id,
        "status": "finished",
        "summary": f"Fake engine summary for goal: {_job_get(job, 'config', {}).get('goal') if isinstance(_job_get(job, 'config', {}), dict) else 'unknown'}",
        "key_findings": ["Test finding A", "Test finding B"],
        "cycles": [],
        "rye_metrics": {"avg_rye": 0.123},
        "sources": [],
        "debug": {"note": "debug_queue.py fake run"},
    }

    try:
        _call_save_job_result(rj, job, fake_result)
    except Exception as e:
        print("[4] ERROR: save_job_result failed:", e)
        raise SystemExit(1)

    print("[4] Called save_job_result(...).")
    print()

    # 5) Confirm result file exists and is readable
    result_path_fn = getattr(rj, "result_path", None)
    if not callable(result_path_fn):
        raise SystemExit("ERROR: result_path not found in agent.run_jobs")

    rp: Path = result_path_fn(got_run_id)
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

    print("[5b] Directory contents AFTER save_job_result")
    print("    pending :", _short_list_dir(p_pending))
    print("    active  :", _short_list_dir(p_active))
    print("    finished:", _short_list_dir(p_finished))
    print("    error   :", _short_list_dir(p_error))
    print()

    # Optional cleanup (best effort)
    if args.cleanup:
        removed = 0
        removed += 1 if _safe_unlink(job_fp) else 0
        removed += 1 if _safe_unlink(rp) else 0
        print(f"[cleanup] Removed {removed} file(s) (best effort).")
        print()

    print("=== DEBUG QUEUE SMOKE TEST COMPLETE (OK) ===")


if __name__ == "__main__":
    main()
