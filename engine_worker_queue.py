"""
Queue-based background worker for the Autonomous Research Agent.

This worker runs on Render (or locally) as a Background Worker and executes
REAL finite-only ARA runs, not just placeholders.

Behavior:
- Watches a queue directory for JSON job files.
- For each job:
    * Loads the job config.
    * Decides mode: "single" or "swarm".
    * Runs the appropriate finite-only engine call on CoreAgent.
    * Computes diagnostics and optional post-run intelligence.
    * Writes a result JSON file.
    * Removes the original job file (or moves it aside on error).

Queue layout (per-process, simple file-based):
    RUN_DIR = ARA_RUNS_DIR or ARA_RUN_DIR or "runs"
    Queue:   RUN_DIR / "queue"
    Results: RUN_DIR / "results"

This is a LIGHTWEIGHT queue front-end that reuses the same CoreAgent,
MemoryStore, presets, RYE diagnostics, and safety clamps as engine_worker.py.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# ARA imports: reuse the main engine helpers + safety rails
# ---------------------------------------------------------------------

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore
from agent.presets import get_preset
from agent.rye_metrics import build_run_diagnostics

from engine_worker import (  # type: ignore[import]
    init_agent_from_config,
    build_goal_and_domain,
    detect_tools,
    _build_source_controls,
    _clamp_int,
    _clamp_minutes,
    _run_post_run_intelligence,
    _log_run_manifest,
    _update_worker_state,
    _heartbeat,
    HARD_MAX_CYCLES,
    HARD_MAX_ROUNDS,
    HARD_MAX_MINUTES,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# For consistency with engine_worker, prefer ARA_RUNS_DIR.
# Fallback to ARA_RUN_DIR or "runs" if not set.
RUN_DIR = Path(
    os.environ.get("ARA_RUNS_DIR") or os.environ.get("ARA_RUN_DIR", "runs")
).resolve()

QUEUE_DIR = RUN_DIR / "queue"
RESULTS_DIR = RUN_DIR / "results"

POLL_SECONDS = 2.0  # how often to look for new jobs

# These are initialized in main()
AGENT: Optional[CoreAgent] = None
BASE_CONFIG: Dict[str, Any] = {}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_dirs() -> None:
    """Make sure queue and results directories exist."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_job(job_path: Path) -> Dict[str, Any]:
    """Load a job JSON file."""
    with job_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_result(job_id: str, payload: Dict[str, Any]) -> None:
    """Write a result JSON file for the completed job."""
    result_path = RESULTS_DIR / f"{job_id}.result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _get_agent() -> CoreAgent:
    """Return the global CoreAgent (initialized in main)."""
    global AGENT
    if AGENT is None:
        raise RuntimeError("CoreAgent not initialized. main() must run first.")
    return AGENT


def _get_base_config() -> Dict[str, Any]:
    """Return the base config loaded with the agent."""
    global BASE_CONFIG
    return dict(BASE_CONFIG)


# ---------------------------------------------------------------------
# Core job execution logic
# ---------------------------------------------------------------------


def _run_single_job(
    agent: CoreAgent,
    *,
    job_id: str,
    job: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a single-agent finite run for this job.

    Job fields (all optional, with sensible defaults):
        goal, domain, max_cycles, max_minutes, stop_rye,
        role, runtime_profile, resume, source_controls, watchdog_minutes
    """
    base_goal, base_domain = build_goal_and_domain()
    goal = str(job.get("goal", base_goal))
    domain = str(job.get("domain", base_domain))

    preset_cfg = get_preset(domain)

    # Cycle and time bounds
    requested_cycles = int(job.get("max_cycles", job.get("cycles", HARD_MAX_CYCLES)))
    max_cycles = _clamp_int(requested_cycles, HARD_MAX_CYCLES, "queue.single.max_cycles")

    raw_max_minutes = job.get("max_minutes")
    max_minutes: Optional[float]
    if raw_max_minutes is not None:
        try:
            max_minutes = float(raw_max_minutes)
        except Exception:
            max_minutes = None
    else:
        max_minutes = None

    max_minutes = _clamp_minutes(max_minutes, "queue.single.max_minutes")

    # Stop RYE threshold
    stop_rye = job.get("stop_rye")
    if stop_rye is not None:
        try:
            stop_rye = float(stop_rye)
        except Exception:
            stop_rye = None

    # Runtime profile + role
    runtime_profile = job.get(
        "runtime_profile",
        preset_cfg.get("default_runtime_profile"),
    )
    role = str(job.get("role", "agent"))
    resume = bool(job.get("resume", True))
    watchdog_minutes = float(job.get("watchdog_minutes", 5.0))

    # Source controls
    config_for_sources = _get_base_config()
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    # Override from job if provided
    if isinstance(job.get("source_controls"), dict):
        override_sc = job["source_controls"]
        for k, v in override_sc.items():
            source_controls[str(k)] = bool(v)

    tool_flags = detect_tools()

    print("=== Queue worker: Single Agent Job (Finite Only) ===")
    print(f"Job id: {job_id}")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Role: {role}")
    print(f"Max cycles (clamped): {max_cycles}")
    print(
        "Max minutes (clamped): "
        f"{max_minutes if max_minutes is not None else 'None (cycles-only guard)'}"
    )
    print(
        "Stop RYE threshold: "
        f"{stop_rye if stop_rye is not None else 'None'}"
    )
    print(f"Runtime profile: {runtime_profile or 'None'}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    print(
        f"[Safety] HARD_MAX_CYCLES={HARD_MAX_CYCLES}, "
        f"HARD_MAX_MINUTES={HARD_MAX_MINUTES}"
    )

    # Update worker state so UI can see this run
    _update_worker_state(
        agent,
        status="running_job",
        mode="single",
        goal=goal,
        domain=domain,
        roles=[role],
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=job_id,
        experiment_mode="queue_worker_single",
        extra={"job_payload": job},
    )
    _heartbeat(agent, label="queue_single_start", run_id=job_id)

    summaries: List[Dict[str, Any]] = agent.run_continuous(
        goal=goal,
        max_cycles=max_cycles,
        stop_rye=stop_rye,
        role=role,
        source_controls=source_controls,
        pdf_bytes=None,
        biomarker_snapshot=None,
        domain=domain,
        max_minutes=max_minutes,
        forever=False,
        resume_from_checkpoint=resume,
        watchdog_interval_minutes=watchdog_minutes,
        runtime_profile=runtime_profile,
    )

    _heartbeat(agent, label="queue_single_finished", run_id=job_id)

    print("=== Queue single-agent run finished cleanly (finite) ===")
    print(f"Total completed cycles: {len(summaries)}")

    diagnostics: Dict[str, Any] = {}
    try:
        diagnostics = build_run_diagnostics(history=summaries, domain=domain, window=10)
        print(f"RYE avg: {diagnostics.get('rye_avg')}")
        print(f"RYE median: {diagnostics.get('rye_median')}")
        print(f"RYE last: {diagnostics.get('rye_last')}")
        print(f"Stability index: {diagnostics.get('stability_index')}")
        print(f"Recovery momentum: {diagnostics.get('recovery_momentum')}")
    except Exception:
        print("Diagnostics computation failed for queue single job.")

    intelligence_info = _run_post_run_intelligence(
        agent,
        mode="single",
        goal=goal,
        domain=domain,
        run_id=job_id,
        history=summaries,
    )

    extra_manifest: Dict[str, Any] = {
        "engine": "queue_single",
        "job_payload": job,
        "diagnostics_snapshot": diagnostics,
    }
    if intelligence_info:
        extra_manifest["intelligence"] = intelligence_info

    try:
        _log_run_manifest(
            agent,
            job_id,
            mode="single",
            domain=domain,
            goal=goal,
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            summaries=summaries,
            extra=extra_manifest,
        )
    except Exception:
        print("Manifest logging failed for queue single job.")

    _update_worker_state(
        agent,
        status="idle",
        mode="single",
        goal=goal,
        domain=domain,
        roles=[role],
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=job_id,
        experiment_mode="queue_worker_single",
        extra={
            "diagnostics": diagnostics,
            "intelligence": intelligence_info if intelligence_info else None,
        },
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "single",
        "goal": goal,
        "domain": domain,
        "runtime_profile": runtime_profile,
        "max_minutes": max_minutes,
        "max_cycles": max_cycles,
        "stop_rye": stop_rye,
        "summaries": summaries,
        "diagnostics": diagnostics,
        "intelligence": intelligence_info,
        "note": "Queue worker executed finite single-agent ARA run.",
    }


def _run_swarm_job(
    agent: CoreAgent,
    *,
    job_id: str,
    job: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a swarm finite run for this job.

    Job fields (all optional, with sensible defaults):
        goal, domain, roles, max_rounds, max_minutes, stop_rye,
        runtime_profile, resume, source_controls, watchdog_minutes
    """
    base_goal, base_domain = build_goal_and_domain()
    goal = str(job.get("goal", base_goal))
    domain = str(job.get("domain", base_domain))

    preset_cfg = get_preset(domain)

    requested_rounds = int(
        job.get("max_rounds", job.get("rounds", HARD_MAX_ROUNDS))
    )
    max_rounds = _clamp_int(
        requested_rounds, HARD_MAX_ROUNDS, "queue.swarm.max_rounds"
    )

    raw_max_minutes = job.get("max_minutes")
    max_minutes: Optional[float]
    if raw_max_minutes is not None:
        try:
            max_minutes = float(raw_max_minutes)
        except Exception:
            max_minutes = None
    else:
        max_minutes = None

    max_minutes = _clamp_minutes(max_minutes, "queue.swarm.max_minutes")

    stop_rye = job.get("stop_rye")
    if stop_rye is not None:
        try:
            stop_rye = float(stop_rye)
        except Exception:
            stop_rye = None

    runtime_profile = job.get(
        "runtime_profile",
        preset_cfg.get("default_runtime_profile"),
    )

    roles = job.get("roles")
    if isinstance(roles, (list, tuple)):
        roles_list = [str(r) for r in roles]
    else:
        try:
            roles_list = agent.get_agent_roles()
        except Exception:
            roles_list = ["agent"]

    resume = bool(job.get("resume", True))
    watchdog_minutes = float(job.get("watchdog_minutes", 5.0))

    # Source controls
    config_for_sources = _get_base_config()
    if "default_source_controls" not in config_for_sources:
        sc_preset = preset_cfg.get("source_controls")
        if isinstance(sc_preset, dict):
            config_for_sources["default_source_controls"] = sc_preset
    source_controls = _build_source_controls(config_for_sources)

    if isinstance(job.get("source_controls"), dict):
        override_sc = job["source_controls"]
        for k, v in override_sc.items():
            source_controls[str(k)] = bool(v)

    tool_flags = detect_tools()

    print("=== Queue worker: Swarm Job (Finite Only) ===")
    print(f"Job id: {job_id}")
    print(f"Goal: {goal}")
    print(f"Domain: {domain}")
    print(f"Roles: {roles_list}")
    print(f"Max rounds (clamped): {max_rounds}")
    print(
        "Max minutes (clamped): "
        f"{max_minutes if max_minutes is not None else 'None (rounds-only guard)'}"
    )
    print(
        "Stop RYE threshold: "
        f"{stop_rye if stop_rye is not None else 'None'}"
    )
    print(f"Runtime profile: {runtime_profile or 'None'}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Watchdog interval (min): {watchdog_minutes}")
    print(f"Source controls: {source_controls}")
    print(f"Tool flags: web={tool_flags['web']}, sandbox={tool_flags['sandbox']}")
    print(
        f"[Safety] HARD_MAX_ROUNDS={HARD_MAX_ROUNDS}, "
        f"HARD_MAX_MINUTES={HARD_MAX_MINUTES}"
    )

    _update_worker_state(
        agent,
        status="running_job",
        mode="swarm",
        goal=goal,
        domain=domain,
        roles=roles_list,
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=job_id,
        experiment_mode="queue_worker_swarm",
        extra={"job_payload": job},
    )
    _heartbeat(agent, label="queue_swarm_start", run_id=job_id)

    summaries: List[Dict[str, Any]] = agent.run_swarm_continuous(
        goal=goal,
        max_rounds=max_rounds,
        stop_rye=stop_rye,
        roles=roles_list,
        source_controls=source_controls,
        pdf_bytes=None,
        biomarker_snapshot=None,
        domain=domain,
        max_minutes=max_minutes,
        forever=False,
        resume_from_checkpoint=resume,
        watchdog_interval_minutes=watchdog_minutes,
        runtime_profile=runtime_profile,
    )

    _heartbeat(agent, label="queue_swarm_finished", run_id=job_id)

    print("=== Queue swarm run finished cleanly (finite) ===")
    print(
        "Total summaries produced across all roles and rounds: "
        f"{len(summaries)}"
    )

    diagnostics: Dict[str, Any] = {}
    try:
        diagnostics = build_run_diagnostics(history=summaries, domain=domain, window=10)
        print(f"RYE avg: {diagnostics.get('rye_avg')}")
        print(f"RYE median: {diagnostics.get('rye_median')}")
        print(f"RYE last: {diagnostics.get('rye_last')}")
        print(f"Stability index: {diagnostics.get('stability_index')}")
        print(f"Recovery momentum: {diagnostics.get('recovery_momentum')}")
    except Exception:
        print("Diagnostics computation failed for queue swarm job.")

    intelligence_info = _run_post_run_intelligence(
        agent,
        mode="swarm",
        goal=goal,
        domain=domain,
        run_id=job_id,
        history=summaries,
    )

    extra_manifest: Dict[str, Any] = {
        "engine": "queue_swarm",
        "roles": roles_list,
        "job_payload": job,
        "diagnostics_snapshot": diagnostics,
    }
    if intelligence_info:
        extra_manifest["intelligence"] = intelligence_info

    try:
        _log_run_manifest(
            agent,
            job_id,
            mode="swarm",
            domain=domain,
            goal=goal,
            runtime_profile=runtime_profile,
            stop_rye=stop_rye,
            max_minutes=max_minutes,
            summaries=summaries,
            extra=extra_manifest,
        )
    except Exception:
        print("Manifest logging failed for queue swarm job.")

    _update_worker_state(
        agent,
        status="idle",
        mode="swarm",
        goal=goal,
        domain=domain,
        roles=roles_list,
        runtime_profile=runtime_profile,
        stop_rye=stop_rye,
        max_minutes=max_minutes,
        run_id=job_id,
        experiment_mode="queue_worker_swarm",
        extra={
            "diagnostics": diagnostics,
            "intelligence": intelligence_info if intelligence_info else None,
        },
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "swarm",
        "goal": goal,
        "domain": domain,
        "runtime_profile": runtime_profile,
        "max_minutes": max_minutes,
        "max_rounds": max_rounds,
        "stop_rye": stop_rye,
        "summaries": summaries,
        "diagnostics": diagnostics,
        "intelligence": intelligence_info,
        "note": "Queue worker executed finite swarm ARA run.",
    }


def _process_job(job_path: Path) -> None:
    """Handle a single job from the queue."""
    agent = _get_agent()
    job = _load_job(job_path)
    job_id = str(job.get("job_id") or job_path.stem)

    print(f"[queue-worker] 🧵 Starting job {job_id} from {job_path}")
    started = datetime.utcnow().isoformat() + "Z"

    mode = str(job.get("mode", job.get("engine_mode", "single"))).lower()
    if mode not in {"single", "swarm"}:
        print(
            f"[queue-worker] Unsupported mode '{mode}' in job {job_id}, "
            "defaulting to 'single'."
        )
        mode = "single"

    try:
        if mode == "swarm":
            result_core = _run_swarm_job(agent, job_id=job_id, job=job)
        else:
            result_core = _run_single_job(agent, job_id=job_id, job=job)

        finished = datetime.utcnow().isoformat() + "Z"

        result_payload = dict(result_core)
        result_payload.setdefault("status", "completed")
        result_payload["started_at"] = started
        result_payload["finished_at"] = finished

        _write_result(job_id, result_payload)
        print(f"[queue-worker] ✅ Finished job {job_id}, wrote result file.")

        # Remove job from queue
        job_path.unlink(missing_ok=True)
    except Exception as exc:
        finished = datetime.utcnow().isoformat() + "Z"
        tb = traceback.format_exc()
        print(f"[queue-worker] ❌ Fatal error in job {job_id}: {exc!r}")
        print(tb)

        error_payload = {
            "job_id": job_id,
            "status": "error",
            "mode": mode,
            "started_at": started,
            "finished_at": finished,
            "error_message": str(exc),
            "traceback": tb,
            "note": "Queue worker hit an error while running this job.",
            "job_payload": job,
        }
        _write_result(job_id, error_payload)

        # Move the bad job aside for inspection
        bad_path = job_path.with_suffix(".error.json")
        try:
            job_path.rename(bad_path)
        except Exception as e2:
            print(
                f"[queue-worker] ⚠️ Failed to move bad job {job_id} "
                f"to error file: {e2!r}"
            )


def main() -> None:
    """
    Main loop: initialize CoreAgent once and watch the queue directory.

    This worker:
        - Reuses the same CoreAgent + MemoryStore config as engine_worker.
        - Applies finite-only safety caps (no forever runs).
        - Processes single and swarm jobs from RUN_DIR / 'queue'.
    """
    global AGENT, BASE_CONFIG

    _ensure_dirs()
    print(f"[queue-worker] RUN_DIR:   {RUN_DIR}")
    print(f"[queue-worker] Queue dir: {QUEUE_DIR}")
    print(f"[queue-worker] Results:   {RESULTS_DIR}")

    # Initialize agent + config using the shared helper
    AGENT, BASE_CONFIG = init_agent_from_config()
    print("[queue-worker] CoreAgent + MemoryStore initialized from config.")

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
            # Do not crash the worker because of one bad job.
            print(f"[queue-worker] ❌ Error processing {job_path}: {exc!r}")
            tb = traceback.format_exc()
            print(tb)
            bad_path = job_path.with_suffix(".fatal.json")
            try:
                job_path.rename(bad_path)
            except Exception as e2:
                print(
                    f"[queue-worker] ⚠️ Failed to move fatally bad job: {e2!r}"
                )
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
