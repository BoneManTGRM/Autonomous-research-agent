# agent/run_jobs.py

from __future__ import annotations

import json
import time
import uuid
import os
import logging
from dataclasses import dataclass, asdict, fields, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

"""
Developer note: run_jobs.py health check and queue behavior

Short version:
    run_jobs.py is structurally correct and is not the reason the cycles
    table is empty in the UI. It is wired correctly to ARA_RUNS_DIR and
    to the runs layout created in start_unified.sh.

What is verified here:

1. BASE_DIR and folders

    ARA_RUNS_DIR is respected and used as the base runs directory:

        _env_runs_raw = os.getenv("ARA_RUNS_DIR")
        _env_runs = _env_runs_raw.strip() if isinstance(_env_runs_raw, str) else None

        if _env_runs:
            BASE_DIR = Path(_env_runs).resolve()
        else:
            BASE_DIR = (REPO_ROOT / "runs").resolve()

    Subfolders:

        PENDING_DIR  = BASE_DIR / "pending"
        ACTIVE_DIR   = BASE_DIR / "active"
        FINISHED_DIR = BASE_DIR / "finished"
        ERROR_DIR    = BASE_DIR / "error"
        QUEUE_DIR    = PENDING_DIR
        LEGACY_QUEUE_DIR = BASE_DIR / "queue"

    All are created at import time. This matches start_unified.sh, which sets
    ARA_RUNS_DIR="/opt/render/project/src/runs", so the queue layout is
    consistent across:

        - start_unified.sh
        - agent/run_jobs.py
        - engine_worker.py (when it imports this module)

2. Job lifecycle and queue behavior

    The key functions for engine workers are:

        - create_job:
            writes metadata JSON into runs/pending and a shadow copy into runs/queue

        - get_next_queued_job:
            returns the oldest queued job using list_jobs(status="queued")

        - claim_next_job:
            marks the next queued job as active and moves it into runs/active

        - load_next_pending_job:
            wrapper around claim_next_job for workers

        - save_job_result:
            writes final result JSON to runs/finished/{run_id}.json
            and moves metadata to runs/finished/{run_id}_job.json

    This is the expected behavior for queue mode.

3. File naming invariants

    For new jobs the canonical naming is:

        pending/<run_id>_job.json
        active/<run_id>_job.json
        active/<run_id>_progress.json
        finished/<run_id>.json          (result payload)
        finished/<run_id>_job.json      (finished metadata for job lists)

    load_next_pending_job and list_jobs only treat *_job.json as runnable
    metadata files. Older jobs that used bare "<run_id>.json" are still
    supported through compatibility helpers.

4. What actually drives the cycles table in the UI

    The cycles / citations / discoveries table in Streamlit depends on:

        1) engine_worker.py
            - uses agent.run_jobs to pick up jobs and write finished results
            - calls CoreAgent.run_continuous or run_swarm_continuous so
              MemoryStore.log_cycle_summary is called for each cycle
            - passes through run_metadata and summaries into result_obj

        2) memory_store.py
            - stores cycle summaries on disk under a path rooted at ARA_RUNS_DIR
            - implements get_cycle_history() to read them back

        3) app_streamlit.py
            - instantiates MemoryStore with the same base path as the worker
            - calls memory_store.get_cycle_history()
            - renders that history as the cycles/citations/discoveries table

Bottom line:
    If queue folders exist and jobs move from pending to active to finished,
    then run_jobs.py is working as intended. If the UI table is still empty,
    focus fixes on engine_worker.py, memory_store.py, and the table renderer
    in app_streamlit.py, not on this queue layer.

Update notes:
    - Canonical job metadata filenames now end in "_job.json" in pending,
      active, finished, and error folders. Finished results stay at
      finished/<run_id>.json, with an optional finished/<run_id>_results.json
      alias on read.
    - load_next_pending_job, claim_next_job and list_jobs prefer "*_job.json"
      and ignore non metadata JSON like "*_progress.json".
    - Backward compatibility is preserved for older "<run_id>.json" jobs.
    - create_job and RunJob.from_dict inject the run_id into the config
      (and common sub-config sections) so the engine and MemoryStore use the
      exact same run_id that the UI uses when generating reports.
    - list_jobs now derives the effective job.status from the folder it was
      loaded from (queued/active/finished/error), and in conflicts a higher
      priority status (finished, error) will override a lower one (active,
      queued). Finished jobs can not incorrectly appear as active in the UI
      even if stale copies exist.
"""

# Public exports (useful for type checkers and explicit imports)
__all__ = [
    "BASE_DIR",
    "PENDING_DIR",
    "ACTIVE_DIR",
    "FINISHED_DIR",
    "ERROR_DIR",
    "QUEUE_DIR",
    "LEGACY_QUEUE_DIR",
    "RunJob",
    "debug_print_layout",
    "_inject_run_id_into_config",
    "_ensure_monitoring_block",
    "MONITORING_DEFAULTS",
    "STATUS_PRIORITY",
    "create_job",
    "enqueue_job",
    "load_job",
    "load_job_by_id",
    "move_job",
    "update_job_status",
    "list_jobs",
    "get_next_queued_job",
    "claim_next_job",
    "progress_path",
    "result_path",
    "error_log_path",
    "load_next_pending_job",
    "save_job_result",
    "mark_job_error",
    "load_job_result",
]

# Resolve repository root so the default runs directory is stable
_THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_FILE_DIR.parent

# Debug toggle (optional, controlled via env var)
DEBUG_RUN_JOBS = os.getenv("ARA_DEBUG_RUNJOBS", "").strip().lower() in ("1", "true", "yes", "on")


def _log(*args: Any) -> None:
    """
    Lightweight debug logger for this module.

    Enable by setting:
        ARA_DEBUG_RUNJOBS=1
    """
    if not DEBUG_RUN_JOBS:
        return
    print("[run_jobs]", *args)
    try:
        import sys

        sys.stdout.flush()
    except Exception:
        pass


def _queue_log(*parts: Any) -> None:
    """
    Queue level logger that always shows up in Render logs.

    Uses the standard logging subsystem when available, with a
    "[Queue]" prefix, and falls back to stdout.
    """
    msg = " ".join(str(p) for p in parts)
    full = f"[Queue] {msg}"
    try:
        logging.getLogger("queue").info(full)
    except Exception:
        print(full)
        try:
            import sys
            sys.stdout.flush()
        except Exception:
            pass


# Monitoring defaults
#
# This block is always present in the final job config under
# config["monitoring"]. The UI and worker can rely on these keys.
MONITORING_DEFAULTS: Dict[str, Any] = {
    # Snapshots and equilibrium view
    "snapshots_enabled": False,           # master toggle for snapshot writing
    "snapshot_interval_cycles": None,     # take snapshot every N cycles (optional)
    "snapshot_interval_minutes": None,    # or every N minutes (optional)
    "snapshot_max_to_keep": 50,           # how many snapshots per run to retain

    # Watchdog heartbeat for MemoryStore diagnostics
    "heartbeat_enabled": True,
    "heartbeat_interval_seconds": 60,     # seconds between heartbeat writes

    # Run state saving for "Last saved run state" diagnostics
    "run_state_enabled": True,
}


def _ensure_monitoring_block(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure config["monitoring"] exists and contains the standard keys.

    Any values already set by the UI or presets are preserved; only missing
    keys are filled with MONITORING_DEFAULTS.

    This function is intentionally idempotent so it can be called multiple
    times safely.
    """
    try:
        cfg: Dict[str, Any] = dict(config)
    except Exception:
        return config

    monitoring = cfg.get("monitoring")
    if not isinstance(monitoring, dict):
        monitoring = {}

    for key, default in MONITORING_DEFAULTS.items():
        monitoring.setdefault(key, default)

    cfg["monitoring"] = monitoring
    return cfg


# Base folder for all runs.
#
# IMPORTANT:
#   On Render your start command (or environment) should set:
#       ARA_RUNS_DIR="/opt/render/project/src/runs"
#
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
# Many older scripts import QUEUE_DIR from this module.
# QUEUE_DIR now points at the canonical pending folder.
QUEUE_DIR = PENDING_DIR

# Optional "queue" folder support (older jobs may still live here).
# Workers that watch runs/queue directly will look here.
LEGACY_QUEUE_DIR = BASE_DIR / "queue"

# Make sure directories exist at import time
for folder in [BASE_DIR, PENDING_DIR, ACTIVE_DIR, FINISHED_DIR, ERROR_DIR, LEGACY_QUEUE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

_log("Initialized BASE_DIR:", BASE_DIR)
_log("PENDING_DIR:", PENDING_DIR, "ACTIVE_DIR:", ACTIVE_DIR, "FINISHED_DIR:", FINISHED_DIR)

# Status priority for conflict resolution in list_jobs
# Higher number wins when the same run_id appears in multiple folders.
STATUS_PRIORITY: Dict[str, int] = {
    "queued": 0,
    "active": 1,
    "finished": 2,
    "error": 3,
}


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
    try:
        pending_list = sorted(p.name for p in PENDING_DIR.glob("*_job.json"))
    except Exception:
        pending_list = []
    print("[run_jobs] Pending jobs visible:", pending_list)
    try:
        active_list = sorted(p.name for p in ACTIVE_DIR.glob("*_job.json"))
    except Exception:
        active_list = []
    print("[run_jobs] Active jobs visible:", active_list)
    try:
        finished_list = sorted(p.name for p in FINISHED_DIR.glob("*_job.json"))
    except Exception:
        finished_list = []
    print("[run_jobs] Finished job metadata visible:", finished_list)
    try:
        error_list = sorted(p.name for p in ERROR_DIR.glob("*_job.json"))
    except Exception:
        error_list = []
    print("[run_jobs] Error jobs visible:", error_list)
    try:
        legacy_list = sorted(p.name for p in LEGACY_QUEUE_DIR.glob("*_job.json"))
    except Exception:
        legacy_list = []
    print("[run_jobs] Legacy queue jobs visible:", legacy_list)
    import sys

    sys.stdout.flush()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write JSON atomically so workers do not see partially written files.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_path, path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def _normalize_status(status: Any) -> str:
    s = str(status or "queued").lower()
    if s in ("pending", "queued"):
        return "queued"
    if s in ("running", "active"):
        return "active"
    if s in ("finished", "error"):
        return s
    return "queued"


def _coerce_positive_int(value: Any) -> Optional[int]:
    """
    Try to convert value to a positive integer.

    Returns:
        int if successful and > 0, otherwise None.
    """
    try:
        v = int(value)
    except Exception:
        return None
    if v <= 0:
        return None
    return v


def _sanitize_limits_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of config where max_cycles, max_rounds, and max_minutes
    are normalized to positive integers when possible.

    No upper cap is enforced here; extremely large values are allowed.
    Global safety limits (if any) should be enforced in engine_worker.py.
    """
    try:
        cfg: Dict[str, Any] = dict(config)
    except Exception:
        return config

    # Helper to apply coercion in multiple nested dicts
    def sanitize_key(obj: Dict[str, Any], key: str) -> None:
        if not isinstance(obj, dict):
            return
        if key in obj:
            val = _coerce_positive_int(obj.get(key))
            if val is None:
                # If value cannot be coerced, drop it rather than leaving junk
                obj.pop(key, None)
            else:
                obj[key] = val

    # Top level
    for key in ("max_cycles", "max_rounds", "max_minutes"):
        sanitize_key(cfg, key)

    # Common nested sections
    for section_key in ("limits", "engine", "runtime"):
        sub = cfg.get(section_key)
        if isinstance(sub, dict):
            for key in ("max_cycles", "max_rounds", "max_minutes"):
                sanitize_key(sub, key)
            cfg[section_key] = sub

    return cfg


def _inject_run_id_into_config(run_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the config passed to the engine carries the same run_id that the
    queue and UI are using.

    This prevents divergence where the engine internally generates a new
    run identifier and writes cycles under a different folder or key.

    Also ensures that config["monitoring"] exists and contains the standard
    monitoring keys (snapshots, heartbeat, run state).
    """
    # First sanitize numeric limits to positive ints
    cfg = _sanitize_limits_in_config(config)
    # Then ensure monitoring block is present
    cfg = _ensure_monitoring_block(cfg)

    try:
        # Top level run_id hint
        cfg.setdefault("run_id", run_id)

        # Common nested sections that may carry their own run_id
        for key in ("engine", "runtime", "run", "agent", "controller"):
            sub = cfg.get(key)
            if isinstance(sub, dict):
                sub.setdefault("run_id", run_id)
                cfg[key] = sub
    except Exception:
        # Never crash job creation because of config munging
        return cfg

    return cfg


def _find_meta_in_folder(run_id: str, folder: Path) -> Optional[Path]:
    """
    Find a job metadata file for this run_id in a folder.

    Supports both the canonical "{run_id}_job.json" and the legacy
    "{run_id}.json" naming so that older runs and newer runs can coexist.
    """
    primary = folder / f"{run_id}_job.json"
    if primary.exists():
        return primary
    alt = folder / f"{run_id}.json"
    if alt.exists():
        return alt
    return None


@dataclass
class RunJob:
    """
    File based representation of a single research run.

    Each job has:
        run_id      : stable identifier used in filenames and UI
        config      : full configuration dict for RunManager or your engine
        status      : queued, active, finished, or error
        created_at  : unix timestamp when job was created
        updated_at  : unix timestamp when job was last updated
        meta        : optional metadata for UI (user prompt, domain, etc)

    STATUS NOTES (normalized):
        - "queued" is the canonical queue status used by engines.
        - "pending" is treated as an alias for "queued".
        - "running" is treated as an alias for "active".

    This file is written by the UI and read by your engine worker.
    """

    run_id: str
    config: Dict[str, Any]
    status: str = "queued"
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
        run_id = str(data["run_id"])
        raw_config = dict(data["config"])
        # Sanitize, inject run_id, and ensure monitoring block
        sanitized_cfg = _sanitize_limits_in_config(raw_config)
        filtered["run_id"] = run_id
        filtered["config"] = _inject_run_id_into_config(run_id, sanitized_cfg)

        # Ensure meta exists and has run_id for UI convenience
        meta = filtered.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("run_id", run_id)
        filtered["meta"] = meta

        return cls(**filtered)

    def save_to(self, folder: Path, filename: Optional[str] = None) -> Path:
        """
        Save the job JSON inside the given folder.

        This writes only the job metadata (config, status, meta, timestamps).
        The engine worker should write the final result into result_path(run_id).

        filename:
            Optional override for the file name. When not provided it defaults
            to "{run_id}_job.json".
        """
        self.updated_at = time.time()
        if filename is None:
            filename = f"{self.run_id}_job.json"
        path = folder / filename
        _atomic_write_json(path, self.to_dict())
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
        Result:   runs/finished/{run_id}.json (or {run_id}_results.json alias)
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

    filename = f"{run_id}_job.json"
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

    Critical invariant:
        The run_id created here is injected into the config so that the
        engine and MemoryStore both write cycles and run state under this
        same identifier. The UI, worker, and reports all agree on run_id.

    Monitoring note:
        This function guarantees config["monitoring"] exists and includes
        snapshot and heartbeat settings, either from the UI or defaults.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    # Sanitize numeric limits and inject run_id into config and common sections.
    # This also ensures that the monitoring block is present.
    cfg_sanitized = _sanitize_limits_in_config(config)
    cfg = _inject_run_id_into_config(run_id, cfg_sanitized)

    # Include run_id in meta for UI convenience, and auto-fill a few
    # important fields if present in the config. This helps the Streamlit UI
    # show richer job information in the job list and diagnostics.
    meta_with_id: Dict[str, Any] = {}
    if isinstance(meta, dict):
        meta_with_id.update(meta)
    meta_with_id.setdefault("run_id", run_id)

    try:
        # Goal
        goal = (
            cfg.get("goal")
            or cfg.get("problem")
            or cfg.get("objective")
            or cfg.get("task")
        )
        if goal and "goal" not in meta_with_id:
            meta_with_id["goal"] = goal

        # Domain
        domain = (cfg.get("domain") or cfg.get("topic") or cfg.get("field"))
        if domain and "domain" not in meta_with_id:
            meta_with_id["domain"] = domain

        # Mode (single, swarm, option_c, etc)
        mode = (
            cfg.get("mode")
            or (isinstance(cfg.get("engine"), dict) and cfg["engine"].get("mode"))
            or None
        )
        if mode and "mode" not in meta_with_id:
            meta_with_id["mode"] = mode

        # Runtime profile
        runtime_profile = (
            cfg.get("runtime_profile")
            or (isinstance(cfg.get("engine"), dict) and cfg["engine"].get("runtime_profile"))
        )
        if runtime_profile and "runtime_profile" not in meta_with_id:
            meta_with_id["runtime_profile"] = runtime_profile

        # Limits
        limits = cfg.get("limits") if isinstance(cfg.get("limits"), dict) else {}
        engine_cfg = cfg.get("engine") if isinstance(cfg.get("engine"), dict) else {}
        runtime_cfg = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}

        max_cycles = (
            cfg.get("max_cycles")
            or limits.get("max_cycles")
            or engine_cfg.get("max_cycles")
            or runtime_cfg.get("max_cycles")
        )
        if max_cycles is not None and "max_cycles" not in meta_with_id:
            meta_with_id["max_cycles"] = max_cycles

        max_rounds = (
            cfg.get("max_rounds")
            or limits.get("max_rounds")
            or engine_cfg.get("max_rounds")
            or runtime_cfg.get("max_rounds")
        )
        if max_rounds is not None and "max_rounds" not in meta_with_id:
            meta_with_id["max_rounds"] = max_rounds

        max_minutes = (
            cfg.get("max_minutes")
            or limits.get("max_minutes")
            or engine_cfg.get("max_minutes")
            or runtime_cfg.get("max_minutes")
        )
        if max_minutes is not None and "max_minutes" not in meta_with_id:
            meta_with_id["max_minutes"] = max_minutes

        # Experiment fingerprint
        experiment_fingerprint = (
            cfg.get("experiment_fingerprint")
            or engine_cfg.get("experiment_fingerprint")
            or runtime_cfg.get("experiment_fingerprint")
        )
        if experiment_fingerprint and "experiment_fingerprint" not in meta_with_id:
            meta_with_id["experiment_fingerprint"] = experiment_fingerprint

        # Monitoring settings surfaced into meta for diagnostics panels
        monitoring_cfg = cfg.get("monitoring") if isinstance(cfg.get("monitoring"), dict) else None
        if isinstance(monitoring_cfg, dict):
            if "snapshots_enabled" in monitoring_cfg and "snapshots_enabled" not in meta_with_id:
                meta_with_id["snapshots_enabled"] = bool(monitoring_cfg.get("snapshots_enabled"))

            if "heartbeat_enabled" in monitoring_cfg and "heartbeat_enabled" not in meta_with_id:
                meta_with_id["heartbeat_enabled"] = bool(monitoring_cfg.get("heartbeat_enabled"))

            if "run_state_enabled" in monitoring_cfg and "run_state_enabled" not in meta_with_id:
                meta_with_id["run_state_enabled"] = bool(monitoring_cfg.get("run_state_enabled"))

            # Numeric monitoring values (coerced to positive ints when present)
            for key in (
                "snapshot_interval_cycles",
                "snapshot_interval_minutes",
                "snapshot_max_to_keep",
                "heartbeat_interval_seconds",
            ):
                if key in monitoring_cfg and key not in meta_with_id:
                    val = monitoring_cfg.get(key)
                    try:
                        ival = int(val)
                    except Exception:
                        continue
                    if ival > 0:
                        meta_with_id[key] = ival
    except Exception:
        # Never fail job creation because of meta enrichment
        pass

    job = RunJob(
        run_id=run_id,
        config=cfg,
        status="queued",
        created_at=time.time(),
        updated_at=time.time(),
        meta=meta_with_id,
    )

    # Save into the canonical pending folder
    pending_path = job.save_to(PENDING_DIR)

    # Shadow copy to queue folder for compatibility with any older watchers
    try:
        job.save_to(LEGACY_QUEUE_DIR)
    except Exception:
        # Compatibility should never crash job creation.
        pass

    _log("Created job", run_id, "status=queued", "BASE_DIR:", BASE_DIR)
    _queue_log(
        "Enqueued job",
        f"run_id={run_id}",
        f"status=queued",
        f"pending_path={pending_path}",
    )
    try:
        import sys

        sys.stdout.flush()
    except Exception:
        pass

    return run_id


def enqueue_job(
    config: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Thin alias around create_job for UIs.

    Streamlit code can call enqueue_job(...) without worrying about
    queue internals.
    """
    return create_job(config=config, meta=meta, run_id=run_id)


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
    # Finished metadata uses "_job.json" and is separate from results
    finished_meta = FINISHED_DIR / f"{run_id}_job.json"
    if finished_meta.exists():
        return load_job(finished_meta)

    # Other statuses: support both "{run_id}_job.json" and "{run_id}.json"
    for folder in [PENDING_DIR, LEGACY_QUEUE_DIR, ACTIVE_DIR, ERROR_DIR]:
        path = _find_meta_in_folder(run_id, folder)
        if path is not None and path.exists():
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
        # Fallback for mixed naming in older runs
        norm_from = from_status.lower()
        candidate_folders: List[Path] = []
        if norm_from in ("queued", "pending"):
            candidate_folders = [PENDING_DIR, LEGACY_QUEUE_DIR]
        elif norm_from in ("running", "active"):
            candidate_folders = [ACTIVE_DIR]
        elif norm_from == "error":
            candidate_folders = [ERROR_DIR]

        src_path = None
        for folder in candidate_folders:
            maybe = _find_meta_in_folder(run_id, folder)
            if maybe is not None and maybe.exists():
                src_path = maybe
                break

        if src_path is None:
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
    _safe_unlink(src_path)

    # Also remove any shadows in queue folders under both naming schemes
    for folder in (PENDING_DIR, LEGACY_QUEUE_DIR):
        _safe_unlink(folder / f"{run_id}.json")
        _safe_unlink(folder / f"{run_id}_job.json")

    job.save_to(dst_path.parent, filename=dst_path.name)
    return job, dst_path


def update_job_status(run_id: str, new_status: str) -> Optional[RunJob]:
    """
    Convenience helper to update a job status without knowing the old state.

    It searches all folders for run_id, then moves it to the requested status.
    Returns the updated job or None if not found.
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
        _safe_unlink(finished_meta)
        job.save_to(dst_path.parent, filename=dst_path.name)
        return job

    # Check other status folders
    for status in ["queued", "pending", "running", "active", "finished", "error"]:
        src_path: Optional[Path] = None
        norm_status = status.lower()
        if norm_status in ("queued", "pending"):
            # Pending or legacy queue
            src_path = _find_meta_in_folder(run_id, PENDING_DIR)
            if src_path is None:
                src_path = _find_meta_in_folder(run_id, LEGACY_QUEUE_DIR)
        elif norm_status in ("running", "active"):
            src_path = _find_meta_in_folder(run_id, ACTIVE_DIR)
        elif norm_status == "finished":
            src_path = FINISHED_DIR / f"{run_id}_job.json"
        elif norm_status == "error":
            src_path = _find_meta_in_folder(run_id, ERROR_DIR)

        if src_path is None or not src_path.exists():
            continue

        job = load_job(src_path)
        job.status = norm_new
        job.updated_at = time.time()
        dst_path = _job_path_for_status(run_id, norm_new)
        _safe_unlink(src_path)

        # remove any stale file in queue dir under both naming schemes
        for folder in (PENDING_DIR, LEGACY_QUEUE_DIR):
            _safe_unlink(folder / f"{run_id}.json")
            _safe_unlink(folder / f"{run_id}_job.json")

        job.save_to(dst_path.parent, filename=dst_path.name)
        return job

    # Also check legacy queue dir directly if not found above
    for legacy_name in (f"{run_id}_job.json", f"{run_id}.json"):
        legacy_path = LEGACY_QUEUE_DIR / legacy_name
        if legacy_path.exists():
            job = load_job(legacy_path)
            job.status = norm_new
            job.updated_at = time.time()
            _safe_unlink(legacy_path)
            dst_path = _job_path_for_status(run_id, norm_new)
            job.save_to(dst_path.parent, filename=dst_path.name)
            return job

    return None


def list_jobs(status: Optional[str] = None, limit: int = 100) -> List[RunJob]:
    """
    List jobs by status for UI or debugging.

    status:
        None        : search all statuses
        queued      : jobs in runs/pending (canonical queue status)
        pending     : alias for queued
        running     : alias for active
        active      : jobs in runs/active
        finished    : jobs in runs/finished (metadata files only)
        error       : jobs in runs/error

    NOTE:
        This function does not rely on the internal status stored in the JSON
        to decide what a job "is". The effective job.status is derived from the
        folder it was loaded from:

            - pending/ or legacy queue -> "queued"
            - active/                  -> "active"
            - finished/                -> "finished"
            - error/                   -> "error"

        This avoids cases where a job that has been moved to finished/ still
        carries an old "active" status in the JSON and appears wrong in the UI.

        For pending and active it only considers "*_job.json" metadata files.
        Legacy "<run_id>.json" jobs are still supported if no job files exist.

    Resolution rule:
        If multiple metadata files for the same run_id exist across folders
        (for example due to a stale active copy and a newer finished copy),
        the record with the higher status priority wins:

            error > finished > active > queued

        If the priority is the same, the record with the newer updated_at
        timestamp wins.
    """
    jobs_by_id: Dict[str, RunJob] = {}

    def collect(folder: Path, folder_status: Optional[str] = None, finished: bool = False) -> None:
        if not folder.exists():
            return

        if finished:
            patterns = ["*_job.json"]
        else:
            # Prefer the canonical "*_job.json". If none exist, fall back
            # to "*.json" for older jobs, while still skipping progress files.
            has_job_files = any(folder.glob("*_job.json"))
            patterns = ["*_job.json"] if has_job_files else ["*.json"]

        for pattern in patterns:
            for path in sorted(folder.glob(pattern)):
                # Skip progress and other non metadata files
                if not finished and path.name.endswith("_progress.json"):
                    continue
                try:
                    job = load_job(path)
                except Exception:
                    continue

                # Force effective status from folder if provided
                if folder_status is not None:
                    job.status = folder_status

                existing = jobs_by_id.get(job.run_id)
                if existing is None:
                    jobs_by_id[job.run_id] = job
                else:
                    existing_prio = STATUS_PRIORITY.get(existing.status, 0)
                    new_prio = STATUS_PRIORITY.get(job.status, 0)

                    if new_prio > existing_prio:
                        jobs_by_id[job.run_id] = job
                    elif new_prio == existing_prio:
                        existing_updated = getattr(existing, "updated_at", existing.created_at)
                        job_updated = getattr(job, "updated_at", job.created_at)
                        if job_updated >= existing_updated:
                            jobs_by_id[job.run_id] = job

    if status is None:
        # Search all status folders plus legacy queue dir
        collect(PENDING_DIR, folder_status="queued")
        collect(LEGACY_QUEUE_DIR, folder_status="queued")
        collect(ACTIVE_DIR, folder_status="active")
        collect(FINISHED_DIR, folder_status="finished", finished=True)
        collect(ERROR_DIR, folder_status="error")
    else:
        norm = status.lower()
        if norm in ("pending", "queued"):
            collect(PENDING_DIR, folder_status="queued")
            collect(LEGACY_QUEUE_DIR, folder_status="queued")
        elif norm in ("running", "active"):
            collect(ACTIVE_DIR, folder_status="active")
        elif norm == "finished":
            collect(FINISHED_DIR, folder_status="finished", finished=True)
        elif norm == "error":
            collect(ERROR_DIR, folder_status="error")

    jobs: List[RunJob] = list(jobs_by_id.values())
    # Sort by created_at descending (newest first for UI)
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


def get_next_queued_job() -> Optional[RunJob]:
    """
    Simple helper for engine workers.

    Returns the oldest queued job (by created_at) from the queue folders,
    or None if there are no queued jobs.
    """
    queued = list_jobs(status="queued", limit=1000)
    if not queued:
        _log("get_next_queued_job: no queued jobs found")
        _queue_log("get_next_queued_job:", "no queued jobs found")
        return None
    # list_jobs returns newest first; for FIFO we take the last (oldest)
    job = queued[-1]
    _log("get_next_queued_job: selected", job.run_id, "created_at", job.created_at)
    _queue_log(
        "get_next_queued_job:",
        f"selected run_id={job.run_id}",
        f"created_at={job.created_at}",
        f"queued_count={len(queued)}",
    )
    return job


def _claim_job_atomic(job: RunJob) -> Optional[RunJob]:
    """
    Atomically claim a specific job by moving its metadata file into ACTIVE_DIR.

    This uses os.replace (atomic rename on same filesystem) to avoid races.
    If another worker claims it first, this returns None.
    """
    run_id = job.run_id

    src_pending = _find_meta_in_folder(run_id, PENDING_DIR)
    src_legacy = _find_meta_in_folder(run_id, LEGACY_QUEUE_DIR)

    src: Optional[Path] = None
    if src_pending is not None and src_pending.exists():
        src = src_pending
    elif src_legacy is not None and src_legacy.exists():
        src = src_legacy
    else:
        _log("claim_job_atomic: no metadata file found for", run_id)
        _queue_log("claim_job_atomic:", f"no metadata file found for run_id={run_id}")
        return None

    # Preserve the file name when moving into ACTIVE_DIR so both
    # "{run_id}.json" and "{run_id}_job.json" keep working.
    dst_active = ACTIVE_DIR / src.name

    try:
        os.replace(str(src), str(dst_active))
    except Exception as e:
        _log("claim_job_atomic: os.replace failed for", run_id, "error:", repr(e))
        _queue_log("claim_job_atomic:", f"os.replace failed for run_id={run_id}", f"error={repr(e)}")
        # If another worker won the race and already created dst_active,
        # treat that as a successful claim elsewhere.
        if dst_active.exists():
            return None
        return None

    # Remove any duplicate shadows in queue folders
    for folder in (PENDING_DIR, LEGACY_QUEUE_DIR):
        _safe_unlink(folder / f"{run_id}.json")
        _safe_unlink(folder / f"{run_id}_job.json")

    # Normalize status inside the job file to "active"
    try:
        claimed = load_job(dst_active)
        claimed.status = "active"
        claimed.updated_at = time.time()
        _atomic_write_json(dst_active, claimed.to_dict())
        _log("claim_job_atomic: claimed", run_id, "at", dst_active)
        _queue_log("claim_job_atomic:", f"claimed run_id={run_id}", f"path={dst_active}")
        return claimed
    except Exception:
        # Best effort fallback if JSON is malformed
        try:
            with dst_active.open("r", encoding="utf8") as f:
                data = json.load(f)
            data["status"] = "active"
            data["updated_at"] = time.time()
            _atomic_write_json(dst_active, data)
            _log("claim_job_atomic: claimed with raw JSON for", run_id)
            _queue_log("claim_job_atomic:", f"claimed (raw JSON) run_id={run_id}", f"path={dst_active}")
            return RunJob.from_dict(data)
        except Exception as e:
            _log("claim_job_atomic: failed to finalize claim for", run_id, "error:", repr(e))
            _queue_log(
                "claim_job_atomic:",
                f"failed to finalize claim for run_id={run_id}",
                f"error={repr(e)}",
            )
            return job


def claim_next_job() -> Optional[RunJob]:
    """
    Atomically claim the next queued job by marking it active.

    Engine workers can call this to get the next job and immediately
    move it to the active folder so that other workers do not pick it up.

    Implementation note:
        This is upgraded to be atomic for multi worker setups.
        It iterates oldest first and attempts an os.replace claim.
    """
    queued = list_jobs(status="queued", limit=2000)
    if not queued:
        _log("claim_next_job: no queued jobs")
        _queue_log("claim_next_job:", "no queued jobs")
        return None

    # Oldest first
    queued.sort(key=lambda j: j.created_at)

    for job in queued:
        claimed = _claim_job_atomic(job)
        if claimed is not None:
            _log("claim_next_job: claimed job", claimed.run_id)
            _queue_log("claim_next_job:", f"claimed run_id={claimed.run_id}")
            return claimed

    _log("claim_next_job: could not claim any queued job")
    _queue_log("claim_next_job:", "could not claim any queued job")
    return None


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
    job = claim_next_job()
    if job is None:
        _log("load_next_pending_job: no job claimed")
        _queue_log("load_next_pending_job:", "no job claimed")
    else:
        _log("load_next_pending_job: returning job", job.run_id)
        _queue_log("load_next_pending_job:", f"returning run_id={job.run_id}")
    return job


def save_job_result(job: RunJob, result_obj: Dict[str, Any]) -> None:
    """
    Write final result JSON for a job and mark it finished.

    Used by engine_worker queue mode.
    """
    # Ensure identifiers are present in the result for UI convenience
    if "job_id" not in result_obj:
        result_obj["job_id"] = job.run_id
    if "run_id" not in result_obj:
        result_obj["run_id"] = job.run_id

    rp = result_path(job.run_id)
    _atomic_write_json(rp, result_obj)

    # Prefer moving the active metadata into finished metadata
    active_meta_candidates = [
        ACTIVE_DIR / f"{job.run_id}_job.json",
        ACTIVE_DIR / f"{job.run_id}.json",  # legacy
    ]
    for active_meta in active_meta_candidates:
        if not active_meta.exists():
            continue
        try:
            j = load_job(active_meta)
        except Exception:
            j = job
        j.status = "finished"
        j.updated_at = time.time()
        _safe_unlink(active_meta)
        j.save_to(FINISHED_DIR, filename=f"{job.run_id}_job.json")
        # Also remove any lingering legacy queue copy
        for folder in (LEGACY_QUEUE_DIR, PENDING_DIR):
            _safe_unlink(folder / f"{job.run_id}.json")
            _safe_unlink(folder / f"{job.run_id}_job.json")
        _log("save_job_result: marked finished", job.run_id)
        _queue_log("save_job_result:", f"marked finished run_id={job.run_id}", f"result_path={rp}")
        return

    # Fallback
    _log("save_job_result: active metadata not found, using update_job_status for", job.run_id)
    _queue_log(
        "save_job_result:",
        f"active metadata not found for run_id={job.run_id}, using update_job_status",
    )
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

    # Prefer moving the active metadata into error
    active_meta_candidates = [
        ACTIVE_DIR / f"{job.run_id}_job.json",
        ACTIVE_DIR / f"{job.run_id}.json",  # legacy
    ]
    for active_meta in active_meta_candidates:
        if not active_meta.exists():
            continue
        try:
            j = load_job(active_meta)
        except Exception:
            j = job
        j.status = "error"
        j.updated_at = time.time()
        _safe_unlink(active_meta)
        j.save_to(ERROR_DIR, filename=f"{job.run_id}_job.json")
        for folder in (LEGACY_QUEUE_DIR, PENDING_DIR):
            _safe_unlink(folder / f"{job.run_id}.json")
            _safe_unlink(folder / f"{job.run_id}_job.json")
        _log("mark_job_error: marked error", job.run_id)
        _queue_log("mark_job_error:", f"marked error run_id={job.run_id}", f"error_path={ep}")
        return

    _log("mark_job_error: active metadata not found, using update_job_status for", job.run_id)
    _queue_log(
        "mark_job_error:",
        f"active metadata not found for run_id={job.run_id}, using update_job_status",
    )
    update_job_status(job.run_id, "error")


# ---------------------------------------------------------------------------
# Optional helpers for UI and debugging
# ---------------------------------------------------------------------------

def load_job_result(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience helper: load the final result JSON for a finished job.

    Accepts either finished/<run_id>.json or finished/<run_id>_results.json
    as the result payload file.
    """
    rp_main = FINISHED_DIR / f"{run_id}.json"
    rp_alt = FINISHED_DIR / f"{run_id}_results.json"
    rp = rp_main if rp_main.exists() else rp_alt
    if not rp.exists():
        return None
    try:
        with rp.open("r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return None
