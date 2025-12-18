# agent/run_jobs.py

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, fields
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

    Queue root:

        If ARA_QUEUE_ROOT is set, it is used as the root for the file based
        queue layout (pending / active / finished / error). This matches
        start_unified.sh, which typically exports:

            ARA_QUEUE_ROOT="$ARA_RUNS_DIR/queue"

        If ARA_QUEUE_ROOT is not set, we fall back to:

            QUEUE_ROOT = BASE_DIR / "queue"

    Primary queue layout (under QUEUE_ROOT):

        PENDING_DIR   = QUEUE_ROOT / "pending"
        ACTIVE_DIR    = QUEUE_ROOT / "active"
        FINISHED_DIR  = QUEUE_ROOT / "finished"
        ERROR_DIR     = QUEUE_ROOT / "error"
        BAD_JOBS_DIR  = QUEUE_ROOT / "bad_jobs"     (quarantine for malformed jobs)
        QUEUE_DIR     = PENDING_DIR                 (canonical queue folder)
        LEGACY_QUEUE_DIR = QUEUE_ROOT               (root alias for older watchers)

    Legacy layout support:

        For backward compatibility with older runs that wrote directly under
        BASE_DIR, we also define:

        LEGACY_PENDING_DIR  = BASE_DIR / "pending"
        LEGACY_ACTIVE_DIR   = BASE_DIR / "active"
        LEGACY_FINISHED_DIR = BASE_DIR / "finished"
        LEGACY_ERROR_DIR    = BASE_DIR / "error"

        list_jobs, load_job_by_id, and the status update helpers will read
        from both the primary QUEUE_ROOT layout and these legacy folders.
        New jobs are always written into QUEUE_ROOT.

2. Job lifecycle and queue behavior

    The key functions for engine workers are:

        - create_job:
            writes metadata JSON into QUEUE_ROOT/pending and a shadow copy
            into QUEUE_ROOT (for older watchers that monitor the root)

        - get_next_queued_job:
            returns the oldest queued job using list_jobs(status="queued")

        - claim_next_job:
            marks the next queued job as active and moves it into
            QUEUE_ROOT/active

        - load_next_pending_job:
            wrapper around claim_next_job for workers

        - save_job_result:
            writes final result JSON to QUEUE_ROOT/finished/{run_id}.json
            and moves metadata to QUEUE_ROOT/finished/{run_id}_job.json

    This is the expected behavior for queue mode.

3. File naming invariants

    For new jobs the canonical naming is:

        pending/<run_id>_job.json
        active/<run_id>_job.json
        active/<run_id>_progress.json
        finished/<run_id>.json          (result payload)
        finished/<run_id>_job.json      (finished metadata for job lists)

    Compatibility with newer workers:
        Some workers (e.g. a hardened internal queue) may archive job state as:

            finished/<run_id>__job.json
            error/<run_id>__job.json

        This module now treats both *_job.json and *__job.json as valid
        metadata files wherever metadata is expected.

    load_next_pending_job and list_jobs only treat metadata files as runnable
    when they are in pending/active. Older jobs that used bare "<run_id>.json"
    are still supported through compatibility helpers (but finished/<run_id>.json
    is reserved for the result payload).

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
    - Canonical job metadata filenames end in "_job.json" in pending,
      active, finished, and error folders. Finished results stay at
      finished/<run_id>.json, with an optional finished/<run_id>_results.json
      alias on read.
    - Newer workers may archive job metadata as "<run_id>__job.json"; this
      module reads both variants.
    - load_next_pending_job, claim_next_job and list_jobs prefer "*_job.json"
      and ignore non metadata JSON like "*_progress.json".
    - Backward compatibility is preserved for older "<run_id>.json" jobs.
    - create_job and RunJob.from_dict inject the run_id into the config
      (and common sub-config sections) so the engine and MemoryStore use the
      exact same run_id that the UI uses when generating reports.
    - list_jobs derives the effective job.status from the folder it was
      loaded from. Higher priority folder status (finished, error) will
      override a lower one (active, queued).

Additional update (2025-12):
    - load_job_by_id no longer returns early if one candidate file is corrupt.
    - list_jobs can now handle mixed legacy "<run_id>.json" and new "*_job.json"
      files in the same folder.
    - Optional shared worker_state.json helpers were added to support a live
      top-of-page status strip and phase counters in Streamlit.
"""

# Public exports (useful for type checkers and explicit imports)
__all__ = [
    "BASE_DIR",
    "QUEUE_ROOT",
    "PENDING_DIR",
    "ACTIVE_DIR",
    "FINISHED_DIR",
    "ERROR_DIR",
    "BAD_JOBS_DIR",
    "QUEUE_DIR",
    "LEGACY_QUEUE_DIR",
    "LOCKS_DIR",
    "WORKER_STATE_PATH",
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
    "write_progress",
    "read_progress",
    "result_path",
    "error_log_path",
    "load_next_pending_job",
    "save_job_result",
    "mark_job_error",
    "load_job_result",
    "read_worker_state",
    "update_worker_state",
]

# Resolve repository root so the default runs directory is stable
_THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_FILE_DIR.parent

# Debug toggle (optional, controlled via env var)
DEBUG_RUN_JOBS = os.getenv("ARA_DEBUG_RUNJOBS", "").strip().lower() in ("1", "true", "yes", "on")

# Internal loggers
_LOGGER = logging.getLogger(__name__)
_QUEUE_LOGGER = logging.getLogger("ara.queue")


def _configure_queue_logger_if_needed() -> None:
    """
    Ensure queue logs have a reasonable chance of showing up even if the
    application didn't configure logging (common in simple workers).

    This avoids surprises where queue events disappear due to a missing
    handler or an INFO-level drop.
    """
    try:
        root = logging.getLogger()
        if _QUEUE_LOGGER.handlers:
            return
        if root.handlers:
            # Let the application's logging config handle it.
            return

        import sys

        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)
        _QUEUE_LOGGER.addHandler(handler)
        _QUEUE_LOGGER.setLevel(logging.INFO)
        _QUEUE_LOGGER.propagate = False
    except Exception:
        # Never fail module import due to logging configuration.
        pass


_configure_queue_logger_if_needed()


def _log(*args: Any) -> None:
    """
    Lightweight debug logger for this module.

    Enable by setting:
        ARA_DEBUG_RUNJOBS=1
    """
    if not DEBUG_RUN_JOBS:
        return
    msg = " ".join(str(a) for a in args)
    try:
        _LOGGER.debug(msg)
    except Exception:
        print("[run_jobs]", msg)
        try:
            import sys

            sys.stdout.flush()
        except Exception:
            pass


def _queue_log(*parts: Any, level: int = logging.INFO) -> None:
    """
    Queue level logger intended for operational visibility.

    Uses the standard logging subsystem and falls back to stdout on failure.
    """
    msg = " ".join(str(p) for p in parts)
    full = f"[Queue] {msg}"
    try:
        _QUEUE_LOGGER.log(level, full)
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
    "snapshots_enabled": False,  # master toggle for snapshot writing
    "snapshot_interval_cycles": None,  # take snapshot every N cycles (optional)
    "snapshot_interval_minutes": None,  # or every N minutes (optional)
    "snapshot_max_to_keep": 50,  # how many snapshots per run to retain
    # Watchdog heartbeat for MemoryStore diagnostics
    "heartbeat_enabled": True,
    "heartbeat_interval_seconds": 60,  # seconds between heartbeat writes
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
    if not isinstance(config, dict):
        return config

    cfg: Dict[str, Any] = dict(config)

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

# Queue root: honor ARA_QUEUE_ROOT if provided, otherwise default
# to BASE_DIR / "queue". This keeps start_unified.sh, engine_worker,
# and Streamlit in sync when they all share the same env.
_env_queue_raw = os.getenv("ARA_QUEUE_ROOT")
_env_queue = _env_queue_raw.strip() if isinstance(_env_queue_raw, str) else None

if _env_queue:
    QUEUE_ROOT = Path(_env_queue).resolve()
else:
    QUEUE_ROOT = (BASE_DIR / "queue").resolve()

# Optional shared worker state file (used by Streamlit to render a fixed top status bar).
# You can override the path with ARA_WORKER_STATE_PATH.
_env_worker_state_raw = os.getenv("ARA_WORKER_STATE_PATH")
_env_worker_state = _env_worker_state_raw.strip() if isinstance(_env_worker_state_raw, str) else None
if _env_worker_state:
    WORKER_STATE_PATH = Path(_env_worker_state).resolve()
else:
    WORKER_STATE_PATH = (BASE_DIR / "worker_state.json").resolve()

# Primary job layout used by the engine worker:
#   - QUEUE_ROOT/pending/   : file based queue of pending jobs (canonical queue)
#   - QUEUE_ROOT/active/    : in progress metadata and optional progress JSON
#   - QUEUE_ROOT/finished/  : final result JSON and finished job metadata
#   - QUEUE_ROOT/error/     : failed job metadata and optional traceback/payload
#   - QUEUE_ROOT/bad_jobs/  : quarantine for malformed or unreadable job files
PENDING_DIR = QUEUE_ROOT / "pending"
ACTIVE_DIR = QUEUE_ROOT / "active"
FINISHED_DIR = QUEUE_ROOT / "finished"
ERROR_DIR = QUEUE_ROOT / "error"
BAD_JOBS_DIR = QUEUE_ROOT / "bad_jobs"

# Canonical queue folder for UIs
QUEUE_DIR = PENDING_DIR

# Legacy layout support (older versions wrote directly under BASE_DIR)
LEGACY_QUEUE_DIR = QUEUE_ROOT  # alias for older watchers expecting "queue" root
LEGACY_PENDING_DIR = BASE_DIR / "pending"
LEGACY_ACTIVE_DIR = BASE_DIR / "active"
LEGACY_FINISHED_DIR = BASE_DIR / "finished"
LEGACY_ERROR_DIR = BASE_DIR / "error"

# Internal lock directory used to claim jobs safely across multiple workers.
# Kept stable for backward compatibility.
_LOCKS_DIR = QUEUE_ROOT / ".locks"
LOCKS_DIR = _LOCKS_DIR  # public alias

# Make sure directories exist at import time
for folder in [
    BASE_DIR,
    QUEUE_ROOT,
    PENDING_DIR,
    ACTIVE_DIR,
    FINISHED_DIR,
    ERROR_DIR,
    BAD_JOBS_DIR,
    LEGACY_QUEUE_DIR,
    LEGACY_PENDING_DIR,
    LEGACY_ACTIVE_DIR,
    LEGACY_FINISHED_DIR,
    LEGACY_ERROR_DIR,
    _LOCKS_DIR,
    WORKER_STATE_PATH.parent,
]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

_log("Initialized BASE_DIR:", BASE_DIR)
_log("QUEUE_ROOT:", QUEUE_ROOT)
_log("PENDING_DIR:", PENDING_DIR, "ACTIVE_DIR:", ACTIVE_DIR, "FINISHED_DIR:", FINISHED_DIR)
_log("WORKER_STATE_PATH:", WORKER_STATE_PATH)

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
    env_queue_val = os.environ.get("ARA_QUEUE_ROOT")
    env_worker_state_val = os.environ.get("ARA_WORKER_STATE_PATH")
    try:
        base_resolved = BASE_DIR.resolve()
    except Exception:
        base_resolved = BASE_DIR

    print("[run_jobs] ARA_RUNS_DIR env (raw):", repr(env_val))
    print("[run_jobs] ARA_RUNS_DIR env (stripped):", repr(_env_runs))
    print("[run_jobs] ARA_QUEUE_ROOT env (raw):", repr(env_queue_val))
    print("[run_jobs] QUEUE_ROOT:", QUEUE_ROOT)
    print("[run_jobs] BASE_DIR:", base_resolved)
    print("[run_jobs] PENDING_DIR:", PENDING_DIR)
    print("[run_jobs] ACTIVE_DIR:", ACTIVE_DIR)
    print("[run_jobs] FINISHED_DIR:", FINISHED_DIR)
    print("[run_jobs] ERROR_DIR:", ERROR_DIR)
    print("[run_jobs] BAD_JOBS_DIR:", BAD_JOBS_DIR)
    print("[run_jobs] LEGACY_QUEUE_DIR:", LEGACY_QUEUE_DIR)
    print("[run_jobs] LEGACY_PENDING_DIR:", LEGACY_PENDING_DIR)
    print("[run_jobs] LEGACY_ACTIVE_DIR:", LEGACY_ACTIVE_DIR)
    print("[run_jobs] LEGACY_FINISHED_DIR:", LEGACY_FINISHED_DIR)
    print("[run_jobs] LEGACY_ERROR_DIR:", LEGACY_ERROR_DIR)
    print("[run_jobs] LOCKS_DIR:", _LOCKS_DIR)
    print("[run_jobs] ARA_WORKER_STATE_PATH env (raw):", repr(env_worker_state_val))
    print("[run_jobs] WORKER_STATE_PATH:", WORKER_STATE_PATH)

    def _names(glob_iter):
        try:
            return sorted(p.name for p in glob_iter)
        except Exception:
            return []

    print("[run_jobs] Pending jobs visible:", _names(PENDING_DIR.glob("*_job.json")))
    print("[run_jobs] Active jobs visible:", _names(ACTIVE_DIR.glob("*_job.json")))
    print("[run_jobs] Finished job metadata visible:", _names(FINISHED_DIR.glob("*_job.json")))
    print("[run_jobs] Error jobs visible:", _names(ERROR_DIR.glob("*_job.json")))
    print("[run_jobs] Legacy queue jobs visible:", _names(LEGACY_QUEUE_DIR.glob("*_job.json")))

    try:
        import sys

        sys.stdout.flush()
    except Exception:
        pass


def _json_default(obj: Any) -> Any:
    """
    Best-effort JSON encoder for non-serializable objects.

    This is intentionally conservative: it prefers stable string forms
    and avoids raising during job/result persistence.
    """
    try:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (uuid.UUID,)):
            return str(obj)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            b = bytes(obj)
            # Avoid accidental huge binary dumps in JSON. Encode.
            return {"__bytes_b64__": base64.b64encode(b).decode("ascii")}
        if isinstance(obj, Exception):
            return {"__exception__": obj.__class__.__name__, "message": str(obj)}
    except Exception:
        pass

    # As a last resort, try common patterns, then string.
    for attr in ("to_dict", "dict", "as_dict"):
        try:
            fn = getattr(obj, attr, None)
            if callable(fn):
                return fn()
        except Exception:
            continue
    try:
        return repr(obj)
    except Exception:
        return str(type(obj))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write JSON atomically so workers do not see partially written files.

    Uses a unique temp file name per write to avoid cross-process races.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_path = path.with_name(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        # Clean up temp file on failures.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Write text atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_path = path.with_name(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _safe_unlink(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def _normalize_status(status: Any) -> str:
    """
    Normalize status strings across queue implementations.

    Canonical statuses:
        queued, active, finished, error

    Accepted aliases:
        pending -> queued
        retrying -> queued
        running -> active
        done/completed/success -> finished
        failed/failure -> error
    """
    s = str(status or "queued").strip().lower()
    if s in ("pending", "queued", "retrying", "requeued"):
        return "queued"
    if s in ("running", "active", "in_progress", "in-progress"):
        return "active"
    if s in ("finished", "done", "completed", "complete", "success", "succeeded", "ok"):
        return "finished"
    if s in ("error", "failed", "failure", "crashed"):
        return "error"
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
    if not isinstance(config, dict):
        return config
    cfg: Dict[str, Any] = dict(config)

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
    cfg = _sanitize_limits_in_config(config if isinstance(config, dict) else {})
    # Then ensure monitoring block is present
    cfg = _ensure_monitoring_block(cfg)

    # Always force run_id at the top level and common nested sections.
    try:
        cfg["run_id"] = run_id

        for key in ("engine", "runtime", "run", "agent", "controller"):
            sub = cfg.get(key)
            if isinstance(sub, dict):
                sub["run_id"] = run_id
                cfg[key] = sub
    except Exception:
        # Never crash job creation because of config munging
        return cfg

    return cfg


def _is_safe_run_id(run_id: str) -> bool:
    if not run_id:
        return False
    if "\x00" in run_id:
        return False
    if os.sep in run_id:
        return False
    if os.altsep and os.altsep in run_id:
        return False
    return True


def _sanitize_run_id_for_filename(run_id: Any, fallback: str = "unknown") -> str:
    """
    Convert an arbitrary run_id-like value into a safe single path component.

    This prevents accidental path traversal if a caller passes a malformed run_id.
    It does not guarantee the returned id exists on disk; it is intended only
    for constructing filenames safely.
    """
    try:
        s = str(run_id).strip()
    except Exception:
        s = ""
    if not s:
        s = fallback
    if "\x00" in s:
        s = s.replace("\x00", "")
    try:
        s = s.replace(os.sep, "_")
    except Exception:
        pass
    if os.altsep:
        try:
            s = s.replace(os.altsep, "_")
        except Exception:
            pass
    return s or fallback


def _find_meta_in_folder(run_id: str, folder: Path) -> Optional[Path]:
    """
    Find a job metadata file for this run_id in a folder.

    Supports:
        - canonical "{run_id}_job.json"
        - alternate "{run_id}__job.json" (some workers archive as this)
        - legacy "{run_id}.json" (older queue versions)

    NOTE:
        In FINISHED_DIR, "{run_id}.json" is typically the result payload, not metadata.
        Callers that need finished metadata should use _find_finished_meta_in_folder().
    """
    primary = folder / f"{run_id}_job.json"
    if primary.exists():
        return primary
    alt2 = folder / f"{run_id}__job.json"
    if alt2.exists():
        return alt2
    alt = folder / f"{run_id}.json"
    if alt.exists():
        return alt
    return None


def _find_finished_meta_in_folder(run_id: str, folder: Path) -> Optional[Path]:
    """
    Finished/error metadata finder that NEVER returns the result payload file.

    Accepts:
        - "{run_id}_job.json"
        - "{run_id}__job.json"
    """
    p1 = folder / f"{run_id}_job.json"
    if p1.exists():
        return p1
    p2 = folder / f"{run_id}__job.json"
    if p2.exists():
        return p2
    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Shared worker state helpers (optional, but useful for Streamlit live UI)
# ---------------------------------------------------------------------------


def read_worker_state(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read the shared worker state file (worker_state.json).

    This is intentionally tolerant: it returns `default` (or {}) on errors.

    Recommended keys:
        - heartbeat_ts (float)
        - status (str): idle|running|backoff|fault|finished
        - run_id (str)
        - phase_index (int)
        - phase_total (int)
        - phase_name (str)
        - cycle (int)
        - updated_at (float)
    """
    base: Dict[str, Any] = dict(default or {})
    path = WORKER_STATE_PATH
    if not path.exists():
        return base
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            base.update(data)
        return base
    except Exception:
        return base


def update_worker_state(update: Dict[str, Any], *, replace: bool = False) -> Path:
    """
    Update (merge) the shared worker state file atomically.

    Args:
        update: dict of keys to set
        replace: if True, overwrite the entire file with `update`

    Returns:
        Path to the worker state file.
    """
    if not isinstance(update, dict):
        update = {}
    if replace:
        state: Dict[str, Any] = dict(update)
    else:
        state = read_worker_state()
        state.update(update)

    # Common timestamp key for UI freshness checks
    state.setdefault("updated_at", time.time())
    _atomic_write_json(WORKER_STATE_PATH, state)
    return WORKER_STATE_PATH


# ---------------------------------------------------------------------------
# Job schema and queue operations
# ---------------------------------------------------------------------------


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
        - "done"/"completed" are treated as an alias for "finished".
        - "failed" is treated as an alias for "error".

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
        if not isinstance(data, dict):
            raise ValueError("RunJob.from_dict expects a dict")

        run_id_raw = data.get("run_id")
        if run_id_raw is None:
            raise ValueError("RunJob.from_dict missing required field: run_id")
        run_id = str(run_id_raw).strip()
        if not _is_safe_run_id(run_id):
            raise ValueError(f"RunJob.from_dict invalid run_id: {run_id!r}")

        cfg_raw = data.get("config")
        if cfg_raw is None:
            raise ValueError("RunJob.from_dict missing required field: config")
        if not isinstance(cfg_raw, dict):
            # Defensive: tolerate non-dict config by coercing to dict if possible.
            try:
                cfg_raw = dict(cfg_raw)  # type: ignore[arg-type]
            except Exception:
                cfg_raw = {}

        now = time.time()
        created_at = _safe_float(data.get("created_at"), now)
        updated_at = _safe_float(data.get("updated_at"), now)

        status = _normalize_status(data.get("status"))

        meta = data.get("meta", None)
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("run_id", run_id)

        # Sanitize, inject run_id, and ensure monitoring block
        sanitized_cfg = _sanitize_limits_in_config(cfg_raw)
        injected_cfg = _inject_run_id_into_config(run_id, sanitized_cfg)

        # Filter to known dataclass fields, but fill required explicitly.
        field_names = {f.name for f in fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in field_names}
        filtered["run_id"] = run_id
        filtered["config"] = injected_cfg
        filtered["status"] = status
        filtered["created_at"] = created_at
        filtered["updated_at"] = updated_at
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
        Metadata: finished/{run_id}_job.json  (canonical)
        Result:   finished/{run_id}.json (or {run_id}_results.json alias)
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


def _run_id_exists(run_id: str) -> bool:
    """
    Check whether any known metadata or result file for this run_id already exists.
    Used to avoid clobbering an existing run when run_id is supplied.
    """
    if not run_id:
        return False
    candidates = [
        # Canonical metadata
        PENDING_DIR / f"{run_id}_job.json",
        ACTIVE_DIR / f"{run_id}_job.json",
        FINISHED_DIR / f"{run_id}_job.json",
        FINISHED_DIR / f"{run_id}__job.json",
        ERROR_DIR / f"{run_id}_job.json",
        ERROR_DIR / f"{run_id}__job.json",
        # Canonical result
        FINISHED_DIR / f"{run_id}.json",
        FINISHED_DIR / f"{run_id}_results.json",
        # Legacy metadata and result layout
        LEGACY_QUEUE_DIR / f"{run_id}_job.json",
        LEGACY_QUEUE_DIR / f"{run_id}.json",
        LEGACY_PENDING_DIR / f"{run_id}_job.json",
        LEGACY_PENDING_DIR / f"{run_id}.json",
        LEGACY_ACTIVE_DIR / f"{run_id}_job.json",
        LEGACY_ACTIVE_DIR / f"{run_id}.json",
        LEGACY_FINISHED_DIR / f"{run_id}_job.json",
        LEGACY_FINISHED_DIR / f"{run_id}__job.json",
        LEGACY_FINISHED_DIR / f"{run_id}.json",
        LEGACY_ERROR_DIR / f"{run_id}_job.json",
        LEGACY_ERROR_DIR / f"{run_id}__job.json",
        LEGACY_ERROR_DIR / f"{run_id}.json",
    ]
    return any(p.exists() for p in candidates)


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
    if not isinstance(config, dict):
        try:
            config = dict(config)  # type: ignore[arg-type]
        except Exception:
            config = {}

    requested_run_id: Optional[str] = None
    if run_id is None:
        run_id = str(uuid.uuid4())
    else:
        requested_run_id = str(run_id).strip()
        run_id = requested_run_id

    if not _is_safe_run_id(run_id):
        _queue_log(
            "create_job:",
            f"unsafe run_id provided, generating new run_id (provided={run_id!r})",
            level=logging.WARNING,
        )
        requested_run_id = run_id
        run_id = str(uuid.uuid4())

    # Avoid clobbering existing runs if a caller supplies a run_id.
    if _run_id_exists(run_id):
        _queue_log(
            "create_job:",
            f"run_id already exists, generating new run_id (existing={run_id})",
            level=logging.WARNING,
        )
        requested_run_id = requested_run_id or run_id
        for _ in range(5):
            candidate = str(uuid.uuid4())
            if not _run_id_exists(candidate):
                run_id = candidate
                break
        else:
            # Extremely unlikely; fall back to time-based unique id.
            run_id = f"{int(time.time())}-{uuid.uuid4()}"

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
    if requested_run_id and requested_run_id != run_id:
        meta_with_id.setdefault("requested_run_id", requested_run_id)

    try:
        # Goal
        goal = cfg.get("goal") or cfg.get("problem") or cfg.get("objective") or cfg.get("task")
        if goal and "goal" not in meta_with_id:
            meta_with_id["goal"] = goal

        # Domain
        domain = cfg.get("domain") or cfg.get("topic") or cfg.get("field")
        if domain and "domain" not in meta_with_id:
            meta_with_id["domain"] = domain

        # Mode (single, swarm, option_c, etc)
        mode = cfg.get("mode") or (isinstance(cfg.get("engine"), dict) and cfg["engine"].get("mode")) or None
        if mode and "mode" not in meta_with_id:
            meta_with_id["mode"] = mode

        # Runtime profile
        runtime_profile = cfg.get("runtime_profile") or (
            isinstance(cfg.get("engine"), dict) and cfg["engine"].get("runtime_profile")
        )
        if runtime_profile and "runtime_profile" not in meta_with_id:
            meta_with_id["runtime_profile"] = runtime_profile

        # Limits
        limits = cfg.get("limits") if isinstance(cfg.get("limits"), dict) else {}
        engine_cfg = cfg.get("engine") if isinstance(cfg.get("engine"), dict) else {}
        runtime_cfg = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}

        max_cycles = (
            cfg.get("max_cycles") or limits.get("max_cycles") or engine_cfg.get("max_cycles") or runtime_cfg.get("max_cycles")
        )
        if max_cycles is not None and "max_cycles" not in meta_with_id:
            meta_with_id["max_cycles"] = max_cycles

        max_rounds = (
            cfg.get("max_rounds") or limits.get("max_rounds") or engine_cfg.get("max_rounds") or runtime_cfg.get("max_rounds")
        )
        if max_rounds is not None and "max_rounds" not in meta_with_id:
            meta_with_id["max_rounds"] = max_rounds

        max_minutes = (
            cfg.get("max_minutes") or limits.get("max_minutes") or engine_cfg.get("max_minutes") or runtime_cfg.get("max_minutes")
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

    # Shadow copy to queue root for compatibility with any older watchers.
    # Best-effort only: failure here should not prevent enqueue.
    try:
        job.save_to(LEGACY_QUEUE_DIR)
    except Exception as e:
        _log("create_job: failed to write shadow copy to LEGACY_QUEUE_DIR:", repr(e))

    # Optional legacy-pending mirror for older deployments that still watch BASE_DIR/pending.
    if os.getenv("ARA_MIRROR_TO_LEGACY_PENDING", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            job.save_to(LEGACY_PENDING_DIR, filename=f"{run_id}_job.json")
        except Exception:
            pass

    _log("Created job", run_id, "status=queued", "QUEUE_ROOT:", QUEUE_ROOT)
    _queue_log("Enqueued job", f"run_id={run_id}", "status=queued", f"pending_path={pending_path}", level=logging.INFO)

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
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return RunJob.from_dict(data)


def load_job_by_id(run_id: str) -> Optional[RunJob]:
    """
    Try to load a job by id from any status folder.
    Returns None if not found.

    Security note:
        run_id is treated as an identifier, not a path. Unsafe values are rejected.
    """
    run_id = str(run_id).strip()
    if not _is_safe_run_id(run_id):
        return None

    # Finished metadata uses "_job.json" (and sometimes "__job.json") and is separate from results.
    for candidate in (
        FINISHED_DIR / f"{run_id}_job.json",
        FINISHED_DIR / f"{run_id}__job.json",
        LEGACY_FINISHED_DIR / f"{run_id}_job.json",
        LEGACY_FINISHED_DIR / f"{run_id}__job.json",
    ):
        if not candidate.exists():
            continue
        try:
            return load_job(candidate)
        except Exception:
            # Do not return early; fall through to other candidates.
            continue

    # Other statuses: support "{run_id}_job.json", "{run_id}__job.json", and legacy "{run_id}.json"
    search_folders = [
        PENDING_DIR,
        LEGACY_QUEUE_DIR,
        LEGACY_PENDING_DIR,
        ACTIVE_DIR,
        LEGACY_ACTIVE_DIR,
        ERROR_DIR,
        LEGACY_ERROR_DIR,
    ]
    for folder in search_folders:
        # In ERROR_DIR, prefer *_job.json/__job.json over a payload {run_id}.json if present.
        if folder in (ERROR_DIR, LEGACY_ERROR_DIR):
            path = _find_finished_meta_in_folder(run_id, folder) or _find_meta_in_folder(run_id, folder)
        else:
            path = _find_meta_in_folder(run_id, folder)

        if path is not None and path.exists():
            try:
                return load_job(path)
            except Exception:
                continue
    return None


def _cleanup_queued_copies(run_id: str) -> None:
    """
    Remove any queued metadata copies for this run_id (canonical pending and legacy queue).
    This is safe to call multiple times.
    """
    for folder in (PENDING_DIR, LEGACY_QUEUE_DIR, LEGACY_PENDING_DIR):
        _safe_unlink(folder / f"{run_id}_job.json")
        _safe_unlink(folder / f"{run_id}__job.json")
        _safe_unlink(folder / f"{run_id}.json")


def move_job(run_id: str, from_status: str, to_status: str) -> Tuple[Optional[RunJob], Optional[Path]]:
    """
    Move job metadata between status folders.

    Returns (job, new_path) or (None, None) if job was not found
    in the expected source folder.
    """
    run_id = str(run_id).strip()
    if not _is_safe_run_id(run_id):
        return None, None

    norm_from = from_status.lower()
    norm_to = to_status.lower()

    # Locate source file robustly (supports legacy naming and legacy folders).
    src: Optional[Path] = None

    # Canonical expected path first
    expected_src = _job_path_for_status(run_id, norm_from)
    if expected_src.exists():
        src = expected_src
    else:
        candidate_folders: List[Path] = []
        if norm_from in ("queued", "pending"):
            candidate_folders = [PENDING_DIR, LEGACY_QUEUE_DIR, LEGACY_PENDING_DIR]
        elif norm_from in ("running", "active"):
            candidate_folders = [ACTIVE_DIR, LEGACY_ACTIVE_DIR]
        elif norm_from == "finished":
            candidate_folders = [FINISHED_DIR, LEGACY_FINISHED_DIR]
        elif norm_from == "error":
            candidate_folders = [ERROR_DIR, LEGACY_ERROR_DIR]

        for folder in candidate_folders:
            if norm_from == "finished":
                maybe = _find_finished_meta_in_folder(run_id, folder)
            elif norm_from == "error":
                # Prefer metadata variants over any payload {run_id}.json
                maybe = _find_finished_meta_in_folder(run_id, folder) or _find_meta_in_folder(run_id, folder)
            else:
                maybe = _find_meta_in_folder(run_id, folder)

            if maybe is not None and maybe.exists():
                src = maybe
                break

    if src is None or not src.exists():
        return None, None

    try:
        job = load_job(src)
    except Exception as e:
        _queue_log("move_job:", f"failed to load job run_id={run_id} from {src}: {repr(e)}", level=logging.WARNING)
        return None, None

    # Normalize destination status
    if norm_to in ("pending", "queued"):
        job.status = "queued"
    elif norm_to in ("running", "active"):
        job.status = "active"
    elif norm_to in ("finished", "error"):
        job.status = norm_to
    else:
        job.status = "queued"

    job.updated_at = time.time()

    dst_path = _job_path_for_status(run_id, job.status)

    # Write destination first, then clean up source (safer on crashes).
    # If the job is already in the destination path, update it in-place and do not unlink.
    try:
        same_path = src.resolve() == dst_path.resolve()
    except Exception:
        same_path = (src == dst_path)

    job.save_to(dst_path.parent, filename=dst_path.name)

    # For backward compatibility with older watchers, queued jobs may also be mirrored into
    # the queue root. This is best-effort and must not fail the move.
    if job.status == "queued":
        try:
            job.save_to(LEGACY_QUEUE_DIR, filename=f"{run_id}_job.json")
        except Exception:
            pass

    if same_path:
        # Still clean queued shadows when moving out of queued.
        if job.status != "queued":
            _cleanup_queued_copies(run_id)
            if job.status in ("finished", "error"):
                _cleanup_claim_lock(run_id)
        return job, dst_path

    # Remove source and any stale queued shadows.
    _safe_unlink(src)
    if job.status != "queued":
        _cleanup_queued_copies(run_id)
        if job.status in ("finished", "error"):
            _cleanup_claim_lock(run_id)

    return job, dst_path


def update_job_status(run_id: str, new_status: str) -> Optional[RunJob]:
    """
    Convenience helper to update a job status without knowing the old state.

    It searches all folders for run_id, then moves it to the requested status.
    Returns the updated job or None if not found.
    """
    run_id = str(run_id).strip()
    if not _is_safe_run_id(run_id):
        return None

    norm_new = _normalize_status(new_status)

    # Find existing job metadata in any folder.
    search: List[Tuple[Path, bool]] = [
        (FINISHED_DIR, True),
        (LEGACY_FINISHED_DIR, True),
        (ACTIVE_DIR, False),
        (LEGACY_ACTIVE_DIR, False),
        (ERROR_DIR, False),
        (LEGACY_ERROR_DIR, False),
        (PENDING_DIR, False),
        (LEGACY_PENDING_DIR, False),
        (LEGACY_QUEUE_DIR, False),
    ]

    src: Optional[Path] = None
    for folder, is_finished in search:
        if is_finished:
            candidate = _find_finished_meta_in_folder(run_id, folder)
            if candidate is not None and candidate.exists():
                src = candidate
                break
        else:
            # For error folders, prefer meta variants over payload JSON if present.
            if folder in (ERROR_DIR, LEGACY_ERROR_DIR):
                candidate = _find_finished_meta_in_folder(run_id, folder) or _find_meta_in_folder(run_id, folder)
            else:
                candidate = _find_meta_in_folder(run_id, folder)

            if candidate is not None and candidate.exists():
                src = candidate
                break

    if src is None or not src.exists():
        return None

    try:
        job = load_job(src)
    except Exception as e:
        _queue_log("update_job_status:", f"failed to load job run_id={run_id} from {src}: {repr(e)}", level=logging.WARNING)
        return None

    job.status = norm_new
    job.updated_at = time.time()

    dst_path = _job_path_for_status(run_id, norm_new)

    # Write destination first, then clean up source (safer on crashes).
    # If the job is already in the destination path, update it in-place and do not unlink.
    try:
        same_path = src.resolve() == dst_path.resolve()
    except Exception:
        same_path = (src == dst_path)

    job.save_to(dst_path.parent, filename=dst_path.name)

    # For backward compatibility with older watchers, queued jobs may also be mirrored into
    # the queue root. This is best-effort and must not fail the status update.
    if norm_new == "queued":
        try:
            job.save_to(LEGACY_QUEUE_DIR, filename=f"{run_id}_job.json")
        except Exception:
            pass

    if not same_path:
        _safe_unlink(src)

    # If moving out of queued, clean queued shadows.
    if norm_new != "queued":
        _cleanup_queued_copies(run_id)
    if norm_new in ("finished", "error"):
        _cleanup_claim_lock(run_id)

    return job


def list_jobs(status: Optional[str] = None, limit: int = 100) -> List[RunJob]:
    """
    List jobs by status for UI or debugging.

    status:
        None        : search all statuses
        queued      : jobs in pending (canonical queue status)
        pending     : alias for queued
        running     : alias for active
        active      : jobs in active
        finished    : jobs in finished (metadata files only)
        error       : jobs in error

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

        For pending and active it prefers "*_job.json" metadata files, but will also
        pick up legacy "<run_id>.json" jobs even when the folder contains a mix of
        both formats.

        For finished it only considers "*_job.json" metadata files (including "__job.json"),
        never the result payload "{run_id}.json".

    Resolution rule:
        If multiple metadata files for the same run_id exist across folders
        (for example due to a stale active copy and a newer finished copy),
        the record with the higher status priority wins:

            error > finished > active > queued

        If the priority is the same, the record with the newer updated_at
        timestamp wins.
    """
    jobs_by_id: Dict[str, RunJob] = {}

    def _should_skip_meta_file(name: str) -> bool:
        if name.startswith("."):
            return True
        # Skip progress/results artifacts
        if name.endswith("_progress.json") or name.endswith("_results.json") or name.endswith("_result.json"):
            return True
        # Skip obvious non-job JSON files some workers might write
        if name.endswith("_state.json") or name == "worker_state.json":
            return True
        return False

    def collect(folder: Path, folder_status: Optional[str] = None, finished: bool = False) -> None:
        if not folder.exists():
            return

        # For finished, do NOT read "<run_id>.json" (reserved for result payload).
        if finished:
            candidates: List[Path] = []
            try:
                candidates = sorted(folder.glob("*_job.json"))
            except Exception:
                candidates = []
        else:
            # Mixed-mode safe: read both new and legacy formats.
            seen: set = set()
            candidates = []
            for pattern in ("*_job.json", "*.json"):
                try:
                    for p in folder.glob(pattern):
                        key = str(p)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(p)
                except Exception:
                    continue
            try:
                candidates.sort()
            except Exception:
                pass

        for path in candidates:
            name = path.name
            if _should_skip_meta_file(name):
                continue
            if not path.is_file():
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
        # Search all status folders plus legacy queue and legacy layout dirs
        collect(PENDING_DIR, folder_status="queued")
        collect(LEGACY_QUEUE_DIR, folder_status="queued")
        collect(LEGACY_PENDING_DIR, folder_status="queued")
        collect(ACTIVE_DIR, folder_status="active")
        collect(LEGACY_ACTIVE_DIR, folder_status="active")
        collect(FINISHED_DIR, folder_status="finished", finished=True)
        collect(LEGACY_FINISHED_DIR, folder_status="finished", finished=True)
        collect(ERROR_DIR, folder_status="error")
        collect(LEGACY_ERROR_DIR, folder_status="error")
    else:
        norm = status.lower()
        if norm in ("pending", "queued"):
            collect(PENDING_DIR, folder_status="queued")
            collect(LEGACY_QUEUE_DIR, folder_status="queued")
            collect(LEGACY_PENDING_DIR, folder_status="queued")
        elif norm in ("running", "active"):
            collect(ACTIVE_DIR, folder_status="active")
            collect(LEGACY_ACTIVE_DIR, folder_status="active")
        elif norm == "finished":
            collect(FINISHED_DIR, folder_status="finished", finished=True)
            collect(LEGACY_FINISHED_DIR, folder_status="finished", finished=True)
        elif norm == "error":
            collect(ERROR_DIR, folder_status="error")
            collect(LEGACY_ERROR_DIR, folder_status="error")

    jobs: List[RunJob] = list(jobs_by_id.values())
    # Sort by created_at descending (newest first for UI)
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    try:
        lim = int(limit)
    except Exception:
        lim = 100

    if lim <= 0:
        return []
    return jobs[:lim]


def get_next_queued_job() -> Optional[RunJob]:
    """
    Simple helper for engine workers.

    Returns the oldest queued job (by created_at) from the queue folders,
    or None if there are no queued jobs.
    """
    queued = list_jobs(status="queued", limit=1000)
    if not queued:
        _log("get_next_queued_job: no queued jobs found")
        _queue_log("get_next_queued_job:", "no queued jobs found", level=logging.DEBUG)
        return None
    # list_jobs returns newest first; for FIFO we take the last (oldest)
    job = queued[-1]
    _log("get_next_queued_job: selected", job.run_id, "created_at", job.created_at)
    _queue_log(
        "get_next_queued_job:",
        f"selected run_id={job.run_id}",
        f"created_at={job.created_at}",
        f"queued_count={len(queued)}",
        level=logging.DEBUG,
    )
    return job


def _claim_lock_path(run_id: str) -> Path:
    return _LOCKS_DIR / f"{run_id}.claim.lock"


def _active_meta_exists(run_id: str) -> bool:
    return any(
        p.exists()
        for p in (
            ACTIVE_DIR / f"{run_id}_job.json",
            ACTIVE_DIR / f"{run_id}__job.json",
            ACTIVE_DIR / f"{run_id}.json",
            LEGACY_ACTIVE_DIR / f"{run_id}_job.json",
            LEGACY_ACTIVE_DIR / f"{run_id}__job.json",
            LEGACY_ACTIVE_DIR / f"{run_id}.json",
        )
    )


def _final_meta_exists(run_id: str) -> bool:
    return any(
        p.exists()
        for p in (
            FINISHED_DIR / f"{run_id}_job.json",
            FINISHED_DIR / f"{run_id}__job.json",
            ERROR_DIR / f"{run_id}_job.json",
            ERROR_DIR / f"{run_id}__job.json",
            LEGACY_FINISHED_DIR / f"{run_id}_job.json",
            LEGACY_FINISHED_DIR / f"{run_id}__job.json",
            LEGACY_ERROR_DIR / f"{run_id}_job.json",
            LEGACY_ERROR_DIR / f"{run_id}__job.json",
        )
    )


def _acquire_claim_lock(run_id: str) -> Optional[Path]:
    """
    Acquire an exclusive lock for claiming a job.

    Returns the lock path if acquired, otherwise None.
    Automatically breaks stale locks that are older than the configured timeout,
    but only when there is no corresponding active/finished/error metadata.
    """
    lock_path = _claim_lock_path(run_id)
    stale_seconds = _coerce_positive_int(os.getenv("ARA_CLAIM_LOCK_STALE_SECONDS", "").strip()) or 900

    # Fast path: try to create lock atomically.
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Possibly stale. Only break if job is not clearly active/finished/error.
        try:
            if _active_meta_exists(run_id) or _final_meta_exists(run_id):
                return None
        except Exception:
            return None

        try:
            age = time.time() - lock_path.stat().st_mtime
        except Exception:
            age = float("inf")

        if age < float(stale_seconds):
            return None

        # Best-effort stale lock cleanup.
        try:
            lock_path.unlink()
        except Exception:
            return None

        # Retry once.
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except Exception:
            return None
    except Exception:
        return None

    try:
        payload = {
            "run_id": run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "claimed_at": time.time(),
        }
        os.write(fd, json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8"))
    except Exception:
        # If we can't write, still hold the lock; it's fine.
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass

    return lock_path


def _release_claim_lock(lock_path: Optional[Path]) -> None:
    if lock_path is None:
        return
    _safe_unlink(lock_path)


def _claim_job_atomic(job: RunJob) -> Optional[RunJob]:
    """
    Atomically claim a specific job by creating ACTIVE metadata and removing queued copies.

    This uses a per-run_id exclusive lock to avoid races in multi-worker setups.
    """
    run_id = job.run_id

    # If already active/finished/error, clean stale queued copies and skip.
    try:
        if _active_meta_exists(run_id) or _final_meta_exists(run_id):
            _cleanup_queued_copies(run_id)
            return None
    except Exception:
        # If filesystem checks fail, be conservative and skip claiming.
        return None

    lock_path = _acquire_claim_lock(run_id)
    if lock_path is None:
        return None

    try:
        # Re-check once under lock.
        if _active_meta_exists(run_id) or _final_meta_exists(run_id):
            _cleanup_queued_copies(run_id)
            return None

        # Load the most current job metadata from any queued location if possible.
        src_candidates = [
            _find_meta_in_folder(run_id, PENDING_DIR),
            _find_meta_in_folder(run_id, LEGACY_PENDING_DIR),
            _find_meta_in_folder(run_id, LEGACY_QUEUE_DIR),
        ]
        src: Optional[Path] = None
        for cand in src_candidates:
            if cand is not None and cand.exists():
                src = cand
                break

        loaded = job
        if src is not None:
            try:
                loaded = load_job(src)
            except Exception:
                loaded = job

        loaded.run_id = run_id
        loaded.config = _inject_run_id_into_config(run_id, loaded.config if isinstance(loaded.config, dict) else {})
        loaded.status = "active"
        loaded.updated_at = time.time()

        # Always write canonical active metadata file name for stability.
        active_meta_path = ACTIVE_DIR / f"{run_id}_job.json"
        loaded.save_to(active_meta_path.parent, filename=active_meta_path.name)

        # Remove queued copies (including shadow copies).
        _cleanup_queued_copies(run_id)

        # Best-effort: create an initial progress file so UIs don't jump 0->100%.
        # (This is safe even if the worker overwrites progress later.)
        try:
            write_progress(
                run_id,
                {
                    "phase_index": 0,
                    "phase_total": 0,
                    "phase_name": "starting",
                    "notes": "claimed",
                },
            )
        except Exception:
            pass

        _log("claim_job_atomic: claimed", run_id, "at", active_meta_path)
        _queue_log("claim_job_atomic:", f"claimed run_id={run_id}", f"path={active_meta_path}", level=logging.INFO)
        return loaded
    except Exception as e:
        _log("claim_job_atomic: failed for", run_id, "error:", repr(e))
        _queue_log("claim_job_atomic:", f"failed for run_id={run_id}", f"error={repr(e)}", level=logging.WARNING)
        return None
    finally:
        _release_claim_lock(lock_path)


def claim_next_job() -> Optional[RunJob]:
    """
    Atomically claim the next queued job by marking it active.

    Engine workers can call this to get the next job and immediately
    move it to the active folder so that other workers do not pick it up.

    Implementation note:
        Uses an exclusive lock per run_id and creates an active metadata file
        (ACTIVE_DIR/<run_id>_job.json) to avoid multi-worker races, including
        the presence of legacy "shadow copies" in the queue root.
    """
    queued = list_jobs(status="queued", limit=2000)
    if not queued:
        _log("claim_next_job: no queued jobs")
        _queue_log("claim_next_job:", "no queued jobs", level=logging.DEBUG)
        return None

    # Oldest first
    queued.sort(key=lambda j: j.created_at)

    for job in queued:
        claimed = _claim_job_atomic(job)
        if claimed is not None:
            _log("claim_next_job: claimed job", claimed.run_id)
            _queue_log("claim_next_job:", f"claimed run_id={claimed.run_id}", level=logging.INFO)
            return claimed

    _log("claim_next_job: could not claim any queued job")
    _queue_log("claim_next_job:", "could not claim any queued job", level=logging.DEBUG)
    return None


def progress_path(run_id: str) -> Path:
    """
    Path where the worker should write live progress JSON for a run.

    Recommended schema (example):
        {
            "run_id": "...",
            "status": "active",
            "phase_index": 2,
            "phase_total": 3,
            "phase_name": "stabilization",
            "cycle": 214,
            "notes": "optional human readable progress",
            "updated_at": 1734481234.12
        }
    """
    safe_id = _sanitize_run_id_for_filename(run_id)
    return ACTIVE_DIR / f"{safe_id}_progress.json"


def write_progress(run_id: str, progress: Dict[str, Any]) -> Path:
    """
    Atomically write live progress JSON for a run.

    This is intentionally lightweight so engine_worker.py can call it before each phase/cycle.
    """
    run_id = str(run_id).strip()
    if not _is_safe_run_id(run_id):
        raise ValueError(f"write_progress called with unsafe run_id: {run_id!r}")

    payload: Dict[str, Any] = dict(progress or {})
    payload.setdefault("run_id", run_id)
    payload.setdefault("status", "active")
    payload.setdefault("updated_at", time.time())
    path = progress_path(run_id)
    _atomic_write_json(path, payload)
    return path


def read_progress(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Read a run's live progress JSON (active/<run_id>_progress.json).
    Returns None if missing/unreadable.
    """
    run_id = str(run_id).strip()
    if not _is_safe_run_id(run_id):
        return None
    path = progress_path(run_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def result_path(run_id: str) -> Path:
    """
    Path where the worker should write final result JSON for a run.

    This returns finished/{run_id}.json.

    Finished jobs now have two files:
        - finished/{run_id}_job.json   (job metadata, used by list_jobs)
        - finished/{run_id}.json       (final result payload, used by UI)

    Some workers also archive job state as finished/{run_id}__job.json; this
    module reads both variants for metadata.
    """
    safe_id = _sanitize_run_id_for_filename(run_id)
    return FINISHED_DIR / f"{safe_id}.json"


def error_log_path(run_id: str) -> Path:
    """
    Path where the worker can write an error message or traceback.

    Note:
        Some workers may write a structured error payload as error/{run_id}.json.
        This function remains a stable path for text logs: error/{run_id}_error.txt.
    """
    safe_id = _sanitize_run_id_for_filename(run_id)
    return ERROR_DIR / f"{safe_id}_error.txt"


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
        _queue_log("load_next_pending_job:", "no job claimed", level=logging.DEBUG)
    else:
        _log("load_next_pending_job: returning job", job.run_id)
        _queue_log("load_next_pending_job:", f"returning run_id={job.run_id}", level=logging.INFO)
        # Best-effort: update worker state so the UI shows "running" immediately.
        try:
            update_worker_state(
                {
                    "status": "running",
                    "run_id": job.run_id,
                    "heartbeat_ts": time.time(),
                    "phase_index": 0,
                    "phase_total": 0,
                    "phase_name": "",
                }
            )
        except Exception:
            pass
    return job


def _cleanup_active_artifacts(run_id: str) -> None:
    """
    Remove active artifacts for a run (metadata variants + progress).
    """
    for p in (
        ACTIVE_DIR / f"{run_id}_job.json",
        ACTIVE_DIR / f"{run_id}__job.json",
        ACTIVE_DIR / f"{run_id}.json",
        ACTIVE_DIR / f"{run_id}_progress.json",
        LEGACY_ACTIVE_DIR / f"{run_id}_job.json",
        LEGACY_ACTIVE_DIR / f"{run_id}__job.json",
        LEGACY_ACTIVE_DIR / f"{run_id}.json",
        LEGACY_ACTIVE_DIR / f"{run_id}_progress.json",
    ):
        _safe_unlink(p)


def _cleanup_claim_lock(run_id: str) -> None:
    _safe_unlink(_claim_lock_path(run_id))


def save_job_result(job: RunJob, result_obj: Dict[str, Any]) -> None:
    """
    Write final result JSON for a job and mark it finished.

    Used by engine_worker queue mode (legacy path). Newer workers may manage
    retries/finalization themselves; this helper remains stable and compatible.
    """
    run_id = str(job.run_id).strip()
    if not _is_safe_run_id(run_id):
        raise ValueError(f"save_job_result called with unsafe run_id: {run_id!r}")

    # Ensure identifiers are present in the result for UI convenience
    if "job_id" not in result_obj:
        result_obj["job_id"] = run_id
    if "run_id" not in result_obj:
        result_obj["run_id"] = run_id

    rp = result_path(run_id)
    _atomic_write_json(rp, result_obj)

    # Optional compatibility alias for older readers.
    try:
        rp_alias = FINISHED_DIR / f"{run_id}_results.json"
        if rp_alias != rp:
            _atomic_write_json(rp_alias, result_obj)
    except Exception:
        pass

    # Produce finished metadata from active metadata if available, else from provided job.
    finished_job = job
    active_meta_candidates = [
        ACTIVE_DIR / f"{run_id}_job.json",
        ACTIVE_DIR / f"{run_id}__job.json",
        ACTIVE_DIR / f"{run_id}.json",  # legacy / alt
        LEGACY_ACTIVE_DIR / f"{run_id}_job.json",
        LEGACY_ACTIVE_DIR / f"{run_id}__job.json",
        LEGACY_ACTIVE_DIR / f"{run_id}.json",
    ]
    for active_meta in active_meta_candidates:
        if not active_meta.exists():
            continue
        try:
            finished_job = load_job(active_meta)
            break
        except Exception:
            continue

    finished_job.run_id = run_id
    finished_job.config = _inject_run_id_into_config(run_id, finished_job.config if isinstance(finished_job.config, dict) else {})
    finished_job.status = "finished"
    finished_job.updated_at = time.time()

    # Write finished metadata (canonical), then clean up active/queued artifacts.
    finished_job.save_to(FINISHED_DIR, filename=f"{run_id}_job.json")
    _cleanup_active_artifacts(run_id)
    _cleanup_queued_copies(run_id)
    _cleanup_claim_lock(run_id)

    _log("save_job_result: marked finished", run_id)
    _queue_log("save_job_result:", f"marked finished run_id={run_id}", f"result_path={rp}", level=logging.INFO)

    # Best-effort: update worker state for live UI.
    try:
        update_worker_state(
            {
                "status": "finished",
                "run_id": run_id,
                "heartbeat_ts": time.time(),
            }
        )
    except Exception:
        pass


def mark_job_error(job: RunJob, error_info: Dict[str, Any]) -> None:
    """
    Write error information for a job and mark it as error.

    error_info can be a string or a dict.
    """
    run_id = str(job.run_id).strip()
    if not _is_safe_run_id(run_id):
        raise ValueError(f"mark_job_error called with unsafe run_id: {run_id!r}")

    ep = error_log_path(run_id)

    try:
        if isinstance(error_info, str):
            _atomic_write_text(ep, error_info)
        else:
            # Preserve legacy behavior: write structured JSON into the ".txt" file.
            text_payload = json.dumps(error_info, indent=2, ensure_ascii=False, default=_json_default)
            _atomic_write_text(ep, text_payload)
    except Exception:
        # Fallback: best-effort text with traceback.
        tb = traceback.format_exc()
        try:
            _atomic_write_text(ep, f"{error_info}\n\n{tb}")
        except Exception:
            pass

    # Optional structured error payload for UIs that prefer JSON.
    #
    # IMPORTANT:
    #   Some older deployments used error/<run_id>.json as *metadata* (with a "config" field).
    #   To avoid clobbering that, we only write/overwrite this payload if the existing file
    #   does not look like job metadata.
    try:
        payload: Dict[str, Any]
        if isinstance(error_info, dict):
            payload = dict(error_info)
        else:
            payload = {"message": str(error_info)}
        payload.setdefault("run_id", run_id)
        payload.setdefault("status", "error")
        payload.setdefault("ts", time.time())

        payload_path = ERROR_DIR / f"{run_id}.json"

        write_payload = True
        if payload_path.exists():
            try:
                with payload_path.open("r", encoding="utf-8") as f:
                    existing_payload = json.load(f)
                if isinstance(existing_payload, dict) and "config" in existing_payload:
                    # Looks like legacy job metadata; do not overwrite.
                    write_payload = False
            except Exception:
                # If unreadable, it's probably not intentional metadata; allow overwrite.
                write_payload = True

        if write_payload:
            _atomic_write_json(payload_path, payload)
    except Exception:
        pass

    # Produce error metadata from active metadata if available, else from provided job.
    error_job = job
    active_meta_candidates = [
        ACTIVE_DIR / f"{run_id}_job.json",
        ACTIVE_DIR / f"{run_id}__job.json",
        ACTIVE_DIR / f"{run_id}.json",  # legacy / alt
        LEGACY_ACTIVE_DIR / f"{run_id}_job.json",
        LEGACY_ACTIVE_DIR / f"{run_id}__job.json",
        LEGACY_ACTIVE_DIR / f"{run_id}.json",
    ]
    for active_meta in active_meta_candidates:
        if not active_meta.exists():
            continue
        try:
            error_job = load_job(active_meta)
            break
        except Exception:
            continue

    error_job.run_id = run_id
    error_job.config = _inject_run_id_into_config(run_id, error_job.config if isinstance(error_job.config, dict) else {})
    error_job.status = "error"
    error_job.updated_at = time.time()

    error_job.save_to(ERROR_DIR, filename=f"{run_id}_job.json")
    _cleanup_active_artifacts(run_id)
    _cleanup_queued_copies(run_id)
    _cleanup_claim_lock(run_id)

    _log("mark_job_error: marked error", run_id)
    _queue_log("mark_job_error:", f"marked error run_id={run_id}", f"error_path={ep}", level=logging.INFO)

    # Best-effort: update worker state for live UI.
    try:
        update_worker_state(
            {
                "status": "fault",
                "run_id": run_id,
                "heartbeat_ts": time.time(),
            }
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Optional helpers for UI and debugging
# ---------------------------------------------------------------------------


def load_job_result(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience helper: load the final result JSON for a finished job.

    Accepts either finished/<run_id>.json or finished/<run_id>_results.json
    as the result payload file. Also supports legacy finished/ layout.
    """
    run_id = str(run_id).strip()
    if not _is_safe_run_id(run_id):
        return None

    rp_main = FINISHED_DIR / f"{run_id}.json"
    rp_alt = FINISHED_DIR / f"{run_id}_results.json"
    rp_legacy_main = LEGACY_FINISHED_DIR / f"{run_id}.json"
    rp_legacy_alt = LEGACY_FINISHED_DIR / f"{run_id}_results.json"

    rp: Optional[Path] = None
    for candidate in (rp_main, rp_alt, rp_legacy_main, rp_legacy_alt):
        if candidate.exists():
            rp = candidate
            break

    if rp is None or not rp.exists():
        return None
    try:
        with rp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
