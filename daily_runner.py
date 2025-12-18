"""
Daily runner for the Autonomous Research Agent.

This script is designed to be triggered once per day by an external scheduler
(cron, GitHub Actions, Render cron, etc.) to:

- Run a small number of cycles on a "daily update" goal (direct mode), OR
  enqueue a job for the engine worker (queue mode)
- Use the same memory file config as the Streamlit app (config/settings.yaml)
- Respect presets (general, longevity, math)
- Print a human readable daily summary with RYE stats

Defaults:
    python daily_runner.py
        -> direct mode (runs cycles in-process), cycles=3, preset=longevity

Queue mode (recommended if you already run an engine worker):
    ARA_DAILY_USE_QUEUE=1 python daily_runner.py
    python daily_runner.py --mode queue --wait

Examples:
    python daily_runner.py --preset longevity --cycles 5
    python daily_runner.py --preset math --cycles 2 --goal-prefix "Daily math scan"
    python daily_runner.py --mode queue --wait --timeout-seconds 1800
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Repo root + imports (supports both package layout and flat fallback)
# -----------------------------------------------------------------------------
_THIS_FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_FILE_DIR
if not (REPO_ROOT / "agent").is_dir() and (_THIS_FILE_DIR.parent / "agent").is_dir():
    REPO_ROOT = _THIS_FILE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _safe_imports() -> Tuple[Any, Any, Any, Any]:
    """
    Return (MemoryStore, CoreAgent, get_preset, run_jobs_module_or_None).

    - MemoryStore/CoreAgent/get_preset may be imported from agent.* or flat fallback.
    - run_jobs is optional (only needed for queue mode).
    """
    run_jobs_mod = None
    try:
        from agent.memory_store import MemoryStore  # type: ignore
        from agent.core_agent import CoreAgent  # type: ignore
        from agent.presets import get_preset  # type: ignore

        try:
            from agent import run_jobs as run_jobs_mod  # type: ignore
        except Exception:
            run_jobs_mod = None

        return MemoryStore, CoreAgent, get_preset, run_jobs_mod
    except ModuleNotFoundError as e:
        missing_name = getattr(e, "name", None)
        if missing_name not in (None, "agent"):
            raise

    # Flat layout fallback
    from memory_store import MemoryStore  # type: ignore
    from core_agent import CoreAgent  # type: ignore
    from presets import get_preset  # type: ignore

    try:
        import run_jobs as run_jobs_mod  # type: ignore
    except Exception:
        run_jobs_mod = None

    return MemoryStore, CoreAgent, get_preset, run_jobs_mod


MemoryStore, CoreAgent, get_preset, _run_jobs = _safe_imports()


# -----------------------------------------------------------------------------
# Settings / paths (match Streamlit behavior)
# -----------------------------------------------------------------------------
CONFIG_PATH_DEFAULT = str(REPO_ROOT / "config" / "settings.yaml")


def _load_settings_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load YAML settings from config/settings.yaml, returning a dict.
    If yaml isn't installed or the file doesn't exist, returns {}.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_memory_path(config_path: str = CONFIG_PATH_DEFAULT) -> str:
    """
    Use the same memory file as the main app, with a safe fallback.

    Priority:
      1) ARA_MEMORY_FILE env var (explicit override)
      2) settings.yaml: memory_file
      3) default: logs/sessions/default_memory.json (relative to REPO_ROOT)
    """
    env_mem = os.getenv("ARA_MEMORY_FILE", "").strip()
    if env_mem:
        return env_mem

    cfg = _load_settings_yaml(config_path)
    mem = cfg.get("memory_file") or cfg.get("memory_path")
    if not mem:
        mem = "logs/sessions/default_memory.json"

    mem_str = str(mem)
    if os.path.isabs(mem_str):
        return mem_str
    return str(REPO_ROOT / mem_str)


def ensure_directories_for_file(file_path: str) -> None:
    """
    Ensure parent directories exist for a file path, plus logs/sessions defaults.
    """
    try:
        fp = Path(file_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / "logs" / "sessions").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Goal + summarization helpers
# -----------------------------------------------------------------------------
def build_daily_goal(prefix: str, domain_label: str, explicit_date_utc: Optional[str] = None) -> str:
    """
    Construct the daily goal string.
    """
    today = explicit_date_utc or datetime.utcnow().strftime("%Y-%m-%d")
    return (
        f"{prefix} for {today}: summarize new or important findings in {domain_label}, "
        "repair and resilience, and update long run hypotheses."
    )


def _extract_rye(summary: Dict[str, Any]) -> Optional[float]:
    v = summary.get("RYE")
    if v is None:
        v = summary.get("rye")
    try:
        return float(v) if isinstance(v, (int, float)) else None
    except Exception:
        return None


def _extract_issues(summary: Dict[str, Any]) -> List[str]:
    issues = summary.get("issues_before") or summary.get("issues") or []
    if not isinstance(issues, list):
        return []
    out: List[str] = []
    for it in issues:
        s = str(it).strip()
        if s:
            out.append(s)
    return out


def summarize_day(summaries: List[Dict[str, Any]]) -> None:
    """
    Print a clear, lay interpretation of how the day went.
    """
    if not summaries:
        print("No cycles were run today.")
        return

    rye_vals: List[float] = []
    issues_total: int = 0
    issues_samples: List[str] = []

    for s in summaries:
        v = _extract_rye(s)
        if v is not None:
            rye_vals.append(v)

        issues = _extract_issues(s)
        issues_total += len(issues)
        for item in issues:
            if len(issues_samples) < 5:
                issues_samples.append(item)

    print("\nDaily summary (human readable):")
    print(f"- Cycles run: {len(summaries)}")

    if rye_vals:
        avg_rye = sum(rye_vals) / len(rye_vals)
        print(f"- RYE range: {min(rye_vals):.3f} to {max(rye_vals):.3f}")
        print(f"- RYE average: {avg_rye:.3f}")

        if avg_rye > 0.5:
            print("- Interpretation: Very productive day. The agent is finding useful repairs.")
        elif avg_rye > 0.1:
            print("- Interpretation: Mixed but net positive. Some useful repairs and updates.")
        elif avg_rye > 0:
            print("- Interpretation: Small gains. The agent is working, but improvements are modest.")
        else:
            print("- Interpretation: No clear net repair today. Consider adjusting goals/sources.")
    else:
        print("- RYE values were not available in these summaries.")

    print(f"- Total issues detected before repair across cycles: {issues_total}")
    if issues_samples:
        print("- Example issues the agent tried to repair today:")
        for ex in issues_samples:
            print(f"  • {ex}")


def _next_cycle_index_from_history(history: List[Dict[str, Any]]) -> int:
    """
    Robust next cycle index that works for both 0-based and 1-based histories.

    Uses max(cycle or cycle_index) + 1 when available; otherwise len(history) + 1.
    """
    if not history:
        return 1

    max_cycle: Optional[int] = None
    for e in history:
        if not isinstance(e, dict):
            continue
        val = e.get("cycle")
        if val is None:
            val = e.get("cycle_index")
        try:
            c = int(val)
        except Exception:
            continue
        if max_cycle is None or c > max_cycle:
            max_cycle = c

    if max_cycle is None:
        return len(history) + 1
    return max_cycle + 1


# -----------------------------------------------------------------------------
# Queue-mode helpers (optional)
# -----------------------------------------------------------------------------
def _queue_available() -> bool:
    return _run_jobs is not None and callable(getattr(_run_jobs, "create_job", None))


def _truthy_env(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _extract_cycle_summaries_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Best-effort extract per-cycle summaries from a finished result JSON.

    Supports common shapes:
      - {"cycles":[{...}...]}
      - {"result":{"cycle_history":[{...}...]}}
      - cycle entries containing "summary" dicts
    """
    if not isinstance(result, dict):
        return []

    base = result.get("result") if isinstance(result.get("result"), dict) else result

    cycles: Optional[List[Any]] = None
    for key in ("cycles", "cycle_history", "history", "tgrm_history", "run_history", "per_cycle"):
        v = base.get(key)
        if isinstance(v, list) and v:
            cycles = v
            break

    if not cycles:
        return []

    out: List[Dict[str, Any]] = []
    for idx, c in enumerate(cycles):
        if not isinstance(c, dict):
            continue
        # If the cycle object embeds a summary dict, prefer that.
        s = c.get("summary")
        if isinstance(s, dict):
            s2 = dict(s)
            s2.setdefault("cycle", c.get("cycle") or c.get("cycle_index") or (idx + 1))
            out.append(s2)
        else:
            c2 = dict(c)
            c2.setdefault("cycle", c2.get("cycle") or c2.get("cycle_index") or (idx + 1))
            out.append(c2)

    return out


def _wait_for_job_completion(
    run_id: str,
    timeout_seconds: int,
    poll_seconds: float,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Wait for a queued run_id to complete by watching run_jobs artifacts.

    Returns:
      ("finished", result_dict) or ("error", result_or_none) or ("timeout", None)
    """
    assert _run_jobs is not None

    load_job_result = getattr(_run_jobs, "load_job_result", None)
    load_job_by_id = getattr(_run_jobs, "load_job_by_id", None)

    start = time.time()
    last_status: Optional[str] = None

    while True:
        elapsed = time.time() - start
        if elapsed > float(timeout_seconds):
            return "timeout", None

        # If run_jobs has load_job_by_id, use it to detect error/finished statuses quickly.
        job_obj = None
        if callable(load_job_by_id):
            try:
                job_obj = load_job_by_id(run_id)
            except Exception:
                job_obj = None

        if job_obj is not None:
            st = getattr(job_obj, "status", None)
            if isinstance(st, str) and st != last_status:
                last_status = st
                print(f"[wait] status now: {st}")

            if st == "error":
                # Try to also load result payload if present
                res = None
                if callable(load_job_result):
                    try:
                        res = load_job_result(run_id)
                    except Exception:
                        res = None
                return "error", res

            if st == "finished":
                res = None
                if callable(load_job_result):
                    try:
                        res = load_job_result(run_id)
                    except Exception:
                        res = None
                return "finished", res

        # Fallback: check for result file existence
        rp_fn = getattr(_run_jobs, "result_path", None)
        if callable(rp_fn):
            try:
                rp = rp_fn(run_id)
                if isinstance(rp, Path) and rp.exists():
                    try:
                        data = json.loads(rp.read_text(encoding="utf-8"))
                        return "finished", data if isinstance(data, dict) else None
                    except Exception:
                        return "finished", None
            except Exception:
                pass

        time.sleep(max(0.2, float(poll_seconds)))


# -----------------------------------------------------------------------------
# Direct-mode runner
# -----------------------------------------------------------------------------
def _run_direct(
    preset_key: str,
    cycles: int,
    goal_prefix: str,
    goal_override: Optional[str],
    config_path: str,
) -> int:
    """
    Run cycles in-process using CoreAgent and MemoryStore.

    Returns exit code (0 success, nonzero on failure).
    """
    preset = get_preset(preset_key)
    if not isinstance(preset, dict):
        preset = {}

    domain_tag = preset.get("domain", preset_key)
    domain_label = preset.get("label", domain_tag)

    source_controls = preset.get("source_controls", {})
    if not isinstance(source_controls, dict):
        source_controls = {}

    goal = (goal_override or "").strip() or build_daily_goal(prefix=goal_prefix, domain_label=str(domain_label))

    memory_path = resolve_memory_path(config_path=config_path)
    ensure_directories_for_file(memory_path)

    memory = MemoryStore(memory_path)
    agent = CoreAgent(memory_store=memory, config={})

    try:
        history = memory.get_cycle_history() or []
    except Exception:
        history = []

    start_idx = _next_cycle_index_from_history(history)

    print("==============================================")
    print("Autonomous Research Agent - Daily Runner (DIRECT)")
    print(f"Date (UTC): {datetime.utcnow().isoformat(timespec='seconds')}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Config: {config_path}")
    print(f"Memory file: {memory_path}")
    print(f"Preset: {preset_key} (domain: {domain_tag})")
    print(f"Daily goal: {goal}")
    print(f"Planned cycles: {cycles}")
    print(f"Starting cycle index: {start_idx}")
    print("==============================================\n")

    summaries: List[Dict[str, Any]] = []
    for i in range(cycles):
        cycle_index = start_idx + i

        # Call run_cycle in a signature-tolerant way
        try:
            fn = getattr(agent, "run_cycle")
        except Exception:
            print("ERROR: CoreAgent has no run_cycle method in this build.")
            return 3

        kwargs: Dict[str, Any] = {
            "goal": goal,
            "cycle_index": cycle_index,
            "role": "agent",
            "source_controls": source_controls,
            "pdf_bytes": None,
            "biomarker_snapshot": None,
            "domain": domain_tag,
        }

        try:
            sig = inspect.signature(fn)
            # Drop kwargs that this build doesn't accept
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            # If we can't inspect, just try the common signature
            pass

        try:
            result = fn(**kwargs)  # type: ignore[misc]
        except TypeError:
            # Fallback: positional call matching the original script
            try:
                result = fn(goal, cycle_index, "agent", source_controls, None, None, domain_tag)  # type: ignore[misc]
            except Exception as e:
                print(f"ERROR: run_cycle failed on cycle {cycle_index}: {e}")
                return 4
        except Exception as e:
            print(f"ERROR: run_cycle failed on cycle {cycle_index}: {e}")
            return 4

        # Normalize summary
        summary = None
        if isinstance(result, dict):
            summary = result.get("summary")
            if not isinstance(summary, dict):
                # Some builds may return summary directly
                summary = result if any(k in result for k in ("RYE", "rye", "delta_R", "delta_r", "issues_before")) else None

        if not isinstance(summary, dict):
            summary = {}

        summaries.append(summary)
        rye = _extract_rye(summary)
        rye_text = f"{rye:.3f}" if isinstance(rye, (int, float)) else "n/a"
        print(f"Cycle {cycle_index} completed. RYE: {rye_text}")

    print("\nDaily run complete. Raw summaries (RYE, issues_before):")
    for s in summaries:
        print("-", _extract_rye(s), _extract_issues(s))

    summarize_day(summaries)
    return 0


# -----------------------------------------------------------------------------
# Queue-mode runner
# -----------------------------------------------------------------------------
def _run_queue(
    preset_key: str,
    cycles: int,
    goal_prefix: str,
    goal_override: Optional[str],
    wait: bool,
    timeout_seconds: int,
    poll_seconds: float,
) -> int:
    """
    Enqueue a finite run via run_jobs.create_job. Optionally wait for completion.
    """
    if not _queue_available():
        print("ERROR: Queue mode requested but agent.run_jobs is not available/importable.")
        return 10

    assert _run_jobs is not None
    create_job = getattr(_run_jobs, "create_job", None)
    if not callable(create_job):
        print("ERROR: agent.run_jobs.create_job not callable.")
        return 11

    preset = get_preset(preset_key)
    if not isinstance(preset, dict):
        preset = {}

    domain_tag = preset.get("domain", preset_key)
    domain_label = preset.get("label", domain_tag)

    source_controls = preset.get("source_controls", {})
    if not isinstance(source_controls, dict):
        source_controls = {}

    goal = (goal_override or "").strip() or build_daily_goal(prefix=goal_prefix, domain_label=str(domain_label))

    today = datetime.utcnow().strftime("%Y-%m-%d")
    requested_run_id = f"daily-{preset_key}-{today}"

    # Minimal finite run_config consistent with your Streamlit/worker setup
    run_config: Dict[str, Any] = {
        "goal": goal,
        "domain": domain_tag,
        "mode": "single",
        "total_cycles": int(cycles),
        "max_cycles": int(cycles),
        "max_rounds": None,
        "max_minutes": None,
        "runtime_hints": {
            "run_mode": "finite_daily",
            "manual_cycles": int(cycles),
            "max_cycles": int(cycles),
        },
        "source_controls": {k: bool(v) for k, v in source_controls.items()},
        # Monitoring defaults (run_jobs will ensure keys exist even if omitted)
        "monitoring": {
            "snapshots_enabled": False,
            "heartbeat_enabled": True,
            "run_state_enabled": True,
        },
        "notes": f"daily_update {today}",
    }

    meta: Dict[str, Any] = {
        "run_label": f"daily_update_{preset_key}_{today}",
        "preset_key": preset_key,
        "domain": domain_tag,
        "mode": "single",
        "requested_from": "daily_runner",
        "scheduled_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    print("==============================================")
    print("Autonomous Research Agent - Daily Runner (QUEUE)")
    print(f"Date (UTC): {datetime.utcnow().isoformat(timespec='seconds')}")
    print(f"Repo root: {REPO_ROOT}")
    print("run_jobs module file:", getattr(_run_jobs, "__file__", "(unknown)"))
    print(f"Preset: {preset_key} (domain: {domain_tag})")
    print(f"Daily goal: {goal}")
    print(f"Planned cycles: {cycles}")
    print(f"Requested run_id: {requested_run_id}")
    print("==============================================\n")

    try:
        run_id = create_job(config=run_config, meta=meta, run_id=requested_run_id)
    except TypeError:
        # Older create_job may not accept run_id kw
        run_id = create_job(config=run_config, meta=meta)

    run_id = str(run_id)
    print(f"[enqueue] Created job run_id={run_id}")

    pending_dir = getattr(_run_jobs, "PENDING_DIR", None)
    if isinstance(pending_dir, Path):
        job_fp = pending_dir / f"{run_id}_job.json"
        print(f"[enqueue] Pending job file (expected): {job_fp}")
    else:
        print("[enqueue] Pending job file: (pending dir unknown in this build)")

    if not wait:
        print("[enqueue] --wait disabled; exiting after enqueue.")
        return 0

    print(f"[wait] Waiting up to {timeout_seconds}s for completion (poll {poll_seconds}s)...")
    status, result = _wait_for_job_completion(run_id, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)

    if status == "timeout":
        print("[wait] TIMEOUT: job did not finish within the timeout window.")
        return 20

    if status == "error":
        print("[wait] ERROR: job ended in error status.")
        if isinstance(result, dict):
            print("[wait] Partial result payload:")
            try:
                print(json.dumps(result, indent=2)[:2000])
            except Exception:
                print(str(result)[:2000])
        return 21

    # finished
    print("[wait] Job finished.")

    # Try to extract per-cycle summaries for daily summary
    summaries: List[Dict[str, Any]] = []
    if isinstance(result, dict):
        summaries = _extract_cycle_summaries_from_result(result)

    # If we couldn't find cycles, try run-level rye_metrics avg_rye as a proxy
    if not summaries and isinstance(result, dict):
        base = result.get("result") if isinstance(result.get("result"), dict) else result
        rye_metrics = base.get("rye_metrics") or base.get("rye") or base.get("metrics") or {}
        if isinstance(rye_metrics, dict):
            avg = rye_metrics.get("avg_rye") or rye_metrics.get("rye_avg")
            try:
                avg_f = float(avg)
            except Exception:
                avg_f = None
            if avg_f is not None:
                summaries = [{"RYE": avg_f, "issues_before": []}]

    if isinstance(result, dict):
        base = result.get("result") if isinstance(result.get("result"), dict) else result
        run_summary = base.get("summary") or base.get("human_summary") or base.get("run_summary")
        if run_summary:
            print("\nRun summary:")
            print(str(run_summary)[:2000])

    summarize_day(summaries)
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Daily runner for the Autonomous Research Agent")

    parser.add_argument(
        "--preset",
        type=str,
        default="longevity",
        choices=["general", "longevity", "math"],
        help="Domain preset to use for the daily scan.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of cycles to run for the daily update.",
    )
    parser.add_argument(
        "--goal-prefix",
        type=str,
        default="Daily science update",
        help="Text prefix for the daily goal string.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="",
        help="Optional full override for the daily goal (if provided, goal-prefix is ignored).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH_DEFAULT,
        help="Path to settings.yaml (used to resolve memory_file).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "direct", "queue"],
        help="Execution mode. auto defaults to direct unless ARA_DAILY_USE_QUEUE=1 and queue is available.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Queue mode only: wait for the job to complete and print a daily summary from results.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Queue mode only: max seconds to wait for completion when --wait is set.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=3.0,
        help="Queue mode only: polling interval while waiting.",
    )

    args = parser.parse_args()

    # Normalize cycles
    try:
        cycles = int(args.cycles)
    except Exception:
        cycles = 3
    if cycles <= 0:
        print("ERROR: --cycles must be a positive integer.")
        raise SystemExit(2)

    goal_override = args.goal.strip() or None

    use_queue_env = _truthy_env("ARA_DAILY_USE_QUEUE")

    mode = args.mode
    if mode == "auto":
        # Default behavior stays compatible with the original script:
        # direct unless the operator explicitly opts into queue mode.
        if use_queue_env and _queue_available():
            mode = "queue"
        else:
            mode = "direct"

    if mode == "queue":
        # If --wait isn't provided, do not wait by default in queue mode unless operator asks.
        wait = bool(args.wait)
        code = _run_queue(
            preset_key=args.preset,
            cycles=cycles,
            goal_prefix=args.goal_prefix,
            goal_override=goal_override,
            wait=wait,
            timeout_seconds=int(args.timeout_seconds),
            poll_seconds=float(args.poll_seconds),
        )
        raise SystemExit(code)

    # direct
    code = _run_direct(
        preset_key=args.preset,
        cycles=cycles,
        goal_prefix=args.goal_prefix,
        goal_override=goal_override,
        config_path=args.config,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
