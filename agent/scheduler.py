"""
scheduler.py

High level scheduler for the Autonomous Research Agent.

Goal
    Coordinate long run maintenance and deep analysis tasks so that:
      - verification passes run regularly on promising hypotheses
      - discovery engine gets periodic focused passes
      - clustering and equilibrium snapshots are updated
      - memory pruning runs in controlled batches
      - everything is logged in a single scheduler state file

Design
    The Scheduler:
      - is stateless between processes except for a small JSON file
      - works with any loop that calls it once per cycle
      - uses both "cycles since" and "minutes since" triggers
      - never calls tools directly, only user provided callbacks

It is intended to be used from engine_worker.py, for example:

    from agent.scheduler import Scheduler, SchedulerCallbacks

    callbacks = SchedulerCallbacks(
        run_verification_batch=run_verification_batch,
        run_discovery_pass=run_discovery_pass,
        run_clustering_pass=run_clustering_pass,
        run_snapshot_pass=run_snapshot_pass,
        run_memory_pruning=run_memory_pruning,
    )

    scheduler = Scheduler(
        run_id=current_run_id,
        config=settings.get("scheduler", {}),
        callbacks=callbacks,
    )

    for cycle_index in loop:
        summary = scheduler.on_cycle_complete(
            cycle_index=cycle_index,
            rye=current_rye,
            domain=current_domain,
        )
        # You can append summary into logs or ignore it.

Scheduler state is stored in:
    logs/scheduler_state.json

This keeps it decoupled from MemoryStore, but very easy to inspect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


SCHEDULER_STATE_PATH = Path("logs/scheduler_state.json")
SCHEDULER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass
class TaskConfig:
    name: str
    every_cycles: Optional[int] = None
    every_minutes: Optional[float] = None
    batch_limit: Optional[int] = None
    enabled: bool = True


@dataclass
class TaskState:
    last_cycle: int = 0
    last_timestamp: Optional[str] = None


@dataclass
class SchedulerState:
    run_id: Optional[str]
    created_at: str
    updated_at: str
    last_cycle_index: int
    tasks: Dict[str, TaskState]


@dataclass
class SchedulerCallbacks:
    """
    Optional callbacks for the scheduler.

    Each callback should either:
      - return a dict summary, or
      - return None

    All callbacks are wrapped in safe error handling.
    """

    run_verification_batch: Optional[Callable[[int], Any]] = None
    run_discovery_pass: Optional[Callable[[], Any]] = None
    run_clustering_pass: Optional[Callable[[], Any]] = None
    run_snapshot_pass: Optional[Callable[[], Any]] = None
    run_memory_pruning: Optional[Callable[[], Any]] = None


class Scheduler:
    """
    Scheduler for long run maintenance tasks.

    It is agnostic to the core agent and tools and only coordinates when
    different subsystems are invoked.

    Task keys:
      - "verification"
      - "discovery"
      - "clustering"
      - "snapshot"
      - "pruning"

    You control frequency through a config dict:

        scheduler:
          verification:
            every_cycles: 50
            every_minutes: 120
            batch_limit: 5
            enabled: true
          discovery:
            every_cycles: 40
          clustering:
            every_cycles: 200
          snapshot:
            every_cycles: 100
          pruning:
            every_cycles: 500
            every_minutes: 1440

    If no scheduler config is given, sensible defaults are used.
    """

    DEFAULT_CONFIG: Dict[str, TaskConfig] = {
        "verification": TaskConfig(
            name="verification",
            every_cycles=40,
            every_minutes=180.0,
            batch_limit=5,
            enabled=True,
        ),
        "discovery": TaskConfig(
            name="discovery",
            every_cycles=50,
            every_minutes=240.0,
            enabled=True,
        ),
        "clustering": TaskConfig(
            name="clustering",
            every_cycles=200,
            every_minutes=720.0,
            enabled=True,
        ),
        "snapshot": TaskConfig(
            name="snapshot",
            every_cycles=100,
            every_minutes=360.0,
            enabled=True,
        ),
        "pruning": TaskConfig(
            name="pruning",
            every_cycles=600,
            every_minutes=1440.0,
            enabled=True,
        ),
    }

    def __init__(
        self,
        run_id: Optional[str],
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[SchedulerCallbacks] = None,
        state_path: Path = SCHEDULER_STATE_PATH,
    ) -> None:
        self.run_id = run_id
        self.state_path = state_path
        self.callbacks = callbacks or SchedulerCallbacks()
        self.task_configs = self._load_task_config(config or {})
        self.state = self._load_state()

    # ------------------------------------------
    # Public API
    # ------------------------------------------
    def on_cycle_complete(
        self,
        cycle_index: int,
        rye: Optional[float] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call this once per TGRM cycle.

        Returns a summary dict describing which maintenance tasks ran,
        any errors, and basic metadata.

        The rye and domain fields are optional and only used for
        logging context. The scheduler itself remains agnostic to
        the actual values.
        """
        now = _utc_now()
        now_iso = now.isoformat(timespec="seconds")

        self.state.last_cycle_index = cycle_index
        self.state.updated_at = now_iso

        tasks_run: List[str] = []
        task_results: Dict[str, Any] = {}
        errors: List[str] = []

        for key, cfg in self.task_configs.items():
            if not cfg.enabled:
                continue
            t_state = self.state.tasks.get(key) or TaskState()
            due, reason = self._is_task_due(cfg, t_state, cycle_index, now)
            if not due:
                continue

            try:
                result = self._run_task(key, cfg)
                tasks_run.append(key)
                if result is not None:
                    task_results[key] = result
                # update task state
                t_state.last_cycle = cycle_index
                t_state.last_timestamp = now_iso
                self.state.tasks[key] = t_state
            except Exception as e:
                msg = f"{key} task raised error: {e}"
                errors.append(msg)

        # persist state
        self._save_state()

        return {
            "timestamp": now_iso,
            "run_id": self.run_id,
            "cycle_index": cycle_index,
            "domain": domain,
            "rye": rye,
            "tasks_run": tasks_run,
            "task_results": task_results,
            "errors": errors,
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Return current scheduler state as simple dict.
        Useful for debugging or UI display.
        """
        return self._state_to_dict(self.state)

    # ------------------------------------------
    # Internal: config loading
    # ------------------------------------------
    def _load_task_config(self, cfg: Dict[str, Any]) -> Dict[str, TaskConfig]:
        """
        Merge user config with default config.
        """
        merged: Dict[str, TaskConfig] = {}

        for key, default_cfg in self.DEFAULT_CONFIG.items():
            user_cfg = cfg.get(key, {}) or {}
            merged[key] = TaskConfig(
                name=default_cfg.name,
                every_cycles=int(user_cfg.get("every_cycles", default_cfg.every_cycles))
                if user_cfg.get("every_cycles", default_cfg.every_cycles) is not None
                else None,
                every_minutes=float(user_cfg.get("every_minutes", default_cfg.every_minutes))
                if user_cfg.get("every_minutes", default_cfg.every_minutes) is not None
                else None,
                batch_limit=int(user_cfg.get("batch_limit", default_cfg.batch_limit))
                if user_cfg.get("batch_limit", default_cfg.batch_limit) is not None
                else None,
                enabled=bool(user_cfg.get("enabled", default_cfg.enabled)),
            )

        # allow user to add custom tasks if they want
        for key, user_cfg in cfg.items():
            if key in merged:
                continue
            merged[key] = TaskConfig(
                name=key,
                every_cycles=user_cfg.get("every_cycles"),
                every_minutes=user_cfg.get("every_minutes"),
                batch_limit=user_cfg.get("batch_limit"),
                enabled=user_cfg.get("enabled", True),
            )

        return merged

    # ------------------------------------------
    # Internal: state load and save
    # ------------------------------------------
    def _load_state(self) -> SchedulerState:
        if not self.state_path.exists():
            return self._new_state()

        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._new_state()

        tasks_state: Dict[str, TaskState] = {}
        raw_tasks = raw.get("tasks", {})
        for key, ts in raw_tasks.items():
            tasks_state[key] = TaskState(
                last_cycle=int(ts.get("last_cycle", 0)),
                last_timestamp=ts.get("last_timestamp"),
            )

        return SchedulerState(
            run_id=raw.get("run_id"),
            created_at=raw.get("created_at", _utc_iso()),
            updated_at=raw.get("updated_at", _utc_iso()),
            last_cycle_index=int(raw.get("last_cycle_index", 0)),
            tasks=tasks_state,
        )

    def _new_state(self) -> SchedulerState:
        now = _utc_iso()
        tasks_state = {key: TaskState() for key in self.task_configs.keys()}
        return SchedulerState(
            run_id=self.run_id,
            created_at=now,
            updated_at=now,
            last_cycle_index=0,
            tasks=tasks_state,
        )

    def _save_state(self) -> None:
        data = self._state_to_dict(self.state)
        try:
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            # Scheduler should never crash the worker on save failure
            return

    def _state_to_dict(self, st: SchedulerState) -> Dict[str, Any]:
        out = {
            "run_id": st.run_id,
            "created_at": st.created_at,
            "updated_at": st.updated_at,
            "last_cycle_index": st.last_cycle_index,
            "tasks": {},
        }
        for key, ts in st.tasks.items():
            out["tasks"][key] = {
                "last_cycle": ts.last_cycle,
                "last_timestamp": ts.last_timestamp,
            }
        return out

    # ------------------------------------------
    # Internal: due checks
    # ------------------------------------------
    def _is_task_due(
        self,
        cfg: TaskConfig,
        state: TaskState,
        current_cycle: int,
        now: datetime,
    ) -> (bool, str):
        """
        Decide if a task is due based on cycles and minutes.

        Returns:
            (is_due, reason_string)
        """
        if not cfg.enabled:
            return False, "disabled"

        cycle_due = False
        time_due = False
        reasons: List[str] = []

        if cfg.every_cycles is not None and cfg.every_cycles > 0:
            cycles_since = max(current_cycle - state.last_cycle, 0)
            if cycles_since >= cfg.every_cycles:
                cycle_due = True
                reasons.append(f"cycles_since={cycles_since} >= every_cycles={cfg.every_cycles}")

        if cfg.every_minutes is not None and cfg.every_minutes > 0:
            last_ts = _parse_ts(state.last_timestamp)
            if last_ts is None:
                time_due = True
                reasons.append("never_run_before")
            else:
                minutes_since = max((now - last_ts).total_seconds() / 60.0, 0.0)
                if minutes_since >= cfg.every_minutes:
                    time_due = True
                    reasons.append(f"minutes_since={minutes_since:.1f} >= every_minutes={cfg.every_minutes}")

        if cycle_due or time_due:
            return True, "; ".join(reasons) or "due"
        return False, "not_due"

    # ------------------------------------------
    # Internal: run tasks
    # ------------------------------------------
    def _run_task(self, key: str, cfg: TaskConfig) -> Any:
        """
        Dispatch to the right callback based on task key.
        All exceptions are handled in caller.
        """
        if key == "verification":
            if self.callbacks.run_verification_batch is None:
                return None
            limit = cfg.batch_limit or 5
            return self.callbacks.run_verification_batch(limit)

        if key == "discovery":
            if self.callbacks.run_discovery_pass is None:
                return None
            return self.callbacks.run_discovery_pass()

        if key == "clustering":
            if self.callbacks.run_clustering_pass is None:
                return None
            return self.callbacks.run_clustering_pass()

        if key == "snapshot":
            if self.callbacks.run_snapshot_pass is None:
                return None
            return self.callbacks.run_snapshot_pass()

        if key == "pruning":
            if self.callbacks.run_memory_pruning is None:
                return None
            return self.callbacks.run_memory_pruning()

        # Custom or unknown tasks can be added later by extending callbacks
        return None
