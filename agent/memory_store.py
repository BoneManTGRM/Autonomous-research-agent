# -*- coding: utf-8 -*-
"""Memory storage for the autonomous research agent.

This module provides a JSON based persistent storage layer. It stores:
- notes
- hypotheses
- citations
- biomarker summaries (future use)
- cycle logs
- run_state metadata (for long continuous runs and auto resume)
- watchdog information (heartbeats for crash diagnostics)
- goal_index summaries (lightweight per goal stats for swarms)
- events (streaming logs and partial updates)
- discoveries (cure, treatment, mechanism, and other key findings)
- run_manifests (compact summaries of long runs)
- tool_events (per tool usage events)
- milestones (key run milestones)
- hypothesis_evolution (how ideas are merged, split, or pruned)
- option_c_diagnostics (deep AGI style diagnostics for frontier runs)
- swarm_contracts (specialization contracts for swarm roles)
- learning_burst (burst learning state for the TGRM loop)
- benchmarks (ARC, math, longevity test batteries, etc.)
- source_index (lightweight index for linking entities to citation ids)
- replay_buffer (high value items for curriculum replay / fast recall)
- msil_snapshots (Meta Skill Intelligence Layer snapshots for learning analytics)

The memory store acts as a lightweight knowledge base for the agent and is
referenced each cycle to retrieve prior context and to persist new findings.

Reparodynamics interpretation:
    MemoryStore is the long term substrate where repairs accumulate.
    Each TGRM cycle writes its improvements here, and future cycles read
    from it to reduce energy cost. When combined with VectorMemory, this
    becomes a semantic, time aware repair substrate.

    The run_state, worker_state, watchdog, goal_index, events, discoveries,
    run_manifests, tool_events, milestones, hypothesis_evolution,
    option_c_diagnostics, swarm_contracts, learning_burst, benchmarks,
    source_index, replay_buffer, and msil_snapshots sections act as a meta
    layer: they record how the system itself is running so that the agent can
    restart and continue repair with minimal extra energy and give swarm level
    analytics (per role and per goal), plus a running log of key cure and
    treatment candidates, tool behavior, benchmark performance, learning curve
    shaping, and high value replay/skill intelligence artifacts.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Optional vector memory integration
try:
    from .vector_memory import VectorMemory  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    VectorMemory = None  # type: ignore[assignment]

# Optional advanced RYE metrics integration
try:
    from . import rye_metrics as _rye_metrics  # type: ignore[import]
except Exception:  # pragma: no cover
    _rye_metrics = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Schema and global caps
# ----------------------------------------------------------------------
MEMORY_SCHEMA_VERSION = 4

# Hard but generous caps for long runs.
# Important: these caps bound how much history is retained inside memory.json.
# They do NOT limit how many cycles the engine can actually execute.
# When a list exceeds its cap, the oldest entries are dropped and new
# cycles or events continue to be logged normally.
MAX_NOTES = 200_000
MAX_CYCLES = 300_000
MAX_HYPOTHESES = 100_000
MAX_CITATIONS = 100_000
MAX_BIOMARKERS = 50_000
MAX_EVENTS = 20_000
MAX_DISCOVERIES = 10_000
MAX_TOOL_EVENTS = 100_000
MAX_RUN_MANIFESTS = 10_000
MAX_MILESTONES = 20_000
MAX_BENCHMARKS = 100_000

# Replay + MSIL caps (kept separate so you can tune without touching core logs)
MAX_REPLAY_BUFFER = 50_000
MAX_MSIL_SNAPSHOTS = 50_000

# Explicit caps for evolution and frontier diagnostics
MAX_HYPOTHESIS_EVOLUTION = 20_000
MAX_OPTION_C_DIAGNOSTICS = 20_000
MAX_SWARM_CONTRACTS = 20_000

# Snapshot settings for Streamlit timeline and long runs
DEFAULT_SNAPSHOT_SECTIONS = [
    "run_state",
    "worker_state",
    "watchdog",
    "goal_index",
    "learning_burst",
    "cycles",
    "discoveries",
    "milestones",
    "benchmarks",
]

# Hard cap per run so snapshot folders cannot explode
MAX_SNAPSHOTS_PER_RUN = 200

# Logs / watchdog filenames (UI often reads these directly)
DEFAULT_LOGS_DIRNAME = "logs"
WATCHDOG_HEARTBEAT_FILENAME = "watchdog_heartbeat.json"

# Small artifact files used by the Streamlit UI and engine_worker.
# Keep this separate from MEMORY_SCHEMA_VERSION (memory.json schema).
RUNS_ARTIFACT_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now_ts() -> float:
    """Return current UTC time as a unix timestamp (seconds since epoch)."""
    return float(datetime.now(timezone.utc).timestamp())


def _to_int(value: Any) -> Optional[int]:
    """Best effort conversion to int that tolerates None and non int types."""
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


class MemoryStore:
    """A lightweight persistent memory store using a JSON file.

    The JSON file is structured with several top level keys:
        - "notes":               free form text notes with metadata
        - "cycles":              logs of each research cycle
        - "hypotheses":          generated hypotheses for each goal
        - "citations":           structured citation objects from web or papers
        - "biomarkers":          placeholder for anti aging or lab data
        - "run_state":           metadata for long running autonomous sessions
        - "worker_state":        live background worker status
        - "watchdog":            timestamps and counters for heartbeats
        - "goal_index":          compact per goal stats, including per role counts
        - "events":              streaming event log
        - "discoveries":         cure, treatment, mechanism, and other key finds
        - "run_manifests":       compact per run summaries for reports
        - "tool_events":         low level tool usage events
        - "milestones":          key run milestones
        - "learning_burst":      burst learning state for TGRM
        - "hypothesis_evolution": evolution of ideas over time
        - "option_c_diagnostics": deep diagnostics for Option C frontier runs
        - "swarm_contracts":     specialization contracts for swarm roles
        - "benchmarks":          benchmark and task results such as ARC, math suites
        - "source_index":        optional index for mapping entities to citations
        - "replay_buffer":       high value items for replay/curriculum
        - "msil_snapshots":      meta skill intelligence snapshots (optional)

    In memory (non persistent) vector memory may also be attached to
    support semantic search and time decayed retrieval if the optional
    VectorMemory class is available.

    This implementation is hardened for long autonomous runs:
        - safe directory handling even for plain filenames (for example "memory.json")
        - atomic write strategy to greatly reduce corruption risk
        - goal_index streaming stats for fast RYE summaries
        - worker_state and events for live status and debugging
        - bounded growth of logs (notes, cycles, hypotheses, citations,
          events, tool_events, hypothesis_evolution, option_c_diagnostics,
          swarm_contracts, benchmarks, replay_buffer, msil_snapshots)
        - per run manifests and milestones for report generation

    Advanced learning layer:
        - goal_index tracks best RYE, last RYE, and basic phase hints
        - optional advanced RYE metrics for learning curves
        - learning profiles and leaderboards for goals and roles
        - learning_burst supports temporary high intensity learning modes
        - hypothesis_evolution tracks how ideas refine or merge
        - option_c_diagnostics and swarm_contracts support frontier AGI runs
        - benchmarks give a persistent trace of test performance (ARC, math, etc.)
        - replay_buffer provides high value items for curriculum replay
        - msil_snapshots provide compact skill intelligence snapshots for analytics

    Watchdog notes:
        - This class maintains two watchdog outputs:
            1) watchdog.json (label->dict) for internal multi-label diagnostics
            2) logs/watchdog_heartbeat.json (flat dict) for UIs that read a single
               "heartbeat" file directly and expect unix timestamps + heartbeat_count
    """

    def __init__(
        self,
        memory_file: Optional[str] = None,
        *,
        base_dir: Optional[str] = None,
        filename: str = "memory.json",
    ) -> None:
        """Create a MemoryStore.

        New unified behavior for ARA:

        - If base_dir is given, it is used as the root for the memory file.
        - Otherwise ARA_RUNS_DIR from the environment is used if present.
        - If neither is set, a local "runs" directory under the current
          working directory is used as a fallback.
        - If memory_file is None, the file "<base_dir>/<filename>" is used.
        - If memory_file is an existing directory, the file
          "<memory_file>/<filename>" is used.
        - If memory_file is a relative path, it is resolved under base_dir.
        """
        # Resolve base_dir
        if base_dir is None:
            env_dir = os.environ.get("ARA_RUNS_DIR")
            if env_dir:
                base_dir = env_dir
            else:
                # If caller passed an explicit path to a file, prefer its directory
                if memory_file and not os.path.isdir(memory_file) and os.path.splitext(memory_file)[1]:
                    base_dir = os.path.dirname(os.path.abspath(memory_file)) or os.getcwd()
                else:
                    base_dir = os.path.join(os.getcwd(), "runs")

        self.base_dir = os.path.abspath(base_dir)

        # Resolve memory_file path
        if memory_file is None:
            memory_path = os.path.join(self.base_dir, filename)
        else:
            if os.path.isdir(memory_file):
                # Treat it as a directory, drop the file inside
                memory_path = os.path.join(memory_file, filename)
            else:
                # If relative path, place it under base_dir so worker and UI agree
                if not os.path.isabs(memory_file):
                    memory_path = os.path.join(self.base_dir, memory_file)
                else:
                    memory_path = memory_file

        self.memory_file = os.path.abspath(memory_path)

        # Logs directory (UI panels often read files from here)
        self.logs_dir = os.path.join(self.base_dir, DEFAULT_LOGS_DIRNAME)
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
        except Exception:
            pass

        # These external JSON files are used by diagnostics panels and
        # background workers for very fast, low risk status checks.
        #
        # IMPORTANT: The Streamlit UI and engine_worker expect these lightweight
        # artifacts under "<runs_root>/logs/". Older deployments sometimes wrote
        # them to "<runs_root>/". We support both:
        #   - Prefer logs/ paths for reads and writes
        #   - Mirror writes to legacy root paths when possible
        self.run_state_path = os.path.join(self.logs_dir, "run_state.json")
        self.worker_state_path = os.path.join(self.logs_dir, "worker_state.json")
        self.watchdog_path = os.path.join(self.base_dir, "watchdog.json")

        # Legacy root-level paths for backward compatibility
        self.legacy_run_state_path = os.path.join(self.base_dir, "run_state.json")
        self.legacy_worker_state_path = os.path.join(self.base_dir, "worker_state.json")

        # Flat heartbeat file for UIs that read a single watchdog file directly
        self.watchdog_heartbeat_path = os.path.join(self.logs_dir, WATCHDOG_HEARTBEAT_FILENAME)

        # Snapshots directory for run specific snapshots
        self.snapshot_root = os.path.join(self.base_dir, "snapshots")
        try:
            os.makedirs(self.snapshot_root, exist_ok=True)
        except Exception:
            pass

        # Ensure the directory exists (supports plain filenames like "memory.json")
        dirpath = os.path.dirname(self.memory_file)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        # Core JSON backed data
        self._data: Dict[str, Any] = {
            "notes": [],
            "cycles": [],
            "hypotheses": [],
            "citations": [],
            "biomarkers": [],
            "run_state": {},  # crash proof run metadata
            "worker_state": {},  # live worker mode and status
            "watchdog": {},  # heartbeat and last seen info
            "goal_index": {},  # compact goal wise stats
            "events": [],  # streaming event log
            "discoveries": [],  # cure, treatment, mechanism candidates
            "run_manifests": {},  # run_id -> manifest dict
            "tool_events": [],  # per tool usage events
            "milestones": [],  # milestones over long runs
            "learning_burst": {
                "active": False,
                "cycles_remaining": None,
                "burst_index": None,
            },
            "hypothesis_evolution": [],
            "option_c_diagnostics": [],
            "swarm_contracts": [],
            "benchmarks": [],
            "source_index": {},
            "replay_buffer": [],
            "msil_snapshots": [],
            "schema_version": MEMORY_SCHEMA_VERSION,
        }

        # Track last save error for debugging long runs
        self._last_save_error: Optional[str] = None

        self._load()
        self._ensure_keys()

        # Optional vector memory for semantic, time decayed retrieval
        if VectorMemory is not None:
            try:
                self.vector_memory: Optional[VectorMemory] = VectorMemory()
            except Exception:
                self.vector_memory = None
        else:
            self.vector_memory = None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def get_base_dir(self) -> str:
        """Return the resolved base directory for this MemoryStore."""
        return self.base_dir

    def has_vector_memory(self) -> bool:
        """Return True if an optional VectorMemory backend is attached."""
        return self.vector_memory is not None

    def has_advanced_rye_metrics(self) -> bool:
        """Return True if optional advanced RYE metrics module is available."""
        return _rye_metrics is not None

    def get_logs_dir(self) -> str:
        """Return the resolved logs directory used for lightweight status files."""
        return self.logs_dir

    # ------------------------------------------------------------------
    # Internal JSON persistence
    # ------------------------------------------------------------------
    def _ensure_keys(self) -> None:
        """Ensure all expected top level keys exist in _data."""
        defaults = {
            "notes": [],
            "cycles": [],
            "hypotheses": [],
            "citations": [],
            "biomarkers": [],
            "run_state": {},
            "worker_state": {},
            "watchdog": {},
            "goal_index": {},
            "events": [],
            "discoveries": [],
            "run_manifests": {},
            "tool_events": [],
            "milestones": [],
            "learning_burst": {},
            "hypothesis_evolution": [],
            "option_c_diagnostics": [],
            "swarm_contracts": [],
            "benchmarks": [],
            "source_index": {},
            "replay_buffer": [],
            "msil_snapshots": [],
        }
        for key, default in defaults.items():
            if key not in self._data:
                self._data[key] = default
            else:
                if key in (
                    "notes",
                    "cycles",
                    "hypotheses",
                    "citations",
                    "biomarkers",
                    "events",
                    "discoveries",
                    "tool_events",
                    "milestones",
                    "hypothesis_evolution",
                    "option_c_diagnostics",
                    "swarm_contracts",
                    "benchmarks",
                    "replay_buffer",
                    "msil_snapshots",
                ):
                    if not isinstance(self._data.get(key), list):
                        self._data[key] = []
                elif key in (
                    "run_manifests",
                    "run_state",
                    "worker_state",
                    "watchdog",
                    "goal_index",
                    "learning_burst",
                    "source_index",
                ):
                    if not isinstance(self._data.get(key), dict):
                        self._data[key] = {}
        # Ensure schema version is present
        if not isinstance(self._data.get("schema_version"), int):
            self._data["schema_version"] = MEMORY_SCHEMA_VERSION

        # Ensure learning_burst has minimal shape
        lb = self._data.get("learning_burst")
        if not isinstance(lb, dict):
            lb = {}
        lb.setdefault("active", False)
        lb.setdefault("cycles_remaining", None)
        lb.setdefault("burst_index", None)
        self._data["learning_burst"] = lb

        # Ensure source_index is a dict
        if not isinstance(self._data.get("source_index"), dict):
            self._data["source_index"] = {}

    def _load(self) -> None:
        """Load memory from disk if the file exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # If the file is corrupt or cannot be read, rename it for forensics
                try:
                    now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                    backup_path = f"{self.memory_file}.corrupt-{now}"
                    os.rename(self.memory_file, backup_path)
                except Exception:
                    pass
                # Start with a fresh structure
                self._data = {
                    "notes": [],
                    "cycles": [],
                    "hypotheses": [],
                    "citations": [],
                    "biomarkers": [],
                    "run_state": {},
                    "worker_state": {},
                    "watchdog": {},
                    "goal_index": {},
                    "events": [],
                    "discoveries": [],
                    "run_manifests": {},
                    "tool_events": [],
                    "milestones": [],
                    "learning_burst": {
                        "active": False,
                        "cycles_remaining": None,
                        "burst_index": None,
                    },
                    "hypothesis_evolution": [],
                    "option_c_diagnostics": [],
                    "swarm_contracts": [],
                    "benchmarks": [],
                    "source_index": {},
                    "replay_buffer": [],
                    "msil_snapshots": [],
                    "schema_version": MEMORY_SCHEMA_VERSION,
                }

    def _save(self) -> None:
        """Persist memory to disk using an atomic write pattern.

        Atomic pattern (write then replace) greatly reduces the chance of
        corrupting the JSON file during very long autonomous runs, where
        the process might be interrupted mid write.
        """
        tmp_path = self.memory_file + ".tmp"
        try:
            self._data["schema_version"] = MEMORY_SCHEMA_VERSION
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.memory_file)
            self._last_save_error = None
        except Exception as e:
            # Track last save error in memory for debugging
            self._last_save_error = f"{type(e).__name__}: {e}"
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _write_json_file(self, path: str, payload: Dict[str, Any]) -> None:
        """Write a small JSON payload to a path using atomic replace."""
        tmp_path = path + ".tmp"
        try:
            dirpath = os.path.dirname(os.path.abspath(path))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        except Exception:
            pass

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _read_json_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a small JSON file and return a dict or None on failure."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    # Public helper if you ever want to force a flush from outside
    def flush(self) -> None:
        """Force a disk flush of current in memory state."""
        self._ensure_keys()
        self._save()

    # Optional snapshot helper for manual backups during long runs
    def export_snapshot(self, path: Optional[str] = None) -> str:
        """Export a full JSON snapshot of the current memory file.

        Args:
            path:
                Optional explicit path. If None, a timestamped
                "<memory_file>.snapshot-YYYYmmddHHMMSS.json" file is used.

        Returns:
            The path of the snapshot file that was written.
        """
        self._ensure_keys()
        now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        if path is None:
            base = os.path.abspath(self.memory_file)
            path = f"{base}.snapshot-{now}.json"

        try:
            snapshot_dir = os.path.dirname(os.path.abspath(path))
            if snapshot_dir and not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir, exist_ok=True)
        except Exception:
            pass

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            return path

        return path

    def write_snapshot(
        self,
        *,
        run_id: str,
        label: Optional[str] = None,
        cycle_index: Optional[int] = None,
        reason: str = "interval",
        include_sections: Optional[List[str]] = None,
        max_snapshots_per_run: Optional[int] = None,
    ) -> Optional[str]:
        """Write a compact snapshot for a specific run_id.

        This is the method engine_worker should call when snapshot_config
        says it is time to record a snapshot.
        """
        if not run_id:
            return None

        sections = include_sections or list(DEFAULT_SNAPSHOT_SECTIONS)
        now = _utc_now_iso()

        # Build payload with only the requested sections
        snapshot_sections: Dict[str, Any] = {}
        for key in sections:
            if key in self._data:
                try:
                    value = self._data.get(key)
                    if isinstance(value, (dict, list)):
                        snapshot_sections[key] = json.loads(json.dumps(value))
                    else:
                        snapshot_sections[key] = value
                except Exception:
                    # If something cannot be serialized, skip that section
                    continue

        payload: Dict[str, Any] = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "created_at": now,
            "run_id": run_id,
            "label": label,
            "cycle_index": cycle_index,
            "reason": reason,
            "sections": sections,
            "data": snapshot_sections,
        }

        # Per run directory under snapshot_root
        run_dir = os.path.join(self.snapshot_root, run_id)
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception:
            return None

        # File name encodes timestamp and cycle index
        cycle_tag = f"c{cycle_index}" if cycle_index is not None else "cNA"
        filename = f"snapshot-{now.replace(':', '').replace('-', '')}-{cycle_tag}.json"
        path = os.path.join(run_dir, filename)

        # Use the same atomic write pattern as _write_json_file
        try:
            self._write_json_file(path, payload)
        except Exception:
            return None

        # Optional simple pruning inside the run folder
        cap = (
            max_snapshots_per_run
            if isinstance(max_snapshots_per_run, int) and max_snapshots_per_run > 0
            else MAX_SNAPSHOTS_PER_RUN
        )
        try:
            files = sorted(
                [
                    f
                    for f in os.listdir(run_dir)
                    if f.startswith("snapshot-") and f.endswith(".json")
                ]
            )
            if len(files) > cap:
                excess = len(files) - cap
                for old_name in files[:excess]:
                    try:
                        os.remove(os.path.join(run_dir, old_name))
                    except Exception:
                        pass
        except Exception:
            pass

        return path

    def maybe_write_snapshot_from_config(
        self,
        *,
        run_id: str,
        cycle_index: Optional[int],
        snapshot_config: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Helper that reads a simple snapshot_config dict.

        Expected keys in snapshot_config:
            enabled: bool
            interval_cycles: Optional[int]
            include_sections: Optional[List[str]]
            max_snapshots_per_run: Optional[int]
            label: Optional[str]
        """
        if not snapshot_config or not snapshot_config.get("enabled"):
            return None

        interval = snapshot_config.get("interval_cycles")
        if isinstance(interval, int) and interval > 0 and cycle_index is not None:
            if cycle_index <= 0 or cycle_index % interval != 0:
                return None

        include_sections = snapshot_config.get("include_sections")
        max_snapshots = snapshot_config.get("max_snapshots_per_run")

        return self.write_snapshot(
            run_id=run_id,
            label=snapshot_config.get("label"),
            cycle_index=cycle_index,
            reason="interval",
            include_sections=include_sections,
            max_snapshots_per_run=max_snapshots,
        )

    # Simple schema and file info for UI or engine_worker dashboards
    def get_schema_info(self) -> Dict[str, Any]:
        """Return basic schema and file info for status panels."""
        info: Dict[str, Any] = {
            "schema_version": int(self._data.get("schema_version", MEMORY_SCHEMA_VERSION)),
            "file_path": os.path.abspath(self.memory_file),
            "file_size_bytes": None,
            "base_dir": self.base_dir,
            "logs_dir": self.logs_dir,
            "watchdog_heartbeat_path": self.watchdog_heartbeat_path,
            "env_ARA_RUNS_DIR": os.environ.get("ARA_RUNS_DIR"),
        }
        try:
            info["file_size_bytes"] = os.path.getsize(self.memory_file)
        except Exception:
            info["file_size_bytes"] = None
        return info

    def get_last_save_error(self) -> Optional[str]:
        """Return the last save error (if any) for debugging."""
        return self._last_save_error

    def get_memory_overview(self) -> Dict[str, Any]:
        """Return a compact overview of memory contents for diagnostics.

        This is a low cost way for UIs or controllers to see how large the
        main collections are and whether the caps are being approached.
        """
        overview: Dict[str, Any] = {
            "schema_version": int(self._data.get("schema_version", MEMORY_SCHEMA_VERSION)),
            "file_path": os.path.abspath(self.memory_file),
            "base_dir": self.base_dir,
            "logs_dir": self.logs_dir,
            "snapshot_root": self.snapshot_root,
            "run_state_path": self.run_state_path,
            "worker_state_path": self.worker_state_path,
            "legacy_run_state_path": getattr(self, "legacy_run_state_path", None),
            "legacy_worker_state_path": getattr(self, "legacy_worker_state_path", None),
            "watchdog_path": self.watchdog_path,
            "watchdog_heartbeat_path": self.watchdog_heartbeat_path,
            "counts": {
                "notes": len(self._data.get("notes", [])),
                "cycles": len(self._data.get("cycles", [])),
                "hypotheses": len(self._data.get("hypotheses", [])),
                "citations": len(self._data.get("citations", [])),
                "biomarkers": len(self._data.get("biomarkers", [])),
                "events": len(self._data.get("events", [])),
                "discoveries": len(self._data.get("discoveries", [])),
                "tool_events": len(self._data.get("tool_events", [])),
                "milestones": len(self._data.get("milestones", [])),
                "hypothesis_evolution": len(self._data.get("hypothesis_evolution", [])),
                "option_c_diagnostics": len(self._data.get("option_c_diagnostics", [])),
                "swarm_contracts": len(self._data.get("swarm_contracts", [])),
                "benchmarks": len(self._data.get("benchmarks", [])),
                "run_manifests": len(self._data.get("run_manifests", {})),
                "replay_buffer": len(self._data.get("replay_buffer", [])),
                "msil_snapshots": len(self._data.get("msil_snapshots", [])),
            },
            "caps": {
                "MAX_NOTES": MAX_NOTES,
                "MAX_CYCLES": MAX_CYCLES,
                "MAX_HYPOTHESES": MAX_HYPOTHESES,
                "MAX_CITATIONS": MAX_CITATIONS,
                "MAX_BIOMARKERS": MAX_BIOMARKERS,
                "MAX_EVENTS": MAX_EVENTS,
                "MAX_DISCOVERIES": MAX_DISCOVERIES,
                "MAX_TOOL_EVENTS": MAX_TOOL_EVENTS,
                "MAX_RUN_MANIFESTS": MAX_RUN_MANIFESTS,
                "MAX_MILESTONES": MAX_MILESTONES,
                "MAX_HYPOTHESIS_EVOLUTION": MAX_HYPOTHESIS_EVOLUTION,
                "MAX_OPTION_C_DIAGNOSTICS": MAX_OPTION_C_DIAGNOSTICS,
                "MAX_SWARM_CONTRACTS": MAX_SWARM_CONTRACTS,
                "MAX_BENCHMARKS": MAX_BENCHMARKS,
                "MAX_REPLAY_BUFFER": MAX_REPLAY_BUFFER,
                "MAX_MSIL_SNAPSHOTS": MAX_MSIL_SNAPSHOTS,
            },
            "last_save_error": self._last_save_error,
        }
        try:
            overview["file_size_bytes"] = os.path.getsize(self.memory_file)
        except Exception:
            overview["file_size_bytes"] = None
        return overview

    # Optional hard reset for dev and testing
    def reset_memory(self, keep_run_state: bool = False) -> None:
        """Reset the memory file to an empty state.

        This is mainly for development and testing. In production you would
        typically never call this.
        """
        run_state_backup = self._data.get("run_state") if keep_run_state else {}
        self._data = {
            "notes": [],
            "cycles": [],
            "hypotheses": [],
            "citations": [],
            "biomarkers": [],
            "run_state": run_state_backup or {},
            "worker_state": {},
            "watchdog": {},
            "goal_index": {},
            "events": [],
            "discoveries": [],
            "run_manifests": {},
            "tool_events": [],
            "milestones": [],
            "learning_burst": {
                "active": False,
                "cycles_remaining": None,
                "burst_index": None,
            },
            "hypothesis_evolution": [],
            "option_c_diagnostics": [],
            "swarm_contracts": [],
            "benchmarks": [],
            "source_index": {},
            "replay_buffer": [],
            "msil_snapshots": [],
            "schema_version": MEMORY_SCHEMA_VERSION,
        }
        self._save()

        # Also clear lightweight external state files so dashboards and workers
        # do not see stale status after a reset.
        try:
            self._write_json_file(self.run_state_path, {})
        except Exception:
            pass
        # Also clear legacy root-level path if configured
        try:
            legacy_path = getattr(self, "legacy_run_state_path", None)
            if isinstance(legacy_path, str) and legacy_path and legacy_path != self.run_state_path:
                self._write_json_file(legacy_path, {})
        except Exception:
            pass
        try:
            self._write_json_file(self.worker_state_path, {})
        except Exception:
            pass
        # Also clear legacy root-level path if configured
        try:
            legacy_path = getattr(self, "legacy_worker_state_path", None)
            if isinstance(legacy_path, str) and legacy_path and legacy_path != self.worker_state_path:
                self._write_json_file(legacy_path, {})
        except Exception:
            pass
        try:
            self._write_json_file(self.watchdog_path, {})
        except Exception:
            pass
        try:
            self._write_json_file(self.watchdog_heartbeat_path, {})
        except Exception:
            pass

    def prune_all(self) -> None:
        """Apply caps to all bounded collections and save.

        Useful after manual edits or schema migrations to bring a large
        memory file back inside the intended limits without losing
        the most recent data.
        """
        caps = {
            "notes": MAX_NOTES,
            "cycles": MAX_CYCLES,
            "hypotheses": MAX_HYPOTHESES,
            "citations": MAX_CITATIONS,
            "biomarkers": MAX_BIOMARKERS,
            "events": MAX_EVENTS,
            "discoveries": MAX_DISCOVERIES,
            "tool_events": MAX_TOOL_EVENTS,
            "milestones": MAX_MILESTONES,
            "hypothesis_evolution": MAX_HYPOTHESIS_EVOLUTION,
            "option_c_diagnostics": MAX_OPTION_C_DIAGNOSTICS,
            "swarm_contracts": MAX_SWARM_CONTRACTS,
            "benchmarks": MAX_BENCHMARKS,
            "replay_buffer": MAX_REPLAY_BUFFER,
            "msil_snapshots": MAX_MSIL_SNAPSHOTS,
        }

        for key, cap in caps.items():
            arr = self._data.get(key)
            if isinstance(arr, list) and len(arr) > cap:
                self._data[key] = arr[-cap:]

        rm = self._data.get("run_manifests")
        if isinstance(rm, dict) and len(rm) > MAX_RUN_MANIFESTS:
            items = sorted(
                rm.items(),
                key=lambda kv: str(kv[1].get("logged_at", "")),
            )
            excess = len(items) - MAX_RUN_MANIFESTS
            for k, _v in items[:excess]:
                rm.pop(k, None)
            self._data["run_manifests"] = rm

        self._save()

    # ------------------------------------------------------------------
    # Internal helpers for goal index
    # ------------------------------------------------------------------
    def _touch_goal_index(
        self,
        goal: str,
        *,
        last_event_type: str,
        role: Optional[str] = None,
        delta_notes: int = 0,
        delta_cycles: int = 0,
        rye_value: Optional[float] = None,
        domain: Optional[str] = None,
        equilibrium_label: Optional[str] = None,
        breakthrough_score: Optional[float] = None,
        cycle_index: Optional[int] = None,
    ) -> None:
        """Update compact per goal stats for swarms and analytics.

        This maintains:
            - note_count, cycle_count
            - per role counts and avg_rye
            - goal level avg_rye, min_rye, max_rye, rye_count
            - best and last RYE per goal
            - last equilibrium label and basic breakthrough info
            - streaming last_rye_delta for speed of learning
        """
        gi = self._data.setdefault("goal_index", {})
        entry = gi.get(goal) or {}
        if not isinstance(entry, dict):
            entry = {}

        now = _utc_now_iso()
        entry.setdefault("created_at", now)
        entry["last_updated"] = now
        entry["last_event_type"] = last_event_type

        if domain is not None:
            entry.setdefault("domain", domain)

        # Note counts
        note_count = int(entry.get("note_count", 0)) + int(delta_notes)
        entry["note_count"] = note_count

        # Cycle counts
        cycle_count = int(entry.get("cycle_count", 0)) + int(delta_cycles)
        entry["cycle_count"] = cycle_count

        # Per role counters for swarms
        roles_dict = entry.get("roles") or {}
        if not isinstance(roles_dict, dict):
            roles_dict = {}
        if role:
            r_stats = roles_dict.get(role) or {}
            if not isinstance(r_stats, dict):
                r_stats = {}
            r_stats["note_count"] = int(r_stats.get("note_count", 0)) + int(delta_notes)
            r_stats["cycle_count"] = int(r_stats.get("cycle_count", 0)) + int(delta_cycles)
            if rye_value is not None:
                prev_avg = r_stats.get("avg_rye")
                prev_n = int(r_stats.get("rye_count", 0))
                if isinstance(prev_avg, (int, float)) and prev_n > 0:
                    new_avg = (prev_avg * prev_n + float(rye_value)) / float(prev_n + 1)
                    r_stats["avg_rye"] = new_avg
                    r_stats["rye_count"] = prev_n + 1
                else:
                    r_stats["avg_rye"] = float(rye_value)
                    r_stats["rye_count"] = 1
            roles_dict[role] = r_stats
        entry["roles"] = roles_dict

        # Goal level RYE stats (streaming) for fast summary
        if rye_value is not None:
            prev_avg = entry.get("avg_rye")
            prev_n = int(entry.get("rye_count", 0))
            v = float(rye_value)

            if isinstance(prev_avg, (int, float)) and prev_n > 0:
                new_avg = (prev_avg * prev_n + v) / float(prev_n + 1)
                entry["avg_rye"] = new_avg
                entry["rye_count"] = prev_n + 1
            else:
                entry["avg_rye"] = v
                entry["rye_count"] = 1

            # Track min and max RYE at the goal index level as well
            prev_min = entry.get("min_rye")
            prev_max = entry.get("max_rye")
            if isinstance(prev_min, (int, float)):
                entry["min_rye"] = min(float(prev_min), v)
            else:
                entry["min_rye"] = v
            if isinstance(prev_max, (int, float)):
                entry["max_rye"] = max(float(prev_max), v)
            else:
                entry["max_rye"] = v

            # Track best RYE and cycle index for fast learning profile
            prev_last_rye = entry.get("last_rye")
            best_rye = entry.get("best_rye")
            if not isinstance(best_rye, (int, float)) or v > float(best_rye):
                entry["best_rye"] = v
                if cycle_index is not None:
                    entry["best_cycle_index"] = int(cycle_index)

            entry["last_rye"] = v
            if isinstance(prev_last_rye, (int, float)):
                try:
                    entry["last_rye_delta"] = v - float(prev_last_rye)
                except Exception:
                    entry["last_rye_delta"] = None
            if cycle_index is not None:
                entry["last_cycle_index"] = int(cycle_index)

        # Learning hints from equilibrium and breakthrough
        if equilibrium_label:
            entry["last_equilibrium_label"] = equilibrium_label

        if isinstance(breakthrough_score, (int, float)):
            entry["last_breakthrough_score"] = float(breakthrough_score)
            prev_best_bs = entry.get("best_breakthrough_score")
            if not isinstance(prev_best_bs, (int, float)) or breakthrough_score > float(prev_best_bs):
                entry["best_breakthrough_score"] = float(breakthrough_score)

        gi[goal] = entry
        self._data["goal_index"] = gi

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    def add_note(
        self,
        goal: str,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        role: str = "agent",
        importance: float = 1.0,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        cycle_index: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a note associated with a research goal."""
        note: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "content": content,
            "tags": tags or [],
            "role": role,
            "importance": float(importance),
        }
        if domain is not None:
            note["domain"] = domain
        if run_id is not None:
            note["run_id"] = run_id
        if cycle_index is not None:
            note["cycle_index"] = int(cycle_index)
        if extra:
            note["extra"] = dict(extra)

        self._data.setdefault("notes", []).append(note)

        # Keep notes bounded for ultra long runs
        if len(self._data["notes"]) > MAX_NOTES:
            self._data["notes"] = self._data["notes"][-MAX_NOTES:]

        self._touch_goal_index(
            goal,
            last_event_type="note",
            role=role,
            delta_notes=1,
            delta_cycles=0,
            rye_value=None,
            domain=domain,
        )
        self._save()

        # Also store in vector memory if available, for semantic retrieval
        if self.vector_memory is not None and content:
            try:
                meta: Dict[str, Any] = {
                    "goal": goal,
                    "tags": tags or [],
                    "role": role,
                    "importance": float(importance),
                }
                if domain is not None:
                    meta["domain"] = domain
                if run_id is not None:
                    meta["run_id"] = run_id
                if cycle_index is not None:
                    meta["cycle_index"] = int(cycle_index)
                self.vector_memory.add_item(text=content, metadata=meta)
            except Exception:
                pass

    def get_notes(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve notes. If goal is None, return all notes."""
        notes = self._data.get("notes", [])
        if goal is None:
            return list(notes)
        return [n for n in notes if n.get("goal") == goal]

    def search_notes(self, query: str, top_k: int = 5, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search notes either via vector memory (if available) or simple keyword search.

        Fallback keyword search is learning aware:
            - ranks by importance
            - breaks ties by recency
        """
        if not query:
            return []

        # If vector memory is present, use it
        if self.vector_memory is not None:
            try:
                items = self.vector_memory.search(query=query, top_k=top_k)
                results: List[Dict[str, Any]] = []
                for it in items:
                    meta = dict(it.metadata)
                    meta.setdefault("content", it.text)
                    meta.setdefault("timestamp", _utc_now_iso())
                    if goal is None or meta.get("goal") == goal:
                        results.append(meta)
                return results
            except Exception:
                pass

        # Keyword based fallback with basic ranking
        notes = self.get_notes(goal)
        query_lower = query.lower()
        matched = [n for n in notes if query_lower in str(n.get("content", "")).lower()]

        def _note_key(n: Dict[str, Any]) -> Tuple[float, str]:
            importance = float(n.get("importance", 1.0))
            ts = str(n.get("timestamp", ""))
            return (importance, ts)

        matched_sorted = sorted(matched, key=_note_key, reverse=True)
        return matched_sorted[:top_k]

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------
    def add_hypothesis(
        self,
        goal: str,
        text: str,
        *,
        score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        domain: Optional[str] = None,
        role: Optional[str] = None,
        run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a generated hypothesis for a given goal."""
        hyp: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "text": text,
            "score": float(score) if score is not None else None,
            "tags": tags or [],
        }
        if domain is not None:
            hyp["domain"] = domain
        if role is not None:
            hyp["role"] = role
        if run_id is not None:
            hyp["run_id"] = run_id
        if extra:
            hyp["extra"] = dict(extra)

        self._data.setdefault("hypotheses", []).append(hyp)

        # Bound hypothesis list growth
        if len(self._data["hypotheses"]) > MAX_HYPOTHESES:
            self._data["hypotheses"] = self._data["hypotheses"][-MAX_HYPOTHESES:]

        self._save()

        # Hypotheses are also valuable semantic memory
        if self.vector_memory is not None and text:
            try:
                meta: Dict[str, Any] = {
                    "goal": goal,
                    "type": "hypothesis",
                    "score": hyp["score"],
                    "tags": hyp["tags"],
                }
                if domain is not None:
                    meta["domain"] = domain
                if role is not None:
                    meta["role"] = role
                if run_id is not None:
                    meta["run_id"] = run_id
                self.vector_memory.add_item(text=text, metadata=meta)
            except Exception:
                pass

    def get_hypotheses(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored hypotheses, optionally filtered by goal."""
        hyps = self._data.get("hypotheses", [])
        if goal is None:
            return list(hyps)
        return [h for h in hyps if h.get("goal") == goal]

    # ------------------------------------------------------------------
    # Hypothesis evolution (merge, split, mutate tracking)
    # ------------------------------------------------------------------
    def add_hypothesis_evolution(self, goal: str, evolution: Dict[str, Any]) -> None:
        """Track evolution of hypotheses for a goal (merge, split, mutate, prune)."""
        entry = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "evolution": evolution,
        }
        arr = self._data.setdefault("hypothesis_evolution", [])
        arr.append(entry)
        if len(arr) > MAX_HYPOTHESIS_EVOLUTION:
            self._data["hypothesis_evolution"] = arr[-MAX_HYPOTHESIS_EVOLUTION:]
        self._save()

    def get_hypothesis_evolution(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve hypothesis evolution entries, optionally filtered by goal."""
        entries = self._data.get("hypothesis_evolution", [])
        if goal is None:
            return list(entries)
        return [e for e in entries if e.get("goal") == goal]

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------
    def add_citation(
        self,
        goal: str,
        citation: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        cycle_index: Optional[int] = None,
        tool_name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a structured citation object.

        This function is tolerant to multiple citation shapes, including:
            - older browser style: {title, snippet, url, source}
            - Tavily style: {provider, source, title, url, content, raw_content, ...}
            - minimal shapes where only url or content is present

        It normalizes the citation for storage and for vector memory.
        """
        # Ensure we have a dict
        if not isinstance(citation, dict):
            citation = {"raw": str(citation)}

        # Shallow copy to avoid mutating caller state
        c = dict(citation)

        # Provider/source normalization
        provider = c.get("provider")
        source = c.get("source") or provider or "web"

        # Title normalization
        title_val = c.get("title")
        if isinstance(title_val, str):
            title = title_val.strip()
        else:
            title = ""

        # URL normalization
        url_val = c.get("url")
        url = str(url_val).strip() if url_val is not None else ""

        # Content or snippet normalization
        snippet = ""
        # Prefer explicit snippet if present
        if isinstance(c.get("snippet"), str) and c["snippet"].strip():
            snippet = c["snippet"].strip()
        else:
            # Fallback sequence for Tavily and other providers
            for key in ("content", "text", "raw_content", "body"):
                val = c.get(key)
                if isinstance(val, str) and val.strip():
                    snippet = val.strip()
                    break

        # Store normalized fields back into the citation dict
        c["source"] = source
        if title:
            c["title"] = title
        elif url:
            c["title"] = url
        else:
            c.setdefault("title", source)

        if url:
            c["url"] = url
        if snippet:
            c["snippet"] = snippet

        # Filter out obvious tool errors so they do not pollute citations
        is_error_like = False
        title_lower = str(c.get("title", "")).lower()
        snippet_lower = str(c.get("snippet", "")).lower()

        if "search error" in title_lower or "semantic scholar error" in title_lower:
            is_error_like = True
        if snippet_lower.startswith("[stub]"):
            is_error_like = True
        if c.get("error"):
            is_error_like = True
        if not url and not snippet:
            is_error_like = True

        if is_error_like:
            # Log as tool error instead of a citation
            try:
                self.log_tool_event(
                    run_id=run_id,
                    goal=goal,
                    domain=domain,
                    role=role,
                    cycle_index=cycle_index,
                    tool_name=tool_name or str(provider or source or "unknown"),
                    status="error",
                    error=str(c.get("error") or c.get("title") or "citation_error"),
                    extra={"raw_citation": c},
                )
            except Exception:
                pass
            return

        entry: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "citation": c,
        }
        if run_id is not None:
            entry["run_id"] = run_id
        if role is not None:
            entry["role"] = role
        if domain is not None:
            entry["domain"] = domain
        if cycle_index is not None:
            entry["cycle_index"] = int(cycle_index)
        if tool_name is not None:
            entry["tool_name"] = tool_name
        if extra:
            entry["extra"] = dict(extra)

        self._data.setdefault("citations", []).append(entry)

        # Bound citation growth
        if len(self._data["citations"]) > MAX_CITATIONS:
            self._data["citations"] = self._data["citations"][-MAX_CITATIONS:]

        self._save()

        # Optional vector memory storage
        if self.vector_memory is not None:
            try:
                text_parts = [
                    str(c.get("title", "")),
                    str(c.get("snippet", "")),
                    str(c.get("content", "")),
                    str(c.get("url", "")),
                ]
                text = " ".join([p for p in text_parts if p])
                meta: Dict[str, Any] = {
                    "goal": goal,
                    "type": "citation",
                    "source": c.get("source", "web"),
                    "url": c.get("url", ""),
                }
                if run_id is not None:
                    meta["run_id"] = run_id
                if role is not None:
                    meta["role"] = role
                if domain is not None:
                    meta["domain"] = domain
                if cycle_index is not None:
                    meta["cycle_index"] = int(cycle_index)
                if tool_name is not None:
                    meta["tool_name"] = tool_name
                if text:
                    self.vector_memory.add_item(text=text, metadata=meta)
            except Exception:
                pass

    def get_citations(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored citations, optionally filtered by goal."""
        entries = self._data.get("citations", [])
        if goal is None:
            return list(entries)
        return [e for e in entries if e.get("goal") == goal]

    def get_citations_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieve citations associated with a specific run_id."""
        entries = self._data.get("citations", [])
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict) and e.get("run_id") == run_id]

    # ------------------------------------------------------------------
    # Biomarkers (placeholder for anti aging or lab data)
    # ------------------------------------------------------------------
    def add_biomarker_snapshot(
        self,
        goal: str,
        data: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        domain: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a biomarker snapshot (anti aging or health metrics)."""
        entry: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "data": data,
        }
        if run_id is not None:
            entry["run_id"] = run_id
        if domain is not None:
            entry["domain"] = domain
        if extra:
            entry["extra"] = dict(extra)

        self._data.setdefault("biomarkers", []).append(entry)

        if len(self._data["biomarkers"]) > MAX_BIOMARKERS:
            self._data["biomarkers"] = self._data["biomarkers"][-MAX_BIOMARKERS:]

        self._save()

    def get_biomarker_history(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve biomarker history, optionally filtered by goal."""
        entries = self._data.get("biomarkers", [])
        if goal is None:
            return list(entries)
        return [e for e in entries if e.get("goal") == goal]

    # ------------------------------------------------------------------
    # Cycle logs
    # ------------------------------------------------------------------
    def log_cycle(self, cycle_data: Dict[str, Any]) -> None:
        """Append cycle data to the cycle log and update learning signals."""
        # Normalize cycle index key for downstream analytics (MSIL, dashboards, etc.)
        # Many callers use "cycle"; some use "cycle_index".
        if isinstance(cycle_data, dict):
            if "cycle_index" not in cycle_data and "cycle" in cycle_data:
                cycle_data["cycle_index"] = cycle_data.get("cycle")
            elif "cycle" not in cycle_data and "cycle_index" in cycle_data:
                # Maintain legacy key for any downstream consumers still reading "cycle"
                cycle_data["cycle"] = cycle_data.get("cycle_index")

        self._data.setdefault("cycles", []).append(cycle_data)

        # Bound cycles growth
        if len(self._data["cycles"]) > MAX_CYCLES:
            self._data["cycles"] = self._data["cycles"][-MAX_CYCLES:]

        # Update compact goal index for swarms and learning
        goal = str(cycle_data.get("goal", ""))
        role = str(cycle_data.get("role", "agent"))
        domain = cycle_data.get("domain")
        rye_val = cycle_data.get("RYE")
        cycle_index = None
        try:
            if "cycle_index" in cycle_data:
                cycle_index = int(cycle_data.get("cycle_index"))
            elif "cycle" in cycle_data:
                cycle_index = int(cycle_data.get("cycle"))
        except Exception:
            cycle_index = None

        equilibrium_label = None
        breakthrough_score = None
        try:
            eq = cycle_data.get("equilibrium") or {}
            if isinstance(eq, dict):
                equilibrium_label = eq.get("equilibrium_label")
        except Exception:
            equilibrium_label = None

        try:
            br = cycle_data.get("breakthrough") or {}
            if isinstance(br, dict):
                breakthrough_score = br.get("breakthrough_score")
        except Exception:
            breakthrough_score = None

        if goal:
            try:
                if isinstance(rye_val, (int, float)):
                    rye_float: Optional[float] = float(rye_val)
                else:
                    rye_float = None
                self._touch_goal_index(
                    goal,
                    last_event_type="cycle",
                    role=role,
                    delta_notes=0,
                    delta_cycles=1,
                    rye_value=rye_float,
                    domain=domain,
                    equilibrium_label=equilibrium_label,
                    breakthrough_score=breakthrough_score,
                    cycle_index=cycle_index,
                )
            except Exception:
                pass

        # Optional automatic milestone when breakthrough_score is high
        try:
            if isinstance(breakthrough_score, (int, float)) and breakthrough_score >= 0.8 and goal:
                flags = []
                br = cycle_data.get("breakthrough") or {}
                if isinstance(br, dict):
                    flags = br.get("flags") or []
                desc = f"Cycle {cycle_index} reached breakthrough_score {breakthrough_score:.3f}"
                if flags:
                    desc += f" with flags {flags}"
                self.log_milestone(
                    run_id=cycle_data.get("run_id"),
                    goal=goal,
                    domain=domain,
                    label="high_breakthrough_score",
                    description=desc,
                    level="info",
                    role=role,
                    cycle_index=cycle_index,
                    extra={
                        "equilibrium_label": equilibrium_label,
                        "flags": flags,
                        "rye": rye_val,
                    },
                )
        except Exception:
            pass

        # Learning signals for downstream analysis and TGRM tuning
        cycle_data.setdefault(
            "learning_signals",
            {
                "delta_notes": cycle_data.get("delta_notes"),
                "delta_hypotheses": cycle_data.get("delta_hypotheses"),
                "tool_efficiency": cycle_data.get("tool_efficiency"),
                "verification_rigidity": (cycle_data.get("verification") or {}).get("rigidity")
                if isinstance(cycle_data.get("verification"), dict)
                else None,
                "novelty_pressure": (cycle_data.get("repair") or {}).get("novelty_pressure")
                if isinstance(cycle_data.get("repair"), dict)
                else None,
                "critic_strength": (cycle_data.get("detect") or {}).get("critic_strength")
                if isinstance(cycle_data.get("detect"), dict)
                else None,
                "burst_mode": cycle_data.get("burst_mode"),
            },
        )

        # Learning burst step: decrement cycles_remaining if active
        try:
            lb = self._data.get("learning_burst") or {}
            if isinstance(lb, dict) and lb.get("active"):
                remaining = lb.get("cycles_remaining")
                if isinstance(remaining, int):
                    if remaining > 1:
                        lb["cycles_remaining"] = remaining - 1
                    else:
                        lb["cycles_remaining"] = 0
                        lb["active"] = False
                lb["last_updated"] = _utc_now_iso()
                self._data["learning_burst"] = lb
        except Exception:
            pass

        self._save()

    def get_cycle_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the history of cycles (oldest to newest).

        Args:
            limit:
                If provided and > 0, returns only the most recent `limit` cycles,
                still ordered oldest->newest (stable for learning windows).
        """
        cycles = self._data.get("cycles", [])
        if not isinstance(cycles, list):
            return []
        if limit is None:
            return list(cycles)
        try:
            lim = int(limit)
        except Exception:
            lim = 0
        if lim <= 0:
            return []
        return list(cycles[-lim:])

    def get_cycle_history_for_goal(self, goal: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Return recent cycles for a goal, oldest to newest, up to limit.

        This helper is used by TGRM learning functions that need an
        ordered recent window for RYE gradient and equilibrium signals.
        """
        history = [c for c in self._data.get("cycles", []) if c.get("goal") == goal]
        history_sorted = sorted(
            history,
            key=lambda c: c.get("timestamp", ""),
            reverse=True,
        )
        # Take most recent limit and reverse so oldest is first
        window = list(reversed(history_sorted[:limit]))
        return window

    def get_cycles_for_run(
        self,
        run_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return cycles associated with a specific run_id.

        Results are sorted by timestamp from newest to oldest.
        If limit is provided, truncate to that many entries.
        """
        history = self._data.get("cycles", [])
        if not isinstance(history, list):
            return []
        filtered = [c for c in history if isinstance(c, dict) and c.get("run_id") == run_id]
        filtered_sorted = sorted(
            filtered,
            key=lambda c: c.get("timestamp", ""),
            reverse=True,
        )
        if limit is not None and limit >= 0:
            return filtered_sorted[:limit]
        return filtered_sorted

    def get_recent_cycles(
        self,
        goal: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return recent cycles, optionally filtered by goal.

        Args:
            goal:
                If provided, only cycles whose "goal" matches are returned.
            limit:
                Maximum number of cycles to return (most recent first).
        """
        history = self._data.get("cycles", [])
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]
        history_sorted = sorted(
            history,
            key=lambda c: c.get("timestamp", ""),
            reverse=True,
        )
        return history_sorted[:limit]

    # ------------------------------------------------------------------
    # Run state and checkpoint helpers
    # ------------------------------------------------------------------
    def save_run_state(
        self,
        state: Optional[Dict[str, Any]] = None,
        *,
        goal: Optional[str] = None,
        mode: Optional[str] = None,
        minutes_remaining: Optional[float] = None,
        last_cycle_index: Optional[int] = None,
        domain: Optional[str] = None,
        role: str = "agent",
        run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist high level run state for crash proof continuous mode.

        Two usage patterns are supported:

            1) New style (from CoreAgent.save_state_to_storage):
                save_run_state(state_dict)

               In this case, the given dict is stored as is, with an
               updated_at timestamp added if missing.

            2) Legacy style:
                save_run_state(
                    goal=...,
                    mode=...,
                    minutes_remaining=...,
                    last_cycle_index=...,
                    domain=...,
                    role=...,
                    run_id=...,
                    extra=...
                )

        In both cases, run_state is kept as a compact summary so the agent
        can resume a long run.
        """
        if state is not None and isinstance(state, dict) and not any(
            v is not None
            for v in (goal, mode, minutes_remaining, last_cycle_index, domain, extra, run_id)
        ):
            # New-style payload: merge into existing run_state so we do not lose
            # last_run_* fields that dashboards use while the worker is idle.
            payload = dict(state)
            now_iso = _utc_now_iso()
            now_ts = _utc_now_ts()
            payload.setdefault("updated_at", now_iso)
            payload.setdefault("utc", payload.get("updated_at") or now_iso)
            payload.setdefault("ts", float(now_ts))
            payload.setdefault("schema_version", RUNS_ARTIFACT_SCHEMA_VERSION)

            # Merge with any existing on-disk run_state (logs/ preferred)
            existing = self.read_run_state() or {}
            if not isinstance(existing, dict):
                existing = {}
            merged = dict(existing)
            merged.update(payload)

            self._data.setdefault("run_state", {})
            self._data["run_state"] = merged
            self._save()

            # Mirror to dedicated JSON files for diagnostics / dashboards
            file_payload = dict(merged)
            file_payload.setdefault("schema_version", RUNS_ARTIFACT_SCHEMA_VERSION)
            try:
                self._write_json_file(self.run_state_path, file_payload)
            except Exception:
                pass
            # Also mirror to legacy root-level path if configured
            try:
                legacy_path = getattr(self, "legacy_run_state_path", None)
                if (
                    isinstance(legacy_path, str)
                    and legacy_path
                    and legacy_path != self.run_state_path
                ):
                    self._write_json_file(legacy_path, file_payload)
            except Exception:
                pass
            return

        # Legacy keyword style
        if goal is None and isinstance(state, dict):
            goal = str(state.get("goal", "")) or None
        if mode is None and isinstance(state, dict):
            mode = str(state.get("mode", "")) or None
        if run_id is None and isinstance(state, dict):
            run_id = state.get("run_id")

        now_iso = _utc_now_iso()
        now_ts = _utc_now_ts()

        legacy_state: Dict[str, Any] = {
            "schema_version": RUNS_ARTIFACT_SCHEMA_VERSION,
            "updated_at": now_iso,
            "utc": now_iso,
            "ts": float(now_ts),
            "status": "running" if run_id else "idle",
            "active_run_id": run_id,
            "goal": goal,
            "mode": mode,
            "domain": domain,
            "role": role,
            "minutes_remaining": minutes_remaining,
            "last_cycle_index": last_cycle_index,
            "run_id": run_id,
        }
        if extra:
            legacy_state["extra"] = extra

        # Merge so we don't blow away last_run_* fields or queue diagnostics.
        existing = self.read_run_state() or {}
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(legacy_state)

        self._data.setdefault("run_state", {})
        self._data["run_state"] = merged
        self._save()

        # Mirror to dedicated JSON files for diagnostics / dashboards
        file_payload = dict(merged)
        file_payload.setdefault("schema_version", RUNS_ARTIFACT_SCHEMA_VERSION)
        try:
            self._write_json_file(self.run_state_path, file_payload)
        except Exception:
            pass
        # Also mirror to legacy root-level path if configured
        try:
            legacy_path = getattr(self, "legacy_run_state_path", None)
            if (
                isinstance(legacy_path, str)
                and legacy_path
                and legacy_path != self.run_state_path
            ):
                self._write_json_file(legacy_path, file_payload)
        except Exception:
            pass

    def load_run_state(self) -> Optional[Dict[str, Any]]:
        """Return the last saved run state, if any (from memory.json)."""
        state = self._data.get("run_state") or {}
        if not isinstance(state, dict) or not state:
            return None
        return dict(state)

    # New file oriented helpers for engine_worker and Streamlit
    def write_run_state(self, state: Dict[str, Any]) -> None:
        """Write run_state to both memory.json and run_state.json."""
        self.save_run_state(state)
        # save_run_state already mirrors to run_state_path

    def update_run_state(self, run_id: str, state: Dict[str, Any]) -> None:
        """Compatibility helper matching older worker expectations.

        Allows calls like update_run_state(run_id, state_dict) while still
        persisting the latest run_state into both memory.json and
        run_state.json.
        """
        if not isinstance(state, dict):
            state = {"raw_state": state}
        payload = dict(state)
        payload.setdefault("run_id", run_id)
        self.write_run_state(payload)

    def read_run_state(self) -> Optional[Dict[str, Any]]:
        """Read run_state from run_state.json files, falling back to memory.json.

        Prefers logs/run_state.json, but falls back to a legacy root-level
        run_state.json if present.
        """
        data = self._read_json_file(self.run_state_path)
        if not isinstance(data, dict) or not data:
            legacy_path = getattr(self, "legacy_run_state_path", None)
            if (
                isinstance(legacy_path, str)
                and legacy_path
                and legacy_path != self.run_state_path
            ):
                data = self._read_json_file(legacy_path)

        if isinstance(data, dict) and data:
            return data

        return self.load_run_state()

    def clear_run_state(self) -> None:
        """Clear saved run state metadata."""
        self._data["run_state"] = {}
        self._save()
        try:
            self._write_json_file(self.run_state_path, {})
        except Exception:
            pass
        # Also clear legacy root-level path if configured
        try:
            legacy_path = getattr(self, "legacy_run_state_path", None)
            if (
                isinstance(legacy_path, str)
                and legacy_path
                and legacy_path != self.run_state_path
            ):
                self._write_json_file(legacy_path, {})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Worker state (live status for engine_worker)
    # ------------------------------------------------------------------
    def update_worker_state(
        self,
        *,
        status: str,
        mode: str,
        goal: str,
        domain: Optional[str] = None,
        roles: Optional[List[str]] = None,
        runtime_profile: Optional[str] = None,
        stop_rye: Optional[float] = None,
        max_minutes: Optional[float] = None,
        max_cycles: Optional[int] = None,
        max_rounds: Optional[int] = None,
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record live worker status for monitoring and UI.

        Example statuses:
            - "starting"
            - "running"
            - "paused"
            - "stopped"
            - "error"
        """
        # Start from existing worker_state.json if it exists so we keep counters.
        # Prefer logs/worker_state.json, but fall back to legacy root-level
        # worker_state.json if present.
        state: Dict[str, Any]
        file_state = self._read_json_file(self.worker_state_path)
        if not isinstance(file_state, dict) or not file_state:
            legacy_path = getattr(self, "legacy_worker_state_path", None)
            if (
                isinstance(legacy_path, str)
                and legacy_path
                and legacy_path != self.worker_state_path
            ):
                file_state = self._read_json_file(legacy_path)

        if isinstance(file_state, dict) and file_state:
            state = dict(file_state)
        else:
            ws = self._data.get("worker_state") or {}
            state = dict(ws) if isinstance(ws, dict) else {}

        # Compute heartbeat info
        now_iso = _utc_now_iso()
        now_ts = _utc_now_ts()
        last_beat = state.get("last_beat") or state.get("last_heartbeat_utc")
        seconds_since_last: Optional[float] = None
        if isinstance(last_beat, str):
            try:
                dt = datetime.fromisoformat(last_beat.replace("Z", "+00:00"))
                seconds_since_last = (datetime.now(timezone.utc) - dt).total_seconds()
            except Exception:
                seconds_since_last = None
        elif isinstance(last_beat, (int, float)):
            try:
                seconds_since_last = float(now_ts) - float(last_beat)
            except Exception:
                seconds_since_last = None

        heartbeat_count = int(state.get("heartbeat_count", 0) or 0) + 1

        # Update core fields
        state.update(
            {
                "schema_version": RUNS_ARTIFACT_SCHEMA_VERSION,
                "utc": now_iso,
                "ts": float(now_ts),
                "updated_at": now_iso,
                "last_beat": now_iso,
                "last_heartbeat_utc": now_iso,
                "seconds_since_last_beat": seconds_since_last,
                "heartbeat_count": heartbeat_count,
                "status": status,
                "mode": mode,
                "goal": goal,
                "domain": domain,
                "roles": roles or [],
                "runtime_profile": runtime_profile,
                "stop_rye": stop_rye,
                "max_minutes": max_minutes,
                "max_cycles": _to_int(max_cycles),
                "max_rounds": _to_int(max_rounds),
                "run_id": run_id,
                "active_run_id": run_id,
                "experiment_mode": experiment_mode,
                "current": _to_int(current),
                "total": _to_int(total),
            }
        )

        if extra:
            existing_extra = state.get("extra") or {}
            if not isinstance(existing_extra, dict):
                existing_extra = {}
            existing_extra.update(extra)
            state["extra"] = existing_extra

        # Persist in-memory and to the small worker_state.json file
        self._data["worker_state"] = state
        self._save()
        try:
            self._write_json_file(self.worker_state_path, dict(state))
        except Exception:
            pass
        # Mirror to legacy root-level worker_state.json for older readers
        try:
            legacy_path = getattr(self, "legacy_worker_state_path", None)
            if (
                isinstance(legacy_path, str)
                and legacy_path
                and legacy_path != self.worker_state_path
            ):
                self._write_json_file(legacy_path, dict(state))
        except Exception:
            pass

    def get_worker_state(self) -> Optional[Dict[str, Any]]:
        """Return current worker_state snapshot.

        Prefers logs/worker_state.json, but falls back to a legacy root-level
        worker_state.json if present, then to in-memory data from memory.json.
        """
        data = self._read_json_file(self.worker_state_path)
        if not isinstance(data, dict) or not data:
            legacy_path = getattr(self, "legacy_worker_state_path", None)
            if (
                isinstance(legacy_path, str)
                and legacy_path
                and legacy_path != self.worker_state_path
            ):
                data = self._read_json_file(legacy_path)

        if isinstance(data, dict) and data:
            return data

        ws = self._data.get("worker_state") or {}
        if not isinstance(ws, dict) or not ws:
            return None
        return dict(ws)

    def read_worker_state(self) -> Optional[Dict[str, Any]]:
        """Compatibility alias for get_worker_state."""
        return self.get_worker_state()

    def clear_worker_state(self) -> None:
        """Clear worker state in memory.json and worker_state.json files."""
        self._data["worker_state"] = {}
        self._save()
        try:
            self._write_json_file(self.worker_state_path, {})
        except Exception:
            pass
        # Also clear legacy root-level path if configured
        try:
            legacy_path = getattr(self, "legacy_worker_state_path", None)
            if (
                isinstance(legacy_path, str)
                and legacy_path
                and legacy_path != self.worker_state_path
            ):
                self._write_json_file(legacy_path, {})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Watchdog heartbeats for long runs
    # ------------------------------------------------------------------
    def _write_flat_watchdog_heartbeat(
        self,
        *,
        label: str,
        run_id: Optional[str],
        now_iso: str,
        now_ts: float,
    ) -> None:
        """Write a flat watchdog heartbeat file for UIs that read it directly.

        The UI panel shown in the screenshots expects a single JSON document
        (not per-label) and commonly reads:
            - last_beat (unix timestamp float)
            - heartbeat_count (int)
            - seconds_since_last_beat (optional; may be displayed as 'n/a' if missing)

        This method:
            - loads existing heartbeat_count if present (persisting across restarts)
            - increments it
            - writes a compact payload to logs/watchdog_heartbeat.json
        """
        prev = self._read_json_file(self.watchdog_heartbeat_path) or {}
        prev_count = prev.get("heartbeat_count")
        if not isinstance(prev_count, int):
            try:
                prev_count = int(float(prev.get("count") if prev.get("count") is not None else prev_count or 0))
            except Exception:
                prev_count = 0

        prev_last = prev.get("last_beat")
        seconds_since_prev: Optional[float] = None
        if isinstance(prev_last, (int, float, str)):
            try:
                seconds_since_prev = float(now_ts) - float(prev_last)
                if seconds_since_prev < 0:
                    seconds_since_prev = None
            except Exception:
                seconds_since_prev = None

        count = int(prev_count) + 1

        payload: Dict[str, Any] = {
            "schema_version": RUNS_ARTIFACT_SCHEMA_VERSION,
            "utc": now_iso,
            # Legacy naming (kept for dashboards that show these fields)
            "last_update_utc": now_iso,
            "last_heartbeat_utc": now_iso,
            "ts": float(now_ts),
            "last_beat": float(now_ts),
            # Counter fields (UI may look for either key)
            "heartbeat_count": int(count),
            "count": int(count),  # compatibility alias
            "seconds_since_last_beat": seconds_since_prev,
            # Simple status hints for UIs
            "status": "running" if run_id else "idle",
            "run_id": run_id,
            "active_run_id": run_id,
            "label": label,
        }

        try:
            self._write_json_file(self.watchdog_heartbeat_path, payload)
        except Exception:
            pass

    def heartbeat(self, label: str = "continuous_run", run_id: Optional[str] = None) -> None:
        """Record a watchdog heartbeat.

        The agent can call this periodically so that:
            - you can see if it is still alive
            - a supervising process can measure downtime

        To avoid heavy disk writes, this updates watchdog in memory and
        writes only to the small watchdog.json file, not the main memory.json.

        Additionally, a flat logs/watchdog_heartbeat.json file is updated for
        UIs that read a single heartbeat file directly.
        """
        wd = self._data.setdefault("watchdog", {})
        now = _utc_now_iso()
        now_ts = _utc_now_ts()
        existing = (self._read_json_file(self.watchdog_path) or {}).get(label) or wd.get(label)
        if isinstance(existing, dict):
            count = int(_to_int(existing.get("count") or existing.get("heartbeat_count")) or 0) + 1
            prev_run_id = existing.get("run_id")
        else:
            count = 1
            prev_run_id = None
        wd[label] = {
            "last_beat": now, "last_beat_ts": float(now_ts),
            "last_heartbeat_utc": now,  # extra field for diagnostics readers
            "count": count, "heartbeat_count": count,
            "run_id": run_id or prev_run_id,
        }

        # Mirror watchdog dict to a dedicated JSON file without rewriting memory.json
        try:
            payload = self._read_json_file(self.watchdog_path) or {}
            payload = payload if isinstance(payload, dict) else {}; payload[label] = dict(wd.get(label) or {}); self._write_json_file(self.watchdog_path, payload)
        except Exception:
            pass

        # Also write the flat heartbeat file commonly used by Streamlit dashboards
        self._write_flat_watchdog_heartbeat(
            label=label,
            run_id=run_id or prev_run_id,
            now_iso=now,
            now_ts=now_ts,
        )

    def record_heartbeat(
        self,
        *,
        label: str = "continuous_run",
        run_id: Optional[str] = None,
    ) -> None:
        """Thin public alias for heartbeat for high frequency callers."""
        self.heartbeat(label=label, run_id=run_id)

    def get_watchdog_info(self, label: str = "continuous_run") -> Dict[str, Any]:
        """Return watchdog data for a label.

        Prefers watchdog.json so separate processes see live heartbeats,
        but falls back to in memory data if the file is missing.

        If the requested label is not found and the label is the common
        'continuous_run', fall back to reading logs/watchdog_heartbeat.json
        (flat heartbeat) and map its fields into the same return shape.
        """
        entry: Dict[str, Any]

        data = self._read_json_file(self.watchdog_path)
        if isinstance(data, dict) and data:
            entry = data.get(label) or {}
        else:
            wd = self._data.get("watchdog") or {}
            if not isinstance(wd, dict):
                wd = {}
            entry = wd.get(label) or {}

        # Fallback: if label missing and user asked for the common heartbeat label,
        # attempt to read the flat heartbeat file used by dashboards.
        if (not entry) and label in ("continuous_run", "watchdog", "heartbeat"):
            flat = self._read_json_file(self.watchdog_heartbeat_path) or {}
            if isinstance(flat, dict) and flat:
                entry = {
                    "last_beat": flat.get("last_update_utc") or flat.get("last_heartbeat_utc"),
                    "count": flat.get("heartbeat_count") if flat.get("heartbeat_count") is not None else flat.get("count"),
                    "run_id": flat.get("run_id"),
                    "_flat_last_beat_ts": flat.get("last_beat") or flat.get("ts"),
                    "_flat_seconds_since_last_beat": flat.get("seconds_since_last_beat"),
                }

        last_beat = entry.get("last_beat") or entry.get("last_heartbeat_utc")
        count = _to_int(entry.get("count") or entry.get("heartbeat_count")) or 0
        run_id = entry.get("run_id")

        seconds_since_last: Optional[float] = None

        # If we have the flat heartbeat timestamp in seconds, prefer it
        flat_ts = entry.get("_flat_last_beat_ts")
        if isinstance(flat_ts, (int, float)):
            try:
                seconds_since_last = float(_utc_now_ts()) - float(flat_ts)
            except Exception:
                seconds_since_last = None
        else:
            # Otherwise parse ISO timestamps if possible
            if isinstance(last_beat, str):
                try:
                    dt = datetime.fromisoformat(last_beat.replace("Z", "+00:00"))
                    seconds_since_last = (datetime.now(timezone.utc) - dt).total_seconds()
                except Exception:
                    seconds_since_last = None
            elif isinstance(last_beat, (int, float)):
                try:
                    seconds_since_last = float(_utc_now_ts()) - float(last_beat)
                except Exception:
                    seconds_since_last = None

        # If the flat file explicitly provides seconds_since_last_beat, keep it
        flat_seconds = entry.get("_flat_seconds_since_last_beat")
        if isinstance(flat_seconds, (int, float)):
            seconds_since_last = float(flat_seconds)

        return {
            "last_beat": last_beat,
            "count": count, "heartbeat_count": count,
            "seconds_since_last": seconds_since_last, "seconds_since_last_beat": seconds_since_last,
            "run_id": run_id,
        }

    def get_heartbeat_stats(self, label: str = "continuous_run") -> Dict[str, Any]:
        """Public alias for get_watchdog_info used by workers and UIs."""
        return self.get_watchdog_info(label=label)

    # File oriented helpers for watchdog
    def write_watchdog_heartbeat(self, label: str = "continuous_run", run_id: Optional[str] = None) -> None:
        """Wrapper so workers can call a named method for heartbeats."""
        self.heartbeat(label=label, run_id=run_id)

    def read_watchdog_status(self, label: str = "continuous_run") -> Dict[str, Any]:
        """Compatibility wrapper that uses the same logic as get_watchdog_info."""
        return self.get_watchdog_info(label=label)

    # ------------------------------------------------------------------
    # Events: streaming log for worker and UI
    # ------------------------------------------------------------------
    def add_event(
        self,
        *,
        kind: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "info",
        goal: Optional[str] = None,
        role: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Append a generic event to the streaming log.

        Examples:
            kind = "worker_status", "partial_cycle", "ui_action"
            level = "info", "warning", "error"
        """
        ev: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "kind": kind,
            "level": level,
            "message": message,
            "payload": payload or {},
        }
        if goal is not None:
            ev["goal"] = goal
        if role is not None:
            ev["role"] = role
        if run_id is not None:
            ev["run_id"] = run_id

        self._data.setdefault("events", []).append(ev)

        # Keep events bounded in size for long runs
        if len(self._data["events"]) > MAX_EVENTS:
            self._data["events"] = self._data["events"][-MAX_EVENTS:]

        self._save()

    def get_events(
        self,
        *,
        limit: int = 200,
        kind: Optional[str] = None,
        level: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent events filtered by kind, level, and run_id."""
        events = self._data.get("events", [])
        if not isinstance(events, list):
            return []

        filtered: List[Dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if kind is not None and ev.get("kind") != kind:
                continue
            if level is not None and ev.get("level") != level:
                continue
            if run_id is not None and ev.get("run_id") != run_id:
                continue
            filtered.append(ev)

        filtered_sorted = sorted(
            filtered,
            key=lambda e: e.get("timestamp", ""),
            reverse=True,
        )
        return filtered_sorted[:limit]

    def record_run_heartbeat(
        self,
        *,
        run_id: str,
        goal: Optional[str] = None,
        status: str = "running",
        mode: Optional[str] = None,
        cycle_index: Optional[int] = None,
        note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience helper to log a per run heartbeat event and watchdog beat.

        Designed for engine_worker so each cycle or round can emit:
            - a streaming event for the UI
            - watchdog beats that can be inspected by dashboards
        """
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "status": status,
            "mode": mode,
            "cycle_index": cycle_index,
        }
        if note is not None:
            payload["note"] = note
        if extra:
            payload.update(extra)

        # Event for UI
        self.add_event(
            kind="run_heartbeat",
            message=f"Run heartbeat status={status}",
            payload=payload,
            level="info",
            goal=goal,
            run_id=run_id,
        )

        # Watchdog beats for this run and the global label used by most UIs
        try:
            if run_id:
                run_label = f"run:{run_id}"
                self.heartbeat(label=run_label, run_id=run_id)
            self.heartbeat(label="continuous_run", run_id=run_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Discoveries: cure, treatment, mechanism, etc.
    # ------------------------------------------------------------------
    def add_discovery(
        self,
        *,
        goal: str,
        kind: str,
        label: str,
        evidence_summary: str,
        score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        equilibrium_label: Optional[str] = None,
        breakthrough_score: Optional[float] = None,
        verification_status: Optional[str] = None,
        priority_rank: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a key discovery, such as a treatment or mechanism candidate.

        Example kinds:
            - "treatment"
            - "cure_candidate"
            - "mechanism"
            - "biomarker"
        """
        entry: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "kind": kind,
            "label": label,
            "evidence_summary": evidence_summary,
            "score": float(score) if score is not None else None,
            "tags": tags or [],
            "citations": citations or [],
        }
        if domain is not None:
            entry["domain"] = domain
        if run_id is not None:
            entry["run_id"] = run_id
        if equilibrium_label is not None:
            entry["equilibrium_label"] = equilibrium_label
        if isinstance(breakthrough_score, (int, float)):
            entry["breakthrough_score"] = float(breakthrough_score)
        if verification_status is not None:
            entry["verification_status"] = verification_status
        if priority_rank is not None:
            entry["priority_rank"] = int(priority_rank)
        if extra:
            entry["extra"] = dict(extra)

        self._data.setdefault("discoveries", []).append(entry)

        # Keep discoveries bounded
        if len(self._data["discoveries"]) > MAX_DISCOVERIES:
            self._data["discoveries"] = self._data["discoveries"][-MAX_DISCOVERIES:]

        self._save()

        # Also push into vector memory to make them easy to retrieve semantically
        if self.vector_memory is not None:
            try:
                text_parts = [label, evidence_summary]
                text = " ".join([p for p in text_parts if p])
                if text:
                    meta: Dict[str, Any] = {
                        "goal": goal,
                        "type": "discovery",
                        "kind": kind,
                        "score": entry["score"],
                        "tags": entry["tags"],
                    }
                    if domain is not None:
                        meta["domain"] = domain
                    if run_id is not None:
                        meta["run_id"] = run_id
                    self.vector_memory.add_item(text=text, metadata=meta)
            except Exception:
                pass

    def get_discoveries(
        self,
        goal: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve discovery entries, optionally filtered by goal and kind."""
        entries = self._data.get("discoveries", [])
        if not isinstance(entries, list):
            return []

        result: List[Dict[str, Any]] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            if goal is not None and e.get("goal") != goal:
                continue
            if kind is not None and e.get("kind") != kind:
                continue
            result.append(e)
        return result

    def get_recent_discoveries(
        self,
        limit: int = 50,
        goal: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent discoveries for dashboards and reports."""
        entries = self.get_discoveries(goal=goal)
        entries_sorted = sorted(
            entries,
            key=lambda d: d.get("timestamp", ""),
            reverse=True,
        )
        return entries_sorted[:limit]

    def get_discoveries_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Return discoveries associated with a specific run_id."""
        entries = self._data.get("discoveries", [])
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict) and e.get("run_id") == run_id]

    # ------------------------------------------------------------------
    # Run manifests for long runs
    # ------------------------------------------------------------------
    def log_run_manifest(self, *args: Any, **kwargs: Any) -> None:
        """Store a compact manifest for a finished or in progress run.

        Supports both:
            log_run_manifest(run_id, manifest_dict)
        and:
            log_run_manifest(manifest_dict)

        The manifest is stored under run_manifests[run_id] with a
        logged_at timestamp added if missing. A bounded number of
        manifests are kept.
        """
        run_id: Optional[str] = None
        manifest: Optional[Dict[str, Any]] = None

        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            run_id = args[0]
            manifest = dict(args[1])
        elif len(args) == 1 and isinstance(args[0], dict):
            manifest = dict(args[0])
            run_id = manifest.get("run_id")
        else:
            run_id = kwargs.get("run_id")
            manifest_arg = kwargs.get("manifest")
            if isinstance(manifest_arg, dict):
                manifest = dict(manifest_arg)

        if manifest is None:
            return
        if not run_id:
            run_id = manifest.get("run_id")
        if not run_id:
            return

        manifest.setdefault("run_id", run_id)
        manifest.setdefault("logged_at", _utc_now_iso())

        rm = self._data.setdefault("run_manifests", {})
        if not isinstance(rm, dict):
            rm = {}
        rm[run_id] = manifest

        # Bound number of manifests
        if len(rm) > MAX_RUN_MANIFESTS:
            items = sorted(
                rm.items(),
                key=lambda kv: str(kv[1].get("logged_at", "")),
            )
            excess = len(items) - MAX_RUN_MANIFESTS
            for k, _v in items[:excess]:
                rm.pop(k, None)

        self._data["run_manifests"] = rm
        self._save()

    def get_run_manifest(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a run manifest by run_id."""
        rm = self._data.get("run_manifests") or {}
        if not isinstance(rm, dict):
            return None
        manifest = rm.get(run_id)
        if not isinstance(manifest, dict):
            return None
        return dict(manifest)

    def list_run_manifests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return a list of recent run manifests sorted by logged_at."""
        rm = self._data.get("run_manifests") or {}
        if not isinstance(rm, dict):
            return []
        items = list(rm.values())
        items_sorted = sorted(
            [m for m in items if isinstance(m, dict)],
            key=lambda m: str(m.get("logged_at", "")),
            reverse=True,
        )
        return items_sorted[:limit]

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Return a compact summary for a specific run_id.

        This is designed for UI tables and meta controllers that need
        quick counts and RYE stats for a run without scanning the whole
        file externally.
        """
        cycles = self.get_cycles_for_run(run_id)
        discoveries = self.get_discoveries_for_run(run_id)
        citations = self.get_citations_for_run(run_id)
        benchmarks = self.get_benchmark_results(run_id=run_id)

        timestamps: List[str] = []
        for c in cycles:
            ts = c.get("timestamp")
            if isinstance(ts, str):
                timestamps.append(ts)
        started_at = min(timestamps) if timestamps else None
        finished_at = max(timestamps) if timestamps else None

        rye_values: List[float] = []
        for c in cycles:
            v = c.get("RYE")
            if isinstance(v, (int, float)):
                rye_values.append(float(v))

        avg_rye: Optional[float] = None
        min_rye: Optional[float] = None
        max_rye: Optional[float] = None
        if rye_values:
            avg_rye = sum(rye_values) / float(len(rye_values))
            min_rye = min(rye_values)
            max_rye = max(rye_values)

        best_cycle: Optional[Dict[str, Any]] = None
        best_rye_val: Optional[float] = None
        for c in cycles:
            v = c.get("RYE")
            if isinstance(v, (int, float)):
                v_float = float(v)
                if best_rye_val is None or v_float > best_rye_val:
                    best_rye_val = v_float
                    best_cycle = c

        goals = sorted(
            {
                str(c.get("goal"))
                for c in cycles
                if isinstance(c.get("goal"), str)
            }
        )

        summary: Dict[str, Any] = {
            "run_id": run_id,
            "goals": goals,
            "counts": {
                "cycles": len(cycles),
                "discoveries": len(discoveries),
                "citations": len(citations),
                "benchmarks": len(benchmarks),
            },
            "rye_basic": {
                "avg": avg_rye,
                "min": min_rye,
                "max": max_rye,
                "count": len(rye_values),
            },
            "timeline": {
                "started_at": started_at,
                "finished_at": finished_at,
            },
            "best_cycle": {
                "cycle_index": (
                    best_cycle.get("cycle_index", best_cycle.get("cycle")) if isinstance(best_cycle, dict) else None
                ),
                "RYE": best_rye_val,
                "timestamp": best_cycle.get("timestamp") if isinstance(best_cycle, dict) else None,
            }
            if best_cycle is not None
            else {},
        }
        return summary

    # ------------------------------------------------------------------
    # Tool events and per tool stats
    # ------------------------------------------------------------------
    def log_tool_event(
        self,
        *,
        run_id: Optional[str],
        goal: Optional[str],
        domain: Optional[str],
        role: Optional[str],
        cycle_index: Optional[int],
        tool_name: str,
        status: str,
        duration_seconds: Optional[float] = None,
        energy_cost: Optional[float] = None,
        rye_delta: Optional[float] = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single tool usage event for diagnostics and tool RYE."""
        ev: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "run_id": run_id,
            "goal": goal,
            "domain": domain,
            "role": role,
            "cycle_index": cycle_index,
            "tool_name": tool_name,
            "status": status,
            "duration_seconds": duration_seconds,
            "energy_cost": energy_cost,
            "rye_delta": rye_delta,
        }
        if error is not None:
            ev["error"] = error
        if extra:
            ev["extra"] = extra

        self._data.setdefault("tool_events", []).append(ev)
        if len(self._data["tool_events"]) > MAX_TOOL_EVENTS:
            self._data["tool_events"] = self._data["tool_events"][-MAX_TOOL_EVENTS:]

        self._save()

    def get_tool_stats(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate basic per tool stats, optionally filtered by run_id.

        Returns a dict:
            {
              "tools": {
                 "browser": {
                     "calls": int,
                     "errors": int,
                     "last_called_at": "iso",
                     "last_error": "message or None"
                 },
                 "sandbox": { ... },
                 ...
              },
              "tool_rye": { ... }  # if provided by rye_metrics
            }
        """
        events = self._data.get("tool_events", [])
        if not isinstance(events, list):
            events = []

        stats: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if run_id is not None and ev.get("run_id") != run_id:
                continue
            name = ev.get("tool_name")
            if not name:
                continue
            st = stats.get(name) or {
                "calls": 0,
                "errors": 0,
                "last_called_at": None,
                "last_error": None,
            }
            st["calls"] = int(st.get("calls", 0)) + 1
            if ev.get("status") == "error":
                st["errors"] = int(st.get("errors", 0)) + 1
                last_error = ev.get("error")
                if last_error:
                    st["last_error"] = last_error
            ts = ev.get("timestamp")
            if isinstance(ts, str):
                prev_ts = st.get("last_called_at")
                if prev_ts is None or str(ts) > str(prev_ts):
                    st["last_called_at"] = ts
            stats[name] = st

        result: Dict[str, Any] = {"tools": stats}

        # Optional per tool RYE and learning metrics if rye_metrics provides it
        if _rye_metrics is not None and hasattr(_rye_metrics, "compute_tool_rye"):
            try:
                result["tool_rye"] = _rye_metrics.compute_tool_rye(
                    events,
                    run_id=run_id,
                )  # type: ignore[attr-defined]
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Option C diagnostics (deep telemetry for frontier runs)
    # ------------------------------------------------------------------
    def log_option_c_diagnostics(self, run_id: str, diagnostics: Dict[str, Any]) -> None:
        """Log advanced diagnostics for AGI style Option C runs."""
        entry = {
            "timestamp": _utc_now_iso(),
            "run_id": run_id,
            "diagnostics": diagnostics,
        }
        arr = self._data.setdefault("option_c_diagnostics", [])
        arr.append(entry)
        if len(arr) > MAX_OPTION_C_DIAGNOSTICS:
            self._data["option_c_diagnostics"] = arr[-MAX_OPTION_C_DIAGNOSTICS:]
        self._save()

    def get_option_c_diagnostics(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve Option C diagnostics, optionally filtered by run_id."""
        arr = self._data.get("option_c_diagnostics", [])
        if run_id is None:
            return list(arr)
        return [d for d in arr if d.get("run_id") == run_id]

    # ------------------------------------------------------------------
    # Swarm contracts (per role negotiation and specialization)
    # ------------------------------------------------------------------
    def log_swarm_contract(self, run_id: str, role: str, contract: Dict[str, Any]) -> None:
        """Record a swarm contract for a specific logical role."""
        entry = {
            "timestamp": _utc_now_iso(),
            "run_id": run_id,
            "role": role,
            "contract": contract,
        }
        arr = self._data.setdefault("swarm_contracts", [])
        arr.append(entry)
        if len(arr) > MAX_SWARM_CONTRACTS:
            self._data["swarm_contracts"] = arr[-MAX_SWARM_CONTRACTS:]
        self._save()

    def get_swarm_contracts(
        self,
        run_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve swarm contracts, optionally filtered by run_id and role."""
        arr = self._data.get("swarm_contracts", [])
        results: List[Dict[str, Any]] = []
        for c in arr:
            if not isinstance(c, dict):
                continue
            if run_id is not None and c.get("run_id") != run_id:
                continue
            if role is not None and c.get("role") != role:
                continue
            results.append(c)
        return results

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------
    def log_milestone(
        self,
        *,
        run_id: Optional[str],
        goal: Optional[str],
        domain: Optional[str],
        label: str,
        description: str,
        level: str = "info",
        role: Optional[str] = None,
        cycle_index: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a key milestone for a run, such as best RYE or phase shift."""
        entry: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "run_id": run_id,
            "goal": goal,
            "domain": domain,
            "label": label,
            "description": description,
            "level": level,
            "role": role,
            "cycle_index": cycle_index,
        }
        if extra:
            entry["extra"] = extra

        self._data.setdefault("milestones", []).append(entry)
        if len(self._data["milestones"]) > MAX_MILESTONES:
            self._data["milestones"] = self._data["milestones"][-MAX_MILESTONES:]

        self._save()

    def get_milestones(
        self,
        *,
        run_id: Optional[str] = None,
        goal: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Retrieve milestones filtered by run_id and goal, most recent first."""
        ms = self._data.get("milestones", [])
        if not isinstance(ms, list):
            return []

        filtered: List[Dict[str, Any]] = []
        for m in ms:
            if not isinstance(m, dict):
                continue
            if run_id is not None and m.get("run_id") != run_id:
                continue
            if goal is not None and m.get("goal") != goal:
                continue
            filtered.append(m)

        filtered_sorted = sorted(
            filtered,
            key=lambda m: m.get("timestamp", ""),
            reverse=True,
        )
        return filtered_sorted[:limit]

    # ------------------------------------------------------------------
    # Learning burst helpers
    # ------------------------------------------------------------------
    def start_learning_burst(
        self,
        *,
        cycles: Optional[int] = None,
        burst_index: Optional[int] = None,
        goal: Optional[str] = None,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        """Activate a learning burst window for TGRM tuning.

        This is a small state flag that other components can read to run
        more intense repair cycles or different presets for a short period.
        """
        now = _utc_now_iso()
        lb = {
            "active": True,
            "cycles_remaining": cycles,
            "burst_index": burst_index,
            "goal": goal,
            "role": role,
            "domain": domain,
            "mode": mode,
            "started_at": now,
            "last_updated": now,
        }
        self._data["learning_burst"] = lb
        self._save()

    def stop_learning_burst(self) -> None:
        """Deactivate any active learning burst."""
        lb = self._data.get("learning_burst") or {}
        if not isinstance(lb, dict):
            lb = {}
        lb["active"] = False
        lb["cycles_remaining"] = 0
        lb["last_updated"] = _utc_now_iso()
        self._data["learning_burst"] = lb
        self._save()

    def update_learning_burst(
        self,
        *,
        cycles_remaining: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the learning_burst state without recreating it."""
        lb = self._data.get("learning_burst") or {}
        if not isinstance(lb, dict):
            lb = {"active": False, "cycles_remaining": None, "burst_index": None}
        if cycles_remaining is not None:
            lb["cycles_remaining"] = cycles_remaining
        if extra:
            payload = dict(lb.get("extra", {}))
            payload.update(extra)
            lb["extra"] = payload
        lb["last_updated"] = _utc_now_iso()
        self._data["learning_burst"] = lb
        self._save()

    def get_learning_burst_state(self) -> Dict[str, Any]:
        """Return the current learning_burst state."""
        lb = self._data.get("learning_burst") or {}
        if not isinstance(lb, dict):
            lb = {}
        return dict(lb)

    # ------------------------------------------------------------------
    # Reporting helpers (for long autonomous runs)
    # ------------------------------------------------------------------
    def get_rye_stats(
        self,
        goal: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        """Compute basic RYE statistics.

        If goal is provided, restrict to that goal.
        If role is also provided, restrict to that logical role.

        Returns:
            (avg_rye, min_rye, max_rye, count)
        """
        # Fast path: if goal is provided and no role filter, try goal_index
        if goal is not None and role is None:
            gi = self._data.get("goal_index") or {}
            if isinstance(gi, dict):
                entry = gi.get(goal)
                if isinstance(entry, dict):
                    avg_val = entry.get("avg_rye")
                    min_val = entry.get("min_rye")
                    max_val = entry.get("max_rye")
                    count_val = entry.get("rye_count")
                    if isinstance(avg_val, (int, float)) and isinstance(count_val, int) and count_val > 0:
                        return (
                            float(avg_val),
                            float(min_val) if isinstance(min_val, (int, float)) else None,
                            float(max_val) if isinstance(max_val, (int, float)) else None,
                            int(count_val),
                        )

        # Fallback: compute from cycles directly
        history = self._data.get("cycles", [])
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]
        if role is not None:
            history = [c for c in history if c.get("role") == role]

        values: List[float] = []
        for c in history:
            v = c.get("RYE")
            if isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            return None, None, None, 0

        avg_rye = sum(values) / len(values)
        min_rye = min(values)
        max_rye = max(values)
        return avg_rye, min_rye, max_rye, len(values)

    def get_advanced_rye_metrics(
        self,
        goal: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """Return advanced RYE metrics for a goal or role using rye_metrics.

        Keys:
            rolling_rye_10
            rolling_rye_50
            median_rye
            efficiency_trend
            regression_slope
            stability_index
            recovery_momentum
        """
        metrics: Dict[str, Optional[float]] = {
            "rolling_rye_10": None,
            "rolling_rye_50": None,
            "median_rye": None,
            "efficiency_trend": None,
            "regression_slope": None,
            "stability_index": None,
            "recovery_momentum": None,
        }

        if _rye_metrics is None:
            return metrics

        history = self._data.get("cycles", [])
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]
        if role is not None:
            history = [c for c in history if c.get("role") == role]

        if not history:
            return metrics

        try:
            metrics["rolling_rye_10"] = _rye_metrics.rolling_rye(history, window=10)
            metrics["rolling_rye_50"] = _rye_metrics.rolling_rye(history, window=50)
            metrics["median_rye"] = _rye_metrics.median_rye(history)
            metrics["efficiency_trend"] = _rye_metrics.efficiency_trend(history)
            metrics["regression_slope"] = _rye_metrics.regression_rye_slope(history)
            metrics["stability_index"] = _rye_metrics.stability_index(history)
            metrics["recovery_momentum"] = _rye_metrics.recovery_momentum(history)
        except Exception:
            pass

        return metrics

    def get_learning_profile(
        self,
        goal: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a learning focused profile for a goal and optional role.

        Includes:
            - counts and basic RYE stats
            - advanced RYE metrics
            - early vs late RYE averages
            - best cycle summary
            - last few equilibrium labels and breakthrough scores
        """
        history = self._data.get("cycles", [])
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]
        if role is not None:
            history = [c for c in history if c.get("role") == role]

        total_cycles = len(history)
        avg_rye, min_rye, max_rye, rye_count = self.get_rye_stats(goal=goal, role=role)
        adv = self.get_advanced_rye_metrics(goal=goal, role=role)

        # Early vs late RYE averages for learning curve shape
        early_avg = None
        late_avg = None
        if history:

            def _cycle_key(c: Dict[str, Any]) -> int:
                try:
                    if c.get("cycle_index") is not None:
                        return int(c.get("cycle_index"))
                    return int(c.get("cycle", 0))
                except Exception:
                    return 0

            sorted_hist = sorted(history, key=_cycle_key)
            n = len(sorted_hist)
            k = max(1, n // 4)
            early_slice = sorted_hist[:k]
            late_slice = sorted_hist[-k:]

            def _avg_rye_slice(slice_hist: List[Dict[str, Any]]) -> Optional[float]:
                vals: List[float] = []
                for c in slice_hist:
                    v = c.get("RYE")
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                if not vals:
                    return None
                return sum(vals) / len(vals)

            early_avg = _avg_rye_slice(early_slice)
            late_avg = _avg_rye_slice(late_slice)

        # Best cycle by RYE
        best_cycle: Optional[Dict[str, Any]] = None
        best_rye_val: Optional[float] = None
        for c in history:
            v = c.get("RYE")
            if isinstance(v, (int, float)):
                v_float = float(v)
                if best_rye_val is None or v_float > best_rye_val:
                    best_rye_val = v_float
                    best_cycle = c

        # Recent phase info
        recent_eq_labels: List[str] = []
        recent_breakthrough_scores: List[float] = []
        for c in sorted(history, key=lambda c: c.get("timestamp", ""), reverse=True)[:50]:
            eq = c.get("equilibrium") or {}
            if isinstance(eq, dict):
                lbl = eq.get("equilibrium_label")
                if isinstance(lbl, str):
                    recent_eq_labels.append(lbl)
            br = c.get("breakthrough") or {}
            if isinstance(br, dict):
                bs = br.get("breakthrough_score")
                if isinstance(bs, (int, float)):
                    recent_breakthrough_scores.append(float(bs))

        profile: Dict[str, Any] = {
            "scope": {
                "goal": goal,
                "role": role,
            },
            "counts": {
                "cycles": total_cycles,
                "rye_count": rye_count,
            },
            "rye_basic": {
                "avg": avg_rye,
                "min": min_rye,
                "max": max_rye,
            },
            "rye_advanced": adv,
            "rye_curve": {
                "early_avg": early_avg,
                "late_avg": late_avg,
                "delta": (late_avg - early_avg) if (early_avg is not None and late_avg is not None) else None,
            },
            "best_cycle": {
                "cycle_index": (
                    best_cycle.get("cycle_index", best_cycle.get("cycle")) if isinstance(best_cycle, dict) else None
                ),
                "RYE": best_rye_val,
                "timestamp": best_cycle.get("timestamp") if isinstance(best_cycle, dict) else None,
                "equilibrium": best_cycle.get("equilibrium") if isinstance(best_cycle, dict) else None,
                "breakthrough": best_cycle.get("breakthrough") if isinstance(best_cycle, dict) else None,
            }
            if best_cycle is not None
            else {},
            "recent_phase": {
                "equilibrium_labels": recent_eq_labels,
                "breakthrough_scores": recent_breakthrough_scores,
            },
        }
        return profile

    def build_text_report(self, goal: Optional[str] = None) -> str:
        """Build a simple text or markdown report for a goal or all goals."""
        if goal:
            title = f"Autonomous Research Report\nGoal: {goal}\n"
        else:
            title = "Autonomous Research Report\n(All goals)\n"

        title += "=" * 40 + "\n\n"

        history = self.get_cycle_history()
        if goal is not None:
            history = [c for c in history if c.get("goal") == goal]

        total_cycles = len(history)
        avg_rye, min_rye, max_rye, rye_count = self.get_rye_stats(goal=goal)

        title += f"Total cycles: {total_cycles}\n"
        if rye_count > 0 and avg_rye is not None:
            title += f"RYE (avg): {avg_rye:.3f}\n"
            if min_rye is not None:
                title += f"RYE (min): {min_rye:.3f}\n"
            if max_rye is not None:
                title += f"RYE (max): {max_rye:.3f}\n"
        else:
            title += "RYE: no data available\n"
        title += "\n"

        # Advanced metrics
        adv = self.get_advanced_rye_metrics(goal=goal)
        if any(v is not None for v in adv.values()):
            title += "Advanced RYE metrics:\n"
            if adv.get("rolling_rye_10") is not None:
                title += f"- Rolling RYE (last 10): {adv['rolling_rye_10']:.3f}\n"
            if adv.get("rolling_rye_50") is not None:
                title += f"- Rolling RYE (last 50): {adv['rolling_rye_50']:.3f}\n"
            if adv.get("median_rye") is not None:
                title += f"- Median RYE: {adv['median_rye']:.3f}\n"
            if adv.get("efficiency_trend") is not None:
                title += f"- Efficiency trend (recent minus early): {adv['efficiency_trend']:.3f}\n"
            if adv.get("regression_slope") is not None:
                title += f"- Regression slope of RYE: {adv['regression_slope']:.4f}\n"
            if adv.get("stability_index") is not None:
                title += f"- Stability index: {adv['stability_index']:.3f}\n"
            if adv.get("recovery_momentum") is not None:
                title += f"- Recovery momentum: {adv['recovery_momentum']:.3f}\n"
            title += "\n"

        # Learning profile snapshot
        profile = self.get_learning_profile(goal=goal)
        curve = profile.get("rye_curve", {}) if isinstance(profile, dict) else {}
        best_cycle = profile.get("best_cycle", {}) if isinstance(profile, dict) else {}
        if profile:
            early_avg = curve.get("early_avg")
            late_avg = curve.get("late_avg")
            delta = curve.get("delta")
            title += "Learning curve snapshot:\n"
            if early_avg is not None:
                title += f"- Early RYE avg: {early_avg:.3f}\n"
            if late_avg is not None:
                title += f"- Late RYE avg: {late_avg:.3f}\n"
            if delta is not None:
                title += f"- RYE delta (late - early): {delta:.3f}\n"
            if best_cycle:
                bc_idx = best_cycle.get("cycle_index")
                bc_rye = best_cycle.get("RYE")
                bc_ts = best_cycle.get("timestamp")
                if bc_idx is not None and bc_rye is not None:
                    title += f"- Best cycle: {bc_idx} with RYE {bc_rye:.3f} at {bc_ts}\n"
            title += "\n"

        notes = self.get_notes(goal=goal)
        title += f"Notes collected: {len(notes)}\n"
        if notes:
            title += "\nRecent notes:\n"
            for n in notes[-5:]:
                ts = n.get("timestamp", "")
                content = str(n.get("content", "")).strip()
                if len(content) > 200:
                    content = content[:200] + "..."
                title += f"- [{ts}] {content}\n"
        title += "\n"

        hyps = self.get_hypotheses(goal=goal)
        title += f"Hypotheses generated: {len(hyps)}\n"
        if hyps:
            title += "\nRecent hypotheses:\n"
            for h in hyps[-5:]:
                ts = h.get("timestamp", "")
                text = str(h.get("text", "")).strip()
                score = h.get("score", None)
                if len(text) > 200:
                    text = text[:200] + "..."
                if isinstance(score, (int, float)):
                    title += f"- [{ts}] ({score:.2f}) {text}\n"
                else:
                    title += f"- [{ts}] {text}\n"
        title += "\n"

        cits = self.get_citations(goal=goal)
        title += f"Citations logged: {len(cits)}\n"
        if cits:
            title += "\nSample citations:\n"
            for e in cits[-5:]:
                ts = e.get("timestamp", "")
                c = e.get("citation", {}) or {}
                src = c.get("source", "web")
                c_title = c.get("title", "")
                url = c.get("url", "")
                title += f"- [{ts}] [{src}] {c_title} - {url}\n"

        disc = self.get_discoveries(goal=goal)
        title += "\nKey discoveries recorded: "
        title += f"{len(disc)}\n"
        if disc:
            title += "\nRecent discoveries:\n"
            for d in disc[-5:]:
                ts = d.get("timestamp", "")
                kind = d.get("kind", "")
                label = d.get("label", "")
                score = d.get("score", None)
                if isinstance(score, (int, float)):
                    title += f"- [{ts}] [{kind}] ({score:.2f}) {label}\n"
                else:
                    title += f"- [{ts}] [{kind}] {label}\n"

        # Benchmarks section (ARC etc.)
        bm = self.get_benchmark_results(goal=goal)
        title += f"\nBenchmark results logged: {len(bm)}\n"
        if bm:
            title += "\nRecent benchmark samples:\n"
            for b in bm[:5]:
                ts = b.get("timestamp", "")
                name = b.get("benchmark", "")
                task_id = b.get("task_id", "")
                score = b.get("score", None)
                passed = b.get("passed", None)
                if isinstance(score, (int, float)):
                    base = f"- [{ts}] {name} task={task_id} score={score:.3f}"
                else:
                    base = f"- [{ts}] {name} task={task_id}"
                if isinstance(passed, bool):
                    base += f" passed={passed}"
                title += base + "\n"

        title += "\nEnd of report.\n"
        return title

    # ------------------------------------------------------------------
    # Convenience getters for report_generator and UI
    # ------------------------------------------------------------------
    def get_all_cycles(self) -> List[Dict[str, Any]]:
        """Alias for getting all cycles (for report generators)."""
        return self._data.get("cycles", [])

    def get_cycles_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Return all cycles for a specific goal."""
        return [c for c in self._data.get("cycles", []) if c.get("goal") == goal]

    def get_all_notes(self) -> List[Dict[str, Any]]:
        """Return all notes regardless of goal."""
        return self._data.get("notes", [])

    def get_notes_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Explicit alias for get_notes(goal=...)."""
        return self.get_notes(goal=goal)

    def get_all_hypotheses(self) -> List[Dict[str, Any]]:
        """Return all hypotheses for all goals."""
        return self._data.get("hypotheses", [])

    def get_hypotheses_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Explicit alias for get_hypotheses(goal=...)."""
        return self.get_hypotheses(goal=goal)

    def get_all_citations(self) -> List[Dict[str, Any]]:
        """Return all citation entries for all goals."""
        return self._data.get("citations", [])

    def get_citations_for_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Explicit alias for get_citations(goal=...)."""
        return self.get_citations(goal=goal)

    # ------------------------------------------------------------------
    # Source index helpers
    # ------------------------------------------------------------------
    def get_source_index(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Return source_index entry for a key, or the entire index if key is None."""
        src = self._data.get("source_index") or {}
        if not isinstance(src, dict):
            src = {}
        if key is None:
            return dict(src)
        entry = src.get(key)
        if isinstance(entry, dict):
            return dict(entry)
        return {}

    def upsert_source_index(
        self,
        key: str,
        value: Dict[str, Any],
        *,
        merge: bool = True,
    ) -> None:
        """Upsert a single source_index entry.

        If merge is True and an entry exists, shallow merge the dicts.
        Otherwise, overwrite the existing entry.
        """
        src = self._data.get("source_index")
        if not isinstance(src, dict):
            src = {}
        existing = src.get(key)
        if merge and isinstance(existing, dict):
            merged = dict(existing)
            merged.update(dict(value))
            src[key] = merged
        else:
            src[key] = dict(value)
        self._data["source_index"] = src
        self._save()

    # ------------------------------------------------------------------
    # Goal index accessors for swarm analytics
    # ------------------------------------------------------------------
    def list_goals(self) -> List[str]:
        """Return a sorted list of known goals from goal_index."""
        gi = self._data.get("goal_index") or {}
        if not isinstance(gi, dict):
            return []
        return sorted(gi.keys())

    def get_goal_index(self, goal: Optional[str] = None) -> Dict[str, Any]:
        """Return compact goal index entry or the full index.

        If goal is None, return the entire goal_index dict.
        """
        gi = self._data.get("goal_index") or {}
        if not isinstance(gi, dict):
            return {}
        if goal is None:
            return dict(gi)
        entry = gi.get(goal, {})
        if isinstance(entry, dict):
            return dict(entry)
        return {}

    def get_goal_summary(self, goal: str) -> Dict[str, Any]:
        """Return a compact, UI ready summary for a single goal."""
        summary: Dict[str, Any] = {
            "goal": goal,
            "index": {},
            "rye_basic": {},
            "rye_advanced": {},
            "counts": {},
            "learning_profile": {},
        }

        gi = self._data.get("goal_index") or {}
        if isinstance(gi, dict):
            entry = gi.get(goal) or {}
            if isinstance(entry, dict):
                summary["index"] = dict(entry)

        avg_rye, min_rye, max_rye, count = self.get_rye_stats(goal=goal)
        summary["rye_basic"] = {
            "avg": avg_rye,
            "min": min_rye,
            "max": max_rye,
            "count": count,
        }
        summary["rye_advanced"] = self.get_advanced_rye_metrics(goal=goal)
        summary["learning_profile"] = self.get_learning_profile(goal=goal)

        notes = self.get_notes(goal=goal)
        hyps = self.get_hypotheses(goal=goal)
        cits = self.get_citations(goal=goal)
        disc = self.get_discoveries(goal=goal)

        summary["counts"] = {
            "notes": len(notes),
            "hypotheses": len(hyps),
            "citations": len(cits),
            "discoveries": len(disc),
            "cycles": len(self.get_cycles_for_goal(goal)),
        }

        return summary

    def get_goal_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return a leaderboard of goals ordered by avg_rye descending."""
        gi = self._data.get("goal_index") or {}
        if not isinstance(gi, dict):
            return []

        rows: List[Dict[str, Any]] = []
        for goal, entry in gi.items():
            if not isinstance(entry, dict):
                continue
            avg_val = entry.get("avg_rye")
            count_val = entry.get("rye_count")
            if not isinstance(avg_val, (int, float)) or not isinstance(count_val, int):
                continue
            rows.append(
                {
                    "goal": goal,
                    "avg_rye": float(avg_val),
                    "rye_count": int(count_val),
                    "note_count": int(entry.get("note_count", 0)),
                    "cycle_count": int(entry.get("cycle_count", 0)),
                    "domain": entry.get("domain"),
                    "best_rye": float(entry.get("best_rye")) if isinstance(entry.get("best_rye"), (int, float)) else None,
                    "best_breakthrough_score": float(entry.get("best_breakthrough_score"))
                    if isinstance(entry.get("best_breakthrough_score"), (int, float))
                    else None,
                }
            )

        rows_sorted = sorted(rows, key=lambda r: (r["avg_rye"], r["rye_count"]), reverse=True)
        return rows_sorted[:limit]

    def get_role_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return a leaderboard of logical roles aggregated across goals.

        Aggregates avg_rye and counts for each role in goal_index["roles"].
        """
        gi = self._data.get("goal_index") or {}
        if not isinstance(gi, dict):
            return []

        agg: Dict[str, Dict[str, Any]] = {}
        for goal, entry in gi.items():
            if not isinstance(entry, dict):
                continue
            roles = entry.get("roles") or {}
            if not isinstance(roles, dict):
                continue
            for role, r_stats in roles.items():
                if not isinstance(r_stats, dict):
                    continue
                avg_rye = r_stats.get("avg_rye")
                rye_count = r_stats.get("rye_count")
                if not isinstance(avg_rye, (int, float)) or not isinstance(rye_count, int) or rye_count <= 0:
                    continue
                row = agg.get(role) or {
                    "role": role,
                    "rye_sum": 0.0,
                    "rye_count": 0,
                    "note_count": 0,
                    "cycle_count": 0,
                    "goals_seen": 0,
                }
                row["rye_sum"] += float(avg_rye) * float(rye_count)
                row["rye_count"] += int(rye_count)
                row["note_count"] += int(r_stats.get("note_count", 0))
                row["cycle_count"] += int(r_stats.get("cycle_count", 0))
                row["goals_seen"] += 1
                agg[role] = row

        rows: List[Dict[str, Any]] = []
        for role, row in agg.items():
            total_rye_count = row.get("rye_count", 0)
            if total_rye_count <= 0:
                continue
            avg_rye_global = row["rye_sum"] / float(total_rye_count)
            rows.append(
                {
                    "role": role,
                    "avg_rye": avg_rye_global,
                    "rye_count": total_rye_count,
                    "note_count": row.get("note_count", 0),
                    "cycle_count": row.get("cycle_count", 0),
                    "goals_seen": row.get("goals_seen", 0),
                }
            )

        rows_sorted = sorted(rows, key=lambda r: (r["avg_rye"], r["rye_count"]), reverse=True)
        return rows_sorted[:limit]

    # ------------------------------------------------------------------
    # Fast learning snapshot for meta controllers
    # ------------------------------------------------------------------
    def get_fast_learning_snapshot(
        self,
        goal: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a compact, ultra fast learning snapshot.

        This is designed for:
            - ultra mode controllers that need a single read per step
            - swarm role schedulers
            - dashboards that only need directional signals

        If goal is None, the goal with highest avg_rye is selected
        (if any goals exist).
        """
        gi = self._data.get("goal_index") or {}
        if not isinstance(gi, dict) or not gi:
            return {
                "goal": None,
                "role": role,
                "avg_rye": None,
                "last_rye": None,
                "last_rye_delta": None,
                "best_rye": None,
                "rye_count": 0,
                "cycle_count": 0,
                "note_count": 0,
                "last_equilibrium_label": None,
                "last_breakthrough_score": None,
                "best_breakthrough_score": None,
                "learning_burst_active": False,
                "learning_burst_cycles_remaining": None,
                "trend": None,
            }

        chosen_goal = goal
        if chosen_goal is None:
            # Pick best goal by avg_rye then rye_count
            best_row = None
            for g, entry in gi.items():
                if not isinstance(entry, dict):
                    continue
                avg_val = entry.get("avg_rye")
                count_val = entry.get("rye_count")
                if not isinstance(avg_val, (int, float)) or not isinstance(count_val, int):
                    continue
                row = {
                    "goal": g,
                    "avg_rye": float(avg_val),
                    "rye_count": int(count_val),
                }
                if best_row is None:
                    best_row = row
                else:
                    if (row["avg_rye"], row["rye_count"]) > (best_row["avg_rye"], best_row["rye_count"]):
                        best_row = row
            if best_row is not None:
                chosen_goal = best_row["goal"]

        entry = gi.get(chosen_goal or "") or {}
        if not isinstance(entry, dict):
            entry = {}

        avg_rye = entry.get("avg_rye")
        last_rye = entry.get("last_rye")
        last_rye_delta = entry.get("last_rye_delta")
        best_rye = entry.get("best_rye")
        rye_count = entry.get("rye_count") if isinstance(entry.get("rye_count"), int) else 0
        cycle_count = entry.get("cycle_count") if isinstance(entry.get("cycle_count"), int) else 0
        note_count = entry.get("note_count") if isinstance(entry.get("note_count"), int) else 0

        last_equilibrium_label = entry.get("last_equilibrium_label")
        last_breakthrough_score = entry.get("last_breakthrough_score")
        best_breakthrough_score = entry.get("best_breakthrough_score")

        # Role level overrides if requested
        role_block = None
        if role is not None:
            roles_dict = entry.get("roles") or {}
            if isinstance(roles_dict, dict):
                rb = roles_dict.get(role)
                if isinstance(rb, dict):
                    role_block = rb

        if role_block:
            if isinstance(role_block.get("avg_rye"), (int, float)):
                avg_rye = float(role_block["avg_rye"])
            if isinstance(role_block.get("rye_count"), int):
                rye_count = int(role_block["rye_count"])
            if isinstance(role_block.get("cycle_count"), int):
                cycle_count = int(role_block["cycle_count"])
            if isinstance(role_block.get("note_count"), int):
                note_count = int(role_block["note_count"])

        # Learning burst state
        lb = self._data.get("learning_burst") or {}
        if not isinstance(lb, dict):
            lb = {}
        lb_active = bool(lb.get("active"))
        lb_cycles_remaining = lb.get("cycles_remaining")

        # Simple directional trend from last_rye_delta
        trend: Optional[str] = None
        if isinstance(last_rye_delta, (int, float)):
            if last_rye_delta > 0.01:
                trend = "improving"
            elif last_rye_delta < -0.01:
                trend = "declining"
            else:
                trend = "flat"

        snapshot: Dict[str, Any] = {
            "goal": chosen_goal,
            "role": role,
            "avg_rye": float(avg_rye) if isinstance(avg_rye, (int, float)) else None,
            "last_rye": float(last_rye) if isinstance(last_rye, (int, float)) else None,
            "last_rye_delta": float(last_rye_delta) if isinstance(last_rye_delta, (int, float)) else None,
            "best_rye": float(best_rye) if isinstance(best_rye, (int, float)) else None,
            "rye_count": rye_count,
            "cycle_count": cycle_count,
            "note_count": note_count,
            "last_equilibrium_label": last_equilibrium_label,
            "last_breakthrough_score": float(last_breakthrough_score)
            if isinstance(last_breakthrough_score, (int, float))
            else None,
            "best_breakthrough_score": float(best_breakthrough_score)
            if isinstance(best_breakthrough_score, (int, float))
            else None,
            "learning_burst_active": lb_active,
            "learning_burst_cycles_remaining": lb_cycles_remaining if isinstance(lb_cycles_remaining, int) else None,
            "trend": trend,
        }
        return snapshot

    # ------------------------------------------------------------------
    # Benchmarks (ARC, math suites, etc.)
    # ------------------------------------------------------------------
    def log_benchmark_result(
        self,
        *,
        benchmark: str,
        task_id: Optional[str] = None,
        score: Optional[float] = None,
        max_score: Optional[float] = None,
        passed: Optional[bool] = None,
        run_id: Optional[str] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        role: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single benchmark result (ARC task, math suite, etc.)."""
        entry: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "benchmark": benchmark,
            "task_id": task_id,
            "score": float(score) if isinstance(score, (int, float)) else None,
            "max_score": float(max_score) if isinstance(max_score, (int, float)) else None,
            "passed": bool(passed) if isinstance(passed, bool) else None,
            "run_id": run_id,
            "goal": goal,
            "domain": domain,
            "role": role,
        }
        if extra:
            entry["extra"] = dict(extra)

        arr = self._data.setdefault("benchmarks", [])
        arr.append(entry)
        if len(arr) > MAX_BENCHMARKS:
            self._data["benchmarks"] = arr[-MAX_BENCHMARKS:]

        self._save()

    def get_benchmark_results(
        self,
        benchmark: Optional[str] = None,
        run_id: Optional[str] = None,
        goal: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Retrieve benchmark results filtered by benchmark, run, goal."""
        arr = self._data.get("benchmarks", [])
        if not isinstance(arr, list):
            return []

        results: List[Dict[str, Any]] = []
        for e in reversed(arr):
            if not isinstance(e, dict):
                continue
            if benchmark is not None and e.get("benchmark") != benchmark:
                continue
            if run_id is not None and e.get("run_id") != run_id:
                continue
            if goal is not None and e.get("goal") != goal:
                continue
            results.append(e)
            if len(results) >= limit:
                break
        return results

    def get_benchmark_summary(
        self,
        benchmark: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a compact summary of benchmark performance."""
        arr = self._data.get("benchmarks", [])
        if not isinstance(arr, list) or not arr:
            return {
                "benchmark": benchmark,
                "count": 0,
                "avg_score": None,
                "best_score": None,
                "pass_rate": None,
            }

        scores: List[float] = []
        passes = 0
        total = 0
        for e in arr:
            if not isinstance(e, dict):
                continue
            if benchmark is not None and e.get("benchmark") != benchmark:
                continue
            total += 1
            sc = e.get("score")
            if isinstance(sc, (int, float)):
                scores.append(float(sc))
            p = e.get("passed")
            if isinstance(p, bool) and p:
                passes += 1

        if total == 0:
            return {
                "benchmark": benchmark,
                "count": 0,
                "avg_score": None,
                "best_score": None,
                "pass_rate": None,
            }

        avg_score = sum(scores) / len(scores) if scores else None
        best_score = max(scores) if scores else None
        pass_rate = passes / float(total) if total > 0 else None

        return {
            "benchmark": benchmark,
            "count": total,
            "avg_score": avg_score,
            "best_score": best_score,
            "pass_rate": pass_rate,
        }

    # ------------------------------------------------------------------
    # Run overview rows for UI tables
    # ------------------------------------------------------------------
    def get_run_table_rows(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return table friendly rows for run overview.

        Each row bundles run_id, goals, counts, basic RYE and timeline
        so that the UI can build a cycles/citations/discoveries table
        without extra joins.

        Uses run_manifests when available, and falls back to scanning
        cycles for runs that do not yet have a manifest, so older or
        legacy runs still appear in the table.
        """
        rows: List[Dict[str, Any]] = []

        # First: rows backed by explicit manifests (preferred)
        manifests = self.list_run_manifests(limit=limit)
        seen_run_ids = set()

        for m in manifests:
            if not isinstance(m, dict):
                continue
            run_id = m.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue

            seen_run_ids.add(run_id)
            summary = self.get_run_summary(run_id)
            counts = summary.get("counts", {}) or {}
            timeline = summary.get("timeline", {}) or {}
            rye_basic = summary.get("rye_basic", {}) or {}

            goals = summary.get("goals", []) or []
            if not isinstance(goals, list):
                goals = []
            goals_str = ", ".join([g for g in goals if isinstance(g, str)])

            row = {
                "run_id": run_id,
                "active_run_id": run_id,
                "label": m.get("label") or m.get("name") or run_id,
                "goals": goals,
                "goals_str": goals_str,
                "cycles": int(counts.get("cycles", 0) or 0),
                "citations": int(counts.get("citations", 0) or 0),
                "discoveries": int(counts.get("discoveries", 0) or 0),
                "benchmarks": int(counts.get("benchmarks", 0) or 0),
                "started_at": timeline.get("started_at") or m.get("started_at"),
                "finished_at": timeline.get("finished_at") or m.get("finished_at"),
                "avg_rye": rye_basic.get("avg"),
                "best_rye": rye_basic.get("max"),
                # Extra fields many UIs like to show, but safe if unused
                "status": m.get("status"),
                "mode": m.get("mode"),
                "domain": m.get("domain"),
                "runtime_profile": m.get("runtime_profile"),
                "created_at": m.get("created_at"),
                "logged_at": m.get("logged_at"),
            }
            rows.append(row)

        # Second: synthesize rows for any run_id present in cycles
        # that does not yet have a manifest entry.
        history = self._data.get("cycles", [])
        if isinstance(history, list):
            runs_without_manifest: Dict[str, List[Dict[str, Any]]] = {}
            for c in history:
                if not isinstance(c, dict):
                    continue
                rid = c.get("run_id")
                if not isinstance(rid, str) or not rid:
                    continue
                if rid in seen_run_ids:
                    continue
                runs_without_manifest.setdefault(rid, []).append(c)

            for run_id, _cycles in runs_without_manifest.items():
                summary = self.get_run_summary(run_id)
                counts = summary.get("counts", {}) or {}
                timeline = summary.get("timeline", {}) or {}
                rye_basic = summary.get("rye_basic", {}) or {}

                goals = summary.get("goals", []) or []
                if not isinstance(goals, list):
                    goals = []
                goals_str = ", ".join([g for g in goals if isinstance(g, str)])

                row = {
                    "run_id": run_id,
                "active_run_id": run_id,
                    "label": run_id,
                    "goals": goals,
                    "goals_str": goals_str,
                    "cycles": int(counts.get("cycles", 0) or 0),
                    "citations": int(counts.get("citations", 0) or 0),
                    "discoveries": int(counts.get("discoveries", 0) or 0),
                    "benchmarks": int(counts.get("benchmarks", 0) or 0),
                    "started_at": timeline.get("started_at"),
                    "finished_at": timeline.get("finished_at"),
                    "avg_rye": rye_basic.get("avg"),
                    "best_rye": rye_basic.get("max"),
                    "status": None,
                    "mode": None,
                    "domain": None,
                    "runtime_profile": None,
                    "created_at": None,
                    "logged_at": None,
                }
                rows.append(row)

        # Sort rows by logged_at then started_at, newest first
        def _row_key(r: Dict[str, Any]) -> str:
            ts = r.get("logged_at") or r.get("started_at") or ""
            return str(ts)

        rows_sorted = sorted(rows, key=_row_key, reverse=True)
        if limit and limit > 0:
            rows_sorted = rows_sorted[:limit]
        return rows_sorted

    def get_run_overview_rows(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Compatibility alias for get_run_table_rows used by some UIs."""
        return self.get_run_table_rows(limit=limit)

    # ------------------------------------------------------------------
    # Run-history compatibility helpers (used by MSIL and other analytics layers)
    # ------------------------------------------------------------------
    def get_run_history(self, run_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return cycles associated with a run_id, oldest -> newest.

        Notes:
            - This is a convenience alias for analytics layers.
            - Existing get_cycles_for_run() returns newest -> oldest.
        """
        if not run_id:
            return []
        newest_first = self.get_cycles_for_run(run_id, limit=limit)
        return list(reversed(newest_first))

    def get_run_stats(self, run_id: str) -> Dict[str, Any]:
        """Return a compact run stats bundle for meta controllers / MSIL."""
        if not run_id:
            return {}
        summary = self.get_run_summary(run_id)
        tool_stats = self.get_tool_stats(run_id=run_id)
        latest_msil = self.get_latest_msil_snapshot(run_id=run_id)
        return {
            "run_id": run_id,
            "summary": summary,
            "tool_stats": tool_stats,
            "latest_msil": latest_msil,
        }

    # ------------------------------------------------------------------
    # Replay buffer (optional; MSIL can use this when available)
    # ------------------------------------------------------------------
    def add_replay_item(
        self,
        *,
        goal: str,
        text: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        rye_score: Optional[float] = None,
        item_type: str = "cycle",
        tags: Optional[List[str]] = None,
        domain: Optional[str] = None,
        role: Optional[str] = None,
        run_id: Optional[str] = None,
        cycle_index: Optional[int] = None,
        source: str = "agent",
        importance: float = 1.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a replay item (high value artifact for future recall).

        This is intentionally lightweight. Prefer storing pointers (run_id,
        cycle_index) over huge payloads to keep memory.json manageable.
        """
        if not goal:
            return

        entry: Dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "goal": goal,
            "item_type": item_type,
            "text": text,
            "payload": payload or {},
            "rye_score": float(rye_score) if isinstance(rye_score, (int, float)) else None,
            "tags": tags or [],
            "domain": domain,
            "role": role,
            "run_id": run_id,
            "cycle_index": int(cycle_index) if isinstance(cycle_index, int) else cycle_index,
            "source": source,
            "importance": float(importance),
        }
        if extra:
            entry["extra"] = dict(extra)

        arr = self._data.setdefault("replay_buffer", [])
        if not isinstance(arr, list):
            arr = []
        arr.append(entry)
        if len(arr) > MAX_REPLAY_BUFFER:
            arr = arr[-MAX_REPLAY_BUFFER:]
        self._data["replay_buffer"] = arr
        self._save()

        # Optional semantic storage
        if self.vector_memory is not None and text:
            try:
                meta: Dict[str, Any] = {
                    "goal": goal,
                    "type": "replay_item",
                    "item_type": item_type,
                    "rye_score": entry.get("rye_score"),
                    "tags": entry.get("tags", []),
                    "source": source,
                }
                if domain is not None:
                    meta["domain"] = domain
                if role is not None:
                    meta["role"] = role
                if run_id is not None:
                    meta["run_id"] = run_id
                if cycle_index is not None:
                    meta["cycle_index"] = int(cycle_index)
                self.vector_memory.add_item(text=text, metadata=meta)
            except Exception:
                pass

    def get_replay_items_for_goal(
        self,
        goal: str,
        *,
        limit: int = 200,
        min_rye: Optional[float] = None,
        run_id: Optional[str] = None,
        prefer_buffer: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return replay items for a goal, most recent first.

        If replay_buffer is empty (or prefer_buffer=False), this falls back to
        deriving replay items from recent cycle history for the goal.
        """
        if not goal:
            return []

        # 1) Prefer explicit replay_buffer if present and non-empty
        buf = self._data.get("replay_buffer", [])
        if prefer_buffer and isinstance(buf, list) and buf:
            out: List[Dict[str, Any]] = []
            lim = max(1, int(limit)) if isinstance(limit, int) else 200
            for e in reversed(buf):  # newest first (append order)
                if not isinstance(e, dict):
                    continue
                if e.get("goal") != goal:
                    continue
                if run_id is not None and e.get("run_id") != run_id:
                    continue
                if min_rye is not None:
                    rs = e.get("rye_score")
                    if not isinstance(rs, (int, float)):
                        continue
                    if float(rs) < float(min_rye):
                        continue
                out.append(dict(e))
                if len(out) >= lim:
                    break
            return out

        # 2) Fallback: derive from recent cycles
        out2: List[Dict[str, Any]] = []
        try:
            lim2 = max(1, int(limit))
        except Exception:
            lim2 = 200
        window = max(lim2, 200)
        cycles = self.get_cycle_history_for_goal(goal, limit=window)  # oldest->newest
        for c in reversed(cycles):  # newest first
            if not isinstance(c, dict):
                continue
            if run_id is not None and c.get("run_id") != run_id:
                continue
            v = c.get("RYE")
            if min_rye is not None:
                if not isinstance(v, (int, float)) or float(v) < float(min_rye):
                    continue
            cy_idx = c.get("cycle_index", c.get("cycle"))
            out2.append(
                {
                    "timestamp": c.get("timestamp"),
                    "goal": goal,
                    "item_type": "cycle_history",
                    "text": None,
                    "payload": {},
                    "rye_score": float(v) if isinstance(v, (int, float)) else None,
                    "tags": [],
                    "domain": c.get("domain"),
                    "role": c.get("role"),
                    "run_id": c.get("run_id"),
                    "cycle_index": cy_idx,
                    "source": "derived_from_cycles",
                    "importance": 1.0,
                }
            )
            if len(out2) >= lim2:
                break
        return out2

    # ------------------------------------------------------------------
    # MSIL snapshots (optional storage for the Meta Skill Intelligence Layer)
    # ------------------------------------------------------------------
    def _compact_msil_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Compact an MSIL snapshot to keep memory.json from growing too fast."""
        if not isinstance(snapshot, dict):
            return {}
        # Keep the key fields used by dashboards and meta-controllers.
        compact: Dict[str, Any] = {
            "timestamp": snapshot.get("timestamp"),
            "goal": snapshot.get("goal"),
            "msil_score": snapshot.get("msil_score"),
            "intelligence_stage": snapshot.get("intelligence_stage"),
            "total_cycles_for_goal": snapshot.get("total_cycles_for_goal"),
            "recent_window": snapshot.get("recent_window"),
            "skills": snapshot.get("skills"),
        }
        # Keep top domain profiles only (to avoid bloat).
        pdp = snapshot.get("per_domain_profiles")
        if isinstance(pdp, list):
            compact["per_domain_profiles"] = pdp[:8]
        # Extras are usually small and useful for debugging.
        extras = snapshot.get("extras")
        if isinstance(extras, dict):
            compact["extras"] = extras
        return compact

    def log_msil_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        goal: Optional[str] = None,
        cycle_index: Optional[int] = None,
        compact: bool = True,
    ) -> None:
        """Persist an MSIL snapshot (typically called once per N cycles)."""
        if not isinstance(snapshot, dict) or not snapshot:
            return

        snap_goal = goal or snapshot.get("goal")
        if not isinstance(snap_goal, str) or not snap_goal:
            return

        ts = snapshot.get("timestamp")
        if not isinstance(ts, str) or not ts:
            ts = _utc_now_iso()

        stored_snapshot = self._compact_msil_snapshot(snapshot) if compact else dict(snapshot)

        entry: Dict[str, Any] = {
            "timestamp": ts,
            "goal": snap_goal,
            "run_id": run_id,
            "cycle_index": cycle_index,
            "snapshot": stored_snapshot,
        }

        arr = self._data.setdefault("msil_snapshots", [])
        if not isinstance(arr, list):
            arr = []
        arr.append(entry)
        if len(arr) > MAX_MSIL_SNAPSHOTS:
            arr = arr[-MAX_MSIL_SNAPSHOTS:]
        self._data["msil_snapshots"] = arr

        # Optional: mirror a few MSIL stats into goal_index for fast dashboards.
        try:
            msil_score = stored_snapshot.get("msil_score")
            if isinstance(msil_score, (int, float)):
                gi = self._data.setdefault("goal_index", {})
                if not isinstance(gi, dict):
                    gi = {}
                g_entry = gi.get(snap_goal) or {}
                if not isinstance(g_entry, dict):
                    g_entry = {}
                g_entry["last_msil_score"] = float(msil_score)
                prev_best = g_entry.get("best_msil_score")
                if not isinstance(prev_best, (int, float)) or float(msil_score) > float(prev_best):
                    g_entry["best_msil_score"] = float(msil_score)
                g_entry["msil_count"] = int(g_entry.get("msil_count", 0)) + 1
                g_entry["msil_last_updated"] = _utc_now_iso()
                gi[snap_goal] = g_entry
                self._data["goal_index"] = gi
        except Exception:
            pass

        self._save()

    def get_msil_snapshots(
        self,
        *,
        goal: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Retrieve MSIL snapshots, most recent first."""
        arr = self._data.get("msil_snapshots", [])
        if not isinstance(arr, list) or not arr:
            return []
        out: List[Dict[str, Any]] = []
        lim = max(1, int(limit)) if isinstance(limit, int) else 200
        for e in reversed(arr):  # newest first
            if not isinstance(e, dict):
                continue
            if goal is not None and e.get("goal") != goal:
                continue
            if run_id is not None and e.get("run_id") != run_id:
                continue
            out.append(dict(e))
            if len(out) >= lim:
                break
        return out

    def get_latest_msil_snapshot(
        self,
        *,
        goal: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent MSIL snapshot for a goal/run_id if present."""
        snaps = self.get_msil_snapshots(goal=goal, run_id=run_id, limit=1)
        if snaps:
            return snaps[0]
        return None
