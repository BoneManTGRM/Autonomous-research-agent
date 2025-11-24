#!/usr/bin/env python
"""
Memory migration utility for the Autonomous Research Agent.

Goals:
- Safely migrate an old memory file to the new location from AppConfig.
- Preserve all learning content while adding minimal structure if missing.
- Normalize containers for sessions, events, and learning_log.
- Build a small learning_stats block for smarter long-run behavior.
- Create timestamped backups before writing anything.
- Print a clear report so you can see exactly what changed.

Usage examples:

  python scripts/migrate_memory.py
  python scripts/migrate_memory.py --source logs/sessions/old_memory.json
  python scripts/migrate_memory.py --target logs/sessions/new_memory.json
  python scripts/migrate_memory.py --dry-run

Environment helpers:

  OLD_MEMORY_FILE      Optional default source path
  MEMORY_FILE          Optional default target override (matches AppConfig)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    # New config system
    from config import build_app_config  # type: ignore
except Exception:
    build_app_config = None  # type: ignore


JsonDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Basic IO helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Tuple[JsonDict, int]:
    if not path.exists():
        raise FileNotFoundError(f"Memory file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    size_bytes = len(raw.encode("utf-8"))
    try:
        data = json.loads(raw) or {}
    except Exception as exc:
        raise RuntimeError(f"Could not parse JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        data = {"data": data}
    return data, size_bytes


def _write_json(path: Path, data: JsonDict) -> int:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _make_backup(path: Path) -> Path:
    if not path.exists():
        return path
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".backup-{ts}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


# ---------------------------------------------------------------------------
# Learning-aware shaping
# ---------------------------------------------------------------------------


def _ensure_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _infer_learning_container(data: JsonDict) -> List[Any]:
    """
    Try to find legacy learning containers and normalize into a list.

    Widely compatible and non-destructive:
    - learning_log (preferred)
    - learning
    - knowledge
    - replay_buffer
    """
    for key in ("learning_log", "learning", "knowledge", "replay_buffer"):
        if key in data:
            return _ensure_list(data[key])
    return []


def _infer_sessions_container(data: JsonDict) -> List[Any]:
    """
    Try to find sessions from a variety of old shapes.

    Priority:
    - sessions
    - history
    - runs
    - cycles
    """
    for key in ("sessions", "history", "runs", "cycles"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def _infer_events_container(data: JsonDict) -> List[Any]:
    """
    Try to find an events-like container without inventing data.
    """
    for key in ("events", "logs", "event_log"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def _build_learning_stats(learning_log: List[Any]) -> JsonDict:
    """
    Build a small, schema-free summary of learning content.

    This is intentionally conservative and only looks for common keys
    (goal, domain, phase, rye, timestamp). Nothing is required.
    """
    total = len(learning_log)
    by_domain: Dict[str, int] = {}
    by_goal: Dict[str, int] = {}
    rye_values: List[float] = []
    last_ts: Union[str, None] = None

    for item in learning_log:
        if not isinstance(item, dict):
            continue

        domain = str(item.get("domain", "")).strip() or "unknown"
        by_domain[domain] = by_domain.get(domain, 0) + 1

        goal = str(item.get("goal", "")).strip()
        if goal:
            by_goal[goal] = by_goal.get(goal, 0) + 1

        rye_val = item.get("RYE") or item.get("rye")
        if isinstance(rye_val, (int, float)):
            rye_values.append(float(rye_val))

        ts = item.get("timestamp") or item.get("time")
        if isinstance(ts, str):
            # we do not parse, just keep the last seen string
            last_ts = ts

    rye_avg = sum(rye_values) / len(rye_values) if rye_values else None
    rye_max = max(rye_values) if rye_values else None

    return {
        "total_learning_items": total,
        "by_domain": by_domain,
        "by_goal_count": by_goal,
        "rye_avg": rye_avg,
        "rye_max": rye_max,
        "last_timestamp": last_ts,
    }


def _ensure_learning_structure(data: JsonDict) -> JsonDict:
    """
    Make sure the memory dict has sane containers for long-term learning,
    without destroying any existing structure.
    """
    # Versioning for future tools
    version = data.get("version")
    if not isinstance(version, int):
        version = 1
    # We bump to at least 2 once migration happens
    if version < 2:
        version = 2
    data["version"] = version

    # Sessions
    sessions = _infer_sessions_container(data)
    data["sessions"] = sessions

    # Learning log
    learning_log = _infer_learning_container(data)
    data["learning_log"] = learning_log

    # Events
    events = _infer_events_container(data)
    data["events"] = events

    # Minimal metadata block
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("engine", "autonomous_research_agent")
    meta["last_migrated"] = datetime.utcnow().isoformat() + "Z"
    meta["schema_version"] = version

    # Learning stats (non-destructive)
    meta["learning_stats"] = _build_learning_stats(learning_log)

    data["metadata"] = meta
    return data


def _summarize(data: JsonDict) -> JsonDict:
    def _count(key: str) -> int:
        val = data.get(key)
        return len(val) if isinstance(val, list) else 0

    keys = sorted(list(data.keys()))
    return {
        "sessions": _count("sessions"),
        "learning_log": _count("learning_log"),
        "events": _count("events"),
        "keys": keys,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate memory file to the new AppConfig location with learning-aware structure."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("OLD_MEMORY_FILE", "logs/sessions/default_memory.json"),
        help="Source memory JSON file (old location).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target memory JSON file. Defaults to AppConfig.memory_file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and analyze but do not write anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine target from AppConfig if not explicitly passed
    if args.target is None:
        if build_app_config is not None:
            cfg = build_app_config()
            target_path = Path(cfg.memory_file)
        else:
            target_path = Path(os.getenv("MEMORY_FILE", "logs/sessions/default_memory.json"))
    else:
        target_path = Path(args.target)

    source_path = Path(args.source)

    print("=== Memory migration start ===")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Dry run: {args.dry_run}")
    print("")

    src_data, src_size = _read_json(source_path)
    src_summary = _summarize(src_data)

    print("Source summary:")
    print(f"  Size: {src_size / 1024:.2f} KB")
    print(f"  Keys: {', '.join(src_summary['keys'])}")
    print(f"  Sessions: {src_summary['sessions']}")
    print(f"  Learning entries: {src_summary['learning_log']}")
    print(f"  Events: {src_summary['events']}")
    print("")

    migrated = _ensure_learning_structure(dict(src_data))
    mig_summary = _summarize(migrated)

    added_keys = [k for k in mig_summary["keys"] if k not in src_summary["keys"]]
    print("After migration shaping (in memory only):")
    print(f"  Keys: {', '.join(mig_summary['keys'])}")
    print(f"  Sessions: {mig_summary['sessions']}")
    print(f"  Learning entries: {mig_summary['learning_log']}")
    print(f"  Events: {mig_summary['events']}")
    if added_keys:
        print(f"  New top-level keys added: {', '.join(added_keys)}")
    print("  Metadata.learning_stats:", migrated.get("metadata", {}).get("learning_stats", {}))
    print("")

    if args.dry_run:
        print("Dry run only. No files were written.")
        print("=== Done (dry run) ===")
        return

    backup_path = _make_backup(target_path)
    if backup_path.exists():
        print(f"Existing target backed up to: {backup_path}")

    new_size = _write_json(target_path, migrated)
    print("")
    print("Migration complete.")
    print(f"Target size: {new_size / 1024:.2f} KB")
    print(f"Target path: {target_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
