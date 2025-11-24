#!/usr/bin/env python
"""
Memory compaction and hygiene utility for the Autonomous Research Agent.

Goals:
- Keep the memory file healthy for long running learning.
- Preserve the most valuable learning content, not just the most recent.
- Trim very old sessions, events, and low value learning entries.
- Rebuild learning stats so the engine can reason about its own history.
- Reduce file size so long runs on Render stay responsive.
- Keep the process transparent with before and after intelligence reports.

Usage examples:

  python scripts/compact_memory.py
  python scripts/compact_memory.py --max-size-mb 50
  python scripts/compact_memory.py --keep-last-sessions 25 --keep-last-learning 500
  python scripts/compact_memory.py --keep-top-learning-rye 200 --min-rye-protect 0.1
  python scripts/compact_memory.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

try:
    # New config system
    from config import build_app_config  # type: ignore
except Exception:
    build_app_config = None  # type: ignore


# --------------------------------------------------------------------
# Basic file helpers
# --------------------------------------------------------------------


def _read_json(path: Path) -> Tuple[Dict[str, Any], int]:
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


def _write_json(path: Path, data: Dict[str, Any]) -> int:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _make_backup(path: Path) -> Path:
    if not path.exists():
        return path
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".compact-backup-{ts}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def _tail_list(obj: Any, keep_last: int) -> List[Any]:
    """
    Return the last keep_last items of a list.
    For non lists, return an empty list.
    """
    if not isinstance(obj, list):
        return []
    if keep_last <= 0:
        return []
    if len(obj) <= keep_last:
        return obj
    return obj[-keep_last:]


# --------------------------------------------------------------------
# Structure and learning helpers
# --------------------------------------------------------------------


def _ensure_learning_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the memory dict so it is safe for learning and compaction.

    This does not destroy existing structure, it only adds missing containers
    and basic metadata.
    """
    # Schema and versioning
    if "schema_version" not in data:
        # Version 2 means "learning aware" layout is present.
        data.setdefault("schema_version", 2)
    if "version" not in data:
        data.setdefault("version", 2)

    # Sessions
    if "sessions" not in data:
        # Many older layouts used "history" or "runs"
        if isinstance(data.get("history"), list):
            data["sessions"] = data["history"]
        elif isinstance(data.get("runs"), list):
            data["sessions"] = data["runs"]
        else:
            data["sessions"] = []

    # Events
    ev = data.get("events")
    if ev is None:
        data["events"] = []
    elif not isinstance(ev, list):
        data["events"] = [ev]

    # Learning log
    learning = data.get("learning_log")
    if learning is None:
        data["learning_log"] = []
    elif not isinstance(learning, list):
        data["learning_log"] = [learning]

    # Discovery log (if present)
    discovery = data.get("discovery_log")
    if discovery is None:
        data["discovery_log"] = []
    elif not isinstance(discovery, list):
        data["discovery_log"] = [discovery]

    # Metadata block
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("engine", "autonomous_research_agent")
    meta.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
    data["metadata"] = meta

    return data


def _extract_rye(entry: Any) -> float:
    """
    Try to extract a numeric RYE value from a learning entry.

    Looks at:
      - entry["RYE"]
      - entry["rye"]
      - entry["score"]
      - entry["metrics"]["RYE"] or ["rye"]
    """
    if not isinstance(entry, dict):
        return 0.0

    for key in ("RYE", "rye", "score"):
        val = entry.get(key)
        if isinstance(val, (int, float)):
            return float(val)

    metrics = entry.get("metrics")
    if isinstance(metrics, dict):
        for key in ("RYE", "rye"):
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                return float(val)

    return 0.0


def _is_discovery(entry: Any, discovery_key: str) -> bool:
    """
    Decide whether a learning entry should be treated as a discovery.
    """
    if not isinstance(entry, dict):
        return False

    # Simple boolean field
    flag = entry.get(discovery_key)
    if isinstance(flag, bool) and flag:
        return True

    # Text flag, like "yes" or "true"
    if isinstance(flag, str) and flag.strip().lower() in {"yes", "true", "1"}:
        return True

    # Tag based
    tags = entry.get("tags") or entry.get("labels")
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, str) and discovery_key.lower() in t.lower():
                return True

    # Discovery score
    disc_score = entry.get("discovery_score")
    if isinstance(disc_score, (int, float)) and disc_score > 0:
        return True

    return False


def _learning_key(entry: Any) -> str:
    """
    Build a stable dedupe key for a learning entry.
    Uses JSON representation as a content hash.
    """
    try:
        return json.dumps(entry, sort_keys=True, ensure_ascii=False)
    except Exception:
        return repr(entry)


def _build_learning_stats(learning: List[Any]) -> Dict[str, Any]:
    """
    Build aggregate stats for learning_log to store under metadata.learning_stats.
    """
    rye_vals: List[float] = []
    domains: Dict[str, int] = {}
    goals: Dict[str, int] = {}

    last_ts: str | None = None

    for item in learning:
        if not isinstance(item, dict):
            continue

        r = _extract_rye(item)
        if r != 0.0:
            rye_vals.append(r)

        d = item.get("domain")
        if isinstance(d, str) and d:
            domains[d] = domains.get(d, 0) + 1

        g = item.get("goal")
        if isinstance(g, str) and g:
            goals[g] = goals.get(g, 0) + 1

        ts = item.get("timestamp") or item.get("created_at")
        if isinstance(ts, str):
            last_ts = ts

    stats: Dict[str, Any] = {
        "total_entries": len(learning),
        "domains": domains,
        "goals": goals,
        "last_entry_timestamp": last_ts,
    }

    if rye_vals:
        stats["rye_avg"] = float(mean(rye_vals))
        stats["rye_median"] = float(median(rye_vals))
        stats["rye_max"] = float(max(rye_vals))
        stats["rye_min"] = float(min(rye_vals))
    else:
        stats["rye_avg"] = None
        stats["rye_median"] = None
        stats["rye_max"] = None
        stats["rye_min"] = None

    stats["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return stats


def _summarize(data: Dict[str, Any]) -> Dict[str, Any]:
    def _count(key: str) -> int:
        val = data.get(key)
        return len(val) if isinstance(val, list) else 0

    return {
        "sessions": _count("sessions"),
        "learning_log": _count("learning_log"),
        "events": _count("events"),
        "discoveries": _count("discovery_log"),
        "keys": sorted(list(data.keys())),
    }


# --------------------------------------------------------------------
# Intelligent compaction logic
# --------------------------------------------------------------------


def _compact_learning(
    learning: List[Any],
    *,
    keep_last_learning: int,
    keep_top_learning_rye: int,
    min_rye_protect: float,
    preserve_discoveries: bool,
    discovery_key: str,
) -> List[Any]:
    """
    Build a compacted learning_log that preserves:
      - the most recent learning entries
      - top RYE entries
      - any entries above a protection RYE threshold
      - flagged discoveries (if enabled)
    """
    if not isinstance(learning, list) or not learning:
        return []

    # 1) recent tail
    recent = _tail_list(learning, keep_last_learning)

    # 2) sort by RYE to find global top entries
    scored = [(idx, _extract_rye(entry), entry) for idx, entry in enumerate(learning)]
    scored.sort(key=lambda t: t[1], reverse=True)
    top_rye_entries: List[Any] = [e for _, _, e in scored[:keep_top_learning_rye] if e is not None]

    # 3) protective entries above RYE threshold
    protected: List[Any] = [e for _, rye, e in scored if rye >= min_rye_protect]

    # 4) discovery based entries
    discoveries: List[Any] = []
    if preserve_discoveries:
        for entry in learning:
            if _is_discovery(entry, discovery_key):
                discoveries.append(entry)

    # Merge all lists, dedup by content
    combined: List[Any] = []
    seen: set[str] = set()

    def _add_batch(batch: List[Any]) -> None:
        for item in batch:
            key = _learning_key(item)
            if key in seen:
                continue
            seen.add(key)
            combined.append(item)

    _add_batch(recent)
    _add_batch(top_rye_entries)
    _add_batch(protected)
    _add_batch(discoveries)

    return combined


# --------------------------------------------------------------------
# CLI and main entry
# --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact the memory JSON file to keep long term learning efficient and intelligent."
    )
    parser.add_argument(
        "--memory-file",
        type=str,
        default=None,
        help="Memory file path. Defaults to AppConfig.memory_file.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=25.0,
        help="Soft size target in megabytes after compaction.",
    )
    parser.add_argument(
        "--keep-last-sessions",
        type=int,
        default=20,
        help="Number of most recent sessions to keep.",
    )
    parser.add_argument(
        "--keep-last-learning",
        type=int,
        default=300,
        help="Number of most recent learning_log entries to always keep.",
    )
    parser.add_argument(
        "--keep-top-learning-rye",
        type=int,
        default=150,
        help="Number of highest RYE learning entries to preserve globally.",
    )
    parser.add_argument(
        "--min-rye-protect",
        type=float,
        default=0.10,
        help="Any learning entry with RYE >= this value is always preserved.",
    )
    parser.add_argument(
        "--keep-last-events",
        type=int,
        default=500,
        help="Number of most recent events to keep if present.",
    )
    parser.add_argument(
        "--preserve-discoveries",
        action="store_true",
        help="Always preserve entries marked as discoveries.",
    )
    parser.add_argument(
        "--discovery-key",
        type=str,
        default="discovery",
        help="Field or tag name used to mark discovery entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen but do not modify the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine memory path from config unless explicit
    if args.memory_file is None:
        if build_app_config is not None:
            cfg = build_app_config()
            mem_path = Path(cfg.memory_file)
        else:
            mem_path = Path(os.getenv("MEMORY_FILE", "logs/sessions/default_memory.json"))
    else:
        mem_path = Path(args.memory_file)

    print("=== Memory compaction start ===")
    print(f"Memory file: {mem_path}")
    print(f"Dry run: {args.dry_run}")
    print(
        "Targets: keep "
        f"{args.keep_last_sessions} sessions, "
        f"{args.keep_last_learning} recent learning entries, "
        f"{args.keep_top_learning_rye} top RYE learning entries, "
        f"min RYE protect {args.min_rye_protect}, "
        f"{args.keep_last_events} events, "
        f"max size {args.max_size_mb:.1f} MB (soft)"
    )
    print(f"Preserve discoveries: {args.preserve_discoveries} (key: {args.discovery_key})")
    print("")

    data, size_before = _read_json(mem_path)

    # Normalize layout before touching anything
    data = _ensure_learning_structure(data)

    summary_before = _summarize(data)

    print("Before compaction:")
    print(f"  Size: {size_before / (1024 * 1024):.2f} MB")
    print(f"  Keys: {', '.join(summary_before['keys'])}")
    print(f"  Sessions: {summary_before['sessions']}")
    print(f"  Learning entries: {summary_before['learning_log']}")
    print(f"  Discoveries: {summary_before['discoveries']}")
    print(f"  Events: {summary_before['events']}")
    print("")

    # Sessions tail compaction
    sessions = data.get("sessions")
    if isinstance(sessions, list):
        data["sessions"] = _tail_list(sessions, args.keep_last_sessions)

    # Events tail compaction
    events = data.get("events")
    if isinstance(events, list):
        data["events"] = _tail_list(events, args.keep_last_events)

    # Learning compaction with intelligence
    learning = data.get("learning_log")
    if isinstance(learning, list):
        data["learning_log"] = _compact_learning(
            learning,
            keep_last_learning=args.keep_last_learning,
            keep_top_learning_rye=args.keep_top_learning_rye,
            min_rye_protect=args.min_rye_protect,
            preserve_discoveries=args.preserve_discoveries,
            discovery_key=args.discovery_key,
        )

    # Rebuild learning stats in metadata
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    learning_log = data.get("learning_log") or []
    if isinstance(learning_log, list):
        meta["learning_stats"] = _build_learning_stats(learning_log)
    meta["last_compacted"] = datetime.utcnow().isoformat() + "Z"
    data["metadata"] = meta

    # After compaction but before writing, estimate size
    tmp_text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    size_after_est = len(tmp_text.encode("utf-8"))

    summary_after = _summarize(data)

    print("After in memory compaction:")
    print(f"  Estimated size: {size_after_est / (1024 * 1024):.2f} MB")
    print(f"  Sessions: {summary_after['sessions']}")
    print(f"  Learning entries: {summary_after['learning_log']}")
    print(f"  Discoveries: {summary_after['discoveries']}")
    print(f"  Events: {summary_after['events']}")
    print("")

    # Simple advisory if still above soft cap
    if size_after_est > args.max_size_mb * 1024 * 1024:
        print("Warning: estimated size is still above the soft max-size-mb target.")
        print("You can rerun with smaller keep-last values or a lower min_rye_protect if needed.")
        print("")

    if args.dry_run:
        print("Dry run only. No changes written.")
        print("=== Done ===")
        return

    backup_path = _make_backup(mem_path)
    if backup_path.exists():
        print(f"Backup created at: {backup_path}")

    size_written = _write_json(mem_path, data)

    print("")
    print("Compaction complete:")
    print(f"  Final size: {size_written / (1024 * 1024):.2f} MB")
    print(f"  File: {mem_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
