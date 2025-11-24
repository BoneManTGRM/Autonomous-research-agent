"""
snapshot_generator.py

Weekly snapshot report generator for the Autonomous Research Agent.

This module produces a structured Markdown summary of long run activity, including:
    - RYE trends and statistics
    - High value discoveries and hypotheses status
    - Tool usage statistics and mode balance
    - Cycle activity counts and stability hints
    - Contradictions resolved and verification health
    - Memory growth metrics and pruning diagnostics
    - Swarm and intelligence profile summaries

Output folder:
    logs/snapshots/week_<number>.md

Trigger this from:
    - engine_worker.py (weekly schedule)
    - core_agent.py (periodic call)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path("logs")
SNAPSHOT_DIR = LOG_DIR / "snapshots"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_VERSION: str = "2025-11-23"


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_json(obj: Any) -> str:
    try:
        return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"
    except Exception:
        return str(obj)


class SnapshotGenerator:
    """
    Build and write weekly snapshot Markdown files.

    Expected usage:

        sg = SnapshotGenerator(run_id="run_001")
        sg.generate(
            week_number=3,
            cycle_stats={"cycles_total": 2048, "cycles_good": 1800},
            rye_stats={"avg": 0.42, "max": 0.88},
            hypotheses={
                "pending": [...],
                "validated": [...],
                "rejected": [...]
            },
            discoveries=[ ... ],
            tool_usage={ ... },
            contradictions=[ ... ],
            memory_stats={ ... },
            extra={ ... },
            run_meta={
                "domain": "longevity",
                "preset": "longevity",
                "runtime_profile": "24_hours",
                "intelligence_profile_name": "longevity_clinical",
            },
            swarm_stats={
                "enabled": True,
                "avg_agents": 5,
                "max_agents": 8,
                "roles_used": {"researcher": 180, "critic": 120, "integrator": 75},
            },
            pruning_summary={ ... },
            intelligence_profile={ ... },
            auto_from_logs=False,
        )
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id

    # --------------------------
    # Markdown helpers
    # --------------------------
    def _md_header(
        self,
        week_number: int,
        run_meta: Optional[Dict[str, Any]],
    ) -> List[str]:
        lines: List[str] = []

        # Machine readable header block
        header_meta: Dict[str, Any] = {
            "snapshot_version": SNAPSHOT_VERSION,
            "week_number": week_number,
            "timestamp": utc_iso(),
            "run_id": self.run_id,
        }
        if run_meta:
            header_meta["run_meta"] = run_meta

        lines.append("```json")
        lines.append(json.dumps(header_meta, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        # Human readable header
        lines.append(f"# Weekly Snapshot Report - Week {week_number}")
        lines.append("")
        lines.append(f"**Timestamp:** {header_meta['timestamp']}")
        if self.run_id:
            lines.append(f"**Run ID:** `{self.run_id}`")
        if run_meta:
            domain = run_meta.get("domain")
            preset = run_meta.get("preset")
            runtime_profile = run_meta.get("runtime_profile")
            intel_name = run_meta.get("intelligence_profile_name")
            if domain:
                lines.append(f"**Domain:** `{domain}`")
            if preset:
                lines.append(f"**Preset:** `{preset}`")
            if runtime_profile:
                lines.append(f"**Runtime profile:** `{runtime_profile}`")
            if intel_name:
                lines.append(f"**Intelligence profile:** `{intel_name}`")
        lines.append("")
        lines.append(f"_Snapshot engine version: {SNAPSHOT_VERSION}_")
        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    def _md_section(self, title: str, content: str) -> List[str]:
        return [f"## {title}", "", content, "", "---", ""]

    def _format_list_block(self, items: List[Any]) -> str:
        if not items:
            return "_None recorded._"
        return "\n".join(f"- {item}" for item in items)

    def _format_dict_pretty(self, d: Dict[str, Any]) -> str:
        if not d:
            return "_None recorded._"
        return _safe_json(d)

    # --------------------------
    # Optional auto log readers
    # --------------------------
    def _tail_file(self, path: Path, max_lines: int = 40) -> Optional[str]:
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        lines = text.splitlines()
        tail = lines[-max_lines:]
        return "\n".join(tail)

    def _auto_discovery_summary(self, limit_lines: int = 40) -> Optional[str]:
        from_discovery = self._tail_file(LOG_DIR / "discovery_log.md", max_lines=limit_lines)
        if not from_discovery:
            return None
        return (
            "Recent discovery log tail (see `logs/discovery_log.md` for full history):\n\n"
            "```markdown\n"
            + from_discovery
            + "\n```"
        )

    def _auto_pruning_summary(self, limit_lines: int = 40) -> Optional[str]:
        from_prune = self._tail_file(LOG_DIR / "memory_pruning_log.md", max_lines=limit_lines)
        if not from_prune:
            return None
        return (
            "Recent memory pruning log tail (see `logs/memory_pruning_log.md` for full history):\n\n"
            "```markdown\n"
            + from_prune
            + "\n```"
        )

    # --------------------------
    # Main generator
    # --------------------------
    def generate(
        self,
        week_number: int,
        cycle_stats: Optional[Dict[str, Any]] = None,
        rye_stats: Optional[Dict[str, Any]] = None,
        hypotheses: Optional[Dict[str, List[str]]] = None,
        discoveries: Optional[List[str]] = None,
        tool_usage: Optional[Dict[str, Any]] = None,
        contradictions: Optional[List[str]] = None,
        memory_stats: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        *,
        run_meta: Optional[Dict[str, Any]] = None,
        swarm_stats: Optional[Dict[str, Any]] = None,
        pruning_summary: Optional[Dict[str, Any]] = None,
        intelligence_profile: Optional[Dict[str, Any]] = None,
        auto_from_logs: bool = False,
    ) -> Path:
        """
        Build and write a snapshot Markdown report.
        Returns the path of the created file.

        All original arguments remain supported. New fields:

            run_meta:
                High level context for the run, for example:
                    {
                        "domain": "longevity",
                        "preset": "longevity",
                        "runtime_profile": "24_hours",
                        "intelligence_profile_name": "longevity_clinical",
                    }

            swarm_stats:
                Swarm operation summary:
                    {
                        "enabled": True,
                        "avg_agents": 5,
                        "max_agents": 8,
                        "roles_used": {"researcher": 180, "critic": 120, ...},
                        "mode": "round_robin",
                    }

            pruning_summary:
                Latest memory_pruner summary dict.

            intelligence_profile:
                Resolved intelligence profile dict from intelligence_profiles.py.

            auto_from_logs:
                If True and some sections are missing, the generator will try to
                pull recent tails from:
                    logs/discovery_log.md
                    logs/memory_pruning_log.md
        """

        file_path = SNAPSHOT_DIR / f"week_{week_number}.md"

        # Optionally enrich from logs
        auto_discovery_block = None
        auto_prune_block = None
        if auto_from_logs:
            if not discoveries:
                auto_discovery_block = self._auto_discovery_summary()
            if not pruning_summary:
                auto_prune_block = self._auto_pruning_summary()

        lines: List[str] = []

        # Header
        lines.extend(self._md_header(week_number, run_meta=run_meta))

        # Run meta section (more detailed)
        if run_meta or intelligence_profile or swarm_stats:
            meta_block: Dict[str, Any] = {}
            if run_meta:
                meta_block["run_meta"] = run_meta
            if intelligence_profile:
                meta_block["intelligence_profile"] = intelligence_profile
            if swarm_stats:
                meta_block["swarm_stats"] = swarm_stats
            lines.extend(self._md_section("Run Context", self._format_dict_pretty(meta_block)))

        # Cycle statistics
        lines.extend(
            self._md_section(
                "Cycle Activity",
                self._format_dict_pretty(cycle_stats or {}),
            )
        )

        # RYE statistics and hints
        rye_block = rye_stats or {}
        if rye_block:
            hints: List[str] = []
            avg_val = rye_block.get("avg") or rye_block.get("mean")
            max_val = rye_block.get("max")
            trend = rye_block.get("trend")
            if isinstance(avg_val, (int, float)):
                if avg_val >= 0.10:
                    hints.append(f"- Average RYE is in the good zone at ~{avg_val:.3f}.")
                elif avg_val >= 0.05:
                    hints.append(f"- Average RYE is in maintenance range at ~{avg_val:.3f}.")
                else:
                    hints.append(f"- Average RYE is in early repair range at ~{avg_val:.3f}.")
            if isinstance(max_val, (int, float)):
                hints.append(f"- Peak RYE this week reached ~{max_val:.3f}.")
            if isinstance(trend, (int, float)):
                if trend > 0.0:
                    hints.append("- RYE trend is improving.")
                elif trend < 0.0:
                    hints.append("- RYE trend is declining; consider rechecking prompts and tools.")
            if hints:
                rye_block["interpretation"] = hints
        lines.extend(
            self._md_section(
                "RYE Performance",
                self._format_dict_pretty(rye_block),
            )
        )

        # Hypotheses sections
        hyp = hypotheses or {}
        lines.extend(
            self._md_section(
                "Hypotheses - Pending",
                self._format_list_block(hyp.get("pending", [])),
            )
        )
        lines.extend(
            self._md_section(
                "Hypotheses - Validated",
                self._format_list_block(hyp.get("validated", [])),
            )
        )
        lines.extend(
            self._md_section(
                "Hypotheses - Rejected",
                self._format_list_block(hyp.get("rejected", [])),
            )
        )

        # Discoveries
        if discoveries:
            discoveries_block = self._format_list_block(discoveries)
        elif auto_discovery_block:
            discoveries_block = auto_discovery_block
        else:
            discoveries_block = "_None recorded or not yet summarized._"

        lines.extend(self._md_section("High Value Discoveries", discoveries_block))

        # Tool usage
        lines.extend(
            self._md_section(
                "Tool Usage Summary",
                self._format_dict_pretty(tool_usage or {}),
            )
        )

        # Contradictions resolved
        lines.extend(
            self._md_section(
                "Contradictions Resolved",
                self._format_list_block(contradictions or []),
            )
        )

        # Memory statistics
        mem_block = memory_stats or {}
        if pruning_summary:
            mem_block = {**mem_block, "latest_pruning_summary": pruning_summary}
        elif auto_prune_block:
            mem_block = {
                **mem_block,
                "latest_pruning_log_tail": auto_prune_block,
            }

        lines.extend(
            self._md_section(
                "Memory Metrics and Pruning",
                self._format_dict_pretty(mem_block),
            )
        )

        # Extra section if provided
        if extra:
            lines.extend(
                self._md_section(
                    "Additional Notes",
                    self._format_dict_pretty(extra),
                )
            )

        # Write file
        file_path.write_text("\n".join(lines), encoding="utf-8")

        return file_path
