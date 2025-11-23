"""
snapshot_generator.py

Weekly snapshot report generator for the Autonomous Research Agent.

This module produces a structured Markdown summary of long-run activity,
including:
    - RYE trends
    - High-value discoveries
    - Hypotheses (pending, validated, rejected)
    - Tool usage statistics
    - Cycle activity counts
    - Any major contradictions resolved
    - Memory growth metrics

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


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
            memory_stats={ ... }
        )
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id

    def _md_header(self, week_number: int) -> List[str]:
        lines = []
        lines.append(f"# Weekly Snapshot Report — Week {week_number}")
        lines.append("")
        lines.append(f"**Timestamp:** {utc_iso()}")
        if self.run_id:
            lines.append(f"**Run ID:** `{self.run_id}`")
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
        try:
            return "```json\n" + json.dumps(d, indent=2, ensure_ascii=False) + "\n```"
        except Exception:
            return str(d)

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
    ) -> Path:
        """
        Build and write a snapshot Markdown report.
        Returns the path of the created file.
        """

        file_path = SNAPSHOT_DIR / f"week_{week_number}.md"

        lines: List[str] = []

        # Header
        lines.extend(self._md_header(week_number))

        # Cycle statistics
        lines.extend(self._md_section(
            "Cycle Activity",
            self._format_dict_pretty(cycle_stats or {})
        ))

        # RYE statistics
        lines.extend(self._md_section(
            "RYE Performance",
            self._format_dict_pretty(rye_stats or {})
        ))

        # Hypotheses sections
        hyp = hypotheses or {}
        lines.extend(self._md_section(
            "Hypotheses — Pending",
            self._format_list_block(hyp.get("pending", []))
        ))
        lines.extend(self._md_section(
            "Hypotheses — Validated",
            self._format_list_block(hyp.get("validated", []))
        ))
        lines.extend(self._md_section(
            "Hypotheses — Rejected",
            self._format_list_block(hyp.get("rejected", []))
        ))

        # Discoveries
        lines.extend(self._md_section(
            "High-Value Discoveries",
            self._format_list_block(discoveries or [])
        ))

        # Tool usage
        lines.extend(self._md_section(
            "Tool Usage Summary",
            self._format_dict_pretty(tool_usage or {})
        ))

        # Contradictions resolved
        lines.extend(self._md_section(
            "Contradictions Resolved",
            self._format_list_block(contradictions or [])
        ))

        # Memory statistics
        lines.extend(self._md_section(
            "Memory Metrics",
            self._format_dict_pretty(memory_stats or {})
        ))

        # Extra section if provided
        if extra:
            lines.extend(self._md_section(
                "Additional Notes",
                self._format_dict_pretty(extra)
            ))

        # Write file
        file_path.write_text("\n".join(lines), encoding="utf-8")

        return file_path
