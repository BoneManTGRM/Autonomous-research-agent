"""
discovery_log.py

Centralized discovery logging for the Autonomous Research Agent.

This module writes human readable Markdown entries to:
    logs/discovery_log.md

It is designed for:
    - Hypotheses (pending, validated, rejected)
    - High RYE events and delta R spikes
    - Contradictions resolved or repaired
    - Notable insights and structural discoveries

Typical usage from TGRM or CoreAgent:

    from pathlib import Path
    from agent.discovery_log import DiscoveryLogger

    logger = DiscoveryLogger.default()

    logger.log_hypothesis(
        title="New candidate mechanism for pathway X",
        description="Summarize the reasoning here...",
        cycle_index=128,
        agent_role="Researcher",
        rye_before=0.42,
        rye_after=0.55,
        delta_r=0.13,
        energy=2.4,
        tags=["longevity", "pathway_x", "hypothesis"]
    )

All entries are appended to a single Markdown file so you can
review discoveries after a 90 day run and export them into papers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR_DEFAULT = Path("logs")
LOG_FILE_NAME_DEFAULT = "discovery_log.md"


def _utc_iso() -> str:
    """Return an ISO formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DiscoveryEvent:
    """Structured representation of a discovery related event."""

    kind: str
    title: str
    description: str

    timestamp: str
    run_id: Optional[str] = None
    cycle_index: Optional[int] = None
    agent_role: Optional[str] = None

    rye_before: Optional[float] = None
    rye_after: Optional[float] = None
    delta_r: Optional[float] = None
    energy: Optional[float] = None

    tags: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

    def to_markdown(self) -> str:
        """Render this event as a Markdown block."""
        lines: List[str] = []

        header_kind = self.kind.strip().upper()
        header_title = self.title.strip() if self.title else "(no title)"

        # Heading
        lines.append(f"## [{self.timestamp}] {header_kind}: {header_title}")
        lines.append("")

        # Meta info as bullet list
        if self.run_id:
            lines.append(f"- Run ID: `{self.run_id}`")
        if self.cycle_index is not None:
            lines.append(f"- Cycle: `{self.cycle_index}`")
        if self.agent_role:
            lines.append(f"- Agent role: `{self.agent_role}`")

        if self.rye_before is not None or self.rye_after is not None:
            before = (
                f"{self.rye_before:.4f}"
                if isinstance(self.rye_before, (int, float))
                else str(self.rye_before)
            )
            after = (
                f"{self.rye_after:.4f}"
                if isinstance(self.rye_after, (int, float))
                else str(self.rye_after)
            )
            lines.append(f"- RYE before: `{before}`")
            lines.append(f"- RYE after: `{after}`")

        if self.delta_r is not None:
            lines.append(f"- Delta R: `{self.delta_r}`")
        if self.energy is not None:
            lines.append(f"- Energy: `{self.energy}`")

        if self.tags:
            tag_str = ", ".join(sorted(set(self.tags)))
            lines.append(f"- Tags: `{tag_str}`")

        if self.extra:
            # Store extra as compact JSON for debug and later analysis
            extra_json = json.dumps(self.extra, ensure_ascii=False, sort_keys=True)
            lines.append(f"- Extra: `{extra_json}`")

        lines.append("")
        lines.append("### Description")
        lines.append("")
        lines.append(self.description.strip() or "(no description provided)")
        lines.append("")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)


class DiscoveryLogger:
    """
    Discovery logger for Reparodynamics experiments.

    Responsibilities:
        - Ensure log directory and file exist.
        - Write well structured Markdown entries for discovery events.
        - Provide convenience methods for common event types.

    You can use one shared logger per process or per run_id.
    """

    def __init__(
        self,
        log_dir: Path | str = LOG_DIR_DEFAULT,
        file_name: str = LOG_FILE_NAME_DEFAULT,
        run_id: Optional[str] = None,
        auto_header: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.file_path = self.log_dir / file_name
        self.run_id = run_id

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if auto_header and not self.file_path.exists():
            self._write_header()

    @classmethod
    def default(cls, run_id: Optional[str] = None) -> "DiscoveryLogger":
        """
        Return a default logger that writes to logs/discovery_log.md.

        You can call this from anywhere without worrying about paths.
        """
        return cls(log_dir=LOG_DIR_DEFAULT, file_name=LOG_FILE_NAME_DEFAULT, run_id=run_id)

    def _write_header(self) -> None:
        """Write an initial header for the discovery log if the file is new."""
        header_lines = [
            "# Discovery Log",
            "",
            "This file records hypotheses, high value repairs, RYE spikes,",
            "contradictions resolved, and other potential discoveries produced",
            "by the Autonomous Research Agent running under Reparodynamics.",
            "",
            "Each entry is appended with a timestamp so you can reconstruct",
            "what the agent discovered during long runs such as a 90 day experiment.",
            "",
            "---",
            "",
        ]
        self.file_path.write_text("\n".join(header_lines), encoding="utf-8")

    def _append_markdown(self, text: str) -> None:
        """Append a Markdown block to the discovery log."""
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

    def log_event(
        self,
        kind: str,
        title: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        rye_before: Optional[float] = None,
        rye_after: Optional[float] = None,
        delta_r: Optional[float] = None,
        energy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> DiscoveryEvent:
        """
        Log a generic discovery event.

        This is the core lower level method. Other helpers call this with
        pre filled kind values for hypotheses, RYE spikes, etc.
        """
        event = DiscoveryEvent(
            kind=kind,
            title=title,
            description=description,
            timestamp=timestamp or _utc_iso(),
            run_id=self.run_id,
            cycle_index=cycle_index,
            agent_role=agent_role,
            rye_before=rye_before,
            rye_after=rye_after,
            delta_r=delta_r,
            energy=energy,
            tags=tags,
            extra=extra,
        )
        md = event.to_markdown()
        self._append_markdown(md)
        return event

    # Convenience helpers

    def log_hypothesis(
        self,
        title: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        rye_before: Optional[float] = None,
        rye_after: Optional[float] = None,
        delta_r: Optional[float] = None,
        energy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryEvent:
        """Log a new candidate hypothesis."""
        combined_tags = (tags or []) + ["hypothesis", "pending"]
        return self.log_event(
            kind="hypothesis",
            title=title,
            description=description,
            cycle_index=cycle_index,
            agent_role=agent_role,
            rye_before=rye_before,
            rye_after=rye_after,
            delta_r=delta_r,
            energy=energy,
            tags=combined_tags,
            extra=extra,
        )

    def log_validated_hypothesis(
        self,
        title: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        rye_before: Optional[float] = None,
        rye_after: Optional[float] = None,
        delta_r: Optional[float] = None,
        energy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryEvent:
        """Log a hypothesis that passed additional checks or verification."""
        combined_tags = (tags or []) + ["hypothesis", "validated"]
        return self.log_event(
            kind="validated_hypothesis",
            title=title,
            description=description,
            cycle_index=cycle_index,
            agent_role=agent_role,
            rye_before=rye_before,
            rye_after=rye_after,
            delta_r=delta_r,
            energy=energy,
            tags=combined_tags,
            extra=extra,
        )

    def log_rejected_hypothesis(
        self,
        title: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryEvent:
        """Log a hypothesis that was rejected after critique or contradiction."""
        combined_tags = (tags or []) + ["hypothesis", "rejected"]
        return self.log_event(
            kind="rejected_hypothesis",
            title=title,
            description=description,
            cycle_index=cycle_index,
            agent_role=agent_role,
            tags=combined_tags,
            extra=extra,
        )

    def log_rye_spike(
        self,
        title: str,
        description: str,
        cycle_index: Optional[int],
        agent_role: Optional[str],
        rye_before: float,
        rye_after: float,
        delta_r: Optional[float] = None,
        energy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryEvent:
        """
        Log a high RYE event, likely associated with a significant repair or insight.
        """
        combined_tags = (tags or []) + ["rye_spike", "high_value"]
        return self.log_event(
            kind="rye_spike",
            title=title,
            description=description,
            cycle_index=cycle_index,
            agent_role=agent_role,
            rye_before=rye_before,
            rye_after=rye_after,
            delta_r=delta_r,
            energy=energy,
            tags=combined_tags,
            extra=extra,
        )

    def log_contradiction_resolved(
        self,
        title: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryEvent:
        """Log an event where a contradiction or inconsistency was resolved."""
        combined_tags = (tags or []) + ["contradiction_resolved"]
        return self.log_event(
            kind="contradiction_resolved",
            title=title,
            description=description,
            cycle_index=cycle_index,
            agent_role=agent_role,
            tags=combined_tags,
            extra=extra,
        )


# Optional: singleton style logger for very simple integrations

_GLOBAL_LOGGER: Optional[DiscoveryLogger] = None


def get_global_logger(run_id: Optional[str] = None) -> DiscoveryLogger:
    """
    Return a process global DiscoveryLogger.

    Useful when you do not want to pass a logger instance through many layers.
    """
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = DiscoveryLogger.default(run_id=run_id)
    else:
        # If a run_id is provided later, update it
        if run_id is not None:
            _GLOBAL_LOGGER.run_id = run_id
    return _GLOBAL_LOGGER
