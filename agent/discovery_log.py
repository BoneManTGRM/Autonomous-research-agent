"""
discovery_log.py

Centralized discovery logging for the Autonomous Research Agent.

This module writes human readable Markdown entries to:
    logs/discovery_log.md

It can also mirror structured JSONL entries to:
    logs/discovery_log.jsonl

It is designed for:
    - Hypotheses (pending, validated, rejected)
    - High RYE events and delta R spikes
    - Contradictions resolved or repaired
    - Notable insights and structural discoveries
    - Mechanisms, treatments, biomarkers, and cure candidates
    - Run milestones and swarm or tool events
    - Long run stability and phase shifts

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

All entries are appended so you can review discoveries after a long run
and export them into papers or structured summaries.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR_DEFAULT = Path("logs")
LOG_FILE_NAME_DEFAULT = "discovery_log.md"
LOG_JSON_FILE_NAME_DEFAULT = "discovery_log.jsonl"
DISCOVERY_LOG_VERSION = "2025-11-23-max2"


def _utc_iso() -> str:
    """Return an ISO formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_event_id() -> str:
    """Generate a short stable event id for cross references."""
    return uuid.uuid4().hex[:12]


@dataclass
class DiscoveryEvent:
    """Structured representation of a discovery related event."""

    kind: str
    title: str
    description: str

    timestamp: str

    # Unique identifier for cross linking
    event_id: str

    # Run and cycle context
    run_id: Optional[str] = None
    cycle_index: Optional[int] = None
    agent_role: Optional[str] = None
    goal: Optional[str] = None
    domain: Optional[str] = None

    # RYE and repair metrics
    rye_before: Optional[float] = None
    rye_after: Optional[float] = None
    delta_r: Optional[float] = None
    energy: Optional[float] = None

    # Search and information quality metrics
    info_gain: Optional[float] = None
    search_energy: Optional[float] = None
    semantic_diversity: Optional[float] = None

    # Swarm context
    swarm_size: Optional[int] = None
    swarm_config: Optional[Dict[str, Any]] = None

    # Classification and metadata
    tags: Optional[List[str]] = None
    severity: Optional[str] = None       # "info", "notice", "major", "critical"
    confidence: Optional[float] = None   # 0 to 1
    extra: Optional[Dict[str, Any]] = None

    def rye_ratio(self) -> Optional[float]:
        """Return delta R per unit energy if both are available and energy is non zero."""
        if self.delta_r is None or self.energy in (None, 0):
            return None
        try:
            return float(self.delta_r) / float(self.energy)
        except Exception:
            return None

    def to_markdown(self) -> str:
        """Render this event as a Markdown block."""
        lines: List[str] = []

        header_kind = self.kind.strip().upper()
        header_title = self.title.strip() if self.title else "(no title)"

        # Heading
        lines.append(f"## [{self.timestamp}] {header_kind}: {header_title}")
        lines.append("")
        lines.append(f"- Event ID: `{self.event_id}`")

        # Meta info as bullet list
        if self.run_id:
            lines.append(f"- Run ID: `{self.run_id}`")
        if self.cycle_index is not None:
            lines.append(f"- Cycle: `{self.cycle_index}`")
        if self.agent_role:
            lines.append(f"- Agent role: `{self.agent_role}`")
        if self.goal:
            lines.append(f"- Goal: `{self.goal}`")
        if self.domain:
            lines.append(f"- Domain: `{self.domain}`")

        # Swarm context
        if self.swarm_size is not None:
            lines.append(f"- Swarm size: `{self.swarm_size}`")
        if self.swarm_config:
            try:
                swarm_json = json.dumps(self.swarm_config, ensure_ascii=False, sort_keys=True)
            except Exception:
                swarm_json = str(self.swarm_config)
            lines.append(f"- Swarm config: `{swarm_json}`")

        # RYE metrics
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

        rr = self.rye_ratio()
        if rr is not None:
            lines.append(f"- R per energy (RYE ratio): `{rr:.4f}`")

        # Search and information quality
        if self.info_gain is not None:
            lines.append(f"- Info gain: `{self.info_gain}`")
        if self.search_energy is not None:
            lines.append(f"- Search energy: `{self.search_energy}`")
        if self.semantic_diversity is not None:
            lines.append(f"- Semantic diversity: `{self.semantic_diversity}`")

        if self.severity:
            lines.append(f"- Severity: `{self.severity}`")
        if isinstance(self.confidence, (int, float)):
            lines.append(f"- Confidence: `{self.confidence:.2f}`")

        if self.tags:
            tag_str = ", ".join(sorted(set(self.tags)))
            lines.append(f"- Tags: `{tag_str}`")

        if self.extra:
            # Store extra as compact JSON for debug and later analysis
            try:
                extra_json = json.dumps(self.extra, ensure_ascii=False, sort_keys=True)
            except Exception:
                extra_json = str(self.extra)
            lines.append(f"- Extra: `{extra_json}`")

        lines.append("")
        lines.append("### Description")
        lines.append("")
        lines.append(self.description.strip() or "(no description provided)")
        lines.append("")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def to_json_ready(self) -> Dict[str, Any]:
        """
        Return a dict ready for JSONL logging.
        Includes explicit R per energy and log version for later parsing.
        """
        data = asdict(self)
        data["rye_ratio"] = self.rye_ratio()
        data["log_version"] = DISCOVERY_LOG_VERSION
        return data


class DiscoveryLogger:
    """
    Discovery logger for Reparodynamics experiments.

    Responsibilities:
        - Ensure log directory and files exist.
        - Write well structured Markdown entries for discovery events.
        - Optionally mirror structured JSONL entries for machine analysis.
        - Provide convenience methods for common event types.

    You can use one shared logger per process or per run_id.
    """

    def __init__(
        self,
        log_dir: Path | str = LOG_DIR_DEFAULT,
        file_name: str = LOG_FILE_NAME_DEFAULT,
        run_id: Optional[str] = None,
        auto_header: bool = True,
        json_mirror: bool = True,
        json_file_name: str = LOG_JSON_FILE_NAME_DEFAULT,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.file_path = self.log_dir / file_name
        self.json_file_path = self.log_dir / json_file_name
        self.run_id = run_id
        self.json_mirror = json_mirror

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if auto_header and not self.file_path.exists():
            self._write_header()

    @classmethod
    def default(cls, run_id: Optional[str] = None) -> "DiscoveryLogger":
        """
        Return a default logger that writes to logs/discovery_log.md
        and logs/discovery_log.jsonl.

        You can call this from anywhere without worrying about paths.
        """
        return cls(
            log_dir=LOG_DIR_DEFAULT,
            file_name=LOG_FILE_NAME_DEFAULT,
            run_id=run_id,
            json_mirror=True,
            json_file_name=LOG_JSON_FILE_NAME_DEFAULT,
        )

    @classmethod
    def from_config(
        cls,
        base_dir: Path | str,
        run_id: Optional[str] = None,
        for_worker: bool = False,
    ) -> "DiscoveryLogger":
        """
        Helper for engine_worker or CoreAgent to create a logger that
        lives alongside other run specific logs.

        If for_worker is True, uses a slightly different file name.
        """
        base = Path(base_dir)
        logs_dir = base / "logs"
        if for_worker:
            md_name = "discovery_log_worker.md"
            json_name = "discovery_log_worker.jsonl"
        else:
            md_name = LOG_FILE_NAME_DEFAULT
            json_name = LOG_JSON_FILE_NAME_DEFAULT
        return cls(
            log_dir=logs_dir,
            file_name=md_name,
            run_id=run_id,
            json_mirror=True,
            json_file_name=json_name,
        )

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
            "what the agent discovered during long runs, including 24 hour",
            "or 90 day stability and swarm experiments.",
            "",
            f"Log schema version: `{DISCOVERY_LOG_VERSION}`",
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

    def _append_json(self, event: DiscoveryEvent) -> None:
        """Append a JSON line representation of the event if json mirroring is enabled."""
        if not self.json_mirror:
            return
        record = event.to_json_ready()
        line = json.dumps(record, ensure_ascii=False)
        with self.json_file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Core logging primitive

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
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        severity: Optional[str] = None,
        confidence: Optional[float] = None,
        info_gain: Optional[float] = None,
        search_energy: Optional[float] = None,
        semantic_diversity: Optional[float] = None,
        swarm_size: Optional[int] = None,
        swarm_config: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> DiscoveryEvent:
        """
        Log a generic discovery event.

        This is the core lower level method. Other helpers call this with
        pre filled kind values for hypotheses, RYE spikes, mechanisms, etc.

        New optional fields:
            info_gain, search_energy, semantic_diversity:
                Used to mirror web search and TGRM stack quality metrics.
            swarm_size, swarm_config:
                Used to tag swarm scale and configuration for the event.
            event_id:
                Optional override if caller wants to set a fixed id.
        """
        event = DiscoveryEvent(
            kind=kind,
            title=title,
            description=description,
            timestamp=timestamp or _utc_iso(),
            event_id=event_id or _new_event_id(),
            run_id=self.run_id,
            cycle_index=cycle_index,
            agent_role=agent_role,
            goal=goal,
            domain=domain,
            rye_before=rye_before,
            rye_after=rye_after,
            delta_r=delta_r,
            energy=energy,
            tags=tags,
            severity=severity,
            confidence=confidence,
            extra=extra,
            info_gain=info_gain,
            search_energy=search_energy,
            semantic_diversity=semantic_diversity,
            swarm_size=swarm_size,
            swarm_config=swarm_config,
        )
        md = event.to_markdown()
        self._append_markdown(md)
        self._append_json(event)
        return event

    # ------------------------------------------------------------------
    # Convenience helpers for hypotheses
    # ------------------------------------------------------------------

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
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
        info_gain: Optional[float] = None,
        search_energy: Optional[float] = None,
        semantic_diversity: Optional[float] = None,
        swarm_size: Optional[int] = None,
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
            goal=goal,
            domain=domain,
            severity="info",
            confidence=confidence,
            info_gain=info_gain,
            search_energy=search_energy,
            semantic_diversity=semantic_diversity,
            swarm_size=swarm_size,
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
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
        info_gain: Optional[float] = None,
        search_energy: Optional[float] = None,
        semantic_diversity: Optional[float] = None,
        swarm_size: Optional[int] = None,
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
            goal=goal,
            domain=domain,
            severity="major",
            confidence=confidence,
            info_gain=info_gain,
            search_energy=search_energy,
            semantic_diversity=semantic_diversity,
            swarm_size=swarm_size,
        )

    def log_rejected_hypothesis(
        self,
        title: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
        info_gain: Optional[float] = None,
        search_energy: Optional[float] = None,
        semantic_diversity: Optional[float] = None,
        swarm_size: Optional[int] = None,
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
            goal=goal,
            domain=domain,
            severity="notice",
            confidence=confidence,
            info_gain=info_gain,
            search_energy=search_energy,
            semantic_diversity=semantic_diversity,
            swarm_size=swarm_size,
        )

    # ------------------------------------------------------------------
    # RYE spikes and major repair events
    # ------------------------------------------------------------------

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
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        severity: str = "major",
        confidence: Optional[float] = None,
        info_gain: Optional[float] = None,
        search_energy: Optional[float] = None,
        semantic_diversity: Optional[float] = None,
        swarm_size: Optional[int] = None,
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
            goal=goal,
            domain=domain,
            severity=severity,
            confidence=confidence,
            info_gain=info_gain,
            search_energy=search_energy,
            semantic_diversity=semantic_diversity,
            swarm_size=swarm_size,
        )

    # ------------------------------------------------------------------
    # Contradictions and repairs
    # ------------------------------------------------------------------

    def log_contradiction_resolved(
        self,
        title: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        severity: str = "info",
        confidence: Optional[float] = None,
        info_gain: Optional[float] = None,
        search_energy: Optional[float] = None,
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
            goal=goal,
            domain=domain,
            severity=severity,
            confidence=confidence,
            info_gain=info_gain,
            search_energy=search_energy,
        )

    # ------------------------------------------------------------------
    # Mechanisms, treatments, biomarkers, and structures
    # ------------------------------------------------------------------

    def log_mechanism(
        self,
        label: str,
        evidence_summary: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> DiscoveryEvent:
        """Log a mechanism discovery or candidate mechanism."""
        extra: Dict[str, Any] = {"citations": citations or []}
        combined_tags = (tags or []) + ["mechanism", "discovery"]
        return self.log_event(
            kind="mechanism",
            title=label,
            description=evidence_summary,
            cycle_index=cycle_index,
            agent_role=agent_role,
            tags=combined_tags,
            extra=extra,
            goal=goal,
            domain=domain,
            severity="major",
            confidence=confidence,
        )

    def log_treatment_candidate(
        self,
        label: str,
        evidence_summary: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> DiscoveryEvent:
        """Log a treatment or stack candidate for cure extraction pipelines."""
        extra: Dict[str, Any] = {"citations": citations or []}
        combined_tags = (tags or []) + ["treatment", "candidate", "intervention"]
        return self.log_event(
            kind="treatment_candidate",
            title=label,
            description=evidence_summary,
            cycle_index=cycle_index,
            agent_role=agent_role,
            tags=combined_tags,
            extra=extra,
            goal=goal,
            domain=domain,
            severity="major",
            confidence=confidence,
        )

    def log_cure_candidate(
        self,
        label: str,
        evidence_summary: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> DiscoveryEvent:
        """Log a very strong cure candidate with high RYE and verification needs."""
        extra: Dict[str, Any] = {"citations": citations or []}
        combined_tags = (tags or []) + ["cure_candidate", "high_stakes"]
        return self.log_event(
            kind="cure_candidate",
            title=label,
            description=evidence_summary,
            cycle_index=cycle_index,
            agent_role=agent_role,
            tags=combined_tags,
            extra=extra,
            goal=goal,
            domain=domain,
            severity="critical",
            confidence=confidence,
        )

    def log_biomarker_shift(
        self,
        label: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> DiscoveryEvent:
        """Log a notable biomarker pattern or shift."""
        combined_tags = (tags or []) + ["biomarker_shift"]
        return self.log_event(
            kind="biomarker_shift",
            title=label,
            description=description,
            cycle_index=cycle_index,
            agent_role=agent_role,
            tags=combined_tags,
            extra=extra,
            goal=goal,
            domain=domain,
            severity="notice",
            confidence=confidence,
        )

    def log_structure_discovery(
        self,
        label: str,
        description: str,
        cycle_index: Optional[int] = None,
        agent_role: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> DiscoveryEvent:
        """Log a structural discovery such as a mathematical framework or pattern."""
        combined_tags = (tags or []) + ["mathematical_structure", "pattern"]
        return self.log_event(
            kind="structure_discovery",
            title=label,
            description=description,
            cycle_index=cycle_index,
            agent_role=agent_role,
            tags=combined_tags,
            extra=extra,
            goal=goal,
            domain=domain,
            severity="major",
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Run milestones and swarm or tool events
    # ------------------------------------------------------------------

    def log_run_milestone(
        self,
        label: str,
        description: str,
        cycle_index: Optional[int] = None,
        run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        severity: str = "info",
        swarm_size: Optional[int] = None,
    ) -> DiscoveryEvent:
        """Log a run milestone such as best RYE so far or phase shift detection."""
        if run_id is not None:
            self.run_id = run_id
        combined_tags = (tags or []) + ["milestone"]
        return self.log_event(
            kind="run_milestone",
            title=label,
            description=description,
            cycle_index=cycle_index,
            agent_role=None,
            tags=combined_tags,
            extra=extra,
            goal=goal,
            domain=domain,
            severity=severity,
            swarm_size=swarm_size,
        )

    def log_swarm_event(
        self,
        label: str,
        description: str,
        cycle_index: Optional[int] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        severity: str = "info",
        swarm_size: Optional[int] = None,
        swarm_config: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryEvent:
        """Log a swarm related event, such as rebalancing or consensus switch."""
        combined_tags = (tags or []) + ["swarm_event"]
        return self.log_event(
            kind="swarm_event",
            title=label,
            description=description,
            cycle_index=cycle_index,
            agent_role="swarm_controller",
            tags=combined_tags,
            extra=extra,
            goal=goal,
            domain=domain,
            severity=severity,
            swarm_size=swarm_size,
            swarm_config=swarm_config,
        )

    def log_tool_event(
        self,
        tool_name: str,
        description: str,
        cycle_index: Optional[int] = None,
        status: str = "success",
        duration_seconds: Optional[float] = None,
        energy_cost: Optional[float] = None,
        rye_delta: Optional[float] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        domain: Optional[str] = None,
        info_gain: Optional[float] = None,
        search_energy: Optional[float] = None,
    ) -> DiscoveryEvent:
        """
        Log a tool event that is particularly important for discovery,
        for example a long browser crawl or a large data pipeline run.
        """
        tool_tags = (tags or []) + ["tool_event", tool_name, status]
        payload: Dict[str, Any] = extra.copy() if extra else {}
        payload.update(
            {
                "tool_name": tool_name,
                "status": status,
                "duration_seconds": duration_seconds,
                "energy_cost": energy_cost,
                "rye_delta": rye_delta,
            }
        )
        severity = "info" if status == "success" else "notice"
        if status == "error":
            severity = "major"
        return self.log_event(
            kind="tool_event",
            title=f"Tool {tool_name} {status}",
            description=description,
            cycle_index=cycle_index,
            tags=tool_tags,
            extra=payload,
            goal=goal,
            domain=domain,
            severity=severity,
            info_gain=info_gain,
            search_energy=search_energy,
        )

    # ------------------------------------------------------------------
    # Integration helpers for MemoryStore style discovery entries
    # ------------------------------------------------------------------

    def log_from_memory_discovery(self, entry: Dict[str, Any]) -> DiscoveryEvent:
        """
        Bridge helper to log a discovery entry coming from MemoryStore.

        Expected MemoryStore shape (best effort, all optional):
            {
                "timestamp": str,
                "goal": str,
                "kind": str,
                "label": str,
                "evidence_summary": str,
                "score": float | None,
                "tags": list,
                "citations": list,
                "domain": str | None,
            }
        """
        kind = str(entry.get("kind", "discovery") or "discovery")
        label = str(entry.get("label", "(no label)") or "(no label)")
        evidence = str(entry.get("evidence_summary", "") or "")
        ts = str(entry.get("timestamp") or _utc_iso())
        goal = entry.get("goal")
        domain = entry.get("domain")
        tags = list(entry.get("tags") or [])
        citations = entry.get("citations") or []
        score = entry.get("score")

        extra: Dict[str, Any] = {
            "citations": citations,
            "memory_entry": entry,
        }

        severity = "major"
        if kind == "cure_candidate":
            severity = "critical"
        elif kind == "treatment":
            severity = "major"
        elif kind in ("mechanism", "biomarker"):
            severity = "major"
        else:
            severity = "info"

        return self.log_event(
            kind=kind,
            title=label,
            description=evidence,
            cycle_index=None,
            agent_role=None,
            tags=tags,
            extra=extra,
            timestamp=ts,
            goal=goal,
            domain=domain,
            severity=severity,
            confidence=score if isinstance(score, (int, float)) else None,
        )


# Optional singleton style logger for very simple integrations

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
        if run_id is not None:
            _GLOBAL_LOGGER.run_id = run_id
    return _GLOBAL_LOGGER
