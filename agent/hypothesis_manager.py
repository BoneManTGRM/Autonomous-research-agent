"""
hypothesis_manager.py

Hypothesis pipeline manager for the Autonomous Research Agent.

Folders created automatically at project root:

    hypotheses/
        pending/
        validated/
        rejected/

Each hypothesis is stored as a Markdown file with a small JSON metadata
header. This lets you inspect them by hand and also parse them later
for reports or papers.

Typical usage:

    from agent.hypothesis_manager import HypothesisManager

    hm = HypothesisManager(run_id="run_001")

    h = hm.create_hypothesis(
        title="New candidate longevity pathway",
        description="Reasoning and evidence here...",
        cycle_index=128,
        agent_role="Researcher",
        rye_before=0.41,
        rye_after=0.53,
        tags=["longevity", "pathway_x"]
    )

    hm.validate_hypothesis(h.hypothesis_id, note="Survived additional checks.")
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT_DIR = Path("hypotheses")
PENDING_DIR = ROOT_DIR / "pending"
VALIDATED_DIR = ROOT_DIR / "validated"
REJECTED_DIR = ROOT_DIR / "rejected"

for d in (PENDING_DIR, VALIDATED_DIR, REJECTED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(text: str, max_length: int = 40) -> str:
    """
    Turn a title into a filesystem friendly slug.
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    return text or "hypothesis"


@dataclass
class HypothesisRecord:
    """
    Structured metadata for a hypothesis.
    """

    hypothesis_id: str
    status: str  # "pending", "validated", "rejected"

    title: str
    description: str

    created_at: str
    updated_at: str

    run_id: Optional[str] = None
    cycle_index: Optional[int] = None
    agent_role: Optional[str] = None

    rye_before: Optional[float] = None
    rye_after: Optional[float] = None
    delta_r: Optional[float] = None
    energy: Optional[float] = None

    tags: Optional[List[str]] = None
    notes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HypothesisManager:
    """
    Manages the lifecycle of hypotheses:
        - creation (pending)
        - validation
        - rejection
        - listing for snapshots and reports
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id

    # ---------- internal helpers ----------

    def _status_dir(self, status: str) -> Path:
        if status == "pending":
            return PENDING_DIR
        if status == "validated":
            return VALIDATED_DIR
        if status == "rejected":
            return REJECTED_DIR
        raise ValueError(f"Unknown hypothesis status: {status}")

    def _build_file_path(self, status: str, title: str, hypothesis_id: str) -> Path:
        slug = _slugify(title)
        filename = f"{slug}_{hypothesis_id}.md"
        return self._status_dir(status) / filename

    def _write_markdown(self, record: HypothesisRecord, path: Path) -> None:
        """
        Write hypothesis metadata and description to a Markdown file.

        Format:

        ```json
        { ...metadata... }
        ```

        # Title
        Description...
        """
        metadata_json = json.dumps(
            record.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
        )

        lines: List[str] = []
        lines.append("```json")
        lines.append(metadata_json)
        lines.append("```")
        lines.append("")
        lines.append(f"# {record.title}")
        lines.append("")
        lines.append(record.description.strip() or "(no description provided)")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _parse_metadata_from_file(self, path: Path) -> Optional[HypothesisRecord]:
        """
        Read the JSON metadata block from a Markdown file and return a record.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

        # Expect metadata JSON between the first ```json and ``` fence
        start = text.find("```json")
        if start == -1:
            return None
        start = text.find("\n", start)
        if start == -1:
            return None
        end = text.find("```", start)
        if end == -1:
            return None

        metadata_str = text[start:end].strip()
        if not metadata_str:
            return None

        try:
            data = json.loads(metadata_str)
        except json.JSONDecodeError:
            return None

        try:
            return HypothesisRecord(**data)
        except TypeError:
            return None

    def _find_file_by_id(self, hypothesis_id: str) -> Optional[Tuple[Path, HypothesisRecord]]:
        """
        Search all status folders for a hypothesis_id and return (path, record).
        """
        for status_dir in (PENDING_DIR, VALIDATED_DIR, REJECTED_DIR):
            for path in status_dir.glob("*.md"):
                rec = self._parse_metadata_from_file(path)
                if rec and rec.hypothesis_id == hypothesis_id:
                    return path, rec
        return None

    # ---------- public API ----------

    def create_hypothesis(
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
    ) -> HypothesisRecord:
        """
        Create a new pending hypothesis and write it to the pending folder.
        """
        now = _utc_iso()
        hypothesis_id = str(uuid.uuid4())

        record = HypothesisRecord(
            hypothesis_id=hypothesis_id,
            status="pending",
            title=title,
            description=description,
            created_at=now,
            updated_at=now,
            run_id=self.run_id,
            cycle_index=cycle_index,
            agent_role=agent_role,
            rye_before=rye_before,
            rye_after=rye_after,
            delta_r=delta_r,
            energy=energy,
            tags=tags or [],
            notes=[],
        )

        path = self._build_file_path("pending", title, hypothesis_id)
        self._write_markdown(record, path)
        return record

    def _update_status(
        self,
        hypothesis_id: str,
        new_status: str,
        note: Optional[str] = None,
    ) -> Optional[HypothesisRecord]:
        """
        Internal helper to move a hypothesis between status folders.
        """
        found = self._find_file_by_id(hypothesis_id)
        if not found:
            return None

        old_path, record = found
        old_status = record.status

        # Update status and timestamp
        record.status = new_status
        record.updated_at = _utc_iso()
        if note:
            record.notes = (record.notes or []) + [note]

        # Remove old file and write new one in the correct folder
        try:
            old_path.unlink()
        except FileNotFoundError:
            pass

        new_path = self._build_file_path(new_status, record.title, record.hypothesis_id)
        self._write_markdown(record, new_path)

        return record

    def validate_hypothesis(
        self,
        hypothesis_id: str,
        note: Optional[str] = None,
    ) -> Optional[HypothesisRecord]:
        """
        Mark a hypothesis as validated and move it into the validated folder.
        """
        return self._update_status(hypothesis_id, "validated", note=note)

    def reject_hypothesis(
        self,
        hypothesis_id: str,
        note: Optional[str] = None,
    ) -> Optional[HypothesisRecord]:
        """
        Mark a hypothesis as rejected and move it into the rejected folder.
        """
        return self._update_status(hypothesis_id, "rejected", note=note)

    def list_hypotheses(self, status: str) -> List[HypothesisRecord]:
        """
        Return all hypotheses with the given status.
        Status must be one of: "pending", "validated", "rejected".
        """
        status_dir = self._status_dir(status)
        records: List[HypothesisRecord] = []
        for path in status_dir.glob("*.md"):
            rec = self._parse_metadata_from_file(path)
            if rec:
                records.append(rec)
        return records

    def summary_strings(self) -> Dict[str, List[str]]:
        """
        Convenience method for snapshot reports.

        Returns a dict with keys:
            "pending", "validated", "rejected"
        Each value is a list of short summary strings.
        """
        result: Dict[str, List[str]] = {}
        for status in ("pending", "validated", "rejected"):
            items = []
            for rec in self.list_hypotheses(status):
                tags = f" [tags: {', '.join(rec.tags)}]" if rec.tags else ""
                cid = f" (cycle {rec.cycle_index})" if rec.cycle_index is not None else ""
                items.append(f"{rec.title}{cid}{tags}")
            result[status] = items
        return result
