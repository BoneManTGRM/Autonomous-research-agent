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

This manager is wired for Reparodynamics style learning:

    - RYE aware hypothesis records (before/after, delta R, energy)
    - confidence, novelty, and evidence-strength fields
    - optional linkage to cycles and citations
    - integration hooks to:
        * discovery_log.DiscoveryLogger
        * MemoryStore.add_hypothesis / add_discovery (if provided)
    - auto evaluation helpers for promoting or rejecting pending ideas

Typical usage:

    from agent.hypothesis_manager import HypothesisManager

    hm = HypothesisManager(run_id="run_001", domain="longevity", goal="extend_healthspan")

    h = hm.create_hypothesis(
        title="New candidate longevity pathway",
        description="Reasoning and evidence here...",
        cycle_index=128,
        agent_role="Researcher",
        rye_before=0.41,
        rye_after=0.53,
        tags=["longevity", "pathway_x"],
        kind="mechanism",
        confidence=0.65,
        novelty=0.4,
        evidence_strength=0.55,
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

# Root folders for hypothesis storage
ROOT_DIR = Path("hypotheses")
PENDING_DIR = ROOT_DIR / "pending"
VALIDATED_DIR = ROOT_DIR / "validated"
REJECTED_DIR = ROOT_DIR / "rejected"

for d in (PENDING_DIR, VALIDATED_DIR, REJECTED_DIR):
    d.mkdir(parents=True, exist_ok=True)

HYPOTHESIS_SCHEMA_VERSION: int = 1


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

    This is the canonical schema written into the JSON header of each
    Markdown file. Extra fields can be safely ignored by older tools.
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

    # Reparodynamics learning hints
    domain: Optional[str] = None
    goal: Optional[str] = None
    kind: Optional[str] = None  # "mechanism", "treatment", "structure", etc.

    confidence: Optional[float] = None      # 0-1 subjective confidence
    novelty: Optional[float] = None         # 0-1 novelty vs known work
    evidence_strength: Optional[float] = None  # 0-1 quality of evidence
    rye_gain_estimate: Optional[float] = None  # estimated delta R impact if true

    linked_citations: Optional[List[Dict[str, Any]]] = None
    linked_cycles: Optional[List[int]] = None

    is_promising: Optional[bool] = None

    schema_version: int = HYPOTHESIS_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HypothesisManager:
    """
    Manages the lifecycle of hypotheses:
        - creation (pending)
        - validation
        - rejection
        - listing and searching
        - basic statistics and auto evaluation for learning loops

    Optional integration points:

        - discovery_log.DiscoveryLogger via lazy import
        - MemoryStore compatible object passed in at construction

    The manager itself does not call tools or external APIs. It only
    reads and writes local files and optionally notifies attached
    in-process components.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        *,
        domain: Optional[str] = None,
        goal: Optional[str] = None,
        auto_log_discoveries: bool = True,
        memory_store: Optional[Any] = None,
    ) -> None:
        self.run_id = run_id
        self.domain = domain
        self.goal = goal
        self.auto_log_discoveries = auto_log_discoveries
        self.memory_store = memory_store

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

        # Backward compatible construction
        try:
            if "schema_version" not in data:
                data["schema_version"] = HYPOTHESIS_SCHEMA_VERSION
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

    def _log_to_discovery_log(
        self,
        event_kind: str,
        record: HypothesisRecord,
        *,
        note: Optional[str] = None,
    ) -> None:
        """
        Best effort integration with discovery_log.DiscoveryLogger.

        This is safe to call even if discovery_log is not available.
        """
        if not self.auto_log_discoveries:
            return
        try:
            # Lazy import to avoid hard dependency or circular imports
            from .discovery_log import get_global_logger  # type: ignore[import]
        except Exception:
            return

        logger = get_global_logger(run_id=self.run_id)
        description_parts: List[str] = []
        description_parts.append(record.description or "")
        if note:
            description_parts.append("")
            description_parts.append(f"Status note: {note}")
        if record.notes:
            description_parts.append("")
            description_parts.append("Additional notes:")
            for n in record.notes:
                description_parts.append(f"- {n}")

        description = "\n".join(part for part in description_parts if part is not None)

        tags: List[str] = list(record.tags or [])
        tags.append(f"status_{record.status}")
        if record.kind:
            tags.append(record.kind)
        if record.domain:
            tags.append(f"domain_{record.domain}")
        if record.goal:
            tags.append(f"goal_{record.goal}")

        logger.log_event(
            kind=event_kind,
            title=record.title,
            description=description,
            cycle_index=record.cycle_index,
            agent_role=record.agent_role,
            rye_before=record.rye_before,
            rye_after=record.rye_after,
            delta_r=record.delta_r,
            energy=record.energy,
            tags=tags,
            extra={
                "hypothesis_id": record.hypothesis_id,
                "status": record.status,
                "domain": record.domain,
                "goal": record.goal,
                "kind": record.kind,
                "confidence": record.confidence,
                "novelty": record.novelty,
                "evidence_strength": record.evidence_strength,
                "rye_gain_estimate": record.rye_gain_estimate,
            },
        )

    def _push_to_memory_store_on_create(self, record: HypothesisRecord) -> None:
        """
        Optional hook: if a MemoryStore-like object is provided, record the
        hypothesis as semantic memory.
        """
        if self.memory_store is None:
            return
        try:
            if hasattr(self.memory_store, "add_hypothesis"):
                goal = record.goal or (self.goal or "global")
                text = record.description or record.title
                score = record.confidence
                tags = record.tags or []
                self.memory_store.add_hypothesis(
                    goal=goal,
                    text=text,
                    score=score,
                    tags=tags,
                )
        except Exception:
            pass

    def _push_to_memory_store_on_validate(self, record: HypothesisRecord) -> None:
        """
        Optional hook: when a hypothesis is validated, also record it as a
        discovery in the MemoryStore if available.
        """
        if self.memory_store is None:
            return
        try:
            if hasattr(self.memory_store, "add_discovery"):
                goal = record.goal or (self.goal or "global")
                label = record.title
                evidence_summary = record.description
                score = record.confidence
                tags = record.tags or []
                kind = record.kind or "hypothesis"
                domain = record.domain or self.domain
                self.memory_store.add_discovery(
                    goal=goal,
                    kind=kind,
                    label=label,
                    evidence_summary=evidence_summary,
                    score=score,
                    tags=tags,
                    citations=record.linked_citations or [],
                    domain=domain,
                )
        except Exception:
            pass

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
        *,
        domain: Optional[str] = None,
        goal: Optional[str] = None,
        kind: Optional[str] = None,
        confidence: Optional[float] = None,
        novelty: Optional[float] = None,
        evidence_strength: Optional[float] = None,
        rye_gain_estimate: Optional[float] = None,
        linked_citations: Optional[List[Dict[str, Any]]] = None,
        linked_cycles: Optional[List[int]] = None,
        is_promising: Optional[bool] = None,
    ) -> HypothesisRecord:
        """
        Create a new pending hypothesis and write it to the pending folder.

        Additional Reparodynamics hints can be provided but are optional.
        """
        now = _utc_iso()
        hypothesis_id = str(uuid.uuid4())

        # If RYE gain estimate not provided, try to infer from before/after
        if rye_gain_estimate is None and rye_before is not None and rye_after is not None:
            try:
                rye_gain_estimate = float(rye_after) - float(rye_before)
            except Exception:
                rye_gain_estimate = None

        # Default domain and goal fallback to manager defaults
        domain_val = domain if domain is not None else self.domain
        goal_val = goal if goal is not None else self.goal

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
            domain=domain_val,
            goal=goal_val,
            kind=kind,
            confidence=confidence,
            novelty=novelty,
            evidence_strength=evidence_strength,
            rye_gain_estimate=rye_gain_estimate,
            linked_citations=linked_citations or [],
            linked_cycles=linked_cycles or ([] if cycle_index is None else [cycle_index]),
            is_promising=is_promising,
        )

        path = self._build_file_path("pending", title, hypothesis_id)
        self._write_markdown(record, path)

        # Learning hooks
        self._push_to_memory_store_on_create(record)
        self._log_to_discovery_log("hypothesis_created", record)

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

        # Learning and logging hooks
        if new_status == "validated":
            self._push_to_memory_store_on_validate(record)
            self._log_to_discovery_log("hypothesis_validated", record, note=note)
        elif new_status == "rejected":
            self._log_to_discovery_log("hypothesis_rejected", record, note=note)
        else:
            self._log_to_discovery_log("hypothesis_status_change", record, note=note)

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

    def append_note(
        self,
        hypothesis_id: str,
        note: str,
    ) -> Optional[HypothesisRecord]:
        """
        Append a note to an existing hypothesis without changing status.
        """
        found = self._find_file_by_id(hypothesis_id)
        if not found:
            return None

        path, record = found
        record.updated_at = _utc_iso()
        record.notes = (record.notes or []) + [note]

        # Rewrite file in place
        self._write_markdown(record, path)

        self._log_to_discovery_log("hypothesis_note", record, note=note)
        return record

    def update_metadata(
        self,
        hypothesis_id: str,
        *,
        confidence: Optional[float] = None,
        novelty: Optional[float] = None,
        evidence_strength: Optional[float] = None,
        rye_gain_estimate: Optional[float] = None,
        tags: Optional[List[str]] = None,
        kind: Optional[str] = None,
        domain: Optional[str] = None,
        goal: Optional[str] = None,
        is_promising: Optional[bool] = None,
    ) -> Optional[HypothesisRecord]:
        """
        Update metadata fields on a hypothesis without changing its status.
        """
        found = self._find_file_by_id(hypothesis_id)
        if not found:
            return None

        path, record = found

        if confidence is not None:
            record.confidence = confidence
        if novelty is not None:
            record.novelty = novelty
        if evidence_strength is not None:
            record.evidence_strength = evidence_strength
        if rye_gain_estimate is not None:
            record.rye_gain_estimate = rye_gain_estimate
        if tags is not None:
            record.tags = tags
        if kind is not None:
            record.kind = kind
        if domain is not None:
            record.domain = domain
        if goal is not None:
            record.goal = goal
        if is_promising is not None:
            record.is_promising = is_promising

        record.updated_at = _utc_iso()

        self._write_markdown(record, path)
        self._log_to_discovery_log("hypothesis_metadata_updated", record)

        return record

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

    def list_all(self) -> Dict[str, List[HypothesisRecord]]:
        """
        Return all hypotheses grouped by status.
        Keys: "pending", "validated", "rejected".
        """
        return {
            "pending": self.list_hypotheses("pending"),
            "validated": self.list_hypotheses("validated"),
            "rejected": self.list_hypotheses("rejected"),
        }

    def search(
        self,
        query: str,
        *,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        domain: Optional[str] = None,
        goal: Optional[str] = None,
        kind: Optional[str] = None,
        max_results: int = 100,
    ) -> List[HypothesisRecord]:
        """
        Simple keyword search across hypotheses.

        Filters:
            - status: limit to a single status if provided
            - tag: must contain this tag
            - domain: match record.domain
            - goal: match record.goal
            - kind: match record.kind
        """
        query_lower = query.lower().strip()
        statuses = [status] if status in ("pending", "validated", "rejected") else (
            "pending",
            "validated",
            "rejected",
        )

        results: List[HypothesisRecord] = []
        for st in statuses:
            records = self.list_hypotheses(st)
            for rec in records:
                if domain is not None and rec.domain != domain:
                    continue
                if goal is not None and rec.goal != goal:
                    continue
                if kind is not None and rec.kind != kind:
                    continue
                if tag is not None:
                    if not rec.tags or tag not in rec.tags:
                        continue

                haystack_parts = [
                    rec.title or "",
                    rec.description or "",
                    " ".join(rec.tags or []),
                ]
                haystack = " ".join(haystack_parts).lower()
                if query_lower in haystack:
                    results.append(rec)
                    if len(results) >= max_results:
                        return results
        return results

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
                conf = (
                    f" [conf: {rec.confidence:.2f}]"
                    if isinstance(rec.confidence, (int, float))
                    else ""
                )
                items.append(f"{rec.title}{cid}{conf}{tags}")
            result[status] = items
        return result

    def stats(self) -> Dict[str, Any]:
        """
        Basic statistics for dashboards and long runs.

        Includes counts by status and simple RYE statistics for validated
        hypotheses.
        """
        pending = self.list_hypotheses("pending")
        validated = self.list_hypotheses("validated")
        rejected = self.list_hypotheses("rejected")

        def _rye_stats(records: List[HypothesisRecord]) -> Dict[str, Optional[float]]:
            gains: List[float] = []
            for r in records:
                if isinstance(r.rye_before, (int, float)) and isinstance(
                    r.rye_after, (int, float)
                ):
                    gains.append(float(r.rye_after) - float(r.rye_before))
                elif isinstance(r.rye_gain_estimate, (int, float)):
                    gains.append(float(r.rye_gain_estimate))
            if not gains:
                return {"avg_gain": None, "min_gain": None, "max_gain": None}
            avg_gain = sum(gains) / len(gains)
            return {"avg_gain": avg_gain, "min_gain": min(gains), "max_gain": max(gains)}

        return {
            "counts": {
                "pending": len(pending),
                "validated": len(validated),
                "rejected": len(rejected),
                "total": len(pending) + len(validated) + len(rejected),
            },
            "validated_rye": _rye_stats(validated),
        }

    def auto_evaluate_pending(
        self,
        *,
        min_rye_gain: float = 0.05,
        min_confidence: float = 0.5,
        require_positive_after: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Simple learning helper to automatically validate or reject pending
        hypotheses based on RYE and confidence thresholds.

        Decision rule:

            - if estimated RYE gain >= min_rye_gain and confidence >= min_confidence
              and (if require_positive_after then rye_after > 0), then validate

            - otherwise, reject

        Returns:
            {
                "validated": [hypothesis_id, ...],
                "rejected": [hypothesis_id, ...],
                "skipped":  [hypothesis_id, ...],   # no RYE or confidence info
            }
        """
        pending = self.list_hypotheses("pending")
        validated_ids: List[str] = []
        rejected_ids: List[str] = []
        skipped_ids: List[str] = []

        for rec in pending:
            # Determine RYE gain
            gain: Optional[float] = None
            if isinstance(rec.rye_before, (int, float)) and isinstance(
                rec.rye_after, (int, float)
            ):
                gain = float(rec.rye_after) - float(rec.rye_before)
            elif isinstance(rec.rye_gain_estimate, (int, float)):
                gain = float(rec.rye_gain_estimate)

            if gain is None or rec.confidence is None:
                skipped_ids.append(rec.hypothesis_id)
                continue

            if require_positive_after and isinstance(rec.rye_after, (int, float)):
                if rec.rye_after <= 0:
                    # Strong negative performance, reject
                    updated = self.reject_hypothesis(
                        rec.hypothesis_id,
                        note="Auto rejected: RYE after non positive.",
                    )
                    if updated:
                        rejected_ids.append(rec.hypothesis_id)
                    else:
                        skipped_ids.append(rec.hypothesis_id)
                    continue

            if gain >= min_rye_gain and rec.confidence >= min_confidence:
                updated = self.validate_hypothesis(
                    rec.hypothesis_id,
                    note=(
                        f"Auto validated: gain={gain:.4f} >= {min_rye_gain:.4f}, "
                        f"confidence={rec.confidence:.2f} >= {min_confidence:.2f}."
                    ),
                )
                if updated:
                    validated_ids.append(rec.hypothesis_id)
                else:
                    skipped_ids.append(rec.hypothesis_id)
            else:
                updated = self.reject_hypothesis(
                    rec.hypothesis_id,
                    note=(
                        f"Auto rejected: gain={gain:.4f}, confidence="
                        f"{(rec.confidence or 0.0):.2f} below thresholds."
                    ),
                )
                if updated:
                    rejected_ids.append(rec.hypothesis_id)
                else:
                    skipped_ids.append(rec.hypothesis_id)

        return {
            "validated": validated_ids,
            "rejected": rejected_ids,
            "skipped": skipped_ids,
        }

    def export_index(self, path: Optional[Path | str] = None) -> Path:
        """
        Export a JSON index of all hypotheses and return the path.

        Useful for building external dashboards or analysis notebooks.
        """
        if path is None:
            path = ROOT_DIR / "hypotheses_index.json"
        out_path = Path(path)

        all_records: List[Dict[str, Any]] = []
        for status in ("pending", "validated", "rejected"):
            for rec in self.list_hypotheses(status):
                data = rec.to_dict()
                data["status"] = status
                all_records.append(data)

        out_path.write_text(
            json.dumps(all_records, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return out_path
