"""PDF specialist agent for the Autonomous Research Agent.

This specialist focuses on:
    - Ingesting PDFs (bytes or file paths).
    - Producing structured summaries tuned for:
        * overview
        * mechanism and pathway mapping (longevity and biology)
        * evidence tables and citation extraction
    - Attaching everything cleanly into MemoryStore with goal and hallmark tags.
    - Optionally feeding a ReplayBuffer for high RYE patterns.

It is designed to plug into CoreAgent.specialist_pool under role "pdf"
and be called through CoreAgent.route_to_specialist(...).

The class is safe if PaperTool or FileTool are missing: it will fall back
to basic text extraction and minimal summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional imports: tolerate missing tools gracefully.
try:
    from ..tools_papers import PaperTool  # type: ignore[attr-defined]
except Exception:
    PaperTool = None  # type: ignore[assignment]

try:
    from ..tools_files import FileTool  # type: ignore[attr-defined]
except Exception:
    FileTool = None  # type: ignore[assignment]


@dataclass
class PdfSummaryRequest:
    """Lightweight request model for the PDF specialist."""
    goal: str
    mode: str = "overview"  # "overview", "mechanism_map", "evidence_table"
    hallmark: Optional[str] = None
    domain: str = "general"
    pdf_bytes: Optional[bytes] = None
    file_path: Optional[str] = None
    run_id: Optional[str] = None
    cycle_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class PdfSummarizerAgent:
    """Specialist agent for summarizing scientific PDFs.

    Typical usage inside CoreAgent:

        pdf_agent = PdfSummarizerAgent(memory_store, config, tools, replay_buffer)
        result = pdf_agent.handle_task({
            "type": "pdf_summary",
            "goal": goal,
            "mode": "mechanism_map",
            "domain": "longevity",
            "hallmark": "mitochondria",
            "pdf_bytes": pdf_bytes,
            "run_id": run_id,
            "cycle_index": cycle_index,
        })
    """

    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[Any] = None,
        replay_buffer: Optional[Any] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or {}
        self.tools = tools
        self.replay_buffer = replay_buffer

        # Internal tools
        self.paper_tool = PaperTool() if PaperTool is not None else None
        self.file_tool = FileTool() if FileTool is not None else None

        # Config knobs for learning speed
        self.max_pages_default: int = int(self.config.get("pdf_max_pages", 18))
        self.max_chars_default: int = int(self.config.get("pdf_max_chars", 45_000))
        self.chunk_size_chars: int = int(self.config.get("pdf_chunk_size_chars", 5_000))
        self.max_chunks_summary: int = int(self.config.get("pdf_max_chunks_summary", 12))

        # Modes: overview, mechanism_map, evidence_table
        self.allowed_modes = {"overview", "mechanism_map", "evidence_table"}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def handle_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generic entry point suitable for CoreAgent.route_to_specialist.

        Expected payload fields:
            type: "pdf_summary" (default)
            goal: research goal string
            pdf_bytes or file_path
            mode: optional summary mode
            hallmark, domain, run_id, cycle_index: optional metadata
        """
        task_type = (payload.get("type") or "pdf_summary").lower()
        if task_type != "pdf_summary":
            return {
                "ok": False,
                "error": f"Unsupported task_type for PdfSummarizerAgent: {task_type}",
            }

        request = PdfSummaryRequest(
            goal=str(payload.get("goal") or "Unnamed PDF goal"),
            mode=str(payload.get("mode") or "overview"),
            hallmark=payload.get("hallmark"),
            domain=str(payload.get("domain") or "general"),
            pdf_bytes=payload.get("pdf_bytes"),
            file_path=payload.get("file_path"),
            run_id=payload.get("run_id"),
            cycle_index=payload.get("cycle_index"),
            extra={k: v for k, v in payload.items() if k not in {
                "type",
                "goal",
                "mode",
                "hallmark",
                "domain",
                "pdf_bytes",
                "file_path",
                "run_id",
                "cycle_index",
            }},
        )

        if request.mode not in self.allowed_modes:
            request.mode = "overview"

        if request.pdf_bytes is None and not request.file_path:
            return {
                "ok": False,
                "error": "PdfSummarizerAgent requires pdf_bytes or file_path.",
            }

        text, ingest_stats = self._ingest_pdf(request)
        summary = self._build_structured_summary(request, text, ingest_stats)

        # Attach to MemoryStore and optional ReplayBuffer
        self._attach_to_memory_store(request, summary)
        self._log_to_replay(request, summary)

        return {
            "ok": True,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # PDF ingestion
    # ------------------------------------------------------------------
    def _ingest_pdf(
        self,
        request: PdfSummaryRequest,
    ) -> Tuple[str, Dict[str, Any]]:
        """Load and extract text from PDF via bytes or file path."""
        stats: Dict[str, Any] = {
            "ingest_source": None,
            "pages_used": None,
            "chars_used": None,
            "fallback_used": False,
        }

        text = ""

        if request.pdf_bytes is not None and self.paper_tool is not None:
            try:
                if hasattr(self.paper_tool, "ingest_bytes"):
                    text = self.paper_tool.ingest_bytes(
                        request.pdf_bytes,
                        max_pages=self.max_pages_default,
                    )
                    stats["ingest_source"] = "paper_tool_bytes"
            except Exception:
                text = ""

        if not text and request.file_path and self.paper_tool is not None:
            try:
                if hasattr(self.paper_tool, "ingest_file"):
                    text = self.paper_tool.ingest_file(
                        request.file_path,
                        max_pages=self.max_pages_default,
                    )
                    stats["ingest_source"] = "paper_tool_file"
            except Exception:
                text = ""

        if not text and request.file_path and self.file_tool is not None:
            try:
                # Some FileTool implementations expose generic read or ingest_pdf.
                if hasattr(self.file_tool, "ingest_pdf"):
                    text = self.file_tool.ingest_pdf(request.file_path)
                    stats["ingest_source"] = "file_tool_ingest_pdf"
                elif hasattr(self.file_tool, "read_text"):
                    text = self.file_tool.read_text(request.file_path)
                    stats["ingest_source"] = "file_tool_read_text"
            except Exception:
                text = ""

        if not text and request.pdf_bytes is not None:
            # Last resort: naive decode, only safe if it is actually text.
            try:
                text = request.pdf_bytes.decode("utf-8", errors="ignore")
                stats["ingest_source"] = "bytes_decode_fallback"
                stats["fallback_used"] = True
            except Exception:
                text = ""

        if not text:
            text = "[PdfSummarizerAgent] Failed to extract text from PDF."

        if len(text) > self.max_chars_default:
            text = text[: self.max_chars_default]
            stats["chars_truncated"] = True
        else:
            stats["chars_truncated"] = False

        stats["chars_used"] = len(text)
        return text, stats

    # ------------------------------------------------------------------
    # Summary construction
    # ------------------------------------------------------------------
    def _build_structured_summary(
        self,
        request: PdfSummaryRequest,
        text: str,
        ingest_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a structured summary with multiple views."""
        chunks = self._chunk_text(text, self.chunk_size_chars, self.max_chunks_summary)

        # Use PaperTool.summarise if available for better summaries per chunk.
        chunk_summaries: List[str] = []
        if self.paper_tool is not None and hasattr(self.paper_tool, "summarise"):
            for chunk in chunks:
                try:
                    s = self.paper_tool.summarise(chunk)
                except Exception:
                    s = chunk[:1500]
                chunk_summaries.append(s)
        else:
            # Fallback: simple truncation of each chunk.
            for chunk in chunks:
                chunk_summaries.append(chunk[:1500])

        overview = self._build_overview_section(request, chunk_summaries)
        mechanism_map = self._build_mechanism_section(request, chunk_summaries)
        evidence_table, citations = self._build_evidence_section(request, text, chunk_summaries)

        now = datetime.utcnow().isoformat() + "Z"

        summary: Dict[str, Any] = {
            "goal": request.goal,
            "mode": request.mode,
            "domain": request.domain,
            "hallmark": request.hallmark,
            "run_id": request.run_id,
            "cycle_index": request.cycle_index,
            "created_at": now,
            "ingest_stats": ingest_stats,
            "overview": overview,
            "mechanism_map": mechanism_map,
            "evidence_table": evidence_table,
            "citations": citations,
            "raw": {
                "chunk_summaries": chunk_summaries,
            },
        }

        # For CoreAgent and reporting, expose a concise short_view like structure.
        summary["short_view"] = {
            "primary_mode": request.mode,
            "chars_used": ingest_stats.get("chars_used"),
            "chunk_count": len(chunks),
            "hallmark": request.hallmark,
            "domain": request.domain,
        }

        return summary

    def _build_overview_section(
        self,
        request: PdfSummaryRequest,
        chunk_summaries: List[str],
    ) -> Dict[str, Any]:
        """High level summary for all modes."""
        if not chunk_summaries:
            text = "[PdfSummarizerAgent] No content available for overview."
        else:
            # Take first two chunk summaries as the core overview.
            first = chunk_summaries[0]
            second = chunk_summaries[1] if len(chunk_summaries) > 1 else ""
            text = first
            if second:
                text = first + "\n\nKey follow up:\n" + second

        return {
            "title": "PDF overview",
            "summary_text": text,
        }

    def _build_mechanism_section(
        self,
        request: PdfSummaryRequest,
        chunk_summaries: List[str],
    ) -> Dict[str, Any]:
        """Mechanism oriented section for longevity or biology domains."""
        if request.domain.lower() not in {"longevity", "biology", "medicine", "health"}:
            return {
                "title": "Mechanism map (not domain focused)",
                "notes": "Mechanism extraction is mainly tuned for longevity and biology domains.",
                "mechanism_bullets": [],
            }

        key_terms = [
            "mechanism",
            "pathway",
            "mTOR",
            "autophagy",
            "senescence",
            "NAD+",
            "inflammation",
            "DNA damage",
            "mitochondria",
            "oxidative stress",
        ]

        bullets: List[str] = []
        for s in chunk_summaries:
            lower = s.lower()
            if any(term.lower() in lower for term in key_terms):
                snippet = s.strip()
                if len(snippet) > 600:
                    snippet = snippet[:600]
                bullets.append(snippet)
            if len(bullets) >= 15:
                break

        return {
            "title": "Mechanism and pathway focused notes",
            "notes": (
                "This section collects chunk summaries that mention classic mechanisms or pathways "
                "relevant to longevity and biology."
            ),
            "mechanism_bullets": bullets,
        }

    def _build_evidence_section(
        self,
        request: PdfSummaryRequest,
        full_text: str,
        chunk_summaries: List[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Evidence oriented section and citation extraction."""
        # If PaperTool exposes a dedicated citation extractor, use it.
        citations: List[Dict[str, Any]] = []

        if self.paper_tool is not None:
            try:
                if hasattr(self.paper_tool, "extract_citations"):
                    raw = self.paper_tool.extract_citations(full_text)
                    for c in raw or []:
                        if not isinstance(c, dict):
                            c = {"title": str(c)}
                        # Tag with minimal provenance. URL may not be available.
                        c = dict(c)
                        c.setdefault("source", "pdf")
                        c.setdefault("channel", "pdf")
                        c.setdefault("goal", request.goal)
                        c.setdefault("domain", request.domain)
                        c.setdefault("hallmark", request.hallmark)
                        citations.append(c)
            except Exception:
                citations = []

        # If no dedicated citation extractor, build a thin evidence table from summaries.
        if not citations:
            for idx, s in enumerate(chunk_summaries):
                trimmed = s.strip()
                if not trimmed:
                    continue
                label = trimmed.split("\n", 1)[0]
                if len(label) > 180:
                    label = label[:180]
                citations.append(
                    {
                        "title": label,
                        "source": "pdf_summary",
                        "channel": "pdf",
                        "goal": request.goal,
                        "domain": request.domain,
                        "hallmark": request.hallmark,
                        "chunk_index": idx,
                    }
                )
                if len(citations) >= 30:
                    break

        rows: List[Dict[str, Any]] = []
        for idx, c in enumerate(citations):
            rows.append(
                {
                    "index": idx,
                    "label": c.get("title"),
                    "channel": c.get("channel", "pdf"),
                    "source": c.get("source", "pdf"),
                    "goal": c.get("goal", request.goal),
                    "hallmark": c.get("hallmark", request.hallmark),
                    "notes": c.get("snippet") or c.get("abstract") or "",
                }
            )

        evidence_table = {
            "title": "Evidence and citation oriented view",
            "rows": rows,
        }
        return evidence_table, citations

    # ------------------------------------------------------------------
    # Memory and replay attachment
    # ------------------------------------------------------------------
    def _attach_to_memory_store(self, request: PdfSummaryRequest, summary: Dict[str, Any]) -> None:
        """Store summary notes and citations into MemoryStore."""
        goal = request.goal
        role = "pdf_specialist"

        overview = summary.get("overview", {})
        mechanism_map = summary.get("mechanism_map", {})
        evidence_table = summary.get("evidence_table", {})
        citations = summary.get("citations", []) or []

        overview_text = overview.get("summary_text") or ""
        mech_bullets = "\n".join(mechanism_map.get("mechanism_bullets", []) or [])
        evidence_rows = evidence_table.get("rows", []) or []

        lines: List[str] = []
        lines.append(f"[{role}] PDF overview for goal:")
        lines.append(goal)
        lines.append("")
        if overview_text:
            lines.append("High level summary:")
            lines.append(overview_text)
            lines.append("")
        if mech_bullets:
            lines.append("Mechanism oriented bullets:")
            lines.append(mech_bullets)
            lines.append("")
        if evidence_rows:
            lines.append("Evidence oriented entries:")
            for row in evidence_rows[:20]:
                label = row.get("label", "")
                notes = row.get("notes", "")
                if notes:
                    snippet = notes[:220]
                    lines.append(f"- {label} :: {snippet}")
                else:
                    lines.append(f"- {label}")
            lines.append("")

        note_text = "\n".join(lines)
        try:
            self.memory_store.add_note(
                goal,
                note_text,
                role=role,
                metadata={
                    "domain": request.domain,
                    "hallmark": request.hallmark,
                    "run_id": request.run_id,
                    "cycle_index": request.cycle_index,
                    "source": "pdf_specialist",
                },
            )
        except Exception:
            # Fall back to minimal signature without metadata
            try:
                self.memory_store.add_note(goal, note_text, role=role)
            except Exception:
                pass

        for c in citations:
            try:
                self.memory_store.add_citation(goal, c)
            except Exception:
                continue

    def _log_to_replay(self, request: PdfSummaryRequest, summary: Dict[str, Any]) -> None:
        """Send high value patterns to ReplayBuffer if available."""
        if self.replay_buffer is None:
            return

        try:
            run_id = summary.get("run_id") or request.run_id
            payload = {
                "item_type": "pdf_summary",
                "goal": request.goal,
                "domain": request.domain,
                "hallmark": request.hallmark,
                "run_id": run_id,
                "cycle_index": request.cycle_index,
                "created_at": summary.get("created_at"),
                "overview": summary.get("overview"),
                "mechanism_map": summary.get("mechanism_map"),
                "evidence_table": summary.get("evidence_table"),
                "citations": summary.get("citations"),
                "short_view": summary.get("short_view"),
            }

            rb = self.replay_buffer
            if hasattr(rb, "add_item"):
                rb.add_item(payload)
            elif hasattr(rb, "log_item"):
                rb.log_item(payload)
            elif hasattr(rb, "add_from_pdf"):
                rb.add_from_pdf(payload)
        except Exception:
            return

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        max_chunks: int,
    ) -> List[str]:
        """Simple character based chunking with a soft cap."""
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length and len(chunks) < max_chunks:
            end = min(start + chunk_size, length)
            chunk = text[start:end]
            chunks.append(chunk)
            start = end

        return chunks
