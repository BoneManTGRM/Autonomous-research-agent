"""Implementation of the TGRM loop for the research agent.

This module defines a `TGRMLoop` class that encapsulates the core logic
of the agent's Targeted Gradient Repair Mechanism (TGRM). The loop iterates
through phases of testing, detecting issues, repairing them, and verifying
improvements. It interacts with the memory store and research tools.

Reparodynamics view
-------------------
The agent is treated as a reparodynamic system:
    - Each cycle tries to reduce defects (gaps, TODOs, unanswered questions)
      while spending as little energy as possible.
    - ΔR measures improvement; E measures cost; RYE = ΔR / E is the core
      efficiency metric.

TGRM phases (implemented below)
-------------------------------
    Test   : evaluate current notes / state
    Detect : find gaps, TODOs, unanswered questions
    Repair : perform targeted web / PubMed / Semantic Scholar actions
    Verify : re-test, compute ΔR and RYE, and log a cycle entry
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

from .rye_metrics import compute_delta_r, compute_energy, compute_rye
from .tools_web import WebResearchTool
from .tools_papers import PaperTool
from .tools_files import FileTool
from .tools_pubmed import PubMedTool
from .tools_semantic_scholar import SemanticScholarTool
from .hypothesis_engine import generate_hypotheses


class TGRMLoop:
    """Encapsulate the TGRM loop logic for one research cycle."""

    def __init__(self, memory_store: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # Core tools
        self.web_tool = WebResearchTool()
        self.paper_tool = PaperTool()
        self.file_tool = FileTool()

        # Scientific tools
        self.pubmed_tool = PubMedTool()
        self.semantic_tool = SemanticScholarTool()

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------
    def run_cycle(
        self,
        goal: str,
        cycle_index: int,
        role: str = "agent",
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Run one TGRM cycle for a given research goal.

        Args:
            goal:
                Human-readable research goal for this cycle.
            cycle_index:
                Global cycle index (used for logging and history).
            role:
                Logical role (e.g., "agent", "researcher", "critic").
            source_controls:
                Optional switches like:
                    {
                      "web": True,
                      "pubmed": True,
                      "semantic": True,
                      "pdf": True,
                      "biomarkers": False,
                    }
                If None, defaults are used (web on, others off).
            pdf_bytes:
                Optional PDF content (uploaded file) that can be ingested
                as part of the Repair phase if "pdf" is enabled.

        Returns:
            Dict with:
                {
                  "summary": human-facing summary,
                  "log":     full cycle log
                }
        """
        src_ctrl = self._normalise_source_controls(source_controls)

        # TEST phase – evaluate current state
        prior_notes = self.memory_store.get_notes(goal)
        status_report = self._test(goal, prior_notes)

        # DETECT phase – identify issues to repair
        issues, issue_descriptions = self._detect(status_report)

        # REPAIR phase – apply targeted repairs
        (
            repair_actions,
            notes_added,
            citations,
            stats,
        ) = self._repair(
            goal=goal,
            issues=issues,
            descriptions=issue_descriptions,
            role=role,
            source_controls=src_ctrl,
            pdf_bytes=pdf_bytes,
        )

        # VERIFY phase – re-evaluate state after repair
        new_notes = self.memory_store.get_notes(goal)
        new_status_report = self._test(goal, new_notes)
        issues_after, _ = self._detect(new_status_report)

        # Hypothesis generation (optional improvement targets)
        hypotheses = generate_hypotheses(goal, new_notes, citations, max_hypotheses=5)
        for h in hypotheses:
            self.memory_store.add_hypothesis(goal, h["text"], score=h.get("confidence"))

        # Compute metrics (Reparodynamics: ΔR / E)
        delta_r = compute_delta_r(
            issues_before=len(issues),
            issues_after=len(issues_after),
            repairs_applied=len(repair_actions),
            contradictions_resolved=stats.get("contradictions_resolved", 0),
            hypotheses_generated=len(hypotheses),
            sources_used=stats.get("sources_used", 0),
        )
        energy_e = compute_energy(
            actions_taken=repair_actions,
            web_calls=stats.get("web_calls", 0),
            pubmed_calls=stats.get("pubmed_calls", 0),
            semantic_calls=stats.get("semantic_calls", 0),
            pdf_ingestions=stats.get("pdf_ingestions", 0),
        )
        rye_value = compute_rye(delta_r, energy_e)

        # Create cycle summary (machine-facing log)
        cycle_summary = {
            "cycle": cycle_index,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "role": role,
            "issues_detected": issue_descriptions,
            "issues_after": issues_after,
            "repairs_applied": repair_actions,
            "citations": citations,
            "hypotheses": hypotheses,
            "stats": stats,
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
        }

        # Log cycle into memory
        self.memory_store.log_cycle(cycle_summary)

        # Human-readable summary
        human_summary = {
            "cycle": cycle_index,
            "role": role,
            "issues_before": issue_descriptions,
            "repairs": [a["description"] for a in repair_actions],
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
            "notes_added": notes_added,
            "citations": citations,
            "hypotheses": hypotheses,
        }
        return {"summary": human_summary, "log": cycle_summary}

    # ------------------------------------------------------------------
    # TGRM phases
    # ------------------------------------------------------------------
    def _test(self, goal: str, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the current state of research for this goal."""
        return {
            "goal": goal,
            "known_notes_count": len(notes),
            "notes": notes,
        }

    def _detect(self, status_report: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify issues or gaps to be repaired."""
        issues: List[str] = []
        descriptions: List[str] = []
        notes = status_report.get("notes", [])

        if not notes:
            issues.append("no_notes")
            descriptions.append("No prior notes found; initial research required.")
        else:
            for note in notes:
                content: str = note.get("content", "")
                if "?" in content:
                    issues.append("question_mark")
                    descriptions.append("Unanswered question detected in notes.")
                if "TODO" in content or "todo" in content:
                    issues.append("todo_item")
                    descriptions.append("TODO item detected in notes; missing information.")
                # Simple placeholder for contradiction marker
                if "CONTRADICTION" in content:
                    issues.append("contradiction")
                    descriptions.append("Marked contradiction in notes; needs resolution.")

        return issues, descriptions

    def _repair(
        self,
        goal: str,
        issues: List[str],
        descriptions: List[str],
        role: str,
        source_controls: Dict[str, bool],
        pdf_bytes: Optional[bytes],
    ):
        """Apply repairs to address detected issues.

        This is where:
        - Tavily web search runs
        - PubMed and Semantic Scholar are queried
        - PDF ingestion can be used when available
        """
        repair_actions: List[Dict[str, str]] = []
        notes_added: List[str] = []
        citations: List[Dict[str, str]] = []

        # Stats for RYE energy + improvement
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
        }

        # Helper to register citations into MemoryStore
        def _log_citations(cites: List[Dict[str, str]]) -> None:
            for c in cites:
                self.memory_store.add_citation(goal, c)

        for issue, desc in zip(issues, descriptions):
            # Main "no notes" entry point
            if issue == "no_notes":
                note_text, new_cites, issue_stats = self._initial_research(goal, role, source_controls, pdf_bytes)
                self.memory_store.add_note(goal, note_text, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Performed initial multi-source research for '{goal}'",
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)

                # accumulate stats
                for k, v in issue_stats.items():
                    stats[k] = stats.get(k, 0) + v

            elif issue in {"question_mark", "todo_item"}:
                # Future: targeted queries built from the specific note/question.
                followup = (
                    f"[{role}] Detected issue '{issue}': {desc}. "
                    "Further targeted research is required (not yet specialised)."
                )
                self.memory_store.add_note(goal, followup, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Logged follow-up requirement for targeted research.",
                    }
                )
                notes_added.append(followup)

            elif issue == "contradiction":
                # Placeholder: we mark that a contradiction was observed.
                # Future work: we could run focused searches to resolve it.
                note = (
                    f"[{role}] Reminder: contradiction flagged in notes. "
                    "Future cycles should prioritise resolving this with additional sources."
                )
                self.memory_store.add_note(goal, note, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Acknowledged contradiction; queued for deeper resolution.",
                    }
                )
                notes_added.append(note)
                stats["contradictions_resolved"] += 0  # not resolved yet

        # Count unique sources used for this cycle
        unique_sources = set()
        for c in citations:
            key = (c.get("source"), c.get("url"))
            unique_sources.add(key)
        stats["sources_used"] = len(unique_sources)

        return repair_actions, notes_added, citations, stats

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _normalise_source_controls(self, source_controls: Optional[Dict[str, bool]]) -> Dict[str, bool]:
        """Provide default source controls if none are given."""
        defaults = {
            "web": True,
            "pubmed": False,
            "semantic": False,
            "pdf": False,
            "biomarkers": False,
        }
        if not source_controls:
            return defaults
        merged = defaults.copy()
        merged.update({k: bool(v) for k, v in source_controls.items()})
        return merged

    def _initial_research(
        self,
        goal: str,
        role: str,
        source_controls: Dict[str, bool],
        pdf_bytes: Optional[bytes],
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, int]]:
        """Perform the initial multi-source research when there are no notes."""
        citations: List[Dict[str, str]] = []
        stats = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Initial research summary for goal:")
        note_lines.append(goal)
        note_lines.append("")

        # Web search
        if source_controls.get("web", True):
            web_results = self.web_tool.search(goal)
            stats["web_calls"] += 1
            web_summary = self.web_tool.summarize_results(web_results)
            web_cites = self.web_tool.to_citations(web_results)
            citations.extend(web_cites)

            note_lines.append("Web summary (Tavily):")
            note_lines.append(web_summary)
            note_lines.append("")
            note_lines.append("Web sources:")
            for c in web_cites:
                note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
            note_lines.append("")

        # PubMed search
        if source_controls.get("pubmed", False):
            pubmed_results = self.pubmed_tool.search(goal, max_results=5)
            stats["pubmed_calls"] += 1
            citations.extend(pubmed_results)
            note_lines.append("PubMed sources:")
            for r in pubmed_results:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        # Semantic Scholar search
        if source_controls.get("semantic", False):
            sem_results = self.semantic_tool.search(goal, max_results=5)
            stats["semantic_calls"] += 1
            citations.extend(sem_results)
            note_lines.append("Semantic Scholar sources:")
            for r in sem_results:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        # Optional PDF ingestion if provided
        if source_controls.get("pdf", False) and pdf_bytes:
            try:
                # We only ingest if PaperTool exposes an appropriate method.
                if hasattr(self.paper_tool, "ingest_bytes"):
                    text = self.paper_tool.ingest_bytes(pdf_bytes)  # type: ignore[attr-defined]
                    stats["pdf_ingestions"] += 1
                    summary = self.paper_tool.summarise(text)
                    note_lines.append("Attached PDF summary:")
                    note_lines.append(summary)
                    note_lines.append("")
                # If ingest_bytes is not implemented, we simply skip.
            except Exception:
                pass

        # Join all lines into a single note text
        note_text = "\n".join(note_lines)
        return note_text, citations, stats
