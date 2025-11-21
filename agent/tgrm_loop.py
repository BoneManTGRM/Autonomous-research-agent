"""Implementation of the TGRM loop for the research agent.

This module defines a `TGRMLoop` class that encapsulates the core logic
of the agent's targeted gradient repair mechanism (TGRM). The loop iterates
through phases of testing, detecting issues, repairing them, and verifying
improvements. It interacts with the memory store and research tools.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from .rye_metrics import compute_delta_r, compute_energy, compute_rye
from .tools_web import WebResearchTool
from .tools_papers import PaperTool
from .tools_files import FileTool


class TGRMLoop:
    """Encapsulate the TGRM loop logic for one research cycle."""

    def __init__(self, memory_store: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # Tools: web, papers, local files
        self.web_tool = WebResearchTool()
        self.paper_tool = PaperTool()
        self.file_tool = FileTool()

    def run_cycle(self, goal: str, cycle_index: int, role: str = "researcher") -> Dict[str, Any]:
        """Run one TGRM cycle for a given research goal and agent role.

        role can be "researcher", "critic", or any tag you want to use.
        """
        # Test phase: evaluate current state
        prior_notes = self.memory_store.get_notes(goal)
        status_report = self._test(goal, prior_notes)

        # Detect phase: identify issues to repair
        issues, issue_descriptions = self._detect(status_report)

        # Repair phase: apply targeted repairs
        repair_actions, notes_added, citations = self._repair(goal, issues, issue_descriptions, role)

        # Verify phase: re evaluate state after repair
        new_notes = self.memory_store.get_notes(goal)
        new_status_report = self._test(goal, new_notes)
        issues_after, _ = self._detect(new_status_report)

        # Compute metrics
        delta_r = compute_delta_r(
            issues_before=len(issues),
            issues_after=len(issues_after),
            repairs_applied=len(repair_actions),
        )
        energy_e = compute_energy(repair_actions)
        rye_value = compute_rye(delta_r, energy_e)

        # Create cycle summary
        cycle_summary = {
            "cycle": cycle_index,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "role": role,
            "issues_detected": issue_descriptions,
            "issues_after": issues_after,
            "repairs_applied": repair_actions,
            "citations": citations,
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
        }

        # Log cycle
        self.memory_store.log_cycle(cycle_summary)

        # Human readable summary
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
        }
        return {"summary": human_summary, "log": cycle_summary}

    # ===== TGRM phases =====

    def _test(self, goal: str, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the current state of research for this goal."""
        return {
            "known_notes_count": len(notes),
            "notes": notes,
        }

    def _detect(self, status_report: Dict[str, Any]):
        """Identify issues or gaps to be repaired."""
        issues: List[str] = []
        descriptions: List[str] = []
        notes = status_report.get("notes", [])
        if not notes:
            issues.append("no_notes")
            descriptions.append("No prior notes found; research required.")
        else:
            for note in notes:
                content: str = note.get("content", "")
                if "?" in content:
                    issues.append("question_mark")
                    descriptions.append("Unanswered question detected in notes.")
                if "TODO" in content or "todo" in content:
                    issues.append("todo_item")
                    descriptions.append("TODO item detected in notes; missing information.")
        return issues, descriptions

    def _repair(
        self,
        goal: str,
        issues: List[str],
        descriptions: List[str],
        role: str,
    ):
        """Apply repairs to address detected issues.

        This is where:
        - real web search runs
        - Semantic Scholar is queried
        - PDF ingestion can be added later if you store URLs in notes
        """
        repair_actions: List[Dict[str, str]] = []
        notes_added: List[str] = []
        citations: List[Dict[str, str]] = []

        for issue, desc in zip(issues, descriptions):
            # Main research entry point
            if issue == "no_notes":
                # 1. Tavily web search
                web_results = self.web_tool.search(goal)
                web_summary = self.web_tool.summarize_results(web_results)

                # 2. Semantic Scholar search for more academic sources
                scholar_results = self.paper_tool.search_semantic_scholar(goal, limit=5)

                # Build citation list
                for r in web_results:
                    citations.append(
                        {
                            "source": "web",
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                        }
                    )
                for r in scholar_results:
                    citations.append(
                        {
                            "source": "semantic_scholar",
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                        }
                    )

                # Build a rich note that includes sources
                note_lines: List[str] = []
                note_lines.append(f"[{role}] Initial research summary:")
                note_lines.append(web_summary)
                note_lines.append("")
                note_lines.append("Web sources:")
                for r in web_results:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    note_lines.append(f"- {title} ({url})")

                note_lines.append("")
                note_lines.append("Semantic Scholar sources:")
                for r in scholar_results:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    note_lines.append(f"- {title} ({url})")

                note_text = "\n".join(note_lines)
                self.memory_store.add_note(goal, note_text)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Performed Tavily and Semantic Scholar search for '{goal}'",
                    }
                )
                notes_added.append(note_text)

            elif issue in {"question_mark", "todo_item"}:
                # For now, log that more directed research is required.
                # Later you can add targeted queries or PDF ingestion based on the specific question.
                followup = (
                    f"[{role}] Detected issue '{issue}': {desc}. "
                    "Further targeted research is required (not yet specialized)."
                )
                self.memory_store.add_note(goal, followup)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Logged follow up requirement for targeted research.",
                    }
                )
                notes_added.append(followup)

        return repair_actions, notes_added, citations
