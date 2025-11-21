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

        # Initialise tools. In a full implementation these could take API keys
        self.web_tool = WebResearchTool()
        self.paper_tool = PaperTool()
        self.file_tool = FileTool()

    def run_cycle(self, goal: str, cycle_index: int) -> Dict[str, Any]:
        """Run one TGRM cycle for a given research goal."""
        # Test phase – evaluate current state
        prior_notes = self.memory_store.get_notes(goal)
        status_report = self._test(goal, prior_notes)

        # Detect phase – identify issues to repair
        issues, issue_descriptions = self._detect(status_report)

        # Repair phase – apply targeted repairs
        repair_actions, notes_added = self._repair(goal, issues, issue_descriptions)

        # Verify phase – re-evaluate state after repair
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
            "issues_detected": issue_descriptions,
            "issues_after": issues_after,
            "repairs_applied": repair_actions,
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
        }

        # Log cycle
        self.memory_store.log_cycle(cycle_summary)

        # Human-readable summary
        human_summary = {
            "cycle": cycle_index,
            "issues_before": issue_descriptions,
            "repairs": [a["description"] for a in repair_actions],
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
            "notes_added": notes_added,
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
    ):
        """Apply repairs to address detected issues."""
        repair_actions: List[Dict[str, str]] = []
        notes_added: List[str] = []

        for issue, desc in zip(issues, descriptions):
            if issue == "no_notes":
                # Perform a web search as a first repair step
                results = self.web_tool.search(goal)
                summary = self.web_tool.summarize_results(results)
                note_text = f"Initial research summary: {summary}"
                self.memory_store.add_note(goal, note_text)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"Performed web search for '{goal}'",
                    }
                )
                notes_added.append(note_text)
            elif issue in {"question_mark", "todo_item"}:
                # For now, log that more directed research is needed
                followup = (
                    f"Detected issue '{issue}': {desc}. "
                    "Further targeted research is required (not implemented in stub)."
                )
                self.memory_store.add_note(goal, followup)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": "Logged follow-up requirement for targeted research.",
                    }
                )
                notes_added.append(followup)

        return repair_actions, notes_added
