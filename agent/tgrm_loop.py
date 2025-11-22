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

Levels and Swarms
-----------------
The loop is now aware of:
    - tgrm_level (1, 2, 3) via config["tgrm_level"] (default 3)
        Level 1: basic defect detection + repair (current behavior).
        Level 2: richer targeted research on questions and TODOs.
        Level 3: domain-aware and role-aware logic for swarms.

    - role: "agent", "researcher", "critic", "planner",
            "synthesizer", "explorer", etc.
      Roles are used to bias how many issues are repaired per cycle
      and how aggressive the research is.

Domain awareness
----------------
The loop now uses the `domain` tag ("general", "longevity", "math", ...)
to surface higher-level issues such as:
    - missing_biomarkers (longevity)
    - missing_mechanisms (longevity)
    - missing_formalism (math: definitions / theorems / proofs)
    - missing_connections (math: links to existing theory)

These appear as additional issue codes and pass through the same
TGRM pipeline (Test → Detect → Repair → Verify) without breaking any
existing behavior.

Engine-worker / 90-day architecture
-----------------------------------
This module is designed to be called by:
    - CoreAgent.run_cycle(...) for single cycles
    - A long-running engine_worker that orchestrates many cycles

To support that:
    - Each run_cycle returns both a machine log and a human summary.
    - Both carry delta_R, energy_E, RYE, hypotheses, citations, and
      candidate_interventions for cure/treatment-style reports.
"""

from __future__ import annotations

from datetime import datetime
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

        # Long run optimization: track which questions have already been
        # researched so we do not re-query the same text a thousand times
        # during 8h or 24h sessions.
        self._seen_questions: set[str] = set()

        # TGRM level: 1 (basic), 2 (targeted), 3 (domain + swarm aware).
        # If not set in config, we default to level 3 to unlock full power.
        try:
            self.tgrm_level: int = int(self.config.get("tgrm_level", 3))
        except Exception:
            self.tgrm_level = 3

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
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one TGRM cycle for a given research goal.

        Args:
            goal:
                Human-readable research goal for this cycle.
            cycle_index:
                Global cycle index (used for logging and history).
            role:
                Logical role (for example "agent", "researcher", "critic").
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
            biomarker_snapshot:
                Optional dict of biomarker or lab values for future
                longevity analysis (currently not used directly here).
            domain:
                Optional domain tag (for example "general", "longevity", "math")
                from presets. Preserved in the log for future
                domain-specific behavior.

        Returns:
            Dict with:
                {
                  "summary": human-facing summary,
                  "log":     full cycle log
                }
        """
        src_ctrl = self._normalise_source_controls(source_controls)
        domain_tag = domain or "general"

        # Long run context: fetch RYE stats for this goal if available.
        # This lets us switch into a maintenance mode when the system is
        # already performing well, which protects RYE during very long runs.
        avg_rye: Optional[float] = None
        total_cycles_for_goal: int = 0
        try:
            if hasattr(self.memory_store, "get_rye_stats"):
                avg, _min_rye, _max_rye, count = self.memory_store.get_rye_stats(goal=goal)
                avg_rye = avg
                total_cycles_for_goal = count
        except Exception:
            avg_rye = None
            total_cycles_for_goal = 0

        # Maintenance mode:
        # If we already have many cycles on this goal and RYE is high,
        # we avoid heavy repeated web/PubMed calls and focus on light
        # refinement. This keeps RYE high and cost low over 24h+ runs.
        maintenance_mode = False
        if (
            self.tgrm_level >= 1
            and avg_rye is not None
            and total_cycles_for_goal >= 20
            and avg_rye >= 0.8
        ):
            maintenance_mode = True

        # TEST phase: evaluate current state
        prior_notes = self.memory_store.get_notes(goal)
        status_report = self._test(goal, prior_notes)

        # DETECT phase: identify issues to repair
        issues, issue_descriptions = self._detect(status_report, domain=domain_tag)

        # REPAIR phase: apply targeted repairs
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
            maintenance_mode=maintenance_mode,
            domain=domain_tag,
        )

        # VERIFY phase: re-evaluate state after repair
        new_notes = self.memory_store.get_notes(goal)
        new_status_report = self._test(goal, new_notes)
        issues_after, _ = self._detect(new_status_report, domain=domain_tag)

        # Hypothesis generation (optional improvement targets)
        # At higher TGRM levels we allow more hypotheses; at level 1 we keep it minimal.
        max_h = 3 if self.tgrm_level == 1 else 5
        hypotheses = generate_hypotheses(goal, new_notes, citations, max_hypotheses=max_h)
        for h in hypotheses:
            self.memory_store.add_hypothesis(goal, h["text"], score=h.get("confidence"))

        # Candidate interventions / cures / treatments (lightweight extractor)
        candidate_interventions = self._extract_candidate_interventions(
            goal=goal,
            domain=domain_tag,
            notes=new_notes,
            citations=citations,
        )

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
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "role": role,
            "domain": domain_tag,
            "tgrm_level": self.tgrm_level,
            # Issues before and after
            "issues_before": issue_descriptions,
            "issues_after": issues_after,
            # Actions and artifacts
            "repairs_applied": repair_actions,
            "notes_added": notes_added,
            "citations": citations,
            "hypotheses": hypotheses,
            "candidate_interventions": candidate_interventions,
            # Raw stats and metrics
            "stats": stats,
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
            # Long run context snapshot
            "avg_rye_for_goal_before_cycle": avg_rye,
            "total_cycles_for_goal_before_cycle": total_cycles_for_goal,
            "maintenance_mode": maintenance_mode,
        }

        # Log cycle into memory
        self.memory_store.log_cycle(cycle_summary)

        # Human-readable summary (what Streamlit / engine_worker shows per cycle)
        human_summary = {
            "cycle": cycle_index,
            "role": role,
            "domain": domain_tag,
            "goal": goal,
            "issues_before": issue_descriptions,
            "issues_after": issues_after,
            "repairs": [a.get("description", "") for a in repair_actions],
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
            "notes_added": notes_added,
            "citations": citations,
            "hypotheses": hypotheses,
            "candidate_interventions": candidate_interventions,
            "maintenance_mode": maintenance_mode,
            "tgrm_level": self.tgrm_level,
        }

        return {"summary": human_summary, "log": cycle_summary}

    # ------------------------------------------------------------------
    # TGRM phases
    # ------------------------------------------------------------------
    def _test(self, goal: str, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the current state of research for this goal.

        Level 1:
            - Count notes only.
        Level 2:
            - Also track approximate citation density.
        Level 3:
            - Keep room for per-domain diagnostic stats (future extension).
        """
        known_notes_count = len(notes)

        # Basic diagnostics: citation markers inside notes
        citation_markers = 0
        if self.tgrm_level >= 2:
            for note in notes:
                content = note.get("content", "")
                if "[" in content and "]" in content:
                    citation_markers += 1

        report: Dict[str, Any] = {
            "goal": goal,
            "known_notes_count": known_notes_count,
            "notes": notes,
        }
        if self.tgrm_level >= 2:
            report["approx_citation_markers"] = citation_markers

        return report

    def _detect(
        self,
        status_report: Dict[str, Any],
        domain: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Identify issues or gaps to be repaired.

        Base (Level 1) behavior:
            - no_notes
            - question_mark
            - todo_item
            - contradiction

        Level 2:
            - can react to low citation density (few citation markers).

        Level 3:
            - domain-aware issues for longevity and math.
        """
        issues: List[str] = []
        descriptions: List[str] = []
        notes = status_report.get("notes", [])

        # --- Core generic issues (Level 1) ---
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

        # --- Level 2: citation-level diagnostics ---
        if self.tgrm_level >= 2 and notes:
            citation_markers = status_report.get("approx_citation_markers", 0)
            # Very rough heuristic: under-cited if fewer markers than notes
            if citation_markers < len(notes):
                issues.append("under_cited")
                descriptions.append(
                    "Evidence base may be thin: fewer citation markers than notes; "
                    "prioritize finding stronger primary sources."
                )

        # --- Level 3: domain-aware issues ---
        if self.tgrm_level >= 3 and notes and domain:
            dom = domain.lower()
            if dom == "longevity":
                self._detect_longevity_issues(notes, issues, descriptions)
            elif dom == "math":
                self._detect_math_issues(notes, issues, descriptions)

        return issues, descriptions

    def _detect_longevity_issues(
        self,
        notes: List[Dict[str, Any]],
        issues: List[str],
        descriptions: List[str],
    ) -> None:
        """Add extra issue types for longevity / anti-aging research."""
        text = "\n".join(n.get("content", "") for n in notes)

        # Missing explicit biomarker discussion
        if all(
            kw.lower() not in text.lower()
            for kw in ["biomarker", "blood", "lab value", "marker", "hdl", "ldl", "triglyceride"]
        ):
            issues.append("missing_biomarkers")
            descriptions.append(
                "Longevity notes lack explicit biomarker discussion; "
                "identify concrete measurable markers and how interventions affect them."
            )

        # Missing mechanisms / pathways
        if all(
            kw.lower() not in text.lower()
            for kw in [
                "mechanism",
                "pathway",
                "mTOR",
                "autophagy",
                "senescence",
                "NAD+",
                "hallmarks of aging",
            ]
        ):
            issues.append("missing_mechanisms")
            descriptions.append(
                "Mechanisms of action are underspecified; map interventions to pathways and aging hallmarks."
            )

    def _detect_math_issues(
        self,
        notes: List[Dict[str, Any]],
        issues: List[str],
        descriptions: List[str],
    ) -> None:
        """Add extra issue types for math / theory research."""
        text = "\n".join(n.get("content", "") for n in notes)

        # Missing formal definitions / theorems
        has_definition = "definition" in text.lower()
        has_theorem = "theorem" in text.lower() or "lemma" in text.lower()
        if not (has_definition and has_theorem):
            issues.append("missing_formalism")
            descriptions.append(
                "Mathematical formalism is incomplete; add explicit definitions and at least one theorem or lemma."
            )

        # Missing links to existing theory
        if all(
            kw.lower() not in text.lower()
            for kw in [
                "lyapunov",
                "markov",
                "information theory",
                "control theory",
                "stability theory",
                "ergodic",
            ]
        ):
            issues.append("missing_connections")
            descriptions.append(
                "Connections to existing mathematical frameworks are thin; "
                "relate Reparodynamics to known stability / control / information theories."
            )

    def _repair(
        self,
        goal: str,
        issues: List[str],
        descriptions: List[str],
        role: str,
        source_controls: Dict[str, bool],
        pdf_bytes: Optional[bytes],
        maintenance_mode: bool = False,
        domain: Optional[str] = None,
    ):
        """Apply repairs to address detected issues.

        This is where:
        - Tavily web search runs
        - PubMed and Semantic Scholar are queried
        - PDF ingestion can be used when available
        - Targeted research is performed for open questions or TODOs

        Long run optimization:
            - In maintenance mode, only a small number of issues are
              processed per cycle to avoid hammering web APIs.
            - Even in normal mode, we cap issues per cycle so that 24h
              runs do not explode in cost.

        Swarm / role awareness:
            - researcher / explorer: allowed to handle more issues per cycle.
            - critic: fewer issues, more focused.
            - synthesizer / planner: often do low-cost notes-only passes.
        """
        repair_actions: List[Dict[str, str]] = []
        notes_added: List[str] = []
        citations: List[Dict[str, str]] = []

        # Stats for RYE energy and improvement
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
        }

        # Base cap per cycle
        if maintenance_mode:
            base_max_issues = 2
        else:
            base_max_issues = 5  # plenty for progress, but bounded for 24h runs

        # Role-based adjustment (swarm-friendly)
        role_lower = (role or "agent").lower()
        if role_lower in {"researcher", "explorer"}:
            max_issues = base_max_issues + 2
        elif role_lower in {"critic", "planner"}:
            max_issues = max(1, base_max_issues - 2)
        elif role_lower in {"synthesizer", "integrator"}:
            # Synthesizers often focus on summarizing; limit expensive repairs.
            max_issues = max(1, base_max_issues - 3)
        else:
            max_issues = base_max_issues

        issues_to_handle = issues[:max_issues]
        descriptions_to_handle = descriptions[:max_issues]

        # Helper to register citations into MemoryStore
        def _log_citations(cites: List[Dict[str, str]]) -> None:
            for c in cites:
                self.memory_store.add_citation(goal, c)

        for issue, desc in zip(issues_to_handle, descriptions_to_handle):
            # Main "no notes" entry point: broad initial sweep
            if issue == "no_notes":
                # Even in maintenance mode, first cycle for a goal should do
                # a proper initial pass.
                note_text, new_cites, issue_stats = self._initial_research(
                    goal=goal,
                    role=role,
                    source_controls=source_controls,
                    pdf_bytes=pdf_bytes,
                    domain=domain,
                    maintenance_mode=maintenance_mode,
                )
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
                # Extract concrete questions or TODO lines from existing notes
                questions = self._extract_questions(goal, issue_type=issue)

                note_text, new_cites, issue_stats = self._targeted_research(
                    goal=goal,
                    role=role,
                    issue=issue,
                    issue_description=desc,
                    source_controls=source_controls,
                    questions=questions,
                    maintenance_mode=maintenance_mode,
                    domain=domain,
                )

                self.memory_store.add_note(goal, note_text, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": (
                            f"[{role}] Performed targeted research for open items "
                            f"(maintenance_mode={maintenance_mode})."
                        ),
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)

                for k, v in issue_stats.items():
                    stats[k] = stats.get(k, 0) + v

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

            elif issue == "under_cited":
                # Level 2+: strengthen the evidence base with more primary sources.
                note_text, new_cites, issue_stats = self._strengthen_citations(
                    goal=goal,
                    role=role,
                    source_controls=source_controls,
                    domain=domain,
                )
                self.memory_store.add_note(goal, note_text, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Searched for additional primary sources to strengthen citations.",
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)
                for k, v in issue_stats.items():
                    stats[k] = stats.get(k, 0) + v

            elif issue in {"missing_biomarkers", "missing_mechanisms"}:
                # Longevity-focused gap filling
                note_text, new_cites, issue_stats = self._domain_gap_research(
                    goal=goal,
                    role=role,
                    issue=issue,
                    description=desc,
                    source_controls=source_controls,
                    domain=domain,
                )
                self.memory_store.add_note(goal, note_text, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Filled longevity gap: {desc}",
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)
                for k, v in issue_stats.items():
                    stats[k] = stats.get(k, 0) + v

            elif issue in {"missing_formalism", "missing_connections"}:
                # Math-focused gap filling
                note_text, new_cites, issue_stats = self._domain_gap_research(
                    goal=goal,
                    role=role,
                    issue=issue,
                    description=desc,
                    source_controls=source_controls,
                    domain=domain,
                )
                self.memory_store.add_note(goal, note_text, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Filled math gap: {desc}",
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)
                for k, v in issue_stats.items():
                    stats[k] = stats.get(k, 0) + v

            else:
                # Unknown / future issue types: low-cost note so we do not silently drop it.
                note = (
                    f"[{role}] Encountered issue '{issue}' with description: {desc}. "
                    "This issue type is not yet fully handled; marking as TODO for future cycles."
                )
                self.memory_store.add_note(goal, note, role=role)
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Recorded unhandled issue type for future logic.",
                    }
                )
                notes_added.append(note)

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

    def _select_web_search_params(
        self,
        role: str,
        maintenance_mode: bool,
        domain: Optional[str],
        purpose: str,
    ) -> Dict[str, Any]:
        """Hybrid-mode selection of Tavily search level / size / topic.

        purpose ∈ {"initial", "targeted", "strengthen", "gap_repair"}
        """
        role_lower = (role or "agent").lower()
        dom = (domain or "general").lower()

        # Topic routing
        if dom in {"longevity", "biology", "medicine", "health"}:
            topic = "science"
        elif dom in {"math", "physics", "chemistry"}:
            topic = "science"
        else:
            topic = "general"

        # Base level from global TGRM level (1–3)
        base_level = max(1, min(self.tgrm_level, 3))

        # Start with base level, then adjust
        level = base_level

        # Maintenance tends to downshift one level if possible
        if maintenance_mode and base_level > 1:
            level = base_level - 1

        # Purpose-specific nudges
        if purpose in {"initial", "gap_repair", "strengthen"}:
            # Let researcher / explorer go deepest when allowed
            if not maintenance_mode and base_level == 3 and role_lower in {"researcher", "explorer"}:
                level = 3
        elif purpose == "targeted":
            if not maintenance_mode and base_level >= 2 and role_lower in {"researcher", "explorer"}:
                level = min(3, base_level)
            elif role_lower in {"critic", "planner", "synthesizer", "integrator"}:
                # Critics / planners often need focused but not super deep pulls
                level = max(1, min(base_level, 2))

        # Max results tuned by purpose
        if purpose == "initial":
            base_max = 5 if not maintenance_mode else 3
        elif purpose == "targeted":
            base_max = 6 if not maintenance_mode else 4
        elif purpose in {"strengthen", "gap_repair"}:
            base_max = 8 if not maintenance_mode else 4
        else:
            base_max = 5

        # Respect level 1 constraints: keep it very cheap
        if level == 1:
            max_results = min(base_max, 3)
        else:
            max_results = base_max

        return {"level": level, "max_results": max_results, "topic": topic}

    def _initial_research(
        self,
        goal: str,
        role: str,
        source_controls: Dict[str, bool],
        pdf_bytes: Optional[bytes],
        domain: Optional[str],
        maintenance_mode: bool,
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
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=maintenance_mode,
                domain=domain,
                purpose="initial",
            )
            web_results = self.web_tool.search(goal, **web_params)
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

    def _extract_questions(self, goal: str, issue_type: str) -> List[str]:
        """Extract concrete question or TODO lines from stored notes.

        This scans existing notes for lines that contain a '?' (for
        'question_mark') or 'TODO'/'todo' (for 'todo_item') and returns
        a deduplicated list of candidate queries.

        Long run optimization:
            - We also track which exact lines have already been used as
              questions in previous cycles, so we do not repeatedly
              hammer the web for the same text.
        """
        notes = self.memory_store.get_notes(goal)
        candidates: List[str] = []

        for note in notes:
            content = note.get("content", "")
            for line in content.splitlines():
                line_strip = line.strip()
                if issue_type == "question_mark" and "?" in line_strip:
                    if len(line_strip) > 10:
                        candidates.append(line_strip)
                elif issue_type == "todo_item" and ("TODO" in line_strip or "todo" in line_strip):
                    if len(line_strip) > 10:
                        candidates.append(line_strip)

        # Deduplicate while preserving order
        seen_local = set()
        unique_questions: List[str] = []
        for q in candidates:
            if q not in seen_local:
                seen_local.add(q)
                unique_questions.append(q)

        # Filter out questions we have already researched in previous cycles
        fresh_questions: List[str] = []
        for q in unique_questions:
            if q not in self._seen_questions:
                self._seen_questions.add(q)
                fresh_questions.append(q)

        # Limit to the first few to control cost; higher TGRM levels can afford a bit more.
        max_q = 3 if self.tgrm_level == 1 else 5
        return fresh_questions[:max_q]

    def _targeted_research(
        self,
        goal: str,
        role: str,
        issue: str,
        issue_description: str,
        source_controls: Dict[str, bool],
        questions: Optional[List[str]] = None,
        maintenance_mode: bool = False,
        domain: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, int]]:
        """Perform focused multi-source research on open questions or TODOs.

        This is the "Option A" full targeted mode:
            - Use exact question or TODO lines when available.
            - Query web, PubMed, and Semantic Scholar for each.
            - Build a rich note that links each question to sources.

        Long run optimization:
            - If there are no new questions for this issue, we log a
              lightweight maintenance note and avoid external calls.
            - In maintenance mode we still answer new questions, but we
              keep the note compact and allow the issue cap in _repair
              to limit cost.
        """
        citations: List[Dict[str, str]] = []
        stats = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
        }

        # If questions is None (no extraction attempted) we fall back to a generic prompt.
        # If questions is an empty list, that means no new questions left. We then
        # produce a lightweight note without external queries.
        if questions is None:
            questions = [f"{goal} - focus on: {issue_description}"]
        elif len(questions) == 0:
            note_lines: List[str] = []
            note_lines.append(f"[{role}] Maintenance pass on open items ({issue}) for goal:")
            note_lines.append(goal)
            note_lines.append("")
            note_lines.append(
                "No new unresolved questions were found beyond what has already been researched. "
                "This cycle performs a light consolidation of existing knowledge without new web or paper calls."
            )
            note_text = "\n".join(note_lines)
            return note_text, citations, stats

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Targeted research on open items ({issue}) for goal:")
        note_lines.append(goal)
        note_lines.append("")
        note_lines.append("Questions / TODOs considered:")
        for q in questions:
            note_lines.append(f"- {q}")
        note_lines.append("")

        # For each question, run multi-source research
        for q in questions:
            note_lines.append("### Question-focused search:")
            note_lines.append(q)
            note_lines.append("")

            # Web search
            if source_controls.get("web", True):
                web_params = self._select_web_search_params(
                    role=role,
                    maintenance_mode=maintenance_mode,
                    domain=domain,
                    purpose="targeted",
                )
                web_results = self.web_tool.search(q, **web_params)
                stats["web_calls"] += 1
                web_summary = self.web_tool.summarize_results(web_results)
                web_cites = self.web_tool.to_citations(web_results)
                citations.extend(web_cites)

                note_lines.append("Web summary (Tavily):")
                note_lines.append(web_summary)
                note_lines.append("Web sources:")
                for c in web_cites:
                    note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
                note_lines.append("")

            # PubMed
            if source_controls.get("pubmed", False):
                pubmed_results = self.pubmed_tool.search(q, max_results=5)
                stats["pubmed_calls"] += 1
                citations.extend(pubmed_results)
                note_lines.append("PubMed sources:")
                for r in pubmed_results:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

            # Semantic Scholar
            if source_controls.get("semantic", False):
                sem_results = self.semantic_tool.search(q, max_results=5)
                stats["semantic_calls"] += 1
                citations.extend(sem_results)
                note_lines.append("Semantic Scholar sources:")
                for r in sem_results:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

        note_text = "\n".join(note_lines)
        return note_text, citations, stats

    def _strengthen_citations(
        self,
        goal: str,
        role: str,
        source_controls: Dict[str, bool],
        domain: Optional[str],
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, int]]:
        """Specialized repair step for 'under_cited' issues.

        Focused on finding stronger primary sources (trials, math papers,
        benchmark studies, etc.) to raise the evidence density.
        """
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
        note_lines.append(f"[{role}] Strengthening citations for goal:")
        note_lines.append(goal)
        note_lines.append("")

        query = f"{goal} primary sources randomized trial benchmark formal paper"
        note_lines.append(f"Search query for stronger evidence: {query}")
        note_lines.append("")

        # Web search
        if source_controls.get("web", True):
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=False,
                domain=domain,
                purpose="strengthen",
            )
            web_results = self.web_tool.search(query, **web_params)
            stats["web_calls"] += 1
            web_summary = self.web_tool.summarize_results(web_results)
            web_cites = self.web_tool.to_citations(web_results)
            citations.extend(web_cites)

            note_lines.append("Web summary (stronger evidence focus):")
            note_lines.append(web_summary)
            note_lines.append("Web sources:")
            for c in web_cites:
                note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
            note_lines.append("")

        # PubMed
        if source_controls.get("pubmed", False):
            pubmed_results = self.pubmed_tool.search(query, max_results=10)
            stats["pubmed_calls"] += 1
            citations.extend(pubmed_results)
            note_lines.append("PubMed sources (stronger evidence):")
            for r in pubmed_results:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        # Semantic Scholar
        if source_controls.get("semantic", False):
            sem_results = self.semantic_tool.search(query, max_results=10)
            stats["semantic_calls"] += 1
            citations.extend(sem_results)
            note_lines.append("Semantic Scholar sources (stronger evidence):")
            for r in sem_results:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        note_text = "\n".join(note_lines)
        return note_text, citations, stats

    def _domain_gap_research(
        self,
        goal: str,
        role: str,
        issue: str,
        description: str,
        source_controls: Dict[str, bool],
        domain: Optional[str],
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, int]]:
        """Generic helper to fill domain-specific gaps.

        For longevity:
            - missing_biomarkers → search for key biomarkers and lab panels.
            - missing_mechanisms → search for pathways, hallmarks, mechanisms.

        For math:
            - missing_formalism → search for definitions, theorems, formal models.
            - missing_connections → search for links to existing stability / control theories.
        """
        dom = (domain or "general").lower()
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
        note_lines.append(f"[{role}] Domain-specific gap repair for goal:")
        note_lines.append(goal)
        note_lines.append("")
        note_lines.append(f"Issue: {issue}")
        note_lines.append(description)
        note_lines.append("")

        # Build a focused query
        if dom == "longevity":
            if issue == "missing_biomarkers":
                query = (
                    f"biomarkers panel clinical endpoints for {goal} healthspan longevity all-cause mortality"
                )
            elif issue == "missing_mechanisms":
                query = (
                    f"mechanisms pathways hallmarks of aging for interventions related to {goal}"
                )
            else:
                query = f"{goal} biomarkers mechanisms healthspan longevity"
        elif dom == "math":
            if issue == "missing_formalism":
                query = (
                    f"formal definition theorem stability framework similar to reparodynamics and RYE"
                )
            elif issue == "missing_connections":
                query = (
                    f"connections between stability theory Lyapunov control Markov processes and repair dynamics"
                )
            else:
                query = f"{goal} mathematical stability formalization"
        else:
            # Generic fallback
            query = f"{goal} {issue} {description}"

        note_lines.append(f"Focused query used for gap repair: {query}")
        note_lines.append("")

        # Web search
        if source_controls.get("web", True):
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=False,
                domain=domain,
                purpose="gap_repair",
            )
            web_results = self.web_tool.search(query, **web_params)
            stats["web_calls"] += 1
            web_summary = self.web_tool.summarize_results(web_results)
            web_cites = self.web_tool.to_citations(web_results)
            citations.extend(web_cites)

            note_lines.append("Web summary (gap repair):")
            note_lines.append(web_summary)
            note_lines.append("Web sources:")
            for c in web_cites:
                note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
            note_lines.append("")

        # PubMed (especially useful in longevity)
        if dom == "longevity" and source_controls.get("pubmed", False):
            pubmed_results = self.pubmed_tool.search(query, max_results=10)
            stats["pubmed_calls"] += 1
            citations.extend(pubmed_results)
            note_lines.append("PubMed sources (gap repair):")
            for r in pubmed_results:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        # Semantic Scholar (useful in both longevity and math)
        if source_controls.get("semantic", False):
            sem_results = self.semantic_tool.search(query, max_results=10)
            stats["semantic_calls"] += 1
            citations.extend(sem_results)
            note_lines.append("Semantic Scholar sources (gap repair):")
            for r in sem_results:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        note_text = "\n".join(note_lines)
        return note_text, citations, stats

    # ------------------------------------------------------------------
    # Candidate intervention extractor
    # ------------------------------------------------------------------
    def _extract_candidate_interventions(
        self,
        goal: str,
        domain: str,
        notes: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Lightweight extractor for candidate interventions / cures / treatments.

        This does NOT try to be medically authoritative. It simply:
            - scans citations for unique titles
            - returns them as candidate entries tagged with goal + domain

        The real filtering / ranking / safety checks can be done later
        in report_generator.py or in downstream human review.
        """
        candidates: List[Dict[str, Any]] = []
        seen_titles: set[str] = set()

        for c in citations:
            title = (c.get("title") or "").strip()
            if not title:
                continue
            if title in seen_titles:
                continue
            seen_titles.add(title)

            entry = {
                "label": title,
                "source": c.get("source"),
                "url": c.get("url"),
                "goal": goal,
                "domain": domain or "general",
            }
            candidates.append(entry)

        return candidates
