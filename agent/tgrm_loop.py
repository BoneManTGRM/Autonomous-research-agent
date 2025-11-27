"""Implementation of the TGRM loop for the research agent.

This module defines a `TGRMLoop` class that encapsulates the core logic
of the agent's Targeted Gradient Repair Mechanism (TGRM). The loop runs
full reparodynamic cycles for a single research goal and returns both a
machine friendly log and a human summary for each cycle.

Reparodynamics view
-------------------
The agent is treated as a reparodynamic system:
    - Each cycle tries to reduce defects (gaps, TODOs, unanswered questions)
      while spending as little energy as possible.
    - delta_R measures improvement; E measures cost; RYE = delta_R / E is the
      core efficiency metric.
    - Short term RYE gradients, equilibrium labels, and stability scores are
      computed from recent history to support long run autonomy.

TGRM phases (implemented below)
-------------------------------
    Test   : evaluate current notes and current state for this goal.
    Detect : find gaps, TODOs, unanswered questions, and contradictions.
    Repair : perform targeted web and scientific actions (web, PubMed,
             Semantic Scholar, PDF ingestion) using the shared Toolbelt or
             tools registry.
    Verify : re test, compute delta_R and RYE, update history, and log a
             detailed cycle entry in the MemoryStore.

Levels and Swarms
-----------------
The loop is aware of:
    - tgrm_level (1, 2, 3) via config["tgrm_level"] (default 3)
        Level 1: basic defect detection and repair.
        Level 2: richer targeted research on questions and TODOs.
        Level 3: domain aware and role aware logic for swarms.

    - role: "agent", "researcher", "critic", "planner",
            "synthesizer", "explorer", etc.
      Roles bias how many issues are repaired per cycle and how aggressive
      external research should be, so a swarm can mix explorers, critics,
      planners, and integrators safely.

Domain awareness
----------------
The loop uses the `domain` tag ("general", "longevity", "math", ...)
to surface higher level issues such as:
    - missing_biomarkers (longevity)
    - missing_mechanisms (longevity)
    - missing_formalism (math: definitions or theorems or proofs)
    - missing_connections (math: links to existing theory)

These appear as additional issue codes and pass through the same
TGRM pipeline without breaking any existing behavior.

Reparodynamic 90 day architecture
---------------------------------
This module is designed to be called by:
    - CoreAgent.run_cycle(...) for single cycles.
    - Continuous runners and swarm controllers for multi day and multi
      agent runs.

To support that:
    - Each run_cycle returns a machine log and a human summary.
    - Both carry delta_R, energy_E, Energy, RYE, hypotheses, citations,
      candidate interventions, and candidate_hypotheses for the discovery
      stack.
    - Extra fields expose RYE gradients, equilibrium status, and a
      breakthrough_score that higher level components can track over
      weeks or months.

Ultra mode
----------
When config["ultra_speed"] is true the loop:
    - handles more issues per cycle before maintenance mode
    - exposes a compact `short_view` block for fast UI and logging
    - adds `meta_signals` for the meta controller to read RYE and
      contradiction rates without parsing full logs.

Longevity two stage mode
------------------------
When run_cycle is called with stage="idea" or stage="verify" and optional
hallmark or subgoal, the loop:
    - tags cycles with hallmark and stage for later reporting
    - can push high value hypotheses into a ReplayBuffer for curriculum
      style learning via _log_replay_candidate

Stability kernel and discovery manager
--------------------------------------
If available, the loop can optionally:
    - stream per cycle metrics into a StabilityKernel for long run
      stability_index, recovery_momentum, and noise estimates
    - stream per cycle signals into a DiscoveryManager for tiered
      discovery classification (tier_0 to tier_3)

Both integrations are soft optional and never break runs if modules
are missing or misconfigured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .rye_metrics import compute_delta_r, compute_energy, compute_rye

# Optional stability kernel integration
try:
    from .stability_kernel import StabilityKernel  # type: ignore[import]
except Exception:  # pragma: no cover
    StabilityKernel = None  # type: ignore[assignment]

# Optional discovery manager integration
try:
    from .discovery_manager import DiscoveryManager  # type: ignore[import]
except Exception:  # pragma: no cover
    DiscoveryManager = None  # type: ignore[assignment]

# Optional imports for external tools and hypothesis engine.
# Each has a safe fallback so the loop still runs if the modules
# or dependencies are missing.

# Web research
try:
    from .tools_web import WebResearchTool  # type: ignore[import]
except Exception:  # pragma: no cover
    WebResearchTool = None  # type: ignore[assignment]

# Paper and PDF tools
try:
    from .tools_papers import PaperTool  # type: ignore[import]
except Exception:  # pragma: no cover
    PaperTool = None  # type: ignore[assignment]

# File tools (currently lightly used but optional)
try:
    from .tools_files import FileTool  # type: ignore[import]
except Exception:  # pragma: no cover
    FileTool = None  # type: ignore[assignment]

# PubMed and Semantic Scholar
try:
    from .tools_pubmed import PubMedTool  # type: ignore[import]
except Exception:  # pragma: no cover
    PubMedTool = None  # type: ignore[assignment]

try:
    from .tools_semantic_scholar import SemanticScholarTool  # type: ignore[import]
except Exception:  # pragma: no cover
    SemanticScholarTool = None  # type: ignore[assignment]

# Hypothesis engine
try:
    from .hypothesis_engine import generate_hypotheses  # type: ignore[import]
except Exception:  # pragma: no cover

    def generate_hypotheses(
        goal: str,
        notes: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        max_hypotheses: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fallback hypothesis generator when hypothesis_engine is unavailable.

        This returns an empty list so the rest of the loop can still run.
        """
        return []


# Optional Toolbelt / ToolUsage import to mirror CoreAgent behavior.
try:
    from .tools import Toolbelt, ToolUsage  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    Toolbelt = None  # type: ignore[assignment]

    class ToolUsage:  # type: ignore[no-redef]
        """Minimal fallback usage tracker if tools.ToolUsage is unavailable."""

        def __init__(self) -> None:
            self.web_calls: int = 0
            self.browser_actions: int = 0
            self.code_execs: int = 0
            self.sql_queries: int = 0
            self.data_loads: int = 0
            self.approx_tokens: int = 0


# ---------------------------------------------------------------------------
# Null or fallback tool implementations
# ---------------------------------------------------------------------------


class _NullWebTool:
    """Fallback web research tool when tools_web or dependencies are missing."""

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        return []

    def summarize_results(self, results: List[Dict[str, Any]]) -> str:
        return "Web search disabled or unavailable for this run."

    def to_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []


class _NullPaperTool:
    """Fallback paper tool when tools_papers is missing."""

    def ingest_bytes(self, pdf_bytes: bytes) -> str:
        return ""

    def summarise(self, text: str) -> str:
        return "PDF ingestion disabled or unavailable for this run."


class _NullFileTool:
    """Fallback file tool when tools_files is missing."""

    def __init__(self) -> None:
        pass


class _NullPubMedTool:
    """Fallback PubMed tool when tools_pubmed is missing."""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        return []


class _NullSemanticScholarTool:
    """Fallback Semantic Scholar tool when tools_semantic_scholar is missing."""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        return []


class TGRMLoop:
    """Encapsulate the TGRM loop logic for one research cycle.

    This class is intentionally self contained and stateless between runs
    except for a few small caches such as _seen_questions. All long run
    state is stored in the MemoryStore so that CoreAgent and engine_worker
    can orchestrate continuous and swarm runs safely.

    Tools:
        - tools can be:
            * a Toolbelt-like instance (with new_usage_tracker, browser, etc.)
            * a plain dict or registry of tool callables
            * None, in which case we try to instantiate Toolbelt if available
              or fall back to an empty registry.

        - For energy accounting we always create a ToolUsage-like tracker
          via a local factory, even if no Toolbelt is present.
    """

    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[Any] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # Shared tools layer (Toolbelt or registry or dict) mirroring CoreAgent.
        if tools is not None:
            self.tools = tools
        else:
            if Toolbelt is not None:
                try:
                    self.tools = Toolbelt()
                except Exception:
                    self.tools = {}
            else:
                self.tools = {}

        # Usage tracker factory so we can always track energy even if tools
        # is just a dict or a minimal Toolbelt without tracking.
        if hasattr(self.tools, "new_usage_tracker"):
            # Toolbelt style API
            self._usage_factory = self.tools.new_usage_tracker  # type: ignore[assignment]
        else:

            def _default_usage_factory() -> ToolUsage:
                return ToolUsage()

            self._usage_factory = _default_usage_factory

        # Core tools with safe fallbacks
        try:
            if WebResearchTool is not None:
                self.web_tool = WebResearchTool()  # type: ignore[call-arg]
            else:
                self.web_tool = _NullWebTool()
        except Exception:
            self.web_tool = _NullWebTool()

        try:
            if PaperTool is not None:
                self.paper_tool = PaperTool()  # type: ignore[call-arg]
            else:
                self.paper_tool = _NullPaperTool()
        except Exception:
            self.paper_tool = _NullPaperTool()

        try:
            if FileTool is not None:
                self.file_tool = FileTool()  # type: ignore[call-arg]
            else:
                self.file_tool = _NullFileTool()
        except Exception:
            self.file_tool = _NullFileTool()

        try:
            if PubMedTool is not None:
                self.pubmed_tool = PubMedTool()  # type: ignore[call-arg]
            else:
                self.pubmed_tool = _NullPubMedTool()
        except Exception:
            self.pubmed_tool = _NullPubMedTool()

        try:
            if SemanticScholarTool is not None:
                self.semantic_tool = SemanticScholarTool()  # type: ignore[call-arg]
            else:
                self.semantic_tool = _NullSemanticScholarTool()
        except Exception:
            self.semantic_tool = _NullSemanticScholarTool()

        # Optional stability kernel and discovery manager
        self.stability_kernel = None
        self.discovery_manager = None

        if StabilityKernel is not None and self.config.get("enable_stability_kernel", True):
            try:
                self.stability_kernel = StabilityKernel()
            except Exception:
                self.stability_kernel = None

        if DiscoveryManager is not None and self.config.get("enable_discovery_manager", True):
            try:
                self.discovery_manager = DiscoveryManager()
            except Exception:
                self.discovery_manager = None

        # Long run optimization: track which questions have already been
        # researched so we do not re query the same text many times during
        # 8 hour or 24 hour sessions.
        self._seen_questions: set[str] = set()

        # TGRM level: 1 (basic), 2 (targeted), 3 (domain plus swarm aware).
        # If not set in config, default to level 3 to unlock full power.
        try:
            self.tgrm_level = int(self.config.get("tgrm_level", 3))
        except Exception:
            self.tgrm_level = 3

        # Sliding window size for short term RYE gradient estimates
        self.rye_window_size = int(self.config.get("rye_window_size", 20))

        # Ultra speed mode and strict pipeline flag (used by meta controller and UI)
        self.ultra_speed = bool(self.config.get("ultra_speed", False))
        self.strict_pipeline = bool(self.config.get("strict_pipeline", True))

    # ------------------------------------------------------------------
    # Small helper for token estimation (for energy accounting)
    # ------------------------------------------------------------------
    def _estimate_tokens(self, text: str) -> int:
        """Very rough token estimate from text length."""
        if not text:
            return 0
        # Rough heuristic: about 4 characters per token
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------
    def _tag_citations(
        self,
        citations: List[Dict[str, Any]],
        *,
        goal: str,
        query: str,
        channel: str,
        phase: str,
    ) -> List[Dict[str, Any]]:
        """Ensure every citation has minimal provenance fields.

        Fields added:
            - goal: research goal for this cycle
            - query: text used to fetch this citation
            - channel: "web", "pubmed", "semantic", "pdf", etc
            - phase: "initial", "targeted", "strengthen", "gap_repair"
            - created_at: ISO timestamp
        """
        stamped: List[Dict[str, Any]] = []
        created_at = datetime.utcnow().isoformat() + "Z"

        for c in citations or []:
            if not isinstance(c, dict):
                c = {"title": str(c)}
            c = dict(c)  # shallow copy
            c.setdefault("goal", goal)
            c.setdefault("query", query)
            c.setdefault("channel", channel)
            c.setdefault("phase", phase)
            c.setdefault("created_at", created_at)
            if not c.get("source"):
                c["source"] = channel
            stamped.append(c)

        return stamped

    # ------------------------------------------------------------------
    # History helpers for RYE gradients and equilibrium status
    # ------------------------------------------------------------------
    def _get_recent_history_for_goal(
        self,
        goal: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return recent history rows for this goal if the store supports it.

        Falls back to empty list if not implemented.
        """
        try:
            if hasattr(self.memory_store, "get_cycle_history_for_goal"):
                rows = self.memory_store.get_cycle_history_for_goal(goal, limit=limit)  # type: ignore[attr-defined]
                if isinstance(rows, list):
                    return rows
        except Exception:
            pass

        try:
            if hasattr(self.memory_store, "get_cycle_history"):
                full = self.memory_store.get_cycle_history()  # type: ignore[attr-defined]
                if isinstance(full, list):
                    filtered = [r for r in full if r.get("goal") == goal]
                    return filtered[-limit:]
        except Exception:
            pass

        return []

    def _compute_rye_gradient_and_equilibrium(
        self,
        goal: str,
        current_rye: Optional[float],
        delta_r: float,
        energy_e: float,
        domain: str,
    ) -> Dict[str, Any]:
        """Compute short term RYE gradient, equilibrium status, and stability score.

        This uses recent history from MemoryStore but never raises even if
        history is missing. It is meant to give higher level components
        an approximate sense of whether the system is:
            - climbing (improving RYE)
            - plateaued (near equilibrium)
            - oscillating or unstable
        """
        result: Dict[str, Any] = {
            "rye_gradient": None,
            "rye_window_mean": None,
            "rye_window_std": None,
            "equilibrium_label": "unknown",
            "equilibrium_score": None,
            "oscillation_score": None,
        }

        if current_rye is None:
            return result

        history = self._get_recent_history_for_goal(
            goal, limit=max(self.rye_window_size, 10)
        )
        rye_values: List[float] = []
        for row in history:
            val = row.get("RYE") or row.get("rye") or row.get("rye_value")
            if isinstance(val, (int, float)):
                rye_values.append(float(val))

        # Include the current cycle in the window
        rye_values.append(float(current_rye))
        if not rye_values:
            return result

        # Sliding window statistics
        window = rye_values[-self.rye_window_size :]
        n = len(window)
        mean_val = sum(window) / n
        var_val = 0.0
        if n > 1:
            var_val = sum((x - mean_val) ** 2 for x in window) / (n - 1)
        std_val = var_val ** 0.5

        result["rye_window_mean"] = mean_val
        result["rye_window_std"] = std_val

        # Simple gradient estimate using last and first in window
        if n > 1:
            gradient = (window[-1] - window[0]) / max(1, n - 1)
        else:
            gradient = 0.0
        result["rye_gradient"] = gradient

        # Equilibrium heuristics:
        #   - high mean RYE with low variance -> stable high equilibrium
        #   - moderate mean with low variance -> stable plateau
        #   - low mean or high variance -> unstable or oscillating
        low_std_threshold = 0.08
        high_std_threshold = 0.25
        high_mean_threshold = 0.82
        mid_mean_threshold = 0.6

        equilibrium_score = 0.0
        oscillation_score = 0.0
        label = "exploring"

        if mean_val >= high_mean_threshold and std_val <= low_std_threshold:
            label = "high_equilibrium"
            equilibrium_score = 0.9
        elif mean_val >= mid_mean_threshold and std_val <= low_std_threshold:
            label = "plateau_equilibrium"
            equilibrium_score = 0.7
        elif std_val >= high_std_threshold:
            label = "oscillating"
            oscillation_score = min(1.0, std_val / 0.6)
        elif mean_val < 0.3:
            label = "low_efficiency"
        else:
            label = "transient"

        # Tiny domain adjustment: longevity work often tolerates lower RYE
        # because primary evidence is harder to obtain. Math expects higher.
        dom_lower = (domain or "general").lower()
        if dom_lower == "longevity":
            equilibrium_score *= 1.05
        elif dom_lower == "math":
            equilibrium_score *= 0.95

        equilibrium_score = max(0.0, min(1.0, equilibrium_score))
        oscillation_score = max(0.0, min(1.0, oscillation_score))

        result["equilibrium_label"] = label
        result["equilibrium_score"] = equilibrium_score
        result["oscillation_score"] = oscillation_score

        return result

    def _compute_breakthrough_score(
        self,
        goal: str,
        domain: str,
        current_rye: Optional[float],
        delta_r: float,
        energy_e: float,
        equilibrium_info: Dict[str, Any],
        issue_code_counts_before: Dict[str, int],
        issue_code_counts_after: Dict[str, int],
        hypotheses: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Heuristic breakthrough score on a 0 to 1 scale.

        This is not a claim of real discovery. It is a reparodynamic
        internal signal indicating whether the system believes it has
        crossed a meaningful threshold for:
            - sustained RYE
            - reduced open issues
            - rich and diverse citations
            - a focused, small set of high value hypotheses
        """
        info: Dict[str, Any] = {
            "breakthrough_score": None,
            "flags": [],
        }

        if current_rye is None:
            info["breakthrough_score"] = 0.0
            return info

        score = 0.0
        flags: List[str] = []

        rye_val = float(current_rye)
        mean_window = equilibrium_info.get("rye_window_mean") or rye_val
        std_window = equilibrium_info.get("rye_window_std") or 0.0

        # 1. Current and window RYE
        if rye_val >= 0.85:
            score += 0.25
            flags.append("high_current_rye")
        elif rye_val >= 0.7:
            score += 0.15

        if mean_window >= 0.8 and std_window <= 0.15:
            score += 0.25
            flags.append("sustained_high_rye")
        elif mean_window >= 0.65 and std_window <= 0.2:
            score += 0.15

        # 2. Issue reduction
        total_before = sum(issue_code_counts_before.values()) or 0
        total_after = sum(issue_code_counts_after.values()) or 0

        if total_before > 0:
            reduction = max(0.0, float(total_before - total_after)) / float(
                total_before
            )
        else:
            reduction = 0.0

        if reduction >= 0.7 and total_after <= 3:
            score += 0.2
            flags.append("large_issue_reduction")
        elif reduction >= 0.4:
            score += 0.1

        # 3. Citation richness
        unique_sources = set()
        for c in citations:
            key = (c.get("source"), c.get("url"))
            unique_sources.add(key)
        source_count = len(unique_sources)

        if source_count >= 50:
            score += 0.15
            flags.append("rich_citation_base")
        elif source_count >= 20:
            score += 0.08

        # 4. Hypothesis focus
        hyp_count = len(hypotheses)
        if 1 <= hyp_count <= 5:
            score += 0.1
            flags.append("focused_hypothesis_set")
        elif hyp_count > 10:
            score -= 0.05

        # 5. Equilibrium label
        eq_label = equilibrium_info.get("equilibrium_label")
        if eq_label == "high_equilibrium":
            score += 0.1
        elif eq_label == "oscillating":
            score -= 0.05

        # Domain adjustments
        dom_lower = domain.lower()
        if dom_lower == "longevity":
            score *= 1.05
        elif dom_lower == "math":
            score *= 1.05

        # Clamp and finalize
        score = max(0.0, min(1.0, score))
        info["breakthrough_score"] = score
        info["flags"] = flags

        return info

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
        stage: str = "idea",
        hallmark: Optional[str] = None,
        subgoal: Optional[str] = None,
        replay_buffer: Optional[Any] = None,
        curriculum_state: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        swarm_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one TGRM cycle for a given research goal.

        Args:
            goal:
                Human readable research goal for this cycle.
            cycle_index:
                Global cycle index used for logging and history.
            role:
                Logical role such as "agent", "researcher", or "critic".
            source_controls:
                Optional switches like:
                    {
                      "web": True,
                      "pubmed": True,
                      "semantic": True,
                      "pdf": True,
                      "biomarkers": False,
                    }
                If None, defaults are used.
            pdf_bytes:
                Optional PDF content that can be ingested as part of the
                Repair phase if "pdf" is enabled.
            biomarker_snapshot:
                Optional dict of biomarker or lab values. Reserved for future
                longevity analysis.
            domain:
                Optional domain tag such as "general", "longevity", or "math".
                Preserved in the log for future domain specific behavior.
            stage:
                Optional TGRM stage hint for longevity style two stage runs:
                "idea" or "verify". Defaults to "idea".
            hallmark:
                Optional hallmark label for longevity runs, such as
                "mitochondria" or "senescence".
            subgoal:
                Optional sub target such as "mito_membrane_potential".
            replay_buffer:
                Optional ReplayBuffer like object where high value hypotheses
                and patterns can be stored for curriculum learning.
            curriculum_state:
                Optional dict with curriculum hints such as phase name,
                progress fraction, or active pathway focus.
            run_id:
                Optional id of the long running experiment or swarm run.
            agent_id:
                Optional id of the agent within a swarm.
            swarm_profile:
                Optional dict describing swarm context such as size, layer,
                and role weights. Safe to omit.
        Returns:
            Dict with:
                {
                  "summary": human facing summary,
                  "log":     full cycle log,
                  "tool_stats": per cycle tool stats,
                  "citations": list of citation dicts
                }
        """
        src_ctrl = self._normalise_source_controls(source_controls)
        domain_tag = domain or "general"

        stage_tag = (stage or "idea").lower()
        if stage_tag not in {"idea", "verify"}:
            stage_tag = "idea"

        # Per cycle tool usage tracker (for energy E and diagnostics)
        tool_usage: ToolUsage = self._usage_factory()

        # Long run context: fetch RYE stats for this goal if available
        avg_rye: Optional[float] = None
        total_cycles_for_goal: int = 0
        try:
            if hasattr(self.memory_store, "get_rye_stats"):
                avg, _min_rye, _max_rye, count = self.memory_store.get_rye_stats(
                    goal=goal
                )  # type: ignore[attr-defined]
                avg_rye = avg
                total_cycles_for_goal = count
        except Exception:
            avg_rye = None
            total_cycles_for_goal = 0

        # Maintenance mode:
        # If there are many cycles on this goal and RYE is high, avoid heavy
        # repeated external calls and focus on light refinement.
        maintenance_mode = False
        if (
            self.tgrm_level >= 1
            and avg_rye is not None
            and total_cycles_for_goal >= 20
            and avg_rye >= 0.8
        ):
            maintenance_mode = True

        # Strict pipeline phase log for Ultra mode and debugging
        phases: Dict[str, Any] = {
            "stage": stage_tag,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "curriculum_state": curriculum_state or {},
        }

        # TEST phase
        prior_notes = self.memory_store.get_notes(goal)
        status_report = self._test(goal, prior_notes)
        phases["test_before"] = {
            "known_notes_count": status_report.get("known_notes_count", 0),
            "approx_citation_markers": status_report.get(
                "approx_citation_markers"
            ),
        }

        # DETECT phase
        issues, issue_descriptions = self._detect(status_report, domain=domain_tag)
        phases["detect_before"] = {
            "issue_codes": issues,
            "issue_descriptions": issue_descriptions,
        }

        # Discovery friendly issue structure before repair
        issue_code_counts_before: Dict[str, int] = {}
        for code in issues:
            issue_code_counts_before[code] = issue_code_counts_before.get(code, 0) + 1

        domain_issue_codes = [
            "missing_biomarkers",
            "missing_mechanisms",
            "missing_formalism",
            "missing_connections",
        ]
        domain_issue_flags_before = {
            code: (code in issues) for code in domain_issue_codes
        }

        has_questions_before = "question_mark" in issues
        has_todos_before = "todo_item" in issues
        has_contradictions_before = "contradiction" in issues

        # REPAIR phase (tool aware)
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
            tool_usage=tool_usage,
        )
        phases["repair"] = {
            "handled_issue_codes": [a.get("issue") for a in repair_actions],
            "notes_added_count": len(notes_added),
            "citations_added_count": len(citations),
        }

        # VERIFY phase
        new_notes = self.memory_store.get_notes(goal)
        new_status_report = self._test(goal, new_notes)
        issues_after, _ = self._detect(new_status_report, domain=domain_tag)

        phases["test_after"] = {
            "known_notes_count": new_status_report.get(
                "known_notes_count", len(new_notes)
            ),
        }
        phases["detect_after"] = {
            "issue_codes": issues_after,
        }

        issue_code_counts_after: Dict[str, int] = {}
        for code in issues_after:
            issue_code_counts_after[code] = issue_code_counts_after.get(code, 0) + 1

        domain_issue_flags_after = {
            code: (code in issues_after) for code in domain_issue_codes
        }
        has_questions_after = "question_mark" in issues_after
        has_todos_after = "todo_item" in issues_after
        has_contradictions_after = "contradiction" in issues_after

        # Hypothesis generation (now structured for discovery and learning)
        max_h = 3 if self.tgrm_level == 1 else 5
        raw_hypotheses = generate_hypotheses(
            goal, new_notes, citations, max_hypotheses=max_h
        )

        hypotheses: List[Dict[str, Any]] = []
        candidate_hypotheses: List[Dict[str, Any]] = []

        for idx, h in enumerate(raw_hypotheses):
            if isinstance(h, str):
                text = h
                conf = None
            elif isinstance(h, dict):
                text = str(h.get("text") or h.get("title") or "")
                conf = h.get("confidence") or h.get("score")
            else:
                text = str(h)
                conf = None

            if not text:
                continue

            hyp_id = f"h_{cycle_index}_{idx}"
            hyp_record: Dict[str, Any] = {
                "id": hyp_id,
                "title": text[:160],
                "description": text,
                "text": text,
                "confidence": conf,
                "tags": [domain_tag, "auto_generated"],
            }
            if hallmark:
                hyp_record["tags"].append(f"hallmark:{hallmark}")
            if subgoal:
                hyp_record["tags"].append(f"subgoal:{subgoal}")

            hypotheses.append(hyp_record)

            candidate_hypotheses.append(
                {
                    "title": hyp_record["title"],
                    "description": hyp_record["description"],
                    "tags": hyp_record["tags"],
                }
            )

            # Store minimal hypothesis text in MemoryStore for long run learning
            try:
                self.memory_store.add_hypothesis(goal, text, score=conf)
            except Exception:
                pass

        # Candidate interventions
        candidate_interventions = self._extract_candidate_interventions(
            goal=goal,
            domain=domain_tag,
            notes=new_notes,
            citations=citations,
        )

        # Metrics (Reparodynamics: delta_R / E)
        # We break delta_R into components for richer analysis.
        delta_r_components = {
            "issue_reduction": max(0, len(issues) - len(issues_after)),
            "repairs_applied": len(repair_actions),
            "hypotheses": len(hypotheses),
            "sources_used": stats.get("sources_used", 0),
            "contradictions_resolved": stats.get("contradictions_resolved", 0),
        }
        delta_r = compute_delta_r(
            issues_before=len(issues),
            issues_after=len(issues_after),
            repairs_applied=len(repair_actions),
            contradictions_resolved=stats.get("contradictions_resolved", 0),
            hypotheses_generated=len(hypotheses),
            sources_used=stats.get("sources_used", 0),
        )

        # Include tokens from this cycle in energy accounting
        energy_e = compute_energy(
            actions_taken=repair_actions,
            web_calls=stats.get("web_calls", 0),
            pubmed_calls=stats.get("pubmed_calls", 0),
            semantic_calls=stats.get("semantic_calls", 0),
            pdf_ingestions=stats.get("pdf_ingestions", 0),
            tokens_estimate=tool_usage.approx_tokens,
            # Optional swarm aware hints (ignored by older compute_energy)
            swarm_size=(swarm_profile or {}).get("swarm_size"),
            swarm_layer=(swarm_profile or {}).get("layer"),
        )
        rye_value = compute_rye(delta_r, energy_e)

        # RYE gradient and equilibrium status
        equilibrium_info = self._compute_rye_gradient_and_equilibrium(
            goal=goal,
            current_rye=rye_value,
            delta_r=delta_r,
            energy_e=energy_e,
            domain=domain_tag,
        )

        # Breakthrough score
        breakthrough_info = self._compute_breakthrough_score(
            goal=goal,
            domain=domain_tag,
            current_rye=rye_value,
            delta_r=delta_r,
            energy_e=energy_e,
            equilibrium_info=equilibrium_info,
            issue_code_counts_before=issue_code_counts_before,
            issue_code_counts_after=issue_code_counts_after,
            hypotheses=hypotheses,
            citations=citations,
        )

        # Optional stability kernel snapshot
        stability_snapshot: Optional[Dict[str, Any]] = None
        if self.stability_kernel is not None:
            try:
                stability_snapshot = self.stability_kernel.update_from_cycle(
                    rye=rye_value,
                    delta_r=delta_r,
                    energy_e=energy_e,
                    equilibrium_info=equilibrium_info,
                    meta={
                        "goal": goal,
                        "domain": domain_tag,
                        "cycle_index": cycle_index,
                        "role": role,
                        "run_id": run_id,
                        "agent_id": agent_id,
                    },
                )
            except Exception:
                stability_snapshot = None

        # Optional discovery manager snapshot
        discovery_snapshot: Optional[Dict[str, Any]] = None
        if self.discovery_manager is not None:
            try:
                discovery_snapshot = self.discovery_manager.update_from_cycle(
                    goal=goal,
                    domain=domain_tag,
                    rye=rye_value,
                    delta_r=delta_r,
                    energy_e=energy_e,
                    breakthrough=breakthrough_info,
                    hypotheses=hypotheses,
                    citations=citations,
                    cycle_index=cycle_index,
                    run_id=run_id,
                )
            except Exception:
                discovery_snapshot = None

        # Short structured view - tuned for Ultra mode and UI
        short_view = {
            "cycle": cycle_index,
            "goal": goal,
            "role": role,
            "domain": domain_tag,
            "stage": stage_tag,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "run_id": run_id,
            "agent_id": agent_id,
            "ultra_speed": self.ultra_speed,
            "maintenance_mode": maintenance_mode,
            "issues_before": issues,
            "issues_after": issues_after,
            "repairs": [a.get("description", "") for a in repair_actions][:5],
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
            "equilibrium_label": equilibrium_info.get("equilibrium_label"),
            "breakthrough_score": breakthrough_info.get("breakthrough_score"),
            "stability_index": (stability_snapshot or {}).get("stability_index"),
            "recovery_momentum": (stability_snapshot or {}).get(
                "recovery_momentum"
            ),
            "discovery_tier": (discovery_snapshot or {}).get("tier"),
        }

        # Meta controller friendly signals
        meta_signals = {
            "rye": rye_value,
            "delta_R": delta_r,
            "energy_E": energy_e,
            "equilibrium_label": equilibrium_info.get("equilibrium_label"),
            "equilibrium_score": equilibrium_info.get("equilibrium_score"),
            "oscillation_score": equilibrium_info.get("oscillation_score"),
            "open_questions": issue_code_counts_after.get("question_mark", 0),
            "todo_items": issue_code_counts_after.get("todo_item", 0),
            "contradictions": issue_code_counts_after.get("contradiction", 0),
            "sources_used": stats.get("sources_used", 0),
            "stage": stage_tag,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "run_id": run_id,
            "agent_id": agent_id,
            "stability": stability_snapshot,
            "discovery": discovery_snapshot,
        }

        # Machine facing log
        cycle_summary: Dict[str, Any] = {
            "cycle": cycle_index,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "goal": goal,
            "role": role,
            "domain": domain_tag,
            "stage": stage_tag,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "run_id": run_id,
            "agent_id": agent_id,
            "tgrm_level": self.tgrm_level,
            "ultra_speed": self.ultra_speed,
            # Issues before and after
            "issues_before": issue_descriptions,
            "issues_after": issues_after,
            "issue_codes_before": issues,
            "issue_code_counts_before": issue_code_counts_before,
            "issue_code_counts_after": issue_code_counts_after,
            "domain_issue_flags_before": domain_issue_flags_before,
            "domain_issue_flags_after": domain_issue_flags_after,
            "has_open_questions_before": has_questions_before,
            "has_open_questions_after": has_questions_after,
            "has_todos_before": has_todos_before,
            "has_todos_after": has_todos_after,
            "has_contradictions_before": has_contradictions_before,
            "has_contradictions_after": has_contradictions_after,
            # Actions and artifacts
            "repairs_applied": repair_actions,
            "notes_added": notes_added,
            "citations": citations,
            "hypotheses": hypotheses,
            "candidate_hypotheses": candidate_hypotheses,
            "candidate_interventions": candidate_interventions,
            # Raw stats and metrics
            "stats": stats,
            "delta_R": delta_r,
            "delta_R_components": delta_r_components,
            "energy_E": energy_e,
            "Energy": energy_e,  # mirror CoreAgent expectations
            "RYE": rye_value,
            # RYE gradient and equilibrium and breakthrough
            "equilibrium": equilibrium_info,
            "breakthrough": breakthrough_info,
            # Stability kernel and discovery manager snapshots
            "stability": stability_snapshot,
            "discovery": discovery_snapshot,
            # Tool usage details for this cycle
            "tool_usage": {
                "web_calls": tool_usage.web_calls,
                "browser_actions": tool_usage.browser_actions,
                "code_execs": getattr(tool_usage, "code_execs", 0),
                "sql_queries": getattr(tool_usage, "sql_queries", 0),
                "data_loads": tool_usage.data_loads,
                "approx_tokens": tool_usage.approx_tokens,
            },
            # Long run context snapshot
            "avg_rye_for_goal_before_cycle": avg_rye,
            "total_cycles_for_goal_before_cycle": total_cycles_for_goal,
            "maintenance_mode": maintenance_mode,
            # Optional biomarker snapshot for longevity style goals
            "biomarker_snapshot": biomarker_snapshot,
            # Strict pipeline and short view
            "phases": phases,
            "short_view": short_view,
            "meta_signals": meta_signals,
            # Swarm profile metadata (if any)
            "swarm_profile": swarm_profile or {},
        }

        # Replay buffer logging and metadata tagging for longevity aware loops
        replay_item_ids = self._log_replay_candidate(cycle_summary, replay_buffer)
        self._tag_cycle_metadata(
            cycle_summary,
            hallmark=hallmark,
            subgoal=subgoal,
            stage=stage_tag,
            curriculum_state=curriculum_state,
            replay_item_ids=replay_item_ids,
        )

        # Log cycle into memory
        self.memory_store.log_cycle(cycle_summary)

        # Human readable summary
        human_summary = {
            "cycle": cycle_index,
            "role": role,
            "domain": domain_tag,
            "stage": stage_tag,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "run_id": run_id,
            "agent_id": agent_id,
            "goal": goal,
            "issues_before": issue_descriptions,
            "issues_after": issues_after,
            "issue_codes_before": issues,
            "issue_codes_after": issues_after,
            "issue_code_counts_before": issue_code_counts_before,
            "issue_code_counts_after": issue_code_counts_after,
            "repairs": [a.get("description", "") for a in repair_actions],
            "delta_R": delta_r,
            "energy_E": energy_e,
            "Energy": energy_e,  # mirror CoreAgent discovery expectations
            "RYE": rye_value,
            "delta_R_components": delta_r_components,
            "equilibrium": equilibrium_info,
            "breakthrough": breakthrough_info,
            "stability": stability_snapshot,
            "discovery": discovery_snapshot,
            "notes_added": notes_added,
            "citations": citations,
            "hypotheses": hypotheses,
            "candidate_hypotheses": candidate_hypotheses,
            "candidate_interventions": candidate_interventions,
            "maintenance_mode": maintenance_mode,
            "tgrm_level": self.tgrm_level,
            "tool_usage": cycle_summary["tool_usage"],
            "short_view": short_view,
            "meta_signals": meta_signals,
            "phases": phases,
            "curriculum_state": curriculum_state,
            "replay_item_ids": replay_item_ids,
            "swarm_profile": swarm_profile or {},
        }

        # Expose stats and citations at the top level so CoreAgent and the UI
        # can see them without digging into the machine log.
        return {
            "summary": human_summary,
            "log": cycle_summary,
            "tool_stats": stats,
            "citations": citations,
        }

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
            - Placeholder for per domain diagnostic stats.
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
            - reacts to low citation density.

        Level 3:
            - adds domain aware issues for longevity and math.
        """
        issues: List[str] = []
        descriptions: List[str] = []
        notes = status_report.get("notes", [])

        # Core generic issues
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
                    descriptions.append(
                        "TODO item detected in notes; missing information."
                    )
                if "CONTRADICTION" in content:
                    issues.append("contradiction")
                    descriptions.append(
                        "Marked contradiction in notes; needs resolution."
                    )

        # Level 2: citation level diagnostics
        if self.tgrm_level >= 2 and notes:
            citation_markers = status_report.get("approx_citation_markers", 0)
            if citation_markers < len(notes):
                issues.append("under_cited")
                descriptions.append(
                    "Evidence base may be thin: fewer citation markers than notes; "
                    "prioritize finding stronger primary sources."
                )

        # Level 3: domain aware issues
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
        """Add extra issue types for longevity or anti aging research."""
        text = "\n".join(n.get("content", "") for n in notes)

        if all(
            kw.lower() not in text.lower()
            for kw in [
                "biomarker",
                "blood",
                "lab value",
                "marker",
                "hdl",
                "ldl",
                "triglyceride",
            ]
        ):
            issues.append("missing_biomarkers")
            descriptions.append(
                "Longevity notes lack explicit biomarker discussion; "
                "identify concrete measurable markers and how interventions affect them."
            )

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
        """Add extra issue types for math or theory research."""
        text = "\n".join(n.get("content", "") for n in notes)

        has_definition = "definition" in text.lower()
        has_theorem = "theorem" in text.lower() or "lemma" in text.lower()
        if not (has_definition and has_theorem):
            issues.append("missing_formalism")
            descriptions.append(
                "Mathematical formalism is incomplete; add explicit definitions and at least one theorem or lemma."
            )

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
                "relate Reparodynamics to known stability, control, or information theories."
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
        tool_usage: Optional[ToolUsage] = None,
    ):
        """Apply repairs to address detected issues."""
        repair_actions: List[Dict[str, str]] = []
        notes_added: List[str] = []
        citations: List[Dict[str, Any]] = []

        # Stats for RYE energy and improvement
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            # extra tool related stats
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
        }

        # Base cap per cycle
        if maintenance_mode:
            base_max_issues = 2
        else:
            base_max_issues = 5

        # Ultra mode pushes a bit harder before maintenance
        if self.ultra_speed and not maintenance_mode:
            base_max_issues += 2

        # Role based adjustment
        role_lower = (role or "agent").lower()
        if role_lower in {"researcher", "explorer"}:
            max_issues = base_max_issues + 2
        elif role_lower in {"critic", "planner"}:
            max_issues = max(1, base_max_issues - 2)
        elif role_lower in {"synthesizer", "integrator"}:
            max_issues = max(1, base_max_issues - 3)
        else:
            max_issues = base_max_issues

        issues_to_handle = issues[:max_issues]
        descriptions_to_handle = descriptions[:max_issues]

        # Helper to register citations into MemoryStore
        def _log_citations(cites: List[Dict[str, Any]]) -> None:
            for c in cites:
                try:
                    self.memory_store.add_citation(goal, c)
                except Exception:
                    pass

        for issue, desc in zip(issues_to_handle, descriptions_to_handle):
            if issue == "no_notes":
                note_text, new_cites, issue_stats = self._initial_research(
                    goal=goal,
                    role=role,
                    source_controls=source_controls,
                    pdf_bytes=pdf_bytes,
                    domain=domain,
                    maintenance_mode=maintenance_mode,
                    tool_usage=tool_usage,
                )
                try:
                    self.memory_store.add_note(goal, note_text, role=role)
                except Exception:
                    pass
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Performed initial multi source research for '{goal}'",
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)

                for k, v in issue_stats.items():
                    stats[k] = stats.get(k, 0) + v

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

            elif issue in {"question_mark", "todo_item"}:
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
                    tool_usage=tool_usage,
                )

                try:
                    self.memory_store.add_note(goal, note_text, role=role)
                except Exception:
                    pass
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

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

            elif issue == "contradiction":
                note = (
                    f"[{role}] Reminder: contradiction flagged in notes. "
                    "Future cycles should prioritise resolving this with additional sources."
                )
                try:
                    self.memory_store.add_note(goal, note, role=role)
                except Exception:
                    pass
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Acknowledged contradiction; queued for deeper resolution.",
                    }
                )
                notes_added.append(note)
                stats["contradictions_resolved"] += 0
                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note)

            elif issue == "under_cited":
                note_text, new_cites, issue_stats = self._strengthen_citations(
                    goal=goal,
                    role=role,
                    source_controls=source_controls,
                    domain=domain,
                    tool_usage=tool_usage,
                )
                try:
                    self.memory_store.add_note(goal, note_text, role=role)
                except Exception:
                    pass
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
                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

            elif issue in {"missing_biomarkers", "missing_mechanisms"}:
                note_text, new_cites, issue_stats = self._domain_gap_research(
                    goal=goal,
                    role=role,
                    issue=issue,
                    description=desc,
                    source_controls=source_controls,
                    domain=domain,
                    tool_usage=tool_usage,
                )
                try:
                    self.memory_store.add_note(goal, note_text, role=role)
                except Exception:
                    pass
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
                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

            elif issue in {"missing_formalism", "missing_connections"}:
                note_text, new_cites, issue_stats = self._domain_gap_research(
                    goal=goal,
                    role=role,
                    issue=issue,
                    description=desc,
                    source_controls=source_controls,
                    domain=domain,
                    tool_usage=tool_usage,
                )
                try:
                    self.memory_store.add_note(goal, note_text, role=role)
                except Exception:
                    pass
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
                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

            else:
                note = (
                    f"[{role}] Encountered issue '{issue}' with description: {desc}. "
                    "This issue type is not yet fully handled; marking as TODO for future cycles."
                )
                try:
                    self.memory_store.add_note(goal, note, role=role)
                except Exception:
                    pass
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Recorded unhandled issue type for future logic.",
                    }
                )
                notes_added.append(note)
                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note)

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
    def _normalise_source_controls(
        self, source_controls: Optional[Dict[str, bool]]
    ) -> Dict[str, bool]:
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
        """Hybrid mode selection of web search level and size.

        purpose in {"initial", "targeted", "strengthen", "gap_repair"}.
        """
        role_lower = (role or "agent").lower()
        dom = (domain or "general").lower()

        if dom in {"longevity", "biology", "medicine", "health"}:
            topic = "science"
        elif dom in {"math", "physics", "chemistry"}:
            topic = "science"
        else:
            topic = "general"

        base_level = max(1, min(self.tgrm_level, 3))
        level = base_level

        if maintenance_mode and base_level > 1:
            level = base_level - 1

        if purpose in {"initial", "gap_repair", "strengthen"}:
            if (
                not maintenance_mode
                and base_level == 3
                and role_lower in {"researcher", "explorer"}
            ):
                level = 3
        elif purpose == "targeted":
            if (
                not maintenance_mode
                and base_level >= 2
                and role_lower in {"researcher", "explorer"}
            ):
                level = min(3, base_level)
            elif role_lower in {"critic", "planner", "synthesizer", "integrator"}:
                level = max(1, min(base_level, 2))

        if purpose == "initial":
            base_max = 5 if not maintenance_mode else 3
        elif purpose == "targeted":
            base_max = 6 if not maintenance_mode else 4
        elif purpose in {"strengthen", "gap_repair"}:
            base_max = 8 if not maintenance_mode else 4
        else:
            base_max = 5

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
        tool_usage: Optional[ToolUsage] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """Perform the initial multi source research when there are no notes."""
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Initial research summary for goal:")
        note_lines.append(goal)
        note_lines.append("")

        first_url: Optional[str] = None

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
            if tool_usage is not None:
                tool_usage.web_calls += 1

            web_summary = self.web_tool.summarize_results(web_results)
            web_cites_raw = self.web_tool.to_citations(web_results)
            web_cites = self._tag_citations(
                web_cites_raw,
                goal=goal,
                query=goal,
                channel="web",
                phase="initial",
            )
            citations.extend(web_cites)

            if tool_usage is not None:
                tool_usage.approx_tokens += self._estimate_tokens(web_summary)

            note_lines.append("Web summary (Tavily or equivalent):")
            note_lines.append(web_summary)
            note_lines.append("")
            note_lines.append("Web sources:")
            for c in web_cites:
                url = c.get("url", "")
                note_lines.append(f"- {c.get('title', '')} ({url})")
                if not first_url and url:
                    first_url = url
            note_lines.append("")

        # Optional headless browser deep dive on the top URL (only if browser exists)
        if first_url and hasattr(self.tools, "browser"):
            try:
                browser = self.tools.browser  # type: ignore[attr-defined]
                browser_result = browser.fetch_page(first_url)
                stats["browser_actions"] += 1
                if tool_usage is not None:
                    tool_usage.browser_actions += 1
                    tool_usage.approx_tokens += self._estimate_tokens(
                        getattr(browser_result, "text_snippet", "") or ""
                    )

                note_lines.append(
                    f"Browser deep dive snippet from: {browser_result.url}"
                )
                if getattr(browser_result, "error", None):
                    note_lines.append(f"(Browser error: {browser_result.error})")
                else:
                    snippet = getattr(browser_result, "text_snippet", "") or ""
                    note_lines.append(snippet[:2000])
                note_lines.append("")
            except Exception:
                # Browser failures must not break the cycle
                pass

        # PubMed search
        if source_controls.get("pubmed", False):
            pubmed_results = self.pubmed_tool.search(goal, max_results=5)
            stats["pubmed_calls"] += 1
            pubmed_cites = self._tag_citations(
                pubmed_results,
                goal=goal,
                query=goal,
                channel="pubmed",
                phase="initial",
            )
            citations.extend(pubmed_cites)

            if tool_usage is not None:
                titles = " ".join(r.get("title", "") or "" for r in pubmed_results)
                tool_usage.approx_tokens += self._estimate_tokens(titles)

            note_lines.append("PubMed sources:")
            for r in pubmed_cites:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        # Semantic Scholar search
        if source_controls.get("semantic", False):
            sem_results = self.semantic_tool.search(goal, max_results=5)
            stats["semantic_calls"] += 1
            sem_cites = self._tag_citations(
                sem_results,
                goal=goal,
                query=goal,
                channel="semantic",
                phase="initial",
            )
            citations.extend(sem_cites)

            if tool_usage is not None:
                titles = " ".join(r.get("title", "") or "" for r in sem_results)
                tool_usage.approx_tokens += self._estimate_tokens(titles)

            note_lines.append("Semantic Scholar sources:")
            for r in sem_cites:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        # Optional PDF ingestion if provided
        if source_controls.get("pdf", False) and pdf_bytes:
            try:
                if hasattr(self.paper_tool, "ingest_bytes"):
                    text = self.paper_tool.ingest_bytes(pdf_bytes)  # type: ignore[attr-defined]
                    stats["pdf_ingestions"] += 1
                    summary = self.paper_tool.summarise(text)
                    note_lines.append("Attached PDF summary:")
                    note_lines.append(summary)
                    note_lines.append("")
                    if tool_usage is not None:
                        tool_usage.approx_tokens += self._estimate_tokens(summary)
            except Exception:
                pass

        note_text = "\n".join(note_lines)
        return note_text, citations, stats

    def _extract_questions(self, goal: str, issue_type: str) -> List[str]:
        """Extract concrete question or TODO lines from stored notes."""
        notes = self.memory_store.get_notes(goal)
        candidates: List[str] = []

        for note in notes:
            content = note.get("content", "")
            for line in content.splitlines():
                line_strip = line.strip()
                if issue_type == "question_mark" and "?" in line_strip:
                    if len(line_strip) > 10:
                        candidates.append(line_strip)
                elif issue_type == "todo_item" and (
                    "TODO" in line_strip or "todo" in line_strip
                ):
                    if len(line_strip) > 10:
                        candidates.append(line_strip)

        seen_local = set()
        unique_questions: List[str] = []
        for q in candidates:
            if q not in seen_local:
                seen_local.add(q)
                unique_questions.append(q)

        fresh_questions: List[str] = []
        for q in unique_questions:
            if q not in self._seen_questions:
                self._seen_questions.add(q)
                fresh_questions.append(q)

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
        tool_usage: Optional[ToolUsage] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """Perform focused multi source research on open questions or TODOs."""
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
        }

        if questions is None:
            questions = [f"{goal} - focus on: {issue_description}"]
        elif len(questions) == 0:
            note_lines: List[str] = []
            note_lines.append(
                f"[{role}] Maintenance pass on open items ({issue}) for goal:"
            )
            note_lines.append(goal)
            note_lines.append("")
            note_lines.append(
                "No new unresolved questions were found beyond what has already been researched. "
                "This cycle performs a light consolidation of existing knowledge without new web or paper calls."
            )
            note_text = "\n".join(note_lines)
            return note_text, citations, stats

        note_lines: List[str] = []
        note_lines.append(
            f"[{role}] Targeted research on open items ({issue}) for goal:"
        )
        note_lines.append(goal)
        note_lines.append("")
        note_lines.append("Questions or TODOs considered:")
        for q in questions:
            note_lines.append(f"- {q}")
        note_lines.append("")

        for q in questions:
            note_lines.append("### Question focused search:")
            note_lines.append(q)
            note_lines.append("")

            if source_controls.get("web", True):
                web_params = self._select_web_search_params(
                    role=role,
                    maintenance_mode=maintenance_mode,
                    domain=domain,
                    purpose="targeted",
                )
                web_results = self.web_tool.search(q, **web_params)
                stats["web_calls"] += 1
                if tool_usage is not None:
                    tool_usage.web_calls += 1

                web_summary = self.web_tool.summarize_results(web_results)
                web_cites_raw = self.web_tool.to_citations(web_results)
                web_cites = self._tag_citations(
                    web_cites_raw,
                    goal=goal,
                    query=q,
                    channel="web",
                    phase="targeted",
                )
                citations.extend(web_cites)

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(web_summary)

                note_lines.append("Web summary (Tavily or equivalent):")
                note_lines.append(web_summary)
                note_lines.append("Web sources:")
                for c in web_cites:
                    note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
                note_lines.append("")

            if source_controls.get("pubmed", False):
                pubmed_results = self.pubmed_tool.search(q, max_results=5)
                stats["pubmed_calls"] += 1
                pubmed_cites = self._tag_citations(
                    pubmed_results,
                    goal=goal,
                    query=q,
                    channel="pubmed",
                    phase="targeted",
                )
                citations.extend(pubmed_cites)

                if tool_usage is not None:
                    titles = " ".join(
                        r.get("title", "") or "" for r in pubmed_results
                    )
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("PubMed sources:")
                for r in pubmed_cites:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

            if source_controls.get("semantic", False):
                sem_results = self.semantic_tool.search(q, max_results=5)
                stats["semantic_calls"] += 1
                sem_cites = self._tag_citations(
                    sem_results,
                    goal=goal,
                    query=q,
                    channel="semantic",
                    phase="targeted",
                )
                citations.extend(sem_cites)

                if tool_usage is not None:
                    titles = " ".join(
                        r.get("title", "") or "" for r in sem_results
                    )
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("Semantic Scholar sources:")
                for r in sem_cites:
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
        tool_usage: Optional[ToolUsage] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """Specialized repair step for 'under_cited' issues."""
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Strengthening citations for goal:")
        note_lines.append(goal)
        note_lines.append("")

        query = f"{goal} primary sources randomized trial benchmark formal paper"
        note_lines.append(f"Search query for stronger evidence: {query}")
        note_lines.append("")

        if source_controls.get("web", True):
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=False,
                domain=domain,
                purpose="strengthen",
            )
            web_results = self.web_tool.search(query, **web_params)
            stats["web_calls"] += 1
            if tool_usage is not None:
                tool_usage.web_calls += 1

            web_summary = self.web_tool.summarize_results(web_results)
            web_cites_raw = self.web_tool.to_citations(web_results)
            web_cites = self._tag_citations(
                web_cites_raw,
                goal=goal,
                query=query,
                channel="web",
                phase="strengthen",
            )
            citations.extend(web_cites)

            if tool_usage is not None:
                tool_usage.approx_tokens += self._estimate_tokens(web_summary)

            note_lines.append("Web summary (stronger evidence focus):")
            note_lines.append(web_summary)
            note_lines.append("Web sources:")
            for c in web_cites:
                note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
            note_lines.append("")

        if source_controls.get("pubmed", False):
            pubmed_results = self.pubmed_tool.search(query, max_results=10)
            stats["pubmed_calls"] += 1
            pubmed_cites = self._tag_citations(
                pubmed_results,
                goal=goal,
                query=query,
                channel="pubmed",
                phase="strengthen",
            )
            citations.extend(pubmed_cites)

            if tool_usage is not None:
                titles = " ".join(r.get("title", "") or "" for r in pubmed_results)
                tool_usage.approx_tokens += self._estimate_tokens(titles)

            note_lines.append("PubMed sources (stronger evidence):")
            for r in pubmed_cites:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        if source_controls.get("semantic", False):
            sem_results = self.semantic_tool.search(query, max_results=10)
            stats["semantic_calls"] += 1
            sem_cites = self._tag_citations(
                sem_results,
                goal=goal,
                query=query,
                channel="semantic",
                phase="strengthen",
            )
            citations.extend(sem_cites)

            if tool_usage is not None:
                titles = " ".join(r.get("title", "") or "" for r in sem_results)
                tool_usage.approx_tokens += self._estimate_tokens(titles)

            note_lines.append("Semantic Scholar sources (stronger evidence):")
            for r in sem_cites:
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
        tool_usage: Optional[ToolUsage] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """Generic helper to fill domain specific gaps."""
        dom = (domain or "general").lower()
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Domain specific gap repair for goal:")
        note_lines.append(goal)
        note_lines.append("")
        note_lines.append(f"Issue: {issue}")
        note_lines.append(description)
        note_lines.append("")

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
                    "formal definition theorem stability framework similar to "
                    "reparodynamics and RYE"
                )
            elif issue == "missing_connections":
                query = (
                    "connections between stability theory Lyapunov control "
                    "Markov processes and repair dynamics"
                )
            else:
                query = f"{goal} mathematical stability formalization"
        else:
            query = f"{goal} {issue} {description}"

        note_lines.append(f"Focused query used for gap repair: {query}")
        note_lines.append("")

        if source_controls.get("web", True):
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=False,
                domain=domain,
                purpose="gap_repair",
            )
            web_results = self.web_tool.search(query, **web_params)
            stats["web_calls"] += 1
            if tool_usage is not None:
                tool_usage.web_calls += 1

            web_summary = self.web_tool.summarize_results(web_results)
            web_cites_raw = self.web_tool.to_citations(web_results)
            web_cites = self._tag_citations(
                web_cites_raw,
                goal=goal,
                query=query,
                channel="web",
                phase="gap_repair",
            )
            citations.extend(web_cites)

            if tool_usage is not None:
                tool_usage.approx_tokens += self._estimate_tokens(web_summary)

            note_lines.append("Web summary (gap repair):")
            note_lines.append(web_summary)
            note_lines.append("Web sources:")
            for c in web_cites:
                note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
            note_lines.append("")

        if dom == "longevity" and source_controls.get("pubmed", False):
            pubmed_results = self.pubmed_tool.search(query, max_results=10)
            stats["pubmed_calls"] += 1
            pubmed_cites = self._tag_citations(
                pubmed_results,
                goal=goal,
                query=query,
                channel="pubmed",
                phase="gap_repair",
            )
            citations.extend(pubmed_cites)

            if tool_usage is not None:
                titles = " ".join(r.get("title", "") or "" for r in pubmed_results)
                tool_usage.approx_tokens += self._estimate_tokens(titles)

            note_lines.append("PubMed sources (gap repair):")
            for r in pubmed_cites:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        if source_controls.get("semantic", False):
            sem_results = self.semantic_tool.search(query, max_results=10)
            stats["semantic_calls"] += 1
            sem_cites = self._tag_citations(
                sem_results,
                goal=goal,
                query=query,
                channel="semantic",
                phase="gap_repair",
            )
            citations.extend(sem_cites)

            if tool_usage is not None:
                titles = " ".join(r.get("title", "") or "" for r in sem_results)
                tool_usage.approx_tokens += self._estimate_tokens(titles)

            note_lines.append("Semantic Scholar sources (gap repair):")
            for r in sem_cites:
                note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
            note_lines.append("")

        note_text = "\n".join(note_lines)
        return note_text, citations, stats

    # ------------------------------------------------------------------
    # Longevity replay helpers
    # ------------------------------------------------------------------
    def _tag_cycle_metadata(
        self,
        cycle_log: Dict[str, Any],
        hallmark: Optional[str],
        subgoal: Optional[str],
        stage: Optional[str],
        curriculum_state: Optional[Dict[str, Any]] = None,
        replay_item_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Attach hallmark, stage, curriculum hints, and replay ids to the log."""
        if stage:
            cycle_log["stage"] = stage
        if hallmark:
            cycle_log["hallmark"] = hallmark
        if subgoal:
            cycle_log["subgoal"] = subgoal

        if curriculum_state:
            cycle_log["curriculum_state"] = dict(curriculum_state)
            phase = curriculum_state.get("phase") or curriculum_state.get("name")
            if phase:
                cycle_log.setdefault("tags", [])
                if f"curriculum:{phase}" not in cycle_log["tags"]:
                    cycle_log["tags"].append(f"curriculum:{phase}")

        if replay_item_ids:
            cycle_log["replay_item_ids"] = list(replay_item_ids)

        return cycle_log

    def _log_replay_candidate(
        self,
        cycle_log: Dict[str, Any],
        replay_buffer: Optional[Any],
    ) -> List[str]:
        """Push top hypotheses and patterns from this cycle into a replay buffer.

        The replay_buffer can be:
            - an object with .add_item(item: dict) -> Any
            - a callable that accepts a single item dict

        Returns:
            List of item ids that were created for this cycle.
        """
        item_ids: List[str] = []
        if replay_buffer is None:
            return item_ids

        hypotheses = cycle_log.get("hypotheses") or []
        citations = cycle_log.get("citations") or []
        biomarker_pattern = cycle_log.get("biomarker_snapshot")
        rye_score = cycle_log.get("RYE")
        energy_cost = cycle_log.get("energy_E")
        hallmark = cycle_log.get("hallmark")
        stage = cycle_log.get("stage")
        run_id = cycle_log.get("run_id")
        timestamp = cycle_log.get("timestamp")
        cycle_idx = cycle_log.get("cycle")

        # Select up to a few best hypotheses by confidence if available
        def hyp_score(h: Dict[str, Any]) -> float:
            val = h.get("confidence")
            try:
                return float(val) if val is not None else 0.0
            except Exception:
                return 0.0

        sorted_hypotheses = sorted(hypotheses, key=hyp_score, reverse=True)
        top_hypotheses = sorted_hypotheses[:5]

        for idx, h in enumerate(top_hypotheses):
            text = (
                h.get("text")
                or h.get("description")
                or h.get("title")
                or ""
            )
            if not text:
                continue

            item_id = f"replay_{cycle_idx}_{idx}"
            item: Dict[str, Any] = {
                "item_id": item_id,
                "hallmark": hallmark,
                "stage": stage,
                "mechanism_chain": text,
                "biomarker_pattern": biomarker_pattern,
                "hypothesis_text": text,
                "rye_score": rye_score,
                "energy_cost": energy_cost,
                "decision": "pending",
                "reason": "auto_logged_from_cycle",
                "source_citations": citations,
                "tags": h.get("tags", []),
                "created_at": timestamp,
                "run_id": run_id,
                "cycle_index": cycle_idx,
            }

            try:
                if hasattr(replay_buffer, "add_item"):
                    replay_buffer.add_item(item)  # type: ignore[attr-defined]
                elif callable(replay_buffer):
                    replay_buffer(item)
                # Only append if we did not raise
                item_ids.append(item_id)
            except Exception:
                # Replay logging must never break the main loop
                continue

        return item_ids

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
        """Lightweight extractor for candidate interventions or treatments."""
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
