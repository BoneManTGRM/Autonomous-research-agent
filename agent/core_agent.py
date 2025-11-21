"""Core agent implementation.

This module defines the `CoreAgent` class which orchestrates the
TGRM loop for running research cycles. It interacts with the memory
store and delegates work to the `TGRMLoop` for each cycle.

Reparodynamics view:
    The CoreAgent is the "system" whose job is to maintain and improve
    the quality of its internal knowledge over time.

TGRM loop (implemented in TGRMLoop):
    - Test   : evaluate current notes / state
    - Detect : find gaps, TODOs, unanswered questions or contradictions
    - Repair : perform targeted actions (web, PubMed, PDFs, biomarkers)
    - Verify : re-test, compute ΔR and RYE, and log the cycle

RYE (Repair Yield per Energy):
    For each cycle, TGRMLoop computes delta_R / E (improvement divided
    by effort). CoreAgent simply orchestrates cycles and exposes higher
    level methods like multi-agent runs and continuous mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .memory_store import MemoryStore
from .tgrm_loop import TGRMLoop


class CoreAgent:
    """High-level controller for the autonomous research agent.

    This class is intentionally thin: it wires UI / CLI controls
    (multi-agent, continuous mode, source preferences, PDF uploads)
    into the lower-level TGRMLoop which performs the actual
    Reparodynamic TGRM cycles and RYE computation.
    """

    def __init__(self, memory_store: MemoryStore, config: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # Underlying TGRM loop that actually performs Test–Detect–Repair–Verify
        self.tgrm_loop = TGRMLoop(memory_store, self.config)

        # Source controls are set from the UI (Streamlit) at runtime.
        # Example:
        #   {
        #       "pubmed": True,
        #       "semantic": False,
        #       "pdf": True,
        #       "biomarkers": False,
        #   }
        self.source_controls: Dict[str, bool] = {}

        # Optional attached PDF bytes (from an uploaded file in the UI)
        self._attached_pdf_bytes: Optional[bytes] = None

    # ------------------------------------------------------------------
    # Public API used by Streamlit and CLI
    # ------------------------------------------------------------------
    def attach_pdf(self, uploaded_file: Any) -> None:
        """Attach a PDF (uploaded via Streamlit) for the next cycles.

        The TGRM loop can then ingest this PDF as part of its REPAIR
        actions. If `uploaded_file` is None, this does nothing.
        """
        if uploaded_file is None:
            return
        try:
            self._attached_pdf_bytes = uploaded_file.read()
        except Exception:
            # Fail silently; core logic should still run without PDF
            self._attached_pdf_bytes = None

    def run_cycle(
        self,
        goal: str,
        cycle_index: int,
        role: str = "agent",
    ) -> Dict[str, Any]:
        """Run a single cycle of research using the TGRM loop.

        Args:
            goal: Research goal for this cycle.
            cycle_index: Global cycle index (used for logging / history).
            role: Logical role for this cycle: "agent", "researcher",
                  "critic", etc. This is recorded in the logs and
                  allows multi-agent setups (researcher + critic).

        Returns:
            Dict[str, Any]: Dictionary containing the human-facing
            summary and the raw log of the cycle.
        """
        # Forward source preferences and optional PDF into the loop.
        # We use a TypeError-safe wrapper so this stays compatible
        # even if TGRMLoop still has the older (goal, cycle_index)
        # signature in your repo while you are upgrading.
        try:
            return self.tgrm_loop.run_cycle(
                goal=goal,
                cycle_index=cycle_index,
                role=role,
                source_controls=self.source_controls,
                pdf_bytes=self._attached_pdf_bytes,
            )
        except TypeError:
            # Backwards compatibility: older TGRMLoop.run_cycle
            # that only accepts (goal, cycle_index).
            return self.tgrm_loop.run_cycle(goal, cycle_index)

    def run_continuous(
        self,
        goal: str,
        max_cycles: int = 100,
        stop_rye: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run multiple TGRM cycles in a row ("continuous mode").

        This is the long-running Reparodynamic mode where the agent
        repeatedly applies Test–Detect–Repair–Verify to gradually
        improve its internal state under a fixed goal.

        Args:
            goal: Research goal to pursue over many cycles.
            max_cycles: Hard cap on the number of cycles to prevent
                        infinite loops in hosted environments.
            stop_rye: Optional RYE threshold. If set, the loop stops
                      early once average RYE over recent cycles
                      falls below this value (indicating diminishing
                      returns on additional repair actions).

        Returns:
            List[Dict[str, Any]]: List of human-facing summaries for
            each completed cycle.
        """
        summaries: List[Dict[str, Any]] = []
        recent_rye: List[float] = []

        # Start counting cycles from the existing history length
        history = self.memory_store.get_cycle_history()
        start_index = len(history)

        for i in range(max_cycles):
            ci = start_index + i
            result = self.run_cycle(goal=goal, cycle_index=ci, role="agent")
            summary = result.get("summary", {})
            summaries.append(summary)

            rye_val = summary.get("RYE")
            if isinstance(rye_val, (int, float)):
                recent_rye.append(float(rye_val))
                # Keep a sliding window of the last N cycles
                if len(recent_rye) > 10:
                    recent_rye.pop(0)

            # Optional stopping condition based on RYE efficiency
            if stop_rye is not None and recent_rye:
                avg_rye = sum(recent_rye) / len(recent_rye)
                if avg_rye < stop_rye:
                    # Reparodynamic interpretation:
                    #   The system's marginal repair yield per energy
                    #   has fallen below the target; further cycles are
                    #   not energetically efficient, so we stop.
                    break

        return summaries
