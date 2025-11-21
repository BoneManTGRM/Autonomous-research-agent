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
    - Verify : re-test, compute delta_R and RYE, and log the cycle

RYE (Repair Yield per Energy):
    For each cycle, TGRMLoop computes delta_R / E (improvement divided
    by effort). CoreAgent simply orchestrates cycles and exposes higher
    level methods like multi-agent runs and continuous mode.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .memory_store import MemoryStore
from .tgrm_loop import TGRMLoop


class CoreAgent:
    """High-level controller for the autonomous research agent.

    This class is intentionally thin: it wires UI or CLI controls
    (multi-agent, continuous mode, source preferences, PDF uploads)
    into the lower-level TGRMLoop which performs the actual
    reparodynamic TGRM cycles and RYE computation.
    """

    def __init__(self, memory_store: MemoryStore, config: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # Underlying TGRM loop that actually performs Test, Detect, Repair, Verify
        self.tgrm_loop = TGRMLoop(memory_store, self.config)

        # Source controls can be set by the UI at runtime.
        # Example:
        #   {
        #       "web": True,
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
        """Attach a PDF (uploaded via Streamlit) for later cycles.

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
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,  # reserved for future biomarker engine
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single cycle of research using the TGRM loop.

        Args:
            goal:
                Research goal for this cycle.
            cycle_index:
                Global cycle index (used for logging and history).
            role:
                Logical role for this cycle: "agent", "researcher",
                "critic", etc. This is recorded in the logs and
                allows multi-agent setups (researcher + critic).
            source_controls:
                Optional override for which sources to use in this cycle.
                If None, falls back to self.source_controls.
            pdf_bytes:
                Optional PDF bytes for this cycle. If None, falls back
                to any previously attached PDF stored in self._attached_pdf_bytes.
            biomarker_snapshot:
                Placeholder for a future biomarker or lab value payload.
            domain:
                Optional domain tag (e.g. "general", "longevity", "math")
                to label this cycle in the logs.

        Returns:
            Dict[str, Any]: Dictionary containing the human-facing
            summary and the raw log of the cycle.
        """
        # Merge source_controls: explicit argument overrides stored default
        effective_source_controls: Dict[str, bool] = {}
        if self.source_controls:
            effective_source_controls.update(self.source_controls)
        if source_controls:
            effective_source_controls.update(source_controls)

        # Choose PDF bytes: per-call bytes override attached PDF
        effective_pdf_bytes: Optional[bytes] = (
            pdf_bytes if pdf_bytes is not None else self._attached_pdf_bytes
        )

        # Forward into the TGRM loop.
        # Use a TypeError-safe wrapper for compatibility with older signatures.
        try:
            return self.tgrm_loop.run_cycle(
                goal=goal,
                cycle_index=cycle_index,
                role=role,
                source_controls=effective_source_controls,
                pdf_bytes=effective_pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
            )
        except TypeError:
            # Backwards compatibility: older TGRMLoop.run_cycle
            # that only accepts (goal, cycle_index) or (goal, cycle_index, role).
            try:
                return self.tgrm_loop.run_cycle(goal, cycle_index, role)
            except TypeError:
                return self.tgrm_loop.run_cycle(goal, cycle_index)

    def run_continuous(
        self,
        goal: str,
        max_cycles: int = 100,
        stop_rye: Optional[float] = None,
        role: str = "agent",
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        max_minutes: Optional[float] = None,
        forever: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run multiple TGRM cycles in a row ("continuous mode").

        This is the long-running reparodynamic mode where the agent
        repeatedly applies Test, Detect, Repair, Verify to gradually
        improve its internal state under a fixed goal.

        Args:
            goal:
                Research goal to pursue over many cycles.
            max_cycles:
                Hard cap on the number of cycles to prevent infinite runs
                in hosted environments (used when `forever` is False).
            stop_rye:
                Optional RYE threshold. If set, the loop stops
                early once average RYE over recent cycles
                falls below this value (indicating diminishing
                returns on additional repair actions).
            role:
                Logical role for the continuous runner, usually "agent".
            source_controls:
                Optional source configuration applied to every cycle.
            pdf_bytes:
                Optional PDF bytes shared across cycles.
            biomarker_snapshot:
                Optional biomarker or lab value payload shared for
                future biomarker-aware logic.
            domain:
                Optional domain tag (e.g. "general", "longevity", "math")
                forwarded into each cycle.
            max_minutes:
                Optional wall-clock time budget in minutes. If provided,
                the loop will stop once this time has elapsed.
            forever:
                If True, ignore max_cycles and keep running until
                stopped by `max_minutes`, `stop_rye`, or the environment.

        Returns:
            List[Dict[str, Any]]: List of human-facing summaries for
            each completed cycle.
        """
        summaries: List[Dict[str, Any]] = []
        recent_rye: List[float] = []

        # ------------------------------------------------------------------
        # Auto map the "hour presets" from app_streamlit to real time
        # without changing any other files.
        #
        # app_streamlit.py currently does:
        #   CYCLES_PER_HOUR_ESTIMATE = 120
        #   1h  -> max_cycles = 120
        #   8h  -> max_cycles = 960
        #   24h -> max_cycles = 2880
        #   Forever -> max_cycles = 10_000_000
        #
        # Here we detect those magic values and convert them back into
        # wall-clock minutes so that the run actually respects time,
        # not just cycle count.
        # ------------------------------------------------------------------
        if max_minutes is None:
            if max_cycles == 120:
                # 1 hour preset
                max_minutes = 60.0
                max_cycles = 1_000_000
            elif max_cycles == 960:
                # 8 hour preset
                max_minutes = 480.0
                max_cycles = 1_000_000
            elif max_cycles == 2880:
                # 24 hour preset
                max_minutes = 1440.0
                max_cycles = 1_000_000
            elif max_cycles >= 10_000_000:
                # "Forever" preset: let environment or stop_rye decide
                forever = True
                # keep a very high cap as a hard safety
                max_cycles = 10_000_000

        # Start counting cycles from the existing history length
        history = self.memory_store.get_cycle_history()
        start_index = len(history)

        start_time = time.monotonic()
        i = 0

        while True:
            # Cycle-based stopping (unless in explicit forever mode)
            if not forever and i >= max_cycles:
                break

            # Time-based stopping (for 1h / 8h / 24h presets)
            if max_minutes is not None:
                elapsed_min = (time.monotonic() - start_time) / 60.0
                if elapsed_min >= max_minutes:
                    break

            ci = start_index + i
            result = self.run_cycle(
                goal=goal,
                cycle_index=ci,
                role=role,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
            )
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
                    # The system's marginal repair yield per energy
                    # has fallen below the target; further cycles are
                    # not energetically efficient, so we stop.
                    break

            i += 1

        return summaries
