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

Multi-agent extension:
    CoreAgent can now orchestrate multiple logical agent roles
    (researcher, critic, planner, synthesizer, explorer, plus up to
    a total of 32 agents) over the same MemoryStore. This makes it
    possible to run:
        - one cycle per role (multi-agent round), or
        - long continuous runs where each "round" consists of many roles.

    The maximum number of logical agents is capped at 32 for safety,
    and can be configured via config["max_agents"] (1–32).

Runtime profiles:
    CoreAgent understands runtime profiles defined in presets.RUNTIME_PROFILES,
    such as "1_hour", "8_hours", "24_hours", and "forever". These profiles
    provide:
        - estimated_cycles
        - rye_stop_threshold
        - energy_scaling (for future use)
        - report_frequency (for UI reporting)

    When a runtime_profile is selected, CoreAgent will:
        - adjust max_minutes and max_cycles for continuous runs
        - auto-apply a stop_rye threshold where appropriate
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .memory_store import MemoryStore
from .tgrm_loop import TGRMLoop
from .presets import RUNTIME_PROFILES


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

        # Checkpoint file for continuous runs
        checkpoint_default = "logs/continuous_checkpoint.json"
        self.checkpoint_path = Path(self.config.get("checkpoint_file", checkpoint_default))

        # Whether continuous runs should try to auto resume from checkpoint
        self.auto_resume_enabled: bool = bool(self.config.get("auto_resume_enabled", True))

        # ------------------------------------------------------------------
        # Multi-agent configuration (up to 32 logical agents)
        # ------------------------------------------------------------------
        # Hard safety cap for number of logical agents this CoreAgent
        # will coordinate in any single round.
        self.max_agents: int = int(self.config.get("max_agents", 32))
        if self.max_agents < 1:
            self.max_agents = 1
        if self.max_agents > 32:
            self.max_agents = 32

        # Logical "base" roles. Additional generic roles are generated
        # as agent_01, agent_02, ... up to self.max_agents.
        base_roles: List[str] = [
            "researcher",
            "critic",
            "planner",
            "synthesizer",
            "explorer",
        ]

        # Generate a full role list up to max_agents
        generated_roles: List[str] = list(base_roles)
        # Fill remaining slots with generic agent_NN labels
        counter = 1
        while len(generated_roles) < self.max_agents:
            generated_roles.append(f"agent_{counter:02d}")
            counter += 1

        # If config provides an explicit list of roles, use it but
        # enforce the max_agents cap.
        config_roles = self.config.get("agent_roles")
        if isinstance(config_roles, (list, tuple)):
            roles: List[str] = [str(r) for r in config_roles if r]
            if not roles:
                roles = generated_roles
        else:
            roles = generated_roles

        # Enforce safety cap
        self.agent_roles: List[str] = roles[: self.max_agents]

    # ------------------------------------------------------------------
    # Internal helpers for crash proofing
    # ------------------------------------------------------------------
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint from disk if present."""
        try:
            if not self.checkpoint_path.exists():
                return None
            with self.checkpoint_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return data
        except Exception:
            return None

    def _save_checkpoint(self, state: Dict[str, Any]) -> None:
        """Persist checkpoint state to disk for auto resume and watchdog."""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with self.checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            # Checkpoint failures must not crash the agent
            return

    def _clear_checkpoint(self) -> None:
        """Remove checkpoint file when a continuous run completes cleanly."""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
        except Exception:
            return

    # ------------------------------------------------------------------
    # Multi-agent utilities
    # ------------------------------------------------------------------
    def get_agent_roles(self) -> List[str]:
        """Return the configured logical agent roles (capped at max_agents)."""
        return list(self.agent_roles[: self.max_agents])

    def spawn_child_agent(self, extra_config: Optional[Dict[str, Any]] = None) -> "CoreAgent":
        """Create a new CoreAgent that shares the same MemoryStore.

        This is a light-weight way for agents to "create" agents.
        Child agents share the same memory substrate but can have their
        own configuration (for example different prompts, domains, etc.).

        The child will inherit max_agents (capped at 32) unless overridden
        in extra_config["max_agents"].
        """
        cfg = dict(self.config)
        if extra_config:
            cfg.update(extra_config)
        return CoreAgent(memory_store=self.memory_store, config=cfg)

    def run_multi_agent_round(
        self,
        goal: str,
        base_cycle_index: int,
        roles: Optional[Sequence[str]] = None,
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a single "round" of multiple logical agents.

        Example round (5 agents, but can go up to 32):
            1. researcher   – primary gatherer and explainer
            2. critic       – attacks weak points, flags gaps
            3. planner      – proposes next experiments and steps
            4. synthesizer  – condenses findings into coherent story
            5. explorer     – searches for surprising / out-of-distribution angles

        All agents share the same MemoryStore and TGRM engine, but run
        with different `role` labels so you can distinguish their
        contributions in the logs and RYE history.

        The number of roles actually run is capped by self.max_agents
        (default 32).
        """
        if roles is None:
            roles_seq: Sequence[str] = self.agent_roles
        else:
            roles_seq = roles

        # Enforce max agent count for safety
        roles_list: List[str] = [str(r) for r in roles_seq][: self.max_agents]

        summaries: List[Dict[str, Any]] = []

        for idx, role in enumerate(roles_list):
            ci = base_cycle_index + idx
            result = self.run_cycle(
                goal=goal,
                cycle_index=ci,
                role=role,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
            )
            summaries.append(result.get("summary", {}))

        return summaries

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
                Optional domain tag (for example "general", "longevity", "math")
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

    # ------------------------------------------------------------------
    # Single-agent continuous mode
    # ------------------------------------------------------------------
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
        resume_from_checkpoint: bool = True,
        watchdog_interval_minutes: float = 5.0,
        runtime_profile: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run multiple TGRM cycles in a row ("continuous mode") for a single role.

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
                falls below this value.
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
                Optional domain tag forwarded into each cycle.
            max_minutes:
                Optional wall-clock time budget in minutes. If provided,
                the loop stops once this time has elapsed.
            forever:
                If True, ignore max_cycles and keep running until
                stopped by `max_minutes`, `stop_rye`, or the environment.
            resume_from_checkpoint:
                If True, try to detect a previous interrupted continuous run from
                the checkpoint file and adjust the remaining time budget.
            watchdog_interval_minutes:
                How often to update the checkpoint heartbeat at minimum.
            runtime_profile:
                Optional name of a runtime profile from presets.RUNTIME_PROFILES,
                for example "1_hour", "8_hours", "24_hours", or "forever".
                When provided, this can adjust max_minutes, max_cycles,
                and stop_rye, unless the caller has explicitly set them.
        Returns:
            List[Dict[str, Any]]: List of human-facing summaries for
            each completed cycle.
        """
        summaries: List[Dict[str, Any]] = []
        recent_rye: List[float] = []

        # Apply runtime profile hints if requested
        profile_cfg: Optional[Dict[str, Any]] = None
        if runtime_profile:
            profile_cfg = RUNTIME_PROFILES.get(runtime_profile)
        if profile_cfg is not None:
            # Only override if caller did not set a stronger preference
            est_cycles = profile_cfg.get("estimated_cycles")
            profile_stop_rye = profile_cfg.get("rye_stop_threshold")

            # Map profile name to a default time budget if not set
            if max_minutes is None:
                if runtime_profile == "1_hour":
                    max_minutes = 60.0
                elif runtime_profile == "8_hours":
                    max_minutes = 8 * 60.0
                elif runtime_profile == "24_hours":
                    max_minutes = 24 * 60.0
                elif runtime_profile == "forever":
                    forever = True

            # Use profile estimated cycles when max_cycles is still at a small default
            if isinstance(est_cycles, (int, float)) and max_cycles <= 100:
                try:
                    max_cycles = max(1, int(est_cycles))
                except Exception:
                    pass

            # Automatically set stop_rye if not provided
            if stop_rye is None and isinstance(profile_stop_rye, (int, float)):
                stop_rye = float(profile_stop_rye)

        # Detect and adapt hour presets from older app versions if
        # max_minutes is still None.
        if max_minutes is None:
            if max_cycles == 120:
                max_minutes = 60.0
                max_cycles = 1_000_000
            elif max_cycles == 960:
                max_minutes = 480.0
                max_cycles = 1_000_000
            elif max_cycles == 2880:
                max_minutes = 1440.0
                max_cycles = 1_000_000
            elif max_cycles >= 10_000_000:
                forever = True
                max_cycles = 10_000_000

        # Try to resume from a previous interrupted continuous run
        checkpoint = None
        if resume_from_checkpoint and self.auto_resume_enabled:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.get("in_progress"):
                if (
                    checkpoint.get("goal") == goal
                    and checkpoint.get("role") == role
                    and checkpoint.get("domain") == (domain or "general")
                    and checkpoint.get("mode", "single") == "single"
                ):
                    remaining = checkpoint.get("remaining_minutes")
                    if isinstance(remaining, (int, float)) and remaining > 0:
                        # Treat remaining minutes as the new budget
                        if max_minutes is None or max_minutes > float(remaining):
                            max_minutes = float(remaining)

        # Start counting cycles from the existing history length
        history = self.memory_store.get_cycle_history()
        start_index = len(history)

        start_time = time.monotonic()
        last_watchdog_update = start_time

        i = 0

        while True:
            # Cycle-based stopping (unless in explicit forever mode)
            if not forever and i >= max_cycles:
                break

            # Time-based stopping
            elapsed_min = (time.monotonic() - start_time) / 60.0
            if max_minutes is not None and elapsed_min >= max_minutes:
                break

            # Watchdog heartbeat and checkpoint before starting the cycle
            now = time.monotonic()
            since_last_heartbeat = (now - last_watchdog_update) / 60.0
            if since_last_heartbeat >= min(watchdog_interval_minutes, max(1.0, watchdog_interval_minutes)):
                remaining_minutes = None
                if max_minutes is not None:
                    remaining_minutes = max(max_minutes - elapsed_min, 0.0)

                checkpoint_state: Dict[str, Any] = {
                    "version": 1,
                    "in_progress": True,
                    "mode": "single",
                    "goal": goal,
                    "role": role,
                    "domain": domain or "general",
                    "max_cycles": max_cycles,
                    "max_minutes": max_minutes,
                    "elapsed_minutes": elapsed_min,
                    "remaining_minutes": remaining_minutes,
                    "stop_rye": stop_rye,
                    "forever": forever,
                    "cycle_history_length": len(history),
                    "local_cycle_index": i,
                    "recent_rye": recent_rye[-10:],
                    "last_heartbeat_ts": time.time(),
                    "runtime_profile": runtime_profile,
                }
                self._save_checkpoint(checkpoint_state)
                # Use MemoryStore watchdog as an additional heartbeat
                try:
                    self.memory_store.heartbeat(label="continuous_run")
                except Exception:
                    pass
                last_watchdog_update = now

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
                    break

            i += 1

        # Continuous run finished cleanly
        self._clear_checkpoint()
        return summaries

    # ------------------------------------------------------------------
    # Swarm-aware continuous mode (multi-agent rounds)
    # ------------------------------------------------------------------
    def run_swarm_continuous(
        self,
        goal: str,
        max_rounds: int = 50,
        stop_rye: Optional[float] = None,
        roles: Optional[Sequence[str]] = None,
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        max_minutes: Optional[float] = None,
        forever: bool = False,
        resume_from_checkpoint: bool = True,
        watchdog_interval_minutes: float = 5.0,
        runtime_profile: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run continuous multi-agent "swarm" rounds.

        Each round runs one TGRM cycle per logical agent role (up to 32),
        sharing a single MemoryStore. This is the hybrid swarm mode:

            - Each role has a different function (researcher, critic, planner, etc.).
            - All roles write to the same MemoryStore and RYE history.
            - The system measures average RYE across the swarm per round.

        Args:
            goal:
                Research goal for the entire swarm.
            max_rounds:
                Hard cap on number of swarm rounds (each round may include
                up to 32 logical agents).
            stop_rye:
                Optional threshold on average swarm RYE. If the rolling
                average over recent rounds drops below this, the run stops.
            roles:
                Optional explicit list of roles. If None, uses self.agent_roles.
                Capped at self.max_agents (default 32).
            source_controls:
                Source configuration shared by all roles.
            pdf_bytes:
                Optional PDF bytes shared across all agents and rounds.
            biomarker_snapshot:
                Optional biomarker payload for future longevity-aware logic.
            domain:
                Optional domain tag ("general", "longevity", "math", etc.).
            max_minutes:
                Optional wall-clock time budget in minutes for the whole swarm run.
            forever:
                If True, ignore max_rounds and keep running until stopped
                by max_minutes, stop_rye, or environment limits.
            resume_from_checkpoint:
                If True, try to resume a previous swarm run (mode="swarm")
                that was interrupted.
            watchdog_interval_minutes:
                Minimum frequency of checkpoint and watchdog heartbeats.
            runtime_profile:
                Optional runtime profile name from presets.RUNTIME_PROFILES.
                Adjusts max_minutes, max_rounds (via estimated_cycles),
                and stop_rye when not explicitly set.
        Returns:
            Flat list of summaries produced by all roles across all rounds.
            Each summary includes the role label, cycle index, and RYE.
        """
        all_summaries: List[Dict[str, Any]] = []
        recent_round_rye: List[float] = []

        # Determine swarm role set
        if roles is None:
            roles_seq: Sequence[str] = self.agent_roles
        else:
            roles_seq = roles
        roles_list: List[str] = [str(r) for r in roles_seq][: self.max_agents]

        # Apply runtime profile hints if requested
        profile_cfg: Optional[Dict[str, Any]] = None
        if runtime_profile:
            profile_cfg = RUNTIME_PROFILES.get(runtime_profile)
        if profile_cfg is not None:
            est_cycles = profile_cfg.get("estimated_cycles")
            profile_stop_rye = profile_cfg.get("rye_stop_threshold")

            # For swarm: treat estimated_cycles as approximate total
            # agent-cycles. Convert to rounds by dividing by number of roles.
            if isinstance(est_cycles, (int, float)) and max_rounds <= 50:
                try:
                    per_round = max(1, len(roles_list) or 1)
                    approx_rounds = max(1, int(est_cycles / per_round))
                    max_rounds = approx_rounds
                except Exception:
                    pass

            if max_minutes is None:
                if runtime_profile == "1_hour":
                    max_minutes = 60.0
                elif runtime_profile == "8_hours":
                    max_minutes = 8 * 60.0
                elif runtime_profile == "24_hours":
                    max_minutes = 24 * 60.0
                elif runtime_profile == "forever":
                    forever = True

            if stop_rye is None and isinstance(profile_stop_rye, (int, float)):
                stop_rye = float(profile_stop_rye)

        # Try to resume from a previous swarm run
        checkpoint = None
        if resume_from_checkpoint and self.auto_resume_enabled:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.get("in_progress"):
                if (
                    checkpoint.get("goal") == goal
                    and checkpoint.get("mode") == "swarm"
                    and checkpoint.get("domain") == (domain or "general")
                ):
                    remaining = checkpoint.get("remaining_minutes")
                    if isinstance(remaining, (int, float)) and remaining > 0:
                        if max_minutes is None or max_minutes > float(remaining):
                            max_minutes = float(remaining)

        # Start cycle indexing after existing history
        history = self.memory_store.get_cycle_history()
        start_index = len(history)

        start_time = time.monotonic()
        last_watchdog_update = start_time

        round_idx = 0

        while True:
            # Round-based stopping (unless in explicit forever mode)
            if not forever and round_idx >= max_rounds:
                break

            # Time-based stopping
            elapsed_min = (time.monotonic() - start_time) / 60.0
            if max_minutes is not None and elapsed_min >= max_minutes:
                break

            # Watchdog and checkpoint
            now = time.monotonic()
            since_last_heartbeat = (now - last_watchdog_update) / 60.0
            if since_last_heartbeat >= min(watchdog_interval_minutes, max(1.0, watchdog_interval_minutes)):
                remaining_minutes = None
                if max_minutes is not None:
                    remaining_minutes = max(max_minutes - elapsed_min, 0.0)

                checkpoint_state: Dict[str, Any] = {
                    "version": 1,
                    "in_progress": True,
                    "mode": "swarm",
                    "goal": goal,
                    "domain": domain or "general",
                    "roles": list(roles_list),
                    "max_rounds": max_rounds,
                    "max_minutes": max_minutes,
                    "elapsed_minutes": elapsed_min,
                    "remaining_minutes": remaining_minutes,
                    "stop_rye": stop_rye,
                    "forever": forever,
                    "cycle_history_length": len(history),
                    "local_round_index": round_idx,
                    "recent_round_rye": recent_round_rye[-10:],
                    "last_heartbeat_ts": time.time(),
                    "runtime_profile": runtime_profile,
                }
                self._save_checkpoint(checkpoint_state)
                try:
                    self.memory_store.heartbeat(label="swarm_run")
                except Exception:
                    pass
                last_watchdog_update = now

            # Compute base cycle index for this round (each role gets a unique cycle index)
            base_ci = start_index + round_idx * len(roles_list or [None])

            round_summaries = self.run_multi_agent_round(
                goal=goal,
                base_cycle_index=base_ci,
                roles=roles_list,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
            )

            all_summaries.extend(round_summaries)

            # Compute average RYE across roles for this round
            round_ryes: List[float] = []
            for s in round_summaries:
                rv = s.get("RYE")
                if isinstance(rv, (int, float)):
                    round_ryes.append(float(rv))

            if round_ryes:
                avg_round_rye = sum(round_ryes) / len(round_ryes)
                recent_round_rye.append(avg_round_rye)
                if len(recent_round_rye) > 10:
                    recent_round_rye.pop(0)

                if stop_rye is not None and recent_round_rye:
                    avg_recent = sum(recent_round_rye) / len(recent_round_rye)
                    if avg_recent < stop_rye:
                        break

            round_idx += 1

        # Swarm run finished cleanly
        self._clear_checkpoint()
        return all_summaries
