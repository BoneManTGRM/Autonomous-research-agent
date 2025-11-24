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
    - Repair : perform targeted actions (web, PubMed, PDFs, biomarkers,
               browser tools, code sandbox, and data pipelines)
    - Verify : re-test, compute delta_R and RYE, and log the cycle

RYE (Repair Yield per Energy):
    For each cycle, TGRMLoop computes delta_R / E (improvement divided
    by effort). CoreAgent orchestrates cycles, multi-agent runs, and
    continuous modes while the TGRM loop integrates tool usage into E.

Multi-agent extension:
    CoreAgent can orchestrate multiple logical agent roles
    (researcher, critic, planner, synthesizer, explorer, plus up to
    a total of 32 agents) over the same MemoryStore. This makes it
    possible to run:
        - one cycle per role (multi-agent round), or
        - long continuous runs where each "round" consists of many roles.

    The maximum number of logical agents is capped at 32 for safety,
    and can be configured via config["max_agents"] (1-32).

Toolbelt and tool registry:
    CoreAgent owns a tools layer that can be:
        - a legacy Toolbelt instance (browser, sandbox, data pipelines)
        - or a TOOL_REGISTRY mapping exported from agent.tools package

    TGRMLoop can use this tools object to perform actions and track tool
    usage for energy accounting. CoreAgent also exposes the registry to
    the UI for status panels and future per tool RYE reporting.

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

Domain presets and experiment modes:
    CoreAgent also uses domain presets (via presets.get_preset) to:
        - pick sensible default runtime profiles per domain
        - suggest RYE stop thresholds
        - keep behavior aligned with longevity / math / general modes

    The experiment_mode (for example discovery, stability_test,
    protocol_builder) can be passed in and tagged on runs so that the
    meta controller and report generator can distinguish different runs.

Discovery extensions:
    CoreAgent can optionally cooperate with:
        - DiscoveryLogger for RYE spikes, hypotheses, and contradictions
        - HypothesisManager for a pending / validated / rejected pipeline
        - SnapshotGenerator for weekly experiment summaries
        - MemoryPruner for long run memory health
        - VerificationEngine for deeper checks of promising hypotheses
        - DiscoveryEngine (if available) for tiered discovery scoring

    These helpers sit on top of your existing TGRM loop and do not
    change its core behavior. They only consume the summary dicts and
    write logs or hypothesis records.

Learning from memory history:
    CoreAgent can analyze past cycle history (via MemoryStore and
    rye_metrics) and derive a "learned" profile that auto tunes:
        - suggested runtime profile (1 hour, 8 hours, 24 hours)
        - suggested stop_rye thresholds
        - diagnostics snapshot for the UI

    This learning is optional, conservative, and backward compatible:
        - If rye_metrics is unavailable, the feature is a no-op.
        - If config["auto_learn_from_memory"] is False, it is disabled.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .memory_store import MemoryStore
from .tgrm_loop import TGRMLoop
from .presets import RUNTIME_PROFILES, CONTINUOUS_MODE_DEFAULTS, get_preset

# Optional tools and registry imports, kept backward compatible.
try:
    # Legacy tools.py may still export Toolbelt
    from .tools import Toolbelt  # type: ignore[attr-defined]
except Exception:
    Toolbelt = None  # type: ignore

# Optional unified TOOL_REGISTRY from agent.tools package
try:
    from .tools import TOOL_REGISTRY  # type: ignore[attr-defined]
except Exception:
    TOOL_REGISTRY: Dict[str, Any] = {}  # type: ignore

# Optional rye_metrics hooks for future diagnostics and run health
try:
    from . import rye_metrics as _rye_metrics_mod  # type: ignore
except Exception:
    _rye_metrics_mod = None  # type: ignore

# Optional intelligence profiles for tuning thresholds and behavior
try:
    from .intelligence_profiles import get_intelligence_profile  # type: ignore[attr-defined]
except Exception:
    get_intelligence_profile = None  # type: ignore

# Optional discovery engine for higher level discovery scoring
try:
    from .discovery_engine import DiscoveryEngine  # type: ignore[attr-defined]
except Exception:
    DiscoveryEngine = None  # type: ignore

# Discovery stack imports
from .discovery_log import DiscoveryLogger, get_global_logger
from .hypothesis_manager import HypothesisManager
from .memory_pruner import MemoryPruner
from .snapshot_generator import SnapshotGenerator
from .verification_engine import VerificationEngine


class CoreAgent:
    """High-level controller for the autonomous research agent.

    This class wires UI or CLI controls (multi-agent, continuous mode,
    source preferences, PDF uploads, swarm roles) into the lower-level
    TGRMLoop which performs the actual reparodynamic TGRM cycles and
    RYE computation.

    It also owns a tools layer that TGRMLoop can use for:
        - headless browser actions
        - code execution sandbox
        - data pipelines (CSV / Excel / SQLite)
        - registry based tools via TOOL_REGISTRY
    """

    def __init__(self, memory_store: MemoryStore, config: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        # ------------------------------------------------------------------
        # Tools and tool registry
        # ------------------------------------------------------------------
        # Snapshot of the TOOL_REGISTRY mapping if available.
        if isinstance(TOOL_REGISTRY, dict):
            self.tool_registry: Dict[str, Any] = dict(TOOL_REGISTRY)
        else:
            self.tool_registry = {}

        # Legacy Toolbelt instance if it exists, otherwise fall back to registry.
        if Toolbelt is not None:
            self.tools = Toolbelt()
        else:
            # TGRMLoop will receive a plain dict of tool callables
            self.tools = self.tool_registry

        # Underlying TGRM loop that actually performs Test, Detect, Repair, Verify.
        # Prefer the newer signature that accepts tools=..., but keep a
        # backwards-compatible fallback for older TGRMLoop implementations.
        try:
            self.tgrm_loop = TGRMLoop(self.memory_store, self.config, tools=self.tools)  # type: ignore[call-arg]
        except TypeError:
            self.tgrm_loop = TGRMLoop(self.memory_store, self.config)

        # Source controls can be set by the UI at runtime.
        # Example:
        #   {
        #       "web": True,
        #       "pubmed": True,
        #       "semantic": False,
        #       "pdf": True,
        #       "biomarkers": False,
        #       "sandbox": False,
        #       "data_connectors": False,
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

        # Logical base roles. Additional generic roles are generated
        # as agent_01, agent_02, up to self.max_agents.
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
        # Discovery stack (sits on top of existing logic)
        # ------------------------------------------------------------------
        # Track last RYE to detect spikes.
        self._last_rye: Optional[float] = None

        # Shared discovery logger and hypothesis pipeline.
        # run_id will be set per run where needed.
        self.discovery_logger: DiscoveryLogger = get_global_logger()
        self.hypothesis_manager: HypothesisManager = HypothesisManager(run_id=None)

        # Long run helpers for 90 day experiments.
        self.memory_pruner: MemoryPruner = MemoryPruner(self.memory_store, run_id=None)
        self.snapshot_generator: SnapshotGenerator = SnapshotGenerator(run_id=None)

        # Verification engine for deeper checks on promising hypotheses.
        # The actual check functions can be wired from engine_worker.
        self.verification_engine: VerificationEngine = VerificationEngine(
            hypothesis_manager=self.hypothesis_manager,
            discovery_logger=self.discovery_logger,
            literature_check_fn=None,
            critique_fn=None,
            data_check_fn=None,
            run_id=None,
        )

        # Optional discovery engine for tiered scoring and more advanced
        # "what counts as a major discovery" logic.
        self.discovery_engine: Optional[Any] = None
        if DiscoveryEngine is not None:
            try:
                self.discovery_engine = DiscoveryEngine(
                    discovery_logger=self.discovery_logger,
                    hypothesis_manager=self.hypothesis_manager,
                    verification_engine=self.verification_engine,
                    memory_store=self.memory_store,
                    config=self.config,
                )
            except Exception:
                self.discovery_engine = None

        # ------------------------------------------------------------------
        # Intelligence profile support
        # ------------------------------------------------------------------
        self.intelligence_profile_name: Optional[str] = self.config.get("intelligence_profile")
        self.intelligence_profile: Dict[str, Any] = {}
        self._memory_prune_cfg: Dict[str, Any] = {}
        self._swarm_cfg: Dict[str, Any] = {}

        if get_intelligence_profile is not None and self.intelligence_profile_name:
            try:
                self.intelligence_profile = get_intelligence_profile(self.intelligence_profile_name)
            except Exception:
                self.intelligence_profile = {}
        else:
            self.intelligence_profile = {}

        # ------------------------------------------------------------------
        # Learned profile from memory history
        # ------------------------------------------------------------------
        # This holds the last learned suggestions from learn_from_memory:
        #   {
        #     "has_history": bool,
        #     "recommended_runtime_profile": Optional[str],
        #     "recommended_stop_rye": Optional[float],
        #     "diagnostics": Optional[dict],
        #   }
        self.learned_from_memory: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Agent-level training / learning configuration (added)
        # ------------------------------------------------------------------
        # Learning is at the agent level (TGRM + RYE + memory), not model weights.
        self.learning_enabled: bool = bool(self.config.get("learning_enabled", True))
        # Minimum cycle history before we trust learned suggestions.
        self.min_training_history: int = int(self.config.get("min_training_history", 50))
        if self.min_training_history < 0:
            self.min_training_history = 0

        # Last training burst summary (short diagnostic/optimization runs).
        self.training_profile: Dict[str, Any] = {}
        # Last high-level learning plan produced by optimize_learning_pipeline().
        self.learning_plan: Dict[str, Any] = {}

        self._apply_intelligence_profile()

    # ------------------------------------------------------------------
    # Intelligence profile helpers
    # ------------------------------------------------------------------
    def _apply_intelligence_profile(self) -> None:
        """Apply intelligence profile settings to verification, pruning, swarm."""
        profile = self.intelligence_profile or {}
        ver_cfg = profile.get("verification", {})
        if isinstance(ver_cfg, dict):
            vt = ver_cfg.get("validate_threshold")
            rt = ver_cfg.get("reject_threshold")
            if isinstance(vt, (int, float)):
                self.verification_engine.validate_threshold = float(vt)
            if isinstance(rt, (int, float)):
                self.verification_engine.reject_threshold = float(rt)

        mem_cfg = profile.get("memory_pruning", {})
        if isinstance(mem_cfg, dict):
            self._memory_prune_cfg = dict(mem_cfg)
        else:
            self._memory_prune_cfg = {}

        swarm_cfg = profile.get("swarm", {})
        if isinstance(swarm_cfg, dict):
            self._swarm_cfg = dict(swarm_cfg)
        else:
            self._swarm_cfg = {}

    def set_intelligence_profile(self, name: str) -> None:
        """Change the active intelligence profile at runtime."""
        self.intelligence_profile_name = name
        if get_intelligence_profile is not None:
            try:
                self.intelligence_profile = get_intelligence_profile(name)
            except Exception:
                self.intelligence_profile = {}
        else:
            self.intelligence_profile = {}
        self._apply_intelligence_profile()

    # ------------------------------------------------------------------
    # Learning from memory history
    # ------------------------------------------------------------------
    def learn_from_memory(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Analyze past cycles and derive learned runtime hints.

        This method is conservative and purely advisory:
            - It never raises.
            - It does not mutate external state except self.learned_from_memory.
            - It can be called by the UI, workers, or continuous modes.

        Returns:
            Dict with keys:
                - "has_history": bool
                - "recommended_runtime_profile": Optional[str]
                - "recommended_stop_rye": Optional[float]
                - "diagnostics": Optional[dict]
        """
        learned: Dict[str, Any] = {
            "has_history": False,
            "recommended_runtime_profile": None,
            "recommended_stop_rye": None,
            "diagnostics": None,
        }

        try:
            history = self.memory_store.get_cycle_history()
        except Exception:
            history = []

        if not history or len(history) < self.min_training_history:
            self.learned_from_memory = learned
            return learned

        learned["has_history"] = True

        diagnostics: Optional[Dict[str, Any]] = None
        effective_domain = domain or "general"

        if _rye_metrics_mod is not None and hasattr(_rye_metrics_mod, "build_run_diagnostics"):
            try:
                diagnostics = _rye_metrics_mod.build_run_diagnostics(  # type: ignore[attr-defined]
                    history,
                    domain=effective_domain,
                )
            except Exception:
                diagnostics = None

        learned["diagnostics"] = diagnostics

        # Simple heuristic for runtime profile and stop_rye based on diagnostics
        recommended_profile: Optional[str] = None
        recommended_stop_rye: Optional[float] = None

        if isinstance(diagnostics, dict):
            avg = diagnostics.get("rye_avg")
            stab = diagnostics.get("stability_index")
            slope = diagnostics.get("trend_slope")
            mid = diagnostics.get("mid_percentile")

            # Choose profile based on average efficiency and stability
            if isinstance(avg, (int, float)) and isinstance(stab, (int, float)):
                if avg >= 0.5 and stab >= 0.6:
                    # High efficiency and stable: safe to attempt a long run
                    recommended_profile = "24_hours"
                elif avg >= 0.2 and stab >= 0.3:
                    # Moderate efficiency: medium run
                    recommended_profile = "8_hours"
                else:
                    # Noisy or weak efficiency: short diagnostic run
                    recommended_profile = "1_hour"

            # Use the mid percentile as a soft baseline for stop_rye
            if isinstance(mid, (int, float)):
                # Keep threshold conservative to avoid premature termination
                recommended_stop_rye = float(mid) * 0.5

            # If trend is clearly negative, bias toward shorter profile
            if isinstance(slope, (int, float)) and slope < 0 and recommended_profile == "24_hours":
                recommended_profile = "8_hours"

        learned["recommended_runtime_profile"] = recommended_profile
        learned["recommended_stop_rye"] = recommended_stop_rye

        self.learned_from_memory = learned
        return learned

    # ------------------------------------------------------------------
    # Internal helpers for crash proofing and run tracking
    # ------------------------------------------------------------------
    def _generate_run_id(self) -> str:
        """Generate a unique run identifier for continuous or worker runs."""
        ts = int(time.time())
        rand = uuid.uuid4().hex[:8]
        return f"run-{ts}-{rand}"

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

    def _log_run_manifest(
        self,
        run_id: str,
        mode: str,
        goal: str,
        domain: str,
        experiment_mode: Optional[str],
        run_metadata: Dict[str, Any],
        summaries: List[Dict[str, Any]],
    ) -> None:
        """Send a compact run manifest into MemoryStore if supported.

        This is intentionally conservative and will silently do nothing
        if MemoryStore does not provide log_run_manifest yet.
        """
        manifest: Dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "goal": goal,
            "domain": domain,
            "experiment_mode": experiment_mode,
            "run_metadata": dict(run_metadata),
            # Caller may trim summaries before passing here if needed.
            "summaries": summaries,
        }
        try:
            if hasattr(self.memory_store, "log_run_manifest"):
                # Preferred future signature: log_run_manifest(run_id, manifest)
                try:
                    self.memory_store.log_run_manifest(run_id, manifest)  # type: ignore[attr-defined]
                except TypeError:
                    # Fallback: some variants might only accept manifest
                    try:
                        self.memory_store.log_run_manifest(manifest)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            return

    # ------------------------------------------------------------------
    # Discovery helpers (non intrusive)
    # ------------------------------------------------------------------
    def _handle_post_cycle_discovery(
        self,
        summary: Dict[str, Any],
        *,
        cycle_index: int,
        role: str,
        domain: Optional[str],
        run_id: Optional[str],
        experiment_mode: Optional[str],
    ) -> None:
        """Inspect a cycle summary and log discovery related signals.

        This method never raises. It reads from summary and writes into
        the discovery logger and hypothesis manager, without changing
        the TGRM core behavior.

        Current behavior:
            - optional DiscoveryEngine tiered processing
            - detect RYE spikes and log them
            - register candidate hypotheses surfaced by the loop
        """
        # First, allow DiscoveryEngine to see the full picture
        if self.discovery_engine is not None:
            try:
                context = {
                    "cycle_index": cycle_index,
                    "role": role,
                    "domain": domain,
                    "run_id": run_id,
                    "experiment_mode": experiment_mode,
                }
                # process_cycle signature is intentionally loose; any mismatch is caught.
                self.discovery_engine.process_cycle(summary, context=context)  # type: ignore[attr-defined]
            except Exception:
                pass

        try:
            # Attach run_id into helpers if present
            if run_id is not None:
                self.discovery_logger.run_id = run_id
                self.hypothesis_manager.run_id = run_id
                self.memory_pruner.run_id = run_id
                self.snapshot_generator.run_id = run_id
                self.verification_engine.run_id = run_id
        except Exception:
            pass

        # 1) RYE spike logging
        try:
            rye_val = summary.get("RYE")
            delta_r = summary.get("delta_R")
            energy = summary.get("Energy") or summary.get("energy")

            if isinstance(rye_val, (int, float)):
                rye_float = float(rye_val)
                previous = self._last_rye

                min_rye_spike = float(self.config.get("rye_spike_min", 0.5))
                spike = False
                if previous is None and rye_float >= min_rye_spike:
                    spike = True
                elif previous is not None and rye_float >= max(min_rye_spike, previous * 1.25):
                    spike = True

                if spike:
                    lines: List[str] = []
                    lines.append(f"RYE reached {rye_float:.4f} on cycle {cycle_index}.")
                    if isinstance(delta_r, (int, float)):
                        lines.append(f"Estimated delta_R: {float(delta_r):.4f}.")
                    if isinstance(energy, (int, float)):
                        lines.append(f"Estimated energy: {float(energy):.4f}.")
                    if domain:
                        lines.append(f"Domain: {domain}.")
                    if experiment_mode:
                        lines.append(f"Experiment mode: {experiment_mode}.")

                    desc = "\n".join(lines)

                    self.discovery_logger.log_rye_spike(
                        title=f"RYE spike at cycle {cycle_index}",
                        description=desc,
                        cycle_index=cycle_index,
                        agent_role=role,
                        rye_before=previous if isinstance(previous, (int, float)) else None,
                        rye_after=rye_float,
                        delta_r=float(delta_r) if isinstance(delta_r, (int, float)) else None,
                        energy=float(energy) if isinstance(energy, (int, float)) else None,
                        tags=[domain] if domain else None,
                        extra={"summary": summary},
                    )

                self._last_rye = rye_float
        except Exception:
            # Discovery hooks must never interfere with the main loop
            pass

        # 2) Hypothesis registration from summary["candidate_hypotheses"]
        try:
            raw_hyp_list = (
                summary.get("candidate_hypotheses")
                or summary.get("hypotheses")
                or []
            )

            if not isinstance(raw_hyp_list, list):
                return

            for item in raw_hyp_list:
                if isinstance(item, str):
                    title = item
                    description = ""
                    tags: Optional[List[str]] = None
                elif isinstance(item, dict):
                    title = str(item.get("title") or "Untitled hypothesis")
                    description = str(item.get("description") or "")
                    raw_tags = item.get("tags") or []
                    tags = [str(t) for t in raw_tags] if isinstance(raw_tags, (list, tuple)) else None
                else:
                    title = str(item)
                    description = ""
                    tags = None

                rec = self.hypothesis_manager.create_hypothesis(
                    title=title,
                    description=description,
                    cycle_index=cycle_index,
                    agent_role=role,
                    rye_before=None if self._last_rye is None else float(self._last_rye),
                    rye_after=summary.get("RYE") if isinstance(summary.get("RYE"), (int, float)) else None,
                    delta_r=summary.get("delta_R") if isinstance(summary.get("delta_R"), (int, float)) else None,
                    energy=summary.get("Energy") if isinstance(summary.get("Energy"), (int, float)) else None,
                    tags=tags,
                )

                self.discovery_logger.log_hypothesis(
                    title=rec.title,
                    description=rec.description,
                    cycle_index=rec.cycle_index,
                    agent_role=rec.agent_role,
                    rye_before=rec.rye_before,
                    rye_after=rec.rye_after,
                    delta_r=rec.delta_r,
                    energy=rec.energy,
                    tags=rec.tags,
                    extra={"hypothesis_id": rec.hypothesis_id},
                )
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

        This is a light-weight way for agents to create agents.
        Child agents share the same memory substrate but can have their
        own configuration (for example different prompts, domains, and so on).

        The child will inherit max_agents (capped at 32) unless overridden
        in extra_config["max_agents"].
        """
        cfg = dict(self.config)
        if extra_config:
            cfg.update(extra_config)
        return CoreAgent(memory_store=self.memory_store, config=cfg)

    # ------------------------------------------------------------------
    # Persistence helpers for background workers
    # ------------------------------------------------------------------
    def load_state_from_storage(self) -> Dict[str, Any]:
        """Load the latest engine run state from persistent storage.

        Priority:
            1. MemoryStore.load_run_state() if available.
            2. JSON checkpoint file (used as a fallback).

        Returns:
            A dict representing the last known engine state.
            Returns {} if nothing valid is found.
        """
        # Prefer MemoryStore-based state if implemented
        try:
            if hasattr(self.memory_store, "load_run_state"):
                state = self.memory_store.load_run_state()  # type: ignore[attr-defined]
            else:
                state = self._load_checkpoint()
        except Exception:
            state = None

        if not isinstance(state, dict):
            return {}
        return state

    def save_state_to_storage(self, state: Dict[str, Any]) -> None:
        """Save engine run state to persistent storage.

        Priority:
            1. MemoryStore.save_run_state() if available.
            2. JSON checkpoint file (fallback).
        """
        if not isinstance(state, dict):
            return

        # Defensive copy so callers can hold their own reference
        payload: Dict[str, Any] = dict(state)

        # Try MemoryStore first
        try:
            if hasattr(self.memory_store, "save_run_state"):
                self.memory_store.save_run_state(payload)  # type: ignore[attr-defined]
                return
        except Exception:
            # If MemoryStore persistence fails, fall back to checkpoint file.
            pass

        # Fallback: store minimal state in the checkpoint file
        try:
            checkpoint_state: Dict[str, Any] = {
                "version": 1,
                "in_progress": bool(payload.get("in_progress", True)),
                "mode": payload.get("mode", "single"),
                "goal": payload.get("goal"),
                "role": payload.get("role", "agent"),
                "domain": payload.get("domain", "general"),
                "run_id": payload.get("run_id"),
                "experiment_mode": payload.get("experiment_mode"),
                "cycle_index": payload.get("cycle_index"),
                "cycles_completed": payload.get("cycles_completed"),
                "last_rye": payload.get("last_rye"),
                "last_update_ts": payload.get("last_update_ts", time.time()),
                "runtime_profile": payload.get("runtime_profile"),
                "raw_state": payload,
            }
            self._save_checkpoint(checkpoint_state)
        except Exception:
            return

    def run_one_cycle_and_return_state(
        self,
        state: Optional[Dict[str, Any]] = None,
        *,
        goal: Optional[str] = None,
        role: str = "agent",
        domain: Optional[str] = None,
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run exactly one TGRM cycle and return updated state plus summary.

        This is the primary entry point for engine_worker.py:

            1. Load state from storage (or start with None).
            2. Call run_one_cycle_and_return_state(...)
            3. Persist the returned state using save_state_to_storage()
            4. Sleep a bit and repeat.

        Args:
            state:
                Existing engine state dict, or None to initialize from scratch.
            goal:
                Optional override of the research goal. If not provided, reads from state["goal"].
            role:
                Logical role label for this cycle ("agent", "researcher", "critic", etc.).
            domain:
                Optional domain tag ("general", "longevity", "math", ...). Overrides state["domain"] if provided.
            source_controls:
                Optional per cycle source configuration.
            pdf_bytes:
                Optional PDF payload for this cycle.
            biomarker_snapshot:
                Optional biomarker payload for future longevity aware logic.
            run_id:
                Optional run identifier for this sequence of cycles. If not provided
                and no run_id exists in state, one will be generated.
            experiment_mode:
                Optional experiment mode label (for example "discovery").

        Returns:
            Dict with keys:
                - "state": updated engine state dict
                - "summary": human-facing cycle summary dict
        """
        # Start from existing state or initialize a new one
        if state is None:
            history = self.memory_store.get_cycle_history()
            next_index = len(history)
            resolved_run_id = run_id or self._generate_run_id()
            state = {
                "mode": "single",
                "goal": goal or "",
                "role": role,
                "domain": domain or "general",
                "cycle_index": next_index,
                "cycles_completed": 0,
                "created_at_ts": time.time(),
                "last_update_ts": time.time(),
                "last_rye": None,
                "in_progress": True,
                "run_id": resolved_run_id,
                "experiment_mode": experiment_mode,
            }
            if source_controls is not None:
                state["source_controls"] = dict(source_controls)
        else:
            # Make a shallow copy so we do not mutate caller's dict in place
            state = dict(state)

        # Resolve run_id and experiment_mode from parameters or state
        effective_run_id: str = run_id or state.get("run_id") or self._generate_run_id()
        state["run_id"] = effective_run_id

        if experiment_mode is None:
            experiment_mode = state.get("experiment_mode")
        state["experiment_mode"] = experiment_mode

        # Resolve goal / role / domain from parameters or state
        effective_goal: str = (goal or state.get("goal") or "").strip()
        state["goal"] = effective_goal

        effective_role: str = role or state.get("role", "agent")
        state["role"] = effective_role

        effective_domain: str = domain or state.get("domain", "general")
        state["domain"] = effective_domain

        # Resolve source controls:
        # parameter > state["source_controls"] > self.source_controls
        if source_controls is not None:
            effective_sources: Dict[str, bool] = dict(source_controls)
            state["source_controls"] = dict(source_controls)
        else:
            stored_sc = state.get("source_controls")
            if isinstance(stored_sc, dict):
                effective_sources = dict(stored_sc)
            else:
                effective_sources = dict(self.source_controls)

        # Determine cycle index
        try:
            ci = int(state.get("cycle_index", 0))
            if ci < 0:
                ci = 0
        except Exception:
            history = self.memory_store.get_cycle_history()
            ci = len(history)
        state["cycle_index"] = ci

        # Choose PDF bytes: if not passed in, we fall back to attached PDF
        if pdf_bytes is None:
            effective_pdf_bytes = self._attached_pdf_bytes
        else:
            effective_pdf_bytes = pdf_bytes

        # Run a single cycle (TGRM plus tools plus RYE)
        result = self.run_cycle(
            goal=effective_goal,
            cycle_index=ci,
            role=effective_role,
            source_controls=effective_sources,
            pdf_bytes=effective_pdf_bytes,
            biomarker_snapshot=biomarker_snapshot,
            domain=effective_domain,
            run_id=effective_run_id,
            experiment_mode=experiment_mode,
        )
        summary = result.get("summary", {})

        # Update engine state with result metadata
        state["last_update_ts"] = time.time()
        state["cycles_completed"] = int(state.get("cycles_completed", 0)) + 1
        state["cycle_index"] = ci + 1
        state["last_summary"] = summary

        rye_val = summary.get("RYE")
        if isinstance(rye_val, (int, float)):
            state["last_rye"] = float(rye_val)

        # Mark as still in progress; the worker or UI can toggle this off if needed
        state["in_progress"] = True

        return {
            "state": state,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Public API used by Streamlit and CLI
    # ------------------------------------------------------------------
    def attach_pdf(self, uploaded_file: Any) -> None:
        """Attach a PDF (uploaded via Streamlit) for later cycles.

        The TGRM loop can then ingest this PDF as part of its REPAIR
        actions. If uploaded_file is None, this does nothing.
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
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single cycle of research using the TGRM loop.

        Args:
            goal:
                Research goal for this cycle.
            cycle_index:
                Global cycle index (used for logging and history).
            role:
                Logical role for this cycle such as "agent", "researcher",
                or "critic". This is recorded in the logs and
                allows multi-agent setups.
            source_controls:
                Optional override for which sources to use in this cycle.
                If None, falls back to self.source_controls.
            pdf_bytes:
                Optional PDF bytes for this cycle. If None, falls back
                to any previously attached PDF stored in self._attached_pdf_bytes.
            biomarker_snapshot:
                Placeholder for a future biomarker or lab value payload.
            domain:
                Optional domain tag such as "general", "longevity", or "math"
                to label this cycle in the logs.
            run_id:
                Optional run identifier for the enclosing run.
            experiment_mode:
                Optional experiment mode label (for example "discovery").

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
        # Use a TypeError safe wrapper for compatibility with older signatures.
        try:
            result = self.tgrm_loop.run_cycle(
                goal=goal,
                cycle_index=cycle_index,
                role=role,
                source_controls=effective_source_controls,
                pdf_bytes=effective_pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
            )
        except TypeError:
            # Backward compatibility: older TGRMLoop.run_cycle
            # that only accepts (goal, cycle_index) or (goal, cycle_index, role).
            try:
                result = self.tgrm_loop.run_cycle(goal, cycle_index, role)
            except TypeError:
                result = self.tgrm_loop.run_cycle(goal, cycle_index)

        summary = result.get("summary", {})

        # Tag summary with run level context if present
        if isinstance(summary, dict):
            if run_id is not None:
                summary.setdefault("run_id", run_id)
            if experiment_mode is not None:
                summary.setdefault("experiment_mode", experiment_mode)
            if domain is not None:
                summary.setdefault("domain", domain)
            summary.setdefault("role", role)
            summary.setdefault("cycle_index", cycle_index)

            # If the TGRM loop returned tool_stats at the result level, mirror
            # them into the summary so the UI and report generator can see them.
            tool_stats = result.get("tool_stats")
            if tool_stats is not None and "tool_stats" not in summary:
                summary["tool_stats"] = tool_stats

            # Discovery aware post processing (new hook, does not change core behavior)
            self._handle_post_cycle_discovery(
                summary,
                cycle_index=cycle_index,
                role=role,
                domain=domain,
                run_id=run_id,
                experiment_mode=experiment_mode,
            )

        return result

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
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run multiple TGRM cycles in a row (continuous mode) for a single role.

        This is the long running reparodynamic mode where the agent
        repeatedly applies Test, Detect, Repair, Verify to gradually
        improve its internal state under a fixed goal.

        Args:
            goal:
                Research goal to pursue over many cycles.
            max_cycles:
                Hard cap on the number of cycles to prevent infinite runs
                in hosted environments (used when forever is False).
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
                future biomarker aware logic.
            domain:
                Optional domain tag forwarded into each cycle.
            max_minutes:
                Optional wall clock time budget in minutes. If provided,
                the loop stops once this time has elapsed (up to per cycle
                granularity).
            forever:
                If True, ignore max_cycles and keep running until
                stopped by max_minutes, stop_rye, or the environment.
            resume_from_checkpoint:
                If True, try to detect a previous interrupted continuous run from
                the checkpoint file and adjust the remaining time budget.
            watchdog_interval_minutes:
                How often to update the checkpoint heartbeat at minimum.
            runtime_profile:
                Optional name of a runtime profile from presets.RUNTIME_PROFILES.
            run_id:
                Optional identifier for this continuous run. If not provided,
                one will be generated.
            experiment_mode:
                Optional experiment mode label (for example "discovery").

        Returns:
            List[Dict[str, Any]]: List of human-facing summaries for
            each completed cycle. Each summary includes run_metadata
            indicating how long the run lasted and why it stopped.
        """
        summaries: List[Dict[str, Any]] = []
        recent_rye: List[float] = []

        # Ensure a run_id exists for this continuous run
        effective_run_id = run_id or self._generate_run_id()

        # Enforce a global failsafe on max_cycles
        max_cycles_failsafe = int(CONTINUOUS_MODE_DEFAULTS.get("max_cycles_failsafe", 10_000_000))
        if max_cycles > max_cycles_failsafe:
            max_cycles = max_cycles_failsafe

        # Domain aware preset defaults
        effective_domain = domain or "general"
        preset_cfg = get_preset(effective_domain)

        # Watchdog interval default from global config if caller did not override
        default_watchdog = float(CONTINUOUS_MODE_DEFAULTS.get("watchdog_interval_minutes", watchdog_interval_minutes))
        if watchdog_interval_minutes == 5.0:
            watchdog_interval_minutes = default_watchdog

        # Optional memory driven auto tuning before we choose runtime profile
        if runtime_profile is None and self.config.get("auto_learn_from_memory", True):
            try:
                learned = self.learn_from_memory(domain=effective_domain)
                lp = learned.get("recommended_runtime_profile")
                if isinstance(lp, str):
                    runtime_profile = lp
                if stop_rye is None:
                    lr = learned.get("recommended_stop_rye")
                    if isinstance(lr, (int, float)):
                        stop_rye = float(lr)
            except Exception:
                # Learning must never break the run
                pass

        # Default runtime profile from domain preset if not provided
        if runtime_profile is None:
            runtime_profile = preset_cfg.get("default_runtime_profile")

        # Apply runtime profile hints if requested
        profile_cfg: Optional[Dict[str, Any]] = None
        if runtime_profile:
            profile_cfg = RUNTIME_PROFILES.get(runtime_profile)

        if profile_cfg is not None:
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
                elif runtime_profile == "90_days":
                    # 90 days in minutes, if ever used in single mode
                    max_minutes = 90 * 24 * 60.0
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

        # If still no explicit stop_rye and preset defines a default
        if stop_rye is None:
            preset_stop = preset_cfg.get("default_rye_stop_threshold")
            if isinstance(preset_stop, (int, float)):
                stop_rye = float(preset_stop)

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
            elif max_cycles >= max_cycles_failsafe:
                forever = True
                max_cycles = max_cycles_failsafe

        # Try to resume from a previous interrupted continuous run
        checkpoint = None
        if resume_from_checkpoint and self.auto_resume_enabled:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.get("in_progress"):
                if (
                    checkpoint.get("goal") == goal
                    and checkpoint.get("role") == role
                    and checkpoint.get("domain") == effective_domain
                    and checkpoint.get("mode", "single") == "single"
                ):
                    remaining = checkpoint.get("remaining_minutes")
                    if isinstance(remaining, (int, float)) and remaining > 0:
                        # Treat remaining minutes as the new budget if tighter
                        if max_minutes is None or max_minutes > float(remaining):
                            max_minutes = float(remaining)

        # Start counting cycles from the existing history length
        history = self.memory_store.get_cycle_history()
        start_index = len(history)

        start_time = time.monotonic()
        last_watchdog_update = start_time

        i = 0
        stop_reason: str = "completed"

        # Heartbeat label for single agent runs
        heartbeat_label = CONTINUOUS_MODE_DEFAULTS.get("heartbeat_labels", {}).get("single", "continuous_single")

        while True:
            # Cycle-based stopping (unless in explicit forever mode)
            if not forever and i >= max_cycles:
                stop_reason = "cycle_cap"
                break

            # Time based stopping
            elapsed_min = (time.monotonic() - start_time) / 60.0
            if max_minutes is not None and elapsed_min >= max_minutes:
                stop_reason = "time_limit"
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
                    "domain": effective_domain,
                    "run_id": effective_run_id,
                    "experiment_mode": experiment_mode,
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
                    self.memory_store.heartbeat(label=heartbeat_label)
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
                domain=effective_domain,
                run_id=effective_run_id,
                experiment_mode=experiment_mode,
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
                    stop_reason = "rye_threshold"
                    break

            i += 1

        # Continuous run finished cleanly
        self._clear_checkpoint()

        # Attach run-level metadata (including how long it actually ran)
        total_elapsed_min = (time.monotonic() - start_time) / 60.0
        run_metadata: Dict[str, Any] = {
            "mode": "single",
            "goal": goal,
            "role": role,
            "domain": effective_domain,
            "run_id": effective_run_id,
            "experiment_mode": experiment_mode,
            "elapsed_minutes": total_elapsed_min,
            "max_minutes": max_minutes,
            "max_cycles": max_cycles,
            "cycles_completed": len(summaries),
            "stop_rye": stop_rye,
            "stop_reason": stop_reason,
            "runtime_profile": runtime_profile,
        }

        # Optional run health score or diagnostics if rye_metrics provides them
        if _rye_metrics_mod is not None:
            try:
                if hasattr(_rye_metrics_mod, "run_health_score"):
                    health_score = _rye_metrics_mod.run_health_score(summaries)  # type: ignore[attr-defined]
                    run_metadata["run_health_score"] = health_score
            except Exception:
                pass

        # Attach the last learned_from_memory snapshot for observability
        if self.learned_from_memory:
            run_metadata["learned_from_memory"] = dict(self.learned_from_memory)

        for s in summaries:
            if isinstance(s, dict):
                s["run_metadata"] = dict(run_metadata)

        # Log a compact manifest into MemoryStore if the new API exists
        self._log_run_manifest(
            run_id=effective_run_id,
            mode="single",
            goal=goal,
            domain=effective_domain,
            experiment_mode=experiment_mode,
            run_metadata=run_metadata,
            summaries=summaries,
        )

        return summaries

    # ------------------------------------------------------------------
    # Swarm aware continuous mode (multi-agent rounds)
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
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run continuous multi-agent swarm rounds.

        Each round runs one TGRM cycle per logical agent role (up to 32),
        sharing a single MemoryStore. This is the hybrid swarm mode:

            - Each role has a different function such as researcher, critic, planner.
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
                Optional biomarker payload for future longevity aware logic.
            domain:
                Optional domain tag such as "general", "longevity", or "math".
            max_minutes:
                Optional wall clock time budget in minutes for the whole swarm run.
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
            run_id:
                Optional identifier for this swarm run. If not provided,
                one will be generated.
            experiment_mode:
                Optional experiment mode label for this swarm run.

        Returns:
            Flat list of summaries produced by all roles across all rounds.
            Each summary includes run_metadata with swarm run duration
            and stop_reason.
        """
        all_summaries: List[Dict[str, Any]] = []
        recent_round_rye: List[float] = []

        # Ensure a run_id exists for this swarm run
        effective_run_id = run_id or self._generate_run_id()

        # Domain aware preset defaults
        effective_domain = domain or "general"
        preset_cfg = get_preset(effective_domain)

        # Determine swarm role set
        if roles is None:
            roles_seq: Sequence[str] = self.agent_roles
        else:
            roles_seq = roles
        roles_list: List[str] = [str(r) for r in roles_seq][: self.max_agents]

        # Apply swarm related overrides from intelligence profile if present
        if self._swarm_cfg:
            try:
                if "max_rounds" in self._swarm_cfg and max_rounds == 50:
                    mr = self._swarm_cfg.get("max_rounds")
                    if isinstance(mr, int) and mr > 0:
                        max_rounds = mr
                if "stop_rye" in self._swarm_cfg and stop_rye is None:
                    sr = self._swarm_cfg.get("stop_rye")
                    if isinstance(sr, (int, float)):
                        stop_rye = float(sr)
            except Exception:
                pass

        # Watchdog interval default from global config if caller did not override
        default_watchdog = float(CONTINUOUS_MODE_DEFAULTS.get("watchdog_interval_minutes", watchdog_interval_minutes))
        if watchdog_interval_minutes == 5.0:
            watchdog_interval_minutes = default_watchdog

        # Optional memory driven auto tuning before runtime profile selection
        if runtime_profile is None and self.config.get("auto_learn_from_memory", True):
            try:
                learned = self.learn_from_memory(domain=effective_domain)
                lp = learned.get("recommended_runtime_profile")
                if isinstance(lp, str):
                    runtime_profile = lp
                if stop_rye is None:
                    lr = learned.get("recommended_stop_rye")
                    if isinstance(lr, (int, float)):
                        stop_rye = float(lr)
            except Exception:
                pass

        # Default runtime profile from domain preset if not provided
        if runtime_profile is None:
            runtime_profile = preset_cfg.get("default_runtime_profile")

        # Apply runtime profile hints if requested
        profile_cfg: Optional[Dict[str, Any]] = None
        if runtime_profile:
            profile_cfg = RUNTIME_PROFILES.get(runtime_profile)

        if profile_cfg is not None:
            est_cycles = profile_cfg.get("estimated_cycles")
            profile_stop_rye = profile_cfg.get("rye_stop_threshold")

            # For swarm: treat estimated_cycles as approximate total
            # agent cycles. Convert to rounds by dividing by number of roles.
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
                elif runtime_profile == "90_days":
                    max_minutes = 90 * 24 * 60.0
                elif runtime_profile == "forever":
                    forever = True

            if stop_rye is None and isinstance(profile_stop_rye, (int, float)):
                stop_rye = float(profile_stop_rye)

        # If still no explicit stop_rye and preset defines a default
        if stop_rye is None:
            preset_stop = preset_cfg.get("default_rye_stop_threshold")
            if isinstance(preset_stop, (int, float)):
                stop_rye = float(preset_stop)

        # Try to resume from a previous swarm run
        checkpoint = None
        if resume_from_checkpoint and self.auto_resume_enabled:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.get("in_progress"):
                if (
                    checkpoint.get("goal") == goal
                    and checkpoint.get("mode") == "swarm"
                    and checkpoint.get("domain") == effective_domain
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
        stop_reason: str = "completed"

        # Heartbeat label for swarm runs
        heartbeat_label = CONTINUOUS_MODE_DEFAULTS.get("heartbeat_labels", {}).get("swarm", "continuous_swarm")

        while True:
            # Round based stopping (unless in explicit forever mode)
            if not forever and round_idx >= max_rounds:
                stop_reason = "round_cap"
                break

            # Time based stopping
            elapsed_min = (time.monotonic() - start_time) / 60.0
            if max_minutes is not None and elapsed_min >= max_minutes:
                stop_reason = "time_limit"
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
                    "domain": effective_domain,
                    "roles": list(roles_list),
                    "run_id": effective_run_id,
                    "experiment_mode": experiment_mode,
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
                    self.memory_store.heartbeat(label=heartbeat_label)
                except Exception:
                    pass
                last_watchdog_update = now

            # Compute base cycle index for this round (each role gets a unique cycle index)
            base_ci = start_index + round_idx * max(1, len(roles_list))

            round_summaries = self.run_multi_agent_round(
                goal=goal,
                base_cycle_index=base_ci,
                roles=roles_list,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=effective_domain,
                run_id=effective_run_id,
                experiment_mode=experiment_mode,
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
                        stop_reason = "rye_threshold"
                        break

            round_idx += 1

        # Swarm run finished cleanly
        self._clear_checkpoint()

        # Attach run-level metadata (including how long it actually ran)
        total_elapsed_min = (time.monotonic() - start_time) / 60.0
        run_metadata: Dict[str, Any] = {
            "mode": "swarm",
            "goal": goal,
            "roles": list(roles_list),
            "domain": effective_domain,
            "run_id": effective_run_id,
            "experiment_mode": experiment_mode,
            "elapsed_minutes": total_elapsed_min,
            "max_minutes": max_minutes,
            "max_rounds": max_rounds,
            "rounds_completed": round_idx,
            "stop_rye": stop_rye,
            "stop_reason": stop_reason,
            "runtime_profile": runtime_profile,
        }

        # Optional run health score or diagnostics if rye_metrics provides them
        if _rye_metrics_mod is not None:
            try:
                if hasattr(_rye_metrics_mod, "run_health_score"):
                    health_score = _rye_metrics_mod.run_health_score(all_summaries)  # type: ignore[attr-defined]
                    run_metadata["run_health_score"] = health_score
            except Exception:
                pass

        # Attach the last learned_from_memory snapshot for observability
        if self.learned_from_memory:
            run_metadata["learned_from_memory"] = dict(self.learned_from_memory)

        for s in all_summaries:
            if isinstance(s, dict):
                s["run_metadata"] = dict(run_metadata)

        # Log a compact manifest into MemoryStore if the new API exists
        self._log_run_manifest(
            run_id=effective_run_id,
            mode="swarm",
            goal=goal,
            domain=effective_domain,
            experiment_mode=experiment_mode,
            run_metadata=run_metadata,
            summaries=all_summaries,
        )

        return all_summaries

    # ------------------------------------------------------------------
    # Multi-agent round helper
    # ------------------------------------------------------------------
    def run_multi_agent_round(
        self,
        goal: str,
        base_cycle_index: int,
        roles: Sequence[str],
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run one multi-agent round: one cycle per role, sharing MemoryStore.

        This is used by swarm continuous mode, but can also be called
        directly if a caller wants to orchestrate a single swarm round.

        Args:
            goal:
                High-level research goal shared by all roles.
            base_cycle_index:
                Global cycle index for the first role in this round.
                Subsequent roles use base_cycle_index + i.
            roles:
                Sequence of role names (for example ["researcher", "critic", ...]).
            source_controls:
                Shared source configuration for all roles.
            pdf_bytes:
                Optional PDF bytes available to each role.
            biomarker_snapshot:
                Optional biomarker snapshot shared across roles.
            domain:
                Optional domain tag such as "general", "longevity", or "math".
            run_id:
                Optional identifier for the enclosing swarm run.
            experiment_mode:
                Optional experiment mode label for this round.

        Returns:
            List of summary dicts, one per role.
        """
        summaries: List[Dict[str, Any]] = []
        effective_domain = domain or "general"
        effective_run_id = run_id or self._generate_run_id()

        for i, role in enumerate(roles):
            ci = base_cycle_index + i
            result = self.run_cycle(
                goal=goal,
                cycle_index=ci,
                role=str(role),
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=effective_domain,
                run_id=effective_run_id,
                experiment_mode=experiment_mode,
            )
            summary = result.get("summary", {})
            summaries.append(summary)

        return summaries

    # ------------------------------------------------------------------
    # Discovery utility methods for workers and external callers
    # ------------------------------------------------------------------
    def prune_memory(
        self,
        min_keep: int = 1000,
        max_drop_fraction: float = 0.3,
    ) -> Dict[str, Any]:
        """Run a memory pruning pass using MemoryPruner.

        This expects the backing MemoryStore to expose a list_entries /
        delete_entries interface either directly or through an adapter.
        If pruning fails, it returns a small summary without raising.
        """
        try:
            # Use profile based overrides if present
            rye_w = 0.5
            rec_w = 0.3
            acc_w = 0.2
            if self._memory_prune_cfg:
                try:
                    rye_w = float(self._memory_prune_cfg.get("rye_weight", rye_w))
                    rec_w = float(self._memory_prune_cfg.get("recency_weight", rec_w))
                    acc_w = float(self._memory_prune_cfg.get("access_weight", acc_w))
                    if "min_keep" in self._memory_prune_cfg and min_keep == 1000:
                        mk = self._memory_prune_cfg.get("min_keep")
                        if isinstance(mk, int) and mk > 0:
                            min_keep = mk
                    if "max_drop_fraction" in self._memory_prune_cfg and max_drop_fraction == 0.3:
                        mdf = self._memory_prune_cfg.get("max_drop_fraction")
                        if isinstance(mdf, (int, float)) and 0.0 <= mdf <= 1.0:
                            max_drop_fraction = float(mdf)
                except Exception:
                    pass

            summary = self.memory_pruner.prune(
                min_keep=min_keep,
                max_drop_fraction=max_drop_fraction,
                rye_weight=rye_w,
                recency_weight=rec_w,
                access_weight=acc_w,
            )
            return summary
        except Exception as exc:
            return {
                "timestamp": time.time(),
                "error": str(exc),
                "dropped": 0,
                "reason": "prune_failed",
            }

    def generate_snapshot(
        self,
        week_number: int,
    ) -> Optional[Path]:
        """Generate a weekly snapshot report using SnapshotGenerator.

        Uses:
            - cycle history from MemoryStore
            - rye_metrics if available
            - hypothesis manager summary
            - memory statistics if provided by MemoryStore
        """
        try:
            history = self.memory_store.get_cycle_history()
            cycle_stats = {
                "cycles_total": len(history),
            }

            rye_stats: Dict[str, Any] = {}
            if _rye_metrics_mod is not None:
                try:
                    if hasattr(_rye_metrics_mod, "rye_summary"):
                        rye_stats = _rye_metrics_mod.rye_summary(history)  # type: ignore[attr-defined]
                except Exception:
                    rye_stats = {}

            hyp_summary = self.hypothesis_manager.summary_strings()

            memory_stats: Dict[str, Any] = {}
            try:
                if hasattr(self.memory_store, "memory_stats"):
                    memory_stats = self.memory_store.memory_stats()  # type: ignore[attr-defined]
            except Exception:
                memory_stats = {}

            path = self.snapshot_generator.generate(
                week_number=week_number,
                cycle_stats=cycle_stats,
                rye_stats=rye_stats,
                hypotheses={
                    "pending": hyp_summary.get("pending", []),
                    "validated": hyp_summary.get("validated", []),
                    "rejected": hyp_summary.get("rejected", []),
                },
                discoveries=[],
                tool_usage={},
                contradictions=[],
                memory_stats=memory_stats,
                extra={},
            )
            return path
        except Exception:
            return None

    def verify_pending_hypotheses(
        self,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """Run verification passes on pending hypotheses.

        This calls the VerificationEngine with the current configuration.
        It is safe to call on a schedule from engine_worker.
        """
        try:
            effective_limit = limit
            if effective_limit is None and self.intelligence_profile:
                ver_cfg = self.intelligence_profile.get("verification", {})
                if isinstance(ver_cfg, dict):
                    bl = ver_cfg.get("batch_limit")
                    if isinstance(bl, int) and bl > 0:
                        effective_limit = bl

            results = self.verification_engine.verify_pending(limit=effective_limit)
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Agent-level training / learning helpers (added)
    # ------------------------------------------------------------------
    def get_learning_status(self) -> Dict[str, Any]:
        """Return a compact view of the agent-level learning state.

        This is useful for the UI or external controllers that want to
        show:

            - whether learning is enabled
            - how much history exists
            - what the last learned suggestions were
            - what the last training burst looked like
        """
        status: Dict[str, Any] = {
            "learning_enabled": self.learning_enabled,
            "min_training_history": self.min_training_history,
            "learned_from_memory": dict(self.learned_from_memory) if self.learned_from_memory else {},
            "training_profile": dict(self.training_profile) if self.training_profile else {},
            "learning_plan": dict(self.learning_plan) if self.learning_plan else {},
        }
        try:
            history = self.memory_store.get_cycle_history()
            status["total_cycles"] = len(history)
        except Exception:
            status["total_cycles"] = None
        return status

    def _update_training_profile_from_summaries(
        self,
        summaries: List[Dict[str, Any]],
        *,
        domain: Optional[str] = None,
        runtime_profile: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> None:
        """Internal helper: derive a training profile snapshot after a burst.

        This consumes the *end state* (history + summaries) and creates
        a compact record in self.training_profile. It does not change
        core behavior or external state.
        """
        profile: Dict[str, Any] = {
            "last_training_at": time.time(),
            "domain": domain or "general",
            "runtime_profile": runtime_profile,
            "experiment_mode": experiment_mode,
            "cycles_in_burst": len(summaries),
        }

        # Simple rollup over this burst
        rye_vals: List[float] = []
        for s in summaries:
            rv = s.get("RYE")
            if isinstance(rv, (int, float)):
                rye_vals.append(float(rv))
        if rye_vals:
            profile["burst_rye_avg"] = sum(rye_vals) / len(rye_vals)
            profile["burst_rye_last"] = rye_vals[-1]

        # Use full history + rye_metrics to attach richer diagnostics if available
        history: List[Dict[str, Any]] = []
        try:
            history = self.memory_store.get_cycle_history()
        except Exception:
            history = []

        diag: Optional[Dict[str, Any]] = None
        option_c: Optional[Dict[str, Any]] = None
        effective_domain = domain or "general"

        if _rye_metrics_mod is not None:
            try:
                if hasattr(_rye_metrics_mod, "build_run_diagnostics"):
                    diag = _rye_metrics_mod.build_run_diagnostics(  # type: ignore[attr-defined]
                        history,
                        domain=effective_domain,
                    )
            except Exception:
                diag = None

            try:
                if hasattr(_rye_metrics_mod, "build_option_c_signature"):
                    option_c = _rye_metrics_mod.build_option_c_signature(  # type: ignore[attr-defined]
                        history,
                        domain=effective_domain,
                        hours_run_so_far=None,
                    )
            except Exception:
                option_c = None

        if diag is not None:
            profile["diagnostics"] = diag
        if option_c is not None:
            profile["option_c_signature"] = option_c

        self.training_profile = profile

    def run_training_burst(
        self,
        goal: str,
        *,
        domain: Optional[str] = None,
        role: str = "agent",
        runtime_profile: Optional[str] = None,
        max_cycles: Optional[int] = None,
        max_minutes: Optional[float] = None,
        stop_rye: Optional[float] = None,
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        experiment_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a short, focused training burst.

        This is where *agent-level* learning happens in practice:
        - It writes new structured knowledge into MemoryStore.
        - It exercises TGRM and tools on a focused goal.
        - It updates self.training_profile and leaves self.learned_from_memory
          available for later calls.

        It does NOT change model weights. It only improves the agent's
        internal state and history.

        Typical use:
            - 1 hour general diagnostic run
            - 1 hour longevity refinement
            - short math-theorem exploration burst
        """
        if not self.learning_enabled:
            # Return empty but do not block the caller
            return []

        effective_domain = domain or "general"
        # Default training runtime profile
        if runtime_profile is None:
            runtime_profile = str(self.config.get("training_runtime_profile", "1_hour"))
            if runtime_profile not in RUNTIME_PROFILES:
                runtime_profile = "1_hour"

        # Default bounds for a "burst": conservative, safe, cheap
        if max_cycles is None:
            # Use profile estimate if present, otherwise a small number
            est_cfg = RUNTIME_PROFILES.get(runtime_profile, {})
            est = est_cfg.get("estimated_cycles")
            if isinstance(est, (int, float)):
                max_cycles = max(10, int(est // 2))
            else:
                max_cycles = 40

        if max_minutes is None:
            if runtime_profile == "1_hour":
                max_minutes = 60.0
            elif runtime_profile == "8_hours":
                max_minutes = 4 * 60.0
            else:
                # Generic small burst
                max_minutes = 60.0

        if experiment_mode is None:
            experiment_mode = "training"

        summaries = self.run_continuous(
            goal=goal,
            max_cycles=max_cycles,
            stop_rye=stop_rye,
            role=role,
            source_controls=source_controls,
            pdf_bytes=pdf_bytes,
            biomarker_snapshot=biomarker_snapshot,
            domain=effective_domain,
            max_minutes=max_minutes,
            forever=False,
            resume_from_checkpoint=False,
            runtime_profile=runtime_profile,
            run_id=None,
            experiment_mode=experiment_mode,
        )

        # After the burst, update training profile from full history + this burst
        self._update_training_profile_from_summaries(
            summaries,
            domain=effective_domain,
            runtime_profile=runtime_profile,
            experiment_mode=experiment_mode,
        )

        return summaries

    def optimize_learning_pipeline(
        self,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Derive a high-level learning plan from existing history.

        This is the "Nova, optimize my learning pipeline" entry point.

        It:
            - calls learn_from_memory(domain)
            - optionally calls rye_metrics Option C for richer diagnostics
            - proposes simple, concrete next actions (as text) without
              mutating core behavior.

        It is safe to call from the UI or a controller.
        """
        plan: Dict[str, Any] = {
            "domain": domain or "general",
            "learning_enabled": self.learning_enabled,
            "actions": [],
            "learned_from_memory": {},
            "option_c_signature": None,
        }

        if not self.learning_enabled:
            plan["actions"].append("Learning is disabled in config (learning_enabled=False). Enable it to train the agent.")
            self.learning_plan = plan
            return plan

        # Step 1: pull learned hints from history
        learned = self.learn_from_memory(domain=domain)
        plan["learned_from_memory"] = dict(learned)

        # Step 2: Option C signature if available (full run self-diagnosis)
        option_c = None
        if _rye_metrics_mod is not None and hasattr(_rye_metrics_mod, "build_option_c_signature"):
            try:
                history = self.memory_store.get_cycle_history()
                if history:
                    option_c = _rye_metrics_mod.build_option_c_signature(  # type: ignore[attr-defined]
                        history,
                        domain=domain or "general",
                        hours_run_so_far=None,
                    )
            except Exception:
                option_c = None
        plan["option_c_signature"] = option_c

        # Step 3: propose next actions (text only, no hard changes)
        actions: List[str] = []

        has_history = bool(learned.get("has_history"))
        rec_profile = learned.get("recommended_runtime_profile")
        rec_stop = learned.get("recommended_stop_rye")

        if not has_history:
            actions.append(
                "Run at least one short training burst (for example 1-hour general or longevity) "
                "to build initial cycle history before auto-tuning."
            )
        else:
            if isinstance(rec_profile, str):
                actions.append(
                    f"Use runtime profile '{rec_profile}' for the next training burst in domain '{plan['domain']}'."
                )
            if isinstance(rec_stop, (int, float)):
                actions.append(
                    f"Start with a conservative RYE stop threshold around {rec_stop:.3f} for diagnostic runs."
                )

        if option_c:
            try:
                tier = option_c.get("run_tier", {}).get("tier")
                env_state = option_c.get("autonomy_safety_envelope", {}).get("state")
                if tier:
                    actions.append(f"Current run tier according to Option C: {tier}.")
                if env_state:
                    actions.append(f"Autonomy–stability safety envelope is classified as: {env_state}.")
            except Exception:
                pass

        # Generic training hygiene suggestions (agent-level learning)
        actions.append(
            "Use run_training_burst(...) with focused goals (for example one clear longevity question) "
            "to teach the agent on compact, high-signal problems."
        )
        actions.append(
            "Periodically call prune_memory(...) to keep MemoryStore healthy and avoid dilution of high-value traces."
        )
        actions.append(
            "After major bursts, regenerate snapshots with generate_snapshot(...) so you can track skill buildup over weeks."
        )

        plan["actions"] = actions
        self.learning_plan = plan
        return plan
