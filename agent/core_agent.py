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
    - Verify : re test, compute delta_R and RYE, and log the cycle

RYE (Repair Yield per Energy):
    For each cycle, TGRMLoop computes delta_R / E (improvement divided
    by effort). CoreAgent orchestrates cycles, multi agent runs, and
    continuous modes while the TGRM loop integrates tool usage into E.

MSIL v2 hooks (dual track inference):
    CoreAgent supports a configurable `msil_mode` and `msil_track_mode`
    that can be passed into upgraded TGRM loops:
        - msil_mode: "v1" or "v2"
        - msil_track_mode: "single" or "dual"
    If an MSIL controller is available, CoreAgent will attach it as
    `self.msil_controller` and expose advisory helpers.

TGRM v3 hooks (multi goal self repair):
    CoreAgent exposes `run_multi_goal_cycle` and
    `run_multi_goal_training_burst` which call
    `TGRMLoop.run_multi_goal_cycle(...)` when available, and otherwise
    fall back to running goals one by one. This sets the stage for
    true multi hypothesis repair in the loop.

Stability Kernel:
    If a StabilityKernel implementation is available, CoreAgent
    attaches it as `self.stability_kernel` and feeds each cycle summary
    into it. When the kernel recommends a halt, redirection, or slowdown,
    CoreAgent will stop continuous runs with a "stability_guard" reason.

Autonomous literature crawler:
    If a LiteratureCrawler is available, CoreAgent exposes
    `run_literature_crawl(...)` that can crawl PubMed, preprints, and
    other literature sources, log findings into MemoryStore, and feed
    DiscoveryLogger and HypothesisManager.

Hierarchical swarm architecture:
    CoreAgent still uses a flat list of roles, but if a SwarmProfileManager
    is available, it can interpret profile names ("lab_style", "validator_heavy")
    and translate them into structured role sets through
    `apply_swarm_profile(...)`. run_swarm_continuous can be called with
    an explicit profile and roles.

RYE v4 style signal decomposition:
    CoreAgent exposes helpers that can call richer rye_metrics functions
    such as Option C signatures and any future v4 metrics, then attach
    these to training_profile and learning plans. The underlying math
    lives in rye_metrics.

Protocol Synthesizer:
    If a ProtocolSynthesizer is available, CoreAgent attaches it and
    exposes `synthesize_protocols_from_validated(...)` which turns
    validated hypotheses and RYE diagnostics into candidate protocols
    and supplement stacks.

Meta Agent for self tuning:
    If a MetaAgent is available, CoreAgent attaches it as
    `self.meta_agent` and exposes `run_meta_tuning(...)` that produces
    advisory suggestions for prompts, roles, swarm shape, and thresholds
    based on learning_plan and meta_state.

Memory partition v2:
    CoreAgent can enable memory tiers:
        - Gold Memory
        - Silver Memory
        - Bronze Memory
        - Dustbin Memory
    When `memory_tiers_enabled` is True and MemoryStore supports
    `configure_tiers(...)`, CoreAgent configures tier labels and exposes
    `get_memory_partition_status()` for quick tier counts.

True long horizon run modes:
    On top of `run_continuous` and `run_swarm_continuous`, CoreAgent
    exposes:
        - run_long_horizon_single(...)
        - run_long_horizon_swarm(...)
    which are convenient wrappers for 24 hour to multi day runs with
    stability guarding and RYE diagnostics plugged in.

Breakthrough Hunter mode:
    CoreAgent supports `breakthrough_hunter_enabled` in config. When on,
    default experiment_mode is "breakthrough" for continuous and swarm
    runs, and RYE thresholds or discovery instrumentation can be tuned
    more aggressively for Tier 3 discovery probability.

Integrated trading mode:
    If a TradingEngine is available, CoreAgent attaches it and exposes:
        - run_trading_session(...)
        - run_trading_swarm(...)
    which are domain "trading" runs coupled with any trading specific
    logic in the TradingEngine while still using TGRM and RYE.

Multi agent extension:
    CoreAgent can orchestrate multiple logical agent roles
    (researcher, critic, planner, synthesizer, explorer, plus up to
    a total of 32 agents) over the same MemoryStore. This makes it
    possible to run:
        - one cycle per role (multi agent round), or
        - long continuous runs where each "round" consists of many roles.

    The maximum number of logical agents is capped at 32 for safety,
    and can be configured via config["max_agents"] (1 to 32).

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
        - auto apply a stop_rye threshold where appropriate

Domain presets and experiment modes:
    CoreAgent also uses domain presets (via presets.get_preset) to:
        - pick sensible default runtime profiles per domain
        - suggest RYE stop thresholds
        - keep behavior aligned with longevity, math, general, or trading modes

    The experiment_mode (for example discovery, stability_test,
    protocol_builder, breakthrough, trading) can be passed in and
    tagged on runs so that the meta controller and report generator
    can distinguish different runs.

Discovery extensions:
    CoreAgent can optionally cooperate with:
        - DiscoveryLogger for RYE spikes, hypotheses, and contradictions
        - HypothesisManager for a pending, validated, rejected pipeline
        - SnapshotGenerator for weekly experiment summaries
        - MemoryPruner for long run memory health
        - VerificationEngine for deeper checks of promising hypotheses
        - DiscoveryEngine (if available) for tiered discovery scoring

    These helpers sit on top of the existing TGRM loop and do not
    change its core behavior. They only consume the summary dicts and
    write logs or hypothesis records.

Learning from memory history:
    CoreAgent can analyze past cycle history (via MemoryStore and
    rye_metrics) and derive a "learned" profile that auto tunes:
        - suggested runtime profile (1 hour, 8 hours, 24 hours)
        - suggested stop_rye thresholds
        - diagnostics snapshot for the UI

    This learning is optional, conservative, and backward compatible:
        - If rye_metrics is unavailable, the feature is a no op.
        - If config["auto_learn_from_memory"] is False, it is disabled.

Ultra speed profiles:
    CoreAgent also supports speed modes for learning optimization:
        - "standard": original behavior
        - "accelerated": tighter cycles and more aggressive use of
          learned runtime profiles
        - "ultra": prepared for aggressive speculative cycles plus a
          verification heavy pipeline in TGRMLoop and related modules

    Speed modes are exposed via config["speed_mode"] and do not break
    existing callers. When other modules are upgraded, they can read
    this mode to coordinate ultra fast, high accuracy runs.

Citation tracking:
    When citations are enabled, CoreAgent will propagate any citations
    returned by TGRMLoop into per cycle summaries, run state, and
    continuous run metadata so that the UI and report generator can
    display clear source provenance.
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

# Optional advanced controllers and kernels
# MSIL: support both msil_controller.py and msil.py (MetaSkillIntelligenceLayer)
try:
    # Preferred controller wrapper if it exists
    from .msil_controller import MSILController  # type: ignore[attr-defined]
except Exception:
    try:
        # Backward compatible: use MetaSkillIntelligenceLayer from msil.py
        from .msil import MetaSkillIntelligenceLayer as MSILController  # type: ignore[attr-defined]
    except Exception:
        MSILController = None  # type: ignore

try:
    from .stability_kernel import StabilityKernel  # type: ignore[attr-defined]
except Exception:
    StabilityKernel = None  # type: ignore

try:
    from .literature_crawler import LiteratureCrawler  # type: ignore[attr-defined]
except Exception:
    LiteratureCrawler = None  # type: ignore

try:
    from .protocol_synthesizer import ProtocolSynthesizer  # type: ignore[attr-defined]
except Exception:
    ProtocolSynthesizer = None  # type: ignore

try:
    from .meta_agent import MetaAgent  # type: ignore[attr-defined]
except Exception:
    MetaAgent = None  # type: ignore

try:
    from .swarm_profiles import SwarmProfileManager  # type: ignore[attr-defined]
except Exception:
    SwarmProfileManager = None  # type: ignore

try:
    from .trading_engine import TradingEngine  # type: ignore[attr-defined]
except Exception:
    TradingEngine = None  # type: ignore

# Optional SwarmOrchestrator engine
try:
    from .swarm_orchestrator import SwarmOrchestrator  # type: ignore[attr-defined]
except Exception:
    SwarmOrchestrator = None  # type: ignore

# Discovery stack imports
from .discovery_log import DiscoveryLogger, get_global_logger
from .hypothesis_manager import HypothesisManager
from .memory_pruner import MemoryPruner
from .snapshot_generator import SnapshotGenerator
from .verification_engine import VerificationEngine


class CoreAgent:
    """High level controller for the autonomous research agent.

    This class wires UI or CLI controls (multi agent, continuous mode,
    source preferences, PDF uploads, swarm roles, advanced upgrades)
    into the lower level TGRMLoop which performs the actual reparodynamic
    TGRM cycles and RYE computation.

    It also owns a tools layer that TGRMLoop can use for:
        - headless browser actions
        - code execution sandbox
        - data pipelines (CSV, Excel, SQLite)
        - registry based tools via TOOL_REGISTRY

    Speed modes:
        - standard: baseline behavior
        - accelerated: advisory auto tuning is more active
        - ultra: advisory auto tuning plus extra meta statistics
                 intended for upgraded TGRM and swarm modules

    Advanced hooks:
        - msil_mode and msil_track_mode for MSIL v2
        - stability_kernel for long horizon stability
        - literature_crawler for autonomous reading
        - protocol_synthesizer for stack and protocol building
        - meta_agent for self tuning
        - swarm_profile_manager for hierarchical swarms
        - trading_engine for trading runs
        - memory tiers for Gold, Silver, Bronze, Dustbin
        - breakthrough_hunter_enabled for discovery focused runs
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

        # Prefer TOOL_REGISTRY (so Tavily + mail tools are visible),
        # and expose legacy Toolbelt under a namespaced key if present.
        if self.tool_registry:
            self.tools: Any = dict(self.tool_registry)
            if Toolbelt is not None:
                try:
                    legacy_toolbelt = Toolbelt()
                    self.tools.setdefault("legacy_toolbelt", legacy_toolbelt)
                except Exception:
                    pass
        elif Toolbelt is not None:
            try:
                self.tools = Toolbelt()
            except Exception:
                self.tools = {}
        else:
            # TGRMLoop will receive a plain dict of tool callables
            self.tools = {}

        # ------------------------------------------------------------------
        # Swarm Orchestrator (hierarchical swarm engine on top of CoreAgent)
        # ------------------------------------------------------------------
        self.swarm_orchestrator: Optional[Any] = None
        if SwarmOrchestrator is not None:
            try:
                self.swarm_orchestrator = SwarmOrchestrator(
                    agent_fn=self._run_swarm_agent_payload,
                    max_workers=int(self.config.get("swarm_max_workers", 32)),
                    max_swarm_size=int(self.config.get("swarm_max_size", 64)),
                    default_mode=str(self.config.get("swarm_default_mode", "burst")),
                    default_domain=str(self.config.get("default_domain", "general")),
                    enable_cross_domain_bridge=bool(self.config.get("swarm_cross_domain_bridge", True)),
                    enable_diagnostics=bool(self.config.get("swarm_diagnostics_enabled", True)),
                    enable_msil=bool(self.config.get("swarm_msil_enabled", True)),
                    enable_stability_kernel=bool(self.config.get("swarm_stability_enabled", True)),
                    enable_discovery_manager=bool(self.config.get("swarm_discovery_enabled", True)),
                    enable_protocol_synthesizer=bool(self.config.get("swarm_protocols_enabled", True)),
                    record_history=bool(self.config.get("swarm_history_enabled", True)),
                )
            except Exception:
                self.swarm_orchestrator = None

        # ------------------------------------------------------------------
        # MSIL controller and modes (v1 vs v2, single vs dual track)
        # ------------------------------------------------------------------
        self.msil_mode: str = str(self.config.get("msil_mode", "v1")).lower()
        if self.msil_mode not in {"v1", "v2"}:
            self.msil_mode = "v1"

        self.msil_track_mode: str = str(self.config.get("msil_track_mode", "single")).lower()
        if self.msil_track_mode not in {"single", "dual"}:
            self.msil_track_mode = "single"

        # Flexible MSIL initialization: supports both MSILController and MetaSkillIntelligenceLayer
        if MSILController is not None:
            msil_instance: Optional[Any] = None
            try:
                # Preferred simple signature
                msil_instance = MSILController(config=self.config)
            except TypeError:
                # Fallback for MetaSkillIntelligenceLayer style constructor
                try:
                    msil_instance = MSILController(
                        memory_store=self.memory_store,
                        config=self.config,
                        rye_metrics_module=_rye_metrics_mod,
                    )
                except Exception:
                    msil_instance = None
            except Exception:
                msil_instance = None
            self.msil_controller = msil_instance
        else:
            self.msil_controller = None

        # Underlying TGRM loop that actually performs Test, Detect, Repair, Verify.
        # Prefer the newer signature that accepts tools=..., but keep a
        # backwards compatible fallback for older TGRMLoop implementations.
        try:
            self.tgrm_loop = TGRMLoop(
                self.memory_store,
                self.config,
                tools=self.tools,  # type: ignore[call-arg]
            )
        except TypeError:
            self.tgrm_loop = TGRMLoop(self.memory_store, self.config)

        # Source controls can be set by the UI at runtime.
        self.source_controls: Dict[str, bool] = {}

        # Optional attached PDF bytes (from an uploaded file in the UI)
        self._attached_pdf_bytes: Optional[bytes] = None

        # Checkpoint file for continuous runs
        checkpoint_default = "logs/continuous_checkpoint.json"
        self.checkpoint_path = Path(self.config.get("checkpoint_file", checkpoint_default))

        # Whether continuous runs should try to auto resume from checkpoint
        self.auto_resume_enabled: bool = bool(self.config.get("auto_resume_enabled", True))

        # ------------------------------------------------------------------
        # Multi agent configuration (up to 32 logical agents)
        # ------------------------------------------------------------------
        self.max_agents: int = int(self.config.get("max_agents", 32))
        if self.max_agents < 1:
            self.max_agents = 1
        if self.max_agents > 32:
            self.max_agents = 32

        base_roles: List[str] = [
            "researcher",
            "critic",
            "planner",
            "synthesizer",
            "explorer",
        ]

        generated_roles: List[str] = list(base_roles)
        counter = 1
        while len(generated_roles) < self.max_agents:
            generated_roles.append(f"agent_{counter:02d}")
            counter += 1

        config_roles = self.config.get("agent_roles")
        if isinstance(config_roles, (list, tuple)):
            roles: List[str] = [str(r) for r in config_roles if r]
            if not roles:
                roles = generated_roles
        else:
            roles = generated_roles

        self.agent_roles: List[str] = roles[: self.max_agents]

        # ------------------------------------------------------------------
        # Discovery stack (sits on top of existing logic)
        # ------------------------------------------------------------------
        self._last_rye: Optional[float] = None

        self.discovery_logger: DiscoveryLogger = get_global_logger()
        self.hypothesis_manager: HypothesisManager = HypothesisManager(run_id=None)

        self.memory_pruner: MemoryPruner = MemoryPruner(self.memory_store, run_id=None)
        self.snapshot_generator: SnapshotGenerator = SnapshotGenerator(run_id=None)

        self.verification_engine: VerificationEngine = VerificationEngine(
            hypothesis_manager=self.hypothesis_manager,
            discovery_logger=self.discovery_logger,
            literature_check_fn=None,
            critique_fn=None,
            data_check_fn=None,
            run_id=None,
        )

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
        # Stability Kernel, Literature Crawler, Protocol Synthesizer,
        # Meta Agent, Swarm Profiles, Trading Engine
        # ------------------------------------------------------------------
        if StabilityKernel is not None:
            try:
                self.stability_kernel = StabilityKernel(config=self.config)
            except Exception:
                self.stability_kernel = None
        else:
            self.stability_kernel = None

        if LiteratureCrawler is not None:
            try:
                self.literature_crawler = LiteratureCrawler(
                    memory_store=self.memory_store,
                    discovery_logger=self.discovery_logger,
                    config=self.config,
                )
            except Exception:
                self.literature_crawler = None
        else:
            self.literature_crawler = None

        if ProtocolSynthesizer is not None:
            try:
                self.protocol_synthesizer = ProtocolSynthesizer(
                    memory_store=self.memory_store,
                    hypothesis_manager=self.hypothesis_manager,
                    verification_engine=self.verification_engine,
                    config=self.config,
                )
            except Exception:
                self.protocol_synthesizer = None
        else:
            self.protocol_synthesizer = None

        if MetaAgent is not None:
            try:
                self.meta_agent = MetaAgent(
                    memory_store=self.memory_store,
                    config=self.config,
                )
            except Exception:
                self.meta_agent = None
        else:
            self.meta_agent = None

        if SwarmProfileManager is not None:
            try:
                self.swarm_profile_manager = SwarmProfileManager(config=self.config)
            except Exception:
                self.swarm_profile_manager = None
        else:
            self.swarm_profile_manager = None

        if TradingEngine is not None:
            try:
                self.trading_engine = TradingEngine(
                    memory_store=self.memory_store,
                    discovery_logger=self.discovery_logger,
                    config=self.config,
                )
            except Exception:
                self.trading_engine = None
        else:
            self.trading_engine = None

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
        self.learned_from_memory: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Agent level training configuration
        # ------------------------------------------------------------------
        self.learning_enabled: bool = bool(self.config.get("learning_enabled", True))
        self.min_training_history: int = int(self.config.get("min_training_history", 50))
        if self.min_training_history < 0:
            self.min_training_history = 0

        self.training_profile: Dict[str, Any] = {}
        self.learning_plan: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Citation tracking configuration
        # ------------------------------------------------------------------
        self.citations_enabled: bool = bool(self.config.get("citations_enabled", True))

        # ------------------------------------------------------------------
        # Speed, Breakthrough Hunter, RYE mode
        # ------------------------------------------------------------------
        self.speed_mode: str = str(self.config.get("speed_mode", "standard")).lower()
        if self.speed_mode not in {"standard", "accelerated", "ultra"}:
            self.speed_mode = "standard"

        self.breakthrough_hunter_enabled: bool = bool(self.config.get("breakthrough_hunter_enabled", False))

        self.rye_mode: str = str(self.config.get("rye_mode", "v3")).lower()
        if self.rye_mode not in {"v2", "v3", "v4"}:
            self.rye_mode = "v3"

        self._meta_state: Dict[str, Any] = {
            "speed_mode": self.speed_mode,
            "last_update_ts": time.time(),
            "single_run_stats": {
                "recent_rye": [],
                "recent_contradictions": [],
            },
            "swarm_run_stats": {
                "recent_round_rye": [],
                "recent_round_contradictions": [],
            },
        }

        # ------------------------------------------------------------------
        # Memory tiers (Gold, Silver, Bronze, Dustbin)
        # ------------------------------------------------------------------
        self.memory_tiers_enabled: bool = bool(self.config.get("memory_tiers_enabled", False))
        self.memory_tiers: List[str] = ["gold", "silver", "bronze", "dustbin"]
        if self.memory_tiers_enabled:
            try:
                if hasattr(self.memory_store, "configure_tiers"):
                    self.memory_store.configure_tiers(self.memory_tiers)  # type: ignore[attr-defined]
            except Exception:
                pass

        # Last meta tuning suggestions
        self.last_meta_tuning: Dict[str, Any] = {}

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
    # Tools inspection helpers
    # ------------------------------------------------------------------
    def get_tool_registry(self) -> Dict[str, Any]:
        """Return a shallow copy of the static TOOL_REGISTRY snapshot."""
        return dict(self.tool_registry)

    def get_tools_status(self) -> Dict[str, Any]:
        """Return a compact view of tool layer status for UI panels."""
        has_legacy_toolbelt = False
        try:
            if Toolbelt is not None:
                if isinstance(self.tools, dict) and "legacy_toolbelt" in self.tools:
                    has_legacy_toolbelt = True
                elif isinstance(self.tools, Toolbelt):
                    has_legacy_toolbelt = True
        except Exception:
            has_legacy_toolbelt = False

        status: Dict[str, Any] = {
            "has_registry": bool(self.tool_registry),
            "registry_keys": list(self.tool_registry.keys()),
            "has_legacy_toolbelt": has_legacy_toolbelt,
            "tools_type": type(self.tools).__name__,
        }
        return status

    # ------------------------------------------------------------------
    # Source controls helpers
    # ------------------------------------------------------------------
    def set_source_controls(self, controls: Dict[str, bool]) -> None:
        """Replace the default source controls with a validated copy."""
        if not isinstance(controls, dict):
            return
        cleaned: Dict[str, bool] = {}
        for k, v in controls.items():
            key = str(k)
            if isinstance(v, bool):
                cleaned[key] = v
        self.source_controls = cleaned

    def update_source_controls(self, controls: Dict[str, bool]) -> None:
        """Update existing source controls with a partial override dict."""
        if not isinstance(controls, dict):
            return
        current = dict(self.source_controls)
        for k, v in controls.items():
            key = str(k)
            if isinstance(v, bool):
                current[key] = v
        self.source_controls = current

    def get_source_controls(self) -> Dict[str, bool]:
        """Return a shallow copy of the current default source controls."""
        return dict(self.source_controls)

    # ------------------------------------------------------------------
    # Speed mode and meta helpers
    # ------------------------------------------------------------------
    def get_speed_mode(self) -> str:
        """Return current speed mode (standard, accelerated, ultra)."""
        return self.speed_mode

    def set_speed_mode(self, mode: str) -> None:
        """Update speed mode at runtime in a safe and conservative way."""
        m = str(mode).lower()
        if m not in {"standard", "accelerated", "ultra"}:
            return
        self.speed_mode = m
        self._meta_state["speed_mode"] = m
        self._meta_state["last_update_ts"] = time.time()

    def _meta_note_single_cycle(self, rye_value: Optional[float], contradictions: Optional[int]) -> None:
        """Track recent single agent RYE and contradictions for meta tuning."""
        try:
            stats = self._meta_state.get("single_run_stats") or {}
            recent_rye = stats.get("recent_rye") or []
            recent_contra = stats.get("recent_contradictions") or []

            if isinstance(rye_value, (int, float)):
                recent_rye.append(float(rye_value))
                if len(recent_rye) > 20:
                    recent_rye.pop(0)

            if isinstance(contradictions, (int, float)):
                recent_contra.append(float(contradictions))
                if len(recent_contra) > 20:
                    recent_contra.pop(0)

            stats["recent_rye"] = recent_rye
            stats["recent_contradictions"] = recent_contra
            self._meta_state["single_run_stats"] = stats
            self._meta_state["last_update_ts"] = time.time()
        except Exception:
            return

    def _meta_note_swarm_round(self, avg_round_rye: Optional[float], avg_round_contradictions: Optional[float]) -> None:
        """Track swarm round RYE and contradictions."""
        try:
            stats = self._meta_state.get("swarm_run_stats") or {}
            recent_rye = stats.get("recent_round_rye") or []
            recent_contra = stats.get("recent_round_contradictions") or []

            if isinstance(avg_round_rye, (int, float)):
                recent_rye.append(float(avg_round_rye))
                if len(recent_rye) > 20:
                    recent_rye.pop(0)

            if isinstance(avg_round_contradictions, (int, float)):
                recent_contra.append(float(avg_round_contradictions))
                if len(recent_contra) > 20:
                    recent_contra.pop(0)

            stats["recent_round_rye"] = recent_rye
            stats["recent_round_contradictions"] = recent_contra
            self._meta_state["swarm_run_stats"] = stats
            self._meta_state["last_update_ts"] = time.time()
        except Exception:
            return

    # ------------------------------------------------------------------
    # Learning from memory history
    # ------------------------------------------------------------------
    def learn_from_memory(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Analyze past cycles and derive learned runtime hints."""
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

        recommended_profile: Optional[str] = None
        recommended_stop_rye: Optional[float] = None

        if isinstance(diagnostics, dict):
            avg = diagnostics.get("rye_avg")
            stab = diagnostics.get("stability_index")
            slope = diagnostics.get("trend_slope")
            mid = diagnostics.get("mid_percentile")

            if isinstance(avg, (int, float)) and isinstance(stab, (int, float)):
                if avg >= 0.5 and stab >= 0.6:
                    recommended_profile = "24_hours"
                elif avg >= 0.2 and stab >= 0.3:
                    recommended_profile = "8_hours"
                else:
                    recommended_profile = "1_hour"

            if isinstance(mid, (int, float)):
                recommended_stop_rye = float(mid) * 0.5

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
        """Send a compact run manifest into MemoryStore if supported."""
        manifest: Dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "goal": goal,
            "domain": domain,
            "experiment_mode": experiment_mode,
            "run_metadata": dict(run_metadata),
            "summaries": summaries,
        }
        try:
            if hasattr(self.memory_store, "log_run_manifest"):
                try:
                    self.memory_store.log_run_manifest(run_id, manifest)  # type: ignore[attr-defined]
                except TypeError:
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
        """Inspect a cycle summary and log discovery related signals."""
        if self.discovery_engine is not None:
            try:
                context = {
                    "cycle_index": cycle_index,
                    "role": role,
                    "domain": domain,
                    "run_id": run_id,
                    "experiment_mode": experiment_mode,
                }
                self.discovery_engine.process_cycle(summary, context=context)  # type: ignore[attr-defined]
            except Exception:
                pass

        try:
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
            pass

        # 2) Hypothesis registration
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

    def get_discovery_status(self) -> Dict[str, Any]:
        """Return a compact view of discovery and hypothesis state."""
        status: Dict[str, Any] = {
            "last_rye": self._last_rye,
            "pending_hypotheses": None,
            "validated_hypotheses": None,
            "rejected_hypotheses": None,
        }
        try:
            hyp_summary = self.hypothesis_manager.summary_strings()
            if isinstance(hyp_summary, dict):
                pending = hyp_summary.get("pending", []) or []
                validated = hyp_summary.get("validated", []) or []
                rejected = hyp_summary.get("rejected", []) or []
                status["pending_hypotheses"] = len(pending)
                status["validated_hypotheses"] = len(validated)
                status["rejected_hypotheses"] = len(rejected)
        except Exception:
            pass

        try:
            if hasattr(self.discovery_logger, "run_id"):
                status["current_run_id"] = getattr(self.discovery_logger, "run_id", None)
        except Exception:
            pass

        return status

    # ------------------------------------------------------------------
    # Stability Kernel and memory tiers status
    # ------------------------------------------------------------------
    def get_stability_status(self) -> Dict[str, Any]:
        """Return stability kernel status if available."""
        if self.stability_kernel is None:
            return {"enabled": False}
        try:
            status = self.stability_kernel.get_status()
            if not isinstance(status, dict):
                status = {}
        except Exception:
            status = {}
        status["enabled"] = True
        return status

    def _apply_stability_kernel(
        self,
        summary: Dict[str, Any],
        *,
        mode: str,
        goal: str,
        domain: str,
        run_id: str,
        experiment_mode: Optional[str],
        elapsed_minutes: float,
    ) -> Optional[Dict[str, Any]]:
        """Feed summary into StabilityKernel and handle any guard signals."""
        if self.stability_kernel is None:
            return None
        try:
            context = {
                "mode": mode,
                "goal": goal,
                "domain": domain,
                "run_id": run_id,
                "experiment_mode": experiment_mode,
                "elapsed_minutes": elapsed_minutes,
                "speed_mode": self.speed_mode,
            }
            self.stability_kernel.observe_cycle(summary, context=context)
            action = self.stability_kernel.recommend_action()
            if isinstance(action, dict):
                return action
        except Exception:
            return None
        return None

    def get_memory_partition_status(self) -> Dict[str, Any]:
        """Return tier level memory statistics if available."""
        status: Dict[str, Any] = {
            "tiers_enabled": self.memory_tiers_enabled,
            "tiers": list(self.memory_tiers),
            "counts": {},
        }
        try:
            if hasattr(self.memory_store, "tier_counts"):
                counts = self.memory_store.tier_counts()  # type: ignore[attr-defined]
                if isinstance(counts, dict):
                    status["counts"] = counts
        except Exception:
            pass
        return status

    # ------------------------------------------------------------------
    # Multi agent utilities
    # ------------------------------------------------------------------
    def get_agent_roles(self) -> List[str]:
        """Return the configured logical agent roles (capped at max_agents)."""
        return list(self.agent_roles[: self.max_agents])

    def get_swarm_status(self) -> Dict[str, Any]:
        """Return a compact view of swarm configuration and recent stats."""
        status: Dict[str, Any] = {
            "max_agents": self.max_agents,
            "agent_roles": list(self.agent_roles[: self.max_agents]),
            "speed_mode": self.speed_mode,
            "recent_round_rye": [],
        }
        try:
            swarm_stats = self._meta_state.get("swarm_run_stats") or {}
            status["recent_round_rye"] = swarm_stats.get("recent_round_rye", []) or []
        except Exception:
            pass
        return status

    def apply_swarm_profile(self, profile_name: str) -> List[str]:
        """Apply a swarm profile to agent roles if a manager is available."""
        if self.swarm_profile_manager is None:
            return self.get_agent_roles()
        try:
            roles = self.swarm_profile_manager.resolve_roles(profile_name)
            if isinstance(roles, (list, tuple)):
                resolved = [str(r) for r in roles][: self.max_agents]
                if resolved:
                    self.agent_roles = resolved
            return self.get_agent_roles()
        except Exception:
            return self.get_agent_roles()

    # ------------------------------------------------------------------
    # Swarm Orchestrator helpers
    # ------------------------------------------------------------------
    def get_swarm_orchestrator_status(self) -> Dict[str, Any]:
        """Return status for the SwarmOrchestrator engine."""
        if self.swarm_orchestrator is None:
            return {
                "enabled": False,
                "reason": "not_initialized",
            }
        try:
            return {
                "enabled": True,
                "max_workers": self.swarm_orchestrator.max_workers,
                "max_swarm_size": self.swarm_orchestrator.max_swarm_size,
                "default_mode": self.swarm_orchestrator.default_mode,
                "default_domain": self.swarm_orchestrator.default_domain,
                "roles": [r.name for r in getattr(self.swarm_orchestrator, "roles", [])],
                "history_length": len(getattr(self.swarm_orchestrator, "history", [])),
            }
        except Exception:
            return {
                "enabled": True,
            }

    def _run_swarm_agent_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter so SwarmOrchestrator can call back into CoreAgent.

        Each swarm agent runs a compact continuous burst using the normal
        CoreAgent run_continuous path, then returns a summary and basic
        diagnostics for the orchestrator.
        """
        try:
            goal = str(payload.get("goal", "")).strip()
            role = str(payload.get("role", "agent"))
            domain = str(payload.get("domain", "general"))

            max_cycles_raw = payload.get("max_cycles", 8)
            try:
                max_cycles = int(max_cycles_raw)
            except Exception:
                max_cycles = 8
            if max_cycles <= 0:
                max_cycles = 1

            max_minutes_raw = payload.get("max_minutes")
            if isinstance(max_minutes_raw, (int, float)):
                max_minutes = float(max_minutes_raw)
                if max_minutes <= 0:
                    max_minutes = None
            else:
                max_minutes = None

            experiment_mode = payload.get("experiment_mode")
            source_controls = payload.get("source_controls")
            pdf_bytes = payload.get("pdf_bytes")
            biomarker_snapshot = payload.get("biomarker_snapshot")
            runtime_profile = payload.get("runtime_profile")

            stop_rye_raw = payload.get("stop_rye")
            stop_rye = float(stop_rye_raw) if isinstance(stop_rye_raw, (int, float)) else None

            summaries = self.run_continuous(
                goal=goal,
                max_cycles=max_cycles,
                stop_rye=stop_rye,
                role=role,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
                max_minutes=max_minutes,
                forever=False,
                resume_from_checkpoint=False,
                runtime_profile=runtime_profile,
                run_id=payload.get("run_id"),
                experiment_mode=experiment_mode,
            )

            summary = summaries[-1] if summaries else {}

            diagnostics: Dict[str, Any] = {}
            rye_diag = summary.get("rye_diagnostics") or summary.get("rye")
            if isinstance(rye_diag, dict):
                diagnostics["rye"] = rye_diag

            result: Dict[str, Any] = {
                "agent_index": payload.get("agent_index"),
                "agent_id": payload.get("agent_id"),
                "role": role,
                "domain": domain,
                "status": "ok",
                "summary": summary,
                "diagnostics": diagnostics,
            }

            trace = summary.get("rye_trace") or summary.get("RYE_trace")
            if trace is not None:
                result["rye_trace"] = trace

            energy = summary.get("Energy") or summary.get("energy") or summary.get("cost_energy")
            if isinstance(energy, (int, float)):
                result["energy"] = float(energy)

            tokens = summary.get("total_tokens") or summary.get("tokens_used")
            if isinstance(tokens, (int, float)):
                result["tokens_used"] = float(tokens)

            if "citations" in summary:
                result["citations"] = summary["citations"]

            return result

        except Exception as exc:
            return {
                "agent_index": payload.get("agent_index"),
                "agent_id": payload.get("agent_id"),
                "role": payload.get("role", "agent"),
                "domain": payload.get("domain", "general"),
                "status": "error",
                "error": str(exc),
            }

    def run_orchestrated_swarm(
        self,
        goal: str,
        *,
        base_preset: Dict[str, Any],
        swarm_size: int = 4,
        max_cycles: int = 6,
        mode: Optional[str] = None,
        domain: Optional[str] = None,
        curriculum_id: Optional[str] = None,
        profile_name: Optional[str] = None,
        seed: Optional[int] = None,
        global_context: Optional[Dict[str, Any]] = None,
        per_agent_overrides: Optional[Sequence[Dict[str, Any]]] = None,
        timeout_s: Optional[float] = None,
        return_agent_payloads: bool = False,
        extra_tags: Optional[Dict[str, Any]] = None,
        **agent_extra: Any,
    ) -> Dict[str, Any]:
        """Run a single orchestrated swarm round on top of CoreAgent."""
        if self.swarm_orchestrator is None:
            return {
                "enabled": False,
                "reason": "swarm_orchestrator_not_available",
            }

        # Defensive clamp for swarm_size and max_cycles per agent
        try:
            swarm_size = int(swarm_size)
        except Exception:
            swarm_size = 4
        if swarm_size <= 0:
            swarm_size = 1

        try:
            max_cycles = int(max_cycles)
        except Exception:
            max_cycles = 6
        if max_cycles <= 0:
            max_cycles = 1

        swarm_summary = self.swarm_orchestrator.run_swarm(
            goal=goal,
            base_preset=base_preset,
            swarm_size=swarm_size,
            max_cycles=max_cycles,
            mode=mode,
            agent_fn=None,
            run_id=agent_extra.pop("run_id", None),
            domain=domain or "general",
            curriculum_id=curriculum_id,
            profile_name=profile_name,
            seed=seed,
            global_context=global_context,
            per_agent_overrides=per_agent_overrides,
            timeout_s=timeout_s,
            return_agent_payloads=return_agent_payloads,
            extra_tags=extra_tags,
            **agent_extra,
        )
        return swarm_summary

    def run_orchestrated_swarm_training_burst(
        self,
        goal: str,
        *,
        base_preset: Dict[str, Any],
        rounds: int = 3,
        swarm_size: int = 4,
        max_cycles: int = 6,
        mode: Optional[str] = None,
        domain: Optional[str] = None,
        runtime_profile: Optional[str] = None,
        curriculum_id: Optional[str] = None,
        profile_name: Optional[str] = None,
        seed: Optional[int] = None,
        global_context: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
        extra_tags: Optional[Dict[str, Any]] = None,
        **agent_extra: Any,
    ) -> List[Dict[str, Any]]:
        """Run several orchestrated swarms in a row as a training burst."""
        results: List[Dict[str, Any]] = []
        if self.swarm_orchestrator is None:
            return results

        try:
            rounds = int(rounds)
        except Exception:
            rounds = 3
        if rounds <= 0:
            rounds = 1

        domain_effective = domain or "general"
        preset_with_runtime = dict(base_preset)
        if runtime_profile is not None:
            preset_with_runtime["runtime_profile"] = runtime_profile

        for _ in range(max(1, rounds)):
            summary = self.run_orchestrated_swarm(
                goal=goal,
                base_preset=preset_with_runtime,
                swarm_size=swarm_size,
                max_cycles=max_cycles,
                mode=mode,
                domain=domain_effective,
                curriculum_id=curriculum_id,
                profile_name=profile_name,
                seed=seed,
                global_context=global_context,
                timeout_s=timeout_s,
                return_agent_payloads=False,
                extra_tags=extra_tags,
                **agent_extra,
            )
            results.append(summary)
        return results

    def spawn_child_agent(self, extra_config: Optional[Dict[str, Any]] = None) -> "CoreAgent":
        """Create a new CoreAgent that shares the same MemoryStore."""
        cfg = dict(self.config)
        if extra_config:
            cfg.update(extra_config)
        return CoreAgent(memory_store=self.memory_store, config=cfg)

    # ------------------------------------------------------------------
    # Persistence helpers for background workers
    # ------------------------------------------------------------------
    def load_state_from_storage(self) -> Dict[str, Any]:
        """Load the latest engine run state from persistent storage."""
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
        """Save engine run state to persistent storage."""
        if not isinstance(state, dict):
            return

        payload: Dict[str, Any] = dict(state)

        try:
            if hasattr(self.memory_store, "save_run_state"):
                self.memory_store.save_run_state(payload)  # type: ignore[attr-defined]
                return
        except Exception:
            pass

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
        """Run exactly one TGRM cycle and return updated state plus summary."""
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
            state = dict(state)

        effective_run_id: str = run_id or state.get("run_id") or self._generate_run_id()
        state["run_id"] = effective_run_id

        if experiment_mode is None:
            experiment_mode = state.get("experiment_mode")
        state["experiment_mode"] = experiment_mode

        effective_goal: str = (goal or state.get("goal") or "").strip()
        state["goal"] = effective_goal

        effective_role: str = role or state.get("role", "agent")
        state["role"] = effective_role

        effective_domain: str = domain or state.get("domain", "general")
        state["domain"] = effective_domain

        if source_controls is not None:
            effective_sources: Dict[str, bool] = dict(source_controls)
            state["source_controls"] = dict(source_controls)
        else:
            stored_sc = state.get("source_controls")
            if isinstance(stored_sc, dict):
                effective_sources = dict(stored_sc)
            else:
                effective_sources = dict(self.source_controls)

        try:
            ci = int(state.get("cycle_index", 0))
            if ci < 0:
                ci = 0
        except Exception:
            history = self.memory_store.get_cycle_history()
            ci = len(history)
        state["cycle_index"] = ci

        if pdf_bytes is None:
            effective_pdf_bytes = self._attached_pdf_bytes
        else:
            effective_pdf_bytes = pdf_bytes

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

        state["last_update_ts"] = time.time()
        state["cycles_completed"] = int(state.get("cycles_completed", 0)) + 1
        state["cycle_index"] = ci + 1
        state["last_summary"] = summary

        if isinstance(summary, dict):
            started_ts = summary.get("started_at_ts")
            completed_ts = summary.get("completed_at_ts")
            if isinstance(started_ts, (int, float)):
                state["last_cycle_started_at_ts"] = float(started_ts)
            if isinstance(completed_ts, (int, float)):
                state["last_cycle_completed_at_ts"] = float(completed_ts)
            else:
                state["last_cycle_completed_at_ts"] = state["last_update_ts"]

        rye_val = summary.get("RYE")
        if isinstance(rye_val, (int, float)):
            state["last_rye"] = float(rye_val)

        if self.citations_enabled and isinstance(summary, dict):
            citations = summary.get("citations")
            if isinstance(citations, list):
                state["last_citations"] = citations

        if isinstance(summary, dict):
            contradictions = None
            contr_val = summary.get("contradictions")
            if isinstance(contr_val, (int, float)):
                contradictions = int(contr_val)
            self._meta_note_single_cycle(
                rye_value=rye_val if isinstance(rye_val, (int, float)) else None,
                contradictions=contradictions,
            )

        state["in_progress"] = True

        return {
            "state": state,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Public API used by Streamlit and CLI
    # ------------------------------------------------------------------
    def attach_pdf(self, uploaded_file: Any) -> None:
        """Attach a PDF (uploaded via Streamlit) for later cycles."""
        if uploaded_file is None:
            return
        try:
            self._attached_pdf_bytes = uploaded_file.read()
        except Exception:
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
        """Run a single cycle of research using the TGRM loop."""
        cycle_started_at_ts = time.time()

        effective_source_controls: Dict[str, bool] = {}
        if self.source_controls:
            effective_source_controls.update(self.source_controls)
        if source_controls:
            effective_source_controls.update(source_controls)

        effective_pdf_bytes: Optional[bytes] = (
            pdf_bytes if pdf_bytes is not None else self._attached_pdf_bytes
        )

        # Preferred advanced signature with MSIL and RYE mode hooks
        try:
            result = self.tgrm_loop.run_cycle(
                goal=goal,
                cycle_index=cycle_index,
                role=role,
                source_controls=effective_source_controls,
                pdf_bytes=effective_pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
                msil_mode=self.msil_mode,             # type: ignore[arg-type]
                msil_track_mode=self.msil_track_mode, # type: ignore[arg-type]
                rye_mode=self.rye_mode,               # type: ignore[arg-type]
                run_id=run_id,
                experiment_mode=experiment_mode,
            )
        except TypeError:
            # Backward compatibility for older TGRMLoop signatures
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
                try:
                    result = self.tgrm_loop.run_cycle(goal, cycle_index, role)
                except TypeError:
                    result = self.tgrm_loop.run_cycle(goal, cycle_index)

        # Normalize into a {"summary": ...} dict so callers and memory store agree
        if isinstance(result, dict) and "summary" in result:
            summary = result.get("summary") or {}
            if not isinstance(summary, dict):
                summary = {"raw_summary": summary}
                result["summary"] = summary
        else:
            # Treat bare dict as summary, wrap it
            if isinstance(result, dict):
                summary = dict(result)
            else:
                summary = {"raw_result": result}
            result = {"summary": summary}

        # Enrich summary with core metadata so MemoryStore and UI have stable fields
        if isinstance(summary, dict):
            if "cycle_index" not in summary:
                summary["cycle_index"] = cycle_index
            summary.setdefault("role", role)
            if domain is not None:
                summary.setdefault("domain", domain)
            if run_id is not None:
                summary.setdefault("run_id", run_id)
            if experiment_mode is not None:
                summary.setdefault("experiment_mode", experiment_mode)
            if goal:
                summary.setdefault("goal", goal)

            summary.setdefault("started_at_ts", cycle_started_at_ts)
            summary.setdefault("completed_at_ts", time.time())
            if "timestamp" not in summary:
                summary["timestamp"] = summary.get("completed_at_ts")

            # Attach tool stats if present on the raw result
            tool_stats = result.get("tool_stats")
            if tool_stats is not None and "tool_stats" not in summary:
                summary["tool_stats"] = tool_stats

            # Attach citations if enabled
            if self.citations_enabled:
                citations = result.get("citations")
                if isinstance(citations, list):
                    summary.setdefault("citations", citations)

            # Discovery hooks (RYE spikes, hypotheses, etc.)
            self._handle_post_cycle_discovery(
                summary,
                cycle_index=cycle_index,
                role=role,
                domain=domain,
                run_id=run_id,
                experiment_mode=experiment_mode,
            )

            # Persist cycle summary into MemoryStore if supported
            try:
                if hasattr(self.memory_store, "log_cycle_summary"):
                    try:
                        # Newer signature: (cycle_index, summary)
                        self.memory_store.log_cycle_summary(cycle_index, summary)  # type: ignore[attr-defined]
                    except TypeError:
                        # Older signature: (summary)
                        self.memory_store.log_cycle_summary(summary)  # type: ignore[attr-defined]
                elif hasattr(self.memory_store, "append_cycle_log"):
                    # Generic append helper some stores may implement
                    self.memory_store.append_cycle_log(summary)  # type: ignore[attr-defined]
            except Exception:
                # Logging failures should never break the main loop
                pass

        return result

    # ------------------------------------------------------------------
    # Multi goal hooks for TGRM v3
    # ------------------------------------------------------------------
    def run_multi_goal_cycle(
        self,
        goals: Sequence[str],
        cycle_index: int,
        role: str = "agent",
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a multi goal cycle if TGRMLoop supports it, otherwise degrade."""
        goals_list = [g for g in goals if str(g).strip()]
        if not goals_list:
            return self.run_cycle(
                goal="",
                cycle_index=cycle_index,
                role=role,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
                run_id=run_id,
                experiment_mode=experiment_mode,
            )

        # Advanced path: TGRMLoop.run_multi_goal_cycle
        if hasattr(self.tgrm_loop, "run_multi_goal_cycle"):
            try:
                effective_source_controls: Dict[str, bool] = {}
                if self.source_controls:
                    effective_source_controls.update(self.source_controls)
                if source_controls:
                    effective_source_controls.update(source_controls)

                eff_pdf_bytes = pdf_bytes if pdf_bytes is not None else self._attached_pdf_bytes

                result = self.tgrm_loop.run_multi_goal_cycle(  # type: ignore[attr-defined]
                    goals=goals_list,
                    cycle_index=cycle_index,
                    role=role,
                    source_controls=effective_source_controls,
                    pdf_bytes=eff_pdf_bytes,
                    biomarker_snapshot=biomarker_snapshot,
                    domain=domain,
                    msil_mode=self.msil_mode,
                    msil_track_mode=self.msil_track_mode,
                    rye_mode=self.rye_mode,
                    run_id=run_id,
                    experiment_mode=experiment_mode,
                )
                summary = result.get("summary", {})
                if isinstance(summary, dict):
                    if run_id is not None:
                        summary.setdefault("run_id", run_id)
                    if experiment_mode is not None:
                        summary.setdefault("experiment_mode", experiment_mode)
                    if domain is not None:
                        summary.setdefault("domain", domain)
                    summary.setdefault("role", role)
                    summary.setdefault("cycle_index", cycle_index)
                    self._handle_post_cycle_discovery(
                        summary,
                        cycle_index=cycle_index,
                        role=role,
                        domain=domain,
                        run_id=run_id,
                        experiment_mode=experiment_mode,
                    )
                return result
            except Exception:
                pass

        # Fallback: run first goal as main and attach others as context string
        combined_goal = goals_list[0]
        if len(goals_list) > 1:
            extra_goals = "; ".join(goals_list[1:])
            combined_goal = f"{goals_list[0]} (plus linked hypotheses: {extra_goals})"
        return self.run_cycle(
            goal=combined_goal,
            cycle_index=cycle_index,
            role=role,
            source_controls=source_controls,
            pdf_bytes=pdf_bytes,
            biomarker_snapshot=biomarker_snapshot,
            domain=domain,
            run_id=run_id,
            experiment_mode=experiment_mode,
        )

    # ------------------------------------------------------------------
    # Single agent continuous mode (finite only)
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

        Note:
            This method enforces finite only semantics. Even if callers
            pass forever=True or select a "forever" runtime_profile, the
            run will still be bounded by max_cycles and or max_minutes
            plus an internal failsafe.
        """
        # Defensive normalization for max_cycles and max_minutes
        try:
            max_cycles = int(max_cycles)
        except Exception:
            max_cycles = 100
        if max_cycles <= 0:
            max_cycles = 1

        if max_minutes is not None:
            try:
                max_minutes = float(max_minutes)
            except Exception:
                max_minutes = None
            if max_minutes is not None and max_minutes <= 0:
                max_minutes = None

        summaries: List[Dict[str, Any]] = []
        recent_rye: List[float] = []

        effective_run_id = run_id or self._generate_run_id()
        run_started_at_ts = time.time()

        # Global failsafe on cycles
        try:
            max_cycles_failsafe = int(CONTINUOUS_MODE_DEFAULTS.get("max_cycles_failsafe", 10_000_000))
        except Exception:
            max_cycles_failsafe = 10_000_000
        if max_cycles > max_cycles_failsafe:
            max_cycles = max_cycles_failsafe

        # Hard override: no true forever runs at CoreAgent level
        forever = False

        effective_domain = domain or "general"
        preset_cfg = get_preset(effective_domain)

        try:
            default_watchdog = float(
                CONTINUOUS_MODE_DEFAULTS.get("watchdog_interval_minutes", watchdog_interval_minutes)
            )
        except Exception:
            default_watchdog = watchdog_interval_minutes
        if watchdog_interval_minutes == 5.0:
            watchdog_interval_minutes = default_watchdog

        # Auto learn runtime profile and stop_rye from history
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

        # Use preset defaults for accelerated or ultra modes
        if runtime_profile is None and self.speed_mode in {"accelerated", "ultra"}:
            default_profile = preset_cfg.get("default_runtime_profile")
            if isinstance(default_profile, str):
                runtime_profile = default_profile

        if runtime_profile is None:
            runtime_profile = preset_cfg.get("default_runtime_profile")

        profile_cfg: Optional[Dict[str, Any]] = None
        if runtime_profile:
            profile_cfg = RUNTIME_PROFILES.get(runtime_profile)

        if profile_cfg is not None:
            est_cycles = profile_cfg.get("estimated_cycles")
            profile_stop_rye = profile_cfg.get("rye_stop_threshold")

            # Convert runtime_profile into a bounded time window
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
                    # Treat "forever" as a long but still finite hint
                    # If no explicit max_minutes, we simply leave it None
                    # and rely on cycle caps and failsafe.
                    pass

            if isinstance(est_cycles, (int, float)) and max_cycles <= 100:
                try:
                    max_cycles = max(1, int(est_cycles))
                except Exception:
                    pass

            if stop_rye is None and isinstance(profile_stop_rye, (int, float)):
                stop_rye = float(profile_stop_rye)

        # Fallback mapping from cycles to time budget in finite only mode
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
                # Clamp to failsafe but do not enable forever
                max_cycles = max_cycles_failsafe

        # Checkpoint resume (finite only)
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
                        if max_minutes is None or max_minutes > float(remaining):
                            max_minutes = float(remaining)

        history = self.memory_store.get_cycle_history()
        start_index = len(history)

        start_time = time.monotonic()
        last_watchdog_update = start_time

        i = 0
        stop_reason: str = "completed"

        heartbeat_label = CONTINUOUS_MODE_DEFAULTS.get("heartbeat_labels", {}).get("single", "continuous_single")

        # Default experiment mode for Breakthrough Hunter
        if experiment_mode is None and self.breakthrough_hunter_enabled:
            experiment_mode = "breakthrough"

        while True:
            if i >= max_cycles:
                stop_reason = "cycle_cap"
                break

            elapsed_min = (time.monotonic() - start_time) / 60.0
            if max_minutes is not None and elapsed_min >= max_minutes:
                stop_reason = "time_limit"
                break

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
                    "forever": False,
                    "cycle_history_length": len(history),
                    "local_cycle_index": i,
                    "recent_rye": recent_rye[-10:],
                    "last_heartbeat_ts": time.time(),
                    "runtime_profile": runtime_profile,
                    "speed_mode": self.speed_mode,
                }
                self._save_checkpoint(checkpoint_state)
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
                if len(recent_rye) > 10:
                    recent_rye.pop(0)

            # Stability Kernel guard
            action = self._apply_stability_kernel(
                summary,
                mode="single",
                goal=goal,
                domain=effective_domain,
                run_id=effective_run_id,
                experiment_mode=experiment_mode,
                elapsed_minutes=elapsed_min,
            )
            if isinstance(action, dict):
                if action.get("halt"):
                    stop_reason = "stability_guard"
                    break

            if stop_rye is not None and recent_rye:
                avg_rye = sum(recent_rye) / len(recent_rye)
                if avg_rye < stop_rye:
                    stop_reason = "rye_threshold"
                    break

            i += 1

        self._clear_checkpoint()

        total_elapsed_min = (time.monotonic() - start_time) / 60.0
        run_completed_at_ts = time.time()
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
            "speed_mode": self.speed_mode,
            "started_at_ts": run_started_at_ts,
            "completed_at_ts": run_completed_at_ts,
        }

        if _rye_metrics_mod is not None:
            try:
                if hasattr(_rye_metrics_mod, "run_health_score"):
                    health_score = _rye_metrics_mod.run_health_score(summaries)  # type: ignore[attr-defined]
                    run_metadata["run_health_score"] = health_score
            except Exception:
                pass

        if self.citations_enabled:
            total_citations = 0
            for s in summaries:
                if isinstance(s, dict):
                    c = s.get("citations")
                    if isinstance(c, list):
                        total_citations += len(c)
            run_metadata["total_citations"] = total_citations

        if self.learned_from_memory:
            run_metadata["learned_from_memory"] = dict(self.learned_from_memory)

        for s in summaries:
            if isinstance(s, dict):
                s["run_metadata"] = dict(run_metadata)

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
    # Swarm aware continuous mode (multi agent rounds, finite only)
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
        """Run continuous multi agent swarm rounds.

        Note:
            This method enforces finite only semantics. Even if callers
            pass forever=True or select a "forever" runtime_profile, the
            run will still be bounded by max_rounds and or max_minutes.
        """
        # Defensive normalization for max_rounds and max_minutes
        try:
            max_rounds = int(max_rounds)
        except Exception:
            max_rounds = 50
        if max_rounds <= 0:
            max_rounds = 1

        if max_minutes is not None:
            try:
                max_minutes = float(max_minutes)
            except Exception:
                max_minutes = None
            if max_minutes is not None and max_minutes <= 0:
                max_minutes = None

        all_summaries: List[Dict[str, Any]] = []
        recent_round_rye: List[float] = []

        effective_run_id = run_id or self._generate_run_id()
        effective_domain = domain or "general"
        preset_cfg = get_preset(effective_domain)
        run_started_at_ts = time.time()

        # Hard override: no true forever runs at CoreAgent level
        forever = False

        if roles is None:
            roles_seq: Sequence[str] = self.agent_roles
        else:
            roles_seq = roles
        roles_list: List[str] = [str(r) for r in roles_seq][: self.max_agents]
        if not roles_list:
            roles_list = ["agent"]

        # Global failsafe on rounds
        try:
            max_rounds_failsafe = int(CONTINUOUS_MODE_DEFAULTS.get("max_rounds_failsafe", 10_000_000))
        except Exception:
            max_rounds_failsafe = 10_000_000
        if max_rounds > max_rounds_failsafe:
            max_rounds = max_rounds_failsafe

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

        try:
            default_watchdog = float(
                CONTINUOUS_MODE_DEFAULTS.get("watchdog_interval_minutes", watchdog_interval_minutes)
            )
        except Exception:
            default_watchdog = watchdog_interval_minutes
        if watchdog_interval_minutes == 5.0:
            watchdog_interval_minutes = default_watchdog

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

        if runtime_profile is None and self.speed_mode in {"accelerated", "ultra"}:
            default_profile = preset_cfg.get("default_runtime_profile")
            if isinstance(default_profile, str):
                runtime_profile = default_profile

        if runtime_profile is None:
            runtime_profile = preset_cfg.get("default_runtime_profile")

        profile_cfg: Optional[Dict[str, Any]] = None
        if runtime_profile:
            profile_cfg = RUNTIME_PROFILES.get(runtime_profile)

        if profile_cfg is not None:
            est_cycles = profile_cfg.get("estimated_cycles")
            profile_stop_rye = profile_cfg.get("rye_stop_threshold")

            # Convert runtime_profile into bounded rounds or time
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
                    # Treat "forever" as a long but still finite hint.
                    # We leave max_minutes unchanged and rely on max_rounds.
                    pass

            if stop_rye is None and isinstance(profile_stop_rye, (int, float)):
                stop_rye = float(profile_stop_rye)

        if stop_rye is None:
            preset_stop = preset_cfg.get("default_rye_stop_threshold")
            if isinstance(preset_stop, (int, float)):
                stop_rye = float(preset_stop)

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

        history = self.memory_store.get_cycle_history()
        start_index = len(history)

        start_time = time.monotonic()
        last_watchdog_update = start_time

        round_idx = 0
        stop_reason: str = "completed"

        heartbeat_label = CONTINUOUS_MODE_DEFAULTS.get("heartbeat_labels", {}).get("swarm", "continuous_swarm")

        if experiment_mode is None and self.breakthrough_hunter_enabled:
            experiment_mode = "breakthrough"

        while True:
            if round_idx >= max_rounds:
                stop_reason = "round_cap"
                break

            elapsed_min = (time.monotonic() - start_time) / 60.0
            if max_minutes is not None and elapsed_min >= max_minutes:
                stop_reason = "time_limit"
                break

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
                    "forever": False,
                    "cycle_history_length": len(history),
                    "local_round_index": round_idx,
                    "recent_round_rye": recent_round_rye[-10:],
                    "last_heartbeat_ts": time.time(),
                    "runtime_profile": runtime_profile,
                    "speed_mode": self.speed_mode,
                }
                self._save_checkpoint(checkpoint_state)
                try:
                    self.memory_store.heartbeat(label=heartbeat_label)
                except Exception:
                    pass
                last_watchdog_update = now

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

            round_ryes: List[float] = []
            round_contras: List[float] = []
            for s in round_summaries:
                rv = s.get("RYE")
                if isinstance(rv, (int, float)):
                    round_ryes.append(float(rv))
                cv = s.get("contradictions")
                if isinstance(cv, (int, float)):
                    round_contras.append(float(cv))

            avg_round_rye = None
            avg_round_contra = None
            if round_ryes:
                avg_round_rye = sum(round_ryes) / len(round_ryes)
                recent_round_rye.append(avg_round_rye)
                if len(recent_round_rye) > 10:
                    recent_round_rye.pop(0)

            if round_contras:
                avg_round_contra = sum(round_contras) / len(round_contras)

            self._meta_note_swarm_round(avg_round_rye, avg_round_contra)

            # Stability Kernel guard
            if round_summaries:
                action = self._apply_stability_kernel(
                    round_summaries[-1],
                    mode="swarm",
                    goal=goal,
                    domain=effective_domain,
                    run_id=effective_run_id,
                    experiment_mode=experiment_mode,
                    elapsed_minutes=elapsed_min,
                )
                if isinstance(action, dict) and action.get("halt"):
                    stop_reason = "stability_guard"
                    break

            if stop_rye is not None and recent_round_rye:
                avg_recent = sum(recent_round_rye) / len(recent_round_rye)
                if avg_recent < stop_rye:
                    stop_reason = "rye_threshold"
                    break

            round_idx += 1

        self._clear_checkpoint()

        total_elapsed_min = (time.monotonic() - start_time) / 60.0
        run_completed_at_ts = time.time()
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
            "speed_mode": self.speed_mode,
            "started_at_ts": run_started_at_ts,
            "completed_at_ts": run_completed_at_ts,
        }

        if _rye_metrics_mod is not None:
            try:
                if hasattr(_rye_metrics_mod, "run_health_score"):
                    health_score = _rye_metrics_mod.run_health_score(all_summaries)  # type: ignore[attr-defined]
                    run_metadata["run_health_score"] = health_score
            except Exception:
                pass

        if self.citations_enabled:
            total_citations = 0
            for s in all_summaries:
                if isinstance(s, dict):
                    c = s.get("citations")
                    if isinstance(c, list):
                        total_citations += len(c)
            run_metadata["total_citations"] = total_citations

        if self.learned_from_memory:
            run_metadata["learned_from_memory"] = dict(self.learned_from_memory)

        for s in all_summaries:
            if isinstance(s, dict):
                s["run_metadata"] = dict(run_metadata)

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
    # Multi agent round helper
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
        """Run one multi agent round: one cycle per role, sharing MemoryStore."""
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
        """Run a memory pruning pass using MemoryPruner."""
        try:
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
        """Generate a weekly snapshot report using SnapshotGenerator."""
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
        """Run verification passes on pending hypotheses."""
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
    # Agent level training and learning helpers
    # ------------------------------------------------------------------
    def get_learning_status(self) -> Dict[str, Any]:
        """Return a compact view of the agent level learning state."""
        status: Dict[str, Any] = {
            "learning_enabled": self.learning_enabled,
            "min_training_history": self.min_training_history,
            "learned_from_memory": dict(self.learned_from_memory) if self.learned_from_memory else {},
            "training_profile": dict(self.training_profile) if self.training_profile else {},
            "learning_plan": dict(self.learning_plan) if self.learning_plan else {},
            "speed_mode": self.speed_mode,
            "citations_enabled": self.citations_enabled,
        }
        try:
            history = self.memory_store.get_cycle_history()
            status["total_cycles"] = len(history)
        except Exception:
            status["total_cycles"] = None

        try:
            status["meta_state"] = {
                "speed_mode": self._meta_state.get("speed_mode"),
                "single_run_recent_rye": (self._meta_state.get("single_run_stats") or {}).get("recent_rye", []),
                "swarm_recent_round_rye": (self._meta_state.get("swarm_run_stats") or {}).get("recent_round_rye", []),
            }
        except Exception:
            pass

        return status

    def _update_training_profile_from_summaries(
        self,
        summaries: List[Dict[str, Any]],
        *,
        domain: Optional[str] = None,
        runtime_profile: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> None:
        """Derive a training profile snapshot after a burst."""
        profile: Dict[str, Any] = {
            "last_training_at": time.time(),
            "domain": domain or "general",
            "runtime_profile": runtime_profile,
            "experiment_mode": experiment_mode,
            "cycles_in_burst": len(summaries),
        }

        rye_vals: List[float] = []
        for s in summaries:
            rv = s.get("RYE")
            if isinstance(rv, (int, float)):
                rye_vals.append(float(rv))
        if rye_vals:
            profile["burst_rye_avg"] = sum(rye_vals) / len(rye_vals)
            profile["burst_rye_last"] = rye_vals[-1]

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
        """Run a short, focused training burst."""
        if not self.learning_enabled:
            return []

        effective_domain = domain or "general"
        if runtime_profile is None:
            runtime_profile = str(self.config.get("training_runtime_profile", "1_hour"))
            if runtime_profile not in RUNTIME_PROFILES:
                runtime_profile = "1_hour"

        if max_cycles is None:
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

        self._update_training_profile_from_summaries(
            summaries,
            domain=effective_domain,
            runtime_profile=runtime_profile,
            experiment_mode=experiment_mode,
        )

        return summaries

    def run_multi_goal_training_burst(
        self,
        goals: Sequence[str],
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
        """Training burst that uses multi goal cycles when available."""
        if hasattr(self.tgrm_loop, "run_multi_goal_cycle"):
            # Use multi goal cycles by orchestrating cycles manually
            if not self.learning_enabled:
                return []
            effective_domain = domain or "general"
            if runtime_profile is None:
                runtime_profile = "1_hour"
            if max_cycles is None:
                max_cycles = 30
            if max_minutes is None:
                max_minutes = 60.0
            if experiment_mode is None:
                experiment_mode = "training_multi_goal"

            summaries: List[Dict[str, Any]] = []
            start_time = time.monotonic()
            history = self.memory_store.get_cycle_history()
            base_index = len(history)

            for i in range(max_cycles):
                elapsed_min = (time.monotonic() - start_time) / 60.0
                if elapsed_min >= max_minutes:
                    break
                ci = base_index + i
                result = self.run_multi_goal_cycle(
                    goals=goals,
                    cycle_index=ci,
                    role=role,
                    source_controls=source_controls,
                    pdf_bytes=pdf_bytes,
                    biomarker_snapshot=biomarker_snapshot,
                    domain=effective_domain,
                    run_id=None,
                    experiment_mode=experiment_mode,
                )
                summary = result.get("summary", {})
                summaries.append(summary)

                rye_val = summary.get("RYE")
                if stop_rye is not None and isinstance(rye_val, (int, float)) and rye_val < stop_rye:
                    break

            self._update_training_profile_from_summaries(
                summaries,
                domain=effective_domain,
                runtime_profile=runtime_profile,
                experiment_mode=experiment_mode,
            )
            return summaries

        # Fallback to single goal training burst using the first goal
        first_goal = ""
        for g in goals:
            gs = str(g).strip()
            if gs:
                first_goal = gs
                break
        return self.run_training_burst(
            goal=first_goal,
            domain=domain,
            role=role,
            runtime_profile=runtime_profile,
            max_cycles=max_cycles,
            max_minutes=max_minutes,
            stop_rye=stop_rye,
            source_controls=source_controls,
            pdf_bytes=pdf_bytes,
            biomarker_snapshot=biomarker_snapshot,
            experiment_mode=experiment_mode,
        )

    def optimize_learning_pipeline(
        self,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Derive a high level learning plan from existing history."""
        plan: Dict[str, Any] = {
            "domain": domain or "general",
            "learning_enabled": self.learning_enabled,
            "speed_mode": self.speed_mode,
            "actions": [],
            "learned_from_memory": {},
            "option_c_signature": None,
        }

        if not self.learning_enabled:
            plan["actions"].append(
                "Learning is disabled in config (learning_enabled=False). Enable it to train the agent."
            )
            self.learning_plan = plan
            return plan

        learned = self.learn_from_memory(domain=domain)
        plan["learned_from_memory"] = dict(learned)

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

        actions: List[str] = []

        has_history = bool(learned.get("has_history"))
        rec_profile = learned.get("recommended_runtime_profile")
        rec_stop = learned.get("recommended_stop_rye")

        if not has_history:
            actions.append(
                "Run at least one short training burst (for example 1 hour general or longevity) "
                "to build initial cycle history before auto tuning."
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
                    actions.append(f"Autonomy stability safety envelope is classified as: {env_state}.")
            except Exception:
                pass

        actions.append(
            "Use run_training_burst(...) with focused goals to teach the agent on compact, high signal problems."
        )
        actions.append(
            "Periodically call prune_memory(...) to keep MemoryStore healthy and avoid dilution of high value traces."
        )
        actions.append(
            "After major bursts, regenerate snapshots with generate_snapshot(...) to track skill buildup over weeks."
        )

        plan["actions"] = actions
        self.learning_plan = plan
        return plan

    # ------------------------------------------------------------------
    # Meta Agent tuning helpers
    # ------------------------------------------------------------------
    def run_meta_tuning(self) -> Dict[str, Any]:
        """Use MetaAgent to propose self tuning updates if available."""
        result: Dict[str, Any] = {
            "enabled": False,
            "suggestions": None,
            "error": None,
        }
        if self.meta_agent is None:
            return result
        try:
            suggest = getattr(self.meta_agent, "suggest_tuning", None)
            if callable(suggest):
                suggestions = suggest(
                    learning_plan=self.learning_plan,
                    training_profile=self.training_profile,
                    meta_state=self._meta_state,
                    config=self.config,
                )
                result["enabled"] = True
                result["suggestions"] = suggestions
                self.last_meta_tuning = suggestions if isinstance(suggestions, dict) else {}
                return result
        except Exception as exc:
            result["error"] = str(exc)
        return result

    # ------------------------------------------------------------------
    # Long horizon and trading helpers
    # ------------------------------------------------------------------
    def run_long_horizon_single(
        self,
        goal: str,
        days: float = 3.0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper for a long horizon single agent run."""
        max_minutes = days * 24.0 * 60.0
        return self.run_continuous(
            goal=goal,
            max_cycles=int(1_000_000),
            max_minutes=max_minutes,
            forever=False,
            resume_from_checkpoint=True,
            runtime_profile="24_hours",
            **kwargs,
        )

    def run_long_horizon_swarm(
        self,
        goal: str,
        days: float = 3.0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper for a long horizon swarm run."""
        max_minutes = days * 24.0 * 60.0
        return self.run_swarm_continuous(
            goal=goal,
            max_rounds=int(1_000_000),
            max_minutes=max_minutes,
            forever=False,
            resume_from_checkpoint=True,
            runtime_profile="24_hours",
            **kwargs,
        )

    def run_trading_session(
        self,
        goal: str,
        *,
        max_minutes: float = 60.0,
        max_cycles: int = 200,
        experiment_mode: Optional[str] = "trading",
    ) -> List[Dict[str, Any]]:
        """Run a trading focused single agent session."""
        return self.run_continuous(
            goal=goal,
            max_cycles=max_cycles,
            domain="trading",
            max_minutes=max_minutes,
            experiment_mode=experiment_mode,
        )

    def run_trading_swarm(
        self,
        goal: str,
        *,
        max_minutes: float = 60.0,
        max_rounds: int = 20,
        experiment_mode: Optional[str] = "trading",
    ) -> List[Dict[str, Any]]:
        """Run a trading focused swarm session."""
        return self.run_swarm_continuous(
            goal=goal,
            max_rounds=max_rounds,
            domain="trading",
            max_minutes=max_minutes,
            experiment_mode=experiment_mode,
        )

    # ------------------------------------------------------------------
    # Autonomous literature crawler helpers
    # ------------------------------------------------------------------
    def run_literature_crawl(
        self,
        query: str,
        max_items: int = 50,
        domains: Optional[Sequence[str]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an autonomous literature crawl if a crawler is attached."""
        if self.literature_crawler is None:
            return {
                "enabled": False,
                "items": [],
                "reason": "crawler_not_available",
            }
        try:
            crawl_fn = getattr(self.literature_crawler, "crawl_and_ingest", None)
            if callable(crawl_fn):
                result = crawl_fn(
                    query=query,
                    max_items=max_items,
                    domains=domains,
                    run_id=run_id or self._generate_run_id(),
                )
                if isinstance(result, dict):
                    result.setdefault("enabled", True)
                    return result
        except Exception as exc:
            return {
                "enabled": True,
                "items": [],
                "error": str(exc),
            }
        return {
            "enabled": True,
            "items": [],
        }
