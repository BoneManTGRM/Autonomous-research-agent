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

import inspect
import re
import os
import time
import traceback
import concurrent.futures
import threading
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from typing import Any, Dict, List, Optional, Tuple, Sequence

# ---- Metrics import (package-safe) ----
try:
    from .rye_metrics import compute_delta_r, compute_rye
except Exception:  # pragma: no cover
    try:
        from rye_metrics import compute_delta_r, compute_rye  # type: ignore
    except Exception:  # pragma: no cover
        # Fallback stubs (keeps the loop running in minimal environments)
        def compute_delta_r(*args, **kwargs) -> float:
            return 0.0

        def compute_rye(*args, **kwargs) -> float:
            return 0.0
# ---------------------------------------------------------------------------
# Global limits and helpers
# ---------------------------------------------------------------------------

# Tavily imposes a maximum query length of 400 characters. We clamp slightly
# below that to leave room for any internal decorations.
MAX_WEB_QUERY_LEN = 380


# ---------------------------------------------------------------------------
# Web search timeout guard
#
# Tavily (and other web providers) can occasionally hang or back off for a very
# long time under load. In a 64-agent swarm, a single hung web call can block
# an entire cycle barrier. This lightweight wrapper enforces a hard timeout so
# the agent continues the cycle with empty web results rather than stalling.
#
# Configure via env vars:
#   WORKER_WEB_TIMEOUT_S (default 25)
#   WORKER_WEB_TIMEOUT_WORKERS (default 4)
#   WORKER_WEB_TIMEOUT_MAX_INFLIGHT (default workers*4)
# ---------------------------------------------------------------------------

_WEB_TIMEOUT_EXECUTOR = None  # lazily initialized ThreadPoolExecutor
_WEB_TIMEOUT_EXECUTOR_LOCK = threading.Lock()
_WEB_TIMEOUT_INFLIGHT = 0
_WEB_TIMEOUT_INFLIGHT_LOCK = threading.Lock()


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return int(default)
    try:
        return int(val)
    except Exception:
        return int(default)


def _get_web_timeout_s() -> float:
    timeout_s = _env_float(
        "WORKER_WEB_TIMEOUT_S",
        _env_float("WEB_TIMEOUT_S", _env_float("TAVILY_TIMEOUT_S", 25.0)),
    )
    # Clamp to a sane range.
    if timeout_s < 3.0:
        timeout_s = 3.0
    if timeout_s > 180.0:
        timeout_s = 180.0
    return float(timeout_s)


def _get_web_timeout_workers() -> int:
    workers = _env_int("WORKER_WEB_TIMEOUT_WORKERS", _env_int("WORKER_WEB_MAX_CONCURRENCY", 4))
    if workers < 1:
        workers = 1
    if workers > 32:
        workers = 32
    return int(workers)


def _get_web_timeout_max_inflight(workers: int) -> int:
    cap = _env_int("WORKER_WEB_TIMEOUT_MAX_INFLIGHT", int(workers) * 4)
    if cap < int(workers):
        cap = int(workers)
    if cap > 256:
        cap = 256
    return int(cap)


def _get_web_timeout_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _WEB_TIMEOUT_EXECUTOR
    if _WEB_TIMEOUT_EXECUTOR is not None:
        return _WEB_TIMEOUT_EXECUTOR
    with _WEB_TIMEOUT_EXECUTOR_LOCK:
        if _WEB_TIMEOUT_EXECUTOR is not None:
            return _WEB_TIMEOUT_EXECUTOR
        max_workers = _get_web_timeout_workers()
        _WEB_TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tgrm-web-timeout",
        )
        return _WEB_TIMEOUT_EXECUTOR


def _decrement_web_inflight(_future: "concurrent.futures.Future[Any]") -> None:
    global _WEB_TIMEOUT_INFLIGHT
    try:
        with _WEB_TIMEOUT_INFLIGHT_LOCK:
            _WEB_TIMEOUT_INFLIGHT = max(0, int(_WEB_TIMEOUT_INFLIGHT) - 1)
    except Exception:
        pass


def _call_with_timeout(
    fn: Any,
    timeout_s: float,
    default: Any,
    allow_exceptions: Tuple[type, ...] = (),
) -> Any:
    """Run fn() with a hard timeout; return `default` on timeout or error.

    NOTE: Python cannot force-kill a running thread, but this prevents the
    *agent* from blocking forever on a hanging provider call. In-flight calls
    are capped to avoid unbounded queue growth if upstream calls hang.
    """
    global _WEB_TIMEOUT_INFLIGHT

    try:
        timeout_s = float(timeout_s)
    except Exception:
        timeout_s = 25.0

    if timeout_s <= 0:
        try:
            return fn()
        except Exception as e:
            if allow_exceptions and isinstance(e, allow_exceptions):
                raise
            return default

    workers = _get_web_timeout_workers()
    max_inflight = _get_web_timeout_max_inflight(workers)

    future: Optional["concurrent.futures.Future[Any]"] = None

    # Guard against unbounded queue growth when upstream calls hang.
    try:
        with _WEB_TIMEOUT_INFLIGHT_LOCK:
            if int(_WEB_TIMEOUT_INFLIGHT) >= int(max_inflight):
                return default
            _WEB_TIMEOUT_INFLIGHT = int(_WEB_TIMEOUT_INFLIGHT) + 1
    except Exception:
        pass

    try:
        executor = _get_web_timeout_executor()
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            try:
                future.cancel()
            except Exception:
                pass
            return default
        except Exception as e:
            if allow_exceptions and isinstance(e, allow_exceptions):
                raise
            return default
    finally:
        try:
            if future is not None:
                future.add_done_callback(_decrement_web_inflight)
            else:
                with _WEB_TIMEOUT_INFLIGHT_LOCK:
                    _WEB_TIMEOUT_INFLIGHT = max(0, int(_WEB_TIMEOUT_INFLIGHT) - 1)
        except Exception:
            pass



def _clamp_query(q: str, limit: int = MAX_WEB_QUERY_LEN) -> str:
    """Clamp web search queries to Tavily-safe length.

    Preserves core meaning while preventing "Query is too long" errors.
    Ensures the returned string length never exceeds `limit`.
    """
    if not isinstance(q, str):
        return ""
    text = q.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: (limit - 3)].rstrip() + "..."


def _first_numeric(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    """Return the first numeric value found for any of the provided keys.

    IMPORTANT: Unlike `dict.get(...) or ...`, this correctly preserves 0.0.
    """
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


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
# We also import the unified web_search helper so that even if tools_web
# is missing, we can still run real Tavily backed search via tools.py.
try:
    from .tools import Toolbelt, ToolUsage, web_search as core_web_search  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    Toolbelt = None  # type: ignore[assignment]
    core_web_search = None  # type: ignore[assignment]

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
# Local energy accounting helper
# ---------------------------------------------------------------------------


def compute_energy(
    actions_taken: List[Dict[str, Any]],
    web_calls: int = 0,
    pubmed_calls: int = 0,
    semantic_calls: int = 0,
    pdf_ingestions: int = 0,
    tokens_estimate: int = 0,
    swarm_size: Optional[int] = None,
    swarm_layer: Optional[str] = None,
) -> float:
    """Compute an energy cost E for a cycle compatible with older setups.

    Energy is a weighted combination of:
        - number of repair actions
        - external calls (web, PubMed, Semantic Scholar, PDF ingestion)
        - approximate tokens processed
        - optional swarm scaling (size and layer)
    """
    base_actions = max(1, len(actions_taken))
    external_calls = int(web_calls) + int(pubmed_calls) + int(semantic_calls) + int(pdf_ingestions)
    token_cost = float(tokens_estimate) / 1000.0

    energy = 0.5 * base_actions + 0.75 * external_calls + 0.2 * token_cost

    if swarm_size:
        energy *= 1.0 + min(2.0, max(0.0, (int(swarm_size) - 1) / 32.0))

    if swarm_layer:
        layer = str(swarm_layer).lower()
        if layer in {"meta", "guardian"}:
            energy *= 1.15
        elif layer in {"exploration", "deep"}:
            energy *= 1.1

    return max(0.1, float(energy))


# ---------------------------------------------------------------------------
# Null or fallback tool implementations
# ---------------------------------------------------------------------------


class _NullWebTool:
    """Fallback web research tool.

    Updated to use the same Tavily backed web_search helper defined in
    tools.py when available, so that TGRM runs get real web results even
    if tools_web.WebResearchTool is missing or disabled.
    """

    def __init__(self) -> None:
        self._web_search_fn = core_web_search

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Return a list of result dicts using tools.web_search if possible."""
        if self._web_search_fn is None:
            return []

        query = _clamp_query(query)

        max_results = kwargs.get("max_results")
        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 8

        # Prefer explicit search_depth; fall back from a "level" convention.
        search_depth = kwargs.get("search_depth")
        if not isinstance(search_depth, str) or not search_depth:
            level = kwargs.get("level")
            if isinstance(level, int) and level <= 1:
                search_depth = "basic"
            else:
                search_depth = "advanced"

        topic = kwargs.get("topic")
        if not isinstance(topic, str) or not topic:
            topic = "general"

        local_usage = ToolUsage()

        # Try the most complete signature first, then degrade gracefully.
        try:
            res = self._web_search_fn(
                query=query,
                tool_usage=local_usage,
                max_results=max_results,
                search_depth=search_depth,
                topic=topic,
            )
        except TypeError:
            try:
                res = self._web_search_fn(
                    query=query,
                    tool_usage=local_usage,
                    max_results=max_results,
                    search_depth=search_depth,
                )
            except TypeError:
                try:
                    res = self._web_search_fn(
                        query=query,
                        max_results=max_results,
                        search_depth=search_depth,
                    )
                except Exception:
                    return []
            except Exception:
                return []
        except Exception:
            return []

        if not isinstance(res, dict):
            return []

        items: List[Dict[str, Any]] = []
        for item in (res.get("results") or []):
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "source": item.get("source") or res.get("provider") or "web",
                    "content": item.get("content") or "",
                    "raw_html": item.get("raw_html") or "",
                }
            )
        return items

    def summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Simple text summary of top web results for note logging."""
        if not results:
            return "Web search returned no usable results for this cycle."

        lines: List[str] = []
        for idx, r in enumerate(results[:5], start=1):
            title = (r.get("title") or "").strip() or "(no title)"
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").replace("\n", " ").strip()
            snippet = content[:400]
            if url:
                lines.append(f"{idx}. {title} [{url}]\n   {snippet}")
            else:
                lines.append(f"{idx}. {title}\n   {snippet}")
        return "\n".join(lines)

    def to_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize search results into citation dicts."""
        citations: List[Dict[str, Any]] = []
        for r in results or []:
            citations.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "source": r.get("source") or "web",
                }
            )
        return citations


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

        # Long run optimization: track which questions have already been researched.
        # IMPORTANT: This is scoped PER GOAL and uses a TTL so that failures can retry later.
        self._seen_questions: Dict[str, Dict[str, float]] = {}
        # Cache: derive tool-friendly search queries from long prompts
        self._goal_query_cache: Dict[str, str] = {}
        self._goal_query_bank_cache: Dict[str, List[str]] = {}
        self._goal_citation_target_cache: Dict[str, int] = {}
        try:
            self._seen_question_ttl_s = float(self.config.get("seen_question_ttl_s", 6 * 3600))
        except Exception:
            self._seen_question_ttl_s = float(6 * 3600)

        # Citation dedupe cache (scoped per goal). This prevents the same external
        # source from being appended over and over across cycles when web search
        # returns the same top results (common with Tavily).
        self._seen_citation_fps: Dict[str, set] = {}

        # TGRM level: 1 (basic), 2 (targeted), 3 (domain plus swarm aware).
        try:
            self.tgrm_level = int(self.config.get("tgrm_level", 3))
        except Exception:
            self.tgrm_level = 3

        # Sliding window size for short term RYE gradient estimates
        try:
            self.rye_window_size = int(self.config.get("rye_window_size", 20))
        except Exception:
            self.rye_window_size = 20

        # Ultra speed mode and strict pipeline flag (used by meta controller and UI)
        self.ultra_speed = bool(self.config.get("ultra_speed", False))
        self.strict_pipeline = bool(self.config.get("strict_pipeline", True))

        # Under-cited trigger threshold: citation_markers / notes_count
        try:
            self.under_cited_ratio = float(self.config.get("under_cited_ratio", 0.5))
        except Exception:
            self.under_cited_ratio = 0.5
        self.under_cited_ratio = max(0.0, min(1.0, self.under_cited_ratio))

        # Maximum note size to store (prevents runaway memory/disk growth)
        try:
            self.max_note_chars = int(self.config.get("max_note_chars", 12000))
        except Exception:
            self.max_note_chars = 12000
        self.max_note_chars = max(1000, self.max_note_chars)

        # Dynamic search energy controls for Tavily and web usage.
        self.search_energy_mode = str(self.config.get("search_energy_mode", "auto")).lower()
        try:
            self.search_energy_base = float(self.config.get("search_energy_base", 1.0))
        except Exception:
            self.search_energy_base = 1.0
        try:
            self.search_energy_min = float(self.config.get("search_energy_min", 0.4))
        except Exception:
            self.search_energy_min = 0.4
        try:
            self.search_energy_max = float(self.config.get("search_energy_max", 1.6))
        except Exception:
            self.search_energy_max = 1.6

        # Per cycle value updated inside run_cycle.
        self.search_energy: float = self.search_energy_base

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _deadline_hit(self, deadline_ts: Optional[float]) -> bool:
        return deadline_ts is not None and time.time() >= deadline_ts

    def _cap_note_text(self, note_text: str) -> str:
        """Cap note size so long runs do not accumulate unbounded note bodies."""
        if not isinstance(note_text, str):
            return ""
        if len(note_text) <= self.max_note_chars:
            return note_text
        truncated = note_text[: self.max_note_chars].rstrip()
        return truncated + "\n\n[truncated]\n"

    def _estimate_tokens(self, text: str) -> int:
        """Very rough token estimate from text length."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _scan_notes_for_basic_markers(self, notes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Scan notes and count markers in a way that supports stable diagnostics.

        Returns counts:
            - question_lines: number of lines containing '?'
            - todo_lines: number of lines containing 'TODO'/'todo'
            - contradiction_lines: number of lines containing 'CONTRADICTION'/'contradiction'
        """
        counts = {"question_lines": 0, "todo_lines": 0, "contradiction_lines": 0}
        if not notes:
            return counts

        for note in notes:
            content = str(note.get("content", "") or "")
            if not content:
                continue
            for line in content.splitlines():
                s = line.strip()
                if not s:
                    continue
                s_lower = s.lower()
                if "?" in s:
                    counts["question_lines"] += 1
                if "todo" in s_lower:
                    counts["todo_lines"] += 1
                if "contradiction" in s_lower:
                    counts["contradiction_lines"] += 1
        return counts

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------

    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize a URL for robust citation de-duplication.

        Tavily and other web sources often return the same page with minor URL
        variations (tracking params, fragments, http/https). Canonicalization
        reduces those variants to a stable key.
        """
        if not isinstance(url, str):
            return ""

        raw = url.strip()
        if not raw:
            return ""

        # Ensure urlparse sees a netloc for bare domains.
        candidate = raw
        if "://" not in candidate and candidate.startswith("www."):
            candidate = "https://" + candidate

        try:
            parsed = urlparse(candidate)
        except Exception:
            return raw.lower()

        scheme = (parsed.scheme or "").lower()
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""

        # Some inputs may still be path-only; fall back to lowercased raw.
        if not netloc and scheme:
            return raw.lower()

        # Normalize common noise
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        # Drop fragments and common tracking query params
        try:
            pairs = parse_qsl(parsed.query or "", keep_blank_values=True)
        except Exception:
            pairs = []

        filtered: List[Tuple[str, str]] = []
        for k, v in pairs:
            kl = (k or "").lower()
            if kl.startswith("utm_"):
                continue
            if kl in {"gclid", "fbclid", "mc_cid", "mc_eid", "igshid"}:
                continue
            filtered.append((k, v))

        # Keep ordering stable for fingerprinting
        filtered.sort(key=lambda kv: ((kv[0] or "").lower(), kv[1] or ""))
        query = urlencode(filtered, doseq=True) if filtered else ""

        # Prefer https in fingerprints when scheme is missing/unknown
        if not scheme:
            scheme = "https"

        try:
            return urlunparse((scheme, netloc, path, "", query, ""))
        except Exception:
            # Fallback to raw lowercased
            return raw.lower()

    def _citation_fingerprint(self, item: Any) -> Optional[str]:
        """Build a stable fingerprint for a citation-like object."""
        doi = ""
        url = ""
        title = ""

        if isinstance(item, dict):
            doi = str(item.get("doi") or item.get("DOI") or "").strip().lower()
            url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
            title = str(item.get("title") or item.get("name") or "").strip().lower()
        else:
            text = str(item).strip()
            if not text:
                return None
            # Heuristic: treat obvious URLs as URLs, otherwise as title
            if text.startswith("http://") or text.startswith("https://") or text.startswith("www."):
                url = text
            else:
                title = text.lower()

        if doi:
            return f"doi:{doi}"

        canon = self._canonicalize_url(url)
        if canon:
            return f"url:{canon}"

        if title:
            return f"title:{title}"

        return None

    def _get_seen_citation_fps(self, goal: str) -> set:
        """Return (and seed) the per-goal set of seen citation fingerprints."""
        if not isinstance(goal, str) or not goal:
            goal = "_"

        seen = self._seen_citation_fps.get(goal)
        if seen is not None:
            return seen

        seen = set()

        # Seed from recent history so restarts/resumes don't re-add the same
        # citations over and over.
        try:
            history_rows = self._get_recent_history_for_goal(goal, limit=200)
            for row in history_rows or []:
                if not isinstance(row, dict):
                    continue
                for k in ("citations", "sources", "source_list"):
                    items = row.get(k)
                    if not isinstance(items, list):
                        continue
                    for it in items:
                        fp = self._citation_fingerprint(it)
                        if fp:
                            seen.add(fp)
        except Exception:
            pass

        self._seen_citation_fps[goal] = seen
        return seen

    def _tag_citations(
        self,
        citations: List[Dict[str, Any]],
        *,
        goal: str,
        query: str,
        channel: str,
        phase: str,
    ) -> List[Dict[str, Any]]:
        """Ensure every citation has minimal provenance fields."""
        stamped: List[Dict[str, Any]] = []
        # Robust de-dupe across cycles: only emit/log citations we have not
        # already seen for this goal.
        seen_fps = self._get_seen_citation_fps(goal)
        created_at = datetime.utcnow().isoformat() + "Z"

        for c in citations or []:
            if not isinstance(c, dict):
                c = {"title": str(c)}
            c = dict(c)
            c.setdefault("goal", goal)
            c.setdefault("query", query)
            c.setdefault("channel", channel)
            c.setdefault("phase", phase)
            c.setdefault("created_at", created_at)
            if not c.get("source"):
                c["source"] = channel

            fp = self._citation_fingerprint(c)
            if fp and fp in seen_fps:
                continue
            if fp:
                seen_fps.add(fp)
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
        """Return recent history rows for this goal if the store supports it."""
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
                    filtered = [r for r in full if isinstance(r, dict) and r.get("goal") == goal]
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
        """Compute short term RYE gradient, equilibrium status, and stability score."""
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

        history = self._get_recent_history_for_goal(goal, limit=max(self.rye_window_size, 10))

        rye_values: List[float] = []
        for row in history:
            if not isinstance(row, dict):
                continue
            v = _first_numeric(row, ("RYE", "rye", "rye_value"))
            if v is not None:
                rye_values.append(v)

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
        std_val = var_val**0.5

        result["rye_window_mean"] = mean_val
        result["rye_window_std"] = std_val

        # Simple gradient estimate using last and first in window
        gradient = (window[-1] - window[0]) / max(1, n - 1) if n > 1 else 0.0
        result["rye_gradient"] = gradient

        # Equilibrium heuristics
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

        # Tiny domain adjustment
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
        """Heuristic breakthrough score on a 0 to 1 scale."""
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
        mean_window = equilibrium_info.get("rye_window_mean")
        if mean_window is None:
            mean_window = rye_val
        std_window = equilibrium_info.get("rye_window_std")
        if std_window is None:
            std_window = 0.0

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
        total_before = sum(int(v) for v in issue_code_counts_before.values()) if issue_code_counts_before else 0
        total_after = sum(int(v) for v in issue_code_counts_after.values()) if issue_code_counts_after else 0

        reduction = (max(0.0, float(total_before - total_after)) / float(total_before)) if total_before > 0 else 0.0

        if reduction >= 0.7 and total_after <= 3:
            score += 0.2
            flags.append("large_issue_reduction")
        elif reduction >= 0.4:
            score += 0.1

        # 3. Citation richness
        unique_sources = set()
        for c in citations:
            if not isinstance(c, dict):
                continue
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
        dom_lower = (domain or "general").lower()
        if dom_lower in {"longevity", "math"}:
            score *= 1.05

        score = max(0.0, min(1.0, score))
        info["breakthrough_score"] = score
        info["flags"] = flags

        return info

    # ------------------------------------------------------------------
    # Search energy control helpers
    # ------------------------------------------------------------------
    def _compute_search_energy(
        self,
        avg_rye: Optional[float],
        total_cycles_for_goal: int,
        stage_tag: str,
        maintenance_mode: bool,
        domain: str,
    ) -> float:
        """Compute a per cycle search_energy multiplier for web usage."""
        base = self.search_energy_base

        if self.search_energy_mode == "fixed":
            return max(self.search_energy_min, min(self.search_energy_max, base))

        energy = base
        dom = (domain or "general").lower()

        # Very early cycles: gently favor higher exploration.
        if avg_rye is None or total_cycles_for_goal < 3:
            energy *= 1.05
        else:
            if avg_rye < 0.4:
                energy *= 1.15
            elif avg_rye > 0.8 and total_cycles_for_goal >= 15:
                energy *= 0.75

        if stage_tag == "verify":
            energy *= 0.8

        if maintenance_mode:
            energy *= 0.7

        if dom == "longevity" and avg_rye is not None and avg_rye < 0.6:
            energy *= 1.1

        energy = max(self.search_energy_min, min(self.search_energy_max, energy))
        return float(energy)

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
        msil_mode: str = "v1",
        msil_track_mode: str = "single",
        rye_mode: str = "v3",
        stage: str = "idea",
        hallmark: Optional[str] = None,
        subgoal: Optional[str] = None,
        replay_buffer: Optional[Any] = None,
        curriculum_state: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        swarm_profile: Optional[Dict[str, Any]] = None,
        deadline_ts: Optional[float] = None,
        experiment_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single TGRM cycle with crash protection.

        The engine worker marks a job as *error* if an exception escapes the agent loop.
        In large swarms, rare edge cases (tool errors, serialization issues, unexpected
        payloads) can bubble up. This wrapper keeps the worker alive by returning a
        structured error payload instead of raising.
        """
        _kwargs = {
            "goal": goal,
            "cycle_index": cycle_index,
            "role": role,
            "source_controls": source_controls,
            "pdf_bytes": pdf_bytes,
            "biomarker_snapshot": biomarker_snapshot,
            "domain": domain,
            "msil_mode": msil_mode,
            "msil_track_mode": msil_track_mode,
            "rye_mode": rye_mode,
            "stage": stage,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "replay_buffer": replay_buffer,
            "curriculum_state": curriculum_state,
            "run_id": run_id,
            "agent_id": agent_id,
            "swarm_profile": swarm_profile,
            "deadline_ts": deadline_ts,
            "experiment_mode": experiment_mode,
        }
        try:
            return self._run_cycle_impl(**_kwargs)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()

            try:
                logger.exception("TGRMLoop.run_cycle crashed: %s", err)
            except Exception:
                pass

            safe_domain = domain or getattr(self, "config", {}).get("domain") or "general"
            summary = {
                "cycle_index": cycle_index,
                "role": role,
                "domain": safe_domain,
                "goal": goal,
                "stage": stage,
                "hallmark": hallmark,
                "subgoal": subgoal,
                "RYE": 0.0,
                "delta_R": 0.0,
                "Energy": 0.0,
                "repairs": [],
                "notes": [f"[ERROR] {err}"],
                "hypotheses": [],
                "errors": 1,
            }
            log = {
                "error": err,
                "traceback": tb,
                "tool_usage": {},
            }
            return {
                "summary": summary,
                "log": log,
                "tool_stats": {"errors": 1},
                "citations": [],
            }

    def _run_cycle_impl(
        self,
        goal: str,
        cycle_index: int,
        role: str = "agent",
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        msil_mode: str = "v1",
        msil_track_mode: str = "single",
        rye_mode: str = "v3",
        stage: str = "idea",
        hallmark: Optional[str] = None,
        subgoal: Optional[str] = None,
        replay_buffer: Optional[Any] = None,
        curriculum_state: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        swarm_profile: Optional[Dict[str, Any]] = None,
        deadline_ts: Optional[float] = None,
        experiment_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one TGRM cycle for a given research goal."""
        cycle_started_ts = time.time()
        cycle_started_iso = datetime.utcfromtimestamp(cycle_started_ts).isoformat() + "Z"

        src_ctrl = self._normalise_source_controls(source_controls)
        domain_tag = domain or "general"

        stage_tag = (stage or "idea").lower()
        if stage_tag not in {"idea", "verify"}:
            stage_tag = "idea"

        interrupted: bool = False
        stop_reason: Optional[str] = None

        def _empty_stats() -> Dict[str, Any]:
            return {
                "web_calls": 0,
                "pubmed_calls": 0,
                "semantic_calls": 0,
                "pdf_ingestions": 0,
                "contradictions_resolved": 0,
                "sources_used": 0,
                "browser_actions": 0,
                "code_execs": 0,
                "data_loads": 0,
                "interrupted": False,
                "stop_reason": None,
            }

        tool_usage: ToolUsage = self._usage_factory()

        # Long run context: fetch RYE stats for this goal if available
        avg_rye: Optional[float] = None
        total_cycles_for_goal: int = 0
        try:
            if hasattr(self.memory_store, "get_rye_stats"):
                avg, _min_rye, _max_rye, count = self.memory_store.get_rye_stats(goal=goal)  # type: ignore[attr-defined]
                avg_rye = avg
                total_cycles_for_goal = count
        except Exception:
            avg_rye = None
            total_cycles_for_goal = 0

        cycle_number_for_goal = (total_cycles_for_goal or 0) + 1

        maintenance_mode = bool(
            self.tgrm_level >= 1
            and avg_rye is not None
            and total_cycles_for_goal >= 20
            and avg_rye >= 0.8
        )

        # Dynamic search energy for this cycle
        self.search_energy = self._compute_search_energy(
            avg_rye=avg_rye,
            total_cycles_for_goal=total_cycles_for_goal,
            stage_tag=stage_tag,
            maintenance_mode=maintenance_mode,
            domain=domain_tag,
        )

        phases: Dict[str, Any] = {
            "stage": stage_tag,
            "hallmark": hallmark,
            "subgoal": subgoal,
            "curriculum_state": curriculum_state or {},
            "search_energy": self.search_energy,
            "msil_mode": msil_mode,
            "msil_track_mode": msil_track_mode,
            "rye_mode": rye_mode,
            "experiment_mode": experiment_mode,
        }

        # TEST phase
        prior_notes = self.memory_store.get_notes(goal)
        status_report = self._test(goal, prior_notes)
        phases["test_before"] = {
            "known_notes_count": status_report.get("known_notes_count", 0),
            "approx_citation_markers": status_report.get("approx_citation_markers"),
            "marker_counts": status_report.get("marker_counts", {}),
        }

        # DETECT phase
        issues, issue_descriptions = self._detect(status_report, domain=domain_tag)
        phases["detect_before"] = {
            "issue_codes": issues,
            "issue_descriptions": issue_descriptions,
        }

        # Issue counts (use marker_counts for question/todo/contradiction)
        issue_code_counts_before: Dict[str, int] = {}
        marker_counts_before = status_report.get("marker_counts", {}) or {}
        if isinstance(marker_counts_before, dict):
            qn = int(marker_counts_before.get("question_lines", 0) or 0)
            td = int(marker_counts_before.get("todo_lines", 0) or 0)
            cd = int(marker_counts_before.get("contradiction_lines", 0) or 0)
        else:
            qn, td, cd = 0, 0, 0

        if "question_mark" in issues:
            issue_code_counts_before["question_mark"] = qn if qn > 0 else 1
        if "todo_item" in issues:
            issue_code_counts_before["todo_item"] = td if td > 0 else 1
        if "contradiction" in issues:
            issue_code_counts_before["contradiction"] = cd if cd > 0 else 1

        # Count remaining issue types as presence=1
        for code in issues:
            if code in {"question_mark", "todo_item", "contradiction"}:
                continue
            issue_code_counts_before[code] = max(1, int(issue_code_counts_before.get(code, 0) or 0))

        domain_issue_codes = [
            "missing_biomarkers",
            "missing_mechanisms",
            "missing_formalism",
            "missing_connections",
        ]
        domain_issue_flags_before = {code: (code in issues) for code in domain_issue_codes}

        has_questions_before = "question_mark" in issues
        has_todos_before = "todo_item" in issues
        has_contradictions_before = "contradiction" in issues

        # If deadline already hit, skip heavy work
        if self._deadline_hit(deadline_ts):
            interrupted = True
            stop_reason = "time_limit"

            repair_actions: List[Dict[str, str]] = []
            notes_added: List[str] = []
            citations: List[Dict[str, Any]] = []
            stats: Dict[str, Any] = _empty_stats()

            new_notes = self.memory_store.get_notes(goal)
            new_status_report = self._test(goal, new_notes)
            issues_after, issue_descriptions_after = self._detect(new_status_report, domain=domain_tag)

            phases["test_after"] = {
                "known_notes_count": new_status_report.get("known_notes_count", len(new_notes)),
                "marker_counts": new_status_report.get("marker_counts", {}),
            }
            phases["detect_after"] = {"issue_codes": issues_after}

            issue_code_counts_after: Dict[str, int] = {}
            marker_counts_after = new_status_report.get("marker_counts", {}) or {}
            if isinstance(marker_counts_after, dict):
                qn_a = int(marker_counts_after.get("question_lines", 0) or 0)
                td_a = int(marker_counts_after.get("todo_lines", 0) or 0)
                cd_a = int(marker_counts_after.get("contradiction_lines", 0) or 0)
            else:
                qn_a, td_a, cd_a = 0, 0, 0

            if "question_mark" in issues_after:
                issue_code_counts_after["question_mark"] = qn_a if qn_a > 0 else 1
            if "todo_item" in issues_after:
                issue_code_counts_after["todo_item"] = td_a if td_a > 0 else 1
            if "contradiction" in issues_after:
                issue_code_counts_after["contradiction"] = cd_a if cd_a > 0 else 1

            for code in issues_after:
                if code in {"question_mark", "todo_item", "contradiction"}:
                    continue
                issue_code_counts_after[code] = max(1, int(issue_code_counts_after.get(code, 0) or 0))

            domain_issue_flags_after = {code: (code in issues_after) for code in domain_issue_codes}
            has_questions_after = "question_mark" in issues_after
            has_todos_after = "todo_item" in issues_after
            has_contradictions_after = "contradiction" in issues_after

        else:
            # REPAIR phase
            repair_actions, notes_added, citations, stats = self._repair(
                goal=goal,
                issues=issues,
                descriptions=issue_descriptions,
                role=role,
                source_controls=src_ctrl,
                pdf_bytes=pdf_bytes,
                maintenance_mode=maintenance_mode,
                domain=domain_tag,
                tool_usage=tool_usage,
                deadline_ts=deadline_ts,
                cycle_index=cycle_index,
            )

            if stats.get("interrupted"):
                interrupted = True
                stop_reason = stats.get("stop_reason") or "time_limit"

            phases["repair"] = {
                "handled_issue_codes": [a.get("issue") for a in repair_actions],
                "notes_added_count": len(notes_added),
                "citations_added_count": len(citations),
                "interrupted": interrupted,
                "stop_reason": stop_reason,
            }

            # VERIFY phase
            new_notes = self.memory_store.get_notes(goal)
            new_status_report = self._test(goal, new_notes)
            issues_after, issue_descriptions_after = self._detect(new_status_report, domain=domain_tag)

            phases["test_after"] = {
                "known_notes_count": new_status_report.get("known_notes_count", len(new_notes)),
                "marker_counts": new_status_report.get("marker_counts", {}),
            }
            phases["detect_after"] = {"issue_codes": issues_after}

            issue_code_counts_after = {}
            marker_counts_after = new_status_report.get("marker_counts", {}) or {}
            if isinstance(marker_counts_after, dict):
                qn_a = int(marker_counts_after.get("question_lines", 0) or 0)
                td_a = int(marker_counts_after.get("todo_lines", 0) or 0)
                cd_a = int(marker_counts_after.get("contradiction_lines", 0) or 0)
            else:
                qn_a, td_a, cd_a = 0, 0, 0

            if "question_mark" in issues_after:
                issue_code_counts_after["question_mark"] = qn_a if qn_a > 0 else 1
            if "todo_item" in issues_after:
                issue_code_counts_after["todo_item"] = td_a if td_a > 0 else 1
            if "contradiction" in issues_after:
                issue_code_counts_after["contradiction"] = cd_a if cd_a > 0 else 1

            for code in issues_after:
                if code in {"question_mark", "todo_item", "contradiction"}:
                    continue
                issue_code_counts_after[code] = max(1, int(issue_code_counts_after.get(code, 0) or 0))

            domain_issue_flags_after = {code: (code in issues_after) for code in domain_issue_codes}
            has_questions_after = "question_mark" in issues_after
            has_todos_after = "todo_item" in issues_after
            has_contradictions_after = "contradiction" in issues_after

        # Hypothesis generation
        max_h = 3 if self.tgrm_level == 1 else 5
        raw_hypotheses = generate_hypotheses(goal, new_notes, citations, max_hypotheses=max_h)

        hypotheses: List[Dict[str, Any]] = []
        candidate_hypotheses: List[Dict[str, Any]] = []

        for idx, h in enumerate(raw_hypotheses):
            if isinstance(h, str):
                text = h
                conf = None
            elif isinstance(h, dict):
                text = str(h.get("text") or h.get("title") or "")
                conf = _first_numeric(h, ("confidence", "score"))
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

            try:
                self.memory_store.add_hypothesis(goal, text, score=conf)
            except Exception:
                pass

        candidate_interventions = self._extract_candidate_interventions(
            goal=goal,
            domain=domain_tag,
            notes=new_notes,
            citations=citations,
        )

        # Metrics (Reparodynamics: delta_R / E)
        issues_before_count = sum(int(v) for v in issue_code_counts_before.values()) if issue_code_counts_before else 0
        issues_after_count = sum(int(v) for v in issue_code_counts_after.values()) if issue_code_counts_after else 0

        delta_r_components = {
            "issue_reduction": max(0, issues_before_count - issues_after_count),
            "repairs_applied": len(repair_actions),
            "hypotheses": len(hypotheses),
            "sources_used": stats.get("sources_used", 0),
            "contradictions_resolved": stats.get("contradictions_resolved", 0),
        }
        delta_r = compute_delta_r(
            issues_before=issues_before_count,
            issues_after=issues_after_count,
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
            tokens_estimate=getattr(tool_usage, "approx_tokens", 0),
            swarm_size=(swarm_profile or {}).get("swarm_size"),
            swarm_layer=(swarm_profile or {}).get("layer"),
        )
        rye_value = compute_rye(delta_r, energy_e)

        equilibrium_info = self._compute_rye_gradient_and_equilibrium(
            goal=goal,
            current_rye=rye_value,
            delta_r=delta_r,
            energy_e=energy_e,
            domain=domain_tag,
        )

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
                        "experiment_mode": experiment_mode,
                        "msil_mode": msil_mode,
                        "msil_track_mode": msil_track_mode,
                        "rye_mode": rye_mode,
                    },
                )
            except Exception:
                stability_snapshot = None

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
                    experiment_mode=experiment_mode,
                    msil_mode=msil_mode,
                    msil_track_mode=msil_track_mode,
                    rye_mode=rye_mode,
                )
            except Exception:
                discovery_snapshot = None

        short_view = {
            "cycle": cycle_index,
            "cycle_number_for_goal": cycle_number_for_goal,
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
            "search_energy": self.search_energy,
            "issues_before": issues,
            "issues_after": issues_after,
            "repairs": [a.get("description", "") for a in repair_actions][:5],
            "delta_R": delta_r,
            "energy_E": energy_e,
            "RYE": rye_value,
            "equilibrium_label": equilibrium_info.get("equilibrium_label"),
            "breakthrough_score": breakthrough_info.get("breakthrough_score"),
            "stability_index": (stability_snapshot or {}).get("stability_index"),
            "recovery_momentum": (stability_snapshot or {}).get("recovery_momentum"),
            "discovery_tier": (discovery_snapshot or {}).get("tier"),
            "interrupted": interrupted,
            "stop_reason": stop_reason,
            "experiment_mode": experiment_mode,
            "msil_mode": msil_mode,
            "msil_track_mode": msil_track_mode,
            "rye_mode": rye_mode,
            "timestamp": cycle_started_iso,
            "ts_utc": cycle_started_iso,
            "ts_epoch": cycle_started_ts,
        }

        meta_signals = {
            "cycle": cycle_index,
            "cycle_number_for_goal": cycle_number_for_goal,
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
            "interrupted": interrupted,
            "stop_reason": stop_reason,
            "search_energy": self.search_energy,
            "experiment_mode": experiment_mode,
            "msil_mode": msil_mode,
            "msil_track_mode": msil_track_mode,
            "rye_mode": rye_mode,
            "timestamp": cycle_started_iso,
            "ts_utc": cycle_started_iso,
            "ts_epoch": cycle_started_ts,
        }

        cycle_summary: Dict[str, Any] = {
            "cycle": cycle_index,
            "cycle_number_for_goal": cycle_number_for_goal,
            "timestamp": cycle_started_iso,
            "ts_utc": cycle_started_iso,
            "ts_epoch": cycle_started_ts,
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
            "msil_mode": msil_mode,
            "msil_track_mode": msil_track_mode,
            "rye_mode": rye_mode,
            "experiment_mode": experiment_mode,
            # Issues before and after
            "issues_before": issue_descriptions,
            "issues_after": issues_after,  # kept for backwards compatibility (codes)
            "issue_descriptions_before": issue_descriptions,
            "issue_descriptions_after": issue_descriptions_after,
            "issue_codes_before": issues,
            "issue_codes_after": issues_after,
            "issue_code_counts_before": issue_code_counts_before,
            "issue_code_counts_after": issue_code_counts_after,
            "implied_issue_count_before": issues_before_count,
            "implied_issue_count_after": issues_after_count,
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
            "Energy": energy_e,
            "RYE": rye_value,
            "search_energy": self.search_energy,
            # RYE gradient and equilibrium and breakthrough
            "equilibrium": equilibrium_info,
            "breakthrough": breakthrough_info,
            # Stability kernel and discovery manager snapshots
            "stability": stability_snapshot,
            "discovery": discovery_snapshot,
            # Tool usage details for this cycle
            "tool_usage": {
                "web_calls": getattr(tool_usage, "web_calls", 0),
                "browser_actions": getattr(tool_usage, "browser_actions", 0),
                "code_execs": getattr(tool_usage, "code_execs", 0),
                "sql_queries": getattr(tool_usage, "sql_queries", 0),
                "data_loads": getattr(tool_usage, "data_loads", 0),
                "approx_tokens": getattr(tool_usage, "approx_tokens", 0),
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
            # Interrupt flags
            "interrupted": interrupted,
            "stop_reason": stop_reason,
        }

        replay_item_ids = self._log_replay_candidate(cycle_summary, replay_buffer)
        self._tag_cycle_metadata(
            cycle_summary,
            hallmark=hallmark,
            subgoal=subgoal,
            stage=stage_tag,
            curriculum_state=curriculum_state,
            replay_item_ids=replay_item_ids,
        )

        self.memory_store.log_cycle(cycle_summary)

        human_summary = {
            "cycle": cycle_index,
            "cycle_number_for_goal": cycle_number_for_goal,
            "timestamp": cycle_started_iso,
            "ts_utc": cycle_started_iso,
            "ts_epoch": cycle_started_ts,
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
            "issue_descriptions_after": issue_descriptions_after,
            "issue_codes_before": issues,
            "issue_codes_after": issues_after,
            "issue_code_counts_before": issue_code_counts_before,
            "issue_code_counts_after": issue_code_counts_after,
            "repairs": [a.get("description", "") for a in repair_actions],
            "delta_R": delta_r,
            "energy_E": energy_e,
            "Energy": energy_e,
            "RYE": rye_value,
            "search_energy": self.search_energy,
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
            "interrupted": interrupted,
            "stop_reason": stop_reason,
            "experiment_mode": experiment_mode,
            "msil_mode": msil_mode,
            "msil_track_mode": msil_track_mode,
            "rye_mode": rye_mode,
        }

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
        """Evaluate the current state of research for this goal."""
        known_notes_count = len(notes)

        citation_markers = 0
        if self.tgrm_level >= 2:
            for note in notes:
                content = str(note.get("content", "") or "")
                if "[" in content and "]" in content:
                    citation_markers += 1

        marker_counts = self._scan_notes_for_basic_markers(notes)

        report: Dict[str, Any] = {
            "goal": goal,
            "known_notes_count": known_notes_count,
            "notes": notes,
            "marker_counts": marker_counts,
        }
        if self.tgrm_level >= 2:
            report["approx_citation_markers"] = citation_markers

        return report

    def _detect(
        self,
        status_report: Dict[str, Any],
        domain: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Identify issues or gaps to be repaired."""
        issues: List[str] = []
        descriptions: List[str] = []
        notes = status_report.get("notes", [])

        marker_counts = status_report.get("marker_counts") or {}
        if not isinstance(marker_counts, dict):
            marker_counts = {}

        question_lines = int(marker_counts.get("question_lines", 0) or 0)
        todo_lines = int(marker_counts.get("todo_lines", 0) or 0)
        contradiction_lines = int(marker_counts.get("contradiction_lines", 0) or 0)

        if not notes:
            issues.append("no_notes")
            descriptions.append("No prior notes found; initial research required.")
        else:
            if question_lines > 0:
                issues.append("question_mark")
                descriptions.append(f"{question_lines} unanswered question line(s) detected in notes.")
            if todo_lines > 0:
                issues.append("todo_item")
                descriptions.append(f"{todo_lines} TODO line(s) detected in notes; missing information.")
            if contradiction_lines > 0:
                issues.append("contradiction")
                descriptions.append(f"{contradiction_lines} contradiction marker line(s) detected; needs resolution.")

        if self.tgrm_level >= 2 and notes:
            citation_markers = int(status_report.get("approx_citation_markers", 0) or 0)
            ratio = citation_markers / max(1, len(notes))
            if ratio < self.under_cited_ratio:
                issues.append("under_cited")
                descriptions.append(
                    f"Evidence base may be thin: citation marker ratio {ratio:.2f} below {self.under_cited_ratio:.2f}; "
                    "prioritize finding stronger primary sources."
                )

        if self.tgrm_level >= 3 and notes and domain:
            dom = str(domain).lower()
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
        text = "\n".join(str(n.get("content", "") or "") for n in notes)

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
            if "missing_biomarkers" not in issues:
                issues.append("missing_biomarkers")
                descriptions.append(
                    "Longevity notes lack explicit biomarker discussion; identify measurable markers and how interventions affect them."
                )

        if all(
            kw.lower() not in text.lower()
            for kw in [
                "mechanism",
                "pathway",
                "mtor",
                "autophagy",
                "senescence",
                "nad+",
                "hallmarks of aging",
            ]
        ):
            if "missing_mechanisms" not in issues:
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
        text = "\n".join(str(n.get("content", "") or "") for n in notes)

        has_definition = "definition" in text.lower()
        has_theorem = "theorem" in text.lower() or "lemma" in text.lower()
        if not (has_definition and has_theorem):
            if "missing_formalism" not in issues:
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
            if "missing_connections" not in issues:
                issues.append("missing_connections")
                descriptions.append(
                    "Connections to existing frameworks are thin; relate Reparodynamics to known stability, control, or information theories."
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
        deadline_ts: Optional[float] = None,
        cycle_index: Optional[int] = None,
    ) -> Tuple[
        List[Dict[str, str]],
        List[str],
        List[Dict[str, Any]],
        Dict[str, Any],
    ]:
        """Apply repairs to address detected issues."""
        repair_actions: List[Dict[str, str]] = []
        notes_added: List[str] = []
        citations: List[Dict[str, Any]] = []

        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
            "interrupted": False,
            "stop_reason": None,
        }

        def _merge_issue_stats(issue_stats: Dict[str, Any]) -> None:
            """Merge per-issue stats safely (no None+int, no stop_reason summation)."""
            if not isinstance(issue_stats, dict):
                return
            # numeric counters
            numeric_keys = {
                "web_calls",
                "pubmed_calls",
                "semantic_calls",
                "pdf_ingestions",
                "contradictions_resolved",
                "sources_used",
                "browser_actions",
                "code_execs",
                "data_loads",
            }
            for k in numeric_keys:
                if k in issue_stats:
                    try:
                        stats[k] = int(stats.get(k, 0) or 0) + int(issue_stats.get(k, 0) or 0)
                    except Exception:
                        # best-effort: leave unchanged if types are weird
                        pass

            if issue_stats.get("interrupted"):
                stats["interrupted"] = True
                # Prefer a specific reason if present; otherwise keep existing, then fallback.
                reason = issue_stats.get("stop_reason")
                if isinstance(reason, str) and reason:
                    stats["stop_reason"] = reason
                elif stats.get("stop_reason") is None:
                    stats["stop_reason"] = "time_limit"

        if maintenance_mode:
            base_max_issues = 2
        else:
            base_max_issues = 5

        if self.ultra_speed and not maintenance_mode:
            base_max_issues += 2

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
        # Evidence expansion: keep accumulating citations even when explicit question markers are sparse.
        citation_target = self._infer_citation_target(goal, default=30)
        citation_count_before = self._count_real_citations(goal)
        can_expand = any(bool(source_controls.get(k)) for k in ("web", "pubmed", "semantic"))
        is_research_role = role_lower.startswith("explorer") or role_lower.startswith("researcher")
        needs_more_citations = citation_count_before < citation_target

        if can_expand and is_research_role and needs_more_citations and not maintenance_mode:
            if "citation_expansion" not in issues_to_handle:
                expansion_query = self._pick_evidence_query(goal, domain, role, cycle_index)
                if len(issues_to_handle) == 0:
                    issues_to_handle = ["citation_expansion"]
                    descriptions_to_handle = [expansion_query]
                elif len(issues_to_handle) < max_issues:
                    issues_to_handle = list(issues_to_handle) + ["citation_expansion"]
                    descriptions_to_handle = list(descriptions_to_handle) + [expansion_query]

        # Keep counts available for downstream logging/telemetry.
        # (Do not embed counts into the search query itself.)
        stats["citation_target"] = citation_target
        stats["citation_count_before"] = citation_count_before


        def _log_citations(cites: List[Dict[str, Any]]) -> None:
            for c in cites:
                if not isinstance(c, dict):
                    continue
                # Drop stub/error placeholders so they don't count toward evidence targets.
                title = str(c.get("title") or "").strip()
                url = str(c.get("url") or c.get("link") or "").strip()
                if (not url) or title.lower().startswith("[stub]"):
                    continue

                # Attach minimal provenance for easier debugging/UX.
                c.setdefault("role", role)
                if domain is not None:
                    c.setdefault("domain", domain)
                if cycle_index is not None:
                    c.setdefault("cycle", cycle_index)

                tool_name = c.get("tool_name") or c.get("tool") or c.get("source")
                try:
                    self.memory_store.add_citation(
                        goal,
                        c,
                        run_id=self.run_id,
                        role=role,
                        domain=domain,
                        cycle_index=cycle_index,
                        tool_name=tool_name,
                    )
                except Exception:
                    # Avoid failing the cycle if citation logging fails.
                    pass

        for issue, desc in zip(issues_to_handle, descriptions_to_handle):
            if self._deadline_hit(deadline_ts):
                stats["interrupted"] = True
                stats["stop_reason"] = "time_limit"
                break

            if issue == "no_notes":
                note_text, new_cites, issue_stats = self._initial_research(
                    goal=goal,
                    role=role,
                    source_controls=source_controls,
                    pdf_bytes=pdf_bytes,
                    domain=domain,
                    maintenance_mode=maintenance_mode,
                    tool_usage=tool_usage,
                    deadline_ts=deadline_ts,
                )
                note_text = self._cap_note_text(note_text)
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
                _merge_issue_stats(issue_stats)

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

                if issue_stats.get("interrupted"):
                    break

            elif is_research_role and issue != "under_cited":
                questions = self._extract_questions(goal, issue_type=issue)

                sc_for_issue = source_controls
                provider: Optional[str] = None
                if issue == "citation_expansion":
                    provider = self._select_expansion_provider(source_controls, role, cycle_index)
                elif self.tgrm_level >= 3 and sum(1 for k in ("web", "pubmed", "semantic") if source_controls.get(k)) > 1:
                    # In swarm/high-evidence mode, rotate providers per issue to reduce rate limits.
                    provider = self._select_expansion_provider(source_controls, f"{role}:{issue}", cycle_index)
                if provider:
                    sc_for_issue = self._only_source_controls(source_controls, provider)
                    stats.setdefault("provider_rotation", []).append({"issue": issue, "provider": provider})

                note_text, new_cites, issue_stats = self._targeted_research(
                    goal=goal,
                    role=role,
                    issue=issue,
                    issue_description=desc,
                    source_controls=sc_for_issue,
                    questions=questions,
                    maintenance_mode=maintenance_mode,
                    domain=domain,
                    tool_usage=tool_usage,
                    deadline_ts=deadline_ts,
                )

                note_text = self._cap_note_text(note_text)
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
                _merge_issue_stats(issue_stats)

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

                if issue_stats.get("interrupted"):
                    break

            elif issue == "contradiction":
                note_text, new_cites, issue_stats = self._resolve_contradictions(
                    goal=goal,
                    role=role,
                    source_controls=source_controls,
                    domain=domain,
                    maintenance_mode=maintenance_mode,
                    tool_usage=tool_usage,
                    deadline_ts=deadline_ts,
                )
                note_text = self._cap_note_text(note_text)
                try:
                    self.memory_store.add_note(goal, note_text, role=role)
                except Exception:
                    pass
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Attempted contradiction resolution with additional sources.",
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)
                _merge_issue_stats(issue_stats)

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

                if issue_stats.get("interrupted"):
                    break

            elif issue == "under_cited":
                note_text, new_cites, issue_stats = self._strengthen_citations(
                    goal=goal,
                    role=role,
                    source_controls=source_controls,
                    domain=domain,
                    tool_usage=tool_usage,
                    deadline_ts=deadline_ts,
                )
                note_text = self._cap_note_text(note_text)
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
                _merge_issue_stats(issue_stats)

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

                if issue_stats.get("interrupted"):
                    break

            elif issue in {"missing_biomarkers", "missing_mechanisms", "missing_formalism", "missing_connections"}:
                note_text, new_cites, issue_stats = self._domain_gap_research(
                    goal=goal,
                    role=role,
                    issue=issue,
                    description=desc,
                    source_controls=source_controls,
                    domain=domain,
                    tool_usage=tool_usage,
                    deadline_ts=deadline_ts,
                )
                note_text = self._cap_note_text(note_text)
                try:
                    self.memory_store.add_note(goal, note_text, role=role)
                except Exception:
                    pass
                repair_actions.append(
                    {
                        "issue": issue,
                        "description": f"[{role}] Filled domain gap: {desc}",
                    }
                )
                notes_added.append(note_text)
                citations.extend(new_cites)
                _log_citations(new_cites)
                _merge_issue_stats(issue_stats)

                if tool_usage is not None:
                    tool_usage.approx_tokens += self._estimate_tokens(note_text)

                if issue_stats.get("interrupted"):
                    break

            else:
                goal_lc = (goal or "").lower()
                if issue == "question_mark" and ("citation hunt" in goal_lc or "citation" in goal_lc):
                    # In citation-hunt runs, do not create TODO issues for question marks.
                    continue

                note = (
                    f"[{role}] Encountered issue '{issue}' with description: {desc}. "
                    "This issue type is not yet fully handled; marking as TODO for future cycles."
                )
                note = self._cap_note_text(note)
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

        # Always recompute sources_used from citations to avoid double counting
        unique_sources = set()
        for c in citations:
            if not isinstance(c, dict):
                continue
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

    def _should_use_web(self, purpose: str, maintenance_mode: bool) -> bool:
        """Decide whether to perform a web search for this purpose."""
        disable_env = os.getenv("DISABLE_WEB_SEARCH", "").strip().lower()
        if disable_env in {"1", "true", "yes", "on"}:
            return False

        try:
            energy_val = float(getattr(self, "search_energy", 1.0))
        except Exception:
            energy_val = 1.0

        if energy_val <= 0.2:
            return False

        if maintenance_mode and energy_val < 0.6 and purpose in {"targeted", "strengthen", "gap_repair"}:
            return False

        return True

    def _select_web_search_params(
        self,
        role: str,
        maintenance_mode: bool,
        domain: Optional[str],
        purpose: str,
        search_energy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Hybrid mode selection of web search parameters.

        Returns a superset of common kwargs and lets _web_search filter what
        the underlying WebResearchTool.search(...) supports:
            - max_results
            - search_depth
            - topic
            - include_raw_content
            - level (legacy)
        """
        role_lower = (role or "agent").lower()
        dom = (domain or "general").lower()

        if dom in {"macro", "markets", "trading", "finance", "economics"}:
            topic = "finance"
        elif dom in {"policy", "geopolitics", "news", "world"}:
            topic = "news"
        else:
            topic = "general"

        base_level = max(1, min(self.tgrm_level, 3))
        level = base_level

        if maintenance_mode and base_level > 1:
            level = base_level - 1

        if purpose in {"initial", "gap_repair", "strengthen"}:
            if (not maintenance_mode and base_level == 3 and role_lower in {"researcher", "explorer"}):
                level = 3
        elif purpose == "targeted":
            if (not maintenance_mode and base_level >= 2 and role_lower in {"researcher", "explorer"}):
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
            search_depth = "basic"
        else:
            max_results = base_max
            search_depth = "advanced"

        if search_energy is None:
            try:
                search_energy = float(getattr(self, "search_energy", 1.0))
            except Exception:
                search_energy = 1.0
        else:
            try:
                search_energy = float(search_energy)
            except Exception:
                search_energy = 1.0

        if search_energy != 1.0:
            scaled = int(round(max_results * search_energy))
            if scaled < 1:
                scaled = 1
            max_cap = max(max_results, base_max * 2)
            max_results = max(1, min(scaled, max_cap))

        params: Dict[str, Any] = {
            "max_results": max_results,
            "topic": topic,
            "search_depth": search_depth,
            "level": level,  # legacy / optional
        }
        if level >= 3:
            params["include_raw_content"] = True

        return params

    def _web_search(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform a web search with resilience to Tavily limits and failures.

        This method prefers the unified `core_web_search` from tools.py when available,
        which incorporates concurrency limits and retry/backoff logic. It falls back
        to `self.web_tool.search` only when the unified search is unavailable.

        Args:
            query: The search query string.
            params: Optional mapping of search parameters such as max_results,
                search_depth, topic, include_raw_content, and level.

        Returns:
            A list of normalized result dictionaries with the keys: title, url,
            source, content, and raw_html. When no results are found or on
            unrecoverable errors, an empty list is returned.
        """
        # Ensure query does not exceed Tavily limits
        query = _clamp_query(query)
        # Normalize params dictionary
        params = dict(params or {})

        # Hard timeout to prevent a single hanging web call from stalling the whole cycle.
        timeout_s = _get_web_timeout_s()

        # If a unified search helper is available, use it. This helper handles
        # Tavily concurrency limits and implements retry/backoff. It returns
        # a dict with a 'results' key which we convert into our normalized
        # result format. We avoid passing ToolUsage into core_web_search here
        # because TGRM energy accounting is handled separately.
        if core_web_search is not None:
            # Extract supported parameters for core_web_search
            # Determine max_results, search_depth and topic from params; allow
            # fallback from legacy 'level'.
            max_results = params.get("max_results")
            if not isinstance(max_results, int) or max_results <= 0:
                max_results = 8

            search_depth = params.get("search_depth")
            if not isinstance(search_depth, str) or not search_depth:
                # Derive search_depth from level: level<=1 -> basic, else advanced
                level = params.get("level")
                if isinstance(level, int) and level <= 1:
                    search_depth = "basic"
                else:
                    search_depth = "advanced"

            topic = params.get("topic")
            if not isinstance(topic, str) or not topic:
                topic = "general"
            # Wrap the provider call with a hard timeout. This prevents one slow/hung
            # Tavily request from blocking the entire swarm cycle.
            def _invoke_core_search() -> Any:
                # Try the most complete signature first; degrade gracefully.
                try:
                    return core_web_search(
                        query=query,
                        max_results=max_results,
                        search_depth=search_depth,
                        topic=topic,
                    )
                except TypeError:
                    try:
                        return core_web_search(
                            query=query,
                            max_results=max_results,
                            search_depth=search_depth,
                        )
                    except TypeError:
                        return core_web_search(
                            query=query,
                            max_results=max_results,
                        )

            res = _call_with_timeout(_invoke_core_search, timeout_s=timeout_s, default=None)

            if isinstance(res, dict) and isinstance(res.get("results"), list):
                items: List[Dict[str, Any]] = []
                for item in res.get("results") or []:
                    if not isinstance(item, dict):
                        continue
                    # Normalize into expected fields. Some fields like content/raw_html
                    # may not be present; map snippet into content.
                    items.append(
                        {
                            "title": item.get("title") or "",
                            "url": item.get("url") or "",
                            "source": item.get("source") or res.get("provider") or "web",
                            "content": item.get("snippet") or item.get("content") or "",
                            "raw_html": item.get("raw_html") or "",
                        }
                    )
                return items
            # If unified search failed or returned an unexpected structure, fall
            # through to the legacy tool below.

        # Fallback to the underlying web tool. Filter params to match the tool
        # signature to prevent TypeError when different implementations accept
        # different keyword sets (e.g. level vs search_depth).
        filtered = params
        try:
            sig = inspect.signature(self.web_tool.search)  # type: ignore[attr-defined]
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_varkw:
                allowed = set(sig.parameters.keys())
                filtered = {k: v for k, v in params.items() if k in allowed}
        except Exception:
            filtered = params

        try:
            return _call_with_timeout(
                lambda: self.web_tool.search(query, **filtered),  # type: ignore[misc]
                timeout_s=timeout_s,
                default=[],
                allow_exceptions=(TypeError,),
            )
        except TypeError:
            # Last resort: try a no-kwargs call
            try:
                return _call_with_timeout(
                    lambda: self.web_tool.search(query),  # type: ignore[misc]
                    timeout_s=timeout_s,
                    default=[],
                )
            except Exception:
                return []
        except Exception:
            return []

    def _initial_research(
        self,
        goal: str,
        role: str,
        source_controls: Dict[str, bool],
        pdf_bytes: Optional[bytes],
        domain: Optional[str],
        maintenance_mode: bool,
        tool_usage: Optional[ToolUsage] = None,
        deadline_ts: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Perform the initial multi source research when there are no notes."""
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
            "interrupted": False,
            "stop_reason": None,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Initial research summary for goal:")
        note_lines.append(goal)
        note_lines.append("")

        if self._deadline_hit(deadline_ts):
            stats["interrupted"] = True
            stats["stop_reason"] = "time_limit"
            note_lines.append("Time limit reached before initial research; skipping external calls this cycle.")
            return "\n".join(note_lines), citations, stats

        first_url: Optional[str] = None

        if source_controls.get("web", True) and self._should_use_web("initial", maintenance_mode):
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=maintenance_mode,
                domain=domain,
                purpose="initial",
            )
            search_query = self._goal_to_search_query(goal, domain=domain)
            try:
                web_results = self._web_search(search_query, web_params)
            except Exception:
                web_results = []
                note_lines.append("Web search failed for initial research; continuing without web results.")
                note_lines.append("")
            else:
                stats["web_calls"] += 1
                if tool_usage is not None:
                    tool_usage.web_calls += 1

                web_summary = self.web_tool.summarize_results(web_results)
                web_cites_raw = self.web_tool.to_citations(web_results)
                web_cites = self._tag_citations(
                    web_cites_raw,
                    goal=goal,
                    query=search_query,
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

        if self._deadline_hit(deadline_ts):
            stats["interrupted"] = True
            stats["stop_reason"] = "time_limit"
            note_lines.append("Time limit hit; skipping remaining initial research sources.")
            return "\n".join(note_lines), citations, stats

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

                note_lines.append(f"Browser deep dive snippet from: {browser_result.url}")
                if getattr(browser_result, "error", None):
                    note_lines.append(f"(Browser error: {browser_result.error})")
                else:
                    snippet = getattr(browser_result, "text_snippet", "") or ""
                    note_lines.append(snippet[:2000])
                note_lines.append("")
            except Exception:
                pass

        if source_controls.get("pubmed", False) and not self._deadline_hit(deadline_ts):
            try:
                pubmed_results = self.pubmed_tool.search(self._goal_to_search_query(goal, domain=domain), max_results=5)
            except Exception:
                pubmed_results = []
                note_lines.append("PubMed search failed for initial research; continuing without PubMed results.")
                note_lines.append("")
            else:
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
                    titles = " ".join((r.get("title", "") or "") for r in pubmed_results if isinstance(r, dict))
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("PubMed sources:")
                for r in pubmed_cites:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

        if source_controls.get("semantic", False) and not self._deadline_hit(deadline_ts):
            try:
                sem_results = self.semantic_tool.search(self._goal_to_search_query(goal, domain=domain), max_results=5)
            except Exception:
                sem_results = []
                note_lines.append(
                    "Semantic Scholar search failed for initial research; continuing without Semantic Scholar results."
                )
                note_lines.append("")
            else:
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
                    titles = " ".join((r.get("title", "") or "") for r in sem_results if isinstance(r, dict))
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("Semantic Scholar sources:")
                for r in sem_cites:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

        if source_controls.get("pdf", False) and pdf_bytes and not self._deadline_hit(deadline_ts):
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
        """Extract concrete question or TODO lines from stored notes.

        Uses a per-goal TTL cache so that repeated long runs do not keep
        re-querying the same question string forever.
        """
        notes = self.memory_store.get_notes(goal)
        candidates: List[str] = []

        for note in notes:
            content = str(note.get("content", "") or "")
            for line in content.splitlines():
                line_strip = line.strip()
                if not line_strip:
                    continue
                lower = line_strip.lower()
                if issue_type == "question_mark":
                    if "?" in line_strip and len(line_strip) > 10:
                        candidates.append(line_strip)
                elif issue_type == "todo_item":
                    if "todo" in lower and len(line_strip) > 10:
                        candidates.append(line_strip)

        seen_local = set()
        unique_questions: List[str] = []
        for q in candidates:
            if q not in seen_local:
                seen_local.add(q)
                unique_questions.append(q)

        now = time.time()
        ttl = max(0.0, float(self._seen_question_ttl_s))
        seen_for_goal = self._seen_questions.setdefault(goal, {})

        fresh_questions: List[str] = []
        for q in unique_questions:
            last = seen_for_goal.get(q)
            if last is not None and (now - float(last)) < ttl:
                continue
            fresh_questions.append(q)
            seen_for_goal[q] = now

        max_q = 3 if self.tgrm_level == 1 else 5
        return fresh_questions[:max_q]

    # ---- Evidence expansion helpers -------------------------------------------------

    def _stable_role_offset(self, role: str) -> int:
        """Stable small integer derived from role string (avoids Python's salted hash)."""
        if not role:
            return 0
        acc = 0
        for ch in role:
            acc = (acc * 31 + ord(ch)) & 0xFFFFFFFF
        return acc

    def _infer_citation_target(self, goal: str, default: int = 30) -> int:
        """Infer a citation target from the prompt (e.g., 'Minimum 50 unique citations')."""
        if not goal:
            return default
        cache = getattr(self, "_goal_citation_target_cache", None)
        if isinstance(cache, dict) and goal in cache:
            try:
                return int(cache[goal])
            except Exception:
                pass

        # Try to parse "Minimum 50 unique citations" or similar.
        m = re.search(r"(?i)\bminimum\s+(\d{1,4})\s+unique\s+citations\b", goal)
        if not m:
            m = re.search(r"(?i)\bcitation\s+target\b[\s\S]{0,250}?\bminimum\s+(\d{1,4})\b", goal)
        target = default
        if m:
            try:
                target = int(m.group(1))
            except Exception:
                target = default

        # Boundaries: keep sane defaults even for malformed prompts.
        if target < 5:
            target = 5
        if target > 300:
            target = 300

        if isinstance(cache, dict):
            cache[goal] = target
        return target

    def _goal_to_search_query(self, goal: str, domain: Optional[str] = None, max_chars: int = 160) -> str:
        """Convert a long run prompt into a compact, tool-friendly search query."""
        goal = (goal or "").strip()
        domain = (domain or "").strip()

        cache = getattr(self, "_goal_query_cache", None)
        cache_key = f"{goal}\n||{domain}"
        if isinstance(cache, dict) and cache_key in cache:
            return cache[cache_key]

        topic: Optional[str] = None
        domain_line: Optional[str] = None

        if goal:
            m = re.search(r"(?im)^\s*TOPIC\s*:\s*(.+)\s*$", goal)
            if m:
                topic = m.group(1).strip()

            m2 = re.search(r"(?im)^\s*DOMAIN\s*:\s*(.+)\s*$", goal)
            if m2:
                domain_line = m2.group(1).strip()

        # Fallback: first non-empty non-heading line.
        if not topic:
            for raw in goal.splitlines():
                s = raw.strip()
                if not s:
                    continue
                # Skip obvious section headers (all caps / very short).
                if re.fullmatch(r"[A-Z0-9\s\-/|]{4,}", s) and s.isupper():
                    continue
                s = re.sub(r"^[-*\d.\s]+", "", s).strip()
                if len(s) < 4:
                    continue
                topic = s
                break

        parts: List[str] = []
        if topic:
            parts.append(topic)

        # Add domain hints if helpful.
        if domain_line:
            dom = domain_line.replace("|", " ")
            dom = dom.replace(",", " ")
            dom = re.sub(r"\s+", " ", dom).strip()
            if dom and (not topic or dom.lower() not in topic.lower()):
                parts.append(dom)

        if domain and domain.lower() not in " ".join(parts).lower():
            parts.append(domain)

        q = " ".join([p for p in parts if p]).strip()

        # Strip common orchestration tokens that pollute searches.
        if q:
            q = re.sub(r"(?i)\b(?:MODE|swarm|agents?|cycles?|run\s+config|hard\s+requirements|begin)\b", " ", q)
            q = re.sub(r"\s+", " ", q).strip()

        if not q:
            q = domain or "research"

        if len(q) > max_chars:
            q = q[:max_chars].rstrip(" -|,:;")

        if isinstance(cache, dict):
            cache[cache_key] = q
        return q

    def _build_evidence_query_bank(self, goal: str, domain: Optional[str] = None) -> List[str]:
        """Build a rotating set of search queries to avoid repeatedly retrieving the same top-N results."""
        goal = (goal or "").strip()
        domain = (domain or "").strip()

        cache = getattr(self, "_goal_query_bank_cache", None)
        cache_key = f"{goal}\n||{domain}"
        if isinstance(cache, dict) and cache_key in cache:
            return list(cache[cache_key])

        base = self._goal_to_search_query(goal, domain=domain)
        base_l = base.lower()

        queries: List[str] = []

        # If this looks like aging/longevity, use a strong geroscience query set.
        is_aging = ("aging" in base_l) or ("ageing" in base_l) or ("longevity" in base_l) or (domain.lower() in {"longevity", "aging", "ageing", "geroscience"})
        if is_aging:
            queries.extend(
                [
                    "hallmarks of aging review",
                    "cellular senescence senolytics review",
                    "inflammaging chronic inflammation aging review",
                    "mitochondrial dysfunction mitophagy aging review",
                    "autophagy proteostasis aging review",
                    "epigenetic clocks biomarkers of aging review",
                    "caloric restriction humans randomized trial systematic review",
                    "intermittent fasting time restricted eating meta-analysis",
                    "exercise mortality meta-analysis cohort study",
                    "rapamycin mTOR inhibition lifespan mammal review",
                    "metformin aging trial TAME review",
                    "NAD+ precursors nicotinamide riboside NMN randomized trial",
                    "spermidine autophagy human study systematic review",
                    "senomorphics SASP inhibitors aging review",
                    "geroscience consensus statement aging interventions",
                ]
            )

        # Pull bullet-list items from the prompt as additional subtopics.
        bullets: List[str] = []
        for raw in goal.splitlines():
            s = raw.strip()
            if s.startswith("-"):
                s = s.lstrip("-").strip()
                # Keep short-ish phrases (avoid full sentences).
                if 3 <= len(s) <= 70 and re.search(r"[A-Za-z]", s):
                    bullets.append(s)
        # De-duplicate while preserving order.
        seen = set()
        bullets_unique: List[str] = []
        for b in bullets:
            bl = b.lower()
            if bl in seen:
                continue
            seen.add(bl)
            bullets_unique.append(b)

        for b in bullets_unique[:20]:
            # Generic templates that work across tools.
            if is_aging:
                queries.append(f"{b} aging review")
            else:
                queries.append(f"{base} {b} review")

        # Always include some broad templates.
        queries.extend(
            [
                f"{base} systematic review",
                f"{base} meta-analysis",
                f"{base} randomized controlled trial",
                f"{base} consensus statement guideline",
            ]
        )

        # Final de-dupe, drop empty / too-long.
        out: List[str] = []
        seen2 = set()
        for q in queries:
            qq = re.sub(r"\s+", " ", (q or "")).strip()
            if not qq:
                continue
            if len(qq) > 220:
                qq = qq[:220].rstrip()
            key = qq.lower()
            if key in seen2:
                continue
            seen2.add(key)
            out.append(qq)

        if not out:
            out = [base]

        if isinstance(cache, dict):
            cache[cache_key] = list(out)
        return out

    def _pick_evidence_query(self, goal: str, domain: Optional[str], role: str, cycle_index: Optional[int]) -> str:
        bank = self._build_evidence_query_bank(goal, domain=domain)
        if not bank:
            return self._goal_to_search_query(goal, domain=domain)

        offset = self._stable_role_offset(role)
        idx = ((cycle_index or 0) + offset) % len(bank)
        return bank[idx]

    def _count_real_citations(self, goal: str) -> int:
        """Count unique, non-stub citations already stored for this goal."""
        try:
            entries = self.memory_store.get_citations(goal)
        except Exception:
            return 0

        seen = set()
        n = 0
        for e in entries or []:
            c = (e or {}).get("citation") or {}
            title = str(c.get("title") or "").strip()
            url = str(c.get("url") or c.get("link") or "").strip()
            if not url:
                continue
            if title.lower().startswith("[stub]"):
                continue
            fp = self._citation_fingerprint(c)
            if fp in seen:
                continue
            seen.add(fp)
            n += 1
        return n

    def _only_source_controls(self, source_controls: Dict[str, bool], provider: str) -> Dict[str, bool]:
        sc = dict(source_controls or {})
        # keep any unknown keys but disable the main providers except the one selected
        for k in ("web", "pubmed", "semantic"):
            sc[k] = bool(k == provider and sc.get(k, False))
        return sc

    def _select_expansion_provider(self, source_controls: Dict[str, bool], role: str, cycle_index: Optional[int]) -> str:
        providers: List[str] = [p for p in ("web", "pubmed", "semantic") if source_controls.get(p)]
        if not providers:
            return "web"
        offset = self._stable_role_offset(role)
        idx = ((cycle_index or 0) + offset) % len(providers)
        return providers[idx]
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
        deadline_ts: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Perform focused multi source research on open questions or TODOs."""
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
            "interrupted": False,
            "stop_reason": None,
        }

        # If we don't have explicit question seeds, fall back to a compact query derived from the goal.
        # (We only keep an empty question list in true maintenance mode.)
        if issue == "citation_expansion" and issue_description:
            questions = [str(issue_description).strip()]
        elif questions is None or (len(questions) == 0 and not maintenance_mode):
            base_query = self._goal_to_search_query(goal, domain=domain)
            questions = [f"{base_query} - focus on: {issue_description}"]

        if len(questions) == 0:
            note_lines: List[str] = []
            note_lines.append(f"[{role}] Maintenance pass on open items ({issue}) for goal:")
            note_lines.append(goal)
            note_lines.append("")
            note_lines.append(
                "No new unresolved questions were found beyond what has already been researched. "
                "This cycle performs a light consolidation of existing knowledge without new web or paper calls."
            )
            return "\n".join(note_lines), citations, stats

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Targeted research on open items ({issue}) for goal:")
        note_lines.append(goal)
        note_lines.append("")
        note_lines.append("Questions or TODOs considered:")
        for q in questions:
            note_lines.append(f"- {q}")
        note_lines.append("")

        for q in questions:
            if self._deadline_hit(deadline_ts):
                stats["interrupted"] = True
                stats["stop_reason"] = "time_limit"
                note_lines.append("Time limit hit during targeted research; stopping remaining searches.")
                break

            note_lines.append("### Question focused search:")
            note_lines.append(q)
            note_lines.append("")

            if source_controls.get("web", True) and self._should_use_web("targeted", maintenance_mode):
                web_params = self._select_web_search_params(
                    role=role,
                    maintenance_mode=maintenance_mode,
                    domain=domain,
                    purpose="targeted",
                )
                search_query = _clamp_query(q)
                try:
                    web_results = self._web_search(search_query, web_params)
                except Exception:
                    web_results = []
                    note_lines.append(
                        "Web search failed for this question; continuing without web results for this item."
                    )
                    note_lines.append("")
                else:
                    stats["web_calls"] += 1
                    if tool_usage is not None:
                        tool_usage.web_calls += 1

                    web_summary = self.web_tool.summarize_results(web_results)
                    web_cites_raw = self.web_tool.to_citations(web_results)
                    web_cites = self._tag_citations(
                        web_cites_raw,
                        goal=goal,
                        query=search_query,
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

            if self._deadline_hit(deadline_ts):
                stats["interrupted"] = True
                stats["stop_reason"] = "time_limit"
                note_lines.append("Time limit hit; skipping PubMed/Semantic Scholar for remaining items.")
                break

            if source_controls.get("pubmed", False):
                try:
                    pubmed_results = self.pubmed_tool.search(q, max_results=5)
                except Exception:
                    pubmed_results = []
                    note_lines.append(
                        "PubMed search failed for this question; continuing without PubMed results for this item."
                    )
                    note_lines.append("")
                else:
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
                        titles = " ".join((r.get("title", "") or "") for r in pubmed_results if isinstance(r, dict))
                        tool_usage.approx_tokens += self._estimate_tokens(titles)

                    note_lines.append("PubMed sources:")
                    for r in pubmed_cites:
                        note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                    note_lines.append("")

            if source_controls.get("semantic", False):
                try:
                    sem_results = self.semantic_tool.search(q, max_results=5)
                except Exception:
                    sem_results = []
                    note_lines.append(
                        "Semantic Scholar search failed for this question; continuing without Semantic Scholar results for this item."
                    )
                    note_lines.append("")
                else:
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
                        titles = " ".join((r.get("title", "") or "") for r in sem_results if isinstance(r, dict))
                        tool_usage.approx_tokens += self._estimate_tokens(titles)

                    note_lines.append("Semantic Scholar sources:")
                    for r in sem_cites:
                        note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                    note_lines.append("")

        return "\n".join(note_lines), citations, stats

    def _strengthen_citations(
        self,
        goal: str,
        role: str,
        source_controls: Dict[str, bool],
        domain: Optional[str],
        tool_usage: Optional[ToolUsage] = None,
        deadline_ts: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Specialized repair step for 'under_cited' issues."""
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
            "interrupted": False,
            "stop_reason": None,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Strengthening citations for goal:")
        note_lines.append(goal)
        note_lines.append("")

        if self._deadline_hit(deadline_ts):
            stats["interrupted"] = True
            stats["stop_reason"] = "time_limit"
            note_lines.append("Time limit reached; skipping citation strengthening this cycle.")
            return "\n".join(note_lines), citations, stats

        query = f"{goal} primary sources randomized trial benchmark formal paper"
        note_lines.append(f"Search query for stronger evidence: {query}")
        note_lines.append("")

        if source_controls.get("web", True) and self._should_use_web("strengthen", maintenance_mode=False):
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=False,
                domain=domain,
                purpose="strengthen",
            )
            search_query = _clamp_query(query)
            try:
                web_results = self._web_search(search_query, web_params)
            except Exception:
                web_results = []
                note_lines.append("Web search failed while strengthening citations; continuing without web results.")
                note_lines.append("")
            else:
                stats["web_calls"] += 1
                if tool_usage is not None:
                    tool_usage.web_calls += 1

                web_summary = self.web_tool.summarize_results(web_results)
                web_cites_raw = self.web_tool.to_citations(web_results)
                web_cites = self._tag_citations(
                    web_cites_raw,
                    goal=goal,
                    query=search_query,
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

        if self._deadline_hit(deadline_ts):
            stats["interrupted"] = True
            stats["stop_reason"] = "time_limit"
            note_lines.append("Time limit hit; skipping PubMed/Semantic Scholar strengthening.")
            return "\n".join(note_lines), citations, stats

        if source_controls.get("pubmed", False):
            try:
                pubmed_results = self.pubmed_tool.search(query, max_results=10)
            except Exception:
                pubmed_results = []
                note_lines.append(
                    "PubMed search failed while strengthening citations; continuing without PubMed results."
                )
                note_lines.append("")
            else:
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
                    titles = " ".join((r.get("title", "") or "") for r in pubmed_results if isinstance(r, dict))
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("PubMed sources (stronger evidence):")
                for r in pubmed_cites:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

        if source_controls.get("semantic", False):
            try:
                sem_results = self.semantic_tool.search(query, max_results=10)
            except Exception:
                sem_results = []
                note_lines.append(
                    "Semantic Scholar search failed while strengthening citations; continuing without Semantic Scholar results."
                )
                note_lines.append("")
            else:
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
                    titles = " ".join((r.get("title", "") or "") for r in sem_results if isinstance(r, dict))
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("Semantic Scholar sources (stronger evidence):")
                for r in sem_cites:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

        return "\n".join(note_lines), citations, stats

    def _domain_gap_research(
        self,
        goal: str,
        role: str,
        issue: str,
        description: str,
        source_controls: Dict[str, bool],
        domain: Optional[str],
        tool_usage: Optional[ToolUsage] = None,
        deadline_ts: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Generic helper to fill domain specific gaps."""
        dom = (domain or "general").lower()
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
            "interrupted": False,
            "stop_reason": None,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Domain specific gap repair for goal:")
        note_lines.append(goal)
        note_lines.append("")
        note_lines.append(f"Issue: {issue}")
        note_lines.append(description)
        note_lines.append("")

        if self._deadline_hit(deadline_ts):
            stats["interrupted"] = True
            stats["stop_reason"] = "time_limit"
            note_lines.append("Time limit reached; skipping gap repair searches this cycle.")
            return "\n".join(note_lines), citations, stats

        if dom == "longevity":
            if issue == "missing_biomarkers":
                query = f"biomarkers panel clinical endpoints for {goal} healthspan longevity all-cause mortality"
            elif issue == "missing_mechanisms":
                query = f"mechanisms pathways hallmarks of aging for interventions related to {goal}"
            else:
                query = f"{goal} biomarkers mechanisms healthspan longevity"
        elif dom == "math":
            if issue == "missing_formalism":
                query = "formal definition theorem stability framework similar to reparodynamics and RYE"
            elif issue == "missing_connections":
                query = "connections between stability theory Lyapunov control Markov processes and repair dynamics"
            else:
                query = f"{goal} mathematical stability formalization"
        else:
            query = f"{goal} {issue} {description}"

        note_lines.append(f"Focused query used for gap repair: {query}")
        note_lines.append("")

        if source_controls.get("web", True) and self._should_use_web("gap_repair", maintenance_mode=False):
            web_params = self._select_web_search_params(
                role=role,
                maintenance_mode=False,
                domain=domain,
                purpose="gap_repair",
            )
            search_query = _clamp_query(query)
            try:
                web_results = self._web_search(search_query, web_params)
            except Exception:
                web_results = []
                note_lines.append("Web search failed during gap repair; continuing without web results.")
                note_lines.append("")
            else:
                stats["web_calls"] += 1
                if tool_usage is not None:
                    tool_usage.web_calls += 1

                web_summary = self.web_tool.summarize_results(web_results)
                web_cites_raw = self.web_tool.to_citations(web_results)
                web_cites = self._tag_citations(
                    web_cites_raw,
                    goal=goal,
                    query=search_query,
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

        if self._deadline_hit(deadline_ts):
            stats["interrupted"] = True
            stats["stop_reason"] = "time_limit"
            note_lines.append("Time limit hit; skipping PubMed/Semantic Scholar gap repair.")
            return "\n".join(note_lines), citations, stats

        if dom == "longevity" and source_controls.get("pubmed", False):
            try:
                pubmed_results = self.pubmed_tool.search(query, max_results=10)
            except Exception:
                pubmed_results = []
                note_lines.append("PubMed search failed during gap repair; continuing without PubMed results.")
                note_lines.append("")
            else:
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
                    titles = " ".join((r.get("title", "") or "") for r in pubmed_results if isinstance(r, dict))
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("PubMed sources (gap repair):")
                for r in pubmed_cites:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

        if source_controls.get("semantic", False):
            try:
                sem_results = self.semantic_tool.search(query, max_results=10)
            except Exception:
                sem_results = []
                note_lines.append(
                    "Semantic Scholar search failed during gap repair; continuing without Semantic Scholar results."
                )
                note_lines.append("")
            else:
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
                    titles = " ".join((r.get("title", "") or "") for r in sem_results if isinstance(r, dict))
                    tool_usage.approx_tokens += self._estimate_tokens(titles)

                note_lines.append("Semantic Scholar sources (gap repair):")
                for r in sem_cites:
                    note_lines.append(f"- {r.get('title', '')} ({r.get('url', '')})")
                note_lines.append("")

        return "\n".join(note_lines), citations, stats

    # ------------------------------------------------------------------
    # Contradiction resolver
    # ------------------------------------------------------------------
    def _extract_contradiction_snippets(self, goal: str, limit: int = 3) -> List[str]:
        notes = self.memory_store.get_notes(goal)
        snippets: List[str] = []
        seen = set()
        for note in notes:
            content = str(note.get("content", "") or "")
            for line in content.splitlines():
                s = line.strip()
                if not s:
                    continue
                if "contradiction" in s.lower():
                    if s not in seen:
                        seen.add(s)
                        snippets.append(s)
                if len(snippets) >= limit:
                    return snippets
        return snippets

    def _resolve_contradictions(
        self,
        goal: str,
        role: str,
        source_controls: Dict[str, bool],
        domain: Optional[str],
        maintenance_mode: bool,
        tool_usage: Optional[ToolUsage] = None,
        deadline_ts: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Attempt to resolve contradictions using additional sources.

        Minimal but real: extract contradiction snippets, run targeted web search,
        and write a short "best current resolution" note with citations.
        """
        citations: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "web_calls": 0,
            "pubmed_calls": 0,
            "semantic_calls": 0,
            "pdf_ingestions": 0,
            "contradictions_resolved": 0,
            "sources_used": 0,
            "browser_actions": 0,
            "code_execs": 0,
            "data_loads": 0,
            "interrupted": False,
            "stop_reason": None,
        }

        note_lines: List[str] = []
        note_lines.append(f"[{role}] Contradiction resolution pass for goal:")
        note_lines.append(goal)
        note_lines.append("")

        if self._deadline_hit(deadline_ts):
            stats["interrupted"] = True
            stats["stop_reason"] = "time_limit"
            note_lines.append("Time limit reached; contradiction resolution deferred.")
            return "\n".join(note_lines), citations, stats

        snippets = self._extract_contradiction_snippets(goal, limit=3)
        if not snippets:
            note_lines.append(
                "Contradiction was flagged by detection, but no explicit contradiction lines were extractable from notes. "
                "Consider marking contradictions with a 'CONTRADICTION:' prefix in notes for better automated resolution."
            )
            return "\n".join(note_lines), citations, stats

        note_lines.append("Contradiction snippets:")
        for s in snippets:
            note_lines.append(f"- {s}")
        note_lines.append("")

        resolved_any = False

        for s in snippets:
            if self._deadline_hit(deadline_ts):
                stats["interrupted"] = True
                stats["stop_reason"] = "time_limit"
                note_lines.append("Time limit hit mid-contradiction resolution; stopping.")
                break

            q = f"{goal} evidence resolve contradiction: {s}"
            note_lines.append("### Evidence search:")
            note_lines.append(q)
            note_lines.append("")

            if source_controls.get("web", True) and self._should_use_web("targeted", maintenance_mode):
                web_params = self._select_web_search_params(
                    role=role,
                    maintenance_mode=maintenance_mode,
                    domain=domain,
                    purpose="targeted",
                )
                search_query = _clamp_query(q)
                try:
                    web_results = self._web_search(search_query, web_params)
                except Exception:
                    web_results = []
                    note_lines.append("Web search failed for contradiction resolution; continuing.")
                    note_lines.append("")
                else:
                    stats["web_calls"] += 1
                    if tool_usage is not None:
                        tool_usage.web_calls += 1

                    web_summary = self.web_tool.summarize_results(web_results)
                    web_cites_raw = self.web_tool.to_citations(web_results)
                    web_cites = self._tag_citations(
                        web_cites_raw,
                        goal=goal,
                        query=search_query,
                        channel="web",
                        phase="contradiction",
                    )
                    citations.extend(web_cites)

                    if tool_usage is not None:
                        tool_usage.approx_tokens += self._estimate_tokens(web_summary)

                    note_lines.append("Web summary (contradiction evidence):")
                    note_lines.append(web_summary)
                    note_lines.append("Web sources:")
                    for c in web_cites:
                        note_lines.append(f"- {c.get('title', '')} ({c.get('url', '')})")
                    note_lines.append("")

                    if web_cites:
                        resolved_any = True

        if resolved_any:
            stats["contradictions_resolved"] = 1

        return "\n".join(note_lines), citations, stats

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
        """Push top hypotheses and patterns from this cycle into a replay buffer."""
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

        def hyp_score(h: Dict[str, Any]) -> float:
            val = h.get("confidence")
            try:
                return float(val) if val is not None else 0.0
            except Exception:
                return 0.0

        sorted_hypotheses = sorted([h for h in hypotheses if isinstance(h, dict)], key=hyp_score, reverse=True)
        top_hypotheses = sorted_hypotheses[:5]

        for idx, h in enumerate(top_hypotheses):
            text = h.get("text") or h.get("description") or h.get("title") or ""
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
                item_ids.append(item_id)
            except Exception:
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
            if not isinstance(c, dict):
                continue
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

    # ------------------------------------------------------------------
    # Multi goal helpers
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
        msil_mode: str = "v1",
        msil_track_mode: str = "single",
        rye_mode: str = "v3",
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper for running one cycle across multiple goals."""
        if not goals:
            composite_goal = ""
        else:
            composite_goal = " | ".join(str(g) for g in goals if g)

        result = self.run_cycle(
            goal=composite_goal,
            cycle_index=cycle_index,
            role=role,
            source_controls=source_controls,
            pdf_bytes=pdf_bytes,
            biomarker_snapshot=biomarker_snapshot,
            domain=domain,
            msil_mode=msil_mode,
            msil_track_mode=msil_track_mode,
            rye_mode=rye_mode,
            stage="idea",
            hallmark=None,
            subgoal=None,
            replay_buffer=None,
            curriculum_state=None,
            run_id=run_id,
            agent_id=None,
            swarm_profile=None,
            deadline_ts=None,
            experiment_mode=experiment_mode,
        )
        result["goals"] = list(goals)
        return result

    def run_multi_goal_training_burst(
        self,
        goals: Sequence[str],
        total_cycles: int,
        role: str = "agent",
        source_controls: Optional[Dict[str, bool]] = None,
        pdf_bytes: Optional[bytes] = None,
        biomarker_snapshot: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        msil_mode: str = "v1",
        msil_track_mode: str = "single",
        rye_mode: str = "v3",
        run_id: Optional[str] = None,
        experiment_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Simple multi goal training burst."""
        goals_list = list(goals) or [""]

        cycles: List[Dict[str, Any]] = []

        for i in range(total_cycles):
            goal = goals_list[i % len(goals_list)]
            res = self.run_cycle(
                goal=goal,
                cycle_index=i,
                role=role,
                source_controls=source_controls,
                pdf_bytes=pdf_bytes,
                biomarker_snapshot=biomarker_snapshot,
                domain=domain,
                msil_mode=msil_mode,
                msil_track_mode=msil_track_mode,
                rye_mode=rye_mode,
                stage="idea",
                hallmark=None,
                subgoal=None,
                replay_buffer=None,
                curriculum_state=None,
                run_id=run_id,
                agent_id=None,
                swarm_profile=None,
                deadline_ts=None,
                experiment_mode=experiment_mode,
            )
            cycles.append(res["summary"])

        rye_values = [c.get("RYE") for c in cycles if isinstance(c.get("RYE"), (int, float))]
        avg_rye = (sum(float(v) for v in rye_values) / len(rye_values)) if rye_values else None

        return {
            "goals": goals_list,
            "total_cycles": total_cycles,
            "avg_rye": avg_rye,
            "cycles": cycles,
            "experiment_mode": experiment_mode,
            "msil_mode": msil_mode,
            "msil_track_mode": msil_track_mode,
            "rye_mode": rye_mode,
            "run_id": run_id,
        }
