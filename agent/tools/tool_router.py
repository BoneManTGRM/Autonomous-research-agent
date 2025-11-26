"""
Ultra Tool Router (A1-Autodiscovery)

This router discovers all tools inside agent.tools.*, loads them safely,
routes tasks based on stage (idea / verify), hallmark context, tool reputation,
RYE cost scoring, and parallel async execution.

Every tool call produces a structured "ToolPacket":
    {
        "tool": str,
        "success": bool,
        "latency": float,
        "cost": float,
        "rye_delta": float,
        "hallmark": hallmark,
        "stage": stage,
        "citations": [...],
        "mechanisms": [...],
        "biomarkers": [...],
        "raw_output": any,
        "error": str or None,
    }

This file upgrades the ARA's learning speed by:
    - parallel fanout calls in idea stage
    - strict verification pipeline in verify stage
    - RYE-aware routing
    - dynamic specialization (critic, biomarker, synergy, pdf)
    - soft failure handling
    - tool reputation & adaptive weighting
"""

from __future__ import annotations
import asyncio
import inspect
import importlib
import pkgutil
import time
from typing import Any, Dict, List, Optional, Callable


# -----------------------------------------------------------------------------
# Helper: lightweight reputation tracker
# -----------------------------------------------------------------------------
class ToolReputation:
    def __init__(self):
        self.stats = {}  # tool_name -> dict

    def update(self, tool_name: str, success: bool, latency: float, rye_delta: float):
        rec = self.stats.setdefault(tool_name, {
            "successes": 0,
            "failures": 0,
            "total_calls": 0,
            "avg_latency": 0.0,
            "avg_rye": 0.0,
        })

        rec["total_calls"] += 1
        if success:
            rec["successes"] += 1
        else:
            rec["failures"] += 1

        # moving averages
        rec["avg_latency"] = (rec["avg_latency"] * 0.9) + (latency * 0.1)
        rec["avg_rye"] = (rec["avg_rye"] * 0.9) + (rye_delta * 0.1)

    def best_tools(self) -> List[str]:
        """Return tools sorted by highest avg_rye."""
        sortable = []
        for name, stats in self.stats.items():
            sortable.append((stats.get("avg_rye", 0.0), name))
        sortable.sort(reverse=True)
        return [name for _, name in sortable]


# -----------------------------------------------------------------------------
# Autodiscovery Loader
# -----------------------------------------------------------------------------
def autodiscover_tools() -> Dict[str, Any]:
    """Import any class inside agent.tools.* ending in 'Tool'."""
    import agent.tools as tools_pkg

    discovered = {}

    for loader, module_name, is_pkg in pkgutil.walk_packages(
        tools_pkg.__path__, prefix=tools_pkg.__name__ + "."
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr in dir(module):
            if attr.lower().endswith("tool"):
                cls = getattr(module, attr)
                if inspect.isclass(cls):
                    try:
                        instance = cls()
                        tool_id = attr.replace("Tool", "").lower()
                        discovered[tool_id] = instance
                    except Exception:
                        continue

    return discovered


# -----------------------------------------------------------------------------
# Main Router
# -----------------------------------------------------------------------------
class ToolRouter:
    def __init__(self, specialist_pool: Optional[Dict[str, Any]] = None):
        self.tools = autodiscover_tools()
        self.reputation = ToolReputation()
        self.specialist_pool = specialist_pool or {}

    # -------------------------------------------------------------------------
    # Core Execution Wrappers
    # -------------------------------------------------------------------------
    async def _run_single(
        self,
        tool_name: str,
        tool: Any,
        payload: Dict[str, Any],
        hallmark: Optional[str],
        stage: str,
    ) -> Dict[str, Any]:
        """Run a single tool call with timing, errors, RYE scoring."""
        start = time.perf_counter()

        try:
            if asyncio.iscoroutinefunction(tool.run):
                out = await tool.run(payload)
            else:
                out = tool.run(payload)

            latency = time.perf_counter() - start
            success = True
            rye_delta = float(out.get("rye_delta", 0.0)) if isinstance(out, dict) else 0.0

            self.reputation.update(tool_name, success, latency, rye_delta)

            return {
                "tool": tool_name,
                "success": True,
                "latency": latency,
                "cost": float(out.get("energy_cost", latency)),
                "rye_delta": rye_delta,
                "hallmark": hallmark,
                "stage": stage,
                "citations": out.get("citations", []) if isinstance(out, dict) else [],
                "mechanisms": out.get("mechanisms", []) if isinstance(out, dict) else [],
                "biomarkers": out.get("biomarkers", []) if isinstance(out, dict) else [],
                "raw_output": out,
                "error": None,
            }

        except Exception as e:
            latency = time.perf_counter() - start
            self.reputation.update(tool_name, False, latency, -0.1)

            return {
                "tool": tool_name,
                "success": False,
                "latency": latency,
                "cost": latency * 2.0,
                "rye_delta": -0.1,
                "hallmark": hallmark,
                "stage": stage,
                "citations": [],
                "mechanisms": [],
                "biomarkers": [],
                "raw_output": None,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Public: Idea stage → Parallel fanout
    # -------------------------------------------------------------------------
    async def route_idea_stage(
        self,
        payload: Dict[str, Any],
        hallmark: Optional[str] = None,
        parallel_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Idea stage:
            - parallel fanout
            - wider tool usage
            - exploratory reasoning
        """
        sem = asyncio.Semaphore(parallel_limit)
        tasks = []

        async def sem_task(tool_name: str, tool_obj: Any):
            async with sem:
                return await self._run_single(
                    tool_name, tool_obj, payload, hallmark, stage="idea"
                )

        for name, tool in self.tools.items():
            tasks.append(asyncio.create_task(sem_task(name, tool)))

        return await asyncio.gather(*tasks)

    # -------------------------------------------------------------------------
    # Public: Verify stage → Strict single-tool or specialist
    # -------------------------------------------------------------------------
    async def route_verify_stage(
        self,
        payload: Dict[str, Any],
        hallmark: Optional[str] = None,
        specialist: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify stage:
            - strict critic mode
            - specialist routing if specified
            - high reliability, low-noise
        """

        # Override: use specialist if available
        if specialist and specialist in self.specialist_pool:
            specialist_agent = self.specialist_pool[specialist]
            try:
                start = time.perf_counter()
                out = await specialist_agent.run(payload)
                latency = time.perf_counter() - start
                return {
                    "tool": f"specialist_{specialist}",
                    "success": True,
                    "latency": latency,
                    "cost": latency,
                    "rye_delta": float(out.get("rye_delta", 0.0)),
                    "hallmark": hallmark,
                    "stage": "verify",
                    "citations": out.get("citations", []) if isinstance(out, dict) else [],
                    "mechanisms": out.get("mechanisms", []) if isinstance(out, dict) else [],
                    "biomarkers": out.get("biomarkers", []) if isinstance(out, dict) else [],
                    "raw_output": out,
                    "error": None,
                }
            except Exception as e:
                return {
                    "tool": f"specialist_{specialist}",
                    "success": False,
                    "latency": 0.0,
                    "cost": 2.0,
                    "rye_delta": -0.2,
                    "hallmark": hallmark,
                    "stage": "verify",
                    "citations": [],
                    "mechanisms": [],
                    "biomarkers": [],
                    "raw_output": None,
                    "error": str(e),
                }

        # If no specialist: choose best known tool
        ranked_tools = self.reputation.best_tools()
        if ranked_tools:
            # use best tool
            best_name = ranked_tools[0]
            best_tool = self.tools.get(best_name)
            if best_tool:
                return await self._run_single(
                    best_name, best_tool, payload, hallmark, stage="verify"
                )

        # fallback: safe browser or web search
        fallback = "browser" if "browser" in self.tools else next(iter(self.tools))
        return await self._run_single(
            fallback, self.tools[fallback], payload, hallmark, stage="verify"
        )

    # -------------------------------------------------------------------------
    # Combined entry: CoreAgent calls this
    # -------------------------------------------------------------------------
    async def route(
        self,
        payload: Dict[str, Any],
        stage: str,
        hallmark: Optional[str] = None,
        specialist: Optional[str] = None,
    ) -> Any:
        """
        The unified public interface.

        stage = "idea"  → parallel fanout
        stage = "verify" → strict specialist/critic
        """
        if stage == "idea":
            return await self.route_idea_stage(payload, hallmark)

        if stage == "verify":
            return await this.route_verify_stage(payload, hallmark, specialist)

        # default fallback
        return await this.route_verify_stage(payload, hallmark)
