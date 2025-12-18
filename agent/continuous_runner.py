# agent/continuous_runner.py
# -*- coding: utf-8 -*-
"""
Helper for running the CoreAgent in continuous mode from scripts.

This is optional: Streamlit already uses CoreAgent.run_continuous.
You can use this from CLI tools or scheduled jobs.

Notes
-----
- If `memory_file` is a relative path, this helper will prefer a shared runs-root
  (ARA_RUNS_DIR or agent.run_jobs.BASE_DIR) so UI + worker/scripts can share state.
- It creates parent directories for the memory file if missing.
- It is resilient to minor signature changes in CoreAgent.run_continuous (stop_rye naming).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core_agent import CoreAgent
from .memory_store import MemoryStore


def _resolve_repo_root() -> Path:
    # agent/ is one level below repo root in the typical layout
    return Path(__file__).resolve().parents[1]


def _resolve_runs_root() -> Path:
    """Resolve shared runs root, matching the worker/UI if possible."""
    env = os.getenv("ARA_RUNS_DIR")
    if env:
        return Path(env)

    # Prefer run_jobs.BASE_DIR if it exists so everything shares a single disk root
    try:
        from .run_jobs import BASE_DIR as _BASE_DIR  # type: ignore

        if isinstance(_BASE_DIR, Path):
            return _BASE_DIR
        if isinstance(_BASE_DIR, str) and _BASE_DIR.strip():
            return Path(_BASE_DIR.strip())
    except Exception:
        pass

    return _resolve_repo_root() / "runs"


def _resolve_memory_path(memory_file: str) -> Path:
    """Resolve memory file path with sensible fallbacks.

    Order (for relative paths):
    1) Current working directory (useful for local scripts)
    2) Repo root (back-compat for older layouts)
    3) Shared runs root (recommended for worker/UI shared artifacts)
    """
    p = Path(memory_file)

    if p.is_absolute():
        return p

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = _resolve_repo_root() / p
    if repo_candidate.exists():
        return repo_candidate

    return _resolve_runs_root() / p


def run_continuous_session(
    goal: str,
    memory_file: str = "logs/sessions/default_memory.json",
    config: Optional[Dict[str, Any]] = None,
    max_cycles: int = 100,
    stop_rye: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run a continuous session of the research agent.

    Args:
        goal: The research goal / prompt.
        memory_file: Where the MemoryStore persists history. If relative, this helper
            resolves it to a shared runs root when possible.
        config: Optional config dict passed to CoreAgent.
        max_cycles: Maximum number of continuous cycles.
        stop_rye: Optional early-stop threshold for RYE if the agent supports it.

    Returns:
        A list of cycle summaries (dicts) as returned by CoreAgent.run_continuous.
    """
    goal_clean = (goal or "").strip()
    if not goal_clean:
        raise ValueError("goal must be a non-empty string")

    cfg: Dict[str, Any] = dict(config or {})

    mem_path = _resolve_memory_path(memory_file)
    try:
        mem_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Non-fatal: MemoryStore may still be able to create the file later
        pass

    # Ensure config knows where memory is (useful if CoreAgent reads it)
    cfg.setdefault("memory_file", str(mem_path))

    mem = MemoryStore(str(mem_path))
    agent = CoreAgent(memory_store=mem, config=cfg)

    # Be resilient to slight signature differences across versions.
    if stop_rye is None:
        return agent.run_continuous(goal=goal_clean, max_cycles=int(max_cycles))

    try:
        return agent.run_continuous(goal=goal_clean, max_cycles=int(max_cycles), stop_rye=float(stop_rye))
    except TypeError:
        # Alternate kwarg names some builds use
        try:
            return agent.run_continuous(
                goal=goal_clean,
                max_cycles=int(max_cycles),
                stop_rye_threshold=float(stop_rye),
            )
        except TypeError:
            return agent.run_continuous(
                goal=goal_clean,
                max_cycles=int(max_cycles),
                rye_stop_threshold=float(stop_rye),
            )
