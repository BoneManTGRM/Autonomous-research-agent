"""
Targeted ARA runtime patch.

This keeps the restored repository intact and applies only two behavior changes:
1. The legacy 24_hours profile no longer becomes a hard 1440-minute stop.
2. Reports include many more cycles and agent output lines by default.
"""

from __future__ import annotations

import os
from typing import Any


# Report-builder defaults. agent/report_builder.py reads these at import time.
os.environ.setdefault("APPENDIX_MAX_CYCLES", "50")
os.environ.setdefault("MAX_AGENT_OUTPUT_LINES", "500")


def _patch_presets() -> None:
    """Prefer unbounded continuation when a preset would otherwise default to 24_hours."""
    try:
        from agent import presets  # type: ignore
    except Exception:
        return

    try:
        for _, preset in getattr(presets, "PRESETS", {}).items():
            if isinstance(preset, dict) and preset.get("default_runtime_profile") == "24_hours":
                preset["default_runtime_profile"] = "forever"
    except Exception:
        pass

    try:
        profiles = getattr(presets, "RUNTIME_PROFILES", {})
        if isinstance(profiles, dict) and isinstance(profiles.get("24_hours"), dict):
            profiles["24_hours"]["estimated_cycles"] = max(int(profiles["24_hours"].get("estimated_cycles", 600)), 20000)
            profiles["24_hours"]["description"] = "Legacy 24-hour profile with hard wall-clock cap disabled by sitecustomize.py."
    except Exception:
        pass


def _patch_core_agent() -> None:
    """Wrap CoreAgent continuous runners so 24_hours never injects max_minutes=1440."""
    try:
        import agent.core_agent as core_agent  # type: ignore
    except Exception:
        return

    CoreAgent = getattr(core_agent, "CoreAgent", None)
    if CoreAgent is None or getattr(CoreAgent, "_ara_no_24h_cap_patch", False):
        return

    original_single = getattr(CoreAgent, "run_continuous", None)
    original_swarm = getattr(CoreAgent, "run_swarm_continuous", None)

    if callable(original_single):
        def run_continuous_no_24h_cap(self: Any, *args: Any, **kwargs: Any):
            if kwargs.get("max_minutes") is None and kwargs.get("runtime_profile") in (None, "24_hours"):
                kwargs["runtime_profile"] = "forever"
            return original_single(self, *args, **kwargs)

        setattr(CoreAgent, "run_continuous", run_continuous_no_24h_cap)

    if callable(original_swarm):
        def run_swarm_continuous_no_24h_cap(self: Any, *args: Any, **kwargs: Any):
            if kwargs.get("max_minutes") is None and kwargs.get("runtime_profile") in (None, "24_hours"):
                kwargs["runtime_profile"] = "forever"
            return original_swarm(self, *args, **kwargs)

        setattr(CoreAgent, "run_swarm_continuous", run_swarm_continuous_no_24h_cap)

    setattr(CoreAgent, "_ara_no_24h_cap_patch", True)


def _patch_report_builder() -> None:
    """If report_builder is already loaded, raise its output limits immediately."""
    try:
        import agent.report_builder as rb  # type: ignore
    except Exception:
        return

    try:
        rb.MAX_APPENDIX_CYCLES = max(int(getattr(rb, "MAX_APPENDIX_CYCLES", 0)), 50)
    except Exception:
        rb.MAX_APPENDIX_CYCLES = 50

    try:
        rb.MAX_AGENT_OUTPUT_LINES = max(int(getattr(rb, "MAX_AGENT_OUTPUT_LINES", 0)), 500)
    except Exception:
        rb.MAX_AGENT_OUTPUT_LINES = 500


_patch_presets()
_patch_core_agent()
_patch_report_builder()
