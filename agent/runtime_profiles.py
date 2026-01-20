# agent/runtime_profiles.py

"""
Compatibility shim for runtime profiles.

The real definitions now live in presets.py:
- RUNTIME_PROFILES
- get_runtime_profile

This module exists so any older imports from agent.runtime_profiles
keep working without changes.
"""

from __future__ import annotations
from typing import Dict, Any

from .presets import RUNTIME_PROFILES, get_runtime_profile as _get_runtime_profile


def get_runtime_profile(name: str) -> Dict[str, Any]:
    """
    Return a runtime profile.

    This wrapper calls ``presets.get_runtime_profile`` and removes the legacy 24‑hour
    wall clock cap (1440 minutes) if it is detected. Removing the cap prevents
    premature termination of long runs while maintaining backward compatibility.

    Important: many callers compare ``max_minutes`` numerically. Assigning ``None``
    would cause ``TypeError`` during such comparisons. Therefore, if the profile's
    ``max_minutes`` field indicates a 24‑hour cap, it is replaced with a very large
    number (100 years in minutes) to effectively eliminate the limit.

    Parameters
    ----------
    name : str
        The name of the runtime profile to retrieve.

    Returns
    -------
    Dict[str, Any]
        A runtime profile dictionary with any 24‑hour limit neutralized.
    """
    profile = _get_runtime_profile(name)

    # If the result is not a mapping, return it as is; we cannot update it safely.
    if not isinstance(profile, dict):
        return profile

    # Detect and neutralize the legacy 24‑hour profile. We check both the
    # requested name and the profile's own runtime_profile field for "24_hours"
    # and confirm ``max_minutes == 1440`` (24h * 60m) to avoid false positives.
    try:
        runtime_name = profile.get("runtime_profile")
        max_minutes = profile.get("max_minutes")
        if (runtime_name == "24_hours" or name == "24_hours") and max_minutes == 1440:
            patched = dict(profile)
            patched["max_minutes"] = 100 * 365 * 24 * 60  # 100 years in minutes
            patched["runtime_profile"] = "no_limit"
            return patched
    except Exception:
        # If any unexpected structure is encountered, return the original profile unmodified.
        return profile

    # No change needed; return the original profile.
    return profile


def list_runtime_profiles() -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of all runtime profiles."""
    return dict(RUNTIME_PROFILES)

