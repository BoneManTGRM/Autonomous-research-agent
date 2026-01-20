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

    This wrapper calls the underlying ``presets.get_runtime_profile`` and then applies a
    safeguard to avoid the hard-coded 24‑hour cap present in some legacy profiles.

    If the retrieved profile includes a ``max_minutes`` field equal to 1440 and its
    ``runtime_profile`` name is ``"24_hours"``, the returned dictionary is copied and
    the ``max_minutes`` entry is set to ``None`` to indicate no time limit. This allows
    existing code to continue working while removing the implicit 24‑hour ceiling.

    Parameters
    ----------
    name: str
        The name of the runtime profile to look up.

    Returns
    -------
    Dict[str, Any]
        The runtime profile configuration with any 24‑hour cap removed.
    """
    profile = _get_runtime_profile(name)
    # Defensive guard: remove the 24‑hour cap if present
    try:
        if profile.get("runtime_profile") == "24_hours" and profile.get("max_minutes") == 1440:
            # copy to avoid mutating the original definition
            profile = dict(profile)
            profile["max_minutes"] = None
    except Exception:
        # If the profile is not a dict or does not support get(), just return it
        pass
    return profile


def list_runtime_profiles() -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of all runtime profiles."""
    return dict(RUNTIME_PROFILES)
