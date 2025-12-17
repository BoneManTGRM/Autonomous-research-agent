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
    """Return a runtime profile, falling back to 24_hours if unknown."""
    return _get_runtime_profile(name)


def list_runtime_profiles() -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of all runtime profiles."""
    return dict(RUNTIME_PROFILES)
