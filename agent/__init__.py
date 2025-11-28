"""
Unified tools package for the Autonomous Research Agent.

This package exposes a single, centralized TOOL_REGISTRY that CoreAgent,
the background worker, and the Streamlit UI can all use to:

- Discover which tools are available (web browser, sandbox, data connectors, etc)
- Instantiate tools in a consistent way
- Toggle tools on/off via environment variables
- Stay backward compatible with older imports (BrowserTool, CodeSandbox, DataConnectors)

Design goals:
- Import-time MUST NEVER crash (missing deps are handled gracefully)
- Registry entries are descriptors (kind/cls/description/tags), not instances
- Engine worker + UI can reliably detect:
    * web / browser capability
    * sandbox / code execution capability
    * data loading / connectors capability
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Type

from .browser_tool import BrowserTool
from .code_sandbox import CodeSandbox
from .data_connectors import DataConnectors

# Optional EXTREME MODE web search bridge (Tavily + RYE metadata)
try:
    from .web_search import web_search_tool as extreme_web_search_tool  # type: ignore
except Exception:
    extreme_web_search_tool = None  # type: ignore

__all__ = [
    "BrowserTool",
    "CodeSandbox",
    "DataConnectors",
    "TOOL_REGISTRY",
    "get_tool_descriptor",
    "build_tool_instance",
    "list_tools",
    "has_tool",
]

# -------------------------------------------------------------------
# Internal registry structure
# -------------------------------------------------------------------
#
# TOOL_REGISTRY is a mapping:
#
#   name -> {
#       "kind": "browser" | "sandbox" | "data",
#       "cls":  <class>,
#       "description": <str>,
#       "tags": [<str>, ...],
#       "enabled": <bool>,
#       "fn": Optional[Callable],   # optional callable entry (e.g. web_search_tool)
#   }
#
# The values are *descriptors*, not instantiated objects. CoreAgent or
# any caller can use build_tool_instance(name, **kwargs) to create a
# concrete tool object when needed.
#
# Example:
#   from agent.tools import TOOL_REGISTRY, build_tool_instance
#   if "browser" in TOOL_REGISTRY:
#       browser = build_tool_instance("browser")
#       # And for unified web search:
#       web_fn = TOOL_REGISTRY["web_search"].get("fn")
#       if web_fn:
#           res = web_fn(query="reparodynamics RYE TGRM")
#
# The Streamlit UI and engine_worker.detect_tools() mostly care about
# the *names* (keys) existing, not the exact descriptor shape.
# -------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _env_flag(name: str, default: bool = True) -> bool:
    """Small helper for environment-based toggles."""
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _safe_register(
    name: str,
    *,
    kind: str,
    cls: Optional[Type[Any]],
    description: str,
    tags: Optional[List[str]] = None,
    enabled: bool = True,
    fn: Optional[Callable[..., Any]] = None,
) -> None:
    """
    Safely register a tool descriptor.

    - Never raises.
    - Skips registration if cls is None or enabled is False.
    - Optionally attaches a callable under 'fn' (for web search, etc).
    """
    if not enabled:
        return
    if cls is None:
        return

    try:
        TOOL_REGISTRY[name] = {
            "kind": kind,
            "cls": cls,
            "description": description,
            "tags": tags or [],
            "enabled": True,
            "fn": fn,
        }
    except Exception:
        # Registry errors must never break import
        return


# -------------------------------------------------------------------
# Registration of tools
# -------------------------------------------------------------------
#
# Environment flags:
#   DISABLE_BROWSER_TOOLS=1  -> skip BrowserTool registration
#   DISABLE_SANDBOX_TOOLS=1  -> skip CodeSandbox registration
#   DISABLE_DATA_TOOLS=1     -> skip DataConnectors registration
# -------------------------------------------------------------------

# Browser / web tools
browser_enabled = not _env_flag("DISABLE_BROWSER_TOOLS", default=False)

# Single unified callable for all web-style entries if EXTREME MODE is present
web_fn: Optional[Callable[..., Any]] = extreme_web_search_tool

_safe_register(
    "browser",
    kind="browser",
    cls=BrowserTool if browser_enabled else None,
    description="HTTP web browsing and scraping helper (Playwright + requests).",
    tags=["web", "browser"],
    fn=None,  # BrowserTool is a class; web_fn is for search
)

# Aliases so detect_tools() and the UI can see web capability
_safe_register(
    "web",
    kind="browser",
    cls=BrowserTool if browser_enabled else None,
    description="Alias for BrowserTool (web capability).",
    tags=["web", "alias"],
    fn=web_fn,  # unified EXTREME MODE search entry if available
)
_safe_register(
    "web_search",
    kind="browser",
    cls=BrowserTool if browser_enabled else None,
    description="Unified web search (EXTREME MODE Tavily + RYE when available).",
    tags=["web", "search", "alias"],
    fn=web_fn,
)
_safe_register(
    "internet",
    kind="browser",
    cls=BrowserTool if browser_enabled else None,
    description="Alias for BrowserTool (internet search).",
    tags=["web", "internet", "alias"],
    fn=web_fn,
)
# Explicit Tavily-style name so agents or LangChain-style configs
# that look for "tavily_search" can still find an EXTREME-capable tool.
_safe_register(
    "tavily_search",
    kind="browser",
    cls=BrowserTool if browser_enabled else None,
    description="Tavily-powered EXTREME web search via web_search tool.",
    tags=["web", "search", "tavily"],
    fn=web_fn,
)

# Sandbox tools
sandbox_enabled = not _env_flag("DISABLE_SANDBOX_TOOLS", default=False)
_safe_register(
    "sandbox",
    kind="sandbox",
    cls=CodeSandbox if sandbox_enabled else None,
    description="Safe, bounded code execution sandbox (no external network).",
    tags=["sandbox", "code", "execution"],
)
_safe_register(
    "code_sandbox",
    kind="sandbox",
    cls=CodeSandbox if sandbox_enabled else None,
    description="Alias for sandbox code execution tool.",
    tags=["sandbox", "alias"],
)
_safe_register(
    "python_sandbox",
    kind="sandbox",
    cls=CodeSandbox if sandbox_enabled else None,
    description="Alias for sandbox code execution tool (Python-focused).",
    tags=["sandbox", "python", "alias"],
)
_safe_register(
    "exec_sandbox",
    kind="sandbox",
    cls=CodeSandbox if sandbox_enabled else None,
    description="Alias for sandbox execution tool.",
    tags=["sandbox", "exec", "alias"],
)

# Data connectors
data_enabled = not _env_flag("DISABLE_DATA_TOOLS", default=False)

# Primary data tool
_safe_register(
    "data_connectors",
    kind="data",
    cls=DataConnectors if data_enabled else None,
    description=(
        "Unified data connector: CSV/XLSX/TSV (and optionally JSON/Parquet/SQL) "
        "for experiments and RYE analysis."
    ),
    tags=["data", "csv", "xlsx", "tsv"],
)

# Generic "data" alias so other components can just check for "data"
_safe_register(
    "data",
    kind="data",
    cls=DataConnectors if data_enabled else None,
    description="Alias for unified data connectors.",
    tags=["data", "alias"],
)

# More specific aliases (for UI/agent hints)
_safe_register(
    "data_csv",
    kind="data",
    cls=DataConnectors if data_enabled else None,
    description="CSV-focused connector (alias of DataConnectors).",
    tags=["data", "csv", "alias"],
)
_safe_register(
    "data_xlsx",
    kind="data",
    cls=DataConnectors if data_enabled else None,
    description="XLSX-focused connector (alias of DataConnectors).",
    tags=["data", "xlsx", "alias"],
)


# -------------------------------------------------------------------
# Public helpers
# -------------------------------------------------------------------
def get_tool_descriptor(name: str) -> Optional[Dict[str, Any]]:
    """
    Return the descriptor for a given tool name, or None if not registered.

    Descriptor fields:
        kind: str
        cls:  type
        description: str
        tags: list[str]
        enabled: bool
        fn: Optional[Callable]
    """
    return TOOL_REGISTRY.get(name)


def build_tool_instance(name: str, **kwargs: Any) -> Any:
    """
    Instantiate a tool by name using its registered class.

    Example:
        browser = build_tool_instance("browser")
        result = browser.fetch_page("https://example.com")

    Returns:
        Instance of the tool class, or None if not found or construction fails.
    """
    desc = TOOL_REGISTRY.get(name)
    if not desc:
        return None
    cls = desc.get("cls")
    if cls is None:
        return None
    try:
        return cls(**kwargs)
    except Exception:
        # Tools must never crash the whole agent; caller can handle None.
        return None


def list_tools() -> List[str]:
    """Return a sorted list of available tool names."""
    return sorted(TOOL_REGISTRY.keys())


def has_tool(name: str) -> bool:
    """Return True if a tool with the given name is registered and enabled."""
    desc = TOOL_REGISTRY.get(name)
    if not desc:
        return False
    return bool(desc.get("enabled", True))
