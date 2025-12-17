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
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict

# Optional core tool classes.
# These imports are all guarded so that missing dependencies never
# break the entire agent at import time.
try:
    from .browser_tool import BrowserTool
except Exception:
    BrowserTool = None  # type: ignore[assignment]

try:
    from .code_sandbox import CodeSandbox
except Exception:
    CodeSandbox = None  # type: ignore[assignment]

try:
    from .data_connectors import DataConnectors
except Exception:
    DataConnectors = None  # type: ignore[assignment]

# Optional EXTREME MODE web search bridge (Tavily + RYE metadata)
try:
    from .web_search import web_search_tool as extreme_web_search_tool  # type: ignore[import]
except Exception:
    extreme_web_search_tool = None  # type: ignore[assignment]

__all__ = [
    "BrowserTool",
    "CodeSandbox",
    "DataConnectors",
    "TOOL_REGISTRY",
    "get_tool_descriptor",
    "get_tool_callable",
    "build_tool_instance",
    "list_tools",
    "has_tool",
    "register_tool",
]

# -------------------------------------------------------------------
# Internal registry structure
# -------------------------------------------------------------------
#
# TOOL_REGISTRY is a mapping:
#
#   name -> {
#       "kind": "browser" | "sandbox" | "data",
#       "cls":  <class> or None,
#       "description": <str>,
#       "tags": [<str>, ...],
#       "enabled": <bool>,
#       "fn": Optional[Callable],   # optional callable entry (e.g. web_search_tool)
#   }
#
# The values are descriptors, not instantiated objects. CoreAgent or
# any caller can use build_tool_instance(name, **kwargs) to create a
# concrete tool object when needed.
#
# Example:
#   from agent.tools import TOOL_REGISTRY, build_tool_instance
#   if "browser" in TOOL_REGISTRY:
#       browser = build_tool_instance("browser")
#   web_fn = TOOL_REGISTRY.get("web_search", {}).get("fn")
#   if web_fn:
#       res = web_fn(query="reparodynamics RYE TGRM")
#
# The Streamlit UI and engine_worker.detect_tools() mostly care about
# the names existing, not the exact descriptor shape.
# -------------------------------------------------------------------


class ToolDescriptor(TypedDict):
    kind: str
    cls: Optional[Type[Any]]
    description: str
    tags: List[str]
    enabled: bool
    fn: Optional[Callable[..., Any]]


TOOL_REGISTRY: Dict[str, ToolDescriptor] = {}


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
    - Skips registration if both cls and fn are None or enabled is False.
    - Allows pure function tools (cls is None, fn is not None).
    """
    if not enabled:
        return
    if cls is None and fn is None:
        return

    try:
        # Deduplicate tags while preserving order
        tag_list = tags or []
        tag_list = list(dict.fromkeys(tag_list))

        TOOL_REGISTRY[name] = {
            "kind": kind,
            "cls": cls,
            "description": description,
            "tags": tag_list,
            "enabled": True,
            "fn": fn,
        }
    except Exception:
        # Registry errors must never break import
        return


def register_tool(
    name: str,
    *,
    kind: str,
    cls: Optional[Type[Any]] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    enabled: bool = True,
    fn: Optional[Callable[..., Any]] = None,
) -> None:
    """
    Public registration hook for downstream modules/plugins.

    This is intentionally thin and safe; it will not raise.
    """
    _safe_register(
        name,
        kind=kind,
        cls=cls,
        description=description,
        tags=tags,
        enabled=enabled,
        fn=fn,
    )


# -------------------------------------------------------------------
# Registration of tools
# -------------------------------------------------------------------
#
# Environment flags (existing):
#   DISABLE_BROWSER_TOOLS=1  -> skip BrowserTool-style registrations
#   DISABLE_SANDBOX_TOOLS=1  -> skip CodeSandbox registrations
#   DISABLE_DATA_TOOLS=1     -> skip DataConnectors registrations
#
# Additional environment flags (new, optional):
#   DISABLE_WEB_TOOLS=1          -> disable *all* web capability (browser + web_search)
#   DISABLE_WEB_SEARCH_TOOLS=1   -> disable function-based web search bridge (Tavily/EXTREME)
#   DISABLE_TAVILY_SEARCH=1      -> alias for DISABLE_WEB_SEARCH_TOOLS
#   DISABLE_EXTREME_WEB_SEARCH=1 -> alias for DISABLE_WEB_SEARCH_TOOLS
# -------------------------------------------------------------------

# Web/browser toggles
web_tools_disabled = _env_flag("DISABLE_WEB_TOOLS", default=False)

# Keep existing semantics: DISABLE_BROWSER_TOOLS disables BrowserTool class-based browsing,
# but does not automatically disable function-based web search unless DISABLE_WEB_TOOLS
# (or the search-specific flags) are set.
browser_enabled = not (_env_flag("DISABLE_BROWSER_TOOLS", default=False) or web_tools_disabled)

web_search_disabled = (
    web_tools_disabled
    or _env_flag("DISABLE_WEB_SEARCH_TOOLS", default=False)
    or _env_flag("DISABLE_TAVILY_SEARCH", default=False)
    or _env_flag("DISABLE_EXTREME_WEB_SEARCH", default=False)
)

# Single unified callable for all web-style search entries if EXTREME MODE is present
web_fn: Optional[Callable[..., Any]] = None
try:
    if not web_search_disabled and callable(extreme_web_search_tool):
        web_fn = extreme_web_search_tool  # type: ignore[assignment]
except Exception:
    web_fn = None

# Browser and web tools
_safe_register(
    "browser",
    kind="browser",
    cls=BrowserTool if browser_enabled else None,
    description="HTTP web browsing and scraping helper (Playwright + requests).",
    tags=["web", "browser"],
    fn=None,
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
# Only register this name if we actually have a callable web search bridge.
_safe_register(
    "tavily_search",
    kind="browser",
    cls=None,
    description="Tavily-powered EXTREME web search via web_search tool.",
    tags=["web", "search", "tavily"],
    enabled=web_fn is not None,
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
def get_tool_descriptor(name: str) -> Optional[ToolDescriptor]:
    """
    Return the descriptor for a given tool name, or None if not registered.

    Descriptor fields:
        kind: str
        cls:  type or None
        description: str
        tags: list[str]
        enabled: bool
        fn: Optional[Callable]
    """
    return TOOL_REGISTRY.get(name)


def get_tool_callable(name: str) -> Optional[Callable[..., Any]]:
    """
    Convenience helper: return the callable 'fn' for a tool if present/callable,
    otherwise None.
    """
    desc = TOOL_REGISTRY.get(name)
    if not desc:
        return None
    fn = desc.get("fn")
    return fn if callable(fn) else None


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
