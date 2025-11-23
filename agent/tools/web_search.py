"""
Web search tool for the Autonomous Research Agent.

Provides:
- Real Tavily search when TAVILY_API_KEY is provided.
- Safe stub mode when key or client missing.
- Clean structured results for TGRM cycles.
- Automatic logging to logs/web_search_log.json.
- In-memory caching to reduce repeated queries.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import the Tavily client
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # if import fails, we run in stub mode


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    source: str = "web"
    score: Optional[float] = None
    favicon: Optional[str] = None


@dataclass
class WebSearchSummary:
    query: str
    results: List[WebResult]
    error: Optional[str]
    stubbed: bool
    response_time: Optional[float] = None
    request_id: Optional[str] = None


# -----------------------------------------------------------------------------
# Logging + caching
# -----------------------------------------------------------------------------
LOG_PATH = Path("logs/web_search_log.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_CACHE: Dict[Tuple[str, int, str, str], WebSearchSummary] = {}
_CACHE_TIMESTAMPS: Dict[Tuple[str, int, str, str], float] = {}
CACHE_TTL_SECONDS = 600.0  # 10 minutes


def _log_event(event: Dict[str, Any]) -> None:
    """Append an event to the web search log without ever crashing."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists():
            with LOG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        else:
            data = []
        data.append(event)
        with LOG_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def _get_tavily_client() -> Tuple[Optional[Any], Optional[str]]:
    """Return an initialized TavilyClient or an error message."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None, "No Tavily API key set. Paste it in the Streamlit sidebar."

    if TavilyClient is None:
        return None, "tavily-python is not installed. Add to requirements.txt."

    try:
        client = TavilyClient(api_key=api_key)
    except Exception as e:
        return None, f"Failed to initialize Tavily client: {e}"

    return client, None


def _from_tavily_response(query: str, raw: Dict[str, Any]) -> WebSearchSummary:
    """Convert Tavily response into our summary object."""
    items = raw.get("results") or raw.get("items") or []
    results = []

    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            WebResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(
                    item.get("content")
                    or item.get("snippet")
                    or item.get("raw_content")
                    or ""
                ),
                score=item.get("score"),
                favicon=item.get("favicon"),
            )
        )

    return WebSearchSummary(
        query=query,
        results=results,
        error=None,
        stubbed=False,
        response_time=raw.get("response_time"),
        request_id=raw.get("request_id"),
    )


def _stub_summary(query: str, message: str) -> WebSearchSummary:
    """Fallback when Tavily is unavailable."""
    return WebSearchSummary(
        query=query,
        results=[],
        error=f"Tavily Search Error: {message}",
        stubbed=True,
        response_time=None,
        request_id=None,
    )


def _cache_get(key: Tuple[str, int, str, str]) -> Optional[WebSearchSummary]:
    """Retrieve cached result if fresh."""
    ts = _CACHE_TIMESTAMPS.get(key)
    if ts is None:
        return None
    if time.time() - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        _CACHE_TIMESTAMPS.pop(key, None)
        return None
    return _CACHE.get(key)


def _cache_set(key: Tuple[str, int, str, str], value: WebSearchSummary) -> None:
    """Store value in cache."""
    _CACHE[key] = value
    _CACHE_TIMESTAMPS[key] = time.time()


# -----------------------------------------------------------------------------
# Main public search tool
# -----------------------------------------------------------------------------
def web_search_tool(
    query: str,
    max_results: int = 6,
    topic: str = "general",
    search_depth: str = "advanced",
    time_range: Optional[str] = None,
    auto_parameters: bool = False,
    include_answer: bool = False,
    include_raw_content: bool = False,
    include_images: bool = False,
) -> Dict[str, Any]:
    """Perform a Tavily web search with safety, logging, and caching."""
    q = (query or "").strip()
    if not q:
        summary = _stub_summary("", "Empty query.")
        _log_event({"event": "empty_query", "summary": asdict(summary)})
        return asdict(summary)

    max_results = max(1, min(max_results, 12))

    cache_key = (q, max_results, topic, search_depth)
    cached = _cache_get(cache_key)
    if cached is not None:
        return asdict(cached)

    client, err = _get_tavily_client()
    if client is None:
        summary = _stub_summary(q, err or "Client unavailable.")
        _log_event({"event": "stubbed_search", "query": q, "error": summary.error})
        _cache_set(cache_key, summary)
        return asdict(summary)

    start = time.time()

    try:
        raw = client.search(
            query=q,
            max_results=max_results,
            topic=topic,
            search_depth=search_depth,
            time_range=time_range,
            auto_parameters=auto_parameters,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
    except Exception as e:
        summary = _stub_summary(q, str(e))
        _log_event({"event": "tavily_exception", "query": q, "error": summary.error})
        _cache_set(cache_key, summary)
        return asdict(summary)

    elapsed = time.time() - start

    summary = _from_tavily_response(q, raw)
    summary.response_time = summary.response_time or elapsed

    _cache_set(cache_key, summary)

    _log_event(
        {
            "event": "web_search",
            "query": q,
            "max_results": max_results,
            "topic": topic,
            "search_depth": search_depth,
            "response_time": summary.response_time,
            "request_id": summary.request_id,
            "num_results": len(summary.results),
        }
    )

    return asdict(summary)
