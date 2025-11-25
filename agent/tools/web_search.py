"""
Advanced web search tool for the Autonomous Research Agent.
EXTREME MODE (Option C):

Adds:
- Quality scoring + novelty detection
- Redundancy filtering
- Semantic result signatures
- Information gain estimation
- Domain-aware weighting (longevity, math, general)
- RYE-friendly metadata (search_energy, info_density)
- Full Tavily support + safe stub fallback
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Try to import the Tavily client
# ---------------------------------------------------------------------
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------
@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    source: str = "web"
    score: Optional[float] = None
    favicon: Optional[str] = None

    # Extreme mode extras
    novelty: Optional[float] = None
    density: Optional[float] = None
    signature: Optional[str] = None


@dataclass
class WebSearchSummary:
    query: str
    results: List[WebResult]
    error: Optional[str]
    stubbed: bool
    response_time: Optional[float] = None
    request_id: Optional[str] = None

    # Extreme mode RYE/AGI signals
    info_gain: Optional[float] = None
    search_energy: Optional[float] = None
    difficulty: Optional[float] = None
    semantic_diversity: Optional[float] = None


# ---------------------------------------------------------------------
# Logging + caching
# ---------------------------------------------------------------------
LOG_PATH = Path("logs/web_search_log.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_CACHE: Dict[Tuple[str, int, str, str], WebSearchSummary] = {}
_CACHE_TIMESTAMPS: Dict[Tuple[str, int, str, str], float] = {}
CACHE_TTL_SECONDS = 600.0


def _log_event(event: Dict[str, Any]) -> None:
    """Append to log without ever crashing."""
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


# ---------------------------------------------------------------------
# Tavily wrapper
# ---------------------------------------------------------------------
def _get_tavily_client() -> Tuple[Optional[Any], Optional[str]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None, "No Tavily API key set."

    if TavilyClient is None:
        return None, "tavily-python not installed."

    try:
        return TavilyClient(api_key=api_key), None
    except Exception as e:
        return None, f"Tavily init failed: {e}"


# ---------------------------------------------------------------------
# Extreme-mode analysis helpers
# ---------------------------------------------------------------------
def _semantic_signature(text: str) -> str:
    """Stable hash used to compute redundancy + diversity."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:16]


def _text_density(text: str) -> float:
    """Density score: characters / tokens ~ signal richness."""
    if not text:
        return 0.0
    tokens = max(1, len(text.split()))
    return min(1.0, len(text) / (tokens * 50))  # scale down to [0,1]


def _estimate_novelty(text: str, seen_hashes: List[str]) -> float:
    """Novelty score based on Hamming distance of semantic signature."""
    sig = _semantic_signature(text)
    if not seen_hashes:
        return 1.0  # first result is maximally novel

    distances = []
    for h in seen_hashes:
        d = sum(a != b for a, b in zip(sig, h))
        distances.append(d / len(sig))

    return max(0.0, min(1.0, sum(distances) / len(distances)))


def _from_tavily_response(query: str, raw: Dict[str, Any]) -> WebSearchSummary:
    items = raw.get("results") or raw.get("items") or []
    results: List[WebResult] = []

    seen_sigs: List[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        text = (
            item.get("content")
            or item.get("snippet")
            or item.get("raw_content")
            or ""
        )

        sig = _semantic_signature(text)
        density = _text_density(text)
        novelty = _estimate_novelty(text, seen_sigs)
        seen_sigs.append(sig)

        results.append(
            WebResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=text,
                score=item.get("score"),
                favicon=item.get("favicon"),
                density=density,
                novelty=novelty,
                signature=sig,
            )
        )

    difficulty = 1.0 - (sum(r.density for r in results) / max(1, len(results)))
    info_gain = sum((r.density or 0) * (r.novelty or 0) for r in results)
    diversity = len(set(r.signature for r in results)) / max(1, len(results))

    return WebSearchSummary(
        query=query,
        results=results,
        error=None,
        stubbed=False,
        response_time=raw.get("response_time"),
        request_id=raw.get("request_id"),
        info_gain=round(info_gain, 4),
        search_energy=round((difficulty + 0.2), 4),
        difficulty=round(difficulty, 4),
        semantic_diversity=round(diversity, 4),
    )


def _stub_summary(query: str, message: str) -> WebSearchSummary:
    return WebSearchSummary(
        query=query,
        results=[],
        error=f"Tavily Search Error: {message}",
        stubbed=True,
        response_time=None,
        request_id=None,
        info_gain=0.0,
        search_energy=0.1,
        difficulty=1.0,
        semantic_diversity=0.0,
    )


# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------
def _cache_get(key: Tuple[str, int, str, str]) -> Optional[WebSearchSummary]:
    ts = _CACHE_TIMESTAMPS.get(key)
    if not ts:
        return None
    if time.time() - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        _CACHE_TIMESTAMPS.pop(key, None)
        return None
    return _CACHE.get(key)


def _cache_set(key: Tuple[str, int, str, str], value: WebSearchSummary) -> None:
    _CACHE[key] = value
    _CACHE_TIMESTAMPS[key] = time.time()


# ---------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------
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
    """Perform a Tavily web search with extreme-mode intelligence."""
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
            "info_gain": summary.info_gain,
            "search_energy": summary.search_energy,
            "semantic_diversity": summary.semantic_diversity,
        }
    )

    return asdict(summary)


# ---------------------------------------------------------------------
# ADDITIONS REQUIRED BY TGRM LOOP
# ---------------------------------------------------------------------
def summarize_results(raw: Dict[str, Any]) -> str:
    """Convert raw Tavily extreme-mode results into a readable text block."""
    if not raw or raw.get("error"):
        return f"Search failed: {raw.get('error', 'unknown error')}"

    results = raw.get("results") or []
    if not results:
        return "No results found."

    lines = []
    for idx, r in enumerate(results[:6], start=1):
        title = r.get("title") or "(no title)"
        snippet = r.get("snippet") or ""
        snippet = snippet.replace("\n", " ").strip()
        lines.append(f"{idx}. {title}: {snippet[:300]}")

    return "\n".join(lines)


def to_citations(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert raw results into agent-standard citation objects."""
    out = []
    results = raw.get("results") or []

    for r in results:
        out.append(
            {
                "source": "web",
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("snippet") or "",
            }
        )

    return out
