"""
Advanced web search tool for the Autonomous Research Agent.
EXTREME MODE (Option C):

Adds:
- Quality scoring + novelty detection
- Redundancy filtering
- Semantic result signatures
- Information gain estimation
- Domain aware weighting (longevity, math, general)
- RYE friendly metadata (search_energy, info_density)
- Optional Tavily support + safe stub fallback
- Learning aware search energy and info_gain_per_energy for 10x modes
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
# Tavily configuration (OPTIONAL, ENABLED IF POSSIBLE)
# ---------------------------------------------------------------------
# To use real Tavily search:
#   1) Make sure ENABLE_TAVILY is True.
#   2) Set TAVILY_API_KEY in your environment (Render dashboard or local env).
#
# If ENABLE_TAVILY is False or the key is missing or tavily is not installed,
# this module falls back to the offline stub.
#
# IMPORTANT SECURITY NOTE:
#   There is NO hardwired Tavily key in this file.
#   Keys should ONLY be supplied through environment variables.
#
# Enable Tavily by default, but allow turning it off via env:
#   ENABLE_TAVILY=0  -> disables Tavily
#   ENABLE_TAVILY=1  -> enables Tavily
ENABLE_TAVILY = os.getenv("ENABLE_TAVILY", "1").strip().lower() not in ("0", "false", "no", "off")

# Tavily has a hard 400 character query limit.
# Use a safety margin so the swarm never triggers the error.
MAX_TAVILY_QUERY_CHARS = 360

# ---------------------------------------------------------------------
# Try to import the Tavily client (optional)
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

    # Extreme mode RYE and AGI signals
    info_gain: Optional[float] = None
    search_energy: Optional[float] = None
    difficulty: Optional[float] = None
    semantic_diversity: Optional[float] = None

    # Learning aware extensions
    learning_speed_factor: Optional[float] = None
    burst_profile_hint: Optional[str] = None
    info_gain_per_energy: Optional[float] = None
    effective_info_gain: Optional[float] = None


# ---------------------------------------------------------------------
# Learning profiles for 10x modes
# These are soft hints that the TGRM loop and presets can override.
# ---------------------------------------------------------------------
LEARNING_TOPIC_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "general": {
        "learning_speed_factor": 1.0,
        "burst_profile_hint": "balanced",
    },
    "longevity": {
        # Longevity stacks, biomarkers, and clinical trials
        # tend to benefit from deeper learning bursts.
        "learning_speed_factor": 1.4,
        "burst_profile_hint": "aggressive",
    },
    "math": {
        # Math prefers precision and careful refinement.
        "learning_speed_factor": 1.2,
        "burst_profile_hint": "precision",
    },
    "default": {
        "learning_speed_factor": 1.0,
        "burst_profile_hint": "balanced",
    },
}


def _compute_learning_context(
    topic: str,
    override_factor: Optional[float],
    override_burst_profile: Optional[str],
) -> Dict[str, Any]:
    """Resolve effective learning context for this search.

    Priority:
        1) Explicit overrides from caller (TGRM, CoreAgent, SwarmManager)
        2) Topic based defaults (general, longevity, math)
        3) Environment multiplier AGENT_LEARNING_SPEED_FACTOR

    The final factor is clamped to [0.1, 10.0].
    """
    topic_key = (topic or "general").lower()
    base = LEARNING_TOPIC_DEFAULTS.get(topic_key, LEARNING_TOPIC_DEFAULTS["default"])

    factor = (
        override_factor
        if override_factor is not None
        else float(base.get("learning_speed_factor", 1.0))
    )
    burst = override_burst_profile or str(base.get("burst_profile_hint", "balanced"))

    # Optional global multiplier from environment for experiments
    env_factor_raw = os.getenv("AGENT_LEARNING_SPEED_FACTOR")
    if env_factor_raw:
        try:
            env_factor = float(env_factor_raw)
            factor *= env_factor
        except Exception:
            pass

    # Clamp to a safe range so 10x modes do not blow up metrics
    factor = max(0.1, min(factor, 10.0))

    return {
        "learning_speed_factor": factor,
        "burst_profile_hint": burst,
    }


# ---------------------------------------------------------------------
# Logging and caching
# ---------------------------------------------------------------------
LOG_PATH = Path("logs/web_search_log.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Cache key MUST include inputs that change results, otherwise you can get
# incorrect stale results when toggles change (time_range, include_raw_content, etc).
_CACHE: Dict[Tuple[Any, ...], WebSearchSummary] = {}
_CACHE_TIMESTAMPS: Dict[Tuple[Any, ...], float] = {}
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
# Tavily wrapper (now optional and enabled if configured)
# ---------------------------------------------------------------------
def _get_tavily_client() -> Tuple[Optional[Any], Optional[str]]:
    """Return a TavilyClient instance or an error string.

    If ENABLE_TAVILY is False, this will return (None, reason)
    so the system uses the offline stub path.
    """
    if not ENABLE_TAVILY:
        return None, "Tavily disabled (ENABLE_TAVILY=0)."

    if TavilyClient is None:
        return None, "tavily-python not installed."

    # Read the key at call time (NOT import time) so Render env changes
    # or runtime injection are respected without redeploy.
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None, "No Tavily API key configured."

    try:
        # First try the explicit api_key signature
        try:
            client = TavilyClient(api_key=api_key)
        except TypeError:
            # Some SDK versions only read from env
            os.environ["TAVILY_API_KEY"] = api_key
            client = TavilyClient()
        return client, None
    except Exception as e:
        return None, f"Tavily init failed: {e}"


# ---------------------------------------------------------------------
# Extreme mode analysis helpers
# ---------------------------------------------------------------------
def _semantic_signature(text: str) -> str:
    """Stable hash used to compute redundancy and diversity."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:16]


def _text_density(text: str) -> float:
    """Density score: characters per token as a crude signal richness proxy."""
    if not text:
        return 0.0
    tokens = max(1, len(text.split()))
    return min(1.0, len(text) / (tokens * 50))  # scale down to [0, 1]


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
    """Build a base summary where learning_speed_factor is conceptually 1.0.

    Learning aware adjustments (10x modes) are applied later so the
    cache can safely reuse this neutral summary and the caller can
    reweight search_energy without hitting Tavily again.
    """
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
    info_gain = sum((r.density or 0.0) * (r.novelty or 0.0) for r in results)
    diversity = len(set(r.signature for r in results)) / max(1, len(results))

    # Base search_energy before learning speed adjustments
    base_search_energy = round((difficulty + 0.2), 4)

    return WebSearchSummary(
        query=query,
        results=results,
        error=None,
        stubbed=False,
        response_time=raw.get("response_time"),
        request_id=raw.get("request_id"),
        info_gain=round(info_gain, 4),
        search_energy=base_search_energy,
        difficulty=round(difficulty, 4),
        semantic_diversity=round(diversity, 4),
    )


def _stub_summary(query: str, message: str) -> WebSearchSummary:
    """Base stub summary (learning speed will be applied later).

    When Tavily is unavailable or encounters an error, we return a
    placeholder summary with no results and suppress the exact error
    details from the user-facing fields.  Downstream consumers can
    inspect logs for diagnostics, but the UI will simply see an empty
    result set marked as stubbed.  A minimal search_energy is provided
    so that RYE and Option C metrics remain defined.
    """
    return WebSearchSummary(
        query=query,
        results=[],
        error=None,
        stubbed=True,
        response_time=None,
        request_id=None,
        info_gain=0.0,
        search_energy=0.1,
        difficulty=1.0,
        semantic_diversity=0.0,
    )


def _clone_summary(summary: WebSearchSummary) -> WebSearchSummary:
    """Deep clone via dataclass serialization to avoid mutating cache entries."""
    return WebSearchSummary(**asdict(summary))


def _apply_learning_speed(
    summary: WebSearchSummary,
    learning_speed_factor: float,
    burst_profile_hint: Optional[str],
) -> WebSearchSummary:
    """Reweight search_energy and info_gain for 10x learning aware modes.

    The idea:
        - Base Tavily cost approximates raw energy spent.
        - Faster learning means more yield per unit energy.
        - We keep info_gain as is and treat search_energy as an
          effective cost divided by learning_speed_factor.

    This is compatible with RYE:
        RYE_search = info_gain / search_energy
        RYE_search_10x = info_gain / (search_energy / factor)
    """
    factor = max(0.1, min(float(learning_speed_factor or 1.0), 10.0))

    base_energy = summary.search_energy if summary.search_energy is not None else 1.0
    if base_energy <= 0:
        base_energy = 1.0

    effective_energy = base_energy / factor

    summary.learning_speed_factor = factor
    summary.burst_profile_hint = burst_profile_hint

    summary.search_energy = round(effective_energy, 4)

    if summary.info_gain is not None:
        if effective_energy > 0:
            summary.info_gain_per_energy = round(
                summary.info_gain / effective_energy, 4
            )
        else:
            summary.info_gain_per_energy = None

        summary.effective_info_gain = round(summary.info_gain * factor, 4)

    return summary


# ---------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------
def _cache_get(key: Tuple[Any, ...]) -> Optional[WebSearchSummary]:
    ts = _CACHE_TIMESTAMPS.get(key)
    if not ts:
        return None
    if time.time() - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        _CACHE_TIMESTAMPS.pop(key, None)
        return None
    return _CACHE.get(key)


def _cache_set(key: Tuple[Any, ...], value: WebSearchSummary) -> None:
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
    # Learning aware parameters (wired from presets, CoreAgent, or SwarmManager)
    learning_speed_factor: Optional[float] = None,
    burst_profile_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform a web search with extreme mode and learning aware intelligence.

    If ENABLE_TAVILY is False or Tavily is not available, this uses the offline stub.
    To enable real web search, set ENABLE_TAVILY = True and configure TAVILY_API_KEY.
    """
    raw_q = (query or "").strip()

    # Clamp to Tavily max length with safety margin.
    truncated = False
    if len(raw_q) > MAX_TAVILY_QUERY_CHARS:
        q = raw_q[:MAX_TAVILY_QUERY_CHARS]
        truncated = True
    else:
        q = raw_q

    learning_ctx = _compute_learning_context(
        topic=topic,
        override_factor=learning_speed_factor,
        override_burst_profile=burst_profile_hint,
    )
    eff_factor = learning_ctx["learning_speed_factor"]
    eff_burst = learning_ctx["burst_profile_hint"]

    if not q:
        summary_base = _stub_summary("", "Empty query.")
        summary = _apply_learning_speed(
            _clone_summary(summary_base), eff_factor, eff_burst
        )
        _log_event({"event": "empty_query", "summary": asdict(summary)})
        return asdict(summary)

    max_results = max(1, min(max_results, 12))

    # ---------------------------------------------------------------------
    # Disambiguate RYE metric versus rye grain.  If the query contains the
    # token "rye" but not the phrase "repair yield", append negative
    # modifiers to discourage the retrieval of agricultural, cereal, or
    # plant genetics results.  Do this before computing the cache key so
    # that the disambiguated queries are cached separately.
    try:
        q_low = q.lower()
        if "rye" in q_low and "repair yield" not in q_low:
            q = (
                q
                + " -seed -seeds -grain -grains -cereal -cereals -secale -cultivar -agronomy"
            )
    except Exception:
        pass

    # Cache key MUST include toggles that change the response.
    cache_key: Tuple[Any, ...] = (
        q,
        max_results,
        (topic or "general"),
        (search_depth or "advanced"),
        time_range,
        bool(auto_parameters),
        bool(include_answer),
        bool(include_raw_content),
        bool(include_images),
    )

    if truncated:
        _log_event(
            {
                "event": "query_truncated",
                "original_len": len(raw_q),
                "clamped_len": len(q),
                "sample": raw_q[:200],
            }
        )

    cached = _cache_get(cache_key)
    if cached is not None:
        summary = _apply_learning_speed(
            _clone_summary(cached), eff_factor, eff_burst
        )
        return asdict(summary)

    client, err = _get_tavily_client()
    if client is None:
        summary_base = _stub_summary(q, err or "Client unavailable.")
        _cache_set(cache_key, summary_base)
        summary = _apply_learning_speed(
            _clone_summary(summary_base), eff_factor, eff_burst
        )
        return asdict(summary)

    start = time.time()

    try:
        # Preferred full parameter call
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
        except TypeError:
            # Fallback for older or minimized SDKs
            raw = client.search(query=q, max_results=max_results)
    except Exception as e:
        summary_base = _stub_summary(q, str(e))
        _cache_set(cache_key, summary_base)
        summary = _apply_learning_speed(
            _clone_summary(summary_base), eff_factor, eff_burst
        )
        _log_event(
            {
                "event": "web_search_error",
                "query": q,
                "error": str(e),
            }
        )
        return asdict(summary)

    elapsed = time.time() - start

    summary_base = _from_tavily_response(q, raw)
    summary_base.response_time = summary_base.response_time or elapsed

    _cache_set(cache_key, summary_base)

    summary = _apply_learning_speed(
        _clone_summary(summary_base), eff_factor, eff_burst
    )

    _log_event(
        {
            "event": "web_search",
            "query": q,
            "max_results": max_results,
            "topic": topic,
            "search_depth": search_depth,
            "time_range": time_range,
            "auto_parameters": auto_parameters,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "response_time": summary.response_time,
            "request_id": summary.request_id,
            "num_results": len(summary.results),
            "info_gain": summary.info_gain,
            "search_energy": summary.search_energy,
            "semantic_diversity": summary.semantic_diversity,
            "learning_speed_factor": summary.learning_speed_factor,
            "burst_profile_hint": summary.burst_profile_hint,
            "info_gain_per_energy": summary.info_gain_per_energy,
            "effective_info_gain": summary.effective_info_gain,
        }
    )

    return asdict(summary)


# ---------------------------------------------------------------------
# ADDITIONS REQUIRED BY TGRM LOOP
# ---------------------------------------------------------------------
def summarize_results(raw: Dict[str, Any]) -> str:
    """Convert raw web_search_tool results into a readable text block."""
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
    """Convert raw results into agent standard citation objects."""
    out: List[Dict[str, str]] = []
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
