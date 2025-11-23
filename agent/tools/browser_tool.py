import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    # Optional Tavily client (preferred when installed)
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # type: ignore[assignment]


class BrowserTool:
    """
    Web search plus lightweight browser fetch for the Autonomous Research Agent.

    Features:
        - Tavily based web search with answer mode and metadata
        - Support for topic, time_range, search depth, auto parameters, images
        - Stubbed offline mode when no API key is available
        - Simple URL fetch for quick page inspection
        - Normalized result shape for the agent
        - Basic logging to logs/web_search_log.json
        - In memory cache to avoid paying twice for identical queries

    Backwards compatibility:
        - Existing calls using search(query, max_results=5) still work
        - Existing calls using fetch_url(url) still work
        - Existing calls using search_with_answer(...) still work
        - Existing code can also call the instance directly: browser("query")
    """

    TAVILY_ENDPOINT = "https://api.tavily.com/search"
    LOG_PATH = Path("logs/web_search_log.json")

    # cache key: (query, max_results, search_depth, topic, time_range)
    _cache: Dict[Tuple[str, int, str, str, Optional[str]], Dict[str, Any]] = {}
    _cache_ts: Dict[Tuple[str, int, str, str, Optional[str]], float] = {}
    CACHE_TTL_SECONDS: float = 600.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_max_results: int = 5,
        timeout: float = 12.0,
        verify_ssl: bool = True,
    ) -> None:
        """
        Args:
            api_key:
                Tavily API key. If omitted, TAVILY_API_KEY from environment is used.
            endpoint:
                Optional override of the Tavily endpoint.
            default_max_results:
                Used when search is called without max_results.
            timeout:
                HTTP timeout in seconds for both search and fetch.
            verify_ssl:
                Whether to verify SSL certificates.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.endpoint = endpoint or self.TAVILY_ENDPOINT
        self.default_max_results = int(default_max_results)
        self.timeout = float(timeout)
        self.verify_ssl = bool(verify_ssl)

        # Ensure log directory exists
        try:
            self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Key and status helpers
    # ------------------------------------------------------------------
    def has_real_search(self) -> bool:
        """Return True if a Tavily key is available."""
        return bool(self.api_key)

    def key_tail(self) -> Optional[str]:
        """Return last 4 characters of the key for display, or None."""
        if not self.api_key:
            return None
        return self.api_key[-4:]

    def status(self) -> Dict[str, Any]:
        """Return a small status summary used by the UI."""
        return {
            "has_key": self.has_real_search(),
            "key_tail": self.key_tail(),
            "endpoint": self.endpoint,
            "mode": "real" if self.has_real_search() else "stub",
        }

    # ------------------------------------------------------------------
    # Logging and caching
    # ------------------------------------------------------------------
    def _log_event(self, event: Dict[str, Any]) -> None:
        """Append an event to the web search log without ever crashing."""
        try:
            if self.LOG_PATH.exists():
                with self.LOG_PATH.open("r", encoding="utf_8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            else:
                data = []
            data.append(event)
            with self.LOG_PATH.open("w", encoding="utf_8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _cache_get(
        self,
        key: Tuple[str, int, str, str, Optional[str]],
    ) -> Optional[Dict[str, Any]]:
        ts = self._cache_ts.get(key)
        if ts is None:
            return None
        if time.time() - ts > self.CACHE_TTL_SECONDS:
            self._cache.pop(key, None)
            self._cache_ts.pop(key, None)
            return None
        return self._cache.get(key)

    def _cache_set(
        self,
        key: Tuple[str, int, str, str, Optional[str]],
        value: Dict[str, Any],
    ) -> None:
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    # ------------------------------------------------------------------
    # Internal HTTP or client helpers
    # ------------------------------------------------------------------
    def _post_tavily_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper for Tavily POST over raw HTTP.

        Never raises. On any error it returns a dict with:
            {"error": "..."}
        """
        if not self.api_key:
            return {"error": "No Tavily API key configured."}

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "X-API-Key": self.api_key,  # compatible with newer Tavily API wrappers
                "Content-Type": "application/json",
            }
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def _tavily_client_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use tavily-python client if available, otherwise fall back to HTTP helper.
        """
        if TavilyClient is None:
            return self._post_tavily_http(payload)

        if not self.api_key:
            return {"error": "No Tavily API key configured."}

        try:
            client = TavilyClient(api_key=self.api_key)  # type: ignore[call-arg]
            # Map payload into client.search parameters
            resp = client.search(
                query=payload.get("query", ""),
                max_results=payload.get("max_results", self.default_max_results),
                topic=payload.get("topic", "general"),
                search_depth=payload.get("search_depth", "basic"),
                time_range=payload.get("time_range"),
                include_answer=payload.get("include_answer", False),
                include_raw_content=payload.get("include_raw_content", False),
                include_images=payload.get("include_images", False),
                auto_parameters=payload.get("auto_parameters", False),
            )
            if isinstance(resp, dict):
                return resp
            return {"results": resp}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _normalize_results(raw_results: Any) -> List[Dict[str, Any]]:
        """
        Normalize Tavily style results into the agent standard shape.

        Input can be:
            - list of dicts
            - dict containing "results" or "items"
        """
        if raw_results is None:
            return []

        if isinstance(raw_results, dict):
            if "results" in raw_results:
                raw_results = raw_results["results"]
            elif "items" in raw_results:
                raw_results = raw_results["items"]

        if not isinstance(raw_results, list):
            return []

        normalized: List[Dict[str, Any]] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            src = item.get("source") or item.get("provider") or "web"
            title = item.get("title") or item.get("name") or "(untitled)"
            url = item.get("url") or item.get("link") or ""
            snippet = (
                item.get("snippet")
                or item.get("content")
                or item.get("raw_content")
                or item.get("text")
                or ""
            )
            score = item.get("score") or item.get("relevance_score")

            normalized.append(
                {
                    "source": str(src),
                    "title": str(title),
                    "url": str(url),
                    "snippet": str(snippet),
                    "score": float(score) if isinstance(score, (int, float)) else None,
                }
            )

        return normalized

    # ------------------------------------------------------------------
    # Public API - search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: Optional[str] = None,
        auto_parameters: bool = False,
        include_images: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search.

        Args:
            query:
                Text query to search for.
            max_results:
                Maximum number of results.
            search_depth:
                "basic" for faster searches, "advanced" for richer context.
            include_answer:
                If True, Tavily will try to generate a direct answer.
            include_raw_content:
                If True, request raw page content where available.
            topic:
                "general", "news", or "finance".
            time_range:
                Optional range "day", "week", "month", "year".
            auto_parameters:
                Let Tavily auto tune some parameters for difficult queries.
            include_images:
                If True, allow Tavily to return image suggestions.

        Returns:
            A list of normalized result dicts:
                {
                    "source": "web" or similar,
                    "title": "...",
                    "url": "https://...",
                    "snippet": "...",
                    "score": float or None,
                }

        When no API key is present, returns a single stub entry.
        """
        query = (query or "").strip()
        max_results = max(1, int(max_results or self.default_max_results))

        if not query:
            return []

        # Stub when no key
        if not self.api_key:
            stub = [
                {
                    "source": "stub",
                    "title": f"Stubbed search result for: {query}",
                    "url": "https://example.com/stub",
                    "snippet": "No Tavily key provided. Configure TAVILY_API_KEY to enable real web search.",
                    "score": None,
                }
            ]
            self._log_event(
                {
                    "event": "stub_search",
                    "query": query,
                    "max_results": max_results,
                }
            )
            return stub

        cache_key = (query, max_results, search_depth, topic, time_range)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached.get("results", [])

        payload: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "topic": topic,
            "time_range": time_range,
            "auto_parameters": auto_parameters,
            "include_images": include_images,
        }

        start = time.time()
        data = self._tavily_client_search(payload)
        elapsed = time.time() - start

        if "error" in data:
            self._log_event(
                {
                    "event": "search_error",
                    "query": query,
                    "error": data["error"],
                    "elapsed_sec": elapsed,
                }
            )
            error_result = [
                {
                    "source": "error",
                    "title": "Search failed",
                    "url": "",
                    "snippet": data["error"],
                    "score": None,
                }
            ]
            self._cache_set(cache_key, {"results": error_result})
            return error_result

        results = self._normalize_results(data.get("results", data))

        self._log_event(
            {
                "event": "search",
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "topic": topic,
                "time_range": time_range,
                "elapsed_sec": elapsed,
                "num_results": len(results),
            }
        )

        self._cache_set(cache_key, {"results": results})
        return results

    def search_with_answer(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        topic: str = "general",
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience helper that returns both:
            - direct answer from Tavily (if available)
            - normalized results

        Returns:
            {
                "answer": str or None,
                "results": [ {source, title, url, snippet, score}, ... ]
            }
        """
        query = (query or "").strip()
        max_results = max(1, int(max_results or self.default_max_results))

        if not query:
            return {"answer": None, "results": []}

        if not self.api_key:
            return {
                "answer": None,
                "results": [
                    {
                        "source": "stub",
                        "title": f"Stubbed search result for: {query}",
                        "url": "https://example.com/stub",
                        "snippet": "No Tavily key provided. Configure TAVILY_API_KEY to enable real web search.",
                        "score": None,
                    }
                ],
            }

        payload: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,
            "include_raw_content": False,
            "topic": topic,
            "time_range": time_range,
        }

        start = time.time()
        data = self._tavily_client_search(payload)
        elapsed = time.time() - start

        if "error" in data:
            self._log_event(
                {
                    "event": "search_with_answer_error",
                    "query": query,
                    "error": data["error"],
                    "elapsed_sec": elapsed,
                }
            )
            return {
                "answer": None,
                "results": [
                    {
                        "source": "error",
                        "title": "Search failed",
                        "url": "",
                        "snippet": data["error"],
                        "score": None,
                    }
                ],
            }

        answer = data.get("answer") if isinstance(data, dict) else None
        results = self._normalize_results(data.get("results", data))

        self._log_event(
            {
                "event": "search_with_answer",
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "topic": topic,
                "time_range": time_range,
                "elapsed_sec": elapsed,
                "num_results": len(results),
            }
        )

        return {"answer": answer, "results": results}

    # ------------------------------------------------------------------
    # Public API - fetch URL
    # ------------------------------------------------------------------
    def fetch_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch a URL and return a lightweight snapshot.

        Returns:
            {
                "url": str,
                "status": int or "error",
                "content": str  (capped to 5000 chars),
                "content_type": str or None,
            }
        """
        try:
            r = requests.get(url, timeout=self.timeout, verify=self.verify_ssl)
            content_type = r.headers.get("Content-Type", "")
            text = r.text or ""
            return {
                "url": url,
                "status": r.status_code,
                "content": text[:5000],  # safety cap
                "content_type": content_type,
            }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "content": str(e),
                "content_type": None,
            }

    # Small aliases so other parts of the agent can call more flexibly
    def __call__(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Allow BrowserTool instance to be called like a function:
            results = browser("reparodynamics", max_results=5)
        """
        return self.search(query=query, max_results=max_results)
