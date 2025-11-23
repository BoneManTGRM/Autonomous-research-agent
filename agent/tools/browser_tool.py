import os
import requests
from typing import Dict, Any, List, Optional, Tuple


class BrowserTool:
    """
    Web search + lightweight browser fetch for the Autonomous Research Agent.

    Features:
        - Tavily based web search with answer mode and metadata
        - Stubbed offline mode when no API key is available
        - Simple URL fetch for quick page inspection
        - Normalized result shape for the agent

    Backwards compatibility:
        - Existing calls using search(query, max_results=5) still work
        - Existing calls using fetch_url(url) still work
    """

    TAVILY_ENDPOINT = "https://api.tavily.com/search"

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
        self.default_max_results = default_max_results
        self.timeout = float(timeout)
        self.verify_ssl = bool(verify_ssl)

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
    # Internal HTTP helper
    # ------------------------------------------------------------------
    def _post_tavily(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper for Tavily POST.

        Never raises. On any error it returns a dict with:
            {"error": "..."}
        """
        if not self.api_key:
            return {"error": "No Tavily API key configured."}

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "X-API-Key": self.api_key,  # works with newer Tavily clients
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

    @staticmethod
    def _normalize_results(raw_results: Any) -> List[Dict[str, Any]]:
        """
        Normalize Tavily style results into the agent standard shape.

        Input can be:
            - list of dicts
            - dict containing "results"
        """
        if raw_results is None:
            return []

        if isinstance(raw_results, dict) and "results" in raw_results:
            raw_results = raw_results["results"]

        if not isinstance(raw_results, list):
            return []

        normalized: List[Dict[str, Any]] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            src = item.get("source") or item.get("provider") or "web"
            title = item.get("title") or item.get("name") or "(untitled)"
            url = item.get("url") or item.get("link") or ""
            snippet = item.get("snippet") or item.get("content") or item.get("text") or ""
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
    # Public API
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_raw_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search.

        Args:
            query:
                Text query to search for.
            max_results:
                Maximum number of results. Defaults to 5; this preserves
                backward compatible behavior with older calls.
            search_depth:
                "basic" for faster searches, "advanced" for richer context.
            include_answer:
                If True, Tavily will try to generate a direct answer which
                is attached to the first result under "answer".
            include_raw_content:
                If True, raw page content text is requested where Tavily supports it.

        Returns:
            A list of normalized result dicts:
                {
                    "source": "tavily" or similar,
                    "title": "...",
                    "url": "https://...",
                    "snippet": "...",
                    "score": float or None,
                }

        When no API key is present, returns a single stub entry.
        """
        max_results = max(1, int(max_results or self.default_max_results))

        if not self.api_key:
            return [
                {
                    "source": "stub",
                    "title": f"Stubbed search result for: {query}",
                    "url": "https://example.com/stub",
                    "snippet": "No Tavily key provided. Configure TAVILY_API_KEY to enable real web search.",
                    "score": None,
                }
            ]

        payload: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        data = self._post_tavily(payload)
        if "error" in data:
            return [
                {
                    "source": "error",
                    "title": "Search failed",
                    "url": "",
                    "snippet": data["error"],
                    "score": None,
                }
            ]

        return self._normalize_results(data.get("results", data))

    def search_with_answer(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
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
        max_results = max(1, int(max_results or self.default_max_results))

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
        }

        data = self._post_tavily(payload)
        if "error" in data:
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
        return {"answer": answer, "results": results}

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
