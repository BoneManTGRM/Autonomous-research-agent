"""Real web research tool using Tavily Search API.

This replaces the old stub search with real internet search.

- If TAVILY_API_KEY is set (via Streamlit Secrets or environment),
  the agent will perform REAL Tavily searches.
- If the key is missing or TavilyClient is unavailable,
  the agent will fall back to a safe stub so it never crashes.

"""

import os
from typing import List, Dict

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # Streamlit Cloud installs this from requirements.txt


class WebResearchTool:
    """Web research tool powered by Tavily Search API."""

    def __init__(self) -> None:
        # Streamlit secrets automatically become environment variables
        self.api_key = os.getenv("TAVILY_API_KEY", None)
        self.client = None

        # If API key exists and TavilyClient imported, initialize real client
        if self.api_key and TavilyClient is not None:
            try:
                self.client = TavilyClient(api_key=self.api_key)
            except Exception:
                self.client = None

    # ----------------------------------------------------------
    # REAL INTERNET SEARCH
    # ----------------------------------------------------------
    def search(self, query: str) -> List[Dict[str, str]]:
        """Perform a REAL web search using Tavily, if available.

        If Tavily is not configured, return a stub result so the
        agent can still complete cycles gracefully.
        """

        if not self.client:
            # STUB FALLBACK MODE
            return [
                {
                    "title": f"[STUB] No Tavily key: Query='{query}'",
                    "snippet": (
                        "Tavily API key missing or Tavily library not loaded. "
                        "Using placeholder result instead of real web search."
                    ),
                    "url": "",
                }
            ]

        # REAL Tavily API call
        try:
            response = self.client.search(
                query=query,
                max_results=5,
                topic="general",
                search_depth="advanced",
            )

            results = []
            for item in response.get("results", []):
                results.append(
                    {
                        "title": item.get("title", "No title"),
                        "snippet": item.get("snippet", item.get("content", "")),
                        "url": item.get("url", ""),
                    }
                )

            if results:
                return results

            # If API returns empty, fall back safely
            return [
                {
                    "title": "No results found",
                    "snippet": "Tavily returned no results for this query.",
                    "url": "",
                }
            ]

        except Exception as e:
            # Safe fallback on API errors
            return [
                {
                    "title": "Tavily Search Error",
                    "snippet": str(e),
                    "url": "",
                }
            ]

    # ----------------------------------------------------------
    # SUMMARIZATION
    # ----------------------------------------------------------
    def summarize_results(self, results: List[Dict[str, str]]) -> str:
        """Create a short summary from search results."""
        if not results:
            return "No results found."

        titles = [r.get("title", "") for r in results[:3]]
        return "; ".join(titles)
