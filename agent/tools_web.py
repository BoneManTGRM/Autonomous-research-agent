"""Real web research tool using Tavily Search API.

This replaces the old stub search with a real internet search.

- If TAVILY_API_KEY is set (via Streamlit secrets), we call Tavily.
- If the key is missing, we fall back to a simple stubbed response
  so the agent still runs instead of crashing.

Tavily docs:
  - pip install tavily-python
  - from tavily import TavilyClient
"""

from typing import List, Dict
import os

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # Streamlit will install this from requirements.txt


class WebResearchTool:
    """Web research tool powered by Tavily Search API."""

    def __init__(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY")
        self.api_key = api_key
        self.client = None

        if api_key and TavilyClient is not None:
            # Real Tavily client (live internet search)
            self.client = TavilyClient(api_key=api_key)

    def search(self, query: str) -> List[Dict[str, str]]:
        """Perform a web search given a query string.

        If Tavily is configured, this hits the real web.
        Otherwise, it returns a small stubbed response.
        """
        # If we don't have a client, fall back to stub so the agent still works
        if self.client is None:
            return [
                {
                    "title": f"[STUB] Overview of Reparodynamics related to {query}",
                    "snippet": (
                        "Stub result: Tavily API key not configured, "
                        "returning placeholder text."
                    ),
                    "url": "https://example.com/reparodynamics-overview",
                }
            ]

        # Real Tavily search
        response = self.client.search(
            query=query,
            max_results=5,           # up to 5 sources
            topic="general",        # general research
            search_depth="advanced" # better quality for research tasks
        )

        results: List[Dict[str, str]] = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url": item.get("url", ""),
                }
            )
        return results

    def summarize_results(self, results: List[Dict[str, str]]) -> str:
        """Generate a simple summary from a list of search results."""
        if not results:
            return "No results found."
        titles = [res.get("title", "") for res in results[:3]]
        return "; ".join(titles)
