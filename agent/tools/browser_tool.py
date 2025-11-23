import os
import requests
from typing import Dict, Any, List, Optional

class BrowserTool:
    """
    Web search + lightweight browser fetch.
    Supports real Tavily search or fallback stub.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            # Stubbed result
            return [{
                "source": "stub",
                "title": f"Stubbed search result for: {query}",
                "url": "https://example.com/stub",
                "snippet": "No Tavily key provided."
            }]

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": max_results},
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            return [{
                "source": "error",
                "title": "Search failed",
                "url": "",
                "snippet": str(e)
            }]

    def fetch_url(self, url: str) -> Dict[str, Any]:
        try:
            r = requests.get(url, timeout=8)
            return {
                "url": url,
                "status": r.status_code,
                "content": r.text[:5000]  # safety cap
            }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "content": str(e)
            }
