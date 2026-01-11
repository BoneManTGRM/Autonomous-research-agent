"""Paper Tool — 90-Day Safe PDF + Paper Engine
Swarm-aware • Long-run hardened • Cached • Retry-capable

Capabilities:
    - Robust PDF ingestion from URL or local file
    - Retry + backoff for long autonomous runs (24h–90d)
    - In-memory caching to avoid re-parsing same PDFs
    - Summarisation with adjustable limits
    - Semantic Scholar search with structured fallback
    - Swarm-aware metadata tagging
"""

from __future__ import annotations

import io
import os
import time
import random
from typing import Dict, List, Tuple, Optional

import requests
from PyPDF2 import PdfReader


class PaperTool:
    """High-reliability scientific document ingestion tool."""

    # Cache for long runs: (url_or_path) -> extracted_text
    _cache: Dict[str, str] = {}

    # Retry settings for long runs
    max_retries = 3

    # Semantic Scholar endpoint
    SEM_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    # ------------------------------------------------------------------
    # PDF / FILE INGESTION
    # ------------------------------------------------------------------
    def ingest(self, source: str) -> str:
        """Ingest a document from URL or local path safely."""

        source = source.strip()
        if source in self._cache:
            return self._cache[source]

        # Remote PDF
        if source.lower().startswith("http") and source.lower().endswith(".pdf"):
            try:
                text = self._download_pdf_with_retries(source)
                self._cache[source] = text
                return text
            except Exception as e:
                return f"[Error downloading or parsing PDF: {e}]"

        # Local file
        if os.path.exists(source):
            try:
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                self._cache[source] = text
                return text
            except Exception as e:
                return f"[Error reading local file: {e}]"

        return f"[Could not ingest source: {source}]"

    def _download_pdf_with_retries(self, url: str) -> str:
        """Long-run hardened download method with retry + backoff."""

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, timeout=25)
                resp.raise_for_status()

                pdf_bytes = io.BytesIO(resp.content)
                reader = PdfReader(pdf_bytes)
                text_chunks: List[str] = []

                for page in reader.pages:
                    try:
                        text_chunks.append(page.extract_text() or "")
                    except Exception:
                        continue

                return "\n".join(text_chunks)

            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep((2 ** attempt) + random.uniform(0.1, 1.3))

        return "[PDF download failed]"

    # ------------------------------------------------------------------
    # SUMMARISATION
    # ------------------------------------------------------------------
    def summarise(self, text: str, max_chars: int = 1200) -> str:
        """Long-run safe summarization for PDF or document text."""

        if not text:
            return "[Empty document]"

        if len(text) <= max_chars:
            return text.strip()

        return text[:max_chars].strip() + " ..."

    # ------------------------------------------------------------------
    # SEMANTIC SCHOLAR SEARCH
    # ------------------------------------------------------------------
    # Cache identical queries
    _sem_cache: Dict[Tuple[str, int], List[Dict[str, str]]] = {}

    def search_semantic_scholar(
        self,
        query: str,
        limit: int = 5,
        *,
        agent_role: Optional[str] = None,
        swarm_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Swarm-aware Semantic Scholar search."""

        if not query:
            return []

        cache_key = (query, limit)
        if cache_key in self._sem_cache:
            return self._tag_sem_results(self._sem_cache[cache_key], agent_role, swarm_id)

        url = self.SEM_URL
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,url,year,venue,abstract"
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            stub = [{
                "title": f"[STUB] Semantic Scholar error for '{query}'",
                "url": "",
                "snippet": str(e),
                "year": "",
                "venue": "",
            }]
            self._sem_cache[cache_key] = stub
            return self._tag_sem_results(stub, agent_role, swarm_id)

        results: List[Dict[str, str]] = []
        for p in data.get("data", []):
            snippet_parts = []
            if p.get("year"):
                snippet_parts.append(f"Year: {p.get('year')}")
            if p.get("venue"):
                snippet_parts.append(f"Venue: {p.get('venue')}")

            snippet = " | ".join(snippet_parts)

            results.append(
                {
                    "title": p.get("title", "No title"),
                    "url": p.get("url", ""),
                    "snippet": snippet,
                    "year": p.get("year", ""),
                    "venue": p.get("venue", ""),
                }
            )

        if not results:
            results = [{
                "title": "No Semantic Scholar results found",
                "url": "",
                "snippet": "",
                "year": "",
                "venue": "",
            }]

        self._sem_cache[cache_key] = results
        return self._tag_sem_results(results, agent_role, swarm_id)

    def _tag_sem_results(
        self,
        results: List[Dict[str, str]],
        agent_role: Optional[str],
        swarm_id: Optional[str],
    ) -> List[Dict[str, str]]:
        """Attach swarm metadata (not stored in cache)."""

        final: List[Dict[str, str]] = []

        for r in results:
            rr = dict(r)
            if agent_role:
                rr["agent_role"] = agent_role
            if swarm_id:
                rr["swarm_id"] = swarm_id
            final.append(rr)

        return final
