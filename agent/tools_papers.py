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
import re


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
        # Normalize and shrink the query to avoid sending full prompts to the API.
        def _shrink_query(q: str, max_chars: int = 240) -> str:
            q = (q or "").strip()
            if not q:
                return ""
            # Drop code blocks which sometimes appear in prompts
            q = re.sub(r"```.*?```", " ", q, flags=re.S)
            # If the prompt contains explicit sections like TOPIC:, Query:, Goal: extract trailing part
            for key in ("TOPIC:", "Topic:", "QUERY:", "Query:", "GOAL:", "Goal:"):
                m = re.search(re.escape(key) + r"\s*(.+)", q)
                if m:
                    q = m.group(1).strip()
                    break
            # Collapse whitespace
            q = re.sub(r"\s+", " ", q).strip()
            # Truncate to max_chars and cut on word boundary
            if len(q) > max_chars:
                q = q[:max_chars]
                if " " in q:
                    q = q.rsplit(" ", 1)[0].strip() or q.strip()
            return q

        clean_query = _shrink_query(query)
        if not clean_query:
            return []

        # ------------------------------------------------------------------
        # Disambiguate RYE metric versus rye grain.  If the search query
        # contains the token "rye" but not the phrase "repair yield", we
        # append negative keywords to discourage the retrieval of agricultural
        # or plant genetics literature.  These negative modifiers are
        # recognized by Semantic Scholar and reduce the chance of irrelevant
        # cereal-related results.
        try:
            q_low = clean_query.lower()
            if "rye" in q_low and "repair yield" not in q_low:
                clean_query = (
                    clean_query
                    + " -seed -seeds -grain -grains -cereal -cereals -secale -cultivar -agronomy"
                )
        except Exception:
            pass

        cache_key = (clean_query, limit)
        if cache_key in self._sem_cache:
            return self._tag_sem_results(self._sem_cache[cache_key], agent_role, swarm_id)

        url = self.SEM_URL
        params = {
            "query": clean_query,
            "limit": limit,
            "fields": "title,url,year,venue,abstract"
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Normalize rate limit errors. When the API quota is exceeded it
            # often raises HTTPError with messages like '429 Too Many Requests (retry_after=..)' which
            # we don't want to store verbatim. Replace such messages with a generic indicator.
            err_msg = str(e)
            if err_msg:
                msg_low = err_msg.lower()
                if "too many requests" in msg_low or "retry_after" in msg_low or "429" in msg_low:
                    err_msg = "Semantic Scholar rate limit exceeded"
            stub = [{
                "title": f"[STUB] Semantic Scholar error for '{query}'",
                "url": "",
                "snippet": err_msg,
                "year": "",
                "venue": "",
            }]
            self._sem_cache[cache_key] = stub
            return self._tag_sem_results(stub, agent_role, swarm_id)

        results: List[Dict[str, str]] = []
        for p in data.get("data", []):
            # Filter out entries that lack key peer-reviewed indicators (year and venue)
            year = p.get("year")
            venue = p.get("venue")
            # Skip entries without a year or venue (likely low quality) and known preprint venues
            venue_lower = str(venue or "").lower()
            if not year or not venue:
                continue
            if any(preprint in venue_lower for preprint in ("arxiv", "biorxiv", "medrxiv", "preprint")):
                continue

            snippet_parts = []
            snippet_parts.append(f"Year: {year}")
            if venue:
                snippet_parts.append(f"Venue: {venue}")
            snippet = " | ".join(snippet_parts)

            results.append(
                {
                    "title": p.get("title", "No title"),
                    "url": p.get("url", ""),
                    "snippet": snippet,
                    "year": str(year),
                    "venue": str(venue),
                    # Provide a source tag for downstream normalization.  We
                    # deliberately omit source on fallback entries to avoid
                    # counting stub results as credible.
                    "source": "semantic_scholar",
                }
            )

        # If filtering yields no results, fall back to include the unfiltered data to avoid empty results
        if not results:
            for p in data.get("data", []):
                title = p.get("title", "No title")
                url = p.get("url", "")
                year = p.get("year", "")
                venue = p.get("venue", "")
                snippet_parts = []
                if year:
                    snippet_parts.append(f"Year: {year}")
                if venue:
                    snippet_parts.append(f"Venue: {venue}")
                snippet = " | ".join(snippet_parts)
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "year": str(year),
                    "venue": str(venue),
                })
            if not results:
                results = [{
                    "title": "No Semantic Scholar results found",
                    "url": "",
                    "snippet": "",
                    "year": "",
                    "venue": "",
                }]

        # Apply longevity domain gating.  We filter results based on
        # keyword relevance and remove unrelated law or syndrome topics.  If
        # no results meet the threshold, we fall back to the original list.
        DOMAIN_KEYWORDS = [
            "longevity",
            "aging",
            "metabolism",
            "senescence",
            "epigenetics",
            "inflammation",
        ]
        def _compute_relevance_score(title: str, snippet: str) -> float:
            text = f"{title} {snippet}".lower()
            matches = sum(1 for kw in DOMAIN_KEYWORDS if kw in text)
            return matches / float(len(DOMAIN_KEYWORDS))

        gated: List[Dict[str, str]] = []
        for r in results:
            try:
                title = str(r.get("title") or "")
                snippet = str(r.get("snippet") or "")
                score = _compute_relevance_score(title, snippet)
            except Exception:
                score = 0.0
            text_l = f"{title} {snippet}".lower()
            unrelated_markers = ["law", "legal", "roman", "syndrome"]
            contains_unrelated = any(m in text_l for m in unrelated_markers)
            if contains_unrelated and score == 0.0:
                continue
            if score >= 0.65:
                gated.append(r)
        # Cache the results for this query (unfiltered results stored)
        self._sem_cache[cache_key] = results
        # Tag and return the gated results (or original if gating empties)
        final_results = gated if gated else results
        return self._tag_sem_results(final_results, agent_role, swarm_id)

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
            # Preserve or attach source tag for credible results.  Skip
            # adding a source for stub or no-result entries so citation
            # normalization can filter them appropriately.
            title_val = rr.get("title") or ""
            try:
                title_low = title_val.strip().lower()
            except Exception:
                title_low = ""
            if "source" not in rr:
                # Only tag if it's not a stub and looks like a real title
                if title_low and not title_low.startswith("[stub]") and "no semantic scholar" not in title_low:
                    rr["source"] = "semantic_scholar"
            final.append(rr)

        return final
