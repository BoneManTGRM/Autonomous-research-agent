"""Paper tools: PDF ingestion, summarisation, and Semantic Scholar search.

This module gives the agent real paper handling ability:
- Download PDFs from URLs
- Extract text from PDFs
- Summarise the text
- Search Semantic Scholar for relevant papers
"""

import io
import os
from typing import Dict, List

import requests
from PyPDF2 import PdfReader


class PaperTool:
    """Tools for working with scientific papers and documents."""

    def ingest(self, source: str) -> str:
        """Ingest a document from a local file or URL.

        If the source looks like a URL and ends in .pdf, we download and parse
        the PDF. Otherwise we try to treat it as a local text file.
        """
        source = source.strip()

        # Remote PDF
        if source.lower().startswith("http") and source.lower().endswith(".pdf"):
            try:
                resp = requests.get(source, timeout=20)
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
            except Exception as e:
                return f"[Error downloading or parsing PDF: {e}]"

        # Local file
        if os.path.exists(source):
            try:
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                return f"[Error reading local file: {e}]"

        # Fallback
        return f"[Could not ingest source: {source}]"

    def summarise(self, text: str, max_chars: int = 800) -> str:
        """Produce a simple summary of the document.

        This is a naive heuristic: we take the first max_chars, trying to
        preserve line breaks. You can later plug in an LLM here.
        """
        if not text:
            return "[Empty document]"

        if len(text) <= max_chars:
            return text.strip()

        summary = text[:max_chars].strip()
        return summary + " ..."

    def search_semantic_scholar(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search Semantic Scholar for papers related to a query.

        Uses the public Semantic Scholar API (no key required).
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,url,year,venue"
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
        except Exception as e:
            return [
                {
                    "title": "Semantic Scholar error",
                    "url": "",
                    "snippet": str(e),
                }
            ]

        results: List[Dict[str, str]] = []
        for paper in data.get("data", []):
            title = paper.get("title", "No title")
            paper_url = paper.get("url", "")
            year = paper.get("year", "")
            venue = paper.get("venue", "")
            snippet_parts = []
            if year:
                snippet_parts.append(f"Year: {year}")
            if venue:
                snippet_parts.append(f"Venue: {venue}")
            snippet = " | ".join(snippet_parts) if snippet_parts else ""
            results.append(
                {
                    "title": title,
                    "url": paper_url,
                    "snippet": snippet,
                }
            )
        if not results:
            results.append(
                {
                    "title": "No Semantic Scholar results",
                    "url": "",
                    "snippet": "",
                }
            )
        return results
