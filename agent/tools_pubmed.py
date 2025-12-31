"""tools_pubmed.py

PubMed (NCBI E-utilities) search helper.

Goals:
- Avoid accidentally sending large prompts as queries (normalize + truncate).
- Be polite to NCBI (rate limiting + retries for 429/5xx).
- Return small citation-friendly records: title / url / snippet.

Environment variables (optional):
- NCBI_API_KEY (or ENTREZ_API_KEY)
- NCBI_EMAIL
- PUBMED_REQUEST_INTERVAL
- PUBMED_MAX_RETRIES
- PUBMED_TIMEOUT
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ESEARCH = f"{_EUTILS_BASE}/esearch.fcgi"
_ESUMMARY = f"{_EUTILS_BASE}/esummary.fcgi"


def _normalize_query(query: str, max_chars: int = 280) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    # Remove code blocks and collapse whitespace
    q = re.sub(r"```.*?```", " ", q, flags=re.S)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > max_chars:
        q = q[:max_chars]
        if " " in q:
            q = q.rsplit(" ", 1)[0].strip() or q.strip()
    return q


class PubMedTool:
    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        tool: str = "tgrm",
        request_interval: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = (api_key or os.environ.get("NCBI_API_KEY") or os.environ.get("ENTREZ_API_KEY") or "").strip()
        self.email = (email or os.environ.get("NCBI_EMAIL") or "").strip()
        self.tool = tool

        # NCBI guidance: ~3 req/sec without key, up to ~10 req/sec with key.
        default_interval = 0.12 if self.api_key else 0.4
        self.request_interval = float(
            request_interval
            if request_interval is not None
            else os.environ.get("PUBMED_REQUEST_INTERVAL", default_interval)
        )
        self.max_retries = int(
            max_retries if max_retries is not None else os.environ.get("PUBMED_MAX_RETRIES", 4)
        )
        self.timeout = float(timeout if timeout is not None else os.environ.get("PUBMED_TIMEOUT", 15))

        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._session = requests.Session()

        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _respect_rate_limit(self) -> None:
        if self.request_interval <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.request_interval:
                time.sleep(self.request_interval - elapsed)
            self._last_request_time = time.time()

    def _request_with_retries(self, url: str, params: Dict[str, Any]) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._respect_rate_limit()
                resp = self._session.get(url, params=params, timeout=self.timeout)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After", "").strip()
                    try:
                        wait_s = float(retry_after) if retry_after else 0.0
                    except Exception:
                        wait_s = 0.0
                    if wait_s <= 0:
                        wait_s = min(60.0, (2 ** attempt) * 2.0)
                    wait_s *= 0.7 + 0.6 * random.random()
                    time.sleep(wait_s)
                    continue

                # Retry transient server errors
                if resp.status_code in (500, 502, 503, 504):
                    wait_s = min(60.0, (2 ** attempt) * 1.0)
                    wait_s *= 0.7 + 0.6 * random.random()
                    time.sleep(wait_s)
                    continue

                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                wait_s = min(60.0, (2 ** attempt) * 1.0)
                wait_s *= 0.7 + 0.6 * random.random()
                time.sleep(wait_s)

        if last_exc:
            raise last_exc
        raise RuntimeError("PubMed request failed")

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        q = _normalize_query(query)
        if not q:
            return [
                {
                    "title": "[STUB] PubMed error for query='(empty)'",
                    "url": "",
                    "snippet": "Empty query",
                }
            ]

        common: Dict[str, Any] = {
            "tool": self.tool,
            "retmode": "json",
        }
        if self.api_key:
            common["api_key"] = self.api_key
        if self.email:
            common["email"] = self.email

        # 1) ESearch to get PubMed IDs
        esearch_params: Dict[str, Any] = {
            **common,
            "db": "pubmed",
            "term": q,
            "retmax": max_results,
            "sort": "relevance",
        }
        esearch = self._request_with_retries(_ESEARCH, esearch_params).json()
        ids = (
            esearch.get("esearchresult", {}).get("idlist", [])
            if isinstance(esearch, dict)
            else []
        )
        if not ids:
            return []

        # 2) ESummary to get titles + metadata
        esummary_params: Dict[str, Any] = {
            **common,
            "db": "pubmed",
            "id": ",".join(ids),
        }
        esummary = self._request_with_retries(_ESUMMARY, esummary_params).json()
        result = esummary.get("result", {}) if isinstance(esummary, dict) else {}
        uids = result.get("uids", []) if isinstance(result, dict) else []

        out: List[Dict[str, Any]] = []
        for pmid in uids:
            item = result.get(pmid, {}) if isinstance(result, dict) else {}
            title = (item.get("title") or "").strip()
            if not title:
                continue

            authors = item.get("authors") or []
            author_str = ", ".join([a.get("name", "") for a in authors if a.get("name")][:3])
            pubdate = (item.get("pubdate") or "").strip()
            journal = (item.get("fulljournalname") or item.get("source") or "").strip()

            snippet_parts = [p for p in [author_str, journal, pubdate] if p]
            snippet = " â¢ ".join(snippet_parts)[:800]

            out.append(
                {
                    "title": title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "snippet": snippet,
                }
            )

        return out
