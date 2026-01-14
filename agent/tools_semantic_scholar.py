"""tools_semantic_scholar.py

Semantic Scholar Graph API client used by the TGRM loop.

Key goals:
- Prevent accidental "full prompt" queries from being sent to the API.
- Avoid 429s by using conservative rate limiting (dynamic default based on API key).
- Provide short, citation-ready results (title/url/snippet) with caching.

Environment variables (optional):
- SEMANTIC_SCHOLAR_API_KEY
- SEMANTIC_SCHOLAR_REQUEST_INTERVAL
- SEMANTIC_SCHOLAR_MAX_CONCURRENT
- SEMANTIC_SCHOLAR_MAX_RETRIES
- SEMANTIC_SCHOLAR_TIMEOUT
- SEMANTIC_SCHOLAR_CACHE_DIR
- SEMANTIC_SCHOLAR_CACHE_TTL_SECONDS
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Include venue field to obtain journal/conference names for filtering peer-reviewed works.
DEFAULT_FIELDS = "title,abstract,url,citationCount,year,venue"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_query(query: str, max_chars: int = 240) -> str:
    """Normalize and shrink a query.

    This protects the API from accidentally receiving large multi-section prompts.
    """
    q = (query or "").strip()
    if not q:
        return ""

    # Drop code blocks (common in prompts)
    q = re.sub(r"```.*?```", " ", q, flags=re.S)

    # Heuristic extraction if it looks like a structured prompt
    for key in ("TOPIC:", "Topic:", "QUERY:", "Query:", "GOAL:", "Goal:"):
        m = re.search(re.escape(key) + r"\s*(.+)", q)
        if m:
            q = m.group(1).strip()
            break

    # Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()

    # Hard truncate
    if len(q) > max_chars:
        q = q[:max_chars]
        # Prefer cutting on word boundary
        if " " in q:
            q = q.rsplit(" ", 1)[0].strip() or q.strip()

    return q


def _global_rate_limit(lock_path: str, min_interval_s: float) -> None:
    """Best-effort cross-process rate limiting (Linux).

    Uses a file lock + a timestamp to ensure at most one request per `min_interval_s`
    across multiple workers in the same container.
    """
    if min_interval_s <= 0:
        return
    try:
        import fcntl  # type: ignore

        lock_file = Path(lock_path)
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        with lock_file.open("a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read().strip()
            try:
                last = float(raw) if raw else 0.0
            except Exception:
                last = 0.0

            now = time.time()
            wait = (last + min_interval_s) - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()

            f.seek(0)
            f.truncate()
            f.write(str(now))
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        # If file locking isn't available, fall back to per-process limiting only.
        return


class SemanticScholarTool:
    def __init__(
        self,
        api_key: Optional[str] = None,
        request_interval: Optional[float] = None,
        max_concurrent: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ):
        self.api_key = (api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")).strip()
        self.headers = {
            "User-Agent": "tgrm/semantic-scholar",
            "Accept": "application/json",
        }
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

        # Dynamic defaults: without an API key, Semantic Scholar rate limits are much tighter.
        default_interval = 0.6 if self.api_key else 3.2
        self.request_interval = float(
            request_interval
            if request_interval is not None
            else os.environ.get("SEMANTIC_SCHOLAR_REQUEST_INTERVAL", default_interval)
        )

        default_concurrent = 2 if self.api_key else 1
        self.max_concurrent = int(
            max_concurrent
            if max_concurrent is not None
            else os.environ.get("SEMANTIC_SCHOLAR_MAX_CONCURRENT", default_concurrent)
        )

        self.max_retries = int(
            max_retries
            if max_retries is not None
            else os.environ.get("SEMANTIC_SCHOLAR_MAX_RETRIES", 5)
        )
        self.timeout = float(
            timeout if timeout is not None else os.environ.get("SEMANTIC_SCHOLAR_TIMEOUT", 15)
        )

        self.cache_dir = Path(
            cache_dir
            if cache_dir is not None
            else os.environ.get("SEMANTIC_SCHOLAR_CACHE_DIR", "/tmp/tgrm_cache/semantic_scholar")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = int(
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else os.environ.get("SEMANTIC_SCHOLAR_CACHE_TTL_SECONDS", 6 * 3600)
        )

        self._semaphore = threading.Semaphore(max(self.max_concurrent, 1))
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._cooldown_until = 0.0

        # In-memory cache: key -> (timestamp, results)
        self._cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self._global_lock_path = os.environ.get(
            "SEMANTIC_SCHOLAR_RATE_LOCK", "/tmp/tgrm_cache/semantic_scholar_rate.lock"
        )

    def _cache_get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        now = time.time()
        entry = self._cache.get(key)
        if entry:
            ts, val = entry
            if now - ts <= self.cache_ttl_seconds:
                return val
            self._cache.pop(key, None)

        cache_file = self.cache_dir / f"{_sha256(key)}.json"
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            ts = float(payload.get("ts", 0))
            if now - ts > self.cache_ttl_seconds:
                return None
            data = payload.get("data")
            if isinstance(data, list):
                return data  # type: ignore[return-value]
        except Exception:
            return None
        return None

    def _cache_set(self, key: str, value: List[Dict[str, Any]]) -> None:
        now = time.time()
        self._cache[key] = (now, value)
        cache_file = self.cache_dir / f"{_sha256(key)}.json"
        try:
            cache_file.write_text(json.dumps({"ts": now, "data": value}), encoding="utf-8")
        except Exception:
            pass

    def _respect_rate_limit(self) -> None:
        # Cross-process best-effort
        _global_rate_limit(self._global_lock_path, self.request_interval)

        # Per-process pacing + cooldown
        with self._lock:
            now = time.time()
            if now < self._cooldown_until:
                sleep_s = self._cooldown_until - now
            else:
                sleep_s = 0.0

            if sleep_s <= 0 and self.request_interval > 0:
                elapsed = now - self._last_request_time
                if elapsed < self.request_interval:
                    sleep_s = self.request_interval - elapsed

            if sleep_s > 0:
                time.sleep(sleep_s)
            self._last_request_time = time.time()

    def _perform_query(
        self,
        query: str,
        limit: int,
        fields: str,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": fields,
        }
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        resp = self._session.get(
            SEMANTIC_SCHOLAR_API_URL,
            params=params,
            headers=self.headers,
            timeout=self.timeout,
        )

        # Special handling for 429 so callers can back off properly
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "").strip()
            try:
                wait_s = float(retry_after) if retry_after else 0.0
            except Exception:
                wait_s = 0.0
            if wait_s <= 0:
                wait_s = max(10.0, self.request_interval * 4)
            raise requests.HTTPError(
                f"429 Too Many Requests (retry_after={wait_s})",
                response=resp,
            )

        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", []) or []

        # Build raw results with year and venue fields when available
        raw_results: List[Dict[str, Any]] = []
        for p in papers:
            title = (p.get("title") or "").strip()
            if not title:
                continue
            url = (p.get("url") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            snippet = abstract[:800] if abstract else ""
            year_val: Optional[str] = None
            # Semantic Scholar may provide 'year' as int; normalize to str
            if "year" in p and p.get("year"):
                try:
                    year_val = str(p.get("year"))
                except Exception:
                    year_val = None
            venue_val: Optional[str] = None
            # The API may return either 'venue' or 'journal' as the publication venue
            if p.get("venue"):
                venue_val = str(p.get("venue")).strip()
            elif p.get("journal"):
                venue_val = str(p.get("journal")).strip()

            # Attach a source tag to aid downstream filtering.  Without this the
            # citation_utils treats Semantic Scholar entries as generic web
            # content and may drop them unless a DOI is present.
            raw_results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "year": year_val or "",
                    "venue": venue_val or "",
                    "source": "semantic_scholar",
                }
            )

        # Filter out results lacking peer-reviewed metadata or from preprint sources
        filtered: List[Dict[str, Any]] = []
        preprint_terms = ("arxiv", "biorxiv", "bioRxiv", "medrxiv", "ssrn", "preprint")
        for r in raw_results:
            v = (r.get("venue") or "").lower()
            y = r.get("year") or ""
            # Require both year and venue
            if not y or not v:
                continue
            if any(term in v for term in preprint_terms):
                continue
            filtered.append(r)

        return filtered if filtered else raw_results

    def search(
        self,
        query: str,
        max_results: int = 5,
        fields: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        normalized = _normalize_query(query)

        # ------------------------------------------------------------------
        # Disambiguate RYE metric versus rye grain.  If the normalized query
        # contains the token "rye" but not the full metric phrase "repair yield",
        # append negative keywords to discourage the API from returning
        # agricultural or plant genetics papers.  This modification must
        # occur before caching and sending the request so that distinct
        # disambiguated queries are cached separately.
        try:
            q_low = normalized.lower()
            if "rye" in q_low and "repair yield" not in q_low:
                normalized = (
                    normalized
                    + " -seed -seeds -grain -grains -cereal -cereals -secale -cultivar -agronomy"
                )
        except Exception:
            pass
        if not normalized:
            return [
                {
                    "title": "[STUB] Semantic Scholar error for query='(empty)'",
                    "url": "",
                    "snippet": "Empty query",
                }
            ]

        fields_str = fields or DEFAULT_FIELDS
        cache_key = f"q={normalized}|limit={max_results}|fields={fields_str}|year={year_range or ''}"

        cached = self._cache_get(cache_key)
        if cached is not None:
            # Apply domain gating on cached results before returning
            return self._apply_domain_gating(cached)

        with self._semaphore:
            last_err: Optional[str] = None
            for attempt in range(self.max_retries + 1):
                try:
                    self._respect_rate_limit()
                    results = self._perform_query(
                        query=normalized,
                        limit=max_results,
                        fields=fields_str,
                        year_range=year_range,
                    )
                    # Apply longevity domain gating on the results before caching.
                    gated_results = self._apply_domain_gating(results)
                    # Cache the *pre-gated* results so subsequent calls have access
                    # to the full set, but return the gated list to the caller.
                    self._cache_set(cache_key, results)
                    return gated_results
                except Exception as e:
                    last_err = str(e)
                    # Determine backoff time
                    backoff = min(60.0, (2 ** attempt) * 1.0)
                    # If Semantic Scholar returned Retry-After in the message, respect it
                    m = re.search(r"retry_after=(\d+(?:\.\d+)?)", last_err)
                    if m:
                        try:
                            backoff = max(backoff, float(m.group(1)))
                        except Exception:
                            pass
                    # Jitter to avoid thundering herd
                    backoff = backoff * (0.7 + 0.6 * random.random())

                    with self._lock:
                        self._cooldown_until = max(self._cooldown_until, time.time() + backoff)

                    if attempt >= self.max_retries:
                        break
                    time.sleep(backoff)

            return [
                # Normalize rate limit errors to a generic message. This prevents raw
                # HTTP 429 error strings from leaking into the citation snippets. If the
                # last error indicates a quota issue, replace the snippet accordingly.
                {
                    "title": f"[STUB] Semantic Scholar error for query='{normalized}'",
                    "url": "",
                    "snippet": (
                        "Semantic Scholar rate limit exceeded"
                        if (
                            isinstance(last_err, str)
                            and (
                                "too many requests" in last_err.lower()
                                or "retry_after" in last_err.lower()
                                or "429" in last_err.lower()
                            )
                        )
                        else (last_err or "Unknown error")
                    ),
                }
            ]

    # ------------------------------------------------------------------
    # Domain gating helper
    # ------------------------------------------------------------------
    def _apply_domain_gating(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of Semantic Scholar results to enforce a longevity
        domain gate.  Each result is assigned a simple relevance score based
        on keyword matches.  Only results with a score >= 0.65 are kept.
        If the filter yields no results, the original list is returned.

        A small set of unrelated markers (law, legal, roman, syndrome) are
        used to hard reject entries when no domain keywords are present.
        """
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

        gated: List[Dict[str, Any]] = []
        for r in results:
            try:
                title = str(r.get("title") or "")
                snippet = str(r.get("snippet") or "")
                score = _compute_relevance_score(title, snippet)
            except Exception:
                score = 0.0
            # Hard reject unrelated topics if no domain keywords
            text_l = f"{r.get('title','')} {r.get('snippet','')}".lower()
            unrelated_markers = ["law", "legal", "roman", "syndrome"]
            contains_unrelated = any(m in text_l for m in unrelated_markers)
            if contains_unrelated and score == 0.0:
                continue
            if score >= 0.65:
                gated.append(r)
        return gated if gated else results
