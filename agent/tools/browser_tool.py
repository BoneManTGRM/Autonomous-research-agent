import os
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from threading import Lock

# Optional HTTP client
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

try:
    # Optional Tavily client (preferred when installed)
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # type: ignore[assignment]


@dataclass
class BrowserPage:
    """Lightweight page snapshot used by TGRM for deep dives."""
    url: str
    text_snippet: str
    status: Any
    content_type: Optional[str] = None
    error: Optional[str] = None


class BrowserTool:
    """
    Web search plus lightweight browser fetch for the Autonomous Research Agent.

    Updates in this version:
        - Fix env bool parsing so TAVILY_STUB_MODE=0 and DISABLE_WEB_SEARCH=0 do not force stub.
        - HTTP fallback includes api_key in JSON body (in addition to headers).
        - Adds retries and backoff for 429 and transient 5xx.
        - Optional light global rate limiting to reduce stampedes in swarm mode.
        - Error payloads include status and response snippet for easier debugging.
        - describe_capabilities() helper for diagnostics and UIs.
    """

    TAVILY_ENDPOINT = "https://api.tavily.com/search"
    LOG_PATH = Path("logs/web_search_log.json")

    # cache key:
    # (query, max_results, search_depth, topic, time_range, include_answer,
    #  include_raw_content, include_images, auto_parameters)
    _cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    _cache_ts: Dict[Tuple[Any, ...], float] = {}
    CACHE_TTL_SECONDS: float = 600.0

    # Tavily hard limit is roughly 400 chars; keep margin
    MAX_TAVILY_QUERY_CHARS: int = 360

    # PubMed URLs break if term is huge (entire JSON config). Keep it short.
    MAX_PUBMED_TERM_CHARS: int = 300

    # Basic retry policy
    MAX_RETRIES: int = 5

    # Optional global throttling (requests per second)
    DEFAULT_TAVILY_RPS: float = 1.0
    _RATE_LOCK: Lock = Lock()
    _NEXT_OK_TS: float = 0.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_max_results: int = 5,
        timeout: float = 12.0,
        verify_ssl: bool = True,
        cache_ttl_seconds: Optional[float] = None,
        force_stub: Optional[bool] = None,
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
            cache_ttl_seconds:
                Override cache TTL (seconds). Falls back to env or default.
            force_stub:
                If True, always run in stub mode even if a key is present.
        """
        self.api_key = api_key
        self.endpoint = endpoint or self.TAVILY_ENDPOINT
        self.default_max_results = int(default_max_results)
        self.timeout = float(timeout)
        self.verify_ssl = bool(verify_ssl)

        # Cache TTL: instance override -> env -> class default
        env_ttl = os.getenv("BROWSER_CACHE_TTL_SECONDS")
        if cache_ttl_seconds is not None:
            self.cache_ttl_seconds = float(cache_ttl_seconds)
        elif env_ttl is not None:
            try:
                self.cache_ttl_seconds = float(env_ttl)
            except Exception:
                self.cache_ttl_seconds = self.CACHE_TTL_SECONDS
        else:
            self.cache_ttl_seconds = self.CACHE_TTL_SECONDS

        # Rate limit (requests per second), set 0 to disable
        env_rps = os.getenv("TAVILY_RPS", "").strip()
        if env_rps:
            try:
                self.tavily_rps = float(env_rps)
            except Exception:
                self.tavily_rps = self.DEFAULT_TAVILY_RPS
        else:
            self.tavily_rps = self.DEFAULT_TAVILY_RPS

        # Stub / offline mode:
        # - Explicit argument wins
        # - Otherwise check env flags correctly (so "0" is False)
        if force_stub is not None:
            self.force_stub = bool(force_stub)
        else:
            enable_tavily = not self._env_false("ENABLE_TAVILY", default_false=False)
            self.force_stub = (
                self._env_true("TAVILY_STUB_MODE")
                or self._env_true("DISABLE_WEB_SEARCH")
                or (not enable_tavily)
            )

        # Ensure log directory exists
        try:
            self.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Env parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _env_true(name: str) -> bool:
        v = os.getenv(name)
        if v is None:
            return False
        v = v.strip().lower()
        return v in ("1", "true", "yes", "on")

    @staticmethod
    def _env_false(name: str, default_false: bool = True) -> bool:
        """
        Returns True if env var explicitly disables a feature.
        If not present, returns default_false.
        """
        v = os.getenv(name)
        if v is None:
            return default_false
        v = v.strip().lower()
        return v in ("0", "false", "no", "off")

    # ------------------------------------------------------------------
    # Key and status helpers
    # ------------------------------------------------------------------
    def _effective_api_key(self) -> Optional[str]:
        """Return the API key from (explicit arg) or environment."""
        return self.api_key or os.getenv("TAVILY_API_KEY")

    def has_real_search(self) -> bool:
        """Return True if a Tavily key is available and not forced into stub mode."""
        return bool(self._effective_api_key()) and not self.force_stub

    def is_stub_mode(self) -> bool:
        """
        Return True if the tool is currently in stub mode.

        Stub mode is active if:
            - force_stub is True, or
            - no Tavily API key is configured.
        """
        return self.force_stub or not self._effective_api_key()

    def key_tail(self) -> Optional[str]:
        """Return last 4 characters of the key for display, or None."""
        key = self._effective_api_key()
        if not key:
            return None
        return key[-4:]

    def status(self) -> Dict[str, Any]:
        """Return a small status summary used by the UI."""
        mode = "real" if self.has_real_search() else "stub"
        stub_reason: Optional[str] = None
        if not self._effective_api_key():
            stub_reason = "no_api_key"
        elif self.force_stub:
            stub_reason = "forced_stub_mode"

        return {
            "has_key": bool(self._effective_api_key()),
            "key_tail": self.key_tail(),
            "endpoint": self.endpoint,
            "mode": mode,
            "stub_reason": stub_reason,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "log_path": str(self.LOG_PATH),
            "requests_available": requests is not None,
            "tavily_client_available": TavilyClient is not None,
            "tavily_rps": self.tavily_rps,
            "enable_tavily_env": not self._env_false("ENABLE_TAVILY", default_false=False),
            "tavily_stub_mode_env": self._env_true("TAVILY_STUB_MODE"),
            "disable_web_search_env": self._env_true("DISABLE_WEB_SEARCH"),
        }

    def describe_capabilities(self) -> Dict[str, Any]:
        """
        Lightweight capabilities summary for diagnostics and higher level tools.
        """
        st = self.status()
        return {
            "stub_mode": st.get("mode") != "real",
            "has_key": st.get("has_key", False),
            "key_tail": st.get("key_tail"),
            "endpoint": st.get("endpoint"),
            "tavily_rps": st.get("tavily_rps"),
            "cache_ttl_seconds": st.get("cache_ttl_seconds"),
            "stub_reason": st.get("stub_reason"),
        }

    # ------------------------------------------------------------------
    # Logging and caching
    # ------------------------------------------------------------------
    def _log_event(self, event: Dict[str, Any]) -> None:
        """Append an event to the web search log without ever crashing."""
        try:
            base = {
                "timestamp": time.time(),
                "mode": "real" if self.has_real_search() else "stub",
            }
            event_full = {**base, **event}

            if self.LOG_PATH.exists():
                with self.LOG_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            else:
                data = []
            data.append(event_full)
            with self.LOG_PATH.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _cache_get(self, key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        ts = self._cache_ts.get(key)
        if ts is None:
            return None
        if time.time() - ts > self.cache_ttl_seconds:
            self._cache.pop(key, None)
            self._cache_ts.pop(key, None)
            return None
        return self._cache.get(key)

    def _cache_set(self, key: Tuple[Any, ...], value: Dict[str, Any]) -> None:
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    # ------------------------------------------------------------------
    # Internal helpers for clamping queries
    # ------------------------------------------------------------------
    def _clamp_query(self, query: str) -> Tuple[str, bool]:
        """Clamp query to Tavily-safe length."""
        q = (query or "").strip()
        if len(q) > self.MAX_TAVILY_QUERY_CHARS:
            return q[: self.MAX_TAVILY_QUERY_CHARS], True
        return q, False

    def _clamp_pubmed_term(self, raw: Any) -> Tuple[str, bool]:
        """
        Build a short PubMed-safe term from a possibly huge object.

        This is intended to be used by the PubMed integration so we never send
        an entire JSON job config in the `term` URL parameter (which causes
        414 Request-URI Too Long errors).
        """
        if isinstance(raw, dict):
            pieces: List[str] = []
            for key in ("goal", "question", "domain", "topic", "hypothesis"):
                v = raw.get(key)
                if isinstance(v, str):
                    pieces.append(v)

            roles = raw.get("roles") or raw.get("role")
            if isinstance(roles, str):
                pieces.append(roles)
            elif isinstance(roles, list):
                for r in roles:
                    if isinstance(r, str):
                        pieces.append(r)

            if not pieces:
                term = json.dumps(raw, separators=(",", ":"))
            else:
                term = " ".join(pieces)
        else:
            term = str(raw or "")

        term = term.replace("\n", " ").strip()
        if len(term) > self.MAX_PUBMED_TERM_CHARS:
            return term[: self.MAX_PUBMED_TERM_CHARS], True
        return term, False

    def build_pubmed_term(self, raw: Any) -> str:
        """Public wrapper for PubMed term building."""
        term, _ = self._clamp_pubmed_term(raw)
        return term

    # ------------------------------------------------------------------
    # Rate limiting and backoff
    # ------------------------------------------------------------------
    def _rate_wait(self) -> None:
        """
        Simple global rate limiter to reduce stampedes during swarm runs.
        Set TAVILY_RPS=0 to disable.
        """
        rps = float(self.tavily_rps or 0.0)
        if rps <= 0.0:
            return

        with self._RATE_LOCK:
            now = time.time()
            wait = max(0.0, self._NEXT_OK_TS - now)
            step = 1.0 / max(rps, 0.1)
            self._NEXT_OK_TS = max(self._NEXT_OK_TS, now) + step

        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _backoff_sleep(attempt: int) -> None:
        # Exponential backoff with jitter
        base = (2 ** attempt)
        time.sleep(base + random.random())

    @staticmethod
    def _snip(text: str, limit: int = 500) -> str:
        t = (text or "").strip()
        if len(t) <= limit:
            return t
        return t[:limit] + "..."

    # ------------------------------------------------------------------
    # Internal HTTP or client helpers
    # ------------------------------------------------------------------
    def _post_tavily_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper for Tavily POST over raw HTTP.

        Never raises. On any error it returns a dict with:
            {"error": "...", "status_code": int|None, "response_snippet": str|None}
        """
        api_key = self._effective_api_key()
        if not api_key:
            return {
                "error": "No Tavily API key configured.",
                "status_code": None,
                "response_snippet": None,
            }

        if requests is None:
            return {
                "error": "requests library is not available.",
                "status_code": None,
                "response_snippet": None,
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_wait()

                body = dict(payload)
                # Important: include api_key in body for compatibility
                body["api_key"] = api_key

                resp = requests.post(
                    self.endpoint,
                    json=body,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

                # Retry on rate limit and transient server errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    self._backoff_sleep(attempt)
                    continue

                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    self._backoff_sleep(attempt)
                    continue
                # Final failure
                status_code = None
                response_snippet = None
                try:
                    status_code = getattr(resp, "status_code", None)  # type: ignore[name-defined]
                    response_snippet = self._snip(getattr(resp, "text", "") or "")  # type: ignore[name-defined]
                except Exception:
                    pass

                err_str = str(e)
                pubstub_flag = False
                if response_snippet and "PubStub" in response_snippet:
                    pubstub_flag = True
                if "PubStub" in err_str:
                    pubstub_flag = True

                if pubstub_flag:
                    err_str = (
                        "Tavily PubStub error. Check your API key, plan, and Tavily account "
                        "configuration. The original error text is in response_snippet."
                    )

                result: Dict[str, Any] = {
                    "error": err_str,
                    "status_code": status_code,
                    "response_snippet": response_snippet,
                }
                if pubstub_flag:
                    result["pubstub"] = True
                    result["raw_error"] = str(e)
                return result

        return {
            "error": "Tavily HTTP retries exhausted.",
            "status_code": None,
            "response_snippet": None,
        }

    def _tavily_client_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use tavily-python client if available, otherwise fall back to HTTP helper.

        Never raises. Returns dict, or {"error": "..."} on failure.
        """
        api_key = self._effective_api_key()
        if not api_key:
            return {"error": "No Tavily API key configured."}

        # If no client, always use HTTP path
        if TavilyClient is None:
            return self._post_tavily_http(payload)

        last_err: Optional[str] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_wait()

                client = TavilyClient(api_key=api_key)  # type: ignore[call-arg]
                try:
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
                except TypeError:
                    # Older or minimal SDKs
                    resp = client.search(
                        query=payload.get("query", ""),
                        max_results=payload.get("max_results", self.default_max_results),
                    )

                if isinstance(resp, dict):
                    return resp
                return {"results": resp}
            except Exception as e:
                last_err = str(e)
                # Backoff and retry a few times, then fall back to HTTP
                if attempt < self.MAX_RETRIES - 1:
                    self._backoff_sleep(attempt)
                    continue
                break

        # Fallback to HTTP after client failures
        http_resp = self._post_tavily_http(payload)
        if "error" in http_resp and last_err:
            http_resp["client_error"] = last_err
        return http_resp

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

        Returns normalized results.

        When in stub mode, returns a single stub entry.
        """
        q, truncated = self._clamp_query(query)
        max_results = max(1, int(max_results or self.default_max_results))

        if not q:
            return []

        # Stub when no key or forced stub
        if self.is_stub_mode():
            stub = [
                {
                    "source": "stub",
                    "title": f"Stubbed search result for: {q}",
                    "url": "https://example.com/stub",
                    "snippet": (
                        "[stub] Stubbed search. Configure TAVILY_API_KEY and "
                        "set TAVILY_STUB_MODE=0, DISABLE_WEB_SEARCH=0, ENABLE_TAVILY=1 "
                        "to enable real web search."
                    ),
                    "score": None,
                }
            ]
            self._log_event(
                {
                    "event": "stub_search",
                    "query": q,
                    "max_results": max_results,
                    "truncated": truncated,
                }
            )
            return stub

        cache_key: Tuple[Any, ...] = (
            q,
            max_results,
            search_depth,
            topic,
            time_range,
            bool(include_answer),
            bool(include_raw_content),
            bool(include_images),
            bool(auto_parameters),
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached.get("results", [])

        payload: Dict[str, Any] = {
            "query": q,
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
                    "query": q,
                    "error": data.get("error"),
                    "status_code": data.get("status_code"),
                    "response_snippet": data.get("response_snippet"),
                    "client_error": data.get("client_error"),
                    "elapsed_sec": elapsed,
                    "truncated": truncated,
                }
            )
            error_result = [
                {
                    "source": "error",
                    "title": "Tavily Search Error",
                    "url": "",
                    "snippet": str(data.get("error") or "unknown_error"),
                    "score": None,
                    "error": str(data.get("error") or "unknown_error"),
                    "status_code": data.get("status_code"),
                    "response_snippet": data.get("response_snippet"),
                    "client_error": data.get("client_error"),
                    "pubstub": data.get("pubstub", False),
                }
            ]
            self._cache_set(cache_key, {"results": error_result})
            return error_result

        results = self._normalize_results(data.get("results", data))
        if len(results) > max_results:
            results = results[:max_results]

        self._log_event(
            {
                "event": "search",
                "query": q,
                "max_results": max_results,
                "search_depth": search_depth,
                "topic": topic,
                "time_range": time_range,
                "elapsed_sec": elapsed,
                "num_results": len(results),
                "truncated": truncated,
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
        """
        q, truncated = self._clamp_query(query)
        max_results = max(1, int(max_results or self.default_max_results))

        if not q:
            return {"answer": None, "results": []}

        if self.is_stub_mode():
            self._log_event(
                {
                    "event": "stub_search_with_answer",
                    "query": q,
                    "max_results": max_results,
                    "truncated": truncated,
                }
            )
            return {
                "answer": None,
                "results": [
                    {
                        "source": "stub",
                        "title": f"Stubbed search result for: {q}",
                        "url": "https://example.com/stub",
                        "snippet": (
                            "[stub] Stubbed search. Configure TAVILY_API_KEY and "
                            "set TAVILY_STUB_MODE=0, DISABLE_WEB_SEARCH=0, ENABLE_TAVILY=1 "
                            "to enable real web search."
                        ),
                        "score": None,
                    }
                ],
            }

        payload: Dict[str, Any] = {
            "query": q,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,
            "include_raw_content": False,
            "topic": topic,
            "time_range": time_range,
            "auto_parameters": False,
            "include_images": False,
        }

        start = time.time()
        data = self._tavily_client_search(payload)
        elapsed = time.time() - start

        if "error" in data:
            self._log_event(
                {
                    "event": "search_with_answer_error",
                    "query": q,
                    "error": data.get("error"),
                    "status_code": data.get("status_code"),
                    "response_snippet": data.get("response_snippet"),
                    "client_error": data.get("client_error"),
                    "elapsed_sec": elapsed,
                    "truncated": truncated,
                }
            )
            return {
                "answer": None,
                "results": [
                    {
                        "source": "error",
                        "title": "Tavily Search Error",
                        "url": "",
                        "snippet": str(data.get("error") or "unknown_error"),
                        "score": None,
                        "error": str(data.get("error") or "unknown_error"),
                        "status_code": data.get("status_code"),
                        "response_snippet": data.get("response_snippet"),
                        "client_error": data.get("client_error"),
                        "pubstub": data.get("pubstub", False),
                    }
                ],
            }

        answer = data.get("answer") if isinstance(data, dict) else None
        results = self._normalize_results(data.get("results", data))
        if len(results) > max_results:
            results = results[:max_results]

        self._log_event(
            {
                "event": "search_with_answer",
                "query": q,
                "max_results": max_results,
                "search_depth": search_depth,
                "topic": topic,
                "time_range": time_range,
                "elapsed_sec": elapsed,
                "num_results": len(results),
                "truncated": truncated,
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
                "content": str (capped),
                "content_type": str or None,
            }
        """
        url = (url or "").strip()
        if not url:
            return {"url": url, "status": "error", "content": "empty_url", "content_type": None}

        if requests is None:
            err_msg = "requests library is not available."
            self._log_event({"event": "fetch_error", "url": url, "error": err_msg})
            return {"url": url, "status": "error", "content": err_msg, "content_type": None}

        try:
            start = time.time()
            r = requests.get(url, timeout=self.timeout, verify=self.verify_ssl)
            elapsed = time.time() - start
            content_type = (r.headers.get("Content-Type") or "").strip()
            text = r.text or ""

            self._log_event(
                {
                    "event": "fetch_url",
                    "url": url,
                    "status": r.status_code,
                    "elapsed_sec": elapsed,
                    "content_type": content_type,
                }
            )

            return {
                "url": url,
                "status": r.status_code,
                "content": text[:8000],
                "content_type": content_type,
            }
        except Exception as e:
            self._log_event({"event": "fetch_error", "url": url, "error": str(e)})
            return {"url": url, "status": "error", "content": str(e), "content_type": None}

    def fetch_page(self, url: str) -> BrowserPage:
        """
        Helper used by TGRM for deep single page inspection.
        """
        result = self.fetch_url(url)
        status = result.get("status")
        content_type = result.get("content_type")
        content = result.get("content", "") or ""
        error: Optional[str] = None

        if status == "error" or isinstance(status, str):
            error = str(content) if content else "fetch_error"
            text_snippet = ""
        else:
            text_snippet = str(content)[:2000]

        return BrowserPage(
            url=result.get("url", url),
            text_snippet=text_snippet,
            status=status,
            content_type=content_type,
            error=error,
        )

    # ------------------------------------------------------------------
    # Optional helpers for summaries and citations
    # ------------------------------------------------------------------
    def summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Create a simple multi line summary from normalized results."""
        if not results:
            return "No results found."

        lines: List[str] = []
        for idx, r in enumerate(results[:6], start=1):
            title = r.get("title") or "(no title)"
            snippet = (r.get("snippet") or "").replace("\n", " ").strip()
            lines.append(f"{idx}. {title}: {snippet[:300]}")

        return "\n".join(lines)

    def to_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert normalized results into standard citation objects."""
        cites: List[Dict[str, str]] = []
        for r in results:
            cites.append(
                {
                    "source": r.get("source") or "web",
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("snippet") or "",
                }
            )
        return cites

    # Small alias so other parts of the agent can call more flexibly
    def __call__(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        return self.search(query=query, max_results=max_results)
