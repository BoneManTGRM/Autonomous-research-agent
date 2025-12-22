# agent/tools.py

"""
Tooling layer for the Autonomous Research Agent.

Includes:
- Headless browser wrapper (navigation, scraping, simple actions)
- Multi step browser automation for forms, clicks, scrolling, downloads
- Code execution sandbox for safe Python snippets (in process and subprocess)
- Lightweight math safe_eval
- Data pipeline helpers for CSV / Excel / Parquet / JSON / NDJSON / basic SQL
- HTML table loaders and simple URL based loaders
- Simple cost accounting hooks so tool usage can be reflected in Energy E

All tools are designed to be:
- Optional (if a dependency is missing, they degrade gracefully)
- Side effect aware (log what happened for RYE and MemoryStore)
- Usable by a swarm of agents without breaking isolation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import io
import json
import os
import sqlite3
import textwrap
import traceback
import math
import subprocess
import time
import threading
import random

# Optional HTTP and HTML parsing for browser like scraping
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

# Optional Playwright for real headless browser automation
try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:
    sync_playwright = None  # type: ignore

# Optional pandas for data pipelines
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# Optional PubMed ingestion tool. This is loaded lazily to allow
# deployments without PubMed support to degrade gracefully. When the
# import fails, PubMedTool will be set to None and pubmed_search will
# return stubbed results.
try:
    from .tools_pubmed import PubMedTool  # type: ignore
except Exception:
    PubMedTool = None  # type: ignore

# Optional SQLAlchemy for richer SQL
try:
    import sqlalchemy  # type: ignore
except Exception:
    sqlalchemy = None  # type: ignore

# Tavily based web search / fetch tool (tools_web.BrowserTool)
try:
    from .tools_web import BrowserTool as TavilyBrowserTool  # type: ignore
except Exception:
    TavilyBrowserTool = None  # type: ignore

# Singleton instance for module level web_search bridge
_WEB_RESEARCH_INSTANCE: Optional[Any] = None

# ---------------------------------------------------------------------------
# Tavily concurrency and backoff controls
#
# When performing web_search via TavilyBrowserTool, it is easy to exhaust
# remote API limits or trigger timeouts when many agents hit the service at
# once. To mitigate this, we gate calls behind a global semaphore and
# implement exponential backoff when certain classes of errors occur. The
# concurrency limit and retry counts can be tuned via environment variables
# `TAVILY_MAX_CONCURRENCY` and `TAVILY_MAX_RETRIES`. If these are not set,
# sensible defaults are used (4 concurrent calls and 3 retries). The
# backoff includes jitter to avoid synchronized retries across agents.

# When only a single web search service is available for all agents, the
# default concurrency limit should be low to avoid saturating that shared
# endpoint. We default to 1 concurrent call when the environment does not
# override it via TAVILY_MAX_CONCURRENCY. You can raise this via the env var
# if multiple independent search services are available.
_DEFAULT_TAVILY_MAX_CONCURRENCY = 1
_DEFAULT_TAVILY_MAX_RETRIES = 3

# Read concurrency and retry settings from environment variables. Fall back to
# defaults if not set or invalid. Keep the values within reasonable bounds
# to avoid resource exhaustion.
def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        val = int(raw)
        # Constrain to at least 1 to avoid deadlock from zero concurrency
        return max(val, 1)
    except Exception:
        return default

_TAVILY_MAX_CONCURRENCY = _parse_int_env("TAVILY_MAX_CONCURRENCY", _DEFAULT_TAVILY_MAX_CONCURRENCY)
_TAVILY_MAX_RETRIES = _parse_int_env("TAVILY_MAX_RETRIES", _DEFAULT_TAVILY_MAX_RETRIES)

# --------------------------------------------------------------------------
# Tavily search hard timeout + fail-fast safety
#
# ARA can appear "stuck" if the underlying Tavily client hangs (socket stall,
# DNS/TLS issues, etc.) because the agent waits for a web_search tool call to
# return before it can advance cycles.
#
# We enforce a hard upper bound on each Tavily search call by running it in a
# daemon thread and joining with a timeout. If it times out, we fail fast and
# fall back to the built-in BrowserTool (DuckDuckGo HTML) so the run keeps
# making forward progress.
#
# You can tune this via env vars on the *worker* service:
#   - TAVILY_TIMEOUT_SECONDS (float, default 20)
#   - TAVILY_DISABLE_AFTER_TIMEOUTS (int, default 1)
#   - TAVILY_COOLDOWN_SECONDS (float, default 900)

def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


_DEFAULT_TAVILY_TIMEOUT_S: float = 20.0
_TAVILY_TIMEOUT_S: float = _parse_float_env("TAVILY_TIMEOUT_SECONDS", _DEFAULT_TAVILY_TIMEOUT_S)

# If Tavily times out repeatedly, disable it temporarily so the run continues
# offline rather than spending minutes retrying.
_TAVILY_DISABLE_AFTER_TIMEOUTS: int = _parse_int_env("TAVILY_DISABLE_AFTER_TIMEOUTS", 1)
_TAVILY_COOLDOWN_SECONDS: float = _parse_float_env("TAVILY_COOLDOWN_SECONDS", 900.0)

_tavily_state_lock = threading.Lock()
_tavily_timeout_count: int = 0
_tavily_disabled_until_ts: float = 0.0


def _log_tools(msg: str) -> None:
    """Best-effort log to stdout (Render captures stdout/stderr)."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _tavily_is_temporarily_disabled() -> bool:
    with _tavily_state_lock:
        return time.time() < _tavily_disabled_until_ts


def _tavily_register_timeout(reason: str) -> None:
    """Track Tavily timeouts and disable it for a cooldown window if needed."""
    global _tavily_timeout_count, _tavily_disabled_until_ts, _WEB_RESEARCH_INSTANCE
    now = time.time()
    with _tavily_state_lock:
        _tavily_timeout_count += 1
        if _tavily_timeout_count >= max(1, _TAVILY_DISABLE_AFTER_TIMEOUTS):
            _tavily_timeout_count = 0
            _tavily_disabled_until_ts = now + max(0.0, _TAVILY_COOLDOWN_SECONDS)
            # Drop the shared instance so future calls can re-init after cooldown.
            _WEB_RESEARCH_INSTANCE = None
            _log_tools(
                f"[tools:web_search] Tavily disabled for {int(_TAVILY_COOLDOWN_SECONDS)}s "
                f"after timeout: {reason}"
            )
        else:
            _log_tools(
                f"[tools:web_search] Tavily timeout ({_tavily_timeout_count}/{_TAVILY_DISABLE_AFTER_TIMEOUTS}): {reason}"
            )

# Global semaphore to limit concurrent Tavily search calls. Protects both
# synchronous and potential asynchronous contexts. Note: Playwright and
# requests within TavilyBrowserTool are already concurrent safe; this is an
# extra guard at the tool layer.
_tavily_semaphore = threading.Semaphore(_TAVILY_MAX_CONCURRENCY)


def _get_web_research_instance() -> Optional[Any]:
    """
    Lazily construct a shared TavilyBrowserTool instance.

    This is the backing implementation for web_search when available.
    """
    global _WEB_RESEARCH_INSTANCE
    if _WEB_RESEARCH_INSTANCE is not None:
        return _WEB_RESEARCH_INSTANCE
    if TavilyBrowserTool is None:
        return None
    try:
        _WEB_RESEARCH_INSTANCE = TavilyBrowserTool()
    except Exception:
        _WEB_RESEARCH_INSTANCE = None
    return _WEB_RESEARCH_INSTANCE


# ----------------------------------------------------------
# Cost accounting structure for tools
# ----------------------------------------------------------


@dataclass
class ToolUsage:
    """Aggregate info about how much energy the tools consumed."""

    web_calls: int = 0
    browser_actions: int = 0
    code_execs: int = 0
    sql_queries: int = 0
    data_loads: int = 0
    api_calls: int = 0
    downloads: int = 0
    approx_tokens: int = 0

    # Extra channels for compute_energy (kept separate from web_calls)
    semantic_calls: int = 0
    pubmed_calls: int = 0
    pdf_ingestions: int = 0

    def add_tokens_for_text(self, text: str) -> None:
        """Very rough token estimate from text length."""
        if not text:
            return
        # Approx 4 characters per token as a rough heuristic
        self.approx_tokens += max(1, len(text) // 4)

    def add_tokens_for_code(self, code: str) -> None:
        """Rough token estimate for code snippets."""
        if not code:
            return
        self.approx_tokens += max(1, len(code) // 3)

    # Convenience helpers for common tool events
    def record_web_call(self, text: str = "") -> None:
        self.web_calls += 1
        if text:
            self.add_tokens_for_text(text)

    def record_semantic_call(self, text: str = "") -> None:
        self.semantic_calls += 1
        if text:
            self.add_tokens_for_text(text)

    def record_pubmed_call(self, text: str = "") -> None:
        self.pubmed_calls += 1
        if text:
            self.add_tokens_for_text(text)

    def record_pdf_ingestion(self, approx_text: str = "") -> None:
        self.pdf_ingestions += 1
        if approx_text:
            self.add_tokens_for_text(approx_text)

    def merge(self, other: "ToolUsage") -> None:
        """Merge another ToolUsage into this instance."""
        self.web_calls += other.web_calls
        self.browser_actions += other.browser_actions
        self.code_execs += other.code_execs
        self.sql_queries += other.sql_queries
        self.data_loads += other.data_loads
        self.api_calls += other.api_calls
        self.downloads += other.downloads
        self.approx_tokens += other.approx_tokens
        self.semantic_calls += other.semantic_calls
        self.pubmed_calls += other.pubmed_calls
        self.pdf_ingestions += other.pdf_ingestions

    def to_energy_kwargs(self) -> Dict[str, Any]:
        """
        Convert tool usage into kwargs for compute_energy.

        The CoreAgent can feed this into compute_energy as:
            energy_e = compute_energy(
                actions_taken=actions,
                web_calls=tool_usage.web_calls,
                semantic_calls=tool_usage.semantic_calls,
                pubmed_calls=tool_usage.pubmed_calls,
                pdf_ingestions=tool_usage.pdf_ingestions,
                tokens_estimate=tool_usage.approx_tokens,
                swarm_size=swarm_size,
                swarm_layering=swarm_layering,
            )
        """
        return {
            "web_calls": self.web_calls,
            "semantic_calls": self.semantic_calls,
            "pubmed_calls": self.pubmed_calls,
            "pdf_ingestions": self.pdf_ingestions,
            "tokens_estimate": self.approx_tokens,
        }


# ----------------------------------------------------------
# Browser tools (Playwright/requests scraper)
# ----------------------------------------------------------


@dataclass
class BrowserResult:
    url: str
    status_code: Optional[int]
    title: str
    text_snippet: str
    html: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BrowserStepLog:
    step_index: int
    op: str
    success: bool
    details: str
    error: Optional[str] = None


@dataclass
class BrowserAutomationResult:
    start_url: str
    final_url: Optional[str]
    title: str
    text_snippet: str
    html: Optional[str]
    screenshot_path: Optional[str]
    steps: List[BrowserStepLog]
    error: Optional[str] = None


class BrowserTool:
    """
    Headless browser and scraper hybrid (Playwright + requests).

    This is focused on navigation, scraping, and simple automation.
    High level web search is handled via the TavilyBrowserTool bridge
    in web_search().
    """

    def __init__(self, user_agent: Optional[str] = None, timeout: float = 20.0) -> None:
        self.user_agent = user_agent or "ReparodynamicsAgent/0.1"
        self.timeout = timeout

    # Simple one shot fetch
    def fetch_page(self, url: str) -> BrowserResult:
        """Fetch a page in a best effort way and return text plus metadata."""
        # Try Playwright first
        if sync_playwright is not None:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page(user_agent=self.user_agent)
                    page.goto(url, timeout=self.timeout * 1000)
                    content = page.content()
                    title = page.title() or ""
                    final_url = page.url
                    browser.close()

                text = content
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)

                snippet = text[:2000]
                # Only keep truncated HTML for traceability
                html_trunc = content[:10000]

                return BrowserResult(
                    url=final_url,
                    status_code=None,
                    title=title,
                    text_snippet=snippet,
                    html=html_trunc,
                    error=None,
                )
            except Exception as e:
                return BrowserResult(
                    url=url,
                    status_code=None,
                    title="",
                    text_snippet="",
                    html=None,
                    error=f"Playwright error: {e}",
                )

        # Fallback: plain HTTP
        if requests is None:
            return BrowserResult(
                url=url,
                status_code=None,
                title="",
                text_snippet="",
                html=None,
                error="requests library not available",
            )

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            status = resp.status_code
            html = resp.text

            if BeautifulSoup is not None:
                soup = BeautifulSoup(html, "html.parser")
                title_el = soup.find("title")
                title = title_el.get_text(strip=True) if title_el else ""
                text = soup.get_text(separator=" ", strip=True)
            else:
                title = ""
                text = html

            snippet = text[:2000]
            html_trunc = html[:10000]

            return BrowserResult(
                url=url,
                status_code=status,
                title=title,
                text_snippet=snippet,
                html=html_trunc,
                error=None,
            )
        except Exception as e:
            return BrowserResult(
                url=url,
                status_code=None,
                title="",
                text_snippet="",
                html=None,
                error=f"HTTP error: {e}",
            )

    # Multi step automation for one session
    def run_actions(
        self,
        start_url: str,
        actions: List[Dict[str, Any]],
        screenshot_dir: Optional[str] = None,
        screenshot_name: str = "automation_final.png",
    ) -> BrowserAutomationResult:
        """
        Run a small browser workflow against a site using Playwright if available.

        Actions are a list of dicts with "op" and other keys.
        Supported ops:
            - "goto": {"op": "goto", "url": "..."}
            - "click": {"op": "click", "selector": "..."}
            - "fill": {"op": "fill", "selector": "...", "value": "..."}
            - "type": {"op": "type", "selector": "...", "value": "..."}
            - "wait_for": {"op": "wait_for", "selector": "...", "timeout_ms": 10000}
            - "scroll_bottom": {"op": "scroll_bottom"}
            - "eval": {"op": "eval", "script": "return document.title;"}
        """
        if sync_playwright is None:
            return BrowserAutomationResult(
                start_url=start_url,
                final_url=None,
                title="",
                text_snippet="",
                html=None,
                screenshot_path=None,
                steps=[
                    BrowserStepLog(
                        step_index=0,
                        op="init",
                        success=False,
                        details="Playwright is not installed; cannot run actions",
                        error="Playwright missing",
                    )
                ],
                error="Playwright missing",
            )

        steps_log: List[BrowserStepLog] = []
        final_html: Optional[str] = None
        final_text: str = ""
        final_title: str = ""
        final_url: Optional[str] = None
        screenshot_path: Optional[str] = None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=self.user_agent)

                # First navigation
                try:
                    page.goto(start_url, timeout=self.timeout * 1000)
                    steps_log.append(
                        BrowserStepLog(
                            step_index=-1,
                            op="goto_initial",
                            success=True,
                            details=f"Navigated to {start_url}",
                        )
                    )
                except Exception as e:
                    steps_log.append(
                        BrowserStepLog(
                            step_index=-1,
                            op="goto_initial",
                            success=False,
                            details=f"Failed to navigate to {start_url}",
                            error=str(e),
                        )
                    )
                    browser.close()
                    return BrowserAutomationResult(
                        start_url=start_url,
                        final_url=None,
                        title="",
                        text_snippet="",
                        html=None,
                        screenshot_path=None,
                        steps=steps_log,
                        error=f"Initial navigation failed: {e}",
                    )

                # Apply actions
                for idx, action in enumerate(actions):
                    op = str(action.get("op", "")).lower().strip()
                    try:
                        if op == "goto":
                            url = action.get("url") or start_url
                            page.goto(url, timeout=self.timeout * 1000)
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=True,
                                    details=f"Navigated to {url}",
                                )
                            )
                        elif op == "click":
                            sel = action["selector"]
                            page.click(sel, timeout=self.timeout * 1000)
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=True,
                                    details=f"Clicked {sel}",
                                )
                            )
                        elif op == "fill":
                            sel = action["selector"]
                            val = action.get("value", "")
                            page.fill(sel, val, timeout=self.timeout * 1000)
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=True,
                                    details=f"Filled {sel} with value of length {len(val)}",
                                )
                            )
                        elif op == "type":
                            sel = action["selector"]
                            val = action.get("value", "")
                            page.type(sel, val, timeout=self.timeout * 1000)
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=True,
                                    details=f"Typed into {sel} with value of length {len(val)}",
                                )
                            )
                        elif op == "wait_for":
                            sel = action["selector"]
                            timeout_ms = int(action.get("timeout_ms", self.timeout * 1000))
                            page.wait_for_selector(sel, timeout=timeout_ms)
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=True,
                                    details=f"Waited for selector {sel}",
                                )
                            )
                        elif op == "scroll_bottom":
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=True,
                                    details="Scrolled to bottom of page",
                                )
                            )
                        elif op == "eval":
                            script = action.get("script", "")
                            res = page.evaluate(script)
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=True,
                                    details=f"Eval success, result: {res}",
                                )
                            )
                        else:
                            steps_log.append(
                                BrowserStepLog(
                                    step_index=idx,
                                    op=op,
                                    success=False,
                                    details=f"Unknown op: {op}",
                                    error="unknown op",
                                )
                            )
                    except Exception as e:
                        steps_log.append(
                            BrowserStepLog(
                                step_index=idx,
                                op=op or "unknown",
                                success=False,
                                details=f"Error running op {op}",
                                error=str(e),
                            )
                        )

                # Final snapshot
                final_url = page.url
                final_html = page.content()
                final_title = page.title() or ""

                text = final_html
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(final_html, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                final_text = text[:2000]
                html_trunc = final_html[:20000]

                # Screenshot
                if screenshot_dir:
                    try:
                        os.makedirs(screenshot_dir, exist_ok=True)
                        shot_path = os.path.join(screenshot_dir, screenshot_name)
                        page.screenshot(path=shot_path, full_page=True)
                        screenshot_path = shot_path
                    except Exception:
                        screenshot_path = None

                browser.close()

                return BrowserAutomationResult(
                    start_url=start_url,
                    final_url=final_url,
                    title=final_title,
                    text_snippet=final_text,
                    html=html_trunc,
                    screenshot_path=screenshot_path,
                    steps=steps_log,
                    error=None,
                )
        except Exception as e:
            return BrowserAutomationResult(
                start_url=start_url,
                final_url=None,
                title="",
                text_snippet="",
                html=None,
                screenshot_path=None,
                steps=steps_log,
                error=str(e),
            )


# ----------------------------------------------------------
# Code execution sandbox
# ----------------------------------------------------------


@dataclass
class CodeExecutionResult:
    stdout: str
    stderr: str
    error: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeSandbox:
    """
    Python code execution sandbox.

    Two levels:
        1) In process exec with restricted builtins
        2) Subprocess exec for extra isolation

    Intended use:
    - Quick math checks
    - Data frame transforms
    - Small simulations
    - Verifying formulas and models
    """

    def __init__(self, max_code_chars: int = 40000) -> None:
        self.max_code_chars = max_code_chars

    def safe_eval(self, expr: str) -> CodeExecutionResult:
        """Evaluate a simple numeric expression with math only."""
        allowed_names = {
            k: getattr(math, k)
            for k in dir(math)
            if not k.startswith("_")
        }
        allowed_names.update({"abs": abs, "min": min, "max": max, "sum": sum})

        try:
            value = eval(expr, {"__builtins__": {}}, allowed_names)
            return CodeExecutionResult(
                stdout=str(value),
                stderr="",
                error=None,
                metadata={"expr": expr},
            )
        except Exception as e:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error=str(e),
                metadata={"expr": expr},
            )

    def run_python(
        self,
        code: str,
        *,
        extra_globals: Optional[Dict[str, Any]] = None,
        allow_numpy: bool = True,
        allow_pandas: bool = True,
    ) -> CodeExecutionResult:
        """Run a Python snippet in process and capture output."""
        if not code:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error="Empty code snippet",
                metadata={},
            )

        if len(code) > self.max_code_chars:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error="Code too long for in process sandbox",
                metadata={"truncated": True},
            )

        # Limit visible builtins
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "print": print,
            "enumerate": enumerate,
            "zip": zip,
        }

        sandbox_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "math": math,
        }

        # Optionally expose numpy and pandas if available
        if allow_numpy:
            try:
                import numpy as _np  # type: ignore

                sandbox_globals["np"] = _np
            except Exception:
                pass

        if allow_pandas and pd is not None:
            sandbox_globals["pd"] = pd  # type: ignore

        if extra_globals:
            sandbox_globals.update(extra_globals)

        sandbox_locals: Dict[str, Any] = {}

        import contextlib
        import io as _io
        import sys  # noqa: F401  (kept for potential future use in sandbox)

        stdout_buffer = _io.StringIO()
        stderr_buffer = _io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                exec(code, sandbox_globals, sandbox_locals)
            err = None
        except Exception:
            err = traceback.format_exc()

        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()

        return CodeExecutionResult(
            stdout=stdout_text,
            stderr=stderr_text,
            error=err,
            metadata={"locals_keys": list(sandbox_locals.keys())},
        )

    def run_python_subprocess(
        self,
        code: str,
        *,
        timeout_sec: int = 30,
        python_executable: str = "python",
    ) -> CodeExecutionResult:
        """
        Run code in a separate Python process for extra isolation.

        This loses direct access to extra_globals but gains:
        - process level timeout
        - separation from the main agent interpreter
        """
        if not code:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error="Empty code snippet",
                metadata={"mode": "subprocess"},
            )

        if len(code) > self.max_code_chars:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error="Code too long for subprocess sandbox",
                metadata={"mode": "subprocess", "truncated": True},
            )

        try:
            proc = subprocess.run(
                [python_executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            err = None
            if proc.returncode != 0:
                err = f"Non zero return code {proc.returncode}"
            return CodeExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                error=err,
                metadata={"mode": "subprocess", "returncode": proc.returncode},
            )
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error=f"Timeout after {timeout_sec} seconds",
                metadata={"mode": "subprocess", "timeout": timeout_sec},
            )
        except Exception as e:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error=str(e),
                metadata={"mode": "subprocess"},
            )


# ----------------------------------------------------------
# Data pipelines
# ----------------------------------------------------------


@dataclass
class DataLoadResult:
    name: str
    rows: int
    cols: int
    preview: List[Dict[str, Any]]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataPipelines:
    """
    Data loading helpers for CSV, Excel, Parquet, JSON, and simple SQL.

    These are used for:
    - Loading experiment datasets
    - Feeding time series into RYE analysis
    - Pulling lab or biomarker tables
    - Inspecting SQL tables via SQLite or optional SQLAlchemy
    """

    def __init__(self) -> None:
        self.has_pandas = pd is not None

    # Core frame helper
    def _df_to_result(self, df: "pd.DataFrame", name: str) -> DataLoadResult:  # type: ignore[name-defined]
        preview = df.head(10).to_dict(orient="records")  # type: ignore

        metadata: Dict[str, Any] = {
            "columns": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        }

        return DataLoadResult(
            name=name,
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
            preview=preview,
            error=None,
            metadata=metadata,
        )

    def load_csv(self, file_bytes: bytes, name: str = "uploaded.csv") -> DataLoadResult:
        if not self.has_pandas:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error="pandas is not installed",
            )
        try:
            buf = io.BytesIO(file_bytes)
            df = pd.read_csv(buf)  # type: ignore
            return self._df_to_result(df, name)
        except Exception as e:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def load_excel(self, file_bytes: bytes, name: str = "uploaded.xlsx") -> DataLoadResult:
        if not self.has_pandas:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error="pandas is not installed",
            )
        try:
            buf = io.BytesIO(file_bytes)
            df = pd.read_excel(buf)  # type: ignore
            return self._df_to_result(df, name)
        except Exception as e:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def load_parquet(self, file_bytes: bytes, name: str = "uploaded.parquet") -> DataLoadResult:
        if not self.has_pandas:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error="pandas is not installed",
            )
        try:
            buf = io.BytesIO(file_bytes)
            df = pd.read_parquet(buf)  # type: ignore
            return self._df_to_result(df, name)
        except Exception as e:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def load_json(self, file_bytes: bytes, name: str = "uploaded.json") -> DataLoadResult:
        if not self.has_pandas:
            # Fallback: simple JSON parse without DataFrame
            try:
                text = file_bytes.decode("utf-8")
                data = json.loads(text)
                if isinstance(data, list):
                    preview = data[:10]
                else:
                    preview = [data]
                return DataLoadResult(
                    name=name,
                    rows=len(preview),
                    cols=len(preview[0]) if preview and isinstance(preview[0], dict) else 0,
                    preview=preview,
                    error=None,
                    metadata={"raw_json": True},
                )
            except Exception as e:
                return DataLoadResult(
                    name=name,
                    rows=0,
                    cols=0,
                    preview=[],
                    error=str(e),
                )
        try:
            text = file_bytes.decode("utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                df = pd.DataFrame(data)  # type: ignore
            else:
                df = pd.json_normalize(data)  # type: ignore
            return self._df_to_result(df, name)
        except Exception as e:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def load_ndjson(self, file_bytes: bytes, name: str = "uploaded.ndjson") -> DataLoadResult:
        if not self.has_pandas:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error="pandas is not installed",
            )
        try:
            text = file_bytes.decode("utf-8")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            records = []
            for ln in lines:
                try:
                    records.append(json.loads(ln))
                except Exception:
                    continue
            df = pd.DataFrame(records)  # type: ignore
            return self._df_to_result(df, name)
        except Exception as e:
            return DataLoadResult(
                name=name,
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def load_html_table_from_url(
        self,
        url: str,
        table_index: int = 0,
        name: Optional[str] = None,
    ) -> DataLoadResult:
        if not self.has_pandas:
            return DataLoadResult(
                name=name or f"html:{url}",
                rows=0,
                cols=0,
                preview=[],
                error="pandas is not installed",
            )
        if requests is None:
            return DataLoadResult(
                name=name or f"html:{url}",
                rows=0,
                cols=0,
                preview=[],
                error="requests is not installed",
            )
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            html = resp.text
            tables = pd.read_html(html)  # type: ignore
            if not tables:
                return DataLoadResult(
                    name=name or f"html:{url}",
                    rows=0,
                    cols=0,
                    preview=[],
                    error="No tables found in HTML",
                )
            index = max(0, min(table_index, len(tables) - 1))
            df = tables[index]
            return self._df_to_result(df, name or f"html_table_{index}")
        except Exception as e:
            return DataLoadResult(
                name=name or f"html:{url}",
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def load_from_url(self, url: str) -> DataLoadResult:
        """
        Convenience helper:
            - If URL ends with .csv, .json, .parquet, .xlsx, try those decoders
            - Otherwise, try HTML table extraction
        """
        lower = url.lower()
        try:
            if any(lower.endswith(ext) for ext in [".csv", ".tsv"]):
                if requests is None:
                    return DataLoadResult(
                        name=url,
                        rows=0,
                        cols=0,
                        preview=[],
                        error="requests is not installed",
                    )
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                return self.load_csv(resp.content, name=os.path.basename(url))

            if lower.endswith(".json"):
                if requests is None:
                    return DataLoadResult(
                        name=url,
                        rows=0,
                        cols=0,
                        preview=[],
                        error="requests is not installed",
                    )
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                return self.load_json(resp.content, name=os.path.basename(url))

            if any(lower.endswith(ext) for ext in [".parquet"]):
                if requests is None:
                    return DataLoadResult(
                        name=url,
                        rows=0,
                        cols=0,
                        preview=[],
                        error="requests is not installed",
                    )
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                return self.load_parquet(resp.content, name=os.path.basename(url))

            if any(lower.endswith(ext) for ext in [".xlsx", ".xls"]):
                if requests is None:
                    return DataLoadResult(
                        name=url,
                        rows=0,
                        cols=0,
                        preview=[],
                        error="requests is not installed",
                    )
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                return self.load_excel(resp.content, name=os.path.basename(url))

            # Fallback to HTML table extraction
            return self.load_html_table_from_url(url, table_index=0, name=url)
        except Exception as e:
            return DataLoadResult(
                name=url,
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def query_sqlite(
        self,
        db_path: str,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> DataLoadResult:
        """Run a read only query against a local SQLite database."""
        params = params or ()
        try:
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall()
            finally:
                conn.close()

            preview_rows = rows[:10]
            preview = [dict(zip(cols, r)) for r in preview_rows]
            return DataLoadResult(
                name=f"sqlite:{db_path}",
                rows=len(rows),
                cols=len(cols),
                preview=preview,
                error=None,
            )
        except Exception as e:
            return DataLoadResult(
                name=f"sqlite:{db_path}",
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )

    def query_sqlalchemy(
        self,
        conn_str: str,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> DataLoadResult:
        """
        Optional richer SQL via SQLAlchemy.

        conn_str example:
            "sqlite:///mydb.sqlite"
            "postgresql://user:pass@host:5432/dbname"
        """
        if sqlalchemy is None or not self.has_pandas:
            return DataLoadResult(
                name=name or f"sqlalchemy:{conn_str}",
                rows=0,
                cols=0,
                preview=[],
                error="sqlalchemy or pandas is not installed",
            )
        try:
            engine = sqlalchemy.create_engine(conn_str)  # type: ignore
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params=params)  # type: ignore
            return self._df_to_result(df, name or f"sqlalchemy:{conn_str}")
        except Exception as e:
            return DataLoadResult(
                name=name or f"sqlalchemy:{conn_str}",
                rows=0,
                cols=0,
                preview=[],
                error=str(e),
            )


# ----------------------------------------------------------
# Tavily based web search bridge
# ----------------------------------------------------------


def web_search(
    query: str,
    *,
    tool_usage: Optional[ToolUsage] = None,
    max_results: int = 8,
    search_depth: str = "advanced",
    topic: str = "general",
    **extra: Any,
) -> Dict[str, Any]:
    """
    Unified web search entry point for the agent.

    This is a bridge into the Tavily based BrowserTool in tools_web.
    It also maintains the classic signature expected by CoreAgent / engine_worker.

    - Records tool_usage.web_calls
    - Uses external search when available
    - Falls back to a browser stub if Tavily is missing or disabled
    """
    if tool_usage is not None:
        tool_usage.record_web_call(query)

    # Global disable flag for fully offline or analysis only runs
    disable_flag = str(os.getenv("DISABLE_WEB_SEARCH", "")).strip().lower()
    if disable_flag in {"1", "true", "yes", "on"}:
        return {
            "query": query,
            "stubbed": True,
            "error": "Web search is disabled by DISABLE_WEB_SEARCH environment variable",
            "results": [],
            "response_time": None,
            "request_id": None,
            "info_gain": None,
            "search_energy": None,
            "difficulty": None,
            "semantic_diversity": None,
        }

    # If Tavily recently stalled, fail fast and fall back so the run keeps
    # progressing. This avoids the "job running" heartbeat loop with no
    # cycle advancement.
    error_str: Optional[str] = None
    tool = None
    if _tavily_is_temporarily_disabled():
        error_str = "Tavily temporarily disabled due to previous timeouts"
    else:
        tool = _get_web_research_instance()

    # Map search_depth to something sensible for TavilyBrowserTool
    depth_norm = (search_depth or "advanced").strip().lower()
    if depth_norm in {"basic", "shallow", "cheap"}:
        mapped_depth = "basic"
    elif depth_norm in {"deep", "deepdive", "max"}:
        mapped_depth = "advanced"
    else:
        mapped_depth = "advanced"

    if tool is not None:
        # Attempt to call Tavily search with concurrency limits and retries
        started = time.time()
        last_exception: Optional[Exception] = None
        results: Optional[List[Dict[str, Any]]] = None
        # Perform up to _TAVILY_MAX_RETRIES attempts. On specific errors
        # (HTTP 429, rate limiting, timeouts), exponential backoff is used.
        for attempt in range(_TAVILY_MAX_RETRIES):
            try:
                # Acquire semaphore before making the remote call to bound
                # concurrency. This protects against many agents flooding
                # Tavily at once.
                #
                # IMPORTANT: Tavily can sometimes hang indefinitely. To
                # prevent the whole run from freezing in cycle 1/2, execute
                # the search in a daemon thread and enforce a hard timeout.
                with _tavily_semaphore:
                    result_container: List[Any] = []
                    error_container: List[Exception] = []

                    def _search_call() -> None:
                        try:
                            res = tool.search(
                                query=query,
                                max_results=max_results,
                                search_depth=mapped_depth,
                                include_answer=False,
                                include_raw_content=False,
                                topic=topic,
                                time_range=extra.get("time_range"),
                                auto_parameters=bool(extra.get("auto_parameters", False)),
                                include_images=bool(extra.get("include_images", False)),
                            )
                            result_container.append(res)
                        except Exception as exc:
                            error_container.append(exc)

                    t = threading.Thread(target=_search_call, daemon=True)
                    t.start()
                    t.join(max(0.1, float(_TAVILY_TIMEOUT_S)))
                    if t.is_alive():
                        raise TimeoutError(
                            f"Tavily search exceeded {_TAVILY_TIMEOUT_S} seconds timeout"
                        )
                    if error_container:
                        raise error_container[0]
                    results = result_container[0] if result_container else None

                # If call succeeds, break out of retry loop
                break
            except Exception as e:
                last_exception = e
                # Inspect exception text for common transient failure patterns.
                msg = str(e).lower()
                # Check for HTTP 429 or rate limit or timeout errors. Include
                # generic timeout keywords to cover both sync and async
                # execution contexts.
                is_rate_limited = any(
                    kw in msg
                    for kw in ["429", "rate limit", "rate-limit", "too many", "quota"]
                )
                is_timeout = any(
                    kw in msg
                    for kw in ["timeout", "timed out", "time out", "connection aborted"]
                ) or isinstance(e, TimeoutError)

                # FAIL FAST on timeouts. Timeouts are the main source of
                # "job claimed" + "still running" with no cycle advance.
                if is_timeout:
                    _tavily_register_timeout(str(e))
                    break

                # For rate limits, retry with exponential backoff + jitter.
                if is_rate_limited:
                    sleep_s = min(30.0, (2 ** attempt) + random.random())
                    time.sleep(sleep_s)
                    continue  # Retry

                # For other exceptions, break and fall back to stub below
                break
        elapsed = time.time() - started

        if results is not None:
            # Inspect capabilities to determine stub status. Some deployments
            # run Tavily in stubbed/offline mode, which is reflected in
            # describe_capabilities().
            caps: Dict[str, Any] = {}
            try:
                caps = tool.describe_capabilities()
            except Exception:
                caps = {}
            stubbed = bool(
                caps.get(
                    "stub_mode",
                    True if caps.get("mode") not in {None, "real"} else False,
                )
            )
            return {
                "query": query,
                "stubbed": stubbed,
                "results": results,
                "response_time": elapsed,
                "request_id": None,
                "info_gain": None,
                "search_energy": None,
                "difficulty": None,
                "semantic_diversity": None,
            }
        # If all retries fail, fall back to stubbed search below. Attach
        # the last exception string for traceability.
        error_str: Optional[str] = None
        if last_exception is not None:
            error_str = str(last_exception)
        # proceed to fallback section after this block

    # Fallback: browser based stub search so the system keeps working
    # Fallback: browser based stub search so the system keeps working. This
    # fallback is used when Tavily is unavailable or repeated failures occur.
    browser = BrowserTool()
    url = f"https://duckduckgo.com/html/?q={query}"
    page = browser.fetch_page(url)

    fallback_error = None
    # Prefer the recorded Tavily failure message if present; otherwise
    # propagate any browser fetch error.
    if 'error_str' in locals() and error_str:
        fallback_error = error_str
    elif page.error:
        fallback_error = page.error
    else:
        fallback_error = "TavilyBrowserTool not available"

    return {
        "query": query,
        "stubbed": True,
        "error": fallback_error,
        "results": [
            {
                "title": page.title or "",
                "url": page.url or url,
                "snippet": page.text_snippet or "",
                "source": "browser_fallback",
            }
        ],
        "response_time": None,
        "request_id": None,
        "info_gain": None,
        "search_energy": None,
        "difficulty": None,
        "semantic_diversity": None,
    }


def web_search_tool(
    query: str,
    *,
    tool_usage: Optional[ToolUsage] = None,
    max_results: int = 8,
    search_depth: str = "advanced",
    topic: str = "general",
    **extra: Any,
) -> Dict[str, Any]:
    """
    Thin wrapper used by tgrm_loop and other modules that expect a function
    named web_search_tool. Delegates to web_search with the same signature.
    """
    return web_search(
        query=query,
        tool_usage=tool_usage,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        **extra,
    )


# ----------------------------------------------------------
# PubMed search capability
# ----------------------------------------------------------

def pubmed_search(
    query: str,
    *,
    tool_usage: Optional[ToolUsage] = None,
    max_results: int = 5,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Unified PubMed search entry point for the agent.

    This wraps the PubMedTool defined in tools_pubmed.py and returns a
    dictionary structure consistent with other search tools. It records
    pubmed-specific tool usage and handles graceful degradation when the
    PubMed client is unavailable or disabled via environment variables.

    Args:
        query: Free text query string to send to PubMed. This will be
            sanitized internally by PubMedTool to remove excessive
            whitespace and truncate long queries.
        tool_usage: Optional ToolUsage tracker for accounting energy usage.
        max_results: Maximum number of results to retrieve (default 5).
        **extra: Ignored, present for signature compatibility.

    Returns:
        A dict with the following keys:
            - ``query``: the original query string.
            - ``stubbed``: boolean indicating whether results are real or stubbed.
            - ``results``: list of result dicts with title, snippet, url, source, pmid.
            - Additional metadata fields (response_time, request_id, etc.) set to None.
            - ``error``: populated only when stubbed is True and there is an error.
    """
    # Record usage tokens for query text
    if tool_usage is not None:
        tool_usage.record_pubmed_call(query)

    # Global disable flag for offline or analysis-only runs
    disable_flag = str(os.getenv("DISABLE_PUBMED_SEARCH", "")).strip().lower()
    if disable_flag in {"1", "true", "yes", "on"}:
        return {
            "query": query,
            "stubbed": True,
            "error": "PubMed search is disabled by DISABLE_PUBMED_SEARCH environment variable",
            "results": [],
            "response_time": None,
            "request_id": None,
            "info_gain": None,
            "search_energy": None,
            "difficulty": None,
            "semantic_diversity": None,
        }

    # If the PubMed tool cannot be imported, degrade gracefully
    if PubMedTool is None:
        return {
            "query": query,
            "stubbed": True,
            "error": "PubMedTool not available (dependency missing)",
            "results": [],
            "response_time": None,
            "request_id": None,
            "info_gain": None,
            "search_energy": None,
            "difficulty": None,
            "semantic_diversity": None,
        }

    # Perform the search via PubMedTool
    import time
    started = time.time()
    try:
        # Instantiate a fresh PubMedTool; email/api_key read from env
        tool = PubMedTool()
        results_list = tool.search(query, max_results=max_results)
        elapsed = time.time() - started
        # Determine if stubbed by checking for stub marker in first title
        stubbed = True
        if results_list:
            first_title = results_list[0].get("title", "")
            if not first_title.startswith("[STUB]"):
                stubbed = False
        return {
            "query": query,
            "stubbed": stubbed,
            "results": results_list,
            "response_time": elapsed,
            "request_id": None,
            "info_gain": None,
            "search_energy": None,
            "difficulty": None,
            "semantic_diversity": None,
        }
    except Exception as e:
        # On unexpected error, return a stub result with error message
        elapsed = time.time() - started
        return {
            "query": query,
            "stubbed": True,
            "error": f"PubMed search error: {e}",
            "results": [],
            "response_time": elapsed,
            "request_id": None,
            "info_gain": None,
            "search_energy": None,
            "difficulty": None,
            "semantic_diversity": None,
        }


def pubmed_search_tool(
    query: str,
    *,
    tool_usage: Optional[ToolUsage] = None,
    max_results: int = 5,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Thin wrapper used by tgrm_loop and other modules that expect a function
    named pubmed_search_tool. Delegates to pubmed_search with the same signature.
    """
    return pubmed_search(
        query=query,
        tool_usage=tool_usage,
        max_results=max_results,
        **extra,
    )


# ----------------------------------------------------------
# Toolbelt facade for CoreAgent
# ----------------------------------------------------------


class Toolbelt:
    """
    Aggregates all tools so CoreAgent can receive a single object.

    Example usage inside CoreAgent:

        tool_usage = self.tools.new_usage_tracker()

        search_res = web_search("NAD longevity review", tool_usage=tool_usage)
        # or
        page = self.tools.browser.fetch_page(url)
        tool_usage.web_calls += 1
        tool_usage.add_tokens_for_text(page.text_snippet or "")

        code_res = self.tools.sandbox.run_python(code)
        tool_usage.code_execs += 1
        tool_usage.add_tokens_for_code(code)

        energy_e = compute_energy(
            actions_taken=actions,
            **tool_usage.to_energy_kwargs(),
            swarm_size=swarm_size,
            swarm_layering=swarm_layering,
        )
    """

    def __init__(self) -> None:
        self.browser = BrowserTool()
        self.sandbox = CodeSandbox()
        self.data = DataPipelines()
        # Expose Tavily based search instance for UIs or diagnostics if needed
        self.web_research = _get_web_research_instance()

    def new_usage_tracker(self) -> ToolUsage:
        """Return a fresh ToolUsage object for one cycle."""
        return ToolUsage()


# ----------------------------------------------------------
# TOOL_REGISTRY for engine_worker and capability detection
# ----------------------------------------------------------


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Web / browser capabilities
    "web": {
        "kind": "web",
        "class": BrowserTool,
        "description": "HTTP/HTML browser and scraper (Playwright plus requests fallback).",
        # Unified search helper (Tavily bridge)
        "fn": web_search_tool,
    },
    "web_search": {
        "kind": "web",
        "class": BrowserTool,
        "description": "Alias for web/browser capability plus Tavily search module when available.",
        "fn": web_search_tool,
    },
    "tavily_search": {
        "kind": "web",
        "class": BrowserTool,
        "description": "Alias name kept for compatibility; currently routed to the generic web_search_tool.",
        "fn": web_search_tool,
    },
    "browser": {
        "kind": "web",
        "class": BrowserTool,
        "description": "Headless browser tool for navigation and scraping.",
    },
    "internet": {
        "kind": "web",
        "class": BrowserTool,
        "description": "Alias indicating internet browsing/scraping.",
    },

    # PubMed search capabilities (independent of Tavily/web)
    # The "pubmed" key exposes the core PubMed tool class for direct use
    # and the fn uses pubmed_search_tool to provide a unified search API.
    "pubmed": {
        "kind": "pubmed",
        "class": PubMedTool,
        "description": "PubMed search tool using NCBI E-utilities (scientific literature).",
        "fn": pubmed_search_tool,
    },
    "pubmed_search": {
        "kind": "pubmed",
        "class": PubMedTool,
        "description": "Alias for the PubMed search tool.",
        "fn": pubmed_search_tool,
    },

    # Code / sandbox capabilities
    "sandbox": {
        "kind": "code",
        "class": CodeSandbox,
        "description": "Python code sandbox (in-process plus subprocess).",
    },
    "code_sandbox": {
        "kind": "code",
        "class": CodeSandbox,
        "description": "Alias for Python code sandbox.",
    },
    "python_sandbox": {
        "kind": "code",
        "class": CodeSandbox,
        "description": "Alias for Python code sandbox.",
    },
    "exec_sandbox": {
        "kind": "code",
        "class": CodeSandbox,
        "description": "Alias for process-isolated Python execution.",
    },

    # Data pipeline capability
    "data": {
        "kind": "data",
        "class": DataPipelines,
        "description": "Data loading and SQL helpers for CSV/Excel/Parquet/JSON/SQL.",
    },
}
