# agent/tools.py

"""
Tooling layer for the Autonomous Research Agent.

Includes:
- Headless browser wrapper (navigation, scraping, simple actions)
- Code execution sandbox for safe Python snippets
- Data pipeline helpers for CSV / Excel / basic SQL
- Simple cost accounting hooks so tool usage can be reflected in Energy E

All tools are designed to be:
- Optional (if dependency is missing, they degrade gracefully)
- Side effect aware (log what happened for RYE and MemoryStore)
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
    approx_tokens: int = 0

    def to_energy_kwargs(self) -> Dict[str, Any]:
        """
        Convert tool usage into kwargs for compute_energy.

        The CoreAgent can feed this into compute_energy as:
            energy_e = compute_energy(
                actions_taken=actions,
                web_calls=tool_usage.web_calls,
                semantic_calls=0,
                pubmed_calls=0,
                pdf_ingestions=0,
                tokens_estimate=tool_usage.approx_tokens,
                swarm_size=swarm_size,
                swarm_layering=swarm_layering,
            )
        """
        return {
            "web_calls": self.web_calls,
            "semantic_calls": 0,
            "pubmed_calls": 0,
            "pdf_ingestions": 0,
            "tokens_estimate": self.approx_tokens,
        }


# ----------------------------------------------------------
# Browser tool
# ----------------------------------------------------------

@dataclass
class BrowserResult:
    url: str
    status_code: Optional[int]
    title: str
    text_snippet: str
    html: Optional[str] = None
    error: Optional[str] = None


class BrowserTool:
    """
    Headless browser and scraper hybrid.

    It tries to use Playwright if available.
    If not, it falls back to simple HTTP GET plus BeautifulSoup parsing.
    """

    def __init__(self, user_agent: Optional[str] = None, timeout: float = 20.0) -> None:
        self.user_agent = user_agent or "ReparodynamicsAgent/0.1"
        self.timeout = timeout

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
                    browser.close()

                if BeautifulSoup is not None:
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                else:
                    text = content

                snippet = text[:2000]
                return BrowserResult(
                    url=url,
                    status_code=None,
                    title=title,
                    text_snippet=snippet,
                    html=None,
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

            return BrowserResult(
                url=url,
                status_code=status,
                title=title,
                text_snippet=snippet,
                html=None,
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
    Very small Python code execution sandbox.

    This is not a full OS sandbox. It is a controlled exec environment with:
    - Limited builtins
    - Optional data inputs
    - Captured stdout and stderr

    Intended use:
    - Quick math checks
    - Data frame transforms
    - Small simulations
    """

    def __init__(self, max_lines: int = 2000) -> None:
        self.max_lines = max_lines

    def run_python(
        self,
        code: str,
        *,
        extra_globals: Optional[Dict[str, Any]] = None,
    ) -> CodeExecutionResult:
        """Run a Python snippet and capture output."""
        # Simple length guard
        if len(code) > 20000:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                error="Code too long for sandbox",
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
        }
        if extra_globals:
            sandbox_globals.update(extra_globals)

        sandbox_locals: Dict[str, Any] = {}

        # Capture stdout and stderr
        import contextlib
        import io as _io
        import sys

        stdout_buffer = _io.StringIO()
        stderr_buffer = _io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
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


class DataPipelines:
    """
    Data loading helpers for CSV, Excel, and simple SQL.

    These are used for:
    - Loading experiment datasets
    - Feeding time series into RYE analysis
    - Pulling lab or biomarker tables
    """

    def __init__(self) -> None:
        if pd is None:
            self.has_pandas = False
        else:
            self.has_pandas = True

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
            preview = df.head(10).to_dict(orient="records")  # type: ignore
            return DataLoadResult(
                name=name,
                rows=int(df.shape[0]),
                cols=int(df.shape[1]),
                preview=preview,
                error=None,
            )
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
            preview = df.head(10).to_dict(orient="records")  # type: ignore
            return DataLoadResult(
                name=name,
                rows=int(df.shape[0]),
                cols=int(df.shape[1]),
                preview=preview,
                error=None,
            )
        except Exception as e:
            return DataLoadResult(
                name=name,
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


# ----------------------------------------------------------
# Toolbelt facade for CoreAgent
# ----------------------------------------------------------

class Toolbelt:
    """
    Aggregates all tools so CoreAgent can receive a single object.

    Example usage inside CoreAgent:

        tool_usage = ToolUsage()
        page = self.tools.browser.fetch_page(url)
        tool_usage.web_calls += 1

        code_res = self.tools.sandbox.run_python(code)
        tool_usage.code_execs += 1

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

    def new_usage_tracker(self) -> ToolUsage:
        """Return a fresh ToolUsage object for one cycle."""
        return ToolUsage()
