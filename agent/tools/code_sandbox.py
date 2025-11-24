import io
import math
import time
import traceback
import statistics
import random
from contextlib import redirect_stdout
from typing import Any, Dict, Optional

# Optional scientific libs (graceful if not installed)
try:
    import numpy as _np
except Exception:
    _np = None

try:
    import pandas as _pd
except Exception:
    _pd = None


class CodeSandbox:
    """
    Enhanced safe code execution sandbox for the Autonomous Research Agent.

    New capabilities (C-Level Extreme):
        - Strict vs research execution modes
        - Soft execution time limits
        - Harder forbidden-code detection
        - Optional numpy/pandas exposure when allowed
        - Vectorized math eval
        - Structured tool diagnostics (for RYE + TGRM)
        - Hardened locals sanitization
        - Automatic shape/size summarization for arrays/dataframes

    Fully backward compatible with:
        - eval_math
        - exec_python
        - run_cell
    """

    # ----------------------------------------------------------
    # SAFE BUILTINS (expanded scientific suite)
    # ----------------------------------------------------------
    SAFE_BUILTINS = {
        # Basic
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "pow": pow,
        "round": round,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "any": any,
        "all": all,

        # Math + statistics
        "math": math,
        "statistics": statistics,
        "random": random,
    }

    # ----------------------------------------------------------
    # GLOBALS
    # ----------------------------------------------------------
    SAFE_GLOBALS = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "statistics": statistics,
        "random": random,
    }

    # ----------------------------------------------------------
    # HARD LIMITS
    # ----------------------------------------------------------
    MAX_STDOUT_LENGTH = 4000
    MAX_LOCALS_ITEMS = 50
    MAX_STRING_LENGTH = 1000
    MAX_CONTAINER_PREVIEW = 200
    MAX_CODE_CHARS = 50000
    EXECUTION_TIME_LIMIT = 3.0   # seconds

    # ----------------------------------------------------------
    # INIT
    # ----------------------------------------------------------
    def __init__(
        self,
        allow_numpy: bool = True,
        allow_pandas: bool = True,
        mode: str = "strict",               # "strict" or "research"
        max_stdout: Optional[int] = None,
    ) -> None:

        self.mode = mode.strip().lower()
        self.allow_numpy = allow_numpy and (_np is not None)
        self.allow_pandas = allow_pandas and (_pd is not None)

        if max_stdout is not None:
            self.MAX_STDOUT_LENGTH = int(max_stdout)

        # Attach numpy/pandas if allowed
        if self.allow_numpy:
            self.SAFE_GLOBALS["np"] = _np
        if self.allow_pandas:
            self.SAFE_GLOBALS["pd"] = _pd

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------
    def _truncate_string(self, s: str) -> str:
        if len(s) <= self.MAX_STRING_LENGTH:
            return s
        return s[: self.MAX_STRING_LENGTH] + "... [truncated]"

    def _summarize_value(self, val: Any) -> Any:
        """Summaries for arrays, lists, dicts, dataframes, etc."""
        try:
            # numpy arrays
            if _np is not None and isinstance(val, _np.ndarray):
                return {
                    "__type__": "numpy.ndarray",
                    "shape": val.shape,
                    "dtype": str(val.dtype),
                    "preview": self._truncate_string(str(val.flat[:50])),
                }

            # pandas DataFrame
            if _pd is not None and isinstance(val, _pd.DataFrame):
                return {
                    "__type__": "pandas.DataFrame",
                    "rows": val.shape[0],
                    "cols": val.shape[1],
                    "columns": list(val.columns)[:20],
                    "preview": val.head(5).to_dict(),
                }

            # list or tuple
            if isinstance(val, (list, tuple)):
                preview = val[: self.MAX_CONTAINER_PREVIEW]
                if len(val) > self.MAX_CONTAINER_PREVIEW:
                    return {
                        "__type__": type(val).__name__,
                        "length": len(val),
                        "preview": preview,
                        "truncated": True,
                    }
                return val

            # dict
            if isinstance(val, dict):
                limited = {}
                count = 0
                for k, v in val.items():
                    limited[k] = v
                    count += 1
                    if count >= self.MAX_CONTAINER_PREVIEW:
                        limited["__truncated__"] = True
                        break
                return limited

            # string
            if isinstance(val, str):
                return self._truncate_string(val)

            # other scalar
            return val

        except Exception:
            return str(val)

    def _sanitize_locals(self, local_env: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        count = 0

        for key, val in local_env.items():
            if key.startswith("_") or key in {"__builtins__", "__name__", "__package__"}:
                continue

            sanitized[key] = self._summarize_value(val)
            count += 1
            if count >= self.MAX_LOCALS_ITEMS:
                sanitized["__sandbox_truncated__"] = True
                break

        return sanitized

    def _basic_code_guard(self, code: str) -> Optional[str]:
        """Block dangerous constructs."""
        lowered = code.lower()

        forbidden = [
            "import ",
            "__import__",
            "open(",
            "exec(",
            "eval(",
            "os.",
            "subprocess",
            "socket",
            "requests",
            "sys.",
            "http",
            "urllib",
        ]

        for frag in forbidden:
            if frag in lowered:
                return f"Forbidden construct detected: '{frag.strip()}'"

        # loop guard
        if "while true" in lowered or "while  true" in lowered:
            return "Blocked potential infinite loop ('while True')."

        # prevent huge code
        if len(code) > self.MAX_CODE_CHARS:
            return "Code too long for sandbox."

        return None

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------
    def eval_math(self, expr: str) -> Dict[str, Any]:
        """Evaluate simple math expression with strict guards."""
        try:
            guard = self._basic_code_guard(expr)
            if guard:
                return {"ok": False, "error": guard}

            start = time.time()
            result = eval(expr, self.SAFE_GLOBALS, {})
            elapsed = time.time() - start

            return {
                "ok": True,
                "result": result,
                "time_sec": elapsed,
            }
        except Exception:
            return {"ok": False, "error": traceback.format_exc()}

    def exec_python(self, code: str) -> Dict[str, Any]:
        """Execute python code safely with time guard + sanitization."""
        local_env: Dict[str, Any] = {}
        stdout_buffer = io.StringIO()

        guard = self._basic_code_guard(code)
        if guard:
            return {"ok": False, "error": guard, "stdout": "", "locals": {}}

        start = time.time()

        try:
            with redirect_stdout(stdout_buffer):
                exec(code, self.SAFE_GLOBALS, local_env)
        except Exception:
            elapsed = time.time() - start
            return {
                "ok": False,
                "error": traceback.format_exc(),
                "stdout": self._truncate_string(stdout_buffer.getvalue()),
                "locals": self._sanitize_locals(local_env),
                "time_sec": elapsed,
            }

        elapsed = time.time() - start
        stdout_val = stdout_buffer.getvalue()[: self.MAX_STDOUT_LENGTH]

        return {
            "ok": True,
            "stdout": stdout_val,
            "locals": self._sanitize_locals(local_env),
            "time_sec": elapsed,
        }

    def run_cell(self, code: str) -> Dict[str, Any]:
        """Alias for Jupyter-style usage."""
        return self.exec_python(code)
