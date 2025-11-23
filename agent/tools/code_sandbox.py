import io
import math
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict, Optional


class CodeSandbox:
    """
    Safe code execution sandbox for the Autonomous Research Agent.

    Capabilities:
        - Lightweight math evaluation for simple expressions
        - Controlled Python execution for small snippets
        - Captured stdout (print) output
        - Sanitized local variable output

    Design goals:
        - No file system access
        - No network access
        - No imports or dangerous builtins
        - Backward compatible with existing eval_math and exec_python calls
    """

    # Builtins that are allowed inside the sandbox
    SAFE_BUILTINS = {
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
        "math": math,
    }

    # Global namespace used for both eval and exec
    SAFE_GLOBALS = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
    }

    # Hard limits so the agent does not flood memory
    MAX_STDOUT_LENGTH = 4000
    MAX_LOCALS_ITEMS = 50
    MAX_STRING_LENGTH = 1000

    def __init__(self, max_stdout: Optional[int] = None) -> None:
        if max_stdout is not None:
            self.MAX_STDOUT_LENGTH = int(max_stdout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _truncate_string(self, s: str) -> str:
        if len(s) <= self.MAX_STRING_LENGTH:
            return s
        return s[: self.MAX_STRING_LENGTH] + "... [truncated]"

    def _sanitize_locals(self, local_env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip internal keys and truncate large string values so the agent
        can safely log and inspect results.
        """
        sanitized: Dict[str, Any] = {}
        count = 0

        for key, val in local_env.items():
            if key.startswith("_"):
                continue
            if key in {"__builtins__", "__name__", "__package__"}:
                continue

            if isinstance(val, str):
                safe_val: Any = self._truncate_string(val)
            else:
                safe_val = val

            sanitized[key] = safe_val
            count += 1
            if count >= self.MAX_LOCALS_ITEMS:
                sanitized["__sandbox_truncated__"] = True
                break

        return sanitized

    def _basic_code_guard(self, code: str) -> Optional[str]:
        """
        Simple static checks to avoid obviously dangerous patterns.
        This is not a full security model, just a first line of defense.
        Returns an error string if something looks unsafe.
        """
        lowered = code.lower()

        forbidden_fragments = [
            "import ",
            "__import__",
            "open(",
            "exec(",
            "eval(",
            "os.system",
            "subprocess",
            "socket",
            "http",
            "requests.",
            "sys.",
        ]

        for frag in forbidden_fragments:
            if frag in lowered:
                return f"Forbidden construct detected in sandbox code: {frag.strip()}"

        # Very naive loop guard
        if "while True" in lowered or "while  True" in lowered:
            return "Potential infinite loop detected (while True)."

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def eval_math(self, expr: str) -> Dict[str, Any]:
        """
        Evaluate a simple math expression in a restricted environment.

        Example:
            sandbox.eval_math("1 + 2 * 3")
        """
        try:
            # Very short guard for obviously unsafe usage
            guard_err = self._basic_code_guard(expr)
            if guard_err:
                return {"ok": False, "error": guard_err}

            val = eval(expr, self.SAFE_GLOBALS, {})
            return {"ok": True, "result": val}
        except Exception:
            return {"ok": False, "error": traceback.format_exc()}

    def exec_python(self, code: str) -> Dict[str, Any]:
        """
        Execute a small Python snippet in a restricted environment.

        Returns:
            {
                "ok": bool,
                "stdout": "captured print output",
                "locals": {name: value, ...}  (sanitized),
                "error": "traceback if ok is False",
            }
        """
        local_env: Dict[str, Any] = {}
        stdout_buffer = io.StringIO()

        # Static safety checks
        guard_err = self._basic_code_guard(code)
        if guard_err:
            return {"ok": False, "error": guard_err, "stdout": "", "locals": {}}

        try:
            with redirect_stdout(stdout_buffer):
                exec(code, self.SAFE_GLOBALS, local_env)
        except Exception:
            return {
                "ok": False,
                "error": traceback.format_exc(),
                "stdout": self._truncate_string(stdout_buffer.getvalue()),
                "locals": self._sanitize_locals(local_env),
            }

        stdout_val = stdout_buffer.getvalue()
        stdout_val = stdout_val[: self.MAX_STDOUT_LENGTH]

        return {
            "ok": True,
            "stdout": stdout_val,
            "locals": self._sanitize_locals(local_env),
        }

    # Optional convenience alias
    def run_cell(self, code: str) -> Dict[str, Any]:
        """
        Alias for exec_python to support notebook style usage inside the agent.
        """
        return self.exec_python(code)
