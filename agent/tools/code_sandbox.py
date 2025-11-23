import traceback
from typing import Any, Dict

class CodeSandbox:
    """
    Safe code execution sandbox for:
    - math expressions
    - controlled python execution
    """

    SAFE_GLOBALS = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "float": float,
            "int": int,
            "pow": pow,
        }
    }

    def eval_math(self, expr: str) -> Dict[str, Any]:
        try:
            val = eval(expr, self.SAFE_GLOBALS, {})
            return {"ok": True, "result": val}
        except Exception:
            return {"ok": False, "error": traceback.format_exc()}

    def exec_python(self, code: str) -> Dict[str, Any]:
        local_env: Dict[str, Any] = {}
        try:
            exec(code, self.SAFE_GLOBALS, local_env)
            return {"ok": True, "locals": local_env}
        except Exception:
            return {"ok": False, "error": traceback.format_exc()}
