from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


def execute_python_code(context_root: Path, code: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
    """Execute Python code in a subprocess with context_root as working directory.

    Uses subprocess instead of multiprocessing to avoid spawn issues when running
    from scripts or interactive environments.
    """
    resolved_context_root = context_root.resolve()

    # Wrap user code with chdir so file paths work correctly
    wrapped = textwrap.dedent(f"""
import os, sys
os.chdir({repr(str(resolved_context_root))})
{code}
""")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(wrapped)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(resolved_context_root),
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "stderr": result.stderr,
            "error": result.stderr if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "stderr": "",
            "error": f"Python execution timed out after {timeout_seconds} seconds.",
        }
    except Exception as exc:
        return {
            "success": False,
            "output": "",
            "stderr": str(exc),
            "error": str(exc),
        }
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
