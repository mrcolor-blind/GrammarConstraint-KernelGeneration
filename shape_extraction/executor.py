"""
Shape Extraction — runs a call site in a sandboxed subprocess to capture
the exact shapes, dtypes, and devices of tensor arguments passed to a function.
"""

import ast
import json
import multiprocessing
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

import torch

_DEFAULT_TIMEOUT = 10  # seconds


def _extract_function_name(source_code: str) -> str:
    """Extract the first function name from source code."""
    tree = ast.parse(source_code)
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            return stmt.name
    raise ValueError("No function definition found in source code.")


def _write_modules_to_temp(
    function_code: str,
    call_site_code: str,
    function_name: str,
    tmp_dir: Path,
) -> tuple[Path, Path]:
    """Write the function and call site to temp files."""
    func_path = tmp_dir / "user_function.py"
    func_path.write_text(function_code, encoding="utf-8")

    # Wrap the call site to intercept the function
    # We need to import the user's function and monkey-patch it
    # Use a placeholder for the call site code to avoid indentation issues
    call_site_placeholder = "__CALL_SITE_CODE_PLACEHOLDER__"
    
    tracer_code = textwrap.dedent(f"""\
import sys
sys.path.insert(0, {repr(str(tmp_dir))})

import torch
import json
import traceback

# Import the user's function module
import user_function

# --- Tracer setup ---
_captured = {{}}
_original = getattr(user_function, {repr(function_name)}, None)

if _original is None:
    raise NameError(f"Function '{function_name}' not found in user_function module")

def _traced_function(*args, **kwargs):
    # Capture tensor metadata
    shapes = {{}}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            shapes[f"arg_{{i}}"] = {{
                "shape": list(arg.shape),
                "dtype": str(arg.dtype).replace("torch.", ""),
                "device": str(arg.device),
            }}
        elif isinstance(arg, (int, float, bool)):
            shapes[f"arg_{{i}}"] = {{
                "value": arg,
                "type": type(arg).__name__,
            }}
    
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            shapes[k] = {{
                "shape": list(v.shape),
                "dtype": str(v.dtype).replace("torch.", ""),
                "device": str(v.device),
            }}
        elif isinstance(v, (int, float, bool, str)):
            shapes[k] = {{
                "value": v,
                "type": type(v).__name__,
            }}
    
    _captured["shapes"] = shapes
    _captured["called"] = True
    
    # Call the original
    return _original(*args, **kwargs)

# Monkey-patch
setattr(user_function, {repr(function_name)}, _traced_function)

# Also inject into the global namespace of this module (__main__) so the call site can use it
globals()[{repr(function_name)}] = _traced_function

# --- Execute call site ---
try:
    {call_site_placeholder}
except Exception as e:
    _captured["error"] = f"{{type(e).__name__}}: {{e}}"
    _captured["traceback"] = traceback.format_exc()

# --- Output result ---
if not _captured.get("called"):
    _captured["error"] = "Function '{function_name}' was never called in the call site."

print("\\n---SHAPE_EXTRACTION_RESULT---")
print(json.dumps(_captured, indent=2))
print("---END_SHAPE_EXTRACTION_RESULT---")
""")

    # The call site code is inserted into the try block of the template.
    # The template line is `    {call_site_placeholder}` (4 spaces before placeholder).
    # The call_site_code should have 4 spaces on each line to match the try block.
    # But when we replace the placeholder, the first line gets the template's 4 spaces
    # PLUS the call_site_code's 4 spaces = 8 spaces. To fix this, we strip the template's
    # leading spaces and indent the call_site_code ourselves.
    
    # First, indent the call site code with 4 spaces
    lines = call_site_code.splitlines(keepends=True)
    indented_lines = []
    for line in lines:
        if line.strip():
            indented_lines.append("    " + line)
        else:
            indented_lines.append(line)
    indented_call_site = "".join(indented_lines)
    
    # Now, in the template, replace the placeholder along with its leading 4 spaces
    # So we replace `    {call_site_placeholder}` with the indented call site
    tracer_code = tracer_code.replace(
        "    " + call_site_placeholder,
        indented_call_site,
    )

    call_path = tmp_dir / "call_site_runner.py"
    call_path.write_text(tracer_code, encoding="utf-8")

    return func_path, call_path


def _run_in_subprocess(
    function_code: str,
    call_site_code: str,
    function_name: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    """
    Run the call site in a sandboxed subprocess.
    Returns the captured shapes or an error dict.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        func_path, call_path = _write_modules_to_temp(
            function_code, call_site_code, function_name, tmp_path
        )

        # Use a subprocess to run the script
        import subprocess

        env = os.environ.copy()
        # Restrict PYTHONPATH to prevent importing arbitrary modules
        env["PYTHONPATH"] = str(tmp_path)

        try:
            result = subprocess.run(
                [sys.executable, str(call_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(tmp_path),
            )
        except subprocess.TimeoutExpired:
            return {
                "error": f"Call site execution timed out after {timeout} seconds.",
                "called": False,
                "shapes": {},
            }
        except Exception as e:
            return {
                "error": f"Subprocess failed: {type(e).__name__}: {e}",
                "called": False,
                "shapes": {},
            }

        # Parse the result from stdout
        stdout = result.stdout
        marker_start = "---SHAPE_EXTRACTION_RESULT---"
        marker_end = "---END_SHAPE_EXTRACTION_RESULT---"

        start_idx = stdout.find(marker_start)
        end_idx = stdout.find(marker_end)

        if start_idx == -1 or end_idx == -1:
            return {
                "error": (
                    "Could not find shape extraction result in subprocess output.\n"
                    f"stdout: {stdout[:500]}\n"
                    f"stderr: {result.stderr[:500]}"
                ),
                "called": False,
                "shapes": {},
            }

        json_str = stdout[start_idx + len(marker_start) : end_idx].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse shape extraction result: {e}",
                "called": False,
                "shapes": {},
            }


def _extract_param_names(function_code: str, function_name: str) -> list[str]:
    """Extract parameter names from the function definition."""
    tree = ast.parse(function_code)
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == function_name:
            return [arg.arg for arg in stmt.args.args]
    return []


def extract_shapes(
    function_code: str,
    call_site_code: str,
    function_name: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Extract the exact shapes of tensor arguments from a call site.
    
    Args:
        function_code: The Python code containing the function definition.
        call_site_code: The Python code that calls the function.
        function_name: The name of the function to intercept. If None, auto-detected.
        timeout: Maximum seconds to wait for the call site to run.
    
    Returns:
        dict with keys:
        - "shapes": dict mapping param_name -> {"shape": [...], "dtype": "", "device": ""}
        - "called": bool
        - "error": str or None
    """
    if function_name is None:
        function_name = _extract_function_name(function_code)

    param_names = _extract_param_names(function_code, function_name)
    
    result = _run_in_subprocess(
        function_code=function_code,
        call_site_code=call_site_code,
        function_name=function_name,
        timeout=timeout,
    )
    
    # Map positional arg_N to parameter names
    if "shapes" in result and param_names:
        mapped_shapes = {}
        for key, value in result["shapes"].items():
            if key.startswith("arg_"):
                idx = int(key.split("_")[1])
                if idx < len(param_names):
                    mapped_shapes[param_names[idx]] = value
            else:
                # Keyword arg, keep as is
                mapped_shapes[key] = value
        result["shapes"] = mapped_shapes
    
    return result


def format_shapes_for_prompt(shapes: dict) -> dict[str, str]:
    """
    Convert the raw extracted shapes into a format suitable for the prompt.
    
    Returns:
        dict mapping parameter name -> shape string like "(1024, 768)"
    """
    result = {}
    for param_name, info in shapes.items():
        if "shape" in info:
            shape_tuple = tuple(info["shape"])
            result[param_name] = str(shape_tuple) if len(shape_tuple) > 1 else f"({shape_tuple[0]},)"
        elif "value" in info:
            # Scalar / non-tensor parameter
            result[param_name] = str(info["value"])
    return result


def format_shapes_for_comparison(shapes: dict) -> dict[str, tuple]:
    """
    Convert the raw extracted shapes into concrete tuples for input generation.
    
    Returns:
        dict mapping parameter name -> (1024, 768) or scalar value
    """
    result = {}
    for param_name, info in shapes.items():
        if "shape" in info:
            result[param_name] = tuple(info["shape"])
        elif "value" in info:
            result[param_name] = info["value"]
    return result


if __name__ == "__main__":
    # Simple self-test
    function_code = """
import torch

def add(input, other, alpha=1):
    return input + alpha * other
"""

    call_site_code = """
import torch
from user_function import add

x = torch.randn(1024, 768, device='cuda')
y = torch.randn(1024, 768, device='cuda')
result = add(x, y, alpha=2.0)
"""

    result = extract_shapes(function_code, call_site_code, "add")
    print("Result:", json.dumps(result, indent=2))
