"""
Modal GPU job for comparing a generated Triton kernel against the user's original PyTorch code.

Performs:
1. Compilation check of both modules
2. Numerical accuracy comparison (same inputs, same outputs?)
3. Timing comparison (if accurate)
4. Returns whether to suggest replacement
"""

import ast
import importlib
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import re

import torch

sys.path.append("/root/project")

from backends.modal.app import benchmark_app, volume

DATA_DIR = "/data"

# Default concrete values for common symbolic dimensions
_DEFAULT_DIM_VALUES = {
    "N": 128,
    "B": 4,
    "S": 128,
    "D": 256,
    "D_in": 256,
    "D_out": 512,
    "H": 32,
    "W": 32,
    "C": 3,
    "C_out": 16,
    "K": 3,
    "H_out": 32,
    "W_out": 32,
}

_RE_IN = re.compile(r"#\s*@in\s+(\w+)\s*:\s*(.+)")


def _parse_input_shapes(source_code: str) -> dict[str, str]:
    """Extract @in shape annotations from source code."""
    shapes = {}
    for line in source_code.splitlines():
        m = _RE_IN.match(line.strip())
        if m:
            shapes[m.group(1).strip()] = m.group(2).strip()
    return shapes


def _parse_function_name(source_code: str) -> str:
    """Extract the first function name from source code."""
    tree = ast.parse(source_code)
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            return stmt.name
    raise ValueError("No function definition found in source code.")


def _parse_param_names(source_code: str) -> list[str]:
    """Extract parameter names from the first function."""
    tree = ast.parse(source_code)
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            return [arg.arg for arg in stmt.args.args]
    return []


def _shape_to_tuple(shape_str: str) -> tuple:
    """Parse a shape string like '(N, D_in)' into a tuple."""
    s = shape_str.strip()
    if s.lower() == "scalar":
        return ()
    if s.lower() == "none":
        return None
    m = re.match(r"\((.*)\)", s)
    if m:
        inner = m.group(1).strip()
        if not inner:
            return ()
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return tuple(parts)
    return (s,)


def _substitute_shape(shape_tuple: tuple, concrete_dims: dict[str, int]) -> tuple:
    """Replace symbolic dimension names with concrete integer values."""
    if shape_tuple is None:
        return None
    result = []
    for dim in shape_tuple:
        if dim == "*":
            result.append(concrete_dims.get("B", 4))
        elif dim in concrete_dims:
            result.append(concrete_dims[dim])
        elif dim.isdigit():
            result.append(int(dim))
        else:
            result.append(_DEFAULT_DIM_VALUES.get(dim, 64))
    return tuple(result)


def _generate_inputs(
    param_names: list[str],
    shapes: dict[str, str],
    concrete_dims: dict[str, int],
    dtype=torch.float32,
    device="cuda",
):
    """Generate random tensors matching the annotated shapes.

    Scalar parameters (shape_str starting with '__scalar__:') are returned
    as Python floats/ints rather than tensors, preventing shape mismatches
    when passing alpha-like arguments to both the reference and generated fns.
    """
    inputs = {}
    for name in param_names:
        shape_str = shapes.get(name)
        if shape_str is None:
            raise ValueError(f"No @in annotation found for parameter '{name}'")

        # Scalar parameter stored as "__scalar__:<value>"
        if isinstance(shape_str, str) and shape_str.startswith("__scalar__:"):
            raw = shape_str[len("__scalar__:"):]
            try:
                val = float(raw)
                inputs[name] = int(val) if val == int(val) else val
            except (ValueError, TypeError):
                inputs[name] = raw
            continue

        shape_tuple = _shape_to_tuple(shape_str)
        if shape_tuple is None:
            inputs[name] = None
            continue
        concrete_shape = _substitute_shape(shape_tuple, concrete_dims)
        inputs[name] = torch.randn(concrete_shape, dtype=dtype, device=device)
    return inputs


def _load_module_from_string(module_name: str, code: str, write_dir: Path):
    """Write code to a real .py file, add to sys.path, and import normally."""
    module_path = write_dir / f"{module_name}.py"
    module_path.write_text(code, encoding="utf-8")
    
    str_dir = str(write_dir)
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)
    
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    return module


def _compile_and_extract(module, function_name: str):
    """Check if module has the function and return it."""
    if not hasattr(module, function_name):
        return None, f"Function '{function_name}' not found in module."
    return getattr(module, function_name), None


def _run_accuracy_check(
    ref_fn,
    gen_fn,
    input_args: list,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """
    Run both functions and compare outputs numerically.
    Returns (pass, max_diff, error_msg).
    """
    try:
        with torch.no_grad():
            ref_out = ref_fn(*input_args)
            gen_out = gen_fn(*input_args)
    except Exception as e:
        return False, None, f"Execution failed: {type(e).__name__}: {e}"

    # Handle None
    if ref_out is None or gen_out is None:
        if ref_out is None and gen_out is None:
            return True, 0.0, None
        return False, None, "Output mismatch: one returned None, the other did not."

    # Handle tuple outputs
    if isinstance(ref_out, tuple):
        if not isinstance(gen_out, tuple) or len(ref_out) != len(gen_out):
            return False, None, "Output structure mismatch: tuple length differs."
        diffs = []
        for ro, go in zip(ref_out, gen_out):
            if not isinstance(ro, torch.Tensor) or not isinstance(go, torch.Tensor):
                return False, None, f"Output type mismatch: {type(ro)} vs {type(go)}"
            diffs.append(torch.max(torch.abs(ro - go)).item())
        max_diff = max(diffs)
        all_close = all(
            torch.allclose(ro, go, rtol=rtol, atol=atol)
            for ro, go in zip(ref_out, gen_out)
        )
    else:
        if not isinstance(ref_out, torch.Tensor) or not isinstance(gen_out, torch.Tensor):
            return False, None, f"Output type mismatch: {type(ref_out)} vs {type(gen_out)}"
        max_diff = torch.max(torch.abs(ref_out - gen_out)).item()
        all_close = torch.allclose(ref_out, gen_out, rtol=rtol, atol=atol)

    if not all_close:
        return False, max_diff, f"Numerical mismatch (max_diff={max_diff:.6e})"

    return True, max_diff, None


def _run_timing(
    ref_fn,
    gen_fn,
    input_args: list,
    device: str,
    warmup_runs: int = 3,
    timed_runs: int = 10,
):
    """
    Measure timing of both functions.
    Returns (ref_time_ms, gen_time_ms, speedup).
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = ref_fn(*input_args)
        _ = gen_fn(*input_args)
    if device == "cuda":
        torch.cuda.synchronize()

    # Reference timing
    start = time.perf_counter()
    for _ in range(timed_runs):
        _ = ref_fn(*input_args)
    if device == "cuda":
        torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / timed_runs

    # Generated timing
    start = time.perf_counter()
    for _ in range(timed_runs):
        _ = gen_fn(*input_args)
    if device == "cuda":
        torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start) / timed_runs

    speedup = ref_time / gen_time if gen_time > 0 else float("inf")
    return ref_time * 1000, gen_time * 1000, speedup


@benchmark_app.function(
    gpu="T4",
    timeout=60 * 10,
    volumes={DATA_DIR: volume},
)
def compare_with_user(
    original_code: str,
    generated_code: str,
    concrete_dims_str: str = "",
    extracted_shapes_json: str = "",
    speedup_threshold: float = 1.1,
    warmup_runs: int = 3,
    timed_runs: int = 10,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """
    Compare a generated Triton kernel against the user's original PyTorch function.
    
    Returns:
    {
        "compilation_pass": bool,
        "accuracy_pass": bool,
        "max_diff": float | None,
        "speedup": float | None,
        "ref_time_ms": float | None,
        "gen_time_ms": float | None,
        "suggest_replacement": bool,
        "reason": str,
        "errors": list[str],
        "device": str,
        "concrete_dims": dict,
    }
    """
    # Parse concrete dims
    concrete_dims = {}
    if concrete_dims_str:
        for pair in concrete_dims_str.split(","):
            if "=" in pair:
                k, v = pair.split("=")
                concrete_dims[k.strip()] = int(v.strip())
    
    # Merge with defaults
    dims = dict(_DEFAULT_DIM_VALUES)
    dims.update(concrete_dims)
    concrete_dims = dims

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logs: list[str] = []
    errors = []

    def _log(msg: str):
        logs.append(msg)
        print(msg, flush=True)

    _log(f"[COMPARE] Starting on device={device}")
    _log(f"[COMPARE] concrete_dims={concrete_dims}")
    
    # --- Step 1: Parse metadata from original code ---
    try:
        function_name = _parse_function_name(original_code)
        param_names = _parse_param_names(original_code)
        
        # Use extracted shapes if available, otherwise fall back to @in annotations
        if extracted_shapes_json:
            import json
            extracted_shapes = json.loads(extracted_shapes_json)
            # Convert extracted shapes to the format expected by _generate_inputs
            input_shapes = {}
            for name, info in extracted_shapes.items():
                if "shape" in info:
                    shape_tuple = tuple(info["shape"])
                    input_shapes[name] = str(shape_tuple) if len(shape_tuple) > 1 else f"({shape_tuple[0]},)"
                elif "value" in info:
                    # Scalar parameter — prefix so _generate_inputs returns it as-is
                    input_shapes[name] = f"__scalar__:{info['value']}"
        else:
            input_shapes = _parse_input_shapes(original_code)
        _log(f"[COMPARE] Parsed function_name={function_name}, params={param_names}, shapes={input_shapes}")
    except Exception as e:
        _log(f"[COMPARE] Failed to parse original code: {e}")
        return {
            "compilation_pass": False,
            "accuracy_pass": False,
            "max_diff": None,
            "speedup": None,
            "ref_time_ms": None,
            "gen_time_ms": None,
            "suggest_replacement": False,
            "reason": f"Failed to parse original code: {e}",
            "errors": [str(e)],
            "device": device,
            "concrete_dims": concrete_dims,
            "logs": logs,
        }

    # --- Step 2: Load both modules ---
    tmp_dir = Path(DATA_DIR) / "tmp_comparison"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    ref_module_name = f"ref_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    gen_module_name = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    _log(f"[COMPARE] Loading ref_module={ref_module_name}, gen_module={gen_module_name}")

    ref_fn = None
    gen_fn = None
    
    try:
        ref_module = _load_module_from_string(ref_module_name, original_code, tmp_dir)
        ref_fn, err = _compile_and_extract(ref_module, function_name)
        if err:
            errors.append(f"Original code: {err}")
            _log(f"[COMPARE] Original code compilation error: {err}")
        else:
            _log("[COMPARE] Original code compilation PASS")
    except Exception as e:
        errors.append(f"Original code compilation failed: {type(e).__name__}: {e}")
        _log(f"[COMPARE] Original code compilation FAIL: {type(e).__name__}: {e}")

    try:
        gen_module = _load_module_from_string(gen_module_name, generated_code, tmp_dir)
        gen_fn, err = _compile_and_extract(gen_module, function_name)
        if err:
            errors.append(f"Generated code: {err}")
            _log(f"[COMPARE] Generated code compilation error: {err}")
        else:
            _log("[COMPARE] Generated code compilation PASS")
    except Exception as e:
        errors.append(f"Generated code compilation failed: {type(e).__name__}: {e}")
        _log(f"[COMPARE] Generated code compilation FAIL: {type(e).__name__}: {e}")

    if not ref_fn or not gen_fn:
        _log("[COMPARE] Compilation failed for one or both modules")
        return {
            "compilation_pass": False,
            "accuracy_pass": False,
            "max_diff": None,
            "speedup": None,
            "ref_time_ms": None,
            "gen_time_ms": None,
            "suggest_replacement": False,
            "reason": "Compilation failed for one or both modules.",
            "errors": errors,
            "device": device,
            "concrete_dims": concrete_dims,
            "logs": logs,
        }

    # --- Step 3: Generate inputs ---
    try:
        inputs = _generate_inputs(param_names, input_shapes, concrete_dims, device=device)
        input_args = [inputs[name] for name in param_names]
        shapes_str = {n: list(inputs[n].shape) if hasattr(inputs[n], 'shape') else str(inputs[n]) for n in param_names}
        _log(f"[COMPARE] Generated inputs: {shapes_str}")
    except Exception as e:
        _log(f"[COMPARE] Failed to generate inputs: {e}")
        return {
            "compilation_pass": True,
            "accuracy_pass": False,
            "max_diff": None,
            "speedup": None,
            "ref_time_ms": None,
            "gen_time_ms": None,
            "suggest_replacement": False,
            "reason": f"Failed to generate inputs: {e}",
            "errors": errors + [str(e)],
            "device": device,
            "concrete_dims": concrete_dims,
            "logs": logs,
        }

    # --- Step 4: Accuracy check ---
    accuracy_pass, max_diff, acc_error = _run_accuracy_check(
        ref_fn, gen_fn, input_args, rtol=rtol, atol=atol
    )

    if acc_error:
        errors.append(acc_error)

    _log(f"[COMPARE] Accuracy check: pass={accuracy_pass}, max_diff={max_diff}")
    if acc_error:
        _log(f"[COMPARE] Accuracy error: {acc_error}")

    if not accuracy_pass:
        return {
            "compilation_pass": True,
            "accuracy_pass": False,
            "max_diff": max_diff,
            "speedup": None,
            "ref_time_ms": None,
            "gen_time_ms": None,
            "suggest_replacement": False,
            "reason": f"Accuracy check failed: {acc_error}",
            "errors": errors,
            "device": device,
            "concrete_dims": concrete_dims,
            "logs": logs,
        }

    # --- Step 5: Timing comparison ---
    ref_time_ms, gen_time_ms, speedup = _run_timing(
        ref_fn, gen_fn, input_args, device,
        warmup_runs=warmup_runs, timed_runs=timed_runs
    )
    _log(f"[COMPARE] Timing: ref={ref_time_ms:.3f}ms, gen={gen_time_ms:.3f}ms, speedup={speedup:.2f}x")

    # --- Step 6: Decision ---
    suggest_replacement = speedup > speedup_threshold
    
    if suggest_replacement:
        reason = (
            f"Triton kernel is {speedup:.2f}x faster than the original PyTorch code "
            f"({ref_time_ms:.2f} ms vs {gen_time_ms:.2f} ms). "
            f"Accuracy verified (max_diff={max_diff:.6e})."
        )
    else:
        reason = (
            f"Triton kernel is {speedup:.2f}x (threshold: {speedup_threshold:.2f}x). "
            f"Not enough speedup to suggest replacement. "
            f"({ref_time_ms:.2f} ms vs {gen_time_ms:.2f} ms)."
        )

    _log(f"[COMPARE] Suggest replacement: {suggest_replacement}")
    _log(f"[COMPARE] Reason: {reason}")

    volume.commit()
    _log("[COMPARE] Finished successfully")

    return {
        "compilation_pass": True,
        "accuracy_pass": True,
        "max_diff": max_diff,
        "speedup": speedup,
        "ref_time_ms": ref_time_ms,
        "gen_time_ms": gen_time_ms,
        "suggest_replacement": suggest_replacement,
        "reason": reason,
        "errors": errors,
        "device": device,
        "concrete_dims": concrete_dims,
        "logs": logs,
    }
