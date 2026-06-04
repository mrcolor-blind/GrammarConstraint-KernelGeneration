"""
Translate Evaluator — numerical accuracy and speedup measurement for generated Triton kernels.
"""

import ast
import importlib.util
import re
import time
from pathlib import Path

import torch
from typing import Optional, Union


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
            # Wildcard batch: use a small default
            result.append(concrete_dims.get("B", 4))
        elif dim in concrete_dims:
            result.append(concrete_dims[dim])
        elif dim.isdigit():
            result.append(int(dim))
        else:
            # Unknown symbol: use default heuristic
            result.append(_DEFAULT_DIM_VALUES.get(dim, 64))
    return tuple(result)


def _load_module_from_path(path: Path):
    """Dynamically import a Python module from a file path.
    Injects torch and triton into the module namespace so user functions
    that rely on these imports (but don't include them in the snippet) work."""
    import torch

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["torch"] = torch
    try:
        import triton
        import triton.language as tl
        module.__dict__["triton"] = triton
        module.__dict__["tl"] = tl
    except ImportError:
        pass  # triton not installed locally; will fail at runtime if kernel is invoked
    spec.loader.exec_module(module)
    return module


def _generate_inputs(
    param_names: list[str],
    shapes: dict[str, str],
    concrete_dims: dict[str, int],
    dtype=torch.float32,
    device="cuda",
):
    """Generate random tensors matching the annotated shapes."""
    inputs = {}
    for name in param_names:
        shape_str = shapes.get(name)
        if shape_str is None:
            raise ValueError(f"No @in annotation found for parameter '{name}'")
        shape_tuple = _shape_to_tuple(shape_str)
        if shape_tuple is None:
            inputs[name] = None
            continue
        concrete_shape = _substitute_shape(shape_tuple, concrete_dims)
        inputs[name] = torch.randn(concrete_shape, dtype=dtype, device=device)
    return inputs


def run_local_evaluation(
    original_path: Path,
    generated_path: Path,
    concrete_dims:Optional[ dict[str, int] ] = None,
    dtype=torch.float32,
    device: str = "cuda",
    warmup_runs: int = 3,
    timed_runs: int = 10,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """
    Evaluate a generated Triton kernel against the original PyTorch function.
    Returns a dict with accuracy and timing results.
    """
    if concrete_dims is None:
        concrete_dims = {}

    # Merge with defaults for missing keys
    dims = dict(_DEFAULT_DIM_VALUES)
    dims.update(concrete_dims)

    original_code = original_path.read_text(encoding="utf-8")
    generated_code = generated_path.read_text(encoding="utf-8")

    # Extract function name and parameter names from original source
    tree = ast.parse(original_code)
    func_def = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            func_def = stmt
            break
    if func_def is None:
        raise ValueError("No function definition found in original source.")

    func_name = func_def.name
    param_names = [arg.arg for arg in func_def.args.args]

    # Parse input shapes
    input_shapes = _parse_input_shapes(original_code)

    # Check CUDA availability
    if not torch.cuda.is_available():
        device = "cpu"

    # Load modules
    original_module = _load_module_from_path(original_path)
    generated_module = _load_module_from_path(generated_path)

    original_fn = getattr(original_module, func_name, None)
    generated_fn = getattr(generated_module, func_name, None)

    if original_fn is None:
        raise ValueError(f"Function '{func_name}' not found in original module.")
    if generated_fn is None:
        raise ValueError(f"Function '{func_name}' not found in generated module.")

    # Generate inputs
    inputs = _generate_inputs(param_names, input_shapes, dims, dtype=dtype, device=device)
    input_args = [inputs[name] for name in param_names]

    # --- Accuracy check ---
    try:
        with torch.no_grad():
            ref_out = original_fn(*input_args)
            gen_out = generated_fn(*input_args)
    except Exception as e:
        return {
            "accuracy_pass": False,
            "error": f"Execution failed: {e}",
            "max_diff": None,
            "speedup": None,
            "ref_time_ms": None,
            "gen_time_ms": None,
        }

    # Handle tuple outputs
    if isinstance(ref_out, tuple):
        if not isinstance(gen_out, tuple) or len(ref_out) != len(gen_out):
            return {
                "accuracy_pass": False,
                "error": "Output structure mismatch: original returns tuple, generated does not.",
                "max_diff": None,
                "speedup": None,
            }
        diffs = []
        for ro, go in zip(ref_out, gen_out):
            diffs.append(torch.max(torch.abs(ro - go)).item())
        max_diff = max(diffs)
        all_close = all(
            torch.allclose(ro, go, rtol=rtol, atol=atol)
            for ro, go in zip(ref_out, gen_out)
        )
    else:
        max_diff = torch.max(torch.abs(ref_out - gen_out)).item()
        all_close = torch.allclose(ref_out, gen_out, rtol=rtol, atol=atol)

    if not all_close:
        return {
            "accuracy_pass": False,
            "error": f"Numerical mismatch (max_diff={max_diff:.6e})",
            "max_diff": max_diff,
            "speedup": None,
            "ref_time_ms": None,
            "gen_time_ms": None,
        }

    # --- Timing ---
    # Warmup
    for _ in range(warmup_runs):
        _ = original_fn(*input_args)
        _ = generated_fn(*input_args)
    if device == "cuda":
        torch.cuda.synchronize()

    # Reference timing
    start = time.perf_counter()
    for _ in range(timed_runs):
        _ = original_fn(*input_args)
    if device == "cuda":
        torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / timed_runs

    # Generated timing
    start = time.perf_counter()
    for _ in range(timed_runs):
        _ = generated_fn(*input_args)
    if device == "cuda":
        torch.cuda.synchronize()
    gen_time = (time.perf_counter() - start) / timed_runs

    speedup = ref_time / gen_time if gen_time > 0 else float("inf")

    return {
        "accuracy_pass": True,
        "max_diff": max_diff,
        "speedup": speedup,
        "ref_time_ms": ref_time * 1000,
        "gen_time_ms": gen_time * 1000,
        "device": str(device),
        "concrete_dims": dims,
    }
