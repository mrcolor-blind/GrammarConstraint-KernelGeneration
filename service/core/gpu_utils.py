"""
GPU Validation helpers — extract parameters and shapes from source code.
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_wrapper_function(source_code: str, original_function_name: str):
    """
    Find the wrapper function in generated code (not the @triton.jit kernel).

    Strategy:
      1. Find a function with the same name as the original function.
      2. If not found, find the first function WITHOUT a @triton.jit decorator.
      3. Fallback: first function in the AST.

    Returns (function_name, param_names, input_shapes_from_signature) or (None, [], {}).
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None, [], {}

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            has_triton_jit = any(
                (isinstance(d, ast.Name) and d.id == "triton")
                or (isinstance(d, ast.Attribute) and d.attr == "jit")
                or (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "jit")
                for d in node.decorator_list
            )
            functions.append((node.name, node, has_triton_jit))

    if not functions:
        return None, [], {}

    # Strategy 1: exact name match
    for name, node, _ in functions:
        if name == original_function_name:
            params = [arg.arg for arg in node.args.args]
            return name, params, {}

    # Strategy 2: first function without @triton.jit
    for name, node, has_triton in functions:
        if not has_triton:
            params = [arg.arg for arg in node.args.args]
            return name, params, {}

    # Strategy 3: first function (fallback)
    name, node, _ = functions[0]
    params = [arg.arg for arg in node.args.args]
    return name, params, {}


def extract_param_names(source_code: str, target_function_name: Optional[str] = None) -> list[str]:
    """Extract parameter names from a specific or the first function definition."""
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if target_function_name is None or node.name == target_function_name:
                    return [arg.arg for arg in node.args.args]
    except SyntaxError:
        pass
    return []


def extract_input_shapes(original_source_code: str, extracted_shapes_json: Optional[str] = None) -> dict[str, str]:
    """
    Extract input shapes.

    Priority:
      1. Use extracted_shapes_json (from call site execution) if available.
      2. Fallback to @in annotations from comments.
      3. Fallback to empty dict.
    """
    shapes: dict[str, str] = {}

    # Priority 1: shapes extracted from call site execution
    if extracted_shapes_json:
        try:
            extracted = json.loads(extracted_shapes_json)
            for name, info in extracted.items():
                if isinstance(info, dict):
                    if "shape" in info:
                        shape_tuple = info["shape"]
                        if isinstance(shape_tuple, list):
                            shapes[name] = str(tuple(shape_tuple))
                        else:
                            shapes[name] = str(shape_tuple)
                    elif "value" in info:
                        shapes[name] = f"__scalar__:{info['value']}"
                else:
                    shapes[name] = str(info)
            if shapes:
                return shapes
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse extracted_shapes_json: {e}")

    # Priority 2: @in annotations from comments
    for line in original_source_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("# @in") or stripped.startswith("#@in"):
            cleaned = re.sub(r"^#\s*@in\s+", "", stripped)
            if ":" in cleaned:
                name_part, shape_part = cleaned.split(":", 1)
                shapes[name_part.strip()] = shape_part.strip()

    return shapes


def build_gpu_validation_payload(
    job_id: str,
    generated_code: str,
    original_source_code: str,
    dims: Optional[dict] = None,
    extracted_shapes_json: Optional[str] = None,
) -> dict:
    """
    Build the JSON payload needed by service/modal_gpu_validator.py.
    """
    original_function_name = _extract_function_name(original_source_code)
    function_name, param_names, _ = _find_wrapper_function(
        generated_code, original_function_name
    )

    if not function_name:
        function_name = _extract_function_name(generated_code)
        param_names = extract_param_names(generated_code, function_name)

    input_shapes = extract_input_shapes(original_source_code, extracted_shapes_json)

    # Fallback: if no shapes found, try to infer from generated code signatures
    if not input_shapes and param_names:
        input_shapes = {name: "(64,)" for name in param_names}

    dims_str = ",".join(f"{k}={v}" for k, v in (dims or {}).items())

    return {
        "job_id": job_id,
        "generated_code": generated_code,
        "function_name": function_name,
        "param_names": param_names,
        "input_shapes": input_shapes,
        "concrete_dims_str": dims_str,
    }


def _extract_function_name(source_code: str) -> str:
    import re
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", source_code)
    return m.group(1) if m else "unknown"
