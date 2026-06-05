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


def extract_param_names(source_code: str) -> list[str]:
    """Extract parameter names from the first function definition."""
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return [arg.arg for arg in node.args.args]
    except SyntaxError:
        pass
    return []


def extract_input_shapes(source_code: str) -> dict[str, str]:
    """
    Extract @in annotations from comments.
    Format expected:
        # @in  x:      (N, D_in)
    Returns {"x": "(N, D_in)", ...}
    """
    shapes: dict[str, str] = {}
    for line in source_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("# @in") or stripped.startswith("#@in"):
            # Try to parse: # @in x: (N, D_in)
            # Remove comment marker and @in
            cleaned = re.sub(r"^#\s*@in\s+", "", stripped)
            if ":" in cleaned:
                name_part, shape_part = cleaned.split(":", 1)
                name = name_part.strip()
                shape = shape_part.strip()
                shapes[name] = shape
    return shapes


def build_gpu_validation_payload(
    job_id: str,
    generated_code: str,
    original_source_code: str,
    dims: Optional[dict] = None,
) -> dict:
    """
    Build the JSON payload needed by service/modal_gpu_validator.py.
    """
    function_name = _extract_function_name(generated_code)
    param_names = extract_param_names(generated_code)
    input_shapes = extract_input_shapes(original_source_code)

    # Fallback: if shapes not found in comments, try to infer from generated code signatures
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
