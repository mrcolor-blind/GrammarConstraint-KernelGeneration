"""
Validator — static validation of generated Triton code.
"""

import ast
import re

from models.domain import OperationGraph, ValidationResult


# Known hallucinated Triton APIs that models often generate
_HALLUCINATED_APIS = [
    r"triton\.lang\b",
    r"tl\.mm\s*\(",
    r"tl\.Scalar\b",
    r"tl\.Tensor\s*\(",
    r"tl\.zero\s*\(",
    r"tl\.zero_like\b",  # sometimes also wrong
]


def validate_code(
    generated_code: str,
    original_graph: OperationGraph,
) -> ValidationResult:
    """
    Run a battery of static checks on the generated Triton code.
    Returns a ValidationResult with errors and warnings.
    """
    errors = []
    warnings = []

    # --- Check 1: Python syntax ---
    try:
        compile(generated_code, "<generated>", "exec")
    except SyntaxError as e:
        errors.append(f"Python syntax error: {e.msg} (line {e.lineno})")
        # Can't continue AST-based checks if syntax is broken
        return ValidationResult(passed=False, errors=errors, warnings=warnings)

    # --- Check 2: AST-based checks ---
    try:
        tree = ast.parse(generated_code)
    except SyntaxError:
        # already caught above, but for type safety
        return ValidationResult(passed=False, errors=errors, warnings=warnings)

    wrapper_name = original_graph.function_name
    wrapper_found = False
    wrapper_params = []
    kernels_found = []
    has_triton_import = False
    has_tl_import = False
    has_triton_lang = False

    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                name = alias.asname if alias.asname else alias.name
                if name == "triton":
                    has_triton_import = True
                if name == "tl":
                    has_tl_import = True
        elif isinstance(stmt, ast.ImportFrom):
            module = stmt.module or ""
            if module == "triton" or module.startswith("triton."):
                has_triton_import = True
            if module == "triton.language":
                for alias in stmt.names:
                    if alias.asname == "tl" or alias.name == "tl":
                        has_tl_import = True

        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == wrapper_name:
                wrapper_found = True
                wrapper_params = [arg.arg for arg in stmt.args.args]
            # Check for @triton.jit decorated kernels
            for dec in stmt.decorator_list:
                dec_name = _attr_to_str(dec)
                if "triton" in dec_name and "jit" in dec_name:
                    kernels_found.append(stmt.name)

    # --- Check 3: Wrapper signature ---
    if not wrapper_found:
        errors.append(
            f"Wrapper function '{wrapper_name}' not found in generated code."
        )
    else:
        expected_params = [p.name for p in original_graph.parameters]
        if wrapper_params != expected_params:
            warnings.append(
                f"Wrapper parameter names differ: expected {expected_params}, got {wrapper_params}."
            )

    # --- Check 4: Triton imports ---
    if not has_triton_import:
        warnings.append("Missing 'import triton' in generated code.")
    if not has_tl_import:
        warnings.append("Missing 'import triton.language as tl' in generated code.")

    # --- Check 5: Hallucinated APIs (regex) ---
    for pattern in _HALLUCINATED_APIS:
        if re.search(pattern, generated_code):
            errors.append(f"Hallucinated API detected: matches pattern '{pattern}'")

    # --- Check 6: Kernel decorator ---
    if not kernels_found:
        warnings.append("No @triton.jit decorated kernels found.")

    # --- Check 7: Grid definition ---
    if "grid=" not in generated_code and "grid =" not in generated_code:
        warnings.append("No 'grid=' found in kernel launch. Ensure kernel is invoked with a grid.")

    passed = len(errors) == 0
    return ValidationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
    )


def _attr_to_str(node: ast.AST) -> str:
    """Convert an attribute chain AST node to a dotted string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attr_to_str(node.value)
        return f"{base}.{node.attr}"
    return ""
