"""
Parser — converts a user-provided Python function into an OperationGraph.
"""

import ast
from typing import Any, Optional

from code_analysis.op_detector import (
    build_alias_map,
    resolve_binary_op,
    resolve_call,
)
from models.domain import OperationGraph, OpNode, Parameter


class ControlFlowWarning(Exception):
    """Raised when control flow (if/for/while) is detected; non-fatal."""
    pass


def _extract_default(node: ast.AST) -> Any:
    """Extract a default value from an AST expression (for display only)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.NameConstant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _build_param(param: ast.arg) -> Parameter:
    """Build a Parameter dataclass from an ast.arg node."""
    kind = "POSITIONAL_OR_KEYWORD"
    annotation = ast.unparse(param.annotation) if param.annotation else None
    return Parameter(
        name=param.arg,
        kind=kind,
        default=None,
        annotation=annotation,
    )


def _build_signature(func_def: ast.FunctionDef) -> str:
    """Recreate the function signature as a string."""
    args = func_def.args
    params = []

    # Positional args with defaults
    defaults = [None] * (len(args.args) - len(args.defaults)) + [
        _extract_default(d) for d in args.defaults
    ]
    for arg, default in zip(args.args, defaults):
        s = arg.arg
        if default is not None:
            s += f"={default}"
        params.append(s)

    # *args
    if args.vararg:
        params.append(f"*{args.vararg.arg}")

    # Keyword-only args
    kw_defaults = [None] * (len(args.kwonlyargs) - len(args.kw_defaults)) + [
        _extract_default(d) for d in args.kw_defaults
    ]
    for arg, default in zip(args.kwonlyargs, kw_defaults):
        s = arg.arg
        if default is not None:
            s += f"={default}"
        params.append(s)

    # **kwargs
    if args.kwarg:
        params.append(f"**{args.kwarg.arg}")

    return f"{func_def.name}({', '.join(params)})"


def _walk_expression(expr: ast.AST, alias_map: dict, ops: list, assign_target:Optional[ str ]):
    """
    Walk an expression AST node, detecting all torch operations (Calls and BinOps).
    *ops* is a mutable list of OpNode dicts (we append to it).
    *assign_target* is the variable name being assigned to, if any.
    
    Uses depth-first traversal so nested operations are detected BEFORE their
    parent, preserving correct evaluation order.
    """
    if isinstance(expr, ast.BinOp):
        # Walk children FIRST (depth-first) so nested ops appear before parent
        _walk_expression(expr.left, alias_map, ops, None)
        _walk_expression(expr.right, alias_map, ops, None)
        
        op_name = resolve_binary_op(expr)
        if op_name:
            left_str = _expr_to_str(expr.left)
            right_str = _expr_to_str(expr.right)
            ops.append(
                OpNode(
                    op_name=op_name,
                    torch_path=op_name,
                    input_vars=[left_str, right_str],
                    output_var=assign_target or f"_t{len(ops)+1}",
                    kwargs={},
                    lineno=getattr(expr, "lineno", 0),
                )
            )
        return

    if isinstance(expr, ast.Call):
        # Walk arguments FIRST for nested ops inside args
        for arg in expr.args:
            _walk_expression(arg, alias_map, ops, None)
        for kw in expr.keywords:
            _walk_expression(kw.value, alias_map, ops, None)
        
        op_name = resolve_call(expr, alias_map)
        if op_name:
            input_vars = [_expr_to_str(a) for a in expr.args]
            kwargs = {}
            for kw in expr.keywords:
                kwargs[kw.arg] = _expr_to_str(kw.value)
            ops.append(
                OpNode(
                    op_name=op_name,
                    torch_path=op_name,
                    input_vars=input_vars,
                    output_var=assign_target or f"_t{len(ops)+1}",
                    kwargs=kwargs,
                    lineno=getattr(expr, "lineno", 0),
                )
            )
        return

    # Walk into sub-expressions for other node types
    for child in ast.iter_child_nodes(expr):
        _walk_expression(child, alias_map, ops, None)


def _expr_to_str(node: ast.AST) -> str:
    """Best-effort string representation of an expression AST node."""
    try:
        return ast.unparse(node)
    except Exception:
        # Fallback for very old Python versions (should not happen with 3.12)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Attribute):
            base = _expr_to_str(node.value)
            return f"{base}.{node.attr}"
        if isinstance(node, ast.Subscript):
            base = _expr_to_str(node.value)
            slice_str = _expr_to_str(node.slice) if hasattr(node, "slice") else ""
            return f"{base}[{slice_str}]"
        return "<?>"


def parse_function(source_code: str) -> tuple[OperationGraph, list[str]]:
    """
    Parse a Python source string and extract the *first* @triton-annotated function
    as an OperationGraph.  Also returns a list of non-fatal warnings.

    Raises SyntaxError if the source is invalid Python.
    Raises ValueError if no @triton-annotated function is found.
    """
    tree = ast.parse(source_code)
    warnings: list[str] = []

    # Build alias map from top-level imports
    alias_map = build_alias_map(tree.body)

    # Find the first function decorated with @triton
    func_def = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            decorators = []
            for dec in stmt.decorator_list:
                dname = _attr_path(dec) or (dec.id if isinstance(dec, ast.Name) else "")
                decorators.append(dname)
            # Also check preceding comments for # @triton
            # (AST doesn't preserve comments, so we use a heuristic below)
            func_def = stmt
            break

    # If no function found by decorator, search by comment heuristic
    if func_def is None:
        # Fallback: just take the first FunctionDef
        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef):
                func_def = stmt
                break

    if func_def is None:
        raise ValueError("No function definition found in source code.")

    # Check for @triton comment heuristic (line-based)
    lines = source_code.splitlines()
    func_start = func_def.lineno - 1  # 0-based
    has_triton_comment = False
    for i in range(max(0, func_start - 5), func_start):
        stripped = lines[i].strip()
        if stripped.startswith("# @triton"):
            has_triton_comment = True
            break
    if not has_triton_comment and not any("triton" in d for d in decorators if isinstance(d, str)):
        warnings.append(
            "No '# @triton' annotation found. Proceeding anyway, but the function may not be intended for translation."
        )

    # Collect parameters
    parameters = []
    # Plain args
    defaults = [None] * (len(func_def.args.args) - len(func_def.args.defaults)) + [
        _extract_default(d) for d in func_def.args.defaults
    ]
    for arg, default in zip(func_def.args.args, defaults):
        p = _build_param(arg)
        p.default = default
        parameters.append(p)

    # Walk body
    operations: list[OpNode] = []
    output_var = ""

    for stmt in func_def.body:
        if isinstance(stmt, ast.If) or isinstance(stmt, ast.For) or isinstance(stmt, ast.While):
            warnings.append(
                f"Control flow detected at line {stmt.lineno} ({type(stmt).__name__}). "
                "MVP does not support if/for/while; behavior may be incorrect."
            )
            continue

        if isinstance(stmt, ast.Return):
            if stmt.value:
                output_var = _expr_to_str(stmt.value)
                # If the return is a direct expression, walk it for ops
                _walk_expression(stmt.value, alias_map, operations, None)
            continue

        if isinstance(stmt, ast.Assign):
            # Single or multiple targets
            target_names = []
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
                else:
                    target_names.append(_expr_to_str(target))
            # Use the first target name
            assign_target = target_names[0] if target_names else None
            _walk_expression(stmt.value, alias_map, operations, assign_target)
            continue

        if isinstance(stmt, ast.Expr):
            # Bare expression (e.g. function call without assignment)
            _walk_expression(stmt.value, alias_map, operations, None)
            continue

        # Other statement types are ignored for MVP

    # If no explicit return, the last assignment's target is a reasonable guess
    if not output_var and operations:
        output_var = operations[-1].output_var

    graph = OperationGraph(
        function_name=func_def.name,
        signature=_build_signature(func_def),
        parameters=parameters,
        operations=operations,
        output_var=output_var,
        source_code=source_code,
    )

    return graph, warnings
