"""
Shape Resolver — parses inline @triton shape annotations and propagates them
through the OperationGraph.
"""

import ast
import re

from models.domain import OperationGraph, OpNode
from typing import Optional, Union


# Regex for inline shape annotations
_RE_IN = re.compile(r"#\s*@in\s+(\w+)\s*:\s*(.+)")
_RE_OUT = re.compile(r"#\s*@out\s*(.+)")
_RE_TRITON = re.compile(r"#\s*@triton\b")


# Mapping from torch op names to their Python operator symbols
_OP_SYMBOLS = {
    "torch.add": " + ",
    "torch.sub": " - ",
    "torch.mul": " * ",
    "torch.div": " / ",
    "torch.floor_divide": " // ",
    "torch.matmul": " @ ",
    "torch.pow": " ** ",
    "torch.remainder": " % ",
}


def _reconstruct_expr(op: OpNode) -> str:
    """Reconstruct the original expression string from an OpNode."""
    sym = _OP_SYMBOLS.get(op.op_name)
    if sym and len(op.input_vars) == 2:
        return f"{op.input_vars[0]}{sym}{op.input_vars[1]}"
    return ""


def _tokenize_shape(shape_str: str) -> Union[tuple, str]:
    """
    Parse a shape string like '(N, D_in)', 'scalar', '(4, 3, 32, 32)'.
    Returns a tuple for multi-dim shapes, 'scalar' for 0-dim, or the raw string.
    """
    s = shape_str.strip()
    if s.lower() == "scalar":
        return ()
    if s.lower() == "none":
        return None
    # Handle union: (C,)Optional[ ]
    if "|" in s:
        return s  # keep as string for now
    # Match (a, b, c) format
    m = re.match(r"\((.*)\)", s)
    if m:
        inner = m.group(1).strip()
        if not inner:
            return ()  # empty tuple -> scalar-like
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        # Handle * for batch dims: (*, D_in) -> ('*', 'D_in')
        return tuple(parts)
    return s


def _shape_to_str(shape: Union[tuple, str, None]) -> str:
    if shape is None:
        return "None"
    if isinstance(shape, str):
        return shape
    if shape == ():
        return "scalar"
    return f"({', '.join(shape)})"


def parse_annotations(source_code: str) -> tuple[dict[str, Union[tuple, str, None]], Union[tuple, str, None]]:
    """
    Parse @in and @out annotations from source code comments.
    Returns (input_shapes dict, output_shape).
    """
    input_shapes = {}
    output_shape = None
    for line in source_code.splitlines():
        stripped = line.strip()
        m_in = _RE_IN.match(stripped)
        if m_in:
            param_name = m_in.group(1).strip()
            shape = _tokenize_shape(m_in.group(2))
            input_shapes[param_name] = shape
        m_out = _RE_OUT.match(stripped)
        if m_out:
            output_shape = _tokenize_shape(m_out.group(1))
    return input_shapes, output_shape


def _broadcast_shapes(a: tuple, b: tuple) ->Optional[ tuple ]:
    """
    Apply PyTorch broadcasting rules to two symbolic shapes.
    Both a and b are tuples of dimension names/strings.
    Returns the broadcasted shape or None if incompatible.
    """
    if not a and not b:
        return ()
    # Align from the right
    max_len = max(len(a), len(b))
    result = []
    for i in range(1, max_len + 1):
        da = a[-i] if i <= len(a) else None
        db = b[-i] if i <= len(b) else None
        if da is None:
            result.append(db)
        elif db is None:
            result.append(da)
        elif da == db:
            result.append(da)
        elif da == "1" or da == "1L":
            result.append(db)
        elif db == "1" or db == "1L":
            result.append(da)
        elif da == "*" or db == "*":
            # wildcard batch dimension
            result.append(da if da != "*" else db)
        else:
            # Incompatible broadcasting
            return None
    return tuple(reversed(result))


def _infer_matmul_shape(a: tuple, b: tuple) ->Optional[ tuple ]:
    """
    Infer output shape of torch.matmul given input shapes.
    Supports batch dims: (..., M, K) @ (..., K, N) -> (..., M, N)
    """
    if len(a) < 1 or len(b) < 1:
        return None
    # Last two dims define the matrix multiply
    a_last = a[-2:]
    b_last = b[-2:]
    # Handle vector cases
    if len(a_last) == 1 and len(b_last) == 2:
        # (K,) @ (K, N) -> (N,)
        if a_last[0] == b_last[0]:
            return b_last[1:]
        return None
    if len(a_last) == 2 and len(b_last) == 1:
        # (M, K) @ (K,) -> (M,)
        if a_last[1] == b_last[0]:
            return a_last[:1]
        return None
    if len(a_last) == 2 and len(b_last) == 2:
        if a_last[1] != b_last[0]:
            return None
        # Batch dims: all preceding dims must broadcast
        batch_a = a[:-2]
        batch_b = b[:-2]
        batch = _broadcast_shapes(batch_a, batch_b)
        if batch is None:
            return None
        return batch + (a_last[0], b_last[1])
    return None


def _infer_shape(op: OpNode, known_shapes: dict[str, Union[tuple, str, None]]) -> Union[tuple, str, None]:
    """
    Infer the output shape of a single OpNode given known variable shapes.
    Returns the inferred shape tuple, 'scalar', '<unknown>', or None.
    """
    op_name = op.op_name
    inputs = op.input_vars

    # Resolve input shapes from known_shapes (best-effort string match)
    input_shapes = []
    for iv in inputs:
        # First try exact match (for intermediate expressions like "x @ weight.T")
        shape = known_shapes.get(iv)
        if shape is not None:
            input_shapes.append(shape)
            continue
        # Strip subscript/indexing for lookup, e.g. "weight.T" -> "weight"
        base = iv.split(".")[0].split("[")[0].strip()
        shape = known_shapes.get(base)
        input_shapes.append(shape)

    # --- element-wise ops (same shape, with broadcasting) ---
    elementwise = {
        "torch.add", "torch.sub", "torch.mul", "torch.div",
        "torch.floor_divide", "torch.remainder", "torch.pow",
        "torch.relu", "torch.gelu", "torch.sigmoid", "torch.tanh",
        "torch.exp", "torch.log", "torch.sqrt", "torch.abs",
        "torch.nn.functional.relu", "torch.nn.functional.gelu",
        "torch.nn.functional.sigmoid", "torch.nn.functional.tanh",
        "torch.clamp", "torch.minimum", "torch.maximum",
        "torch.neg", "torch.reciprocal",
    }
    if op_name in elementwise:
        if len(input_shapes) >= 2:
            s1 = input_shapes[0]
            s2 = input_shapes[1]
            if isinstance(s1, tuple) and isinstance(s2, tuple):
                return _broadcast_shapes(s1, s2)
            if isinstance(s1, tuple):
                return s1
            if isinstance(s2, tuple):
                return s2
            return s1 if s1 is not None else s2
        if len(input_shapes) == 1:
            return input_shapes[0]
        return "<unknown>"

    # --- matmul ---
    if op_name == "torch.matmul":
        if len(input_shapes) >= 2:
            a = input_shapes[0]
            b = input_shapes[1]
            # Handle .T suffix: swap last two dims of b
            if len(inputs) >= 2 and ".T" in inputs[1] and isinstance(b, tuple) and len(b) >= 2:
                b = b[:-2] + (b[-1], b[-2])
            if isinstance(a, tuple) and isinstance(b, tuple):
                return _infer_matmul_shape(a, b)
        return "<unknown>"

    # --- reductions ---
    reductions = {
        "torch.mean", "torch.sum", "torch.var", "torch.std",
        "torch.max", "torch.min", "torch.argmax", "torch.softmax",
    }
    if op_name in reductions:
        # Without knowing dim/keepdim, we can't infer precisely
        # Default: assume keepdim=True and dim is last axis -> same shape as input
        if len(input_shapes) >= 1:
            s = input_shapes[0]
            if isinstance(s, tuple) and len(s) >= 1:
                # Most common: reduce over last dim, keepdim=True
                return s  # same shape as input when keepdim=True
            return s
        return "<unknown>"

    # --- conv2d ---
    if op_name in ("torch.nn.functional.conv2d", "torch.conv2d"):
        # H_out = (H + 2*pad - dil*(K-1) - 1)//stride + 1
        # Without stride/padding info, we can't compute exact H_out/W_out
        # Return symbolic: (N, C_out, H_out, W_out) if we know input is (N,C,H,W)
        if len(input_shapes) >= 1:
            s = input_shapes[0]
            if isinstance(s, tuple) and len(s) == 4:
                return (s[0], "C_out", "H_out", "W_out")
        return "<unknown>"

    # --- normalization ---
    if op_name in ("torch.nn.functional.batch_norm", "torch.nn.functional.layer_norm"):
        if len(input_shapes) >= 1:
            return input_shapes[0]
        return "<unknown>"

    return "<unknown>"


def resolve_shapes(graph: OperationGraph, source_code: str) -> tuple[OperationGraph, list[str]]:
    """
    Attach shapes to every OpNode in *graph* by parsing inline annotations
    and propagating forward through the operation list.
    Returns the modified graph and a list of warnings.
    """
    warnings: list[str] = []
    input_shapes, output_shape = parse_annotations(source_code)

    # Validate that all @in names exist as parameters
    param_names = {p.name for p in graph.parameters}
    for name in input_shapes:
        if name not in param_names:
            warnings.append(f"@in annotation '{name}' does not match any function parameter.")

    # Build a map of variable -> shape from inputs
    known_shapes: dict[str, Union[tuple, str, None]] = {}
    for param in graph.parameters:
        shape = input_shapes.get(param.name)
        if shape is not None:
            known_shapes[param.name] = shape
            param.shape = _shape_to_str(shape)

    # Propagate through operations
    for op in graph.operations:
        inferred = _infer_shape(op, known_shapes)
        op.shape = _shape_to_str(inferred) if inferred is not None else "<unknown>"
        # Update known_shapes with the output variable
        known_shapes[op.output_var] = inferred
        # Also map the original expression string for parent ops to find
        expr_str = _reconstruct_expr(op)
        if expr_str:
            known_shapes[expr_str] = inferred

    # Check if we can validate the output shape annotation
    if output_shape is not None:
        final_shape = known_shapes.get(graph.output_var)
        # We just note it; exact symbolic equality is hard, so we just warn
        if final_shape is None:
            warnings.append(
                f"Output variable '{graph.output_var}' shape could not be inferred; "
                "annotated output shape may not match."
            )

    return graph, warnings
