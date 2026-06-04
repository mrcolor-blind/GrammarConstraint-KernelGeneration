"""
Op Detector — resolves Python AST call nodes into fully-qualified torch operator names.
"""

import ast
from typing import Optional, Union


# Map Python binary operators to torch function names
_BINARY_OP_MAP = {
    ast.Add: "torch.add",
    ast.Sub: "torch.sub",
    ast.Mult: "torch.mul",
    ast.Div: "torch.div",
    ast.FloorDiv: "torch.floor_divide",
    ast.MatMult: "torch.matmul",
    ast.Pow: "torch.pow",
    ast.Mod: "torch.remainder",
}

# Common torch.nn.functional aliases
_TORCH_FN_ALIASES = {
    "F": "torch.nn.functional",
    "nn.functional": "torch.nn.functional",
    "torch.nn": "torch.nn",
}

# Known torch module prefixes
_TORCH_PREFIXES = ("torch.", "torch.nn.functional.")


def resolve_binary_op(op_node: ast.BinOp) ->Optional[ str ]:
    """Map a Python BinOp AST node to a torch function name."""
    return _BINARY_OP_MAP.get(type(op_node.op))


def resolve_call(node: ast.Call, alias_map: dict[str, str]) ->Optional[ str ]:
    """
    Resolve an ast.Call node to a fully-qualified torch operator name.
    Returns None if the call is not a known torch operator.
    """
    if isinstance(node.func, ast.Name):
        name = node.func.id
        # Direct torch call, e.g. torch.add
        if name in alias_map:
            resolved = alias_map[name]
            # If alias maps to a full module path, append function name
            if resolved.startswith("torch.") and "." in resolved[6:]:
                return resolved  # already fully qualified
            return f"{resolved}.{name}" if resolved.startswith("torch") else resolved
        return name if name.startswith("torch.") else None

    if isinstance(node.func, ast.Attribute):
        # e.g. torch.add, F.relu, x.add, torch.nn.functional.conv2d
        path = _attr_path(node.func)
        if path is None:
            return None

        # Direct torch path
        if path.startswith("torch."):
            return path

        # Alias resolution: F.relu -> torch.nn.functional.relu
        top = path.split(".")[0]
        if top in alias_map:
            prefix = alias_map[top]
            rest = ".".join(path.split(".")[1:])
            return f"{prefix}.{rest}" if rest else prefix

        # Method call on a tensor: x.add(y) -> torch.Tensor.add -> fallback torch.add
        if len(path.split(".")) == 2:
            method = path.split(".")[1]
            # Many torch.Tensor methods map directly to torch functions
            return f"torch.{method}"

    return None


def _attr_path(node: ast.AST) ->Optional[ str ]:
    """Flatten an Attribute chain into a dotted string, e.g. torch.nn.functional.relu."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attr_path(node.value)
        if prefix is None:
            return None
        return f"{prefix}.{node.attr}"
    return None


def build_alias_map(body: list[ast.stmt]) -> dict[str, str]:
    """
    Scan top-level import statements and build an alias map.
    Returns dict of short_name -> fully_qualified_module.
    """
    alias_map = {}
    for stmt in body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                name = alias.asname if alias.asname else alias.name
                alias_map[name] = alias.name
        elif isinstance(stmt, ast.ImportFrom):
            module = stmt.module or ""
            for alias in stmt.names:
                name = alias.asname if alias.asname else alias.name
                if module:
                    alias_map[name] = f"{module}.{alias.name}"
                else:
                    alias_map[name] = alias.name
    return alias_map
