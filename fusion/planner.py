"""
Fusion Planner — groups operations into fused Triton kernels using conservative heuristics.
"""

from models.domain import FusionPlan, FusedGroup, OpNode, OperationGraph


# Classification of known operators
_ELEMENT_WISE = {
    "torch.add", "torch.sub", "torch.mul", "torch.div",
    "torch.floor_divide", "torch.remainder", "torch.pow",
    "torch.relu", "torch.gelu", "torch.sigmoid", "torch.tanh",
    "torch.exp", "torch.log", "torch.sqrt", "torch.abs",
    "torch.nn.functional.relu", "torch.nn.functional.gelu",
    "torch.nn.functional.sigmoid",
    "torch.maximum", "torch.minimum", "torch.clamp",
}

_COMPUTE_INTENSIVE = {
    "torch.matmul", "torch.bmm", "torch.mm",
    "torch.conv1d", "torch.conv2d", "torch.conv3d",
    "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d",
    "torch.nn.functional.linear",
}

_REDUCTION = {
    "torch.mean", "torch.sum", "torch.var", "torch.std",
    "torch.max", "torch.min", "torch.argmax", "torch.argmin",
    "torch.softmax", "torch.log_softmax",
    "torch.nn.functional.softmax", "torch.nn.functional.log_softmax",
}

_NORMALIZATION = {
    "torch.nn.functional.batch_norm", "torch.nn.functional.layer_norm",
    "torch.nn.functional.group_norm", "torch.nn.functional.instance_norm",
    "torch.batch_norm", "torch.layer_norm", "torch.group_norm",
    "torch.rms_norm",
}

_LAYOUT = {
    "torch.reshape", "torch.view", "torch.permute", "torch.transpose",
    "torch.flatten", "torch.unsqueeze", "torch.squeeze",
    "torch.nn.functional.interpolate",
}

_REGULARIZATION = {
    "torch.nn.functional.dropout", "torch.nn.functional.drop_path",
    "torch.dropout", "torch.drop_path",
}


def classify_op(op_name: str) -> str:
    """Classify an operator into a fusion category."""
    if op_name in _ELEMENT_WISE:
        return "element_wise"
    if op_name in _COMPUTE_INTENSIVE:
        return "compute_intensive"
    if op_name in _REDUCTION:
        return "reduction"
    if op_name in _NORMALIZATION:
        return "normalization"
    if op_name in _LAYOUT:
        return "layout"
    if op_name in _REGULARIZATION:
        return "regularization"
    return "other"


def is_fusible(prev: OpNode, curr: OpNode) -> bool:
    """
    Determine whether *curr* can be fused into the same kernel group as *prev*.
    Conservative rules from PLAN.md §5.2 / §7.4.
    """
    prev_cat = classify_op(prev.op_name)
    curr_cat = classify_op(curr.op_name)

    # Rule 1: element-wise + element-wise → always fuse
    if prev_cat == "element_wise" and curr_cat == "element_wise":
        return True

    # Rule 2: compute_intensive + element-wise → fuse (data already in registers)
    if prev_cat == "compute_intensive" and curr_cat == "element_wise":
        return True

    # Rule 3: reduction ends a group → nothing after it fuses
    if prev_cat == "reduction":
        return False

    # Rule 4: layout changes → new group
    if curr_cat == "layout":
        return False

    # Rule 5: dropout / regularization → own kernel
    if prev_cat == "regularization" or curr_cat == "regularization":
        return False

    # Rule 6: normalization → own kernel
    if prev_cat == "normalization" or curr_cat == "normalization":
        return False

    # Rule 7: other ops → don't fuse (conservative)
    if prev_cat == "other" or curr_cat == "other":
        return False

    # Default: do not fuse
    return False


def plan_fusion(graph: OperationGraph) -> FusionPlan:
    """
    Build a FusionPlan from an OperationGraph using conservative heuristics.
    """
    groups: list[FusedGroup] = []
    current_group: list[OpNode] = []
    group_id = 0

    for op in graph.operations:
        if not current_group:
            current_group.append(op)
        elif is_fusible(current_group[-1], op):
            current_group.append(op)
        else:
            # Flush current group
            groups.append(
                _build_fused_group(group_id, current_group, graph)
            )
            group_id += 1
            current_group = [op]

    if current_group:
        groups.append(
            _build_fused_group(group_id, current_group, graph)
        )

    return FusionPlan(groups=groups, strategy="auto")


def _build_fused_group(group_id: int, ops: list[OpNode], graph: OperationGraph) -> FusedGroup:
    """Construct a FusedGroup with shapes and reasoning."""
    # Derive a fused name from the ops
    op_short_names = []
    for op in ops:
        # Take last component of op_name, e.g. torch.add -> add
        short = op.op_name.split(".")[-1]
        op_short_names.append(short)
    fused_name = "fused_" + "_".join(op_short_names)

    # Collect input shapes from the first op
    input_shapes = {}
    first_op = ops[0]
    for iv in first_op.input_vars:
        base = iv.split(".")[0].split("[")[0].strip()
        # Look for shape in graph parameters
        for p in graph.parameters:
            if p.name == base:
                input_shapes[iv] = "(see annotation)"
                break
        else:
            input_shapes[iv] = "<inferred>"

    # Output shape from last op
    last_op = ops[-1]
    output_shape = last_op.shape or "<unknown>"

    # Build reasoning string
    cats = [classify_op(o.op_name) for o in ops]
    reasons = []
    for i, (op, cat) in enumerate(zip(ops, cats)):
        if i == 0:
            if cat == "compute_intensive":
                reasons.append(f"{op.op_name} is compute-intensive → initiates group")
            else:
                reasons.append(f"{op.op_name} ({cat}) → starts group")
        else:
            reasons.append(f"{op.op_name} ({cat}) → fused with previous")

    reasoning = "; ".join(reasons)

    return FusedGroup(
        group_id=group_id,
        operations=ops,
        fused_name=fused_name,
        input_shapes=input_shapes,
        output_shape=output_shape,
        reasoning=reasoning,
    )
