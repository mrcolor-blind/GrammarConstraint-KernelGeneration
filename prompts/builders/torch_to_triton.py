"""
Prompt Builder — constructs the user message for multi-operation PyTorch → Triton translation.
"""

from pathlib import Path

from models.domain import FusionPlan, OperationGraph, OpContext


TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "templates"
    / "torch_translation.txt"
)


class TorchToTritonPromptBuilder:
    def __init__(self):
        self.template = TEMPLATE_PATH.read_text()

    def build(
        self,
        graph: OperationGraph,
        fusion_plan: FusionPlan,
        contexts: dict[str, OpContext],
    ) -> list[dict]:
        """
        Build an OpenAI-style chat message list.
        Returns [system_message, user_message].
        """
        system_msg = {"role": "system", "content": self.template}
        user_content = self._render_user_message(graph, fusion_plan, contexts)
        user_msg = {"role": "user", "content": user_content}
        return [system_msg, user_msg]

    @staticmethod
    def _render_user_message(
        graph: OperationGraph,
        fusion_plan: FusionPlan,
        contexts: dict[str, OpContext],
    ) -> str:
        # Check if we can use "direct mode" (single op, TritonBench context)
        is_single_op = len(graph.operations) == 1
        is_tritonbench = False
        full_instruction = None

        # ── 1. Check if the function name itself is a composite TritonBench operator ──
        # Examples: gelu_std, add_gelu, sum_std, etc.
        from datasets.tritonbench.registry import get_registry
        registry = get_registry()
        if registry.is_bench_operator(graph.function_name):
            entry = registry.get_entry(graph.function_name)
            if entry and entry.get("instruction"):
                is_tritonbench = True
                full_instruction = entry["instruction"]

        if is_single_op and not is_tritonbench and graph.operations:
            # Fallback: check if the single op is in TritonBench (old behavior)
            op = graph.operations[0]
            ctx = contexts.get(op.op_name)
            if ctx and ctx.source == "tritonbench_json" and ctx.full_instruction:
                is_tritonbench = True
                full_instruction = ctx.full_instruction

        if is_tritonbench and full_instruction:
            return TorchToTritonPromptBuilder._render_direct_mode(
                graph, full_instruction
            )

        return TorchToTritonPromptBuilder._render_fusion_mode(
            graph, fusion_plan, contexts
        )

    @staticmethod
    def _render_direct_mode(
        graph: OperationGraph,
        full_instruction: str,
    ) -> str:
        """
        Render a lean prompt for single-op functions that are in TritonBench.
        Puts the full TritonBench instruction front and center, plus user-specific metadata.
        """
        lines = []
        lines.append("=" * 66)
        lines.append(f"FUNCTION NAME: {graph.function_name}")
        lines.append(f"ORIGINAL SIGNATURE: {graph.signature}")
        lines.append("")

        # Input shapes from parameters
        lines.append("INPUT SHAPES:")
        for p in graph.parameters:
            shape_str = p.shape if p.shape else "(not annotated)"
            lines.append(f"  {p.name}: {shape_str}")
        lines.append("")

        lines.append(f"OUTPUT VARIABLE: {graph.output_var}")
        lines.append("")
        lines.append("=" * 66)
        lines.append("")
        lines.append("The following operator is defined in TritonBench with its complete specification:")
        lines.append("")
        lines.append("-" * 66)
        lines.append(full_instruction)
        lines.append("-" * 66)
        lines.append("")
        lines.append("Now generate a Triton kernel for this exact user function with the following requirements:")
        lines.append("- The wrapper must match the ORIGINAL SIGNATURE exactly.")
        lines.append("- Use the input shapes provided above to design appropriate BLOCK_SIZE and grid.")
        lines.append("- Return ONLY valid Python code. No markdown fences, no explanations.")
        lines.append("")
        lines.append("Generate the complete, self-contained Python module now.")

        return "\n".join(lines)

    @staticmethod
    def _render_fusion_mode(
        graph: OperationGraph,
        fusion_plan: FusionPlan,
        contexts: dict[str, OpContext],
    ) -> str:
        """
        Render the full fusion-mode prompt for multi-op or non-TritonBench functions.
        """
        lines = []
        lines.append("=" * 66)
        lines.append(f"FUNCTION NAME: {graph.function_name}")
        lines.append(f"ORIGINAL SIGNATURE: {graph.signature}")
        lines.append("")

        # Input shapes from parameters
        lines.append("INPUT SHAPES:")
        for p in graph.parameters:
            shape_str = p.shape if p.shape else "(not annotated)"
            lines.append(f"  {p.name}: {shape_str}")
        lines.append("")

        lines.append(f"OUTPUT VARIABLE: {graph.output_var}")
        lines.append("")
        lines.append("=" * 66)

        # Fusion groups
        for group in fusion_plan.groups:
            lines.append("")
            lines.append(f"FUSION GROUP {group.group_id + 1}: {group.fused_name}")
            lines.append("-" * 66)
            lines.append("")

            for op in group.operations:
                ctx = contexts.get(op.op_name)
                if ctx is None:
                    ctx = OpContext(
                        op_name=op.op_name,
                        source="name_only",
                        confidence="low",
                        functional_description="",
                    )

                lines.append(f"  Op: {op.op_name}")
                lines.append(f"    Source: {ctx.source} (confidence: {ctx.confidence})")
                if ctx.math_formula:
                    lines.append(f"    Math: {ctx.math_formula}")
                lines.append(f"    Inputs: {op.input_vars}")
                lines.append(f"    Output: {op.output_var}")
                if op.shape:
                    lines.append(f"    Shape: {op.shape}")
                if ctx.broadcasting:
                    lines.append(f"    Broadcasting: {ctx.broadcasting}")
                if ctx.edge_cases:
                    lines.append(f"    Edge cases: {ctx.edge_cases}")
                lines.append("")

            lines.append(f"  FUSION REASONING: {group.reasoning}")
            lines.append("")
            lines.append("  SUGGESTED APPROACH:")
            lines.append(_suggest_approach(group))
            lines.append("")
            lines.append(f"  KERNEL NAME: {group.fused_name}_kernel")
            lines.append(f"  WRAPPER NAME: {graph.function_name}")
            lines.append("=" * 66)

        # Implementation requirements
        lines.append("")
        lines.append("IMPLEMENTATION REQUIREMENTS:")
        for group in fusion_plan.groups:
            lines.append(
                f"- Kernel {group.group_id + 1} ({group.fused_name}_kernel): "
                f"Implement fused operations: "
                + ", ".join(o.op_name for o in group.operations)
            )
        lines.append(f"- Wrapper: {graph.function_name}({', '.join(p.name for p in graph.parameters)})")
        lines.append("")
        lines.append("Generate the complete, self-contained Python module now.")

        return "\n".join(lines)


def _suggest_approach(group) -> str:
    """Generate a suggested tiling / implementation strategy for a fused group."""
    ops = group.operations
    cats = set()
    for op in ops:
        # simple classification from fusion module
        from fusion.planner import classify_op
        cats.add(classify_op(op.op_name))

    if "compute_intensive" in cats:
        return (
            "    - Use 2D tiling with appropriate BLOCK_M, BLOCK_N, BLOCK_K.\n"
            "    - Load tiles into shared memory for matmul/conv.\n"
            "    - Fuse subsequent element-wise ops directly in registers.\n"
            "    - Grid launch should cover output dimensions."
        )
    if "reduction" in cats:
        return (
            "    - One program per reduction group (e.g., one thread block per row).\n"
            "    - Load input, reduce in shared memory or warp shuffle, write result.\n"
            "    - For mean/var: compute sum then divide."
        )
    if "element_wise" in cats and len(ops) > 1:
        return (
            "    - Simple element-wise fusion: load, compute chain, store.\n"
            "    - One program per output element (1D grid over flattened output).\n"
            "    - All ops fused into a single tl.load / compute / tl.store sequence."
        )
    return (
        "    - Implement each operation in the group sequentially within the kernel.\n"
        "    - Choose grid/block sizes matching the output tensor shape."
    )
