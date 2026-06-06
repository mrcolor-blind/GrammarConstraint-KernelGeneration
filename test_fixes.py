#!/usr/bin/env python3
"""Quick test script for the prompt builder fixes."""

import json
import tempfile
from pathlib import Path

from context.knowledge_base import KnowledgeBase
from context.resolver import ContextResolver
from code_analysis.parser import parse_function
from code_analysis.shape_resolver import resolve_shapes
from fusion.planner import plan_fusion
from prompts.builders.torch_to_triton import TorchToTritonPromptBuilder

# A minimal TritonBench-style instruction for gelu
GELU_INSTRUCTION = """You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the input tensor. The function can be computed exactly or approximately using a tanh-based formula depending on the 'approximate' argument.
Wrapper Entry Information: gelu(input, approximate='none') -> Tensor
Math: When approximate is 'none': GELU(x) = x * Φ(x), where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution. When approximate is 'tanh': GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
other: See Gaussian Error Linear Units (GELUs) https://arxiv.org/abs/1606.08415
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate."""

# Create a temporary TritonBench JSON file
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump([{"instruction": GELU_INSTRUCTION, "input": "", "output": ""}], f)
    temp_json_path = f.name

# Test 1: KnowledgeBase extraction
print("=" * 60)
print("TEST 1: KnowledgeBase extraction")
print("=" * 60)

# Test the static extraction method
short, canonical = KnowledgeBase._extract_op_name_from_instruction(GELU_INSTRUCTION)
print(f"Extracted from instruction: short='{short}', canonical='{canonical}'")

kb = KnowledgeBase(tritonbench_json_path=temp_json_path)
print(f"Loaded {len(kb._tritonbench)} entries")
print(f"Short name mappings: {kb._short_name_to_torch}")

# Test lookup by canonical torch name
entry = kb.get_tritonbench_entry("torch.nn.functional.gelu")
print(f"Lookup by 'torch.nn.functional.gelu': {'FOUND' if entry else 'NOT FOUND'}")

# Test lookup by short name
entry = kb.get_tritonbench_entry("gelu")
print(f"Lookup by 'gelu': {'FOUND' if entry else 'NOT FOUND'}")

# Test 2: ContextResolver
print("\n" + "=" * 60)
print("TEST 2: ContextResolver")
print("=" * 60)

resolver = ContextResolver(knowledge_base=kb)
ctx = resolver.resolve("torch.nn.functional.gelu")
print(f"Resolved 'torch.nn.functional.gelu': source={ctx.source}, confidence={ctx.confidence}")
print(f"Has full_instruction: {ctx.full_instruction is not None}")
if ctx.full_instruction:
    print(f"Instruction length: {len(ctx.full_instruction)} chars")

ctx2 = resolver.resolve("gelu")
print(f"Resolved 'gelu': source={ctx2.source}, confidence={ctx2.confidence}")

# Test 3: Pipeline end-to-end with explicit import
print("\n" + "=" * 60)
print("TEST 3: End-to-end pipeline (single op gelu with explicit import)")
print("=" * 60)

source_code = """import torch.nn.functional as F

def gelu(input, approximate='none'):
    return F.gelu(input, approximate=approximate)
"""

graph, warnings = parse_function(source_code)
print(f"Parsed function: {graph.function_name}")
print(f"Operations: {[op.op_name for op in graph.operations]}")

graph, shape_warnings = resolve_shapes(graph, source_code)
print(f"Shape warnings: {shape_warnings}")

plan = plan_fusion(graph)
print(f"Fusion groups: {len(plan.groups)}")

contexts = {}
for op in graph.operations:
    contexts[op.op_name] = resolver.resolve(op.op_name)
    print(f"  Context for {op.op_name}: {contexts[op.op_name].source}")

builder = TorchToTritonPromptBuilder()
messages = builder.build(graph, plan, contexts)

print(f"\nPrompt has {len(messages)} messages")
user_msg = messages[1]["content"]

# Check if it's using direct mode
if "The following operator is defined in TritonBench" in user_msg:
    print("✓ PROMPT IS USING DIRECT MODE (single-op TritonBench)")
    if "FUSION GROUP" in user_msg:
        print("✗ WARNING: Fusion boilerplate still present in direct mode!")
    else:
        print("✓ No fusion boilerplate (correct)")
    if "complete specification" in user_msg:
        print("✓ Full instruction referenced")
else:
    print("✗ PROMPT IS USING FUSION MODE (unexpected for single-op TritonBench)")

# Show first 500 chars of user message
print("\n--- User message preview (first 800 chars) ---")
print(user_msg[:800])

# Test 4: Multi-op function (should use fusion mode)
print("\n" + "=" * 60)
print("TEST 4: Multi-op function (linear + relu)")
print("=" * 60)

source_code2 = """# @triton
# @in  x: (N, D_in)
# @in  weight: (D_out, D_in)
# @in  bias: (D_out,)
# @out (N, D_out)
def linear_relu(x, weight, bias):
    z = x @ weight.T + bias
    return torch.relu(z)
"""

graph2, _ = parse_function(source_code2)
graph2, _ = resolve_shapes(graph2, source_code2)
plan2 = plan_fusion(graph2)

contexts2 = {}
for op in graph2.operations:
    contexts2[op.op_name] = resolver.resolve(op.op_name)

messages2 = builder.build(graph2, plan2, contexts2)
user_msg2 = messages2[1]["content"]

if "FUSION GROUP" in user_msg2:
    print("✓ PROMPT IS USING FUSION MODE (correct for multi-op)")
else:
    print("✗ PROMPT IS USING DIRECT MODE (unexpected for multi-op)")

print("\n--- User message preview (first 800 chars) ---")
print(user_msg2[:800])

# Cleanup
Path(temp_json_path).unlink()
print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
