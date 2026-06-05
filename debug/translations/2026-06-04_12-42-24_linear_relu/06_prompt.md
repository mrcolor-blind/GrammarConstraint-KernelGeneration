## SYSTEM

You are an expert in Triton programming. Given a PyTorch function and its complete mathematical description, generate a self-contained Python module containing:

1. Required imports (torch, triton, triton.language as tl)
2. One Triton kernel per fused operation group
3. A wrapper function with the EXACT same name and signature as the original PyTorch function

CRITICAL RULES:
- Use ONLY valid Triton APIs. The correct imports are:
    import triton
    import triton.language as tl
- Do NOT use: triton.lang, tl.mm(), tl.Scalar, tl.Tensor (as argument type), tl.zero(), or any other hallucinated API.
- For matmul: implement tiled matrix multiplication with tl.dot().
- For element-wise ops: use tl.load + compute + tl.store pattern.
- For reductions: use tl.reduce() or manual reduction loops.
- Design appropriate BLOCK_SIZE constants and grid launch.
- Return ONLY valid Python code. No markdown fences, no explanations.
- The wrapper must have the EXACT function name and EXACT signature provided.
- Preserve all parameter names and their order.


---

## USER

==================================================================
FUNCTION NAME: linear_relu
ORIGINAL SIGNATURE: linear_relu(x, weight, bias)

INPUT SHAPES:
  x: (N, D_in)
  weight: (D_out, D_in)
  bias: (D_out)

OUTPUT VARIABLE: torch.relu(z)

==================================================================

FUSION GROUP 1: fused_matmul_add_relu
------------------------------------------------------------------

  Op: torch.matmul
    Source: torch_docstring (confidence: medium)
    Inputs: ['x', 'weight.T']
    Output: _t1
    Shape: (N, D_out)
    Broadcasting: Supports broadcasting to a common shape.

  Op: torch.add
    Source: torch_docstring (confidence: medium)
    Math: \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i
    Inputs: ['x @ weight.T', 'bias']
    Output: z
    Shape: (N, D_out)
    Broadcasting: Supports broadcasting to a common shape.

  Op: torch.relu
    Source: name_only (confidence: low)
    Inputs: ['z']
    Output: _t3
    Shape: (N, D_out)

  FUSION REASONING: torch.matmul is compute-intensive → initiates group; torch.add (element_wise) → fused with previous; torch.relu (element_wise) → fused with previous

  SUGGESTED APPROACH:
    - Use 2D tiling with appropriate BLOCK_M, BLOCK_N, BLOCK_K.
    - Load tiles into shared memory for matmul/conv.
    - Fuse subsequent element-wise ops directly in registers.
    - Grid launch should cover output dimensions.

  KERNEL NAME: fused_matmul_add_relu_kernel
  WRAPPER NAME: linear_relu
==================================================================

IMPLEMENTATION REQUIREMENTS:
- Kernel 1 (fused_matmul_add_relu_kernel): Implement fused operations: torch.matmul, torch.add, torch.relu
- Wrapper: linear_relu(x, weight, bias)

Generate the complete, self-contained Python module now.

---
