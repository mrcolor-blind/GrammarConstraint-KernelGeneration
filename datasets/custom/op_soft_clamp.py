import torch

# @triton
# @in  x:   (M, N)
# @out (M, N)
def soft_clamp(x, low=-1.0, high=1.0):
    # Diferentiable clamp: low + (high-low) * x^2 / (1 + x^2) cuando x>0 etc.
    # Simple version: clamp con torch.clamp (no está en TritonBench)
    return torch.clamp(x, min=low, max=high)
