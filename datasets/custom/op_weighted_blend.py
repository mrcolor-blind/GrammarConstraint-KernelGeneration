import torch

# @triton
# @in  x: (N,)
# @in  y: (N,)
# @out (N,)
def weighted_blend(x, y, alpha=0.5):
    return alpha * x + (1.0 - alpha) * y
