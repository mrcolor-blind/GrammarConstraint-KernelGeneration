import torch

# @triton
# @in  x: (N,)
# @out (N,)
def scale(x, alpha=2.0):
    return alpha * x
