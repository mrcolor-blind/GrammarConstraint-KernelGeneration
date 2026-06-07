import torch

# @triton
# @in  x: (M, N)
# @out (M, N)
def mean_center(x):
    return x - x.mean(dim=-1, keepdim=True)
