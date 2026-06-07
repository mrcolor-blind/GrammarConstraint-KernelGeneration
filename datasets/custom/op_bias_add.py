import torch

# @triton
# @in  x:    (M, N)
# @in  bias: (N,)
# @out (M, N)
def bias_add(x, bias):
    return x + bias
