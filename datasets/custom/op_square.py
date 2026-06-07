import torch

# @triton
# @in  x: (N,)
# @out (N,)
def square(x):
    return x * x
