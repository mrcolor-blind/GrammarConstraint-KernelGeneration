import torch

# @triton
# @in  x: (N,)
# @in  w: (N,)
# @in  b: (N,)
# @out (N,)
def affine(x, w, b):
    return w * x + b
