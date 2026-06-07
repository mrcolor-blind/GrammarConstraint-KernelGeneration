import torch

# @triton
# @in  x: (M, N)
# @out (M, N)
def swish_approx(x):
    # x * sigmoid_approx(x) where sigmoid_approx = 1 / (1 + x*x)
    return x / (1.0 + x * x)
