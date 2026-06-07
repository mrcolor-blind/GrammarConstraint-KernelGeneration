import torch

# @triton
# @in  x:      (N, D)
# @in  weight: (D,)
# @out (N, D)
def rms_norm(x, weight, eps=1e-6):
    rms   = (x * x).mean(dim=-1, keepdim=True)
    x_hat = x / (rms + eps)
    return weight * x_hat
