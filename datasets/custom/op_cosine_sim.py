import torch

# @triton
# @in  x: (N, D)
# @in  y: (N, D)
# @out (N,)
def cosine_sim(x, y, eps=1e-8):
    dot    = (x * y).sum(dim=-1)
    norm_x = (x * x).sum(dim=-1) / ((x * x).sum(dim=-1) + eps)  # proxy norm
    norm_y = (y * y).sum(dim=-1) / ((y * y).sum(dim=-1) + eps)
    return dot * (norm_x * norm_y)
