import torch
from user_function import affine

x = torch.randn(65536, device='cpu')
w = torch.randn(65536, device='cpu')
b = torch.randn(65536, device='cpu')
result = affine(x, w, b)
