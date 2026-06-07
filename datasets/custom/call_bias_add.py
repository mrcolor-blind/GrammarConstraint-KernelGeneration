import torch
from user_function import bias_add

x    = torch.randn(512, 1024, device='cpu')
bias = torch.randn(1024, device='cpu')
result = bias_add(x, bias)
