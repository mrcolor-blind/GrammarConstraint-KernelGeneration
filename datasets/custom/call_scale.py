import torch
from user_function import scale

x = torch.randn(65536, device='cpu')
result = scale(x, alpha=3.14)
