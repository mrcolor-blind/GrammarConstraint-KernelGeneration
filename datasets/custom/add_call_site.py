import torch
from user_function import add

x = torch.randn(1024, 768, device='cpu')
y = torch.randn(1024, 768, device='cpu')
result = add(x, y, alpha=2.0)
