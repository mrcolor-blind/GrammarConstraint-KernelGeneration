import torch
from user_function import soft_clamp

x = torch.randn(512, 1024, device='cpu') * 3.0
result = soft_clamp(x, low=-1.0, high=1.0)
