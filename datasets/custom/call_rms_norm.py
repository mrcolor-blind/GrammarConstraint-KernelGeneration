import torch
from user_function import rms_norm

x      = torch.randn(512, 768, device='cpu')
weight = torch.ones(768, device='cpu')
result = rms_norm(x, weight)
