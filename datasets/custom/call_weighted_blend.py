import torch
from user_function import weighted_blend

x = torch.randn(65536, device='cpu')
y = torch.randn(65536, device='cpu')
result = weighted_blend(x, y, alpha=0.3)
