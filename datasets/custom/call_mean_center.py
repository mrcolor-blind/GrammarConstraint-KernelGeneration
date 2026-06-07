import torch
from user_function import mean_center

x = torch.randn(512, 1024, device='cpu')
result = mean_center(x)
