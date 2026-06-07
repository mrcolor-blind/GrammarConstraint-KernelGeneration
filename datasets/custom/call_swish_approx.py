import torch
from user_function import swish_approx

x = torch.randn(512, 1024, device='cpu')
result = swish_approx(x)
