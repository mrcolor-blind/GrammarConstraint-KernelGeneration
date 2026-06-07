import torch
from user_function import square

x = torch.randn(65536, device='cpu')
result = square(x)
