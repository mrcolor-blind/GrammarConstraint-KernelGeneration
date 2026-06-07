import torch
from user_function import cosine_sim

x = torch.randn(1024, 256, device='cpu')
y = torch.randn(1024, 256, device='cpu')
result = cosine_sim(x, y)
