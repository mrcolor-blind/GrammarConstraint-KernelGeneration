from user_function import linear_relu
import torch

x = torch.randn(128, 256)
weight = torch.randn(512, 256)
bias = torch.randn(512)
out = linear_relu(x, weight, bias)