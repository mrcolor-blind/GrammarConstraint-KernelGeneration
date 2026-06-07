import torch

def linear_relu(x, weight, bias):
    z = x @ weight.T + bias
    return torch.relu(z)