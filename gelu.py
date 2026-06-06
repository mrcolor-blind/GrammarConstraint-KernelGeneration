import torch

def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
    return torch.gelu(input, approximate=approximate)