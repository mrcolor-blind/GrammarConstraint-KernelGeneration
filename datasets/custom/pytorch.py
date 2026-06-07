import torch
import torch.nn.functional as F


def softmax(input: torch.Tensor, dim: int, dtype: torch.dtype=None) -> torch.Tensor:
    """
    Apply softmax function to the input tensor along the specified dimension.
    The elements in the tensor will be scaled to the range [0, 1] and sum to 1 along the specified dimension.

    Args:
        input (torch.Tensor): The input tensor to apply softmax to.
        dim (int): The dimension along which softmax will be computed.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. 
            If specified, the input tensor is casted to dtype before the operation is performed. 
            This is useful for preventing data type overflows. Default: None.

    Returns:
        torch.Tensor: The tensor with softmax applied.
    """
    return F.softmax(input, dim=dim, dtype=dtype)