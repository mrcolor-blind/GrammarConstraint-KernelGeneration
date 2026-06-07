import torch

def addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta: float=1, alpha: float=1, out: torch.Tensor=None) -> torch.Tensor:
    """
    Performs matrix multiplication of mat1 and mat2, and adds input to the result.

    Parameters:
        input (torch.Tensor): Matrix to be added.
        mat1 (torch.Tensor): The first matrix to be matrix-multiplied.
        mat2 (torch.Tensor): The second matrix to be matrix-multiplied.
        beta (float, optional): Multiplier for input (default is 1).
        alpha (float, optional): Multiplier for mat1 @ mat2 (default is 1).
        out (torch.Tensor, optional): The output tensor to store the result.

    Returns:
        torch.Tensor: The resulting tensor after performing the operation.
    
    This function performs the matrix multiplication of mat1 and mat2, scales the result by alpha,
    and then adds it to the input matrix scaled by beta. The resulting matrix is returned.
    
    If input is sparse, the result will have the same layout as input. If out is provided,
    it must have the same layout as input. If beta is 0, the input will be ignored, and nan or inf
    in input will not be propagated.
    """
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)