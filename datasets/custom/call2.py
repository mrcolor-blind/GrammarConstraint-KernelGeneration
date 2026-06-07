from user_function import add_mean
import torch

def test_addmm():
    results = {}

    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0)

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = addmm(input3, mat1_3, mat2_3, beta=0.0)

    return results

test_results = test_addmm()