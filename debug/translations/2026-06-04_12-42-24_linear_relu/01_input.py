# @triton
# @in  x:      (N, D_in)
# @in  weight: (D_out, D_in)
# @in  bias:   (D_out,)
# @out (N, D_out)
def linear_relu(x, weight, bias):
    z = x @ weight.T + bias
    return torch.relu(z)
