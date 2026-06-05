# @triton
# @in  input: (N,)
# @out (N,)
def gelu(input, approximate='none'):
    return input * 0.5 * (1.0 + torch.erf(input / 1.4142135623730951))
