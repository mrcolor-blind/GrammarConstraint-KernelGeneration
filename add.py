# @triton
# @in  input: (N,)
# @in  other: (N,)
# @out (N,)
def add(input, other, alpha=1):
    return input + alpha * other
