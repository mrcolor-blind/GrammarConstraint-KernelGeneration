"""
Quick test for _resolve_bench_operator logic.
Run with: python3 -m evaluation.test_smart_evaluator
"""

from evaluation.smart_evaluator import _resolve_bench_operator
from datasets.tritonbench.registry import get_registry

# Test cases
registry = get_registry()

test_cases = [
    # (function_name, torch_op_names, expected_result)
    ("gelu_std", ["torch.nn.functional.gelu", "torch.std"], "gelu_std"),
    ("add_gelu", ["torch.add", "torch.nn.functional.gelu"], "add_gelu"),
    ("gelu", ["torch.nn.functional.gelu", "torch.add"], "gelu"),
    ("gelu", ["torch.add", "torch.mul"], "gelu"),
    ("gelu", ["torch.matmul"], "matmul"),  # Contradiction
    ("add", ["torch.add"], "add"),
    ("add", ["torch.matmul"], "matmul"),  # Contradiction
    ("add", ["torch.mul", "torch.add"], "add"),
    ("add", ["torch.mul", "torch.matmul"], "matmul"),  # Contradiction
    ("my_custom", ["torch.nn.functional.gelu", "torch.std"], None),  # Not in bench
    ("my_custom", ["torch.nn.functional.gelu"], "gelu"),  # Not in bench, 1 specific
    ("my_custom", ["torch.add", "torch.mul"], None),  # Not in bench, ambiguous
    ("silu_batch_norm", ["torch.nn.functional.batch_norm", "torch.sigmoid", "torch.mul"], "silu_batch_norm"),  # batch_norm is substring
    ("sigmoid_batch_norm", ["torch.nn.functional.batch_norm", "torch.sigmoid", "torch.mul"], "sigmoid_batch_norm"),  # both are substrings
]

print("Testing _resolve_bench_operator:")
print("=" * 60)

all_passed = True
for fn_name, ops, expected in test_cases:
    result = _resolve_bench_operator(fn_name, ops, registry)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  {status}: {fn_name} ({ops}) -> {result!r} (expected {expected!r})")

print("=" * 60)
if all_passed:
    print("All tests passed!")
else:
    print("Some tests failed!")
