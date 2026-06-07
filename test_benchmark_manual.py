#!/usr/bin/env python3
"""
Test script to run the TritonBench benchmark manually with a specific piece of code.

Usage:
    python3 test_benchmark_manual.py <operator_name> <generated_code_file>
    
Example:
    python3 test_benchmark_manual.py exp_mean debug/translations/2026-06-06_21-10-09_exp_mean/08_extracted.py
"""

import json
import sys
from pathlib import Path

# Read operator name
operator_name = sys.argv[1] if len(sys.argv) > 1 else "exp_mean"

# Read generated code file
code_file = sys.argv[2] if len(sys.argv) > 2 else f"debug/translations/2026-06-06_21-10-09_{operator_name}/08_extracted.py"

if not Path(code_file).exists():
    # Try alternative path
    code_file = f"debug/translations/2026-06-06_21-10-09_{operator_name}/10_final.py"

generated_code = Path(code_file).read_text()

# Read instruction from TritonBench dataset
from datasets.tritonbench.registry import get_registry
registry = get_registry()
entry = registry.get_entry(operator_name)

if not entry:
    print(f"Error: Operator '{operator_name}' not found in TritonBench dataset")
    sys.exit(1)

instruction = entry["instruction"]

print(f"Testing benchmark for: {operator_name}")
print(f"Code file: {code_file}")
print(f"Code length: {len(generated_code)} chars")
print(f"Instruction length: {len(instruction)} chars")
print("=" * 60)

# Run the benchmark via Modal
from backends.modal.app import benchmark_app
from backends.modal.jobs.bench_evaluation_single import bench_evaluation_single

print("\nCalling bench_evaluation_single via Modal...")
print("This may take a few minutes...")
print()

# Call the Modal function within the app context
with benchmark_app.run():
    result = bench_evaluation_single.remote(
        operator_name=operator_name,
        generated_code=generated_code,
        instruction=instruction,
    )

print("\n" + "=" * 60)
print("BENCHMARK RESULT:")
print("=" * 60)
print(json.dumps(result, indent=2, default=str))

# Check if we have logs
if "logs" in result:
    print("\n" + "=" * 60)
    print("BENCHMARK LOGS:")
    print("=" * 60)
    for log in result["logs"]:
        print(log)

# Check speedup specifically
speedup = result.get("speedup")
if speedup is not None:
    print(f"\n✓ Speedup: {speedup:.2f}x")
    if speedup > 1.0:
        print("✓ Triton is FASTER than PyTorch")
    else:
        print("✗ Triton is SLOWER than PyTorch")
else:
    print("\n✗ Speedup: N/A (could not measure)")
    print("Possible reasons:")
    print("  - The efficiency script didn't find a 'speed up:' line")
    print("  - The generated code crashed during performance testing")
    print("  - The benchmark input sizes are too small for reliable measurement")
    print("  - The generated code is extremely slow and timed out")
