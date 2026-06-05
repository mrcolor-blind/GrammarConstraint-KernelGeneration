"""
Modal GPU job for evaluating translated Triton kernels.
"""

import json
import sys
from pathlib import Path

import modal
import torch

sys.path.append("/root/project")

from backends.modal.app import benchmark_app, volume
from evaluation.translate_evaluator import run_local_evaluation

DATA_DIR = "/data"


@benchmark_app.function(
    gpu="T4",
    timeout=60 * 30,
    volumes={DATA_DIR: volume},
)
def translate_evaluation(
    run_id: str,
    concrete_dims_str: str = "",
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> dict:
    """
    Evaluate a generated translation inside a Modal GPU container.
    *run_id* corresponds to the debug/translations/<run_id>/ folder.
    """
    run_dir = Path(DATA_DIR) / "translations" / run_id
    if not run_dir.exists():
        return {
            "accuracy_pass": False,
            "error": f"Run directory not found: {run_dir}",
        }

    input_file = run_dir / "01_input.py"
    generated_file = run_dir / "10_final.py"

    if not input_file.exists():
        return {
            "accuracy_pass": False,
            "error": f"Missing original file: {input_file}",
        }
    if not generated_file.exists():
        return {
            "accuracy_pass": False,
            "error": f"Missing generated file: {generated_file}",
        }

    # Parse concrete dims
    concrete_dims = {}
    if concrete_dims_str:
        for pair in concrete_dims_str.split(","):
            if "=" in pair:
                k, v = pair.split("=")
                concrete_dims[k.strip()] = int(v.strip())

    try:
        result = run_local_evaluation(
            original_path=input_file,
            generated_path=generated_file,
            concrete_dims=concrete_dims,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )
    except Exception as e:
        return {
            "accuracy_pass": False,
            "error": f"Evaluation failed: {e}",
        }

    # Persist result back to debug dir
    result_file = run_dir / "evaluation_result.json"
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()

    return result
