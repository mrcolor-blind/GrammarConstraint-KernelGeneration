"""
Smart Evaluator — elige la estrategia de evaluación correcta:

  • Operador en TritonBench  →  bench_evaluation_single (call_acc + exec_acc + speedup)
  • Operador custom           →  compare_with_user     (accuracy vs PyTorch del usuario)

Uso:
    from evaluation.smart_evaluator import smart_evaluate

    result = smart_evaluate(
        function_name="gelu",
        generated_code="import triton...",
        original_code="def gelu(input): ...",   # solo para path custom
        concrete_dims_str="N=128",
        extracted_shapes_json="",
        speedup_threshold=1.0,
    )
    # result["strategy"] == "tritonbench" | "user_comparison"
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def smart_evaluate(
    function_name: str,
    generated_code: str,
    original_code: str = "",
    concrete_dims_str: str = "",
    extracted_shapes_json: str = "",
    speedup_threshold: float = 1.0,
) -> dict:
    """
    Evalúa el kernel generado con la estrategia más apropiada.

    Retorna un dict unificado con al menos:
        strategy, compilation_pass, accuracy_pass, speedup, errors
    """
    from datasets.tritonbench.registry import get_registry

    registry = get_registry()

    if registry.is_bench_operator(function_name):
        logger.info(
            f"[smart_evaluate] '{function_name}' encontrado en TritonBench "
            f"→ usando bench_evaluation_single"
        )
        return _run_bench(function_name, generated_code, registry)
    else:
        logger.info(
            f"[smart_evaluate] '{function_name}' NO está en TritonBench "
            f"→ usando compare_with_user"
        )
        return _run_user_comparison(
            original_code=original_code,
            generated_code=generated_code,
            concrete_dims_str=concrete_dims_str,
            extracted_shapes_json=extracted_shapes_json,
            speedup_threshold=speedup_threshold,
        )


def _run_bench(function_name: str, generated_code: str, registry) -> dict:
    """Llama a bench_evaluation_single vía Modal."""
    entry = registry.get_entry(function_name)
    if entry is None:
        return {
            "strategy": "tritonbench",
            "compilation_pass": False,
            "accuracy_pass": False,
            "speedup": None,
            "errors": [f"No dataset entry found for '{function_name}'"],
        }

    try:
        from backends.modal.jobs.bench_evaluation_single import bench_evaluation_single
        result = bench_evaluation_single.remote(
            operator_name=function_name,
            generated_code=generated_code,
            instruction=entry["instruction"],
        )
        result["strategy"] = "tritonbench"
        return result
    except Exception as e:
        logger.error(f"bench_evaluation_single failed: {e}")
        return {
            "strategy": "tritonbench",
            "compilation_pass": False,
            "accuracy_pass": False,
            "speedup": None,
            "errors": [f"{type(e).__name__}: {e}"],
        }


def _run_user_comparison(
    original_code: str,
    generated_code: str,
    concrete_dims_str: str,
    extracted_shapes_json: str,
    speedup_threshold: float,
) -> dict:
    """Llama a compare_with_user vía Modal."""
    try:
        from backends.modal.jobs.compare_with_user import compare_with_user
        result = compare_with_user.remote(
            original_code=original_code,
            generated_code=generated_code,
            concrete_dims_str=concrete_dims_str,
            extracted_shapes_json=extracted_shapes_json,
            speedup_threshold=speedup_threshold,
        )
        result["strategy"] = "user_comparison"
        return result
    except Exception as e:
        logger.error(f"compare_with_user failed: {e}")
        return {
            "strategy": "user_comparison",
            "compilation_pass": False,
            "accuracy_pass": False,
            "speedup": None,
            "errors": [f"{type(e).__name__}: {e}"],
        }
