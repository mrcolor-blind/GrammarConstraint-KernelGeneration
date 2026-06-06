"""
Smart Evaluator — elige la estrategia de evaluación correcta.

Lógica de matching (en orden de prioridad):

  1. Busca qué operaciones torch usa la función original (torch.add, torch.gelu…).
     Normaliza: "torch.add" → "add". Si UNA SOLA de esas ops está en TritonBench
     de forma inequívoca → usa bench_evaluation_single con esa entrada.

  2. Si hay ambigüedad (múltiples ops en bench) o ninguna op confirma el match,
     cae a compare_with_user contra el PyTorch original del usuario.

Esto evita falsos positivos: una función llamada "add" que internamente
solo use torch.matmul NO se compararía contra el entry "add" de TritonBench.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Ops tan genéricas que por sí solas no identifican un operador TritonBench
_AMBIGUOUS_OPS = {"mul", "add", "sub", "div"}


def _resolve_bench_operator(
    function_name: str,
    torch_op_names: list[str],   # e.g. ["torch.mul", "torch.add"]
    registry,
) -> Optional[str]:
    """
    Devuelve el nombre del operador TritonBench a usar, o None.

    Reglas:
    1. Normaliza torch_op_names → short names (sin prefijo "torch.")
    2. Filtra las que existen en TritonBench
    3. Si hay exactamente UNA op específica (no ambigua) en bench → úsala
    4. Si la única match es ambigua (mul, add…) pero el function_name también
       está en bench → confirmar con function_name
    5. Si function_name está en bench y NO hay ops contradictorias → úsalo
    6. Caso contrario → None (compare_with_user)
    """
    # Normalizar: "torch.add" → "add", "torch.nn.functional.gelu" → "gelu"
    short_ops = {op.split(".")[-1] for op in torch_op_names}

    bench_matches = short_ops & registry.operator_names()

    # Quita ops ambiguas para la decisión primaria
    specific_matches = bench_matches - _AMBIGUOUS_OPS

    if len(specific_matches) == 1:
        # Una sola op específica identifica el operador claramente
        return specific_matches.pop()

    if len(specific_matches) > 1:
        # Múltiples ops específicas en bench → función fusionada custom
        # Si el function_name es uno de ellos, úsalo; si no, ambiguo
        if function_name in specific_matches:
            return function_name
        logger.debug(
            f"[smart_evaluate] Múltiples ops específicas en bench: {specific_matches}. "
            f"function_name='{function_name}' no confirma ninguna → compare_with_user"
        )
        return None

    # No hay ops específicas → solo ops ambiguas o ninguna
    # Usa function_name si está en bench Y las ops no lo contradicen
    if registry.is_bench_operator(function_name):
        # Verificación negativa: ¿hay alguna op en bench que CONTRADIGA el match?
        # (p.ej. función "add" pero ops son solo torch.matmul → contradice)
        contradicting = bench_matches - {function_name}
        if contradicting:
            logger.debug(
                f"[smart_evaluate] function_name='{function_name}' está en bench "
                f"pero ops contradicen: {contradicting} → compare_with_user"
            )
            return None
        return function_name

    return None


def smart_evaluate(
    function_name: str,
    generated_code: str,
    original_code: str = "",
    concrete_dims_str: str = "",
    extracted_shapes_json: str = "",
    speedup_threshold: float = 1.0,
    torch_op_names: Optional[list[str]] = None,
) -> dict:
    """
    Evalúa el kernel generado con la estrategia más apropiada.

    Args:
        torch_op_names: Lista de nombres de ops torch usadas en la función
                        original (e.g. ["torch.add", "torch.mul"]).
                        Si no se pasa, intenta extraerlos de original_code.

    Retorna dict con al menos:
        strategy, compilation_pass, accuracy_pass, speedup, errors
    """
    from datasets.tritonbench.registry import get_registry

    registry = get_registry()

    # Si no se pasan op names, intentar extraerlos del source
    if torch_op_names is None:
        torch_op_names = _extract_ops_from_source(original_code)

    bench_op = _resolve_bench_operator(function_name, torch_op_names, registry)

    if bench_op:
        logger.info(
            f"[smart_evaluate] '{function_name}' → TritonBench entry '{bench_op}' "
            f"(ops: {torch_op_names})"
        )
        return _run_bench(bench_op, generated_code, registry)
    else:
        logger.info(
            f"[smart_evaluate] '{function_name}' → compare_with_user "
            f"(ops: {torch_op_names}, no TritonBench match)"
        )
        return _run_user_comparison(
            original_code=original_code,
            generated_code=generated_code,
            concrete_dims_str=concrete_dims_str,
            extracted_shapes_json=extracted_shapes_json,
            speedup_threshold=speedup_threshold,
        )


def _extract_ops_from_source(source_code: str) -> list[str]:
    """Extrae op names de torch del source code usando el parser del pipeline."""
    if not source_code:
        return []
    try:
        from code_analysis.parser import parse_function
        graph, _ = parse_function(source_code)
        return [op.op_name for op in graph.operations]
    except Exception:
        return []


def _run_bench(bench_op: str, generated_code: str, registry) -> dict:
    """Llama a bench_evaluation_single vía Modal."""
    entry = registry.get_entry(bench_op)
    if entry is None:
        return {
            "strategy": "tritonbench",
            "compilation_pass": False,
            "accuracy_pass": False,
            "speedup": None,
            "errors": [f"No dataset entry found for '{bench_op}'"],
        }

    try:
        from backends.modal.jobs.bench_evaluation_single import bench_evaluation_single
        result = bench_evaluation_single.remote(
            operator_name=bench_op,
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
