"""
Smart Evaluator — elige la estrategia de evaluación correcta.

Lógica de matching (en orden de prioridad):

  1. Si el nombre de la función del usuario está en TritonBench, confirma con
     las operaciones torch internas:
     - Si todas las ops específicas detectadas son substrings del nombre
       → usa TritonBench con ese nombre (ej: gelu_std → match gelu_std).
     - Si una op específica contradice el nombre (no es substring)
       → usa esa op en su lugar (ej: función "add" que usa matmul → match matmul).
     - Si no hay ops específicas → usa el nombre de la función (ops ambiguas
       como add/mul/sub/div no contradicen).

  2. Si el nombre NO está en TritonBench, busca una única op específica
     detectada en el código. Si hay exactamente una → úsala.

  3. Si hay ambigüedad (múltiples ops específicas) o ninguna op confirma el
     match, cae a compare_with_user contra el PyTorch original del usuario.

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

    Reglas (en orden de prioridad):
    1. Normaliza torch_op_names → short names (sin prefijo "torch.")
    2. Si function_name está en TritonBench:
       a) Si hay ops específicas → verificar que alguna sea substring de function_name.
          - Si al menos una op específica es substring → match confirmado
          - Si ninguna es substring → hay contradicciones
            - Si exactamente UNA contradicción → usar esa op
            - Si múltiples contradicciones → ambiguo, None
            - Si NO hay ops específicas → usar function_name (ops ambiguas no contradicen)
       b) Si no hay ops en bench → confirmar con function_name
    3. Si function_name NO está en bench:
       - Si hay exactamente UNA op específica en bench → úsala
       - Si hay múltiples ops específicas → ambiguo, None
       - Si solo ops ambiguas → None
    """
    # Normalizar: "torch.add" → "add", "torch.nn.functional.gelu" → "gelu"
    short_ops = {op.split(".")[-1] for op in torch_op_names}
    bench_matches = short_ops & registry.operator_names()
    specific_matches = bench_matches - _AMBIGUOUS_OPS

    # ── 1. function_name está en TritonBench ──────────────────────────────────
    if registry.is_bench_operator(function_name):
        # Check if ANY specific op is a substring of function_name.
        # This handles composite operators (e.g., batch_norm in silu_batch_norm)
        # and single ops (e.g., gelu in gelu).
        matching_ops = {op for op in specific_matches if op in function_name}
        if matching_ops:
            return function_name

        # No specific op matches function_name → check for contradictions
        real_contradictions = {op for op in specific_matches if op not in function_name}
        if real_contradictions:
            if len(real_contradictions) == 1:
                return real_contradictions.pop()
            # Múltiples contradicciones → ambiguo, compara con usuario
            logger.debug(
                f"[smart_evaluate] function_name='{function_name}' contradicho por "
                f"múltiples ops: {real_contradictions} → compare_with_user"
            )
            return None

        # No specific contradictions → use function_name (ambiguous ops don't contradict)
        return function_name

    # ── 2. function_name NO está en TritonBench ─────────────────────────────
    # Fallback a la lógica original: match por ops internos
    if len(specific_matches) == 1:
        return specific_matches.pop()

    if len(specific_matches) > 1:
        # Múltiples ops específicas en bench → función fusionada custom
        # Si el function_name es uno de ellos, ya se hubiera manejado arriba
        # (aquí function_name NO está en bench)
        logger.debug(
            f"[smart_evaluate] Múltiples ops específicas en bench: {specific_matches}. "
            f"function_name='{function_name}' no confirma ninguna → compare_with_user"
        )
        return None

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
