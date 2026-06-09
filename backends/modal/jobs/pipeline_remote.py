"""
Pipeline Remote — corre el TranslationPipeline completo dentro de un
contenedor Modal con GPU T4.

Ventajas vs correr localmente:
  • Shape extraction usa device='cuda' — no más workaround de CPU
  • GPU_VALIDATE y COMPARE hacen fan-out directo a otras funciones Modal
    (sin subprocess, sin problemas de autenticación)
  • El modelo de generación se llama desde Modal → misma red, más rápido
"""

import json
import sys
import tempfile
from pathlib import Path

import modal

sys.path.append("/root/project")

from backends.modal.app import benchmark_app, volume
from backends.modal.jobs.bench_evaluation_single import bench_evaluation_single  # registro
from backends.modal.jobs.compare_with_user import compare_with_user              # registro
from backends.modal.jobs.translate_validation import translate_validation         # registro

DATA_DIR = "/data"


@benchmark_app.function(
    gpu="T4",
    timeout=60 * 30,       # 30 min max por pipeline
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("triton-grammar-constrains")],
    include_source=True,
)
def run_pipeline_remote(
    source_code: str,
    call_site_code: str = "",
    provider_name: str = "nvidia",
    model_name: str = "qwen/qwen3.5-397b-a17b",
    concrete_dims_json: str = "{}",
    speedup_threshold: float = 1.0,
    compare: bool = False,
    modal_validate: bool = True,
) -> dict:
    """
    Corre el pipeline completo (parse→generate→validate→compare) en Modal.

    Dentro de este contexto:
      • CUDA disponible → shape extraction con device='cuda'
      • translate_validation.remote() / compare_with_user.remote() funcionan
        vía fan-out (no subprocess)
    """
    import os
    concrete_dims = json.loads(concrete_dims_json)

    # Asegurar que la API key esté disponible
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if nvidia_key:
        os.environ["NVIDIA_API_KEY"] = nvidia_key

    from orchestration.translation_pipeline import TranslationPipeline
    from utils import debug_logger

    # Apuntar debug_logger al volumen Modal → todos los artefactos (01…10) se guardan ahí
    debug_root = Path(DATA_DIR) / "translations"
    debug_logger.set_debug_root(debug_root)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(source_code)
        source_path = f.name

    pipeline = TranslationPipeline(
        provider_name=provider_name,
        model_name=model_name,
        modal_validate=modal_validate,
        compare_with_user=compare,
        speedup_threshold=speedup_threshold,
        concrete_dims=concrete_dims,
        call_site_code=call_site_code,
        debug_root=debug_root,
    )

    ctx = pipeline.run(
        file_path=source_path,
        source_code=source_code,
        call_site_code=call_site_code,
    )

    volume.commit()  # flush todos los artefactos al volumen

    # Serializar resultado
    result = {
        "run_id": ctx.run_id,
        "generated_code": ctx.generated_code,
        "validation": None,
        "gpu_validation": None,
        "user_comparison": None,
        "shape_extraction": None,
        "errors": [],
    }

    if ctx.validation_result:
        result["validation"] = {
            "passed": ctx.validation_result.passed,
            "errors": ctx.validation_result.errors,
            "warnings": ctx.validation_result.warnings,
        }

    if ctx.gpu_validation_result:
        gvr = ctx.gpu_validation_result
        result["gpu_validation"] = {
            "compilation_pass": gvr.compilation_pass,
            "execution_pass": gvr.execution_pass,
            "output_shape": gvr.output_shape,
            "device": gvr.device,
            "errors": gvr.errors,
        }

    if ctx.user_comparison_result:
        ucr = ctx.user_comparison_result
        result["user_comparison"] = {
            "strategy": getattr(ucr, "strategy", "user_comparison"),
            "compilation_pass": ucr.compilation_pass,
            "accuracy_pass": ucr.accuracy_pass,
            "max_diff": ucr.max_diff,
            "speedup": ucr.speedup,
            "ref_time_ms": ucr.ref_time_ms,
            "gen_time_ms": ucr.gen_time_ms,
            "suggest_replacement": ucr.suggest_replacement,
            "reason": ucr.reason,
            "errors": ucr.errors,
        }

    if ctx.shape_extraction_result:
        ser = ctx.shape_extraction_result
        result["shape_extraction"] = {
            "success": ser.success,
            "shapes": ser.shapes if ser.success else {},
            "error": ser.error,
        }

    return result
