"""
Pipeline Runner — wrapper around TranslationPipeline that integrates with the service DB.
"""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from orchestration.translation_pipeline import TranslationPipeline
from service.core.config import DEFAULT_MODEL
from service.db import crud

logger = logging.getLogger(__name__)


def _build_concrete_dims(dims_dict: Optional[dict]) -> dict[str, int]:
    """Convert dims dict to the format expected by TranslationPipeline."""
    if not dims_dict:
        return {}
    result = {}
    for k, v in dims_dict.items():
        try:
            result[str(k)] = int(v)
        except (ValueError, TypeError):
            logger.warning(f"Ignoring invalid dim: {k}={v}")
    return result


def run_translation(
    db: Session,
    job_id: str,
    source_code: str,
    provider: str,
    model: Optional[str] = None,
    dims: Optional[dict] = None,
    call_site_code: Optional[str] = None,
) -> dict:
    """
    Execute the full translation pipeline and persist results to the DB.

    GPU validation is always enabled via Modal.

    Returns a dict with the final result (matches TranslateResponse shape).
    """
    model = model or DEFAULT_MODEL
    concrete_dims = _build_concrete_dims(dims)

    # Mark job as running
    crud.update_job_status(db, job_id, "running")

    errors: list[str] = []
    validation_result: Optional[dict] = None
    gpu_validation_result: Optional[dict] = None
    generated_code: Optional[str] = None
    run_id: Optional[str] = None
    extracted_shapes: Optional[dict] = None

    try:
        # Ensure NVIDIA_API_KEY is available in the environment for the provider
        if "NVIDIA_API_KEY" not in os.environ:
            raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")

        pipeline = TranslationPipeline(
            provider_name=provider,
            model_name=model,
            modal_validate=True,  # GPU validation is always enabled
            concrete_dims=concrete_dims,
            call_site_code=call_site_code or "",
        )

        # Write source to a temp file so the pipeline can parse it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(source_code)
            tmp_path = tmp.name

        try:
            ctx = pipeline.run(file_path=tmp_path, source_code=source_code)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        run_id = ctx.run_id
        generated_code = ctx.generated_code

        # Validation result
        if ctx.validation_result:
            validation_result = {
                "passed": ctx.validation_result.passed,
                "errors": ctx.validation_result.errors,
                "warnings": ctx.validation_result.warnings,
            }

        # GPU validation result
        if ctx.gpu_validation_result:
            gpu_validation_result = {
                "compilation_pass": ctx.gpu_validation_result.compilation_pass,
                "execution_pass": ctx.gpu_validation_result.execution_pass,
                "errors": ctx.gpu_validation_result.errors,
                "output_shape": ctx.gpu_validation_result.output_shape,
                "device": ctx.gpu_validation_result.device,
                "pytorch_time_ms": ctx.gpu_validation_result.pytorch_time_ms,
            }

        # Extracted shapes from call site execution
        if ctx.shape_extraction_result and ctx.shape_extraction_result.shapes:
            extracted_shapes = ctx.shape_extraction_result.shapes

        # Collect any pipeline-stage errors from debug artifacts if available
        if not generated_code:
            errors.append("Pipeline completed but no code was generated.")

        status = "completed"

    except Exception as exc:
        logger.exception(f"Translation failed for job {job_id}")
        status = "failed"
        errors.append(f"{type(exc).__name__}: {exc}")

    # Persist everything
    crud.save_job_result(
        db,
        job_id=job_id,
        run_id=run_id,
        generated_code=generated_code,
        validation_json=validation_result,
        gpu_validation_json=gpu_validation_result,
        extracted_shapes_json=extracted_shapes,
        errors=errors if errors else None,
    )
    crud.update_job_status(db, job_id, status)

    # Also create a Kernel record if we have generated code
    if generated_code:
        func_name = _guess_function_name(source_code)
        crud.create_kernel(
            db,
            job_id=job_id,
            function_name=func_name,
            source_code=source_code,
            generated_code=generated_code,
        )

    return {
        "job_id": job_id,
        "status": status,
        "provider": provider,
        "model": model,
        "run_id": run_id,
        "source_code": source_code,
        "generated_code": generated_code,
        "validation": validation_result or {"passed": False, "errors": [], "warnings": []},
        "gpu_validation": gpu_validation_result,
        "errors": errors,
    }


def _guess_function_name(source_code: str) -> str:
    """Quick heuristic to extract the function name from source."""
    import re
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", source_code)
    return m.group(1) if m else "unknown"


def run_evaluation(
    db: Session,
    job_id: str,
    dims: Optional[dict] = None,
) -> dict:
    """
    Run numerical evaluation of a previously translated kernel.
    Reuses the generated code stored in the DB.
    """
    job = crud.get_job(db, job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    if not job.generated_code:
        raise ValueError(f"Job {job_id} has no generated code to evaluate.")

    # Write source and generated to temp files for the evaluator
    concrete_dims = _build_concrete_dims(dims)
    
    # We need to locate the debug artifacts or reconstruct the files
    # The evaluator expects two .py files: original and generated
    # We'll write them to a temp directory
    import tempfile
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = Path(tmpdir) / "original.py"
        generated_path = Path(tmpdir) / "generated.py"
        original_path.write_text(job.source_code or "", encoding="utf-8")
        generated_path.write_text(job.generated_code, encoding="utf-8")

        try:
            from evaluation.translate_evaluator import run_local_evaluation
            result = run_local_evaluation(
                original_path=original_path,
                generated_path=generated_path,
                concrete_dims=concrete_dims if concrete_dims else None,
            )
        except Exception as exc:
            logger.exception(f"Evaluation failed for job {job_id}")
            return {
                "job_id": job_id,
                "accuracy_pass": False,
                "max_error": None,
                "speedup": None,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }

    # Wrap in a consistent shape
    return {
        "job_id": job_id,
        "accuracy_pass": result.get("accuracy_pass", False),
        "max_error": result.get("max_error"),
        "speedup": result.get("speedup"),
        "errors": result.get("errors", []),
    }
