"""
FastAPI route definitions.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import os
import subprocess
import tempfile
from pathlib import Path

from sqlalchemy import text

from service.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    JobDetail,
    JobListResponse,
    JobSummary,
    TranslateRequest,
    TranslateResponse,
    ValidationOut,
    GpuValidationOut,
    UserComparisonOut,
)
from service.core.gpu_utils import build_gpu_validation_payload
from service.core.pipeline_runner import run_evaluation, run_translation
from service.db import crud
from service.db.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def _modal_env() -> dict:
    """Build env dict with Modal credentials explicitly set for subprocess calls."""
    env = os.environ.copy()
    # Ensure Modal token vars are present even if inherited env is incomplete
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        val = os.environ.get(key, "")
        if val:
            env[key] = val
        else:
            logger.warning(f"Missing env var: {key}")
    return env


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------

@router.post("/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest, db: Session = Depends(get_db)):
    """
    Translate a PyTorch function to Triton.

    Runs the full pipeline (parse, shapes, context, fusion, prompt, generate, validate)
    and optionally validates on a Modal GPU.
    """
    # Create job record
    job = crud.create_job(
        db,
        job_type="translate",
        status="pending",
        provider=payload.provider,
        model=payload.model,
        source_code=payload.source_code,
        dims=payload.dims,
    )

    # Run pipeline (synchronous for MVP) — GPU validation always enabled
    result = run_translation(
        db=db,
        job_id=job.id,
        source_code=payload.source_code,
        provider=payload.provider,
        model=payload.model,
        dims=payload.dims,
    )

    # Build response
    validation = result.get("validation", {})
    gpu_val = result.get("gpu_validation")

    return TranslateResponse(
        job_id=result["job_id"],
        status=result["status"],
        provider=result["provider"],
        model=result["model"],
        run_id=result.get("run_id"),
        source_code=result.get("source_code"),
        generated_code=result.get("generated_code"),
        validation=ValidationOut(**validation),
        gpu_validation=GpuValidationOut(**gpu_val) if gpu_val else None,
        errors=result.get("errors", []),
    )


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest, db: Session = Depends(get_db)):
    """
    Numerically evaluate a previously translated kernel against the original PyTorch function.
    """
    result = run_evaluation(
        db=db,
        job_id=payload.job_id,
        dims=payload.dims,
    )
    return EvaluateResponse(**result)


# ---------------------------------------------------------------------------
# GPU Validation (direct Modal .remote() call — no subprocess)
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/gpu-validate", response_model=GpuValidationOut)
def gpu_validate(job_id: str, db: Session = Depends(get_db)):
    """
    Run GPU compilation + execution smoke test on Modal for a previously translated job.
    Calls translate_validation.remote() directly — no subprocess, no CLI auth issues.
    """
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.generated_code:
        raise HTTPException(status_code=400, detail="Job has no generated code to validate")

    dims = {}
    if job.dims:
        try:
            dims = json.loads(job.dims)
        except Exception:
            pass

    payload = build_gpu_validation_payload(
        job_id=job_id,
        generated_code=job.generated_code,
        original_source_code=job.source_code or "",
        dims=dims,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        tmp_path = tmp.name
    output_path = tmp_path.replace(".json", "_output.json")

    logs: list[str] = []
    try:
        logger.info(f"Starting Modal GPU validation for job {job_id}")
        logger.info(f"Modal command: modal run service/modal_gpu_validator.py --json-file {tmp_path} --output-file {output_path}")

        process = subprocess.Popen(
            ["modal", "run", "service/modal_gpu_validator.py",
             "--json-file", tmp_path, "--output-file", output_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd="/app", env=_modal_env(),
        )

        for line in process.stdout:
            line = line.rstrip()
            logs.append(line)
            logger.info(f"[modal-gpu] {line}")

        returncode = process.wait(timeout=600)
        if returncode != 0:
            logger.error(f"Modal subprocess exited with code {returncode}")
            logs.append(f"Modal subprocess exited with code {returncode}")

        output_file = Path(output_path)
        if output_file.exists():
            try:
                data = json.loads(output_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                data = {"error": f"Invalid JSON in output file: {exc}"}
        else:
            logger.error(f"Modal output file not found. Last logs: {' | '.join(logs[-5:])}")
            data = {"error": f"Modal subprocess failed. Last logs: {' | '.join(logs[-5:])}"}

        if "error" in data:
            gpu_val = GpuValidationOut(
                compilation_pass=False, execution_pass=False,
                errors=[data["error"]], logs=logs,
            )
            crud.save_job_result(db, job_id=job_id, gpu_validation_json=gpu_val.model_dump())
            return gpu_val

        gpu_val = GpuValidationOut(
            compilation_pass=data.get("compilation_pass", False),
            execution_pass=data.get("execution_pass", False),
            errors=data.get("errors", []),
            output_shape=data.get("output_shape"),
            device=data.get("device"),
            logs=logs,
        )
        crud.save_job_result(db, job_id=job_id, gpu_validation_json=gpu_val.model_dump())
        return gpu_val

    except subprocess.TimeoutExpired:
        logger.error("Modal subprocess timed out after 600s")
        logs.append("Modal GPU validation timed out after 10 minutes")
        gpu_val = GpuValidationOut(
            compilation_pass=False, execution_pass=False,
            errors=["Modal GPU validation timed out after 10 minutes"],
            logs=logs,
        )
        crud.save_job_result(db, job_id=job_id, gpu_validation_json=gpu_val.model_dump())
        return gpu_val
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Compare generated kernel vs original PyTorch (accuracy + speedup)
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/compare", response_model=UserComparisonOut)
def compare_kernel(job_id: str, db: Session = Depends(get_db)):
    """
    Smart evaluation: usa TritonBench si el operador está en el dataset,
    compare_with_user contra PyTorch del usuario si no lo está.
    """
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.generated_code:
        raise HTTPException(status_code=400, detail="Job has no generated code to compare")
    if not job.source_code:
        raise HTTPException(status_code=400, detail="Job has no original source code to compare against")

    dims = {}
    if job.dims:
        try:
            dims = json.loads(job.dims)
        except Exception:
            pass

    dims_str = ",".join(f"{k}={v}" for k, v in dims.items())

    from service.core.gpu_utils import _extract_function_name
    from evaluation.smart_evaluator import smart_evaluate

    function_name = _extract_function_name(job.source_code)

    try:
        result_dict = smart_evaluate(
            function_name=function_name,
            generated_code=job.generated_code,
            original_code=job.source_code,
            concrete_dims_str=dims_str,
            speedup_threshold=1.0,
        )
    except Exception as exc:
        logger.error(f"smart_evaluate failed: {exc}")
        cmp = UserComparisonOut(
            compilation_pass=False,
            accuracy_pass=False,
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        crud.save_job_result(db, job_id=job_id, comparison_json=cmp.model_dump())
        return cmp

    # Extract and log Modal remote logs
    modal_logs = result_dict.get("logs", [])
    for line in modal_logs:
        logger.info(f"[modal-compare] {line}")

    cmp = UserComparisonOut(
        compilation_pass=result_dict.get("compilation_pass", False),
        accuracy_pass=result_dict.get("accuracy_pass", False),
        max_diff=result_dict.get("max_diff"),
        speedup=result_dict.get("speedup"),
        ref_time_ms=result_dict.get("ref_time_ms"),
        gen_time_ms=result_dict.get("gen_time_ms"),
        suggest_replacement=result_dict.get("suggest_replacement", False),
        reason=result_dict.get("reason", ""),
        errors=result_dict.get("errors", []),
        device=result_dict.get("device"),
        concrete_dims=result_dict.get("concrete_dims"),
        logs=modal_logs,
    )
    crud.save_job_result(db, job_id=job_id, comparison_json=cmp.model_dump())
    return cmp


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@router.get("/runs/{job_id}", response_model=JobDetail)
def get_run(job_id: str, db: Session = Depends(get_db)):
    """Retrieve full details of a single job."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Parse JSON columns
    validation = None
    if job.validation_json:
        try:
            validation = ValidationOut(**json.loads(job.validation_json))
        except Exception:
            validation = ValidationOut(errors=["Failed to parse validation_json"])

    gpu_validation = None
    if job.gpu_validation_json:
        try:
            gpu_validation = GpuValidationOut(**json.loads(job.gpu_validation_json))
        except Exception:
            gpu_validation = None

    comparison = None
    if job.comparison_json:
        try:
            comparison = UserComparisonOut(**json.loads(job.comparison_json))
        except Exception:
            comparison = None

    errors = []
    if job.errors:
        try:
            errors = json.loads(job.errors)
        except Exception:
            errors = [job.errors]

    return JobDetail(
        job_id=job.id,
        status=job.status,
        type=job.type,
        provider=job.provider,
        model=job.model,
        run_id=job.run_id,
        source_code=job.source_code,
        generated_code=job.generated_code,
        validation=validation,
        gpu_validation=gpu_validation,
        comparison_json=comparison,
        errors=errors,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get("/runs", response_model=JobListResponse)
def list_runs(
    status: Optional[str] = Query(default=None, description="Filter by status"),
    type_: Optional[str] = Query(default=None, alias="type", description="Filter by job type"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """List jobs with optional filtering and pagination."""
    items_raw, total = crud.list_jobs(
        db, status=status, job_type=type_, limit=limit, offset=offset
    )

    items = []
    for job in items_raw:
        func_name = None
        if job.kernel:
            func_name = job.kernel.function_name
        items.append(
            JobSummary(
                job_id=job.id,
                status=job.status,
                type=job.type,
                provider=job.provider,
                model=job.model,
                run_id=job.run_id,
                function_name=func_name,
                created_at=job.created_at.isoformat() if job.created_at else None,
            )
        )

    return JobListResponse(total=total, limit=limit, offset=offset, items=items)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    """Health check — verifies DB connectivity."""
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
        logger.warning("Health check: DB connection failed")

    return HealthResponse(status="ok" if db_ok else "degraded", db_connected=db_ok)
