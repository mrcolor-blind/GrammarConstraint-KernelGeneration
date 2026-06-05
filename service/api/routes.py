"""
FastAPI route definitions.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

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
)
from service.core.gpu_utils import build_gpu_validation_payload
from service.core.pipeline_runner import run_evaluation, run_translation
from service.db import crud
from service.db.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


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
# GPU Validation (via Modal subprocess)
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/gpu-validate", response_model=GpuValidationOut)
def gpu_validate(job_id: str, db: Session = Depends(get_db)):
    """
    Run GPU compilation + execution validation on Modal for a previously translated job.

    This executes `modal run service/modal_gpu_validator.py` as a subprocess
    inside the Docker container. The Modal local entrypoint reads a JSON payload,
    calls `translate_validation.remote()` on a GPU in the cloud, and prints JSON.
    """
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.generated_code:
        raise HTTPException(status_code=400, detail="Job has no generated code to validate")

    # Parse dims from DB
    dims = {}
    if job.dims:
        try:
            dims = json.loads(job.dims)
        except Exception:
            pass

    # Build payload for the Modal entrypoint
    payload = build_gpu_validation_payload(
        job_id=job_id,
        generated_code=job.generated_code,
        original_source_code=job.source_code or "",
        dims=dims,
    )

    # Write payload to a temp JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        tmp_path = tmp.name

    output_path = tmp_path.replace(".json", "_output.json")

    try:
        # Execute Modal entrypoint as subprocess (runs locally in container,
        # but translate_validation.remote() runs on Modal GPU cloud)
        result = subprocess.run(
            [
                "modal",
                "run",
                "service/modal_gpu_validator.py",
                "--json-file",
                tmp_path,
                "--output-file",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes (GPU spin-up + compilation + execution)
        )

        # Read result from output file (avoids Modal stdout noise)
        output_file = Path(output_path)
        if output_file.exists():
            try:
                data = json.loads(output_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.error(f"Failed to parse Modal output file: {exc}")
                data = {"error": f"Invalid JSON in output file: {exc}"}
        else:
            stderr = result.stderr.strip()
            logger.error(f"Modal output file not found. stderr: {stderr}")
            data = {"error": f"Modal subprocess failed: {stderr[:500]}"}

        if "error" in data:
            gpu_val = GpuValidationOut(
                compilation_pass=False,
                execution_pass=False,
                errors=[data["error"]],
            )
            crud.save_job_result(db, job_id=job_id, gpu_validation_json=gpu_val.model_dump())
            return gpu_val

        # Build response
        gpu_val = GpuValidationOut(
            compilation_pass=data.get("compilation_pass", False),
            execution_pass=data.get("execution_pass", False),
            errors=data.get("errors", []),
            output_shape=data.get("output_shape"),
            device=data.get("device"),
        )

        # Persist success
        crud.save_job_result(db, job_id=job_id, gpu_validation_json=gpu_val.model_dump())
        return gpu_val

    except subprocess.TimeoutExpired:
        logger.error("Modal subprocess timed out after 600s")
        gpu_val = GpuValidationOut(
            compilation_pass=False,
            execution_pass=False,
            errors=["Modal GPU validation timed out after 10 minutes"],
        )
        crud.save_job_result(db, job_id=job_id, gpu_validation_json=gpu_val.model_dump())
        return gpu_val

    finally:
        # Clean up temp files
        Path(tmp_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


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
