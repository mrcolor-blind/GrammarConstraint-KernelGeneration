"""
CRUD operations for Job and Kernel.
"""

import json
from typing import Optional

from sqlalchemy.orm import Session

from service.models.sqlalchemy_models import Job, Kernel


def create_job(
    db: Session,
    job_type: str,
    status: str = "pending",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    source_code: Optional[str] = None,
    dims: Optional[dict] = None,
) -> Job:
    db_job = Job(
        type=job_type,
        status=status,
        provider=provider,
        model=model,
        source_code=source_code,
        dims=json.dumps(dims) if dims else None,
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    return db.query(Job).filter(Job.id == job_id).first()


def list_jobs(
    db: Session,
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Job], int]:
    query = db.query(Job)
    if status:
        query = query.filter(Job.status == status)
    if job_type:
        query = query.filter(Job.type == job_type)
    total = query.count()
    items = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    return items, total


def update_job_status(db: Session, job_id: str, status: str):
    job = get_job(db, job_id)
    if job:
        job.status = status
        db.commit()
        db.refresh(job)
    return job


def save_job_result(
    db: Session,
    job_id: str,
    run_id: Optional[str] = None,
    generated_code: Optional[str] = None,
    validation_json: Optional[dict] = None,
    gpu_validation_json: Optional[dict] = None,
    comparison_json: Optional[dict] = None,
    errors: Optional[list] = None,
):
    job = get_job(db, job_id)
    if not job:
        return None
    if run_id is not None:
        job.run_id = run_id
    if generated_code is not None:
        job.generated_code = generated_code
    if validation_json is not None:
        job.validation_json = json.dumps(validation_json)
    if gpu_validation_json is not None:
        job.gpu_validation_json = json.dumps(gpu_validation_json)
    if comparison_json is not None:
        job.comparison_json = json.dumps(comparison_json)
    if errors is not None:
        job.errors = json.dumps(errors)
    db.commit()
    db.refresh(job)
    return job


def create_kernel(
    db: Session,
    job_id: str,
    function_name: Optional[str] = None,
    source_code: Optional[str] = None,
    generated_code: Optional[str] = None,
) -> Kernel:
    db_kernel = Kernel(
        job_id=job_id,
        function_name=function_name,
        source_code=source_code,
        generated_code=generated_code,
    )
    db.add(db_kernel)
    db.commit()
    db.refresh(db_kernel)
    return db_kernel
