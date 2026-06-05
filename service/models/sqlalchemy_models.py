"""
SQLAlchemy ORM models for the service.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship

from service.db.database import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    type = Column(String(20), nullable=False)          # 'translate' | 'evaluate'
    status = Column(String(20), nullable=False)          # 'pending' | 'running' | 'completed' | 'failed'
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=_utc_now)
    completed_at = Column(DateTime, nullable=True)

    # Inputs (JSON strings)
    source_code = Column(Text, nullable=True)
    dims = Column(Text, nullable=True)
    run_id = Column(String(100), nullable=True)

    # Outputs (JSON strings)
    generated_code = Column(Text, nullable=True)
    validation_json = Column(Text, nullable=True)
    gpu_validation_json = Column(Text, nullable=True)
    errors = Column(Text, nullable=True)

    # Relationship
    kernel = relationship("Kernel", back_populates="job", uselist=False)


class Kernel(Base):
    __tablename__ = "kernels"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    function_name = Column(String(200), nullable=True)
    source_code = Column(Text, nullable=True)
    generated_code = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utc_now)

    job = relationship("Job", back_populates="kernel")
