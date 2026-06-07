"""
Pydantic request/response schemas for the service API.
"""

from typing import Optional

from pydantic import BaseModel, Field

from service.core.config import DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------

class ValidationOut(BaseModel):
    passed: bool = False
    errors: list[str] = []
    warnings: list[str] = []


class GpuValidationOut(BaseModel):
    compilation_pass: bool = False
    execution_pass: bool = False
    errors: list[str] = []
    output_shape: Optional[str] = None
    device: Optional[str] = None
    logs: list[str] = []
    pytorch_time_ms: Optional[float] = None


class UserComparisonOut(BaseModel):
    """Result of comparing generated Triton kernel vs original PyTorch fn."""
    compilation_pass: bool = False
    accuracy_pass: bool = False
    max_diff: Optional[float] = None
    speedup: Optional[float] = None
    ref_time_ms: Optional[float] = None
    gen_time_ms: Optional[float] = None
    suggest_replacement: bool = False
    reason: str = ""
    errors: list[str] = []
    device: Optional[str] = None
    concrete_dims: Optional[dict] = None
    logs: list[str] = []


class TranslateRequest(BaseModel):
    source_code: str = Field(..., description="Python source code containing the function to translate")
    call_site_code: Optional[str] = Field(default=None, description="Python code that calls the function to extract real tensor shapes")
    provider: str = Field(default="nvidia-grammar", description="LLM provider name")
    model: Optional[str] = Field(default=DEFAULT_MODEL, description=f"Model identifier (default: {DEFAULT_MODEL})")
    dims: Optional[dict[str, int]] = Field(default=None, description='Concrete dimensions for symbolic shapes, e.g. {"N": 128, "D_in": 256}')
    gpu_validate: bool = Field(default=True, description="Always runs GPU compilation + execution smoke test on Modal")


class TranslateResponse(BaseModel):
    job_id: str
    status: str
    provider: str
    model: str
    run_id: Optional[str] = None
    source_code: Optional[str] = None
    generated_code: Optional[str] = None
    validation: ValidationOut
    gpu_validation: Optional[GpuValidationOut] = None
    errors: list[str] = []
    completed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    job_id: str = Field(..., description="Job ID of a previous translation")
    dims: Optional[dict[str, int]] = Field(default=None, description='Concrete dimensions for symbolic shapes')


class EvaluateResponse(BaseModel):
    job_id: str
    accuracy_pass: bool = False
    max_error: Optional[float] = None
    speedup: Optional[float] = None
    errors: list[str] = []


# ---------------------------------------------------------------------------
# Runs (listing / detail)
# ---------------------------------------------------------------------------

class JobSummary(BaseModel):
    job_id: str
    status: str
    type: str
    provider: Optional[str] = None
    model: Optional[str] = None
    run_id: Optional[str] = None
    function_name: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class JobDetail(BaseModel):
    job_id: str
    status: str
    type: str
    provider: Optional[str] = None
    model: Optional[str] = None
    run_id: Optional[str] = None
    source_code: Optional[str] = None
    call_site_code: Optional[str] = None
    generated_code: Optional[str] = None
    validation: Optional[ValidationOut] = None
    gpu_validation: Optional[GpuValidationOut] = None
    comparison_json: Optional[UserComparisonOut] = None
    errors: list[str] = []
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[JobSummary]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    version: str = "0.1.0"
