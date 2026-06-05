from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Parameter:
    """Represents a function parameter."""
    name: str
    kind: str  # e.g. 'POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY'
    default: Any = None
    annotation: Optional[str] = None
    shape: Optional[str] = None  # populated by shape_resolver


@dataclass
class OpNode:
    """Represents a single operation (function call) inside the user function."""
    op_name: str  # fully-qualified name, e.g. "torch.matmul"
    torch_path: str  # resolved path if applicable
    input_vars: list[str]  # e.g. ["x", "weight.T"]
    output_var: str  # e.g. "z"
    kwargs: dict = field(default_factory=dict)
    lineno: int = 0
    shape: Optional[str] = None  # resolved later by shape_resolver


@dataclass
class OperationGraph:
    """AST-derived description of the user function."""
    function_name: str
    signature: str
    parameters: list[Parameter]
    operations: list[OpNode]
    output_var: str
    source_code: str = ""


@dataclass
class ParamDesc:
    """Describes a parameter of a torch operator."""
    name: str
    type_str: Optional[str] = None
    default: Any = None
    required: bool = True
    description: str = ""


@dataclass
class OpContext:
    """Rich context for a single torch operator."""
    op_name: str
    source: str  # "tritonbench_json" | "torch_docstring" | "inspect_signature" | "name_only"
    confidence: str  # "high" | "medium" | "low"
    functional_description: str = ""
    math_formula: Optional[str] = None
    signature: str = ""
    parameters: list[ParamDesc] = field(default_factory=list)
    shapes_info: Optional[str] = None
    broadcasting: Optional[str] = None
    edge_cases: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class FusedGroup:
    """A group of operations fused into a single Triton kernel."""
    group_id: int
    operations: list[OpNode]
    fused_name: str
    input_shapes: dict[str, str]
    output_shape: str
    reasoning: str = ""


@dataclass
class FusionPlan:
    """The full fusion plan for an OperationGraph."""
    groups: list[FusedGroup]
    strategy: str = "auto"


@dataclass
class StageResult:
    """Result of a single pipeline stage attempt."""
    success: bool
    data: Any = None
    error: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of static validation on generated Triton code."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class GpuValidationResult:
    """Result of GPU compilation + execution smoke test on Modal."""
    compilation_pass: bool
    execution_pass: bool
    errors: list[str] = field(default_factory=list)
    output_shape: Optional[str] = None
    device: Optional[str] = None


@dataclass
class PipelineContext:
    """Mutable context passed through every pipeline stage."""
    source_code: str = ""
    file_path: str = ""
    run_id: str = ""
    operation_graph: Optional[OperationGraph] = None
    fusion_plan: Optional[FusionPlan] = None
    contexts: dict[str, OpContext] = field(default_factory=dict)
    generated_code: str = ""
    raw_responses: list[str] = field(default_factory=list)
    prompt_messages: list[dict] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    gpu_validation_result: Optional[GpuValidationResult] = None
    attempt_counts: dict[str, int] = field(default_factory=dict)
    evaluation_result: Optional[dict] = None
