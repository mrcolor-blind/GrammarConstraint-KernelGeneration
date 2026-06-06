"""
Debug Logger — persists numbered artifacts for full traceability.
"""

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from models.domain import (
    FusionPlan,
    OperationGraph,
    OpContext,
    PipelineContext,
    StageResult,
    ValidationResult,
)


_DEBUG_ROOT = Path("debug") / "translations"


def set_debug_root(path: Path):
    """Override the default debug root (e.g., for Modal volume /data/translations)."""
    global _DEBUG_ROOT
    _DEBUG_ROOT = path


def _to_json(obj):
    """Serialize dataclasses / lists / dicts for JSON output."""
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        # strip empty / None fields for readability
        return {k: v for k, v in d.items() if v not in (None, "", [], {})}
    return obj


def make_run_id(function_name: str) -> str:
    """Unique run identifier:  timestamp + function name."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in function_name)
    return f"{ts}_{safe_name}"


def init_debug_dir(run_id: str) -> Path:
    """Create and return the artifact directory for a run."""
    d = _DEBUG_ROOT / run_id
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def persist_source_code(debug_dir: Path, source_code: str):
    (debug_dir / "01_input.py").write_text(source_code, encoding="utf-8")


def persist_call_site(debug_dir: Path, call_site_code: str):
    (debug_dir / "01b_call_site.py").write_text(call_site_code, encoding="utf-8")


def persist_parse(debug_dir: Path, graph: OperationGraph):
    (debug_dir / "02_parse.json").write_text(
        json.dumps(_to_json(graph), indent=2),
        encoding="utf-8",
    )


def persist_shapes(debug_dir: Path, graph: OperationGraph):
    shapes = {}
    for i, op in enumerate(graph.operations):
        shapes[f"op_{i}_{op.op_name}"] = {
            "input_vars": op.input_vars,
            "output_var": op.output_var,
            "shape": op.shape,
        }
    (debug_dir / "03_shapes.json").write_text(
        json.dumps(shapes, indent=2),
        encoding="utf-8",
    )


def persist_contexts(debug_dir: Path, contexts: dict[str, OpContext]):
    d = {k: _to_json(v) for k, v in contexts.items()}
    (debug_dir / "04_context.json").write_text(
        json.dumps(d, indent=2),
        encoding="utf-8",
    )


def persist_fusion(debug_dir: Path, plan: FusionPlan):
    (debug_dir / "05_fusion.json").write_text(
        json.dumps(_to_json(plan), indent=2),
        encoding="utf-8",
    )


def persist_prompt(debug_dir: Path, messages: list[dict]):
    lines = []
    for msg in messages:
        lines.append(f"## {msg['role'].upper()}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
    (debug_dir / "06_prompt.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def persist_generation_attempt(debug_dir: Path, attempt: int, raw_response: str):
    (debug_dir / f"07_generation_attempt{attempt}.txt").write_text(
        raw_response,
        encoding="utf-8",
    )


def persist_extracted_code(debug_dir: Path, code: str):
    (debug_dir / "08_extracted.py").write_text(code, encoding="utf-8")


def persist_validation(debug_dir: Path, result: ValidationResult):
    (debug_dir / "09_validation.json").write_text(
        json.dumps(_to_json(result), indent=2),
        encoding="utf-8",
    )


def persist_gpu_validation(debug_dir: Path, result):
    from models.domain import GpuValidationResult
    if isinstance(result, GpuValidationResult):
        (debug_dir / "09b_gpu_validation.json").write_text(
            json.dumps(_to_json(result), indent=2),
            encoding="utf-8",
        )


def persist_user_comparison(debug_dir: Path, result):
    from models.domain import UserComparisonResult
    if isinstance(result, UserComparisonResult):
        (debug_dir / "09c_user_comparison.json").write_text(
            json.dumps(_to_json(result), indent=2),
            encoding="utf-8",
        )


def persist_shape_extraction(debug_dir: Path, result):
    from models.domain import ShapeExtractionResult
    if isinstance(result, ShapeExtractionResult):
        (debug_dir / "03b_shape_extraction.json").write_text(
            json.dumps(_to_json(result), indent=2),
            encoding="utf-8",
        )


def persist_final_code(debug_dir: Path, code: str):
    (debug_dir / "10_final.py").write_text(code, encoding="utf-8")


def write_summary(debug_dir: Path, ctx: PipelineContext):
    """Human-readable summary of the entire run."""
    lines = [
        "# Translation Summary",
        "",
        f"**Run ID:** {ctx.run_id}",
        f"**File:** {ctx.file_path}",
        f"**Function:** {ctx.operation_graph.function_name if ctx.operation_graph else 'N/A'}",
        "",
        "## Stage Attempts",
        "",
    ]
    for stage_name, count in ctx.attempt_counts.items():
        lines.append(f"- {stage_name}: {count} attempt(s)")

    if ctx.validation_result:
        lines.append("")
        lines.append("## Validation")
        vr = ctx.validation_result
        lines.append(f"- Passed: {vr.passed}")
        if vr.errors:
            lines.append("- Errors:")
            for e in vr.errors:
                lines.append(f"  - {e}")
        if vr.warnings:
            lines.append("- Warnings:")
            for w in vr.warnings:
                lines.append(f"  - {w}")

    if ctx.gpu_validation_result:
        lines.append("")
        lines.append("## GPU Validation (Modal)")
        gvr = ctx.gpu_validation_result
        lines.append(f"- Compilation: {'PASS' if gvr.compilation_pass else 'FAIL'}")
        lines.append(f"- Execution: {'PASS' if gvr.execution_pass else 'FAIL'}")
        if gvr.output_shape:
            lines.append(f"- Output shape: {gvr.output_shape}")
        if gvr.device:
            lines.append(f"- Device: {gvr.device}")
        if gvr.errors:
            lines.append("- Errors:")
            for e in gvr.errors:
                lines.append(f"  - {e}")

    if ctx.shape_extraction_result:
        lines.append("")
        lines.append("## Shape Extraction")
        ser = ctx.shape_extraction_result
        lines.append(f"- Success: {'PASS' if ser.success else 'FAIL'}")
        if ser.error:
            lines.append(f"- Error: {ser.error}")
        if ser.shapes:
            lines.append("- Extracted shapes:")
            for name, info in ser.shapes.items():
                if "shape" in info:
                    lines.append(f"  - {name}: {info['shape']} (dtype={info.get('dtype', 'N/A')}, device={info.get('device', 'N/A')})")
                elif "value" in info:
                    lines.append(f"  - {name}: {info['value']} ({info.get('type', 'N/A')})")

    if ctx.user_comparison_result:
        lines.append("")
        lines.append("## User Comparison (Triton vs. PyTorch)")
        ucr = ctx.user_comparison_result
        lines.append(f"- Compilation: {'PASS' if ucr.compilation_pass else 'FAIL'}")
        lines.append(f"- Accuracy: {'PASS' if ucr.accuracy_pass else 'FAIL'}")
        if ucr.max_diff is not None:
            lines.append(f"- Max diff: {ucr.max_diff:.6e}")
        if ucr.speedup is not None:
            lines.append(f"- Speedup: {ucr.speedup:.2f}x")
        if ucr.ref_time_ms is not None:
            lines.append(f"- PyTorch time: {ucr.ref_time_ms:.2f} ms")
        if ucr.gen_time_ms is not None:
            lines.append(f"- Triton time: {ucr.gen_time_ms:.2f} ms")
        lines.append(f"- Suggest replacement: {'YES' if ucr.suggest_replacement else 'NO'}")
        if ucr.reason:
            lines.append(f"- Reason: {ucr.reason}")
        if ucr.errors:
            lines.append("- Errors:")
            for e in ucr.errors:
                lines.append(f"  - {e}")

    if ctx.evaluation_result:
        lines.append("")
        lines.append("## Evaluation")
        for k, v in ctx.evaluation_result.items():
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Artifacts")
    for f in sorted(debug_dir.iterdir()):
        lines.append(f"- {f.name}")

    (debug_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
