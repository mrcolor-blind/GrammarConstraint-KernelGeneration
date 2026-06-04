"""
Translation Pipeline — orchestrates all stages from PyTorch source → Triton kernel.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from code_analysis.parser import parse_function
from code_analysis.shape_resolver import resolve_shapes
from context.resolver import ContextResolver
from fusion.planner import plan_fusion
from models.domain import (
    FusionPlan,
    OperationGraph,
    OpContext,
    PipelineContext,
    StageResult,
    ValidationResult,
)
from orchestration.pipeline_stage import PipelineStage
from prompts.builders.torch_to_triton import TorchToTritonPromptBuilder
from utils import debug_logger
from validation.validator import validate_code
from typing import Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

class ParseStage(PipelineStage):
    name = "PARSE"

    def _try(self, ctx: PipelineContext) -> StageResult:
        try:
            graph, warnings = parse_function(ctx.source_code)
            ctx.operation_graph = graph
            return StageResult(success=True, data=graph, warnings=warnings)
        except SyntaxError as e:
            return StageResult(success=False, error=f"SyntaxError: {e}")
        except ValueError as e:
            return StageResult(success=False, error=str(e))

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        if attempt == 1:
            # Strip non-standard comments and try again
            lines = []
            for line in ctx.source_code.splitlines():
                stripped = line.strip()
                # Keep standard comments and code
                if stripped.startswith("# @"):
                    # Keep triton annotations
                    lines.append(line)
                elif stripped.startswith("#") and not stripped.startswith("# @"):
                    # Skip other comments
                    continue
                else:
                    lines.append(line)
            ctx.source_code = "\n".join(lines)
        return ctx


class ShapeResolveStage(PipelineStage):
    name = "SHAPES"

    def _try(self, ctx: PipelineContext) -> StageResult:
        if ctx.operation_graph is None:
            return StageResult(success=False, error="No OperationGraph available.")
        try:
            graph, warnings = resolve_shapes(ctx.operation_graph, ctx.source_code)
            ctx.operation_graph = graph
            return StageResult(success=True, data=graph, warnings=warnings)
        except Exception as e:
            return StageResult(success=False, error=str(e))

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        if attempt == 1:
            # Try to infer shapes from type hints in the AST
            # (implemented as a no-op for MVP; could scan ast.arg.annotation)
            pass
        return ctx


class ContextResolveStage(PipelineStage):
    name = "CONTEXT"

    def __init__(self):
        super().__init__()
        self.resolver = ContextResolver()

    def _try(self, ctx: PipelineContext) -> StageResult:
        if ctx.operation_graph is None:
            return StageResult(success=False, error="No OperationGraph available.")

        contexts: dict[str, OpContext] = {}
        warnings = []

        for op in ctx.operation_graph.operations:
            op_ctx = self.resolver.resolve(op.op_name)
            contexts[op.op_name] = op_ctx
            if op_ctx.confidence == "low":
                warnings.append(
                    f"Low confidence context for {op.op_name} (source={op_ctx.source})"
                )

        ctx.contexts = contexts
        return StageResult(success=True, data=contexts, warnings=warnings)

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        # Context resolver already has internal fallback tiers.
        # No extra retry mutation needed.
        return ctx


class FusionPlannerStage(PipelineStage):
    name = "FUSION"

    def _try(self, ctx: PipelineContext) -> StageResult:
        if ctx.operation_graph is None:
            return StageResult(success=False, error="No OperationGraph available.")
        try:
            plan = plan_fusion(ctx.operation_graph)
            ctx.fusion_plan = plan
            return StageResult(success=True, data=plan)
        except Exception as e:
            return StageResult(success=False, error=str(e))

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        if attempt == 1:
            # Fallback: more aggressive fusion (just one big group)
            # This mutates the graph conceptually but we recompute in _try
            pass
        elif attempt == 2:
            # Fallback: no fusion (each op in its own group)
            pass
        return ctx


class PromptBuilderStage(PipelineStage):
    name = "PROMPT"

    def __init__(self):
        super().__init__()
        self.builder = TorchToTritonPromptBuilder()

    def _try(self, ctx: PipelineContext) -> StageResult:
        if ctx.operation_graph is None or ctx.fusion_plan is None:
            return StageResult(success=False, error="Missing graph or fusion plan.")
        try:
            messages = self.builder.build(
                graph=ctx.operation_graph,
                fusion_plan=ctx.fusion_plan,
                contexts=ctx.contexts or {},
            )
            ctx.prompt_messages = messages
            return StageResult(success=True, data=messages)
        except Exception as e:
            return StageResult(success=False, error=str(e))

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        # Prompt builder is deterministic; no retry mutation needed.
        return ctx


class GenerationStage(PipelineStage):
    name = "GENERATE"

    def __init__(self, provider_name: str, model_name: str):
        super().__init__()
        self.provider_name = provider_name
        self.model_name = model_name
        self._provider = None

    @property
    def provider(self):
        if self._provider is None:
            from models.registry.model_registry import load_provider
            self._provider = load_provider(self.provider_name)
        return self._provider

    def _try(self, ctx: PipelineContext) -> StageResult:
        if not ctx.prompt_messages:
            return StageResult(success=False, error="No prompt messages available.")
        try:
            raw_response = self.provider.generate(
                messages=ctx.prompt_messages,
                model=self.model_name,
            )
            ctx.raw_responses.append(raw_response)
            code = _extract_code(raw_response)
            ctx.generated_code = code
            return StageResult(success=True, data=code)
        except Exception as e:
            return StageResult(success=False, error=str(e))

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        if attempt == 1:
            # Append error feedback to the user message
            ctx.prompt_messages[-1]["content"] += (
                f"\n\n[FEEDBACK] Your previous attempt failed with:\n{error}\n"
                f"Please fix the issues and regenerate the code."
            )
        elif attempt == 2:
            ctx.prompt_messages[-1]["content"] += (
                "\n\n[FEEDBACK] Think carefully about each line before writing it.\n"
                "Valid Triton imports: import triton; import triton.language as tl\n"
                "Valid Triton APIs: tl.load(), tl.store(), tl.arange(), tl.dot(), "
                "tl.reduce(), tl.maximum(), tl.sqrt(), tl.exp(), tl.zeros_like()\n"
                "Do not use any API not listed above."
            )
        return ctx


class ValidationStage(PipelineStage):
    name = "VALIDATE"

    def _try(self, ctx: PipelineContext) -> StageResult:
        if not ctx.generated_code:
            return StageResult(success=False, error="No generated code available.")
        if ctx.operation_graph is None:
            return StageResult(success=False, error="No OperationGraph available.")
        try:
            result = validate_code(ctx.generated_code, ctx.operation_graph)
            ctx.validation_result = result
            return StageResult(
                success=result.passed,
                data=result,
                error="",
                warnings=result.errors + result.warnings,
            )
        except Exception as e:
            return StageResult(success=False, error=str(e))

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        if attempt == 1:
            # Try auto-fixing common import errors
            code = ctx.generated_code
            code = code.replace("import triton.lang as tl", "import triton.language as tl")
            code = code.replace("triton.lang", "triton.language")
            ctx.generated_code = code
        elif attempt == 2:
            # Generate a minimal stub (not ideal, but as last resort)
            # For MVP, we'll let the generation stage retry instead
            pass
        return ctx


class GpuValidationStage(PipelineStage):
    """GPU compilation + execution smoke test via Modal."""
    name = "GPU_VALIDATE"
    max_attempts = 1  # No repair loop; just report

    def __init__(self, run_id: str, concrete_dims: dict[str, int]):
        super().__init__()
        self.run_id = run_id
        self.concrete_dims = concrete_dims

    def _try(self, ctx: PipelineContext) -> StageResult:
        if not ctx.generated_code:
            return StageResult(success=False, error="No generated code available.")
        if ctx.operation_graph is None:
            return StageResult(success=False, error="No OperationGraph available.")

        # Lazy import to avoid Modal dependency on local runs
        try:
            from backends.modal.jobs.translate_validation import translate_validation
        except ImportError as e:
            return StageResult(
                success=False,
                error=f"Modal not available for GPU validation: {e}",
            )

        # Build input_shapes dict from parameters
        input_shapes = {}
        for p in ctx.operation_graph.parameters:
            if p.shape:
                input_shapes[p.name] = p.shape

        dims_str = ",".join(f"{k}={v}" for k, v in self.concrete_dims.items())

        try:
            result_dict = translate_validation.remote(
                generated_code=ctx.generated_code,
                function_name=ctx.operation_graph.function_name,
                param_names=[p.name for p in ctx.operation_graph.parameters],
                input_shapes=input_shapes,
                concrete_dims_str=dims_str,
            )
            from models.domain import GpuValidationResult
            result = GpuValidationResult(
                compilation_pass=result_dict.get("compilation_pass", False),
                execution_pass=result_dict.get("execution_pass", False),
                errors=result_dict.get("errors", []),
                output_shape=result_dict.get("output_shape"),
                device=result_dict.get("device"),
            )
            ctx.gpu_validation_result = result
            # GPU validation is informational; stage always "succeeds" so pipeline continues
            return StageResult(
                success=True,
                data=result,
                warnings=result.errors,
            )
        except Exception as e:
            return StageResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Remove markdown fences and extract clean Python code."""
    s = text.strip()
    match = re.search(
        r"```(?:python|py)?\s*\n(.*?)\n```",
        s,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip() + "\n"
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip() + "\n"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class TranslationPipeline:
    """Runs the full PyTorch → Triton translation pipeline."""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        modal_validate: bool = False,
        concrete_dims: Optional[dict[str, int]] = None,
        debug_root: Optional[Path] = None,
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.modal_validate = modal_validate
        self.concrete_dims = concrete_dims or {}
        self.debug_root = debug_root

    def run(self, file_path: str, source_code: Optional[str] = None) -> PipelineContext:
        """
        Execute the full pipeline.
        Returns the final PipelineContext with all artifacts and results.
        """
        if source_code is None:
            source_code = Path(file_path).read_text(encoding="utf-8")

        # Override debug root for Modal volume if provided
        if self.debug_root:
            debug_logger.set_debug_root(self.debug_root)

        run_id = debug_logger.make_run_id(
            self._guess_function_name(source_code)
        )
        debug_dir = debug_logger.init_debug_dir(run_id)

        ctx = PipelineContext(
            source_code=source_code,
            file_path=file_path,
            run_id=run_id,
        )

        debug_logger.persist_source_code(debug_dir, source_code)
        logger.info(f"[{run_id}] Starting translation pipeline for {file_path}")

        stages = [
            ParseStage(),
            ShapeResolveStage(),
            ContextResolveStage(),
            FusionPlannerStage(),
            PromptBuilderStage(),
            GenerationStage(self.provider_name, self.model_name),
            ValidationStage(),
        ]

        for stage in stages:
            logger.info(f"[{run_id}] Running stage: {stage.name}")
            result = stage.run(ctx)
            self._persist_stage_artifact(debug_dir, stage.name, ctx, result)

            if not result.success:
                logger.error(
                    f"[{run_id}] Stage {stage.name} failed after {stage.max_attempts} attempts."
                )
                debug_logger.write_summary(debug_dir, ctx)
                return ctx

        # Optional GPU validation on Modal (informational, does not block)
        if self.modal_validate:
            gpu_stage = GpuValidationStage(run_id=run_id, concrete_dims=self.concrete_dims)
            logger.info(f"[{run_id}] Running optional stage: {gpu_stage.name}")
            gpu_result = gpu_stage.run(ctx)
            self._persist_stage_artifact(debug_dir, gpu_stage.name, ctx, gpu_result)
            if gpu_result.success and ctx.gpu_validation_result:
                gvr = ctx.gpu_validation_result
                logger.info(
                    f"[{run_id}] GPU validation: compilation={'PASS' if gvr.compilation_pass else 'FAIL'}, "
                    f"execution={'PASS' if gvr.execution_pass else 'FAIL'}"
                )
            else:
                logger.warning(
                    f"[{run_id}] GPU validation stage failed: {gpu_result.error}"
                )

        # Final success
        debug_logger.persist_final_code(debug_dir, ctx.generated_code)
        debug_logger.write_summary(debug_dir, ctx)
        logger.info(f"[{run_id}] Translation complete. Artifacts in {debug_dir}")
        return ctx

    @staticmethod
    def _guess_function_name(source_code: str) -> str:
        """Quick heuristic to extract the function name from source."""
        m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", source_code)
        return m.group(1) if m else "unknown"

    def _persist_stage_artifact(
        self,
        debug_dir: Path,
        stage_name: str,
        ctx: PipelineContext,
        result: StageResult,
    ):
        """Persist intermediate artifacts after each stage."""
        if stage_name == "PARSE" and ctx.operation_graph:
            debug_logger.persist_parse(debug_dir, ctx.operation_graph)
        elif stage_name == "SHAPES" and ctx.operation_graph:
            debug_logger.persist_shapes(debug_dir, ctx.operation_graph)
        elif stage_name == "CONTEXT" and ctx.contexts:
            debug_logger.persist_contexts(debug_dir, ctx.contexts)
        elif stage_name == "FUSION" and ctx.fusion_plan:
            debug_logger.persist_fusion(debug_dir, ctx.fusion_plan)
        elif stage_name == "PROMPT" and ctx.prompt_messages:
            debug_logger.persist_prompt(debug_dir, ctx.prompt_messages)
        elif stage_name == "GENERATE":
            attempt = ctx.attempt_counts.get("GENERATE", 1)
            # Persist each raw generation attempt
            for i, raw in enumerate(ctx.raw_responses, start=1):
                debug_logger.persist_generation_attempt(debug_dir, i, raw)
            if ctx.generated_code:
                debug_logger.persist_extracted_code(debug_dir, ctx.generated_code)
        elif stage_name == "VALIDATE" and ctx.validation_result:
            debug_logger.persist_validation(debug_dir, ctx.validation_result)
        elif stage_name == "GPU_VALIDATE" and ctx.gpu_validation_result:
            debug_logger.persist_gpu_validation(debug_dir, ctx.gpu_validation_result)
