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


class ShapeExtractionStage(PipelineStage):
    """Extract exact shapes from a call site via runtime execution."""
    name = "EXTRACT_SHAPES"

    def _try(self, ctx: PipelineContext) -> StageResult:
        if ctx.operation_graph is None:
            return StageResult(success=False, error="No OperationGraph available.")
        if not ctx.call_site_code:
            return StageResult(
                success=False,
                error="No call site code provided. Use --call-site to provide a code snippet that calls the function.",
            )

        try:
            from shape_extraction.executor import (
                extract_shapes,
                format_shapes_for_prompt,
            )
            from models.domain import ShapeExtractionResult

            result = extract_shapes(
                function_code=ctx.source_code,
                call_site_code=ctx.call_site_code,
                function_name=ctx.operation_graph.function_name,
            )

            if not result.get("called"):
                error = result.get("error", "Function was never called.")
                ctx.shape_extraction_result = ShapeExtractionResult(
                    success=False,
                    error=error,
                    called=False,
                )
                return StageResult(success=False, error=error)

            if result.get("error"):
                # Called but had an error during execution
                ctx.shape_extraction_result = ShapeExtractionResult(
                    success=False,
                    error=result["error"],
                    called=True,
                )
                return StageResult(success=False, error=result["error"])

            # Success: map shapes to parameters
            extracted_shapes = result.get("shapes", {})
            ctx.shape_extraction_result = ShapeExtractionResult(
                success=True,
                shapes=extracted_shapes,
                called=True,
            )

            # Attach shapes to OperationGraph parameters
            for param in ctx.operation_graph.parameters:
                if param.name in extracted_shapes:
                    info = extracted_shapes[param.name]
                    if "shape" in info:
                        shape_tuple = tuple(info["shape"])
                        param.shape = str(shape_tuple) if len(shape_tuple) > 1 else f"({shape_tuple[0]},)"
                    elif "value" in info:
                        param.shape = str(info["value"])

            # Also attach to OpNodes
            known_shapes = {}
            for param in ctx.operation_graph.parameters:
                if param.shape:
                    known_shapes[param.name] = param.shape

            for op in ctx.operation_graph.operations:
                for i, iv in enumerate(op.input_vars):
                    base = iv.split(".")[0].split("[")[0].strip()
                    if base in known_shapes:
                        op.shape = known_shapes[base]

            warnings = []
            for param in ctx.operation_graph.parameters:
                if param.name not in extracted_shapes:
                    warnings.append(f"Shape not extracted for parameter '{param.name}'.")

            return StageResult(success=True, data=extracted_shapes, warnings=warnings)

        except Exception as e:
            return StageResult(success=False, error=str(e))

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
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

        import json, os, subprocess, tempfile
        from pathlib import Path

        # Build input_shapes dict from parameters
        input_shapes = {}
        for p in ctx.operation_graph.parameters:
            if p.shape:
                input_shapes[p.name] = p.shape

        dims_str = ",".join(f"{k}={v}" for k, v in self.concrete_dims.items())

        payload = {
            "job_id": self.run_id,
            "generated_code": ctx.generated_code,
            "original_source_code": ctx.source_code,
            "function_name": ctx.operation_graph.function_name,
            "param_names": [p.name for p in ctx.operation_graph.parameters],
            "input_shapes": input_shapes,
            "concrete_dims_str": dims_str,
        }

        # Pass Modal credentials explicitly so the subprocess can authenticate
        modal_env = os.environ.copy()
        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
            if os.environ.get(key):
                modal_env[key] = os.environ[key]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name
        output_path = tmp_path.replace(".json", "_output.json")

        try:
            proc = subprocess.run(
                ["modal", "run", "service/modal_gpu_validator.py",
                 "--json-file", tmp_path, "--output-file", output_path],
                capture_output=True, text=True, timeout=600,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=modal_env,
            )
            output_file = Path(output_path)
            if output_file.exists():
                result_dict = json.loads(output_file.read_text(encoding="utf-8"))
                if "error" in result_dict:
                    return StageResult(success=True, warnings=[result_dict["error"]])
            else:
                return StageResult(success=True, warnings=[f"Modal GPU validation failed: {proc.stderr[:300]}"])
        except subprocess.TimeoutExpired:
            return StageResult(success=True, warnings=["GPU validation timed out"])
        except Exception as exc:
            return StageResult(success=True, warnings=[str(exc)])
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

        try:
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


class CompareWithUserStage(PipelineStage):
    """Compare generated Triton kernel against user's original PyTorch code."""
    name = "COMPARE_WITH_USER"
    max_attempts = 1  # No repair loop; just report

    def __init__(self, concrete_dims: dict[str, int], speedup_threshold: float = 1.1):
        super().__init__()
        self.concrete_dims = concrete_dims
        self.speedup_threshold = speedup_threshold

    def _try(self, ctx: PipelineContext) -> StageResult:
        if not ctx.generated_code:
            return StageResult(success=False, error="No generated code available.")
        if not ctx.source_code:
            return StageResult(success=False, error="No original source code available.")

        import json
        from evaluation.smart_evaluator import smart_evaluate

        dims_str = ",".join(f"{k}={v}" for k, v in self.concrete_dims.items())
        extracted_shapes = {}
        if ctx.shape_extraction_result and ctx.shape_extraction_result.shapes:
            extracted_shapes = ctx.shape_extraction_result.shapes

        function_name = (
            ctx.operation_graph.function_name
            if ctx.operation_graph else "unknown"
        )
        torch_op_names = (
            [op.op_name for op in ctx.operation_graph.operations]
            if ctx.operation_graph else []
        )

        try:
            result_dict = smart_evaluate(
                function_name=function_name,
                generated_code=ctx.generated_code,
                original_code=ctx.source_code,
                concrete_dims_str=dims_str,
                extracted_shapes_json=json.dumps(extracted_shapes) if extracted_shapes else "",
                speedup_threshold=self.speedup_threshold,
                torch_op_names=torch_op_names,
            )
            from models.domain import UserComparisonResult
            result = UserComparisonResult(
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
            )
            ctx.user_comparison_result = result
            # Comparison is informational; stage always "succeeds" so pipeline continues
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
        compare_with_user: bool = False,
        speedup_threshold: float = 1.1,
        concrete_dims: Optional[dict[str, int]] = None,
        debug_root: Optional[Path] = None,
        call_site_code: str = "",
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.modal_validate = modal_validate
        self.compare_with_user = compare_with_user
        self.speedup_threshold = speedup_threshold
        self.concrete_dims = concrete_dims or {}
        self.debug_root = debug_root
        self.call_site_code = call_site_code

    def run(self, file_path: str, source_code: Optional[str] = None, call_site_code: Optional[str] = None) -> PipelineContext:
        """
        Execute the full pipeline.
        Returns the final PipelineContext with all artifacts and results.
        """
        if source_code is None:
            source_code = Path(file_path).read_text(encoding="utf-8")
        
        if call_site_code is None:
            call_site_code = self.call_site_code

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
            call_site_code=call_site_code,
        )

        debug_logger.persist_source_code(debug_dir, source_code)
        if call_site_code:
            debug_logger.persist_call_site(debug_dir, call_site_code)
        logger.info(f"[{run_id}] Starting translation pipeline for {file_path}")

        # Determine which shape stage to use
        if call_site_code:
            shape_stage = ShapeExtractionStage()
        else:
            shape_stage = ShapeResolveStage()

        stages = [
            ParseStage(),
            shape_stage,
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

        # Optional: Compare generated Triton against user's PyTorch code
        if self.compare_with_user:
            comp_stage = CompareWithUserStage(
                concrete_dims=self.concrete_dims,
                speedup_threshold=self.speedup_threshold,
            )
            logger.info(f"[{run_id}] Running optional stage: {comp_stage.name}")
            comp_result = comp_stage.run(ctx)
            self._persist_stage_artifact(debug_dir, comp_stage.name, ctx, comp_result)
            if comp_result.success and ctx.user_comparison_result:
                ucr = ctx.user_comparison_result
                logger.info(
                    f"[{run_id}] User comparison: compilation={'PASS' if ucr.compilation_pass else 'FAIL'}, "
                    f"accuracy={'PASS' if ucr.accuracy_pass else 'FAIL'}, "
                    f"speedup={ucr.speedup:.2f}x " if ucr.speedup is not None else f"speedup=N/A, "
                    f"suggest={ucr.suggest_replacement}"
                )
            else:
                logger.warning(
                    f"[{run_id}] User comparison stage failed: {comp_result.error}"
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
        elif stage_name == "EXTRACT_SHAPES" and ctx.shape_extraction_result:
            debug_logger.persist_shape_extraction(debug_dir, ctx.shape_extraction_result)
            if ctx.operation_graph:
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
        elif stage_name == "COMPARE_WITH_USER" and ctx.user_comparison_result:
            debug_logger.persist_user_comparison(debug_dir, ctx.user_comparison_result)
