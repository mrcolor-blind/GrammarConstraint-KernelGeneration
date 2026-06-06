from pathlib import Path

from orchestration.pipelines.benchmark_pipeline import (
    BenchmarkPipeline,
)
from orchestration.pipelines.production_pipeline import (
    ProductionPipeline,
)
from orchestration.translation_pipeline import TranslationPipeline

from backends.modal.app import benchmark_app, production_app
from backends.modal.jobs.translate_validation import translate_validation
from backends.modal.jobs.compare_with_user import compare_with_user
from backends.modal.jobs.bench_evaluation_single import bench_evaluation_single  # registra con benchmark_app
from evaluation.smart_evaluator import smart_evaluate
from datasets.tritonbench.registry import get_registry


@benchmark_app.local_entrypoint()
def benchmark(
    provider: str = "nvidia",
    model: str = "mistralai/devstral-small-2507",
    dataset: str = "simp",
    limit: int = 0,
    operator: str = "",
):
    summary = BenchmarkPipeline().run(
        provider=provider,
        model=model,
        dataset=dataset,
        limit=limit or None,
        operator=operator or None,
    )

    print(summary)


@production_app.local_entrypoint()
def production(
    provider: str = "nvidia",
    model: str = "mistralai/devstral-small-2507",
    dataset: str = "simp",
    limit: int = 0,
    operator: str = "",
):
    summary = ProductionPipeline().run(
        provider=provider,
        model=model,
        dataset=dataset,
        limit=limit or None,
        operator=operator or None,
    )

    print(summary)


@benchmark_app.local_entrypoint()
def translate(
    file: str = "linear_relu.py",
    provider: str = "nvidia",
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
    modal_validate: bool = False,
    compare_with_user: bool = False,
    speedup_threshold: float = 1.1,
    call_site: str = "",
    dims: str = "",
):
    """
    Translate a PyTorch function to Triton, with optional GPU validation and
    comparison against the user's PyTorch code on Modal.
    
    If --call-site is provided, the pipeline will execute the call site code
    locally to extract exact tensor shapes for the prompt and benchmark.
    
    Local stages run on your machine; GPU validation and comparison run inside
    a Modal container.
    Artifacts are saved locally to debug/translations/.
    """
    source_path = Path(file)
    if not source_path.exists():
        print(f"Error: file not found: {file}")
        return

    source_code = source_path.read_text(encoding="utf-8")

    # Parse concrete dims
    concrete_dims = {}
    if dims:
        for pair in dims.split(","):
            if "=" in pair:
                k, v = pair.split("=")
                concrete_dims[k.strip()] = int(v.strip())

    # Read call site code if provided as a file path
    call_site_code = ""
    if call_site:
        call_site_path = Path(call_site)
        if call_site_path.exists():
            call_site_code = call_site_path.read_text(encoding="utf-8")
        else:
            # If not a file, treat as inline code
            call_site_code = call_site

    pipeline = TranslationPipeline(
        provider_name=provider,
        model_name=model,
        modal_validate=modal_validate,
        compare_with_user=compare_with_user,
        speedup_threshold=speedup_threshold,
        concrete_dims=concrete_dims,
        call_site_code=call_site_code,
    )

    ctx = pipeline.run(
        file_path=str(source_path),
        source_code=source_code,
        call_site_code=call_site_code,
    )

    # Print summary
    print(f"\nRun ID: {ctx.run_id}")
    
    if ctx.shape_extraction_result:
        ser = ctx.shape_extraction_result
        print(f"\nShape Extraction:")
        print(f"  Success: {'PASS' if ser.success else 'FAIL'}")
        if ser.error:
            print(f"  Error: {ser.error}")
        if ser.shapes:
            print(f"  Extracted shapes:")
            for name, info in ser.shapes.items():
                if "shape" in info:
                    print(f"    {name}: {info['shape']} (dtype={info.get('dtype', 'N/A')}, device={info.get('device', 'N/A')})")
                elif "value" in info:
                    print(f"    {name}: {info['value']} ({info.get('type', 'N/A')})")

    if ctx.validation_result:
        vr = ctx.validation_result
        print(f"Static Validation: {'PASS' if vr.passed else 'FAIL'}")
        if vr.errors:
            for e in vr.errors:
                print(f"  Error: {e}")
        if vr.warnings:
            for w in vr.warnings:
                print(f"  Warning: {w}")

    if ctx.gpu_validation_result:
        gvr = ctx.gpu_validation_result
        print(f"\nGPU Validation (Modal):")
        print(f"  Compilation: {'PASS' if gvr.compilation_pass else 'FAIL'}")
        print(f"  Execution: {'PASS' if gvr.execution_pass else 'FAIL'}")
        if gvr.output_shape:
            print(f"  Output shape: {gvr.output_shape}")
        if gvr.device:
            print(f"  Device: {gvr.device}")
        if gvr.errors:
            for e in gvr.errors:
                print(f"  Error: {e}")

    if ctx.user_comparison_result:
        ucr = ctx.user_comparison_result
        strategy = getattr(ucr, "strategy", "user_comparison")
        label = "TritonBench Evaluation" if strategy == "tritonbench" else "User Comparison (Modal)"
        print(f"\n{label}:")
        print(f"  Strategy: {strategy}")
        print(f"  Compilation: {'PASS' if ucr.compilation_pass else 'FAIL'}")
        print(f"  Accuracy: {'PASS' if ucr.accuracy_pass else 'FAIL'}")
        if ucr.max_diff is not None:
            print(f"  Max diff: {ucr.max_diff:.6e}")
        if ucr.speedup is not None:
            print(f"  Speedup: {ucr.speedup:.2f}x")
        if ucr.ref_time_ms is not None:
            print(f"  PyTorch time: {ucr.ref_time_ms:.2f} ms")
        if ucr.gen_time_ms is not None:
            print(f"  Triton time: {ucr.gen_time_ms:.2f} ms")
        print(f"  Suggest replacement: {'YES' if ucr.suggest_replacement else 'NO'}")
        if ucr.reason:
            print(f"  Reason: {ucr.reason}")
        if ucr.errors:
            for e in ucr.errors:
                print(f"  Error: {e}")

    print(f"\nArtifacts: debug/translations/{ctx.run_id}/")

    if ctx.generated_code:
        print("\n--- Generated Code ---\n")
        print(ctx.generated_code)
    else:
        print("No generated code produced.")