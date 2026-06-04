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
    dims: str = "",
):
    """
    Translate a PyTorch function to Triton, with optional GPU validation on Modal.
    Local stages run on your machine; GPU validation runs inside a Modal container.
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

    pipeline = TranslationPipeline(
        provider_name=provider,
        model_name=model,
        modal_validate=modal_validate,
        concrete_dims=concrete_dims,
    )

    ctx = pipeline.run(
        file_path=str(source_path),
        source_code=source_code,
    )

    # Print summary
    print(f"\nRun ID: {ctx.run_id}")
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

    print(f"\nArtifacts: debug/translations/{ctx.run_id}/")

    if ctx.generated_code:
        print("\n--- Generated Code ---\n")
        print(ctx.generated_code)
    else:
        print("No generated code produced.")