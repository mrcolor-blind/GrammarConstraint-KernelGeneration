from pathlib import Path

from orchestration.pipelines.benchmark_pipeline import (
    BenchmarkPipeline,
)
from orchestration.pipelines.production_pipeline import (
    ProductionPipeline,
)
from orchestration.translation_pipeline import TranslationPipeline

from backends.modal.app import benchmark_app, production_app

# Modal deploy looks for a variable named 'app'
app = benchmark_app
from backends.modal.jobs.translate_validation import translate_validation
from backends.modal.jobs.compare_with_user import compare_with_user
from backends.modal.jobs.bench_evaluation_single import bench_evaluation_single  # registra con benchmark_app
from backends.modal.jobs.pipeline_remote import run_pipeline_remote              # pipeline completo en GPU
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
    all_calls: bool = False,
):
    """
    Translate a PyTorch function to Triton, with optional GPU validation and
    comparison against the user's PyTorch code on Modal.
    
    If --call-site is provided, the pipeline will execute the call site code
    locally to extract exact tensor shapes for the prompt and benchmark.
    If --all-calls is set, shapes from ALL test cases are captured, deduplicated
    by structural pattern, and shown to the LLM in the prompt.
    
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
        all_calls=all_calls,
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


@benchmark_app.local_entrypoint()
def translate_remote(
    file: str = "",
    call_site: str = "",
    provider: str = "nvidia",
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
    modal_validate: bool = False,
    compare_with_user: bool = False,
    speedup_threshold: float = 1.1,
    dims: str = "",
    all_calls: bool = False,
):
    """
    Corre el pipeline COMPLETO en Modal GPU T4 (shape extraction + LLM + compare, todo con CUDA).
    El output hace streaming nativo en la terminal.

    Ejemplo:
        modal run backends/modal/entrypoints.py::translate_remote \\
            --file datasets/custom/op_affine.py \\
            --call-site datasets/custom/call_affine.py \\
            --provider nvidia \\
            --model nvidia/llama-3.3-nemotron-super-49b-v1 \\
            --compare-with-user
    """
    import json

    if not file:
        print("Error: --file is required")
        return

    source_path = Path(file)
    if not source_path.exists():
        print(f"Error: file not found: {file}")
        return

    source_code = source_path.read_text(encoding="utf-8")

    call_site_code = ""
    if call_site:
        cs_path = Path(call_site)
        call_site_code = cs_path.read_text(encoding="utf-8") if cs_path.exists() else call_site

    concrete_dims: dict[str, int] = {}
    if dims:
        for pair in dims.split(","):
            if "=" in pair:
                k, v = pair.split("=")
                concrete_dims[k.strip()] = int(v.strip())

    print(f"Dispatching to Modal GPU T4: {source_path.name}")
    if call_site:
        print(f"  Call site: {call_site}")
    print(f"  Provider: {provider} | Model: {model}")
    print(f"  Compare: {compare_with_user} | GPU validate: {modal_validate}")

    result = run_pipeline_remote.remote(
        source_code=source_code,
        call_site_code=call_site_code,
        provider_name=provider,
        model_name=model,
        concrete_dims_json=json.dumps(concrete_dims),
        speedup_threshold=speedup_threshold,
        compare=compare_with_user,
        modal_validate=modal_validate,
    )

    run_id = result.get("run_id", "unknown")
    generated_code = result.get("generated_code", "")

    print(f"\nRun ID: {run_id}")

    if result.get("shape_extraction"):
        se = result["shape_extraction"]
        print(f"\nShape Extraction: {'PASS' if se['success'] else 'FAIL'}")
        for name, info in (se.get("shapes") or {}).items():
            if "shape" in info:
                print(f"  {name}: {info['shape']} dtype={info.get('dtype')}")
            elif "value" in info:
                print(f"  {name}: {info['value']} (scalar)")

    val = result.get("validation")
    if val:
        print(f"Static Validation: {'PASS' if val['passed'] else 'FAIL'}")
        for e in val.get("errors", []):
            print(f"  Error: {e}")
        for w in val.get("warnings", []):
            print(f"  Warning: {w}")

    gv = result.get("gpu_validation")
    if gv:
        print(f"\nGPU Validation:")
        print(f"  Compilation: {'PASS' if gv['compilation_pass'] else 'FAIL'}")
        print(f"  Execution:   {'PASS' if gv['execution_pass'] else 'FAIL'}")
        if gv.get("output_shape"):
            print(f"  Output shape: {gv['output_shape']}")
        for e in gv.get("errors", []):
            print(f"  Error: {e}")

    uc = result.get("user_comparison")
    if uc:
        strategy = uc.get("strategy", "compare")
        print(f"\nComparison ({strategy}):")
        print(f"  Compilation: {'PASS' if uc['compilation_pass'] else 'FAIL'}")
        print(f"  Accuracy:    {'PASS' if uc['accuracy_pass'] else 'FAIL'}")
        if uc.get("speedup") is not None:
            print(f"  Speedup:     {uc['speedup']:.2f}x")
        if uc.get("max_diff") is not None:
            print(f"  Max diff:    {uc['max_diff']:.2e}")
        print(f"  Suggest replacement: {'YES' if uc.get('suggest_replacement') else 'NO'}")
        if uc.get("reason"):
            print(f"  Reason: {uc['reason']}")
        for e in uc.get("errors", []):
            print(f"  Error: {e}")

    print(f"\nModal volume artifacts: translations/{run_id}/")

    if generated_code:
        print("\n--- Generated Code ---\n")
        print(generated_code)
    else:
        print("\nNo generated code produced.")