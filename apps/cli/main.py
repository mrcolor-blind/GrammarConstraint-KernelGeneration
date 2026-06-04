"""
CLI — entry point for the translation pipeline and benchmark pipelines.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for local execution
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from orchestration.translation_pipeline import TranslationPipeline
from utils import debug_logger
from typing import Optional, Union


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# New translation commands
# ---------------------------------------------------------------------------

def cmd_translate(args):
    setup_logging(args.verbose)

    source_path = Path(args.file)
    if not source_path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    source_code = source_path.read_text(encoding="utf-8")

    if args.dry_run:
        print("=== DRY RUN ===")
        print("Pipeline will run up to prompt generation, no LLM call.")
        print("(Not yet implemented — running full pipeline)")

    pipeline = TranslationPipeline(
        provider_name=args.provider,
        model_name=args.model,
    )

    ctx = pipeline.run(
        file_path=str(source_path),
        source_code=source_code,
    )

    # Output
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{ctx.run_id}.py"
        if ctx.generated_code:
            out_file.write_text(ctx.generated_code, encoding="utf-8")
            print(f"Generated code written to: {out_file}")
        else:
            print("No generated code produced.", file=sys.stderr)
    else:
        if ctx.generated_code:
            print(ctx.generated_code)
        else:
            print("No generated code produced.", file=sys.stderr)

    # Summary
    print(f"\nRun ID: {ctx.run_id}")
    if ctx.validation_result:
        vr = ctx.validation_result
        print(f"Validation: {'PASS' if vr.passed else 'FAIL'}")
        if vr.errors:
            for e in vr.errors:
                print(f"  Error: {e}")
        if vr.warnings:
            for w in vr.warnings:
                print(f"  Warning: {w}")
    print(f"Debug artifacts: debug/translations/{ctx.run_id}/")

    if not ctx.generated_code or (ctx.validation_result and not ctx.validation_result.passed):
        sys.exit(1)


def cmd_inspect(args):
    run_dir = Path("debug/translations") / args.run
    if not run_dir.exists():
        print(f"Error: run not found: {args.run}", file=sys.stderr)
        sys.exit(1)

    summary_file = run_dir / "summary.md"
    if summary_file.exists():
        print(summary_file.read_text())
    else:
        print(f"Artifacts for run '{args.run}':")
        for f in sorted(run_dir.iterdir()):
            print(f"  {f.name}")


def cmd_evaluate(args):
    """Evaluate a generated translation against the original PyTorch function."""
    setup_logging(args.verbose)

    from evaluation.translate_evaluator import run_local_evaluation

    run_dir = Path("debug/translations") / args.run
    if not run_dir.exists():
        print(f"Error: run not found: {args.run}", file=sys.stderr)
        sys.exit(1)

    input_file = run_dir / "01_input.py"
    generated_file = run_dir / "10_final.py"

    if not input_file.exists():
        print(f"Error: missing {input_file}", file=sys.stderr)
        sys.exit(1)
    if not generated_file.exists():
        print(f"Error: missing generated code {generated_file}", file=sys.stderr)
        sys.exit(1)

    result = run_local_evaluation(
        original_path=input_file,
        generated_path=generated_file,
        concrete_dims=_parse_concrete_dims(args.dims) if args.dims else None,
    )

    print(json.dumps(result, indent=2))

    if not result.get("accuracy_pass", False):
        sys.exit(1)


def _parse_concrete_dims(dims_str:Optional[ str ]) -> dict[str, int]:
    if not dims_str:
        return {}
    result = {}
    for pair in dims_str.split(","):
        k, v = pair.split("=")
        result[k.strip()] = int(v.strip())
    return result


# ---------------------------------------------------------------------------
# Legacy benchmark commands
# ---------------------------------------------------------------------------

def cmd_benchmark(args):
    # Lazy imports to avoid pulling in Modal dependencies when not needed
    from orchestration.pipelines.benchmark_pipeline import BenchmarkPipeline
    from orchestration.pipelines.production_pipeline import ProductionPipeline

    parser_inner = argparse.ArgumentParser()
    parser_inner.add_argument("--provider", required=True)
    parser_inner.add_argument("--model", required=True)
    parser_inner.add_argument("--dataset", default="simp")
    parser_inner.add_argument("--limit", type=int, default=None)
    parser_inner.add_argument("--operator", default=None)
    parser_inner.add_argument("--pipeline", choices=["bench", "prod"], default="bench")
    inner_args = parser_inner.parse_args(args.remainder)

    if inner_args.pipeline == "prod":
        pipeline = ProductionPipeline()
    else:
        pipeline = BenchmarkPipeline()

    summary = pipeline.run(
        provider=inner_args.provider,
        model=inner_args.model,
        dataset=inner_args.dataset,
        limit=inner_args.limit,
        operator=inner_args.operator,
    )
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch to Triton Translation & Benchmark CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- translate ---
    translate_parser = subparsers.add_parser(
        "translate",
        help="Translate a PyTorch function to Triton",
    )
    translate_parser.add_argument(
        "--file",
        required=True,
        help="Path to the Python file containing the @triton-annotated function",
    )
    translate_parser.add_argument(
        "--provider",
        required=True,
        help="LLM provider name (e.g., nvidia, openai, gemini)",
    )
    translate_parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., nvidia/llama-3.3-nemotron-super-49b-v1)",
    )
    translate_parser.add_argument(
        "--output",
        default=None,
        help="Directory to write the generated .py file (default: print to stdout)",
    )
    translate_parser.add_argument(
        "--local",
        action="store_true",
        help="Run LLM generation locally (no Modal)",
    )
    translate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline up to prompt, skip LLM call",
    )
    translate_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # --- inspect ---
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a previous translation run",
    )
    inspect_parser.add_argument(
        "--run",
        required=True,
        help="Run ID (e.g., 2026-06-03_14-22-10_linear_relu)",
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a generated translation numerically and for speedup",
    )
    eval_parser.add_argument(
        "--run",
        required=True,
        help="Run ID to evaluate",
    )
    eval_parser.add_argument(
        "--dims",
        default=None,
        help='Concrete dimensions for symbolic shapes, e.g., "N=128,D_in=256"',
    )
    eval_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # --- benchmark (legacy) ---
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the legacy benchmark/production pipeline",
    )
    benchmark_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the benchmark pipeline",
    )

    args = parser.parse_args()

    if args.command == "translate":
        cmd_translate(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
