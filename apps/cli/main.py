import argparse
import json

from orchestration.pipelines.benchmark_pipeline import (
    BenchmarkPipeline,
)
from orchestration.pipelines.production_pipeline import (
    ProductionPipeline,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        default="simp",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--operator",
        default=None,
    )

    parser.add_argument(
        "--pipeline",
        choices=["bench", "prod"],
        default="bench",
    )

    args = parser.parse_args()

    if args.pipeline == "prod":
        pipeline = ProductionPipeline()
    else:
        pipeline = BenchmarkPipeline()

    summary = pipeline.run(
        provider=args.provider,
        model=args.model,
        dataset=args.dataset,
        limit=args.limit,
        operator=args.operator,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()