import argparse
import json

from orchestration.pipelines.benchmark_pipeline import (
    BenchmarkPipeline,
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

    args = parser.parse_args()

    pipeline = BenchmarkPipeline()

    summary = pipeline.run(
        provider=args.provider,
        model=args.model,
        dataset=args.dataset,
        limit=args.limit,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()