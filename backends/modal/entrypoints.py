from orchestration.pipelines.benchmark_pipeline import (
    BenchmarkPipeline,
)

from backends.modal.app import app


@app.local_entrypoint()
def main(
    provider: str = "nvidia",
    model: str = "mistralai/devstral-small-2507",
    dataset: str = "simp",
    limit: int = 0,
):
    pipeline = BenchmarkPipeline()

    summary = pipeline.run(
        provider=provider,
        model=model,
        dataset=dataset,
        limit=limit or None,
    )

    print(summary)