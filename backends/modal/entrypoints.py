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
    operator: str = ""
):
    pipeline = BenchmarkPipeline()

    summary = pipeline.run(
        provider=provider,
        model=model,
        dataset=dataset,
        limit=limit or None,
        operator=operator or None,
    )

    print(summary)