from orchestration.pipelines.benchmark_pipeline import (
    BenchmarkPipeline,
)
from orchestration.pipelines.production_pipeline import (
    ProductionPipeline,
)

from backends.modal.app import benchmark_app, production_app


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