from backends.modal.jobs.bench_evaluation import benchEvaluation
from backends.modal.jobs.bench_generation import benchGeneration
from typing import Optional, Union


class BenchRunner:
    def generate(
        self,
        provider: str,
        model: str,
        dataset: str,
        output_path: str,
        limit:Optional[ int ] = None,
        operator:Optional[ str ] = None,
    ):
        return benchGeneration.remote(
            provider_name=provider,
            model_name=model,
            dataset=dataset,
            output_path=output_path,
            limit=limit,
            operator=operator,
        )

    def evaluate(
        self,
        predictions_path: str,
        output_subdir: str = "results",
    ):
        return benchEvaluation.remote(
            predictions_path=predictions_path,
            output_subdir=output_subdir,
        )
