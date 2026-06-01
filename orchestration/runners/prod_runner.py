from backends.modal.jobs.prod_evaluation import prodEvaluation
from backends.modal.jobs.prod_generation import prodGeneration


class ProdRunner:
    def generate(
        self,
        provider: str,
        model: str,
        dataset: str,
        output_path: str,
        limit: int | None = None,
        operator: str | None = None,
    ):
        return prodGeneration.remote(
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
        return prodEvaluation.remote(
            predictions_path=predictions_path,
            output_subdir=output_subdir,
        )
