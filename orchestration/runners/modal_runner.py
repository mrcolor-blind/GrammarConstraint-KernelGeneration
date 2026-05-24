from backends.modal.jobs.evaluation import evaluate_predictions
from backends.modal.jobs.generation import generate_predictions


class ModalRunner:
    def generate(
        self,
        provider: str,
        model: str,
        dataset: str,
        output_path: str,
        limit: int | None = None,
        operator: str | None = None,
    ):
        return generate_predictions.remote(
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
        return evaluate_predictions.remote(
            predictions_path=predictions_path,
            output_subdir=output_subdir,
        )