from typing import Optional, Union
class ModalRunner:
    def __init__(self, pipeline: str = "benchmark"):
        self.pipeline = pipeline
        
        if pipeline == "production":
            from backends.modal.jobs.production_evaluation import evaluate_predictions as prod_eval
            from backends.modal.jobs.production_generation import generate_predictions as prod_gen
            self.generate_predictions = prod_gen
            self.evaluate_predictions = prod_eval
        else:
            from backends.modal.jobs.evaluation import evaluate_predictions
            from backends.modal.jobs.generation import generate_predictions
            self.generate_predictions = generate_predictions
            self.evaluate_predictions = evaluate_predictions

    def generate(
        self,
        provider: str,
        model: str,
        dataset: str,
        output_path: str,
        limit:Optional[ int ] = None,
        operator:Optional[ str ] = None,
    ):
        return self.generate_predictions.remote(
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
        return self.evaluate_predictions.remote(
            predictions_path=predictions_path,
            output_subdir=output_subdir,
        )