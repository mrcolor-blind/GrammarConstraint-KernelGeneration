from orchestration.runners.prod_runner import ProdRunner


class ProductionPipeline:
    def __init__(self):
        self.runner = ProdRunner()

    def run(
        self,
        provider: str,
        model: str,
        dataset: str = "simp",
        limit: int | None = None,
        operator: str | None = None, 
    ):
        tag = f"{provider}_{model.replace('/', '_').replace(':', '_')}_{dataset}"
        if operator:
            tag += f"_{operator}"

        predictions_path = f"predictions/{tag}.jsonl"
        results_dir = f"results/{tag}"

        remote_predictions = self.runner.generate(
            provider=provider,
            model=model,
            dataset=dataset,
            output_path=predictions_path,
            limit=limit,
            operator=operator,
        )

        summary = self.runner.evaluate(
            predictions_path=remote_predictions,
            output_subdir=results_dir,
        )
        return summary
