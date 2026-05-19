from orchestration.runners.modal_runner import ModalRunner


class BenchmarkPipeline:
    def __init__(self):
        self.runner = ModalRunner()

    def run(
        self,
        provider: str,
        model: str,
        dataset: str = "simp",
        limit: int | None = None,
    ):
        tag = (
            f"{provider}_"
            f"{model.replace('/', '_').replace(':', '_')}_"
            f"{dataset}"
        )

        predictions_path = (
            f"predictions/{tag}.jsonl"
        )

        results_dir = (
            f"results/{tag}"
        )

        remote_predictions = self.runner.generate(
            provider=provider,
            model=model,
            dataset=dataset,
            output_path=predictions_path,
            limit=limit,
        )

        summary = self.runner.evaluate(
            predictions_path=remote_predictions,
            output_subdir=results_dir,
        )

        return summary