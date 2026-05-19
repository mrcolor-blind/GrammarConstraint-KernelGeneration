import json
from pathlib import Path


class TritonBenchLoader:
    def __init__(self, repo_dir: str):
        self.repo_dir = Path(repo_dir)

    def load_alpaca(self, dataset: str = "simp"):
        path = (
            self.repo_dir /
            f"data/TritonBench_T_{dataset}_alpac_v1.json"
        )

        return json.loads(path.read_text())