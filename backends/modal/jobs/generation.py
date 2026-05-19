import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

import sys

sys.path.append("/root/project")

from backends.modal.app import app, volume
from datasets.tritonbench.loader import TritonBenchLoader
from models.registry.model_registry import load_provider
from prompts.builders.triton_prompt_builder import TritonPromptBuilder



DATA_DIR = "/data"
REPO_DIR = "/opt/TritonBench"


def extract_code(text: str) -> str:
    import re

    s = text.strip()

    match = re.search(
        r"```(?:python|py)?\s*\n(.*?)\n```",
        s,
        re.DOTALL,
    )

    if match:
        return match.group(1).strip() + "\n"

    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)

    return s.strip() + "\n"


@app.function(
    include_source=True,    
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: volume},
    secrets=[
        modal.Secret.from_name("triton-grammar-constrains")
    ]
)
def generate_predictions(
    provider_name: str,
    model_name: str,
    dataset: str = "simp",
    output_path: str = "predictions/predictions.jsonl",
    limit: int | None = None,
    concurrency: int = 8,
):
    provider = load_provider(provider_name)

    loader = TritonBenchLoader(REPO_DIR)
    items = loader.load_alpaca(dataset)

    if limit:
        items = items[:limit]

    prompt_builder = TritonPromptBuilder()

    def process_item(idx_item):
        idx, item = idx_item

        try:
            messages = prompt_builder.build(
                instruction=item["instruction"],
                input_text=item.get("input", ""),
            )

            raw_response = provider.generate(
                messages=messages,
                model=model_name,
            )

            print("=" * 80)
            print("RAW RESPONSE")
            print("=" * 80)
            print(raw_response[:5000])

            code = extract_code(raw_response)

            print("=" * 80)
            print("EXTRACTED CODE")
            print("=" * 80)
            print(code[:5000])

            namespace = {}

            try:
                exec(code, namespace)

                print("=" * 80)
                print("EXEC SUCCESS")
                print(namespace.keys())

            except Exception as e:
                print("=" * 80)
                print("EXEC FAILED")
                print(e)

        except Exception as e:
            code = f"# generation failed: {e}\n"

        return idx, {
            "instruction": item["instruction"],
            "predict": code,
        }

    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(process_item, (i, item))
            for i, item in enumerate(items)
        ]

        completed = 0

        for future in as_completed(futures):
            idx, result = future.result()

            results[idx] = result

            completed += 1

            if completed % 5 == 0 or completed == len(items):
                print(f"{completed}/{len(items)} complete")

    out_path = Path(DATA_DIR) / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    volume.commit()

    return output_path