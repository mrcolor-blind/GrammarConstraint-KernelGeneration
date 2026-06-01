import json
import re
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

def make_operator_name(item: dict, idx: int) -> str:
    instruction = item["instruction"]

    match = re.search(
        r"Wrapper Entry Information:\s*(?:def\s+)?([a-zA-Z0-9_]+)\(",
        instruction,
    )

    if match:
        fn_name = match.group(1)
        return f"{idx:04d}_{fn_name}"

    return f"{idx:04d}_unknown_operator"

def extract_operator_name(instruction: str) -> str | None:
    match = re.search(
        r"Wrapper Entry Information:\s*(?:def\s+)?([a-zA-Z0-9_]+)\(",
        instruction,
    )
    return match.group(1) if match else None

def write_debug_file(
    debug_dir: Path,
    op_name: str,
    provider_name: str,
    model_name: str,
    messages,
    raw_response: str,
    code: str,
    exec_status: str,
):
    out_file = debug_dir / f"{op_name}.txt"

    with out_file.open("w") as f:
        f.write("=== METADATA ===\n\n")
        f.write(f"PROVIDER: {provider_name}\n")
        f.write(f"MODEL: {model_name}\n")
        f.write(f"OPERATOR: {op_name}\n")
        f.write(f"EXEC_STATUS: {exec_status}\n")

        f.write("\n\n=== PROMPT ===\n\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            f.write(f"[{role}]\n")
            f.write(content)
            f.write("\n\n")

        f.write("=== RAW RESPONSE ===\n\n")
        f.write(raw_response)

        f.write("\n\n=== EXTRACTED CODE ===\n\n")
        f.write(code)


@app.function(
    include_source=True,
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: volume},
    secrets=[
        modal.Secret.from_name("triton-grammar-constrains")
    ]
)
def prodGeneration(
    provider_name: str,
    model_name: str,
    dataset: str = "simp",
    output_path: str = "predictions/predictions.jsonl",
    limit: int | None = None,
    concurrency: int = 8,
    operator: str | None = None,
):
    provider = load_provider(provider_name)

    loader = TritonBenchLoader(REPO_DIR)
    items = loader.load_alpaca(dataset)

    if operator:
        filtered_items = []

        for item in items:
            instruction = item.get(
                "instruction",
                "",
            )

            op_name = extract_operator_name(
                instruction
            )

            if op_name == operator:
                filtered_items.append(item)

        items = filtered_items

        print("=" * 80)
        print(
            f"Filtered dataset to "
            f"{len(items)} entries "
            f"for operator '{operator}'"
        )

        if not items:
            raise ValueError(
                f"No entries found for "
                f"operator '{operator}'"
            )

    if limit:
        items = items[:limit]

    prompt_builder = TritonPromptBuilder()

    results = [None] * len(items)

    out_path = Path(DATA_DIR) / output_path

    debug_dir = out_path.parent / "debug"

    debug_dir.mkdir(parents=True, exist_ok=True)

    def process_item(idx_item):
        idx, item = idx_item

        op_name = make_operator_name(item, idx)

        print("=" * 80)
        print(f"[{idx + 1}/{len(items)}] Processing: {op_name}")

        try:
            messages = prompt_builder.build(
                instruction=item["instruction"],
                input_text=item.get("input", ""),
            )

            raw_response = provider.generate(
                messages=messages,
                model=model_name,
            )

            code = extract_code(raw_response)

            namespace = {}

            try:
                exec(code, namespace)

                exec_status = "SUCCESS"

                print(f"[EXEC SUCCESS] {op_name}")

            except Exception as e:
                exec_status = f"FAILED: {e}"

                print(f"[EXEC FAILED] {op_name}")
                print(e)

        except Exception as e:
            messages = []
            raw_response = str(e)
            code = f"# generation failed: {e}\n"
            exec_status = "GENERATION_FAILED"

            print(f"[GENERATION FAILED] {op_name}")
            print(e)

        debug = {
            "operator": op_name,
            "messages": messages,
            "raw_response": raw_response,
            "exec_status": exec_status,
        }

        result = {
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "predict": code,
            "debug": debug,
        }

        write_debug_file(
            debug_dir=debug_dir,
            op_name=op_name,
            provider_name=provider_name,
            model_name=model_name,
            messages=messages,
            raw_response=raw_response,
            code=code,
            exec_status=exec_status,
        )

        return idx, result

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

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    volume.commit()

    return output_path
