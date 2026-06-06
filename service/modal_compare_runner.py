"""
Modal Compare Runner — local entrypoint that runs compare_with_user via Modal.

Usage (from inside the Docker container):
    modal run service/modal_compare_runner.py \
        --json-file /tmp/compare_input.json \
        --output-file /tmp/compare_output.json

The JSON file must contain:
{
    "original_code": "...",
    "generated_code": "...",
    "concrete_dims_str": "N=128",
    "speedup_threshold": 1.0
}
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backends.modal.app import benchmark_app
from backends.modal.jobs.compare_with_user import compare_with_user


@benchmark_app.local_entrypoint()
def main(
    json_file: str = "",
    output_file: str = "",
):
    if not json_file or not output_file:
        print(json.dumps({"error": "Missing --json-file or --output-file"}), file=sys.stderr)
        sys.exit(1)

    json_path = Path(json_file)
    if not json_path.exists():
        print(json.dumps({"error": f"JSON file not found: {json_file}"}), file=sys.stderr)
        sys.exit(1)

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    try:
        result = compare_with_user.remote(
            original_code=payload["original_code"],
            generated_code=payload["generated_code"],
            concrete_dims_str=payload.get("concrete_dims_str", ""),
            extracted_shapes_json=payload.get("extracted_shapes_json", ""),
            speedup_threshold=payload.get("speedup_threshold", 1.0),
        )
        Path(output_file).write_text(json.dumps(result), encoding="utf-8")
        print(f"Compare result written to {output_file}", file=sys.stderr)
    except Exception as exc:
        error_result = {"error": f"{type(exc).__name__}: {exc}"}
        Path(output_file).write_text(json.dumps(error_result), encoding="utf-8")
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)
