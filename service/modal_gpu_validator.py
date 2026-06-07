"""
Modal GPU Validator — local entrypoint that runs GPU validation via Modal.

Usage (from inside the Docker container):
    modal run service/modal_gpu_validator.py \
        --json-file /tmp/gpu_validate_input.json \
        --output-file /tmp/gpu_validate_output.json

The JSON file must contain:
{
    "generated_code": "import triton\n...",
    "function_name": "linear_relu",
    "param_names": ["x", "weight", "bias"],
    "input_shapes": {"x": "(N, D_in)", "weight": "(D_out, D_in)", "bias": "(D_out,)"},
    "concrete_dims_str": "N=128,D_in=256,D_out=512"
}

Output: writes a JSON dict to --output-file with compilation_pass, execution_pass, errors, output_shape, device.
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backends.modal.app import benchmark_app
from backends.modal.jobs.translate_validation import translate_validation


@benchmark_app.local_entrypoint()
def main(
    json_file: str = "",
    output_file: str = "",
):
    """
    Entrypoint that reads validation input from a JSON file,
    calls translate_validation.remote(), and writes the result as JSON to output_file.
    """
    if not json_file:
        print(json.dumps({"error": "Missing --json-file argument"}), file=sys.stderr)
        sys.exit(1)

    if not output_file:
        print(json.dumps({"error": "Missing --output-file argument"}), file=sys.stderr)
        sys.exit(1)

    json_path = Path(json_file)
    if not json_path.exists():
        print(json.dumps({"error": f"JSON file not found: {json_file}"}), file=sys.stderr)
        sys.exit(1)

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    required = ["generated_code", "function_name", "param_names", "input_shapes"]
    missing = [k for k in required if k not in payload]
    if missing:
        print(json.dumps({"error": f"Missing keys in JSON: {missing}"}), file=sys.stderr)
        sys.exit(1)

    try:
        result = translate_validation.remote(
            generated_code=payload["generated_code"],
            function_name=payload["function_name"],
            param_names=payload["param_names"],
            input_shapes=payload["input_shapes"],
            concrete_dims_str=payload.get("concrete_dims_str", ""),
            original_code=payload.get("original_source_code", ""),
        )
        # Write result to output file (avoids Modal stdout noise)
        Path(output_file).write_text(json.dumps(result), encoding="utf-8")
        print(f"GPU validation result written to {output_file}", file=sys.stderr)
    except Exception as exc:
        error_result = {"error": f"{type(exc).__name__}: {exc}"}
        Path(output_file).write_text(json.dumps(error_result), encoding="utf-8")
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)
