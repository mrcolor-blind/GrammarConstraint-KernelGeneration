"""
Modal GPU job for validating translated Triton kernels.
Performs compilation + execution smoke test on a real GPU.
"""

import importlib.util
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch

sys.path.append("/root/project")

from backends.modal.app import benchmark_app, volume

DATA_DIR = "/data"


def _generate_dummy_inputs(param_names, shapes, device="cuda"):
    """Generate random tensors matching annotated shapes with concrete defaults."""
    dim_defaults = {
        "N": 128, "B": 4, "S": 128, "D": 256,
        "D_in": 256, "D_out": 512, "H": 32, "W": 32,
        "C": 3, "C_out": 16, "K": 3, "H_out": 32, "W_out": 32,
    }
    inputs = {}
    for name in param_names:
        shape_str = shapes.get(name, "")
        import re
        m = re.match(r"\((.*)\)", shape_str.strip())
        if m:
            dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
            concrete_dims = []
            for d in dims:
                if d.isdigit():
                    concrete_dims.append(int(d))
                else:
                    concrete_dims.append(dim_defaults.get(d, 64))
            inputs[name] = torch.randn(concrete_dims, dtype=torch.float32, device=device)
        elif shape_str.lower() == "scalar":
            inputs[name] = torch.randn((), dtype=torch.float32, device=device)
        else:
            inputs[name] = torch.randn((64,), dtype=torch.float32, device=device)
    return inputs


def _load_module_from_string(module_name: str, code: str, write_dir: Path):
    """Write code to a real .py file, add to sys.path, and import normally.
    
    This is the most reliable way to make inspect.getsource() work,
    which Triton's JIT compiler requires.
    """
    module_path = write_dir / f"{module_name}.py"
    module_path.write_text(code, encoding="utf-8")
    
    # Add the directory to sys.path so importlib can find it normally
    str_dir = str(write_dir)
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)
    
    # Use standard import (most reliable for inspect.getsource)
    import importlib
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    return module


@benchmark_app.function(
    gpu="T4",
    timeout=60 * 10,
    volumes={DATA_DIR: volume},
)
def translate_validation(
    generated_code: str,
    function_name: str,
    param_names: list,
    input_shapes: dict,
    concrete_dims_str: str = "",
) -> dict:
    """
    Validate a generated translation inside a Modal GPU container.
    Writes code to a temp file so Triton JIT can introspect source.
    """
    # Parse any user-provided concrete dims
    concrete_dims = {}
    if concrete_dims_str:
        for pair in concrete_dims_str.split(","):
            if "=" in pair:
                k, v = pair.split("=")
                concrete_dims[k.strip()] = int(v.strip())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    errors = []
    compilation_pass = False
    execution_pass = False
    output_shape = None

    # Write code to a real file in /data (so inspect.getsource works)
    tmp_dir = Path(DATA_DIR) / "tmp_validation"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    # --- Check 1: Compilation ---
    try:
        module = _load_module_from_string(module_name, generated_code, tmp_dir)
        if not hasattr(module, function_name):
            errors.append(f"Function '{function_name}' not found in generated module.")
        else:
            compilation_pass = True
    except SyntaxError as e:
        errors.append(f"SyntaxError during compilation: {e.msg} (line {e.lineno})")
    except Exception as e:
        errors.append(f"Compilation failed: {type(e).__name__}: {e}")

    # --- Check 2: Execution (smoke test) ---
    if compilation_pass and hasattr(module, function_name):
        fn = getattr(module, function_name)
        try:
            dummy_inputs = _generate_dummy_inputs(param_names, input_shapes, device=device)
            input_args = [dummy_inputs[name] for name in param_names]
            result = fn(*input_args)

            if result is None:
                errors.append("Wrapper returned None.")
            elif not isinstance(result, torch.Tensor):
                errors.append(f"Wrapper returned {type(result).__name__}, expected torch.Tensor.")
            else:
                execution_pass = True
                output_shape = str(tuple(result.shape))
        except Exception as e:
            errors.append(f"Execution failed: {type(e).__name__}: {e}")

    volume.commit()

    return {
        "compilation_pass": compilation_pass,
        "execution_pass": execution_pass,
        "errors": errors,
        "output_shape": output_shape,
        "device": device,
    }
