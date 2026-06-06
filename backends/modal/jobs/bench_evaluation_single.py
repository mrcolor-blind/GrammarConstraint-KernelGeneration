"""
Single-operator TritonBench evaluation.

Escribe un JSONL de una sola predicción en el volumen Modal y corre
call_acc → exec_acc → efficiency, igual que benchEvaluation pero para
un kernel individual generado por el pipeline de traducción.

Devuelve el mismo schema que compare_with_user para que el caller
no necesite distinguir entre los dos paths.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

sys.path.append("/root/project")

from backends.modal.app import benchmark_app, volume

DATA_DIR = "/data"
REPO_DIR = "/opt/TritonBench"


@benchmark_app.function(
    include_source=True,
    gpu="T4",
    timeout=60 * 60 * 2,
    volumes={DATA_DIR: volume},
    secrets=[
        modal.Secret.from_name("triton-grammar-constrains")
    ],
)
def bench_evaluation_single(
    operator_name: str,
    generated_code: str,
    instruction: str,
) -> dict:
    """
    Evalúa un kernel individual contra TritonBench.

    Args:
        operator_name:  Nombre del operador (e.g. "gelu", "add").
        generated_code: Código Python generado por el pipeline.
        instruction:    La instrucción original del dataset TritonBench
                        (necesaria para call_acc que compara firmas).

    Returns dict con:
        call_accuracy, exec_accuracy, speedup, errors, strategy="tritonbench"
    """
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    os.environ["PYTHONPATH"] = (
        eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

    # ── Directorio de trabajo para este run ────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{operator_name}"
    work_dir = Path(DATA_DIR) / "bench_single" / run_id
    predictions_path = work_dir / "predictions.jsonl"
    call_acc_dir = work_dir / "call_acc"
    perf_results_dir = work_dir / "perf_results"

    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Escribir la predicción en formato JSONL ────────────────────────────
    prediction = {
        "instruction": instruction,
        "input": "",
        "predict": generated_code,
        "debug": {"operator": f"0000_{operator_name}", "exec_status": "PENDING"},
    }
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    errors: list[str] = []
    call_passed = 0
    exec_passed = 0
    speedup = None

    try:
        import call_acc
        import exe_acc

        # ── Phase 1: call accuracy ─────────────────────────────────────────
        call_acc_dir.mkdir(parents=True, exist_ok=True)
        try:
            call_acc.call_4file(
                str(predictions_path),
                str(call_acc_dir),
                gpus=[0],
            )
            survivors = list(call_acc_dir.glob("*.py"))
            call_passed = len(survivors)
        except Exception as e:
            errors.append(f"call_acc failed: {e}")

        # ── Phase 2: execution accuracy ────────────────────────────────────
        if call_passed > 0:
            try:
                exe_acc.execute_4folder(str(call_acc_dir), gpus=[0])
                exec_survivors = list(call_acc_dir.glob("*.py"))
                exec_passed = len(exec_survivors)
            except Exception as e:
                errors.append(f"exec_acc failed: {e}")

        # ── Phase 3: efficiency / speedup ──────────────────────────────────
        if exec_passed > 0:
            try:
                perf_root = f"{REPO_DIR}/performance_metrics/perf_T"
                perf_results_dir.mkdir(parents=True, exist_ok=True)

                subprocess.run(
                    [sys.executable, "run_bench/write_file.py",
                     "--input_folder_path", str(call_acc_dir),
                     "--results_path", str(perf_results_dir)],
                    cwd=perf_root, check=True, capture_output=True,
                )
                subprocess.run(
                    [sys.executable, "run_bench/multiprocess_gpu_run.py"],
                    cwd=perf_root, check=True, capture_output=True,
                )
                eff = subprocess.run(
                    [sys.executable, "2_efficiency.py",
                     "--gen_folder", str(perf_results_dir)],
                    cwd=eval_dir, capture_output=True, text=True,
                )
                for line in eff.stdout.splitlines():
                    if line.startswith("speed up:"):
                        try:
                            speedup = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
            except Exception as e:
                errors.append(f"efficiency failed: {e}")

    except ImportError as e:
        errors.append(f"TritonBench eval scripts not available: {e}")
    finally:
        # Limpieza
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        volume.commit()

    return {
        "strategy": "tritonbench",
        "operator": operator_name,
        "call_accuracy": call_passed > 0,
        "exec_accuracy": exec_passed > 0,
        "speedup": speedup,
        "errors": errors,
        # Campos compatibles con UserComparisonOut para que el schema unifique
        "compilation_pass": call_passed > 0,
        "accuracy_pass": exec_passed > 0,
        "max_diff": None,          # TritonBench no reporta diff numérico
        "ref_time_ms": None,
        "gen_time_ms": None,
        "suggest_replacement": exec_passed > 0 and (speedup or 0) > 1.0,
        "reason": (
            f"TritonBench evaluation: call_acc={'PASS' if call_passed else 'FAIL'}, "
            f"exec_acc={'PASS' if exec_passed else 'FAIL'}, "
            f"speedup={f'{speedup:.2f}x' if speedup else 'N/A'}."
        ),
        "device": "cuda",
        "concrete_dims": {},
    }
