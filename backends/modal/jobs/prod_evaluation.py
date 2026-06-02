import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

sys.path.append("/root/project")

from backends.modal.app import production_app, volume

DATA_DIR = "/data"
REPO_DIR = "/opt/TritonBench"


@production_app.function(
    include_source=True,
    gpu="T4",
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: volume},
    secrets=[
        modal.Secret.from_name("triton-grammar-constrains")
    ]
)
def prodEvaluation(
    predictions_path: str,
    output_subdir: str = "results",
):
    pred_full = Path(DATA_DIR) / predictions_path

    if not pred_full.exists():
        raise FileNotFoundError(pred_full)

    out_dir = Path(DATA_DIR) / output_subdir

    out_dir.mkdir(parents=True, exist_ok=True)

    call_acc_dir = out_dir / "call_acc"
    perf_results_dir = out_dir / "perf_results"

    if call_acc_dir.exists():
        shutil.rmtree(call_acc_dir)

    if perf_results_dir.exists():
        shutil.rmtree(perf_results_dir)

    eval_dir = f"{REPO_DIR}/EVAL/eval_T"

    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    os.environ["PYTHONPATH"] = (
        eval_dir + os.pathsep +
        os.environ.get("PYTHONPATH", "")
    )

    import call_acc
    import exe_acc

    total = sum(1 for _ in pred_full.open())

    print("\n=== Phase 1: call accuracy ===\n")


    call_acc.call_4file(
        str(pred_full),
        str(call_acc_dir),
        gpus=[0],
    )

    call_survivors = sorted(
        p.name for p in call_acc_dir.glob("*.py")
    )

    print(f"call_acc survivors: {len(call_survivors)} / {total}")

    print("\n=== Phase 2: execution accuracy ===\n")

    if call_survivors:
        exe_acc.execute_4folder(
            str(call_acc_dir),
            gpus=[0],
        )

    exec_survivors = sorted(
        p.name for p in call_acc_dir.glob("*.py")
    )

    print(f"exec_acc survivors: {len(exec_survivors)} / {total}")

    print("\n=== Phase 3: efficiency ===\n")

    speedup = None
    efficiency_output = ""

    if exec_survivors:
        perf_root = f"{REPO_DIR}/performance_metrics/perf_T"

        subprocess.run(
            [
                sys.executable,
                "run_bench/write_file.py",
                "--input_folder_path",
                str(call_acc_dir),
                "--results_path",
                str(perf_results_dir),
            ],
            cwd=perf_root,
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "run_bench/multiprocess_gpu_run.py",
            ],
            cwd=perf_root,
            check=True,
        )

        eff = subprocess.run(
            [
                sys.executable,
                "2_efficiency.py",
                "--gen_folder",
                str(perf_results_dir),
            ],
            cwd=eval_dir,
            capture_output=True,
            text=True,
        )

        efficiency_output = eff.stdout

        for line in eff.stdout.splitlines():
            if line.startswith("speed up:"):
                try:
                    speedup = float(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError:
                    pass

    volume.commit()

    summary = {
        "total_predictions": total,
        "call_acc": {
            "passed": len(call_survivors),
            "rate": round(
                100 * len(call_survivors) / total,
                2,
            ) if total else 0,
        },
        "exec_acc": {
            "passed": len(exec_survivors),
            "rate": round(
                100 * len(exec_survivors) / total,
                2,
            ) if total else 0,
        },
        "speedup": speedup,
        "raw_efficiency_output": efficiency_output[-2000:],
    }

    return summary
