import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"
REPO_DIR = "/opt/TritonBench"
TRITONBENCH_DIR = "/opt/TritonBench"

PATCH_CALL_ACC = (
    f"""sed -i """
    f"""-e 's|^statis_path = .*|statis_path = "{TRITONBENCH_DIR}/data/TritonBench_T_v1.jsonl"|' """
    f"""-e 's|^py_folder = .*|py_folder = "{TRITONBENCH_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{TRITONBENCH_DIR}/EVAL/eval_T/0_call_acc.py"""
)

PATCH_EXE_ACC = (
    f"""sed -i """
    f"""-e 's|^gold_folder = .*|gold_folder = "{TRITONBENCH_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{TRITONBENCH_DIR}/EVAL/eval_T/1_exe_acc.py"""
)

PATCH_PERF = (
    f"""sed -i 's|^gpu_count = .*|gpu_count = 1|' """
    f"""{TRITONBENCH_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py"""
)

# List of path substrings to exclude from the image upload
_EXCLUDED_SUBSTRINGS = [
    # Virtual environments
    "/venv/", "/.venv/", "/env/",
    # Local debug artifacts
    "/debug/", "/predictions/", "/results/",
    # Python cache
    "/__pycache__/", ".pyc", ".pyo", ".egg-info/", ".pytest_cache/",
    # Git
    "/.git/", ".gitignore",
    # Build artifacts
    "/dist/", "/build/", ".egg",
    # IDE
    "/.vscode/", "/.idea/", ".swp", ".swo",
    # OS
    ".DS_Store", "Thumbs.db",
    # Jupyter
    ".ipynb_checkpoints/", ".ipynb",
    # Modal
    "/.modal/",
    # VS Code Extension
    "/apps/vscode-extension/",
    # Secrets / local env
    ".env", ".envrc",
    # Local datasets (not needed at runtime — TritonBench is cloned in the image)
    "/datasets/tritonbench/",  # Exclude the whole tritonbench dataset dir
    # Tests / local scripts
    "test_", "_test.py",
    # Docs
    "docker-compose.yml", ".dockerignore", "Dockerfile",
    "SERVICE.md", "README.md",
    # Unused backends
    "/backends/local/", "/backends/future/",
]


def _should_ignore(path: Path) -> bool:
    """Return True if the file should be ignored (skipped) in the image upload."""
    path_str = str(path)
    for excluded in _EXCLUDED_SUBSTRINGS:
        if excluded in path_str:
            return True
    return False


def _create_benchmark_image():
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04",
            add_python="3.12",
        )
        .apt_install(
            "git",
            "build-essential",
        )
        .pip_install(
            "torch==2.5.1",
            "triton==3.1.0",
            "tqdm==4.66.5",
            "numpy<2",
            "openai>=1.50",
            "google-genai",
        )
        .run_commands(
            f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}"
        )
        .run_commands(
            PATCH_CALL_ACC,
            PATCH_EXE_ACC,
            PATCH_PERF,
        )
        .run_commands(
            f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py"
        )
        .run_commands(
            f"ln -s {REPO_DIR}/EVAL/eval_T/1_exe_acc.py {REPO_DIR}/EVAL/eval_T/exe_acc.py"
        )
        .add_local_dir(
            str(PROJECT_ROOT),
            remote_path="/root/project",
            ignore=_should_ignore,
        )
    )


def _create_production_image():
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04",
            add_python="3.12",
        )
        .apt_install(
            "git",
            "build-essential",
        )
        .pip_install(
            "torch==2.5.1",
            "triton==3.1.0",
            "numpy<2",
            "openai>=1.50",
            "google-genai",
            "tqdm==4.66.5",
        )
        .add_local_dir(
            str(PROJECT_ROOT),
            remote_path="/root/project",
            ignore=_should_ignore,
        )
    )

benchmark_image = _create_benchmark_image()
production_image = _create_production_image()
