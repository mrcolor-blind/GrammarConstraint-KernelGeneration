""" REPO_DIR = "/opt/TritonBench"

PATCH_CALL_ACC = (
    f\"\"\"sed -i \"\"\"
    f\"\"\"-e 's|^statis_path = .*|statis_path = \\\"{REPO_DIR}/data/TritonBench_T_v1.jsonl\\\"|' \"\"\"
    f\"\"\"-e 's|^py_folder = .*|py_folder = \\\"{REPO_DIR}/data/TritonBench_T_v1/\\\"|' \"\"\"
    f\"\"\"-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' \"\"\"
    f\"\"\"{REPO_DIR}/EVAL/eval_T/0_call_acc.py\"\"\"
)

PATCH_EXE_ACC = (
    f\"\"\"sed -i \"\"\"
    f\"\"\"-e 's|^gold_folder = .*|gold_folder = \\\"{REPO_DIR}/data/TritonBench_T_v1/\\\"|' \"\"\"
    f\"\"\"-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' \"\"\"
    f\"\"\"{REPO_DIR}/EVAL/eval_T/1_exe_acc.py\"\"\"
)

PATCH_PERF = (
    f\"\"\"sed -i 's|^gpu_count = .*|gpu_count = 1|' \"\"\"
    f\"\"\"{REPO_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py\"\"\"
) """