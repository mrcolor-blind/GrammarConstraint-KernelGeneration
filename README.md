# Grammar-Constrained GPU Kernel (Triton) Generation

Notebook Flujo TritonBench:
https://colab.research.google.com/drive/1r820uomX4ivjKng-Sw7Vncc6JWq9Rm6R?usp=sharing 


A research pipeline for generating Triton kernels using LLMs and evaluating them with TritonBench using Modal.

---

# Overview

This project:

1. Generates Triton kernel implementations using an LLM provider
2. Evaluates generations using TritonBench
3. Runs remotely on Modal GPU containers
4. Supports different providers/models
5. Uses an end-to-end benchmark pipeline

Current supported workflow:

- Provider: NVIDIA
- Model example:
  - `mistralai/devstral-small-2507`

---

# Requirements

## 1. Install Python

Recommended:

- Python 3.12 or 3.13

Verify:

```bash
python --version
```

---

## 2. Install Git

Verify:

```bash
git --version
```

---

## 3. Install Modal

Official website:

:contentReference[oaicite:0]{index=0}

Install:

```bash
pip install modal
```

Verify:

```bash
modal --version
```

---

# Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd GrammarConstraint-KernelGeneration
```

---

# Modal Authentication

Login to Modal:

```bash
modal token new
```

This opens a browser for authentication.

Verify login:

```bash
modal profile current
```

---

# Create Modal Secret

The NVIDIA provider requires an NVIDIA API key.

Create a secret in Modal:

```bash
modal secret create triton-grammar-constrains NVIDIA_API_KEY=YOUR_API_KEY
```

Verify:

```bash
modal secret list
```

You should see:

```text
triton-grammar-constrains
```

---

# NVIDIA API Key

Obtain an API key from:

:contentReference[oaicite:1]{index=1}

The key is used by the NVIDIA provider backend.

---

# Project Structure

Important directories:

```text
backends/
models/
prompts/
orchestration/
evaluation/
```

Important entrypoint:

```text
backends/modal/entrypoints.py
```

---

# Running the Pipeline

Run:

```bash
modal run backends/modal/entrypoints.py::main --provider nvidia --model mistralai/devstral-small-2507 --limit 1
```

---

# Command Arguments

## Provider

Example:

```bash
--provider nvidia
```

Currently supported:

- `nvidia`

---

## Model

Example:

```bash
--model mistralai/devstral-small-2507
```

Other models can be substituted if supported by the provider.

---

## Limit

Example:

```bash
--limit 1
```

Controls how many TritonBench tasks are evaluated.

Recommended during debugging:

```bash
--limit 1
```

Larger runs:

```bash
--limit 10
```

or:

```bash
--limit 100
```

---

# What Happens Internally

The pipeline performs:

## Phase 1 — Generation

- Loads benchmark task
- Builds prompt
- Sends request to model provider
- Generates Triton implementation

---

## Phase 2 — Call Accuracy

Runs TritonBench call accuracy evaluation.

Checks:

- Function exists
- Signature is correct
- Invocation succeeds

Output example:

```text
call_acc survivors: 1 / 1
```

---

## Phase 3 — Execution Accuracy

Runs execution correctness tests.

Checks:

- Numerical correctness
- Runtime behavior
- Execution success

Output example:

```text
exec_acc survivors: 1 / 1
```

---

## Phase 4 — Efficiency

Runs TritonBench performance evaluation.

Measures:

- Runtime
- Throughput
- Performance metrics

---

# Expected First Run Behavior

The first execution may:

- Build multiple Modal images
- Install PyTorch
- Clone TritonBench
- Patch TritonBench evaluation scripts

This can take several minutes.

Subsequent runs are much faster because Modal caches images.

---

# Important Modal Notes

You may see:

```text
WARNING: The NVIDIA Driver was not detected.
```

during image build stages.

This is normal for non-GPU build containers.

Actual evaluation containers will use GPUs when configured correctly.

---

# TritonBench Compatibility Patch

This project patches TritonBench paths automatically inside Modal images.

Patched files:

```text
/opt/TritonBench/EVAL/eval_T/0_call_acc.py
/opt/TritonBench/EVAL/eval_T/1_exe_acc.py
```

The pipeline redirects:

```python
statis_path
gold_folder
py_folder
py_interpreter
```

to Modal-compatible paths.

No manual TritonBench modifications are required locally.

---

# Common Errors

# 1. Missing NVIDIA_API_KEY

Error:

```text
KeyError: 'NVIDIA_API_KEY'
```

Solution:

Ensure the Modal secret exists:

```bash
modal secret list
```

Ensure the function includes:

```python
secrets=[modal.Secret.from_name("triton-grammar-constrains")]
```

---

# 2. Missing Prompt Template

Error:

```text
FileNotFoundError: prompts/templates/triton_translation.txt
```

Solution:

Ensure the file exists:

```text
prompts/templates/triton_translation.txt
```

---

# 3. Function Not Defined

Error example:

```text
NameError: name 'fused_bmm_rmsnorm_gelu_dropout_sub' is not defined
```

Meaning:

- Infrastructure worked
- Evaluation worked
- Model generation failed

Usually caused by:

- Markdown code fences
- Wrong function name
- Empty output
- Invalid generation

---

# Recommended Debugging

Add temporary logging after generation:

```python
print("=" * 80)
print(prediction[:3000])
print("=" * 80)
```

Also verify generated namespace:

```python
namespace = {}
exec(prediction, namespace)
print(namespace.keys())
```

---

# Recommended Prompt Constraints

Models should be instructed with constraints such as:

```text
Return ONLY valid Python code.

Do not include markdown fences.

Do not include explanations.

Preserve the exact function name.

Return a complete implementation.
```

---

# Example Successful Pipeline Output

```text
=== Phase 1: call accuracy ===

call_acc survivors: 1 / 1

=== Phase 2: execution accuracy ===

exec_acc survivors: 1 / 1

=== Phase 3: efficiency ===
```

---

# Development Workflow

Typical iteration loop:

```bash
modal run backends/modal/entrypoints.py::main --provider nvidia --model mistralai/devstral-small-2507 --limit 1
```

Then:

1. Inspect generated outputs
2. Improve prompts
3. Improve postprocessing
4. Re-run benchmark

---

# Recommended Initial Testing

Use:

```bash
--limit 1
```

until:

- generation format is stable
- function names are preserved
- evaluation passes consistently

Then increase benchmark size.

---

# Useful Modal Commands

## View secrets

```bash
modal secret list
```

---

## Open Modal dashboard

:contentReference[oaicite:2]{index=2}

---

## Check current profile

```bash
modal profile current
```

---

# Current Status

Infrastructure status:

- Modal integration: working
- TritonBench integration: working
- Evaluation pipeline: working
- NVIDIA provider: working

Current remaining challenge:

- Improving model generation quality and formatting consistency

---