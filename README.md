# Grammar-Constrained GPU Kernel (Triton) Generation

Notebook Flujo TritonBench:
https://colab.research.google.com/drive/1r820uomX4ivjKng-Sw7Vncc6JWq9Rm6R?usp=sharing 

# Pipeline Testing Guide

## 1. Install Local Dependencies

Install Modal locally:

```bash
pip install modal
```

---

## 2. Authenticate Modal

```bash
modal setup
```

---

## 3. Create Modal Secrets

Create a single secret containing all provider API keys:

```bash
modal secret create grammar-constrains OPENAI_API_KEY=your_openai_key NVIDIA_API_KEY=your_nvidia_key GEMINI_API_KEY=your_gemini_key
```

---

## 4. Attach Secrets to the Generation Job

File:

```text
backends/modal/jobs/generation.py
```

Add:

```python
secrets=[
    modal.Secret.from_name("triton-grammar-constrains")
]
```

Final decorator:

```python
@app.function(
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: volume},
    secrets=[
        modal.Secret.from_name("tritonforge-llm")
    ],
)
```

---

## 5. Fix NVIDIA Provider

File:

```text
models/providers/nvidia_provider.py
```

Use:

```python
import os

from openai import OpenAI

from models.interfaces.base_provider import BaseProvider


class NvidiaProvider(BaseProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
        )

    def generate(self, messages, model):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=8192,
        )

        return response.choices[0].message.content
```

---

## 6. Fix OpenAI Provider

File:

```text
models/providers/openai_provider.py
```

Add:

```python
api_key=os.environ["OPENAI_API_KEY"]
```

Example:

```python
self.client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
```

---

## 7. Fix Gemini Provider

File:

```text
models/providers/gemini_provider.py
```

Add:

```python
api_key=os.environ["GEMINI_API_KEY"]
```

---

## 8. Restore TritonBench Import Symlinks

File:

```text
backends/modal/image.py
```

Add:

```python
.run_commands(
    (
        f"ln -s "
        f"{REPO_DIR}/EVAL/eval_T/0_call_acc.py "
        f"{REPO_DIR}/EVAL/eval_T/call_acc.py"
    ),

    (
        f"ln -s "
        f"{REPO_DIR}/EVAL/eval_T/1_exe_acc.py "
        f"{REPO_DIR}/EVAL/eval_T/exe_acc.py"
    ),
)
```

These are NOT patches.

They only create importable aliases:

```python
import call_acc
import exe_acc
```

---

## 9. Configure PYTHONPATH

Linux/macOS:

```bash
export PYTHONPATH=.
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="."
```

---

## 10. Run First Smoke Test

From repository root:

```bash
python -m apps.cli.main \
    --provider nvidia \
    --model mistralai/devstral-small-2507 \
    --limit 2
```

---

# Expected Pipeline Flow

```text
Operator
    ↓
Prompt Builder
    ↓
LLM Provider
    ↓
Modal Generation Job
    ↓
Predictions JSONL
    ↓
Modal Evaluation Job
    ↓
call_acc
    ↓
exec_acc
    ↓
speedup
    ↓
summary metrics
```

---

# Expected Behavior

## Step 1

Modal builds the container image.

First build may take several minutes.

---

## Step 2

Generation job starts:

```python
generate_predictions.remote(...)
```

---

## Step 3

Predictions are saved into the Modal Volume.

---

## Step 4

Evaluation job starts on a remote GPU (T4).

---

## Step 5

TritonBench phases execute:

- call accuracy
- execution accuracy
- efficiency benchmark

---

## Step 6

CLI prints a summary:

```json
{
  "call_acc": {
    "passed": 1,
    "rate": 50.0
  },
  "exec_acc": {
    "passed": 1,
    "rate": 50.0
  },
  "speedup": 1.42
}
```

---

# Inspect Modal Artifacts

List files stored in the Modal Volume:

```bash
modal volume ls triton-grammar-constrains-volume
```

Artifacts are stored under:

```text
/data/predictions/
/data/results/
```

---

# Common First Errors

## 1. TritonBench Hardcoded Paths

TritonBench upstream contains bad path assumptions.

If evaluation fails immediately, you will likely need:

- PATCH_CALL_ACC
- PATCH_EXE_ACC

---

## 2. CUDA / Triton Version Mismatch

Can happen depending on Triton version updates.

---

## 3. Invalid Generated Triton

Very common initially.

This is expected.

---

## 4. API Rate Limits

Especially common with Gemini.

---

# Recommended Initial Testing Strategy

DO NOT benchmark the full dataset first.

Recommended progression:

## Stage 1

```bash
--limit 1
```

## Stage 2

```bash
--limit 5
```

## Stage 3

```bash
dataset=simp
```

## Stage 4

```bash
dataset=comp
```

Only scale after the full pipeline works end-to-end.