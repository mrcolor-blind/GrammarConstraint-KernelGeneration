# VS Code Extension Integration

**How the VS Code extension consumes the Translation Service.**

## Service Base URL

```
http://localhost:8000/api/v1
```

## Endpoints

### 1. `POST /translate` — Translate PyTorch to Triton

**Minimal request (uses default model):**
```typescript
const response = await fetch('http://localhost:8000/api/v1/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    source_code: userCode,           // string: the Python function
    provider: 'nvidia-grammar',      // optional, default
    // model is optional, defaults to qwen/qwen3.5-397b-a17b
  })
});

const result = await response.json();
// result.generated_code contains the Triton kernel
```

**With concrete dimensions:**
```typescript
const response = await fetch('http://localhost:8000/api/v1/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    source_code: userCode,
    provider: 'nvidia-grammar',
    dims: { N: 128, D_in: 256, D_out: 512 }
  })
});
```

**Important fields in the response:**

| Field | Type | Meaning |
|-------|------|---------|
| `job_id` | `string` | Unique identifier for this translation |
| `status` | `string` | `completed` or `failed` |
| `generated_code` | `string \| null` | The Triton kernel code (null if failed) |
| `validation.passed` | `boolean` | Static validation result |
| `validation.errors` | `string[]` | Syntax / signature errors |
| `validation.warnings` | `string[]` | Non-blocking warnings |
| `gpu_validation` | `object \| null` | `null` initially — use `POST /jobs/{id}/gpu-validate` to populate |
| `errors` | `string[]` | Pipeline-level errors (e.g. missing API key) |

### 2. `POST /jobs/{job_id}/gpu-validate` — GPU Validation (Modal)

Run compilation + execution smoke test on a real GPU via Modal.

```typescript
const response = await fetch(`http://localhost:8000/api/v1/jobs/${result.job_id}/gpu-validate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' }
});

const gpuResult = await response.json();
// gpuResult.compilation_pass  → boolean
// gpuResult.execution_pass    → boolean
// gpuResult.output_shape      → string (e.g. "(128, 512)")
// gpuResult.device            → string (e.g. "cuda")
// gpuResult.errors            → string[]
```

**How it works internally:**
The service runs `modal run service/modal_gpu_validator.py` as a subprocess inside the Docker container. The Modal local entrypoint reads the generated code from the database, calls `translate_validation.remote()` on a Modal GPU in the cloud, and returns the result.

**Timeout:** This endpoint can take 2–5 minutes (Modal GPU cold-start + compilation + execution). Use `AbortController` with a 6-minute timeout.

### 3. `POST /evaluate` — Numerical Evaluation

Compare the generated Triton kernel against the original PyTorch function.

```typescript
const response = await fetch('http://localhost:8000/api/v1/evaluate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    job_id: result.job_id,           // from the translate response
    dims: { N: 128, D_in: 256, D_out: 512 }
  })
});

const evalResult = await response.json();
// evalResult.accuracy_pass  → boolean
// evalResult.max_error      → number (L∞ error)
// evalResult.speedup        → number (Triton time / PyTorch time)
```

### 4. `GET /runs/{job_id}` — Retrieve Job Details

```typescript
const response = await fetch(`http://localhost:8000/api/v1/runs/${jobId}`);
const job = await response.json();
```

### 5. `GET /runs` — List Previous Jobs

```typescript
const response = await fetch('http://localhost:8000/api/v1/runs?limit=10');
const list = await response.json();
// list.total → number of jobs
// list.items → array of job summaries
```

## Error Handling

**Timeout:**
- `/translate`: ~30–90 seconds. Use 120-second timeout.
- `/jobs/{id}/gpu-validate`: ~2–5 minutes. Use 360-second timeout (Modal GPU cold-start).

**Common error responses:**

| Status | Body | Meaning |
|--------|------|---------|
| `200` | `status: "failed"`, `errors: [...]` | Pipeline failed (e.g. no API key, invalid code) |
| `404` | `detail: "Job not found"` | Wrong `job_id` |
| `422` | `detail: [...]` | Invalid request body (FastAPI validation) |

## UI Flow Suggestion

1. **User selects a `@triton`-annotated function** in the editor.
2. **Extension shows a "Translate to Triton" button** (e.g. in a CodeLens or context menu).
3. **On click, extension sends `POST /translate`** with the selected code.
4. **Show a progress indicator** (the call is synchronous but may take a while).
5. **On response:**
   - If `status === "completed"` and `generated_code` exists → open a new editor tab with the Triton code.
   - If `validation.warnings.length > 0` → show warnings in a notification.
   - If `status === "failed"` → show `errors` in an error message.
6. **"GPU Validate" button** (after translation succeeds) → sends `POST /jobs/{id}/gpu-validate` and shows compilation + execution result.
7. **Optional: "Evaluate" button** → sends `POST /evaluate` and shows accuracy + speedup in a webview panel.

## Artifacts

The service also writes debug artifacts to `debug/translations/<run_id>/` inside the container (and mounted to the host via `docker-compose.yml`).

These are useful for debugging but not required for the extension workflow.

## Prerequisites

Before the extension can call the service:

1. **Copy `.env.example` to `.env` and fill in your keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your NVIDIA_API_KEY and Modal tokens
   ```
2. **Start the service:**
   ```bash
   docker-compose up --build
   ```
3. **Health check should pass:**
   ```bash
   curl http://localhost:8000/api/v1/health
   # → {"status":"ok","db_connected":true,"version":"0.1.0"}
   ```
