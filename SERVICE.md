# Servicio de Traducción PyTorch → Triton

Servicio REST dockerizado para traducir funciones PyTorch a kernels Triton usando LLMs (vía NVIDIA API) y validar su compilación/ejecución en GPU a través de Modal.

---

## ¿Qué hace?

Este servicio toma código PyTorch con operaciones como `matmul`, `relu`, `add`, etc., y genera automáticamente un kernel Triton equivalente. Luego valida que ese kernel compile y ejecute correctamente en una GPU real (vía Modal cloud).

El pipeline completo es:

```
PyTorch source code
    │
    ▼
Parse (AST) → Shapes → Context → Fusion → Prompt
    │
    ▼
LLM Generation (NVIDIA API, model: qwen/qwen3.5-397b-a17b)
    │
    ▼
Static Validation (sintaxis, firma, imports)
    │
    ▼
GPU Validation (compilación + ejecución en Modal GPU)
    │
    ▼
Resultado JSON con código generado y métricas
```

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────────┐
│                     Cliente (VS Code, curl, etc.)                │
│     POST /api/v1/translate                                       │
│     POST /api/v1/jobs/{id}/gpu-validate                          │
│     POST /api/v1/evaluate                                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│              Servicio Dockerizado (FastAPI + SQLite)               │
│                                                                    │
│   ┌──────────────────┐     ┌────────────────────────────────┐   │
│   │  API REST          │────▶│  TranslationPipeline           │   │
│   │  FastAPI           │     │  (parse → generate → validate) │   │
│   └──────────────────┘     └────────────────────────────────┘   │
│            │                              │                       │
│   ┌────────▼────────┐          ┌─────────▼──────────┐          │
│   │  SQLite (jobs,   │          │  Modal subprocess   │          │
│   │  kernels)         │          │  (GPU validation)   │          │
│   └──────────────────┘          └─────────────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

**Componentes clave:**

| Componente | Archivo | Función |
|---|---|---|
| API REST | `service/main.py` | Entrypoint FastAPI |
| Rutas | `service/api/routes.py` | Endpoints `/translate`, `/evaluate`, `/jobs/{id}/gpu-validate` |
| Pipeline | `orchestration/translation_pipeline.py` | Lógica de traducción (parse, shapes, context, fusion, prompt, generate, validate) |
| GPU Validator | `service/modal_gpu_validator.py` | Entrypoint Modal que llama a GPU cloud |
| DB | `service/db/` | SQLite con tablas `jobs` y `kernels` |
| Config | `.env` | Variables de entorno (NVIDIA API key, Modal tokens) |

---

## Requisitos previos

- **Docker** y **docker-compose** instalados.
- **NVIDIA_API_KEY**: API key de [NVIDIA](https://build.nvidia.com/) para generación con LLM.
- **MODAL_TOKEN_ID** y **MODAL_TOKEN_SECRET**: Tokens de [Modal](https://modal.com/) para validación GPU.

---

## Ejecución paso a paso

### 1. Configurar variables de entorno

Copia el template y rellena tus claves:

```bash
cp .env.example .env
```

Edita `.env`:

```bash
# .env (no se sube a Git gracias a .gitignore)
NVIDIA_API_KEY=nvapi-tu-clave-real-aqui
MODAL_TOKEN_ID=tk-tu-token-id
MODAL_TOKEN_SECRET=ts-tu-token-secret
```

### 2. Iniciar el servicio

```bash
docker-compose up --build
```

**Nota:** El primer build descarga ~3GB de dependencias (PyTorch, CUDA, transformers, etc.). Puede tardar 15-20 minutos. Builds posteriores usan cache.

### 3. Verificar que el servicio está corriendo

```bash
curl http://localhost:8000/api/v1/health
```

Debería devolver:

```json
{"status":"ok","db_connected":true,"version":"0.1.0"}
```

### 4. Usar el servicio

#### 4.1 Traducir PyTorch a Triton

```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "source_code": "def linear_relu(x, weight, bias):\n    z = x @ weight.T + bias\n    return torch.relu(z)",
    "provider": "nvidia-grammar",
    "dims": {"N": 128, "D_in": 256, "D_out": 512}
  }' | python3 -m json.tool
```

**Respuesta:**

```json
{
  "job_id": "...",
  "status": "completed",
  "generated_code": "import triton\n...",
  "validation": {"passed": true, "errors": [], "warnings": []},
  "gpu_validation": null
}
```

#### 4.2 Validar en GPU (paso separado, toma 2-5 min)

```bash
curl -X POST http://localhost:8000/api/v1/jobs/{job_id}/gpu-validate | python3 -m json.tool
```

**Respuesta:**

```json
{
  "compilation_pass": true,
  "execution_pass": true,
  "output_shape": "(128, 512)",
  "device": "cuda",
  "errors": []
}
```

#### 4.3 Consultar un job

```bash
curl http://localhost:8000/api/v1/runs/{job_id} | python3 -m json.tool
```

#### 4.4 Listar jobs anteriores

```bash
curl "http://localhost:8000/api/v1/runs?limit=10" | python3 -m json.tool
```

#### 4.5 Evaluar numéricamente (comparar vs PyTorch original)

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"job_id": "...", "dims": {"N": 128, "D_in": 256, "D_out": 512}}' | python3 -m json.tool
```

**Respuesta:**

```json
{
  "job_id": "...",
  "accuracy_pass": true,
  "max_error": 1e-06,
  "speedup": 2.3,
  "errors": []
}
```

---

## Endpoints disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/v1/translate` | Traduce PyTorch a Triton |
| `POST` | `/api/v1/jobs/{job_id}/gpu-validate` | Compila y ejecuta en GPU via Modal |
| `POST` | `/api/v1/evaluate` | Evaluación numérica (accuracy + speedup) |
| `GET`  | `/api/v1/runs/{job_id}` | Detalle de un job |
| `GET`  | `/api/v1/runs` | Lista jobs con paginación |
| `GET`  | `/api/v1/health` | Healthcheck del servicio |

**Documentación interactiva:** Abre `http://localhost:8000/docs` para Swagger UI.

---

## Flujo típico

1. **El usuario envía código PyTorch** a `POST /translate`.
2. El servicio parsea el código, resuelve shapes, busca contexto de operadores, planifica fusión.
3. Construye un prompt enriquecido y lo envía al LLM (NVIDIA API).
4. Recibe código Triton, valida sintaxis estática.
5. Devuelve resultado al usuario.
6. **(Opcional)** El usuario llama `POST /jobs/{id}/gpu-validate` para ejecutar compilación + smoke test en una GPU real de Modal.
7. El servicio ejecuta `modal run service/modal_gpu_validator.py` como subprocess, que internamente llama `translate_validation.remote()` en GPU cloud.
8. El resultado de GPU se guarda en SQLite y se devuelve al usuario.

---

## Limitaciones conocidas

| Limitación | Detalle |
|------------|---------|
| **GPU Validation requiere paso separado** | No ocurre automáticamente en `/translate`. El usuario debe llamar explícitamente `POST /jobs/{id}/gpu-validate`. |
| **Cold-start de Modal GPU** | La primera validación GPU puede tardar 2-5 minutos mientras Modal arranca el contenedor GPU. |
| **Calidad del código generado** | Depende del LLM. Puede producir typos o kernels incompletos. El servicio reporta estos errores en `validation.errors`. |
| **Solo funciones puras** | No soporta `nn.Module`, estado interno, o control de flujo dinámico (`if`/`for`). |

---

## Desarrollo (cambios sin rebuild)

El `docker-compose.yml` usa **volume mounts** para que los cambios de código se reflejen instantáneamente:

```yaml
volumes:
  - ./service:/app/service
  - ./orchestration:/app/orchestration
  - ./backends:/app/backends
  # ... etc
```

Si editas archivos Python localmente, solo necesitas reiniciar:

```bash
docker-compose restart
```

---

## Detener el servicio

```bash
docker-compose down
```

Para limpiar también imágenes y volúmenes:

```bash
docker-compose down --rmi all -v
```

---

## Notas adicionales

- **Modelo default:** `qwen/qwen3.5-397b-a17b`. Es configurable en el request pero el default siempre apunta a este modelo.
- **Base de datos:** SQLite se almacena en `./data/service.db` (montado como volumen).
- **Artefactos de debug:** El pipeline guarda artefactos en `debug/translations/<run_id>/`.
