# SERVICE-PLAN.md

## PyTorch → Triton Translation Service

**Status:** Draft — Pendiente de revisión con el equipo.  
**Autor:** Adolfo  
**Fecha:** 2026-06-05

---

## 1. Objetivo

Dockerizar el pipeline de traducción PyTorch → Triton como un **servicio REST** con FastAPI y una **mini base de datos SQLite**, listo para ser consumido por una extensión de VS Code (Task 3 de Arturo y Adolfo).

**Scope:**
- Exponer los endpoints que necesita la extensión: `translate` y `evaluate`.
- Persistir jobs y kernels en una DB mínima.
- Reutilizar al máximo el código existente del proyecto.
- Sin autenticación (entorno de desarrollo interno).

---

## 2. Contexto del Proyecto Base

El proyecto `GrammarConstraint-KernelGeneration` ya tiene:

| Componente | Ubicación | Estado |
|---|---|---|
| Pipeline de traducción | `orchestration/translation_pipeline.py` | Funcional (Parse → Shapes → Context → Fusion → Prompt → Generate → Validate → GPU Validate) |
| CLI | `apps/cli/main.py` | Comandos: `translate`, `inspect`, `evaluate`, `benchmark` |
| Providers LLM | `models/providers/` | NVIDIA (con/sin grammar), OpenAI, Gemini, vLLM |
| Validación GPU | `backends/modal/jobs/translate_validation.py` | Ejecuta en Modal via `@benchmark_app.function(...)` |
| Gramática EBNF | `grammars/triton_kernel.ebnf` | Para guided decoding con `nvidia-grammar` |
| Modelo de dominio | `models/domain.py` | `PipelineContext`, `ValidationResult`, `GpuValidationResult`, etc. |

**Este servicio NO reescribe el pipeline.** Lo envuelve en una API REST.

---

## 3. Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cliente (VS Code Extension)                        │
│    POST /api/v1/translate   → traducir PyTorch a Triton             │
│    POST /api/v1/evaluate    → evaluar un kernel generado            │
│    GET  /api/v1/runs/{id}   → consultar estado/resultado            │
│    GET  /api/v1/runs        → listar traducciones previas           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────────┐
│                    Servicio Dockerizado (FastAPI)                     │
│  ┌──────────────────┐    ┌─────────────────────────────────────┐   │
│  │  API Layer         │    │  Pipeline Engine (reutilizado)      │   │
│  │  - Validación      │───→│  - TranslationPipeline              │   │
│  │  - Serialización   │    │  - NvidiaProvider (local)           │   │
│  │  - Respuestas JSON │    │  - ValidationStage                  │   │
│  └──────────────────┘    └─────────────────────────────────────┘   │
│           │                              │                          │
│  ┌────────▼────────┐        ┌───────────▼────────────┐             │
│  │  SQLite (mini DB)│        │  Modal (GPU opcional)  │             │
│  │  - jobs table    │        │  - translate_validation  │             │
│  │  - kernels table │        │  - compilation + exec    │             │
│  └──────────────────┘        └────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

**Flujo típico:**
1. VS Code envía código PyTorch a `POST /api/v1/translate`.
2. El servicio crea un `job` en SQLite con `status='running'`.
3. Ejecuta `TranslationPipeline` localmente (usa `NVIDIA_API_KEY` del contenedor).
4. Si `gpu_validate=true`, delega la validación GPU a Modal (reutiliza `translate_validation.remote`).
5. Guarda `generated_code`, `validation_json`, `gpu_validation_json` en SQLite.
6. Actualiza `status='completed'` y devuelve el resultado JSON.
7. VS Code muestra el código Triton en un editor lateral.
8. (Opcional) VS Code llama `POST /api/v1/evaluate` para comparar numéricamente vs PyTorch.

---

## 4. Modelo de Datos (SQLite)

**Dos tablas.** SQLite es un archivo local (`/app/data/service.db`), sin necesidad de otro contenedor. Ideal para MVP.

### 4.1 Tabla `jobs`

Una fila por cada request de `translate` o `evaluate`.

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | UUID v4 (ej: `550e8400-e29b-...`) |
| `type` | `TEXT NOT NULL` | `'translate'` \| `'evaluate'` |
| `status` | `TEXT NOT NULL` | `'pending'` \| `'running'` \| `'completed'` \| `'failed'` |
| `provider` | `TEXT` | Ej: `nvidia-grammar`, `nvidia` |
| `model` | `TEXT` | Ej: `qwen/qwen3.5-397b-a17b` |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` |
| `completed_at` | `TIMESTAMP` | Se actualiza al terminar |
| `source_code` | `TEXT` | Código PyTorch original (JSON string) |
| `dims` | `TEXT` | Dimensiones concretas, ej: `"N=128,D_in=256"` |
| `run_id` | `TEXT` | Run ID interno del pipeline (ej: `2026-06-05_14-22-10_linear_relu`) |
| `generated_code` | `TEXT` | Código Triton resultante |
| `validation_json` | `TEXT` | `ValidationResult` serializado a JSON |
| `gpu_validation_json` | `TEXT` | `GpuValidationResult` serializado a JSON |
| `errors` | `TEXT` | Lista de errores (JSON array) |

### 4.2 Tabla `kernels`

Código generado reusable (para evaluate y listing).

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | UUID v4 |
| `job_id` | `TEXT NOT NULL` | FK → `jobs(id)` |
| `function_name` | `TEXT` | Nombre de la función (ej: `linear_relu`) |
| `source_code` | `TEXT` | Código PyTorch original |
| `generated_code` | `TEXT` | Código Triton generado |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` |

**Razón de dos tablas:** `jobs` es el log de operaciones; `kernels` es el catálogo de resultados reutilizables. En el futuro se puede buscar kernels por función sin recorrer todo el historial de jobs.

---

## 5. API REST

Base URL: `http://localhost:8000/api/v1`

Auto-documentación: `http://localhost:8000/docs` (Swagger UI de FastAPI).

### 5.1 `POST /translate` — Traducir PyTorch a Triton

**Request body (JSON):**

```json
{
  "source_code": "def linear_relu(x, weight, bias):\n    z = x @ weight.T + bias\n    return torch.relu(z)",
  "provider": "nvidia-grammar",
  "model": "qwen/qwen3.5-397b-a17b",
  "dims": {"N": 128, "D_in": 256, "D_out": 512},
  "gpu_validate": true
}
```

**Campos:**

| Campo | Tipo | Requerido | Default | Descripción |
|---|---|---|---|---|
| `source_code` | `string` | Sí | — | Código Python con la función `@triton` anotada |
| `provider` | `string` | No | `nvidia-grammar` | Provider LLM a usar |
| `model` | `string` | No | `qwen/qwen3.5-397b-a17b` | **Siempre este modelo por defecto** |
| `dims` | `dict[str, int]` | No | `{}` | Dimensiones concretas para shapes simbólicas |
| `gpu_validate` | `boolean` | No | `false` | ¿Ejecutar compilación + smoke test en Modal GPU? |

**Response (200 OK, síncrono):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "provider": "nvidia-grammar",
  "model": "qwen/qwen3.5-397b-a17b",
  "run_id": "2026-06-05_14-22-10_linear_relu",
  "source_code": "def linear_relu(x, weight, bias): ...",
  "generated_code": "import triton\n...",
  "validation": {
    "passed": true,
    "errors": [],
    "warnings": ["BLOCK_SIZE may be suboptimal"]
  },
  "gpu_validation": {
    "compilation_pass": true,
    "execution_pass": true,
    "output_shape": "(128, 512)",
    "device": "cuda",
    "errors": []
  },
  "artifacts_url": "debug/translations/2026-06-05_14-22-10_linear_relu/",
  "created_at": "2026-06-05T14:22:10Z",
  "completed_at": "2026-06-05T14:23:45Z"
}
```

**Nota sobre síncrono vs async:**
- El pipeline puede tardar 30–120 segundos (LLM API + validación GPU).
- Para el MVP, el endpoint es **síncrono** (FastAPI maneja el timeout). VS Code espera.
- Si en el futuro necesitamos escalar, se puede convertir a **async** con `background_tasks` y polling (`GET /runs/{id}`). Por ahora, KISS.

### 5.2 `POST /evaluate` — Evaluación numérica

Compara el kernel Triton generado contra la función PyTorch original en un dataset de prueba.

**Request body (JSON):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "dims": {"N": 128, "D_in": 256, "D_out": 512}
}
```

**Response (200 OK):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "accuracy_pass": true,
  "max_error": 1.0e-06,
  "speedup": 2.3,
  "errors": []
}
```

### 5.3 `GET /runs/{job_id}` — Consultar un job

Devuelve el estado y el resultado completo de un job (translate o evaluate).

### 5.4 `GET /runs` — Listar jobs

Lista todos los jobs. Parámetros de query opcionales:

| Query param | Tipo | Descripción |
|---|---|---|
| `status` | `string` | Filtrar por `pending`, `running`, `completed`, `failed` |
| `type` | `string` | Filtrar por `translate`, `evaluate` |
| `limit` | `int` | Default: 50 |
| `offset` | `int` | Default: 0 |

**Response:**

```json
{
  "total": 127,
  "limit": 50,
  "offset": 0,
  "items": [
    { "job_id": "...", "status": "completed", "type": "translate", "function_name": "linear_relu", "created_at": "..." },
    ...
  ]
}
```

### 5.5 `GET /health` — Healthcheck

```json
{
  "status": "ok",
  "db_connected": true,
  "version": "0.1.0"
}
```

---

## 6. Estructura de Archivos Nuevos

Todo el servicio vive bajo `service/`. No tocamos el código existente, solo lo importamos.

```
service/
├── main.py              # FastAPI app, entrypoint uvicorn
├── api/
│   ├── __init__.py
│   ├── routes.py        # Endpoints: translate, evaluate, runs, health
│   └── schemas.py       # Pydantic models (Request/Response/JobOut/KernelOut)
├── db/
│   ├── __init__.py
│   ├── database.py      # Engine SQLite, SessionLocal, creación de tablas
│   └── crud.py          # Funciones: create_job, get_job, list_jobs, update_job, save_result
├── models/
│   ├── __init__.py
│   └── sqlalchemy_models.py  # Tablas Job y Kernel (SQLAlchemy ORM)
├── core/
│   ├── __init__.py
│   ├── config.py        # Settings pydantic (env vars, DEFAULT_MODEL)
│   └── pipeline_runner.py  # Wrapper: invoca TranslationPipeline + guarda en DB
├── Dockerfile
└── docker-compose.yml
```

**Archivos existentes que se reutilizan (sin tocar):**

| Archivo existente | Cómo se reutiliza |
|---|---|
| `orchestration/translation_pipeline.py` | `PipelineRunner` crea una instancia de `TranslationPipeline` y llama `.run()` |
| `models/registry/model_registry.py` | `load_provider()` para seleccionar provider (nvidia-grammar, etc.) |
| `backends/modal/jobs/translate_validation.py` | Si `gpu_validate=true`, se llama a `translate_validation.remote()` |
| `evaluation/translate_evaluator.py` | Endpoint `/evaluate` importa `run_local_evaluation()` |

---

## 7. Plan de Implementación Paso a Paso

### **Paso 1: Estructura y dependencias** (30 min)
- Crear directorio `service/` con subdirectorios (`api/`, `db/`, `models/`, `core/`).
- Añadir al `requirements.txt` del proyecto raíz: `fastapi`, `uvicorn[standard]`, `sqlalchemy`, `pydantic-settings`.
- Crear `service/core/config.py` con `Settings` de Pydantic y `DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"`.

### **Paso 2: Capa de Base de Datos** (45 min)
- `service/db/database.py`: Engine SQLite apuntando a `/app/data/service.db`, `SessionLocal`, `get_db()`.
- `service/models/sqlalchemy_models.py`: Clases `Job` y `Kernel` (SQLAlchemy declarative base).
- `service/db/crud.py`: `create_job()`, `get_job()`, `list_jobs()`, `update_job_status()`, `save_job_result()`.
- `service/main.py` (temporal): Healthcheck que verifique que la DB se crea correctamente.

### **Paso 3: Pipeline Runner** (45 min)
- `service/core/pipeline_runner.py`:
  - `run_translation(job_id, source_code, provider, model, dims, gpu_validate)`.
  - Crea instancia de `TranslationPipeline`.
  - Maneja `try/except`: si el pipeline falla, guarda errores en DB y marca `status='failed'`.
  - Si `gpu_validate=True`, invoca `translate_validation.remote()` (Modal) — reutiliza el código existente.
  - Guarda todo el resultado en DB vía funciones CRUD.
- Usa el modelo default si `model` es `None`.

### **Paso 4: API REST (FastAPI)** (60 min)
- `service/api/schemas.py`: `TranslateRequest`, `TranslateResponse`, `EvaluateRequest`, `EvaluateResponse`, `JobOut`, `JobListResponse`, `HealthResponse`.
- `service/api/routes.py`:
  - `POST /api/v1/translate` → llama a `pipeline_runner.run_translation()`.
  - `POST /api/v1/evaluate` → recupera kernel de DB, ejecuta `run_local_evaluation()`.
  - `GET /api/v1/runs/{job_id}` → `crud.get_job()`.
  - `GET /api/v1/runs` → `crud.list_jobs()`.
  - `GET /api/v1/health` → verifica DB.
- `service/main.py`: Crea `FastAPI(app)`, incluye `routes.router`, evento startup para crear tablas.

### **Paso 5: Dockerización** (30 min)
- **`Dockerfile`** (en `service/`):
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- **`docker-compose.yml`** (en raíz):
  ```yaml
  version: '3.8'
  services:
    api:
      build:
        context: .
        dockerfile: service/Dockerfile
      ports:
        - "8000:8000"
      environment:
        - NVIDIA_API_KEY=${NVIDIA_API_KEY}
        - MODAL_TOKEN_ID=${MODAL_TOKEN_ID}
        - MODAL_TOKEN_SECRET=${MODAL_TOKEN_SECRET}
      volumes:
        - ./debug:/app/debug
        - ./data:/app/data   # SQLite persistente
  ```

### **Paso 6: Pruebas manuales** (30 min)
- `docker-compose up --build`
- `curl http://localhost:8000/api/v1/health`
- `curl -X POST http://localhost:8000/api/v1/translate -H "Content-Type: application/json" -d '{"source_code": "...", "provider": "nvidia-grammar"}'`
- Verificar que `debug/translations/` y `data/service.db` se crean.

### **Paso 7: Documentación de integración VS Code** (15 min)
- Agregar sección en este mismo archivo (o un `VS_CODE_INTEGRATION.md`) con:
  - URL base del servicio.
  - Ejemplos de fetch en TypeScript.
  - Manejo de errores (timeout de 120s).
  - Cómo mostrar el resultado en un webview panel de VS Code.

**Tiempo total estimado: ~4.25 horas de trabajo concentrado.**

---

## 8. Decisiones de Diseño

| Aspecto | Decisión | Razón |
|---|---|---|
| **Framework API** | **FastAPI** | Async nativo, auto-docs Swagger/OpenAPI, estándar en Python, fácil de integrar con VS Code via HTTP |
| **Base de datos** | **SQLite** (archivo local) | Mini, sin contenedor extra, suficiente para MVP. Si escala, migración a PostgreSQL es trivial con SQLAlchemy |
| **Pipeline de generación** | **Local en el contenedor** | El usuario especificó "local". Requiere `NVIDIA_API_KEY` en el entorno del contenedor. Las llamadas a la API de NVIDIA salen directamente del contenedor |
| **Validación GPU** | **Delegar a Modal** | Reutiliza `translate_validation.remote()` existente. El contenedor no necesita GPU propia |
| **Autenticación** | **Ninguna** | Acordado para desarrollo interno. Si se necesita en el futuro, se añade un middleware de API key simple |
| **Modelo default** | **`qwen/qwen3.5-397b-a17b`** | Decisión del equipo. El campo `model` es opcional en la API y siempre cae en este valor |
| **Síncrono vs Async** | **Síncrono para MVP** | El pipeline tarda ~30–120s. VS Code espera. Si se necesita escala, se convierte a background tasks + polling |
| **Docker** | **Docker + docker-compose** | Un solo contenedor para la API. SQLite es un volumen montado. No se necesita orquestación compleja |

---

## 9. Variables de Entorno Necesarias

El servicio lee del entorno del contenedor (pasadas via `docker-compose.yml` o `.env`):

| Variable | Requerida | Descripción |
|---|---|---|
| `NVIDIA_API_KEY` | **Sí** | API key de NVIDIA Inference API. Necesaria para que `NvidiaProvider` haga llamadas a `integrate.api.nvidia.com` |
| `MODAL_TOKEN_ID` | No (solo si `gpu_validate=true`) | Token ID de Modal para validación GPU remota |
| `MODAL_TOKEN_SECRET` | No (solo si `gpu_validate=true`) | Token secret de Modal |
| `DATABASE_URL` | No | Default: `sqlite:///data/service.db`. Se puede sobrescribir |

**Ejemplo `.env`:**
```bash
NVIDIA_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
MODAL_TOKEN_ID=tk-XXXXXXXX
MODAL_TOKEN_SECRET=ts-XXXXXXXX
```

---

## 10. Ajustes a Archivos Existentes del Proyecto

| # | Archivo | Ajuste |
|---|---|---|
| 1 | `requirements.txt` (raíz) | Añadir: `fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`, `sqlalchemy>=2.0.0`, `pydantic-settings>=2.0.0` |
| 2 | `.gitignore` (raíz) | Añadir: `/data/`, `*.db` para no commitear la base de datos SQLite |

**Ningún archivo Python existente se modifica.** Todo el servicio es código nuevo bajo `service/`.

---

## 11. Integración con VS Code Extension (Task 3)

La extensión (desarrollada por Arturo y Adolfo) consume el servicio via HTTP estándar.

### 11.1 Flujo típico en la extensión

```typescript
// 1. El usuario selecciona una función PyTorch en el editor
// 2. La extensión extrae el texto del editor
const sourceCode = editor.document.getText(selection);

// 3. POST /translate
const response = await fetch('http://localhost:8000/api/v1/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    source_code: sourceCode,
    provider: 'nvidia-grammar',
    // model es opcional, usa default qwen/qwen3.5-397b-a17b
    gpu_validate: true
  })
});

const result = await response.json();

// 4. Mostrar código Triton generado en un panel lateral (webview)
// result.generated_code contiene el código Triton

// 5. (Opcional) Evaluar numéricamente
const evalResponse = await fetch('http://localhost:8000/api/v1/evaluate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    job_id: result.job_id,
    dims: { N: 128, D_in: 256, D_out: 512 }
  })
});

const evalResult = await evalResponse.json();
// evalResult.accuracy_pass, evalResult.speedup
```

### 11.2 Notas para el equipo de VS Code

- **Timeout:** Usar `AbortController` con timeout de 120 segundos (el pipeline puede tardar).
- **Errores:** Si `status === 'failed'`, mostrar `errors` al usuario.
- **Warnings:** Mostrar `validation.warnings` como advertencias no bloqueantes.
- **GPU:** Si `gpu_validate=false`, no hay `gpu_validation` en la respuesta.
- **Artifacts:** `artifacts_url` apunta a la carpeta local `debug/translations/<run_id>/`.

---

## 12. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| **Timeout de VS Code** (120s no es suficiente) | Media | Alto | Hacer el endpoint async con background tasks en Paso 2 si se detecta que tarda más de 60s |
| **NVIDIA API key expuesta** en el contenedor | Baja | Medio | Usar Docker secrets o Modal secrets en producción. Para dev, `.env` local |
| **SQLite no escala** a múltiples workers | Baja | Medio | FastAPI por defecto corre 1 worker. Si se necesitan más, migrar a PostgreSQL |
| **Modal no disponible** (gpu_validate falla) | Media | Bajo | `gpu_validate` es opcional. El pipeline sigue funcionando sin validación GPU |
| **Pipeline local consume mucha memoria** | Baja | Medio | Monitorear. El contenedor de Docker puede tener `memory_limit` configurado |

---

## 13. Checklist de Aprobación

Antes de empezar la implementación, el equipo debe revisar y aprobar:

- [ ] **Arquitectura:** ¿Están de acuerdo con FastAPI + SQLite + contenedor único?
- [ ] **API:** ¿Los endpoints `/translate`, `/evaluate`, `/runs` cubren las necesidades de la extensión?
- [ ] **Modelo default:** ¿Confirmado `qwen/qwen3.5-397b-a17b` como default?
- [ ] **Autenticación:** ¿OK sin auth para el MVP?
- [ ] **Scope:** ¿Solo translate + evaluate, o necesitan también `benchmark`/`inspect`?
- [ ] **Prioridad:** ¿Empezamos con Pasos 1–5 (servicio funcional) antes de documentar integración VS Code?

---

## 14. Timeline Tentativo

| Día | Qué | Responsable |
|---|---|---|
| **Día 1 (hoy)** | Revisar y aprobar este plan con el equipo | Adolfo + Arturo |
| **Día 1–2** | Implementar Pasos 1–4 (estructura, DB, runner, API) | Adolfo |
| **Día 2–3** | Dockerizar y probar manualmente (Paso 5–6) | Adolfo |
| **Día 3** | Documentar integración VS Code (Paso 7) | Adolfo + Arturo |
| **Día 3–4** | Arturo integra la extensión con el servicio | Arturo |
| **Día 4–5** | Testing conjunto y ajustes | Adolfo + Arturo |

---

## 15. Notas para el Equipo

**Arturo (Task 1 — Testear pipeline):**
> Por favor revisa si los endpoints `/translate` y `/evaluate` cubren todo lo que necesitas para la extensión. ¿Falta algún campo en el request/response? ¿Necesitas `DELETE /runs/{id}` o algún otro endpoint?

**Arturo + Adolfo (Task 3 — VS Code Extension):**
> La sección 11 tiene el sketch de cómo la extensión consume el servicio. ¿Necesitan WebSocket en vez de HTTP polling? ¿Prefieren que `/translate` sea async (devuelve job_id inmediato y luego hacen polling)?

**Dudas o cambios:**
> Si quieren ajustar algo (cambiar SQLite por PostgreSQL, añadir auth, hacer async, etc.), este es el momento. Una vez aprobado, ejecuto el plan paso a paso.

---

*Plan escrito para revisión del equipo. Aprobación pendiente.*
