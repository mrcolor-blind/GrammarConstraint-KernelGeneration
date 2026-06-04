# Plan: Traducción Automática de PyTorch → Triton

## Tabla de Contenidos

1. [Problema](#1-problema)
2. [Estado Actual del Proyecto](#2-estado-actual-del-proyecto)
3. [El Desafío Central: El Vacío de Contexto](#3-el-desafío-central-el-vacío-de-contexto)
4. [Visión General de la Solución](#4-visión-general-de-la-solución)
5. [Decisiones de Diseño](#5-decisiones-de-diseño)
6. [Formato de Entrada del Usuario](#6-formato-de-entrada-del-usuario)
7. [Arquitectura Detallada](#7-arquitectura-detallada)
8. [Diseño del Prompt](#8-diseño-del-prompt)
9. [Sistema de Reparación Modular (3 intentos por etapa)](#9-sistema-de-reparación-modular-3-intentos-por-etapa)
10. [Debugging y Trazabilidad](#10-debugging-y-trazabilidad)
11. [Criterios de Éxito: Niveles 1 al 6](#11-criterios-de-éxito-niveles-1-al-6)
12. [Orden de Implementación (P0–P9)](#12-orden-de-implementación-p0p9)
13. [Estructura de Archivos](#13-estructura-de-archivos)
14. [Riesgos y Mitigaciones](#14-riesgos-y-mitigaciones)
15. [Apéndice: Ejemplos de Referencia](#15-apéndice-ejemplos-de-referencia)

---

## 1. Problema

Queremos que un usuario pueda escribir código PyTorch paralelizable en GPU y traducirlo automáticamente a **Triton**, un lenguaje/compilador para escribir kernels de GPU de alto rendimiento.

### ¿Por qué Triton?

Triton permite escribir kernels de GPU en Python (similar a CUDA pero mucho más productivo) con rendimiento cercano a kernels escritos a mano en CUDA. Es el backend que usan proyectos como PyTorch 2.0 (`torch.compile`), Flash Attention, y vLLM.

### ¿Por qué traducir a Triton en vez de usar `torch.compile`?

`torch.compile` es una caja negra. Queremos darle al usuario **control y transparencia** sobre el kernel generado. También queremos que el usuario pueda **fusionar operaciones** (combinar múltiples operadores en un solo kernel) para eliminar viajes redondos a memoria global, que es donde se gana rendimiento real.

### El caso de uso concreto

- El usuario escribe una función PyTorch que contiene operaciones como `matmul`, `add`, `relu`, `conv2d`, `batch_norm`, etc.
- El sistema analiza ese código, entiende qué hace cada operación, decide cuáles fusionar, y genera kernels Triton equivalentes.
- El resultado es un archivo `.py` auto-contenido con kernels Triton que reemplazan la función original.

---

## 2. Estado Actual del Proyecto

El proyecto actualmente tiene un pipeline funcional para el caso **más simple** de traducción: operador único con contexto preexistente. Esto es lo que ya funciona:

```
TritonBench JSON (descripción pre-escrita de 1 operador)
        │
        ▼
   TritonPromptBuilder  ──►  LLM (NVIDIA API)  ──►  Código Triton
        │
        ▼
   Evaluación con TritonBench (call accuracy, exec accuracy, efficiency)
        │
        ▼
   Modal (GPUs en la nube)
```

**Componentes existentes que se reutilizarán:**

| Componente | Archivo | Qué hace |
|------------|---------|----------|
| Proveedores LLM | `models/providers/*` | OpenAI, NVIDIA, Gemini — ya funcionan |
| Cargador TritonBench | `datasets/tritonbench/loader.py` | Carga `TritonBench_T_simp_alpac_v1.json` |
| Prompt builder actual | `prompts/builders/triton_prompt_builder.py` | Construye mensajes para 1 operador |
| Extracción de código | `models/providers/nvidia_provider.py` | Limpia markdown fences, reasoning tokens |
| Evaluación | `evaluation/`, `backends/modal/jobs/` | Call accuracy, exec accuracy, efficiency en GPU |
| Pipelines | `orchestration/pipelines/` | Orquesta generación + evaluación |

**Lo que NO existe aún** y necesitamos construir:

- Análisis de código PyTorch arbitrario (AST)
- Resolución de contexto para operadores no cubiertos por TritonBench
- Planificación de fusión multi-operador
- Prompt builder para múltiples operaciones
- Validación del código generado (sintaxis, firma, compilación)
- Loop de reparación automática cuando el LLM genera código inválido
- Sistema de debugging que permita inspeccionar cada etapa del pipeline

---

## 3. El Desafío Central: El Vacío de Contexto

### El problema

Cuando el LLM recibe código PyTorch como:

```python
def linear_relu(x, weight, bias):
    z = x @ weight.T + bias
    return torch.relu(z)
```

...ve nombres de función como `matmul`, `add` y `relu`. Pero necesita saber **mucho más** para generar Triton correcto:

| Lo que el LLM necesita saber | ¿Está en el código? |
|------------------------------|---------------------|
| Qué hace matemáticamente cada operador | ❌ |
| Fórmula exacta (LaTeX o matemática) | ❌ |
| Shapes de entrada, intermedios y salida | ❌ |
| Semántica de broadcasting | ❌ |
| Type promotion y edge cases (NaN, inf, cero) | ❌ |
| Estabilidad numérica | ❌ |
| Parámetros exactos con defaults | Parcialmente (en la llamada) |

Los LLMs **sí saben** qué hacen operadores comunes como `matmul` y `relu` (están en sus datos de entrenamiento). Pero:
- **Operadores esotéricos** (`airy_ai`, `bessel_j1`, `digamma`) tienen casi nulo contexto en los datos de entrenamiento del LLM.
- El LLM alucina APIs de Triton (`triton.lang`, `tl.mm()`, `tl.Scalar`) porque no conoce bien la API real.
- Sin shapes, el LLM no puede diseñar el grid/block size, y el kernel no compila.

### La solución en capas (THINKING.md)

El plan original del equipo (THINKING.md) propone 4 fuentes de contexto:

1. **TritonBench JSON** — ~130 operadores con descripciones de altísima calidad (fórmulas LaTeX, broadcasting detallado, edge cases). Pero solo cubre el 6% de los operadores de PyTorch.

2. **`__doc__` de PyTorch** — los docstrings de los operadores (`torch.add.__doc__`). Cubren casi el 100% de operadores built-in, pero la calidad es muy variable: `torch.add` tiene fórmula LaTeX y broadcasting; `torch.special.airy_ai` tiene 1 línea ("Airy function").

3. **`inspect.signature`** — da la firma exacta de cada operador (nombres de parámetros, defaults, keyword-only). Cubre el 100%, pero solo da información sintáctica, no semántica.

4. **AST** — parsea el código del usuario para detectar qué operadores se usan y en qué orden.

### Conclusión del análisis

Ninguna fuente es suficiente sola. La **combinación en capas con fallback** es la estrategia correcta:

```
Para cada operador detectado:
  Tier 1: ¿Está en TritonBench JSON?  →  contexto de máxima calidad
  Tier 2: ¿Tiene buen __doc__?         →  descripción + fórmula + parámetros
  Tier 3: inspect.signature            →  firma exacta garantizada
  Tier 4: Solo el nombre               →  el LLM usará su conocimiento previo
```

---

## 4. Visión General de la Solución

### Flujo de alto nivel

```
┌──────────────────────────────────────────────────────────────────┐
│                        USUARIO                                    │
│  Escribe función PyTorch + anotaciones de shapes (@triton)        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. PARSE         AST → función, firma, lista de ops en orden    │
│  2. RESOLVE SHAPES Cruzar anotaciones del usuario con el grafo   │
│  3. RESOLVE CONTEXTO TritonBench → __doc__ → signature → nombre   │
│  4. PLAN FUSIÓN    Heurísticas automáticas                       │
│  5. BUILD PROMPT   Construir mensaje rico para el LLM            │
│  6. GENERATE       LLM → respuesta → extraer código Triton       │
│  7. VALIDATE       Compilar, verificar firma, verificar imports  │
│  8. REPAIR         Si falla, reintentar con errores (máx 3)       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        SALIDA                                     │
│  Archivo .py con kernels Triton auto-contenidos                   │
│  Carpeta debug/ con trazabilidad completa de cada etapa          │
└──────────────────────────────────────────────────────────────────┘
```

### Alcance del MVP (Minimum Viable Product)

**Incluye:** Funciones puras de PyTorch con shapes anotadas, fusión automática, niveles 1 a 4 (ver Sección 11).

**No incluye (por ahora):** Clases `nn.Module` con estado interno (pesos, buffers), control de flujo dinámico (if/else/loops), tracing automático de shapes.

### Principios de diseño

1. **Local-first.** Las etapas P0–P5 (parse, shapes, contexto, fusión, prompt) corren sin GPU y sin Modal. Solo la ejecución final necesita Modal.

2. **Modular con repair loops.** Cada etapa es independiente y tiene su propio loop de hasta 3 intentos. Si una etapa se agota, el pipeline se detiene con diagnóstico preciso.

3. **Trazabilidad total.** Cada ejecución guarda artefactos intermedios en `debug/translations/<run_id>/`. Se puede inspeccionar cualquier etapa sin re-ejecutar.

---

## 5. Decisiones de Diseño

### 5.1 Shapes: anotaciones inline en el código fuente

**Formato elegido:**

```python
# @triton
# @in  x:      (N, D_in)
# @in  weight: (D_out, D_in)
# @in  bias:   (D_out,)
# @out (N, D_out)
def linear_relu(x, weight, bias):
    z = x @ weight.T + bias
    return torch.relu(z)
```

**¿Por qué anotaciones inline y no JSON externo o flags de CLI?**

| Alternativa | Problema |
|-------------|----------|
| JSON externo | El usuario tiene que mantener 2 archivos sincronizados. Si renombra un parámetro en el `.py`, el JSON queda desactualizado. |
| Flags de CLI (`--shapes "x=(2,768),w=(512,768)"`) | Inviable para funciones con 8+ parámetros. La línea de comando se vuelve ilegible. |
| Type hints de Python (`x: Tensor["N", "D_in"]`) | Requiere Python 3.12+ con `typing` experimental. No todos los entornos lo soportan. |
| **Anotaciones inline** | ✅ Un solo archivo. Parseo trivial. Sobrevive si faltan. Extensible. |

**Sintaxis completa de anotaciones:**

```python
# @triton                       ←  marca que esta función debe traducirse
# @triton dtype=float32         ←  tipo de datos por defecto (opcional)
# @in  nombre_param: shape      ←  shape de cada parámetro de entrada
# @out shape                    ←  shape del tensor de salida
# @out shape                    ←  múltiples @out para tuplas de retorno
```

**Sintaxis de shapes:**

| Notación | Significado | Ejemplo |
|----------|-------------|---------|
| `(B, C, H, W)` | Dimensiones simbólicas (nombres de variables) | `(N, D_in)` |
| `(2, 64, 224)` | Dimensiones concretas | `(4, 3, 32, 32)` |
| `scalar` | Tensor 0-dimensional (escalar) | `scalar` |
| `(C,)` | Vector 1D | `(D_out,)` |
| `None` | Parámetro opcional sin tensor | `None` |
| `(C,) | None` | Puede ser tensor o None | `(D_out,) | None` |
| `*` | Cualquier número de dimensiones batch | `(*, D_in)` |

### 5.2 Fusión: automática

El planner automático aplica reglas fijas basadas en el tipo de operación y las shapes. Estas reglas son conservadoras (prefieren no fusionar a fusionar mal):

| Regla | Razonamiento |
|-------|-------------|
| **Element-wise consecutivos → fusionar** | No cambian shapes, no requieren sincronización. Ej: `add → relu → mul`. |
| **Matmul/Conv → inician grupo** | Requieren tiling complejo (bloques 2D, memoria compartida). Fusionar el element-wise que viene después (bias, relu) es gratis porque los datos ya están en registros. |
| **Reducción (mean, sum, var) → termina grupo** | Requieren sincronización entre threads. Lo que viene después de una reducción trabaja con un tensor de menor dimensión; es más limpio empezar un kernel nuevo. |
| **Reshape/permute/transpose → nuevo grupo** | Cambian el layout en memoria. Fusionar a través de un reshape puede matar la coalescencia de accesos a memoria. |
| **Dropout → kernel separado** | Requiere generación de números aleatorios y máscara booleana. La semántica de training/eval es compleja. Mejor aislarlo. |
| **Normalización (BatchNorm, LayerNorm, RMSNorm) → kernel propio** | Son reducciones + element-wise, pero tienen parámetros aprendibles y estadísticas acumuladas. Son kernels bien conocidos que es mejor mantener separados. |

### 5.3 Alcance del MVP: funciones puras + shapes anotadas (Niveles 1-4)

Esto significa:
- ✅ Funciones sueltas (`def mi_funcion(a, b, c): ...`)
- ✅ Sin clases `nn.Module`
- ✅ Sin control de flujo (if/else/loops)
- ✅ Shapes provistas por el usuario (anotaciones inline)
- ✅ Fácil de probar, depurar, y demostrar

---

## 6. Formato de Entrada del Usuario

### Ejemplo completo: Linear + ReLU (Nivel 3)

Archivo `linear_relu.py`:

```python
# @triton
# @in  x:      (N, D_in)
# @in  weight: (D_out, D_in)
# @in  bias:   (D_out,)
# @out (N, D_out)
def linear_relu(x, weight, bias):
    z = x @ weight.T + bias
    return torch.relu(z)
```

### Ejemplo completo: Conv + BatchNorm + ReLU (Nivel 4)

Archivo `conv_bn_relu.py`:

```python
# @triton
# @in  x:         (N, C, H, W)
# @in  weight:    (C_out, C, K, K)
# @in  bias:      (C_out,)
# @in  bn_weight: (C_out,)
# @in  bn_bias:   (C_out,)
# @in  bn_mean:   (C_out,)
# @in  bn_var:    (C_out,)
# @out (N, C_out, H_out, W_out)
def conv_bn_relu(x, weight, bias, bn_weight, bn_bias, bn_mean, bn_var):
    x = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)
    x = (x - bn_mean[None, :, None, None]) / torch.sqrt(bn_var[None, :, None, None] + 1e-5)
    x = x * bn_weight[None, :, None, None] + bn_bias[None, :, None, None]
    return torch.relu(x)
```

---

## 7. Arquitectura Detallada

### 7.1 Componente 1: Parser (`code_analysis/parser.py`)

**Entrada:** Código fuente Python (string o archivo `.py`).

**Salida:** `OperationGraph` — una estructura que describe todas las operaciones detectadas.

```python
@dataclass
class OperationGraph:
    function_name: str
    signature: str                       # "linear_relu(x, weight, bias)"
    parameters: list[Parameter]           # [{name: "x", kind: POSITIONAL}, ...]
    operations: list[OpNode]              # en orden de ejecución
    output_var: str                       # nombre de la variable de retorno

@dataclass
class OpNode:
    op_name: str                         # "torch.matmul", "torch.relu"
    torch_path: str                      # ruta fully-qualified si es posible
    input_vars: list[str]                # ["x", "weight.T"]
    output_var: str                      # "z"
    kwargs: dict                         # {"alpha": 1} para torch.add
    lineno: int                          # línea en el archivo original
```

**Proceso:**

1. `ast.parse(codigo_fuente)` → AST de Python.
2. Extraer `FunctionDef`: nombre, parámetros, defaults.
3. Caminar el cuerpo de la función en orden de ejecución.
4. Para cada nodo `Call`, resolver: ¿es una llamada a torch? ¿a torch.nn.functional?
5. Resolver aliases: `import torch.nn.functional as F` → `F.relu` = `torch.nn.functional.relu`.
6. Detectar `return` para identificar la variable de salida.
7. Construir el `OperationGraph`.

**¿Por qué AST y no regex?** Porque queremos manejar código real: llamadas anidadas, atributos encadenados (`x.T`), imports con alias. Un regex no puede distinguir `F.relu(x)` de `mi_funcion.relu(x)`.

**Limitaciones aceptadas:** No analizamos control de flujo (if/else, loops) para el MVP. Si detectamos un `If` o `For`, el parser emite una advertencia.

**Ejemplo de lo que el parser produce para `linear_relu`:**

```json
{
  "function_name": "linear_relu",
  "signature": "linear_relu(x, weight, bias)",
  "parameters": [
    {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
    {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD"},
    {"name": "bias", "kind": "POSITIONAL_OR_KEYWORD"}
  ],
  "operations": [
    {
      "op_name": "torch.matmul",
      "input_vars": ["x", "weight.T"],
      "output_var": "_t1"
    },
    {
      "op_name": "torch.add",
      "input_vars": ["_t1", "bias"],
      "output_var": "z"
    },
    {
      "op_name": "torch.relu",
      "input_vars": ["z"],
      "output_var": "_t2"
    }
  ],
  "output_var": "_t2"
}
```

### 7.2 Componente 2: Shape Resolution (`code_analysis/shape_resolver.py`)

**Entrada:** `OperationGraph` + anotaciones del usuario (extraídas de comentarios `@triton`).

**Salida:** `OperationGraph` con `.shape` poblado en cada `OpNode`.

**Proceso:**

1. Extraer anotaciones de shapes de los comentarios `@in` y `@out`.
2. Validar que cada `@in` corresponde a un parámetro real de la función.
3. Adjuntar shapes a los nodos de entrada.
4. **Propagar shapes hacia adelante** por el grafo usando reglas de inferencia:

| Operación | Regla de inferencia de shape de salida |
|-----------|----------------------------------------|
| `matmul(a, b)` | `a=(..., M, K), b=(..., K, N)` → `(..., M, N)` |
| `add(a, b)` | Broadcasting: el máximo en cada dimensión |
| `relu(a)` | Misma shape que `a` |
| `conv2d(x, w, stride, padding, ...)` | Fórmula: `H_out = (H + 2*pad - dil*(k-1) - 1)//stride + 1` |

**¿Por qué propagación forward y no solo adjuntar?** Porque el usuario solo provee shapes de entrada. Las shapes intermedias (entre matmul y add, entre add y relu) deben inferirse automáticamente para que el LLM tenga el panorama completo.

**¿Qué pasa si las shapes tienen nombres simbólicos como `N` y `D_in`?** La propagación funciona igual con símbolos. `(N, D_in) @ (D_in, D_out)` → `(N, D_out)`. No necesitamos valores concretos.

**Fallback:** Si una shape no se puede inferir (ej: operación compleja sin regla), se marca como `<unknown>` y el LLM trabajará con eso.

### 7.3 Componente 3: Context Resolution (`context/resolver.py`)

**Entrada:** Nombre de operador (ej: `"torch.matmul"`).

**Salida:** `OpContext` con descripción enriquecida.

```python
@dataclass
class OpContext:
    op_name: str
    source: str              # "tritonbench_json" | "torch_docstring" | "inspect_signature" | "name_only"
    confidence: str          # "high" | "medium" | "low"
    functional_description: str
    math_formula: str | None
    signature: str
    parameters: list[ParamDesc]
    shapes_info: str | None
    broadcasting: str | None
    edge_cases: str | None
    notes: str | None
```

**Proceso de resolución en 4 tiers:**

```
Tier 1: TritonBench JSON
  ↓ (no encontrado)
Tier 2: torch.<op>.__doc__
  ↓ (docstring vacío o muy corto)
Tier 3: inspect.signature(torch.<op>)
  ↓ (función no encontrada en torch)
Tier 4: Solo el nombre del operador (el LLM usará su conocimiento previo)
```

**Tier 1 — TritonBench JSON:**

- Busca el operador en `TritonBench_T_comp_alpac_v1.json` (la variante "complete").
- Si existe, extrae: `instruction` (contiene descripción funcional, firma, matemáticas, notas).
- **Confianza: alta.** Es la fuente de mayor calidad.
- Cobertura: ~130 operadores.

**Tier 2 — `__doc__` de PyTorch:**

- `getattr(torch, 'add').__doc__` → docstring.
- Parseo básico del docstring para extraer:
  - Primera línea → descripción funcional.
  - `.. math::` → fórmula LaTeX.
  - `Args:` → lista de parámetros con tipos.
  - `Keyword args:` → parámetros keyword-only.
- **Confianza: media.** Calidad muy variable entre operadores.
- Cobertura: casi 100% de operadores built-in.

**Lógica de decisión Tier 1 vs Tier 2:**

El resolver inteligente combina ambos: si TritonBench tiene el operador, usa esa descripción (más rica). Si no, usa `__doc__`. Si el `__doc__` es muy corto (< 3 líneas), marca confianza como `low` e intenta complementar con Tier 3.

**¿Qué pasa con operadores como `torch.Tensor.add` (métodos de tensor)?**

Los métodos de tensor (`x.add(y)`) no se usan en el código que el usuario escribe si sigue la convención de PyTorch (funcional). Pero si aparecen, el AST los detecta como `Call(Attribute(value=Name('x'), attr='add'))`. En ese caso, intentamos resolver `torch.Tensor.add` → `torch.add` como fallback.

**Tier 3 — `inspect.signature`:**

- `inspect.signature(torch.add)` → objeto `Signature`.
- Extrae: nombres de parámetros, defaults, si son keyword-only.
- **Confianza: media-alta** para la firma, **baja** para semántica.
- Cobertura: 100% de funciones Python de PyTorch.

**¿Qué información da `inspect.signature`?**

```python
import inspect, torch
sig = inspect.signature(torch.add)
print(sig)
# (input, other, *, alpha=1, out=None)
```

Esto nos dice que:
- `input` y `other` son posicionales (obligatorios).
- `alpha` y `out` son keyword-only (tienen default, se pasan por nombre).
- `alpha` tiene default `1`, `out` tiene default `None`.

Esta información es crítica para que el LLM genere un wrapper con la firma correcta.

**Tier 4 — Solo nombre:**

- Si el operador no está en torch, no tiene docstring, y no tiene signature → enviamos solo el nombre al LLM.
- **Confianza: baja.** El LLM usará su conocimiento previo (datos de entrenamiento).
- Ejemplo: el usuario escribió `mi_funcion_custom(x)` que no es de torch. El LLM podría saber qué hace si es una función conocida.

**Caché:** El resolver cachea resultados en memoria. Si el mismo operador aparece 3 veces en el código del usuario, solo se resuelve una vez.

### 7.4 Componente 4: Fusion Planner (`fusion/planner.py`)

**Entrada:** `OperationGraph` con shapes resueltos + contextos.

**Salida:** `FusionPlan` — agrupación de operaciones en kernels.

```python
@dataclass
class FusionPlan:
    groups: list[FusedGroup]
    strategy: str            # "auto"

@dataclass
class FusedGroup:
    group_id: int
    operations: list[OpNode] # en orden
    fused_name: str          # "fused_matmul_add_relu"
    input_shapes: dict
    output_shape: str
    reasoning: str           # por qué se fusionaron (para debugging)
```

**Algoritmo de fusión automática:**

```
groups = []
current_group = []

para cada op en operations:
    si current_group está vacío:
        agregar op a current_group
    sino si es_fusible(current_group[-1], op):
        agregar op a current_group
    sino:
        groups.append(current_group)
        current_group = [op]

groups.append(current_group)
```

**Función `es_fusible(op_anterior, op_actual)`:**

```python
def es_fusible(prev: OpNode, curr: OpNode) -> bool:
    # Regla 1: Element-wise sobre element-wise → SIEMPRE fusionar
    if prev.es_elementwise() and curr.es_elementwise():
        return True

    # Regla 2: Matmul/Conv → el element-wise que sigue se fusiona
    # (porque los datos ya están en registros)
    if prev.es_compute_intensive() and curr.es_elementwise():
        return True

    # Regla 3: Reducción → NUNCA fusionar lo que sigue
    # (cambia dimensionalidad, requiere sincronización)
    if prev.es_reduccion():
        return False

    # Regla 4: Cambio de layout (reshape, permute) → NUEVO grupo
    if curr.cambia_layout():
        return False

    # Regla 5: Dropout → siempre en su propio kernel
    if "dropout" in prev.op_name or "dropout" in curr.op_name:
        return False

    # Por defecto: no fusionar (conservador)
    return False
```

**Clasificación de operadores:**

El planner consulta el `OpContext` de cada operador para determinar su tipo:

| Tipo | Ejemplos | Política de fusión |
|------|----------|-------------------|
| `element_wise` | relu, gelu, add, mul, sigmoid, tanh, exp, log, sqrt | Fusionar con lo que sea |
| `compute_intensive` | matmul, conv2d, conv1d, bmm | Inician grupo; fusionar element-wise siguiente |
| `reduction` | mean, sum, var, std, max, min, argmax, softmax | Terminan grupo; lo siguiente va en kernel nuevo |
| `normalization` | batch_norm, layer_norm, rms_norm, group_norm | Kernel propio (tienen estado interno) |
| `layout` | reshape, permute, transpose, view, flatten | Nuevo grupo |
| `regularization` | dropout, drop_path | Kernel propio |
| `other` | index_select, gather, scatter, index_fill | Kernel propio (accesos no coalescidos) |

**Plan de fusión para `linear_relu`:**

```
Group 1: [matmul, add, relu]
  Razonamiento: matmul es compute_intensive → inicia grupo.
                add es element_wise → se fusiona (datos en registros).
                relu es element_wise → se fusiona (sin overhead).
  Fused name:  fused_linear_relu
  Input:       x(N,D_in), weight(D_out,D_in), bias(D_out,)
  Output:      (N, D_out)
```

**Plan de fusión para `conv_bn_relu`:**

```
Group 1: [conv2d, sub, div, mul, add, relu]
  Razonamiento: conv2d inicia grupo. Las 5 operaciones siguientes son
                element-wise (broadcasting de mean/var/weight/bias).
  Fused name:  fused_conv_bn_relu
  Output:      (N, C_out, H_out, W_out)
```

### 7.5 Componente 5: Prompt Builder (`prompts/builders/torch_to_triton.py`)

**Entrada:** `OperationGraph` + `FusionPlan` + contextos resueltos.

**Salida:** `messages[]` (formato OpenAI chat) listo para enviar al LLM.

**System Prompt (nuevo template: `torch_translation.txt`):**

```
You are an expert in Triton programming. Given a PyTorch function and
its complete mathematical description, generate a self-contained
Python module containing:

1. Required imports (torch, triton, triton.language as tl)
2. One Triton kernel per fused operation group
3. A wrapper function with the EXACT same name and signature as the
   original PyTorch function

CRITICAL RULES:
- Use ONLY valid Triton APIs. The correct import is:
    import triton
    import triton.language as tl
- Do NOT use: triton.lang, tl.mm(), tl.Scalar, tl.Tensor (as arg type),
  tl.zero(), or any other hallucinated API.
- For matmul: implement tiled matrix multiplication with tl.dot().
- For reductions: use tl.reduce() or manual reduction loops.
- Return ONLY valid Python code. No markdown fences, no explanations.
- The wrapper must have the EXACT function name provided.
```

**User Message (generado dinámicamente):**

Para cada grupo fusionado, se incluye un bloque con:

```
──────────────────────────────────────────────────────────────────
FUNCTION: <nombre de la función>
SIGNATURE: <firma exacta extraída del AST>

INPUT SHAPES:
  <param>: <shape> (<descripción opcional>)
  ...

OUTPUT SHAPE: <shape>

FUSION GROUP 1: <fused_name>
┌─────────────────────────────────────────────────────────────────┐
│ Op 1: <nombre> — <descripción corta>                            │
│   Source: <TritonBench | __doc__> (confidence: <high|medium>)   │
│   Math: <fórmula LaTeX o descripción matemática>                │
│   Input: <var> shape=<shape>                                    │
│   Output: <var> shape=<shape>                                   │
│                                                                  │
│ Op 2: <nombre> — <descripción corta>                            │
│   ...                                                            │
│                                                                  │
│ FUSION REASONING: <por qué estas ops están juntas>              │
│ SUGGESTED APPROACH: <estrategia de tiling sugerida>             │
└─────────────────────────────────────────────────────────────────┘

FUSION GROUP 2: <fused_name>
...

IMPLEMENTATION REQUIREMENTS:
- Kernel 1 (fused_<name>): <descripción de lo que debe hacer>
- Kernel 2 (fused_<name>): <descripción>
- Wrapper: <nombre_fn>(<params>) -> Tensor
```

**Ejemplo concreto del prompt para `linear_relu`:**

Ver [Apéndice A](#apéndice-a-ejemplo-de-prompt-completo-para-linear_relu).

### 7.6 Componente 6: LLM Generation

**Reutiliza los providers existentes:** `NvidiaProvider`, `OpenAIProvider`, `GeminiProvider`.

Parámetros por defecto:
- `temperature=0.15` (baja — queremos determinismo)
- `max_completion_tokens=8192`
- `top_p=0.95`
- `seed=42` (reproducibilidad)

**Post-procesamiento:**

1. `extract_code(response)` — remueve markdown fences, reasoning tokens, texto extra.
2. Validación inicial rápida: ¿el código tiene `import triton`? ¿Tiene una función con el nombre esperado?

### 7.7 Componente 7: Validator (`validation/validator.py`)

**Entrada:** Código Triton generado + `OperationGraph` original (para verificar firma).

**Salida:** `ValidationResult { pass: bool, errors: list[str], warnings: list[str] }`

**Checks:**

| Check | Cómo se hace | Es bloqueante |
|-------|-------------|---------------|
| **Python syntax** | `compile(code, '<generated>', 'exec')` | ✅ Sí |
| **Wrapper signature** | Extraer AST del código generado, comparar parámetros con el original | ✅ Sí |
| **Triton imports** | Verificar que `import triton` y `import triton.language as tl` existen, que NO existe `triton.lang` | ⚠️ Warning |
| **Triton API usage** | Regex para detectar APIs alucinadas: `tl.mm(`, `tl.Scalar`, `tl.Tensor(`, `triton.lang` | ✅ Sí |
| **Kernel annotation** | Verificar que los kernels tienen `@triton.jit` | ✅ Sí |
| **Grid definition** | Verificar que el kernel se lanza con `grid=` | ⚠️ Warning |
| **Function naming** | La wrapper tiene el mismo nombre que la función original? | ✅ Sí |

**¿Por qué no ejecutamos el código en esta etapa?**

Porque ejecutar requiere GPU con Triton instalado. Esta validación es puramente estática y corre en CPU. La ejecución real se hace después, vía Modal.

### 7.8 Componente 8: Repair Loop

Cada etapa del pipeline implementa la interfaz `PipelineStage`:

```python
class PipelineStage(ABC):
    name: str

    def run(self, ctx: PipelineContext) -> StageResult:
        for attempt in range(1, 4):  # max 3 intentos
            result = self._try(ctx)
            if result.success:
                log.info(f"[{self.name}] ✓ passed on attempt {attempt}")
                return result
            log.warning(f"[{self.name}] attempt {attempt}/3 failed: {result.error}")
            ctx = self._prepare_retry(ctx, result.error, attempt)
        raise StageExhaustedError(self.name, result.error)
```

**Estrategias de reintento por etapa:**

| Etapa | Intento 1 | Intento 2 | Intento 3 |
|-------|-----------|-----------|-----------|
| **PARSE** | `ast.parse()` normal | Reintentar quitando comentarios no estándar | Emitir error detallado y abortar |
| **SHAPES** | Leer anotaciones `@in`/`@out` | Buscar type hints, docstrings, nombres de variable | Emitir error con listado de shapes faltantes y abortar |
| **CONTEXT** | TritonBench JSON | `__doc__` del operador | Solo enviar nombre del operador al LLM |
| **FUSION** | Heurísticas estándar | Reglas más agresivas (fusionar todo) | No fusionar (1 kernel por operación) |
| **GENERATE** | Prompt normal | Prompt + errores del intento anterior | Prompt + "think step by step" + ejemplos |
| **VALIDATE** | `compile()` | Reparar imports automáticamente | Generar un stub mínimo para pasar |

**Loop de repair en GENERATE:**

Este es el más importante. Si el código generado no pasa validación:

```
Intento 1: Prompt normal
  → Error: "import triton.lang as tl" (API alucinada)

Intento 2: Mismo prompt + feedback:
  "Your previous attempt had errors:
   - Invalid import: 'triton.lang' does not exist.
     Use 'import triton.language as tl' instead.
   Please fix and regenerate."
  → Error: "NameError: name 'tl.float32' is not defined"

Intento 3: Prompt + feedback + "think step by step":
  "Think carefully about each line before writing it.
   Previous errors: ...
   Valid Triton imports: import triton; import triton.language as tl
   Valid Triton APIs: tl.load(), tl.store(), tl.arange(), tl.dot(),
     tl.reduce(), tl.maximum(), tl.sqrt(), tl.exp(), tl.zeros_like()"
  → Éxito o fallo final
```

---

## 8. Diseño del Prompt

### 8.1 System Prompt (template: `prompts/templates/torch_translation.txt`)

```
You are an expert in Triton programming. Given a PyTorch function and
its complete mathematical description, generate a self-contained
Python module containing:

1. Required imports (torch, triton, triton.language as tl)
2. One Triton kernel per fused operation group
3. A wrapper function with the EXACT same name and signature as the
   original PyTorch function

CRITICAL RULES:
- Use ONLY valid Triton APIs. The correct import is:
    import triton
    import triton.language as tl
- Do NOT use: triton.lang, tl.mm(), tl.Scalar, tl.Tensor (as arg type),
  tl.zero(), or any other hallucinated API.
- For matmul: implement tiled matrix multiplication with tl.dot().
- For element-wise ops: use tl.load + compute + tl.store pattern.
- For reductions: use tl.reduce() or manual reduction loops.
- Design appropriate BLOCK_SIZE constants and grid launch.
- Return ONLY valid Python code. No markdown fences, no explanations.
- The wrapper must have the EXACT function name provided.
```

### 8.2 User Message (generado dinámicamente por `torch_to_triton.py`)

Ver [Apéndice A](#apéndice-a-ejemplo-de-prompt-completo-para-linear_relu) para un ejemplo completo renderizado.

**Campos adaptativos del mensaje:**

| Campo | Se incluye si... |
|-------|-----------------|
| Math formula | El `OpContext` tiene `math_formula` (Tier 1 o Tier 2 con LaTeX) |
| Broadcasting notes | El operador involucra broadcasting |
| Edge cases | TritonBench lo incluye (NaN, inf, división por cero) |
| Suggested approach | El planner de fusión tiene una estrategia recomendada (ej: "tile over N dimension, accumulate in registers") |
| Shape propagation | Las shapes intermedias inferidas por el shape resolver |
| Confidence warning | Si algún operador tiene confianza `low`, se advierte al LLM |

---

## 9. Sistema de Reparación Modular (3 intentos por etapa)

Ver Sección 7.8 para el diseño completo.

**Principio clave:** Cada etapa es independiente. Si CONTEXT falla para el operador 3, no re-ejecutamos PARSE ni SHAPES. Solo re-ejecutamos CONTEXT para ese operador. Esto minimiza llamadas al LLM y tiempo total.

**Logging de intentos:**

```
[3/6] RESOLVING CONTEXT
  ✓ matmul: TritonBench JSON (confidence: high)
  ✓ add:    __doc__ (confidence: medium)
  ⚠ bessel_j1: __doc__ gave 2 lines (confidence: low)
  → Attempt 2: checking TritonBench alias...
  → Attempt 3: falling back to name-only
  ⚠ bessel_j1: resolved (confidence: low, source: name_only)
  ✓ relu:   __doc__ (confidence: medium)
```

---

## 10. Debugging y Trazabilidad

### 10.1 Estructura de artefactos

Cada ejecución produce una carpeta con trazabilidad completa:

```
debug/translations/
└── 2026-06-03_14-22-10_linear_relu/
    ├── 01_input.py                    # Código original del usuario
    ├── 02_parse.json                  # OperationGraph serializado
    ├── 03_shapes.json                 # Shapes resueltos por nodo
    ├── 04_context.json                # Contexto resuelto por operador
    ├── 05_fusion.json                 # FusionPlan serializado
    ├── 06_prompt.md                   # Prompt completo (system + user)
    ├── 07_generation_attempt1.txt     # Respuesta raw del LLM (intento 1)
    ├── 07_generation_attempt2.txt     # Respuesta raw (intento 2, si hubo)
    ├── 07_generation_attempt3.txt     # Respuesta raw (intento 3, si hubo)
    ├── 08_extracted.py                # Código Triton extraído (versión final)
    ├── 09_validation.json             # Resultado de validación + errores
    ├── 10_final.py                    # Código Triton listo para ejecutar
    └── summary.md                     # Resumen legible de todo el proceso
```

### 10.2 Comandos de debug

```bash
# Ejecución completa local (sin Modal, sin LLM real — prueba el pipeline)
python apps/cli/main.py translate --file test.py --dry-run

# Ejecución local con LLM real
python apps/cli/main.py translate --file test.py --local

# Inspeccionar una traducción anterior
python apps/cli/main.py inspect --run 2026-06-03_14-22-10_linear_relu

# Reintentar solo una etapa específica de una run anterior
python apps/cli/main.py translate --file test.py --resume-from fusion

# Modo verbose (muestra cada paso en consola)
python apps/cli/main.py translate --file test.py --local -v
```

### 10.3 Output de consola en modo verbose

```
$ python apps/cli/main.py translate --file test.py --local -v

══════════════════════════════════════════════════════════════════
  PyTorch → Triton Translation Pipeline
  Run ID: 2026-06-03_14-22-10_linear_relu
  File:   test.py
══════════════════════════════════════════════════════════════════

[1/6] PARSING
  ✓ Function detected: linear_relu
  ✓ Signature: linear_relu(x, weight, bias)
  ✓ Operations found: 3
    1. torch.matmul (line 3)
    2. torch.add    (line 3)
    3. torch.relu   (line 4)

[2/6] RESOLVING SHAPES
  ✓ @in  x:      (N, D_in)
  ✓ @in  weight: (D_out, D_in)
  ✓ @in  bias:   (D_out,)
  ✓ @out (N, D_out)
  ✓ Propagation: (N, D_in) @ (D_in, D_out) → (N, D_out)
  ✓ All shapes resolved

[3/6] RESOLVING CONTEXT
  ✓ matmul: TritonBench JSON          (confidence: high,   141 lines)
  ✓ add:    __doc__                   (confidence: medium, 28 lines)
  ✓ relu:   __doc__                   (confidence: medium, 14 lines)

[4/6] PLANNING FUSION
  Strategy: auto
  ✓ Group 1: [matmul, add, relu]
    Reason: matmul(+) → element-wise → element-wise

[5/6] GENERATING (nvidia/deepseek-ai/deepseek-v4-pro)
  Attempt 1... ✓ 2341 tokens, 4.2s
  Code extracted: 87 lines

[6/6] VALIDATING
  ✓ Python syntax: OK
  ✓ Wrapper signature: matches linear_relu(x, weight, bias)
  ✓ Triton imports: valid (triton, triton.language)
  ⚠ Performance: BLOCK_SIZE=32 may be suboptimal (D_in unconstrained)
  OVERALL: PASS (5/6 checks, 1 warning)

══════════════════════════════════════════════════════════════════
  Translation complete.
  Output: debug/translations/2026-06-03_14-22-10_linear_relu/
    → 10_final.py (ready for GPU execution)
══════════════════════════════════════════════════════════════════
```

---

## 11. Criterios de Éxito: Niveles 1 al 6

Cada nivel es un hito demostrable. Si se logra, se puede presentar como evidencia de que el sistema funciona para esa clase de problemas.

### Nivel 1 — Element-wise puro (prueba de concepto)

```python
# @triton
# @in  x: (N,)
# @out (N,)
def gelu_activation(x):
    return 0.5 * x * (1.0 + torch.tanh(
        0.7978845608 * (x + 0.044715 * x**3)
    ))
```

**Qué demuestra:** El pipeline end-to-end funciona. Parse, contexto, prompt, generación, validación.

**Validación:** El kernel Triton produce los mismos valores numéricos que PyTorch (tolerancia `1e-5`).

### Nivel 2 — Reducción + element-wise

```python
# @triton
# @in  x:      (N, D)
# @in  weight: (D,)
# @in  bias:   (D,)
# @out (N, D)
def layernorm(x, weight, bias):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5) * weight + bias
```

**Qué demuestra:** Reducción paralela dentro de un kernel Triton. Grid design (un programa por fila).

### Nivel 3 — Matmul + activación (el clásico)

```python
# @triton
# @in  x:      (N, D_in)
# @in  weight: (D_out, D_in)
# @in  bias:   (D_out,)
# @out (N, D_out)
def linear_relu(x, weight, bias):
    return torch.relu(x @ weight.T + bias)
```

**Qué demuestra:** Tiling 2D, memoria compartida, fusión de bias+relu en registros. **Si esto funciona con speedup medible, el proyecto es defendible.**

### Nivel 4 — Conv + BN + ReLU (valor práctico real)

```python
# @triton
# @in  x:         (N, C, H, W)
# @in  weight:    (C_out, C, K, K)
# @in  bias:      (C_out,)
# @in  bn_weight: (C_out,)
# @in  bn_bias:   (C_out,)
# @in  bn_mean:   (C_out,)
# @in  bn_var:    (C_out,)
# @out (N, C_out, H_out, W_out)
def conv_bn_relu(x, weight, bias, bn_weight, bn_bias, bn_mean, bn_var):
    x = F.conv2d(x, weight, bias, stride=1, padding=1)
    x = (x - bn_mean[None, :, None, None]) / torch.sqrt(bn_var[None, :, None, None] + 1e-5)
    x = x * bn_weight[None, :, None, None] + bn_bias[None, :, None, None]
    return F.relu(x)
```

**Qué demuestra:** Convolución en Triton (kernel más complejo que matmul), broadcasting 4D, fusión que elimina 3 round-trips a memoria. **Si muestra speedup, es un resultado sólido.**

### Nivel 5 — Flash Attention simplificado (aspiracional)

```python
# @triton
# @in  q:     (B, H, S, D)
# @in  k:     (B, H, S, D)
# @in  v:     (B, H, S, D)
# @in  scale: scalar
# @out (B, H, S, D)
def scaled_dot_product_attention(q, k, v, scale):
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return attn @ v
```

**Qué demuestra:** Tiling 2D (online softmax), evita materializar la matriz de atención `S×S`. **Nivel de paper.** Si funciona para `S ≤ 512`, es extraordinario.

### Nivel 6 — Subgrafo transformer MLP (stretch goal)

```python
# @triton
# @in  x:  (B, S, D)
# @in  w1: (D, 4*D)
# @in  b1: (4*D,)
# @in  w2: (4*D, D)
# @in  b2: (D,)
# @out (B, S, D)
def transformer_mlp(x, w1, b1, w2, b2):
    x = F.linear(x, w1, b1)
    x = F.gelu(x)
    x = F.dropout(x, p=0.1, training=True)
    x = F.linear(x, w2, b2)
    x = F.dropout(x, p=0.1, training=True)
    return x
```

**Qué demuestra:** Traducción de un programa completo (no solo un operador). El planificador de fusión automática debe detectar: grupo 1 = `[linear, add, gelu]`, grupo 2 = `[dropout]`, grupo 3 = `[linear, add]`, grupo 4 = `[dropout]`. **Si el sistema, sin intervención humana, parte esto en kernels óptimos y obtiene speedup, es nivel publicación.**

---

## 12. Orden de Implementación (P0–P9)

| Paso | Qué | Depende de | Prueba de que funciona |
|------|-----|------------|------------------------|
| **P0** | `code_analysis/parser.py` + `op_detector.py` | Nada | Parsear `datasets/custom/operator.py`, verificar que detecta `torch.add` |
| **P1** | `code_analysis/shape_resolver.py` | P0 | Parsear test con anotaciones, verificar que shapes se adjuntan al grafo |
| **P2** | `context/knowledge_base.py` + `context/resolver.py` | Nada | `resolver.resolve("torch.add")` → contexto. `resolver.resolve("torch.matmul")` → TritonBench. Resolver 10 operadores variados |
| **P3** | `fusion/planner.py` | P0, P1 | Grafo de linear_relu → propone 1 grupo. Grafo de conv_bn_relu → propone 1 grupo. Grafo con dropout → 2 grupos |
| **P4** | `prompts/builders/torch_to_triton.py` + `prompts/templates/torch_translation.txt` | P2, P3 | Generar el prompt para linear_relu. Hacer review manual. Ajustar template |
| **P5** | `validation/validator.py` | Nada | Pasar código Triton inválido (imports malos), verificar que detecta errores. Pasar código válido, verificar que pasa |
| **P6** | Integrar en CLI (`apps/cli/main.py` subcomando `translate`) | P4, P5 | `python apps/cli/main.py translate --file test.py --local` corre el pipeline completo |
| **P7** | Repair loop (3 intentos por etapa, `PipelineStage` ABC) | P5, P6 | Forzar error en GENERATE (prompt malo), verificar que reintenta. Probar con bessel_j1 (operador sin docstring) |
| **P8** | Conectar con Modal (`backends/modal/jobs/translation.py`) | P6 | Ejecutar en GPU real. Correr evaluación de TritonBench (call + exec accuracy) sobre el código generado |
| **P9** | UI futura (streamlit/gradio) — opcional, post-MVP | P6 | Interfaz web para pegar código, ver resultado |

**Tiempo estimado por paso:** P0–P2: 1-2 días cada uno. P3–P5: 2-3 días cada uno. P6–P8: 1-2 días cada uno. **Total: 2-3 semanas para MVP funcional.**

---

## 13. Estructura de Archivos

```
GrammarConstraint-KernelGeneration/
├── apps/
│   └── cli/
│       └── main.py                     # [MODIFICAR] Nuevo subcomando 'translate'
│
├── code_analysis/                       # [NUEVO] Análisis de código PyTorch
│   ├── __init__.py
│   ├── parser.py                        # P0 — AST → OperationGraph
│   ├── op_detector.py                   # P0 — Detección de torch.* calls
│   └── shape_resolver.py               # P1 — Resolución de shapes
│
├── context/                             # [NUEVO] Resolución de contexto
│   ├── __init__.py
│   ├── knowledge_base.py                # P2 — Carga JSON + __doc__
│   └── resolver.py                      # P2 — Tiered resolution
│
├── fusion/                              # [NUEVO] Planificación de fusión
│   ├── __init__.py
│   └── planner.py                       # P3 — Heurísticas automáticas
│
├── prompts/
│   ├── builders/
│   │   ├── triton_prompt_builder.py     # [EXISTENTE] — Se reutiliza
│   │   └── torch_to_triton.py           # [NUEVO] P4 — Builder multi-op
│   └── templates/
│       ├── triton_translation.txt       # [EXISTENTE] — Template actual
│       └── torch_translation.txt        # [NUEVO] P4 — Template multi-op
│
├── validation/                          # [NUEVO] Validación de código
│   ├── __init__.py
│   └── validator.py                     # P5 — Syntax, signature, API checks
│
├── models/                              # [EXISTENTE] — Se reutiliza
│   └── providers/
│
├── backends/
│   └── modal/
│       └── jobs/
│           └── translation.py           # [NUEVO] P8 — Job de traducción en Modal
│
├── datasets/                            # [EXISTENTE] — Se reutiliza
│   ├── custom/operator.py
│   └── tritonbench/loader.py
│
├── debug/
│   └── translations/                    # [NUEVO] — Artefactos de debugging
│       └── <run_id>/
│           ├── 01_input.py
│           ├── 02_parse.json
│           ├── ...
│           └── summary.md
│
├── PLAN.md                              # [NUEVO] — Este documento
├── README.md
├── THINKING.md
└── requirements.txt
```

---

## 14. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| **LLM alucina APIs de Triton** (`triton.lang`, `tl.mm()`) | Alta | Alto | Validator detecta estas APIs. Repair loop corrige (3 intentos). System prompt con lista explícita de APIs válidas. |
| **LLM no sabe generar kernel de reducción** (mean, var, sum) | Media | Alto | Incluir ejemplos de reducción en el prompt. Usar Tier 1 (TritonBench) que tiene kernels de referencia. |
| **`__doc__` vacío para operadores esotéricos** | Media | Medio | Tier 4 fallback: el LLM usa su conocimiento previo. Para operadores muy esotéricos, el usuario tendrá que proporcionar más contexto. |
| **Shapes mal inferidas** (broadcasting complejo) | Media | Alto | El usuario es responsable de proveer shapes correctas. El shape resolver valida consistencia y emite warnings. Si no se pueden resolver, el pipeline aborta con diagnóstico. |
| **Fusión automática subóptima** (demasiado agresiva o conservadora) | Alta | Bajo | Las reglas son conservadoras por diseño (prefieren no fusionar). Los repair loops tienen fallbacks progresivos (más agresivo → sin fusión). |
| **Código generado no compila en GPU** (error de Triton en runtime) | Media | Alto | P6 (Modal) ejecuta compilación real. Si falla, el repair loop reintenta con el mensaje de error de Triton. |
| **Speedup negativo** (el kernel Triton es más lento que PyTorch) | Alta | Medio | Aceptable para MVP. El foco es corrección funcional, no rendimiento. El tuning de BLOCK_SIZE y grid se puede hacer después. |
| **Cambios en la API de PyTorch** (nuevos operadores, deprecated) | Baja | Bajo | `__doc__` y `inspect.signature` siempre reflejan la versión instalada. El JSON de TritonBench puede actualizarse periódicamente. |

---

## 15. Apéndice: Ejemplos de Referencia

### Apéndice A: Ejemplo de Prompt Completo para `linear_relu`

```
──────────────────────────────────────────────────────────────────
FUNCTION NAME: linear_relu
ORIGINAL SIGNATURE: linear_relu(x, weight, bias) -> Tensor

INPUT SHAPES:
  x:      (N, D_in)     — input features, batch of N vectors
  weight: (D_out, D_in)  — linear transformation weight matrix
  bias:   (D_out,)       — bias vector, broadcasts over N

OUTPUT SHAPE: (N, D_out)

──────────────────────────────────────────────────────────────────
FUSION GROUP 1: fused_matmul_add_relu
──────────────────────────────────────────────────────────────────

  Op 1: torch.matmul(x, weight.T) — matrix multiplication
    Source: TritonBench JSON (confidence: high)
    Math: out[i,k] = Σⱼ x[i,j] · weight[k,j]
    Shapes: (N, D_in) @ (D_in, D_out) → (N, D_out)
    Notes: Supports broadcasting on batch dimensions.

  Op 2: torch.add(_, bias) — element-wise add with broadcast
    Source: __doc__ (confidence: medium)
    Math: out[i,k] = z[i,k] + bias[k]
    Shapes: (N, D_out) + (D_out,) → (N, D_out)
    Broadcasting: bias (D_out,) → (1, D_out) → (N, D_out)
    Notes: Supports type promotion.

  Op 3: torch.relu(_) — rectified linear unit
    Source: __doc__ (confidence: medium)
    Math: out[i,k] = max(0, y[i,k])
    Element-wise, no shape change.

  FUSION REASONING:
    matmul is compute-intensive with tiling → starts the group.
    add and relu are element-wise → fused at no cost (data stays in registers).

  SUGGESTED APPROACH:
    - Tile over N dimension with BLOCK_N rows per program.
    - Each program loads a BLOCK_N × BLOCK_K tile of x and
      BLOCK_K × BLOCK_DOUT tile of weight.T into shared memory.
    - Accumulate partial dot products in registers (BLOCK_N × D_out).
    - Apply bias + relu directly in registers.
    - Write final BLOCK_N × D_out block to output.

  KERNEL NAME: fused_linear_relu_kernel
  WRAPPER NAME: linear_relu
──────────────────────────────────────────────────────────────────

IMPLEMENTATION REQUIREMENTS:
  1. One Triton kernel: fused_linear_relu_kernel
  2. One wrapper function: linear_relu(x, weight, bias) -> Tensor
  3. The wrapper must call the kernel with appropriate grid.

Generate the complete, self-contained Python module now.
```

### Apéndice B: ¿Qué contiene `torch.add.__doc__`?

```
add(input, other, *, alpha=1, out=None) -> Tensor

Adds other, scaled by alpha, to input.

.. math::
    \text{out}_i = \text{input}_i + \text{alpha} \times \text{other}_i

Supports broadcasting to a common shape,
type promotion, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to add to input.

Keyword args:
    alpha (Number): the multiplier for other.
    out (Tensor, optional): the output tensor.
```

28 líneas. Contiene: firma, descripción, fórmula LaTeX, broadcasting, type promotion, argumentos, keyword args. **Excelente calidad para Tier 2.**

### Apéndice C: ¿Qué contiene `torch.special.airy_ai.__doc__`?

```
airy_ai(input, *, out=None) -> Tensor

Airy function :math:`\text{Ai}\left(\text{input}\right)`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.
```

5 líneas. Contiene: firma, una línea de descripción, args mínimos. **Calidad pobre — el LLM necesitará conocimiento previo.**

### Apéndice D: Tabla de operadores y calidad de docstring

| Operador | Líneas de docstring | ¿Tiene LaTeX? | ¿Tiene shapes? | ¿Tiene edge cases? | Confianza Tier 2 |
|----------|---------------------|---------------|----------------|---------------------|------------------|
| `torch.add` | 28 | ✅ | ❌ (broadcasting sí) | ❌ | Alta |
| `torch.matmul` | 80+ | ❌ | ✅ (múltiples casos) | ✅ | Alta |
| `torch.relu` | 14 | ✅ | ✅ | ❌ | Media |
| `torch.softmax` | 25 | ✅ | ✅ | ❌ | Media |
| `F.conv2d` | 50+ | ❌ | ✅ (detallado) | ✅ | Alta |
| `F.gelu` | 20 | ✅ | ❌ | ❌ | Media |
| `F.batch_norm` | 40+ | ✅ | ✅ | ❌ | Alta |
| `torch.special.airy_ai` | 5 | ❌ | ❌ | ❌ | **Baja** |
| `torch.bessel_j1` | 5 | ❌ | ❌ | ❌ | **Baja** |
| `torch.zeta` | 5 | ✅ | ❌ | ❌ | Baja |
