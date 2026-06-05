# Guía de Estudio — Quiz + SoW
## TC3002B · Grammar-Constrained Kernel Generation

---

## PARTE 1 — QUIZ: 10 Temas Clave

---

### TEMA 1 — Constrained Decoding: qué es y para qué sirve en el reto

**Concepto central:**
Los LLMs generan texto token por token, eligiendo en cada paso el token más probable. Sin restricciones, pueden generar texto inválido (markdown fences, código con syntax errors, loops infinitos).

**Constrained decoding** = en cada paso de decodificación, se aplica una **máscara de bits** (bitmask) sobre el vocabulario del modelo que pone `-inf` en los logits de tokens inválidos **antes** del softmax. El modelo solo puede elegir tokens que sean válidos según la gramática en ese estado.

```
Paso N:  modelo produce logits[V] para todo el vocab
         XGrammar produce bitmask[V/32] de int32
         apply_token_bitmask_inplace: logits inválidos → -inf
         softmax → solo tokens válidos tienen prob > 0
         modelo elige siguiente token
         accept_token → PDA avanza al siguiente estado
```

**Por qué se usa en el reto:**
El LLM genera código Triton. Sin constraint, puede producir:
- Markdown fences (` ```python `) que rompen el `exec()`
- `tl.store(ptr, val) = tl.load(...)` — SyntaxError Python
- Loops infinitos de `if_out_provided = out is not None`
- Módulos incompletos (kernel sin `tl.store`)

El constraint fuerza estructura `preamble → @triton.jit kernel → wrapper`, Python sintácticamente válido.

**Pregunta tipo quiz:**
> ¿En qué paso del pipeline de generación actúa el constrained decoding?
> a) Antes de enviar el prompt al LLM
> b) **En cada paso de decodificación, modificando los logits antes del softmax** ✓
> c) Como post-procesamiento después de generar todo el texto
> d) En el tokenizer, antes de codificar el prompt

---

### TEMA 2 — XGrammar: rol en el pipeline

**Las 5 clases y cuándo corren:**

| Clase | Qué hace | Frecuencia |
|---|---|---|
| `Grammar` | Define el EBNF (la gramática) | 1 vez |
| `TokenizerInfo` | Vocabulario + tokens especiales del modelo | 1 vez por tokenizer |
| `GrammarCompiler` | Construye el PDA + cache adaptativa de máscaras | 1 vez por (gramática, tokenizer) |
| `CompiledGrammar` | Handle cacheable y serializable al resultado compilado | Compartido entre requests |
| `GrammarMatcher` | Runtime: `fill_next_token_bitmask` + `accept_token` | **En cada token decodificado** |

**Por qué es rápido (insight clave):**
XGrammar pre-computa las máscaras para ~99% de los tokens del vocabulario que son **context-independent** (su validez no depende del estado del PDA). Solo los ~120 tokens **context-dependent** se recalculan en runtime. Esto hace que `fill_next_token_bitmask` sea O(1) para la mayoría de pasos.

**Integración en el pipeline del reto:**
```
nvidia_provider.py  →  extra_body={"guided_grammar": ebnf_str}
                                    ↓
                        NVIDIA NIM (self-hosted)
                        vLLM backend con XGrammar
                        aplica máscara en cada token
```

**Pregunta tipo quiz:**
> ¿Cuál componente de XGrammar se ejecuta en CADA paso de decodificación?
> a) GrammarCompiler
> b) TokenizerInfo
> c) **GrammarMatcher** ✓
> d) CompiledGrammar

---

### TEMA 3 — Triton vs CUDA: diferencias y arquitectura GPU

**Triton:**
- DSL en Python sobre CUDA
- Programa a nivel de **tile/bloque** (no thread individual)
- `@triton.jit` compila a PTX (mismo binario que CUDA)
- Backend de `torch.compile`, Flash Attention, vLLM
- Más productivo: el compilador de Triton optimiza automáticamente accesos a memoria

**CUDA:**
- C/C++, acceso directo a cada hilo (threadIdx, blockIdx)
- Control total pero complejidad alta
- Requiere gestión manual de memoria compartida, warps, etc.

**Arquitectura GPU relevante:**
```
GPU
├── SM (Streaming Multiprocessor) × N
│   ├── Warp (32 threads SIMT)
│   ├── Shared memory / L1 cache (SRAM, ~100KB, rápida)
│   └── Registros (más rápidos que shared memory)
└── Global memory (HBM/DRAM, GB, lenta ~1TB/s)
```

**Por qué Triton importa en el reto:**
El objetivo es generar kernels Triton que reemplacen operaciones PyTorch. La ganancia de rendimiento viene de **fusión de operadores** — en vez de 3 viajes a HBM (matmul → add → relu), un solo kernel Triton hace todo en SRAM.

**Modelo de ejecución Triton:**
```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)          # ← qué bloque soy
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # ← tile
    mask = offsets < N
    x = tl.load(ptr + offsets, mask=mask)  # ← carga tile a SRAM
    # ... operaciones sobre el tile ...
    tl.store(out_ptr + offsets, result, mask=mask)  # ← guarda
```

**Pregunta tipo quiz:**
> ¿A qué nivel de abstracción opera un kernel Triton comparado con CUDA?
> a) Thread individual (threadIdx)
> b) Warp (32 threads)
> c) **Tile/bloque (group of threads)** ✓
> d) GPU completa

---

### TEMA 4 — TritonBench y métricas de evaluación

**TritonBench:** Benchmark de ~130 operadores PyTorch con descripción de alta calidad (fórmula matemática, shapes, edge cases, type promotion). Fuente de ground truth para evaluar LLMs que generan kernels.

**Las 3 métricas en orden:**

```
Fase 1: call_acc (call accuracy)
    ¿La función existe y tiene la firma correcta?
    exec(código_generado) → ¿existe la función? → ¿se puede llamar?
    Survivors: N / total

Fase 2: exec_acc (execution accuracy)  
    ¿El resultado numérico es correcto?
    output_triton ≈ output_pytorch (torch.allclose)
    Solo corre si pasó call_acc

Fase 3: efficiency / speedup
    ¿Qué tan rápido vs PyTorch baseline?
    speedup = tiempo_pytorch / tiempo_triton
    Solo corre si pasó exec_acc
```

**Por qué `could not get source code`:**
TritonBench usa `inspect.getsource()` sobre funciones definidas via `exec()`. Esto siempre falla porque `exec()` no crea un módulo en disco. Es un bug/limitación de TritonBench, no del código generado.

**Pregunta tipo quiz:**
> Un kernel generado pasa call_acc pero falla exec_acc. ¿Qué significa?
> a) El código tiene un SyntaxError
> b) La función no existe en el módulo generado
> c) **La función existe y se puede llamar, pero produce resultados numéricos incorrectos** ✓
> d) El kernel es más lento que PyTorch

---

### TEMA 5 — CFG, EBNF y Análisis Sintáctico (módulo Compiladores)

**Gramática Libre de Contexto G = (V, T, P, S):**
- **V**: no-terminales (construcciones: `program`, `if_stmt`, `expr`)
- **T**: terminales (tokens: `KW_IF`, `IDENTIFIER`, `COLON`)
- **P**: producciones (`if_stmt → KW_IF expr COLON suite`)
- **S**: símbolo inicial (`program`)

**LALR(1) — cómo parsea Bison:**
- Lee tokens de izquierda a derecha
- Mantiene una pila y una tabla de acciones
- **Shift**: empuja el token a la pila
- **Reduce**: cuando el tope coincide con el RHS de una producción, reemplaza por el LHS
- El "1" = 1 token de lookahead para decidir

**Diferencia YACC (triton.y) vs XGrammar (triton_kernel.ebnf):**

| Aspecto | triton.y (YACC) | triton_kernel.ebnf (XGrammar) |
|---|---|---|
| Algoritmo | Bottom-up LALR(1) | Top-down LL/PDA |
| Terminales | Códigos de token (`KW_IF`) | Patrones de texto (`"if "`) |
| Indentación | Tokens INDENT/DEDENT del lexer | Espacios literales (`"    "`) |
| Recursión izq. | Permitida | **Debe eliminarse** |
| Propósito | Verificar código existente | Guiar generación token a token |

**Pregunta tipo quiz:**
> ¿Por qué la gramática en XGrammar NO puede tener recursión izquierda?
> a) Porque XGrammar usa LALR(1) como Bison
> b) **Porque XGrammar usa un PDA de estilo LL (top-down) que entraría en loop infinito** ✓
> c) Porque los LLMs no generan tokens recursivos
> d) Por limitaciones del formato EBNF

---

### TEMA 6 — Pipeline completo del reto

```
Usuario da código PyTorch
        ↓
    AST Parser (P0)         ← detecta operadores
        ↓
 Context Resolver (P1)      ← TritonBench JSON > __doc__ > signature
        ↓
 Fusion Planner (P2)        ← decide qué fusionar
        ↓
 Prompt Builder (P3)        ← construye mensaje rico para LLM
        ↓
 LLM Generation (P4)        ← NVIDIA API / vLLM + grammar constraint
        ↓
 Code Validator (P5)        ← sintaxis, firma, imports
        ↓
 Repair Loop (max 3) (P6)   ← si falla, reintenta con el error
        ↓
 TritonBench Eval (P7)      ← call_acc → exec_acc → speedup
        ↓
 Debug artifacts (P8)       ← debug/ con trazabilidad completa
```

**Por qué grammar constraint en P4:**
Sin constraint, ~30% de las generaciones tienen syntax errors o markdown fences que rompen el `exec()`. Con constraint: solo errores semánticos (que requieren el repair loop).

---

### TEMA 7 — Módulo IA: LLMs y token generation

**Cómo genera un LLM:**
1. Input → tokenizer → token IDs
2. Transformer → logits[vocab_size] para el último token
3. Softmax → probabilidades
4. Sampling (greedy / top-p / temperature) → siguiente token
5. Repetir hasta EOS o max_tokens

**Temperature y top-p:**
- `temperature=0`: greedy (siempre el token más probable)
- `temperature>0`: muestrea aleatoriamente (más diversidad)
- `top_p=0.95`: solo considera los tokens cuya probabilidad acumulada suma 95%

**Razonamiento (DeepSeek-R1 style):**
Algunos modelos producen `reasoning_content` (tokens de pensamiento interno) separados del `content` (respuesta). El provider actual filtra el reasoning y solo retorna el content.

**Pregunta tipo quiz:**
> ¿Qué hace `apply_token_bitmask_inplace` en el loop de XGrammar?
> a) Elimina físicamente los tokens inválidos del vocabulario
> b) **Pone -inf en los logits de tokens inválidos antes del softmax** ✓
> c) Filtra los tokens después del softmax
> d) Modifica las probabilidades del modelo permanentemente

---

### TEMA 8 — Módulo Estadística: métricas y evaluación

**Speedup:**
```
speedup = tiempo_pytorch_baseline / tiempo_triton_generado
speedup > 1 → kernel más rápido que PyTorch
speedup < 1 → kernel más lento (regresión)
```

**Por qué exec_acc no garantiza buen speedup:**
Un kernel puede ser numéricamente correcto pero usar un patrón ineficiente (e.g., accesos no coalescentes a memoria global, sin aprovechar memoria compartida).

**Varianza en benchmarks de GPU:**
El tiempo de ejecución de kernels GPU tiene alta varianza. Se usan múltiples corridas con warmup y se reporta la mediana o percentil 50.

**Distribución de resultados esperada:**
En TritonBench con modelos actuales sin constraint:
- call_acc: ~60-80% (muchos modelos generan código que exec pero con nombre incorrecto)
- exec_acc: ~30-50% de los que pasaron call_acc
- speedup > 1: ~10-30% de los que pasaron exec_acc

---

### TEMA 9 — Módulo Research: metodología y ablation

**Ablation study:**
Experimento donde se desactiva un componente para medir su contribución. En este reto:
- Baseline: sin grammar constraint (provider `nvidia`)
- Tratamiento: con grammar constraint (provider `nvidia-grammar`)
- Métrica: call_acc, exec_acc, speedup sobre mismo dataset y operador

**Los resultados observados en el proyecto:**
- Sin grammar (`nvidia`): `add` pasa call_acc y exec_acc ✓
- Con grammar v1: `tl.store() = tl.load()` — SyntaxError ✗
- Con grammar v2: loop infinito `if_out_provided = out is not None` × 300 ✗
- Con grammar v3: kernel truncado, error semántico `float.stride(0)` ✗

**Conclusión research:**
Grammar constraint garantiza ausencia de Python SyntaxErrors pero puede degradar calidad semántica. Trade-off confirmado empíricamente. Próximo paso: ablation con más operadores y modelos más grandes.

---

### TEMA 10 — Arquitectura del proyecto

**Estructura de archivos clave:**
```
GrammarConstraint-KernelGeneration/
├── backends/modal/
│   ├── app.py              ← benchmark_app y production_app (Modal)
│   ├── entrypoints.py      ← CLI: benchmark() y production()
│   ├── image.py            ← Docker images con PyTorch + Triton
│   └── jobs/
│       ├── bench_generation.py  ← genera predicciones en Modal GPU
│       └── bench_evaluation.py  ← evalúa con TritonBench
├── models/providers/
│   ├── nvidia_provider.py       ← NVIDIA API (stream=True normal,
│   │                               stream=False + guided_grammar)
│   └── vllm_grammar_provider.py ← vLLM local con XGrammar
├── grammars/
│   ├── triton_kernel.ebnf       ← la gramática EBNF
│   └── loader.py                ← compila gramática para HF models
├── prompts/templates/
│   └── triton_translation.txt   ← system prompt del LLM
└── PLAN.md                      ← plan arquitectónico del equipo
```

**Cómo correr:**
```bash
# Sin grammar (baseline)
modal run backends/modal/entrypoints.py::benchmark \
  --provider nvidia --model qwen/qwen3.5-397b-a17b --operator add

# Con grammar constraint
modal run backends/modal/entrypoints.py::benchmark \
  --provider nvidia-grammar --model qwen/qwen3.5-397b-a17b --operator add
```

---

## PARTE 2 — STATEMENT OF WORK (SoW)

### Guía de escritura basada en TU contribución real

---

### SECCIÓN 1 — Qué hice

**Lista de tareas/features/entregables (escribe estos en tu SoW):**

1. **Especificación de la CFG del Triton-DSL** (`triton-lex/cfg_triton_dsl.md`)
   - Definí formalmente G = (V, T, P, S) con 35 no-terminales, 51 terminales y ~70 producciones
   - Escribí la especificación informal (en lenguaje natural) y la formal en notación BNF

2. **Lexer en Flex** (`triton-lex/triton.l`)
   - Implementé el analizador léxico: manejo de indentación con pila (INDENT/DEDENT), tablas de símbolos (listas enlazadas para IDs, números, strings), cola de tokens para emitir múltiples tokens desde una sola regla

3. **Parser en Bison/YACC** (`triton-lex/triton.y`)
   - Implementé el parser LALR(1) con la CFG completa, declaraciones de precedencia (%left/%right para 10 niveles), error rules específicas por construcción gramatical con `%define parse.error verbose`

4. **Integración Flex+Bison** (`triton-lex/triton.h`, build.sh, test.sh)
   - Diseñé la arquitectura de integración: `YY_DECL` para renombrar la función de Flex, wrapper `yylex()` con cola de tokens, `triton.h` como header compartido

5. **Gramática XGrammar** (`grammars/triton_kernel.ebnf`)
   - Traduje la CFG de YACC a formato EBNF de XGrammar para constrained decoding
   - Eliminé recursión izquierda, reemplacé tokens por patrones de texto, modelé indentación con espacios literales

6. **Integración en el pipeline** (`models/providers/nvidia_provider.py`, `model_registry.py`)
   - Modifiqué `NvidiaProvider` para soportar `guided_grammar` via `extra_body`
   - Añadí `VllmGrammarProvider` para vLLM local, actualicé `model_registry`

7. **Debugging iterativo de la gramática** (3 iteraciones observando outputs reales)
   - v1 → fix: añadí `target` para prevenir asignación a function calls
   - v2 → fix: restructuré `module` con estructura explícita para prevenir loops infinitos
   - v3 → pendiente: kernel body truncation (trade-off semántico)

---

### SECCIÓN 2 — Decisiones técnicas que tomé

**Decisión 1: Cola de tokens en el lexer (no YY_USER_ACTION)**

*El problema:* La regla `\n[ \t]*` en Flex necesita emitir NEWLINE + posiblemente múltiples DEDENTs, pero `yylex()` solo puede retornar UN token por llamada.

*La decisión:* Implementé una cola circular estática (`tok_queue[210]`) y un wrapper `yylex()` que drena la cola antes de llamar a `flex_lex()`. Usé `YY_DECL` para renombrar la función generada por Flex a `flex_lex()`, preservando `yylex()` como punto de entrada público para Bison.

*Por qué esta opción y no alternatives:* `YY_USER_ACTION` no permite retornar múltiples tokens. Usar `unput()` para "devolver" tokens al buffer es frágil. La cola es O(1) y permite reset al vaciarse.

*Trade-off reconocido:* Requiere que la cola sea suficientemente grande (210 = MAX_DEPTH + margen), pero dado que MAX_DEPTH=100, es seguro.

---

**Decisión 2: Estructura explícita en la gramática XGrammar**

*El problema:* La primera versión tenía `module ::= top_item+`. Esto causaba un loop infinito: después del wrapper, `top_item+` seguía siendo válido y el modelo repetía `if_out_provided = out is not None` hasta max_tokens.

*La decisión:* Reestructuré a `module ::= preamble kernel_section wrapper_section`. La gramática termina exactamente cuando el wrapper function's suite cierra. El EOS token se vuelve el único token válido en ese estado.

*Por qué funciona:* Con la nueva estructura, después del wrapper, el autómata del PDA está en estado aceptador. XGrammar solo permite EOS. El modelo se detiene naturalmente.

*Trade-off reconocido:* La estructura fija (imports → kernels → wrapper) no soporta módulos con helper functions al nivel superior. Se podría relajar `preamble` para incluir `func_def` si necesario.

---

**Decisión 3: `target` vs `expr` en el lado izquierdo de asignaciones**

*El problema:* La generación v1 produjo `tl.store(out_ptr, result) = tl.load(...)` — Python SyntaxError: "cannot assign to function call".

*La decisión:* Definí un no-terminal `target ::= name target_suffix*` donde `target_suffix` solo permite `.attr` y `[subscript]`. Esto excluye function calls (`primary_suffix ::= "(arg_list)"`) del lado izquierdo.

*Por qué funciona:* En Python, los assignment targets válidos son: name, attribute access, subscript. Nunca function calls. La restricción refleja la semántica real de Python.

*Trade-off reconocido:* El modelo encontró el workaround `tl.store[out_ptr] = result` (subscript assignment) que es sintácticamente válido Python pero semánticamente incorrecto para Triton. Esto indica que la gramática previene SyntaxErrors pero no todos los errores semánticos.

---

### SECCIÓN 3 — Cómo podría evidenciarlo

- **Lexer y parser:** archivos `triton.l`, `triton.y`, `triton.h`, `build.sh` en rama `main` del repo `triton-lex/`
- **Especificación CFG:** documento `triton-lex/cfg_triton_dsl.md`
- **Gramática XGrammar:** `grammars/triton_kernel.ebnf` y `grammars/loader.py` en repo `GrammarConstraint-KernelGeneration`
- **Integración pipeline:** `models/providers/nvidia_provider.py` (commits con mensaje "feat: grammar-constrained NvidiaProvider"), `models/registry/model_registry.py`
- **Debug outputs de las 3 iteraciones:** archivos en Modal Volume `/data/debug/0000_add.txt` de los runs `ap-vneCOGWOAZDJaubaRksKGL`, `ap-9FkXqi9iB4Wct61bHFpDOA`, `ap-F5qt9gkbzU72jAm8Sn4nSw`, `ap-XwYSWEn3zdoPF7yaHutcEM` (visible en modal.com dashboard)
- **Notebook de integración XGrammar:** `Notebooks/triton_grammar_xgrammar.ipynb`

---

## RESUMEN RÁPIDO — Para repasar 10 min antes

| Concepto | Una línea |
|---|---|
| Constrained decoding | Máscara de bits sobre logits en cada token decodificado |
| XGrammar GrammarMatcher | Runtime que llama `fill_next_token_bitmask` + `accept_token` por token |
| Triton vs CUDA | Triton opera a nivel tile/bloque; CUDA a nivel thread individual |
| tl.load / tl.store | Cargan/guardan tiles de memoria global a SRAM (operaciones vectorizadas con máscara) |
| call_acc | ¿La función existe y tiene la firma correcta? |
| exec_acc | ¿El resultado numérico es correcto vs PyTorch? |
| LALR(1) | LR parser con 1 token lookahead; shift/reduce sobre una pila |
| guided_grammar | Parámetro extra_body que pasa EBNF al servidor NIM/vLLM |
| Por qué no recursión izq. en XGrammar | PDA LL (top-down) entra en loop infinito |
| Trade-off del constraint | Garantiza sintaxis válida; puede degradar calidad semántica |
