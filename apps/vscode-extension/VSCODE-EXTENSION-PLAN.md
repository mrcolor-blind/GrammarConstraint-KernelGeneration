# Plan de Extension VS Code: Triton Translator

## 1. Ubicacion

```
apps/vscode-extension/
├── package.json
├── tsconfig.json
├── .vscodeignore
├── src/
│   ├── extension.ts
│   ├── api/
│   │   └── client.ts
│   ├── panels/
│   │   └── TritonPanel.ts
│   └── utils/
│       └── parser.ts
└── media/
    ├── main.js
    └── style.css
```

## 2. Objetivo

Extensión de VS Code que permite a un usuario traducir una funcion PyTorch a un kernel Triton utilizando el servicio REST dockerizado expuesto en `http://localhost:8000/api/v1`.

El usuario coloca el cursor sobre una funcion, la extension detecta automaticamente el bloque completo (incluyendo comentarios decoradores), extrae las dimensiones de los comentarios `@in/@out`, permite al usuario introducir los valores numericos de esas dimensiones, y luego ejecuta los 3 pasos del pipeline en botones independientes, mostrando todos los resultados en una pestana webview.

## 3. Requisitos del usuario

| # | Requisito | Decision |
|---|-----------|----------|
| 1 | Extraccion de codigo | Automatica desde el archivo abierto. El cursor debe estar dentro de la funcion. Subir hasta `def`, capturar todo el bloque indentado. Manejar decoradores complejos (subir hasta la linea sin indentacion previa a `def`). |
| 2 | Flujo de ejecucion | Botones independientes: `Traducir` -> `Validar GPU` -> `Evaluar`. El usuario activa manualmente cada paso. |
| 3 | Dimensiones (`dims`) | Extraidas de los comentarios `# @in` y `# @out`. Mostradas en un formulario editable para que el usuario introduzca los valores numericos. |
| 4 | URL del servicio | Fija: `http://localhost:8000/api/v1` (sin configuracion adicional). |
| 5 | Ubicacion | `apps/vscode-extension/` (al lado de `apps/cli/`). |
| 6 | Sin comentarios `@triton` | Bloquear con warning y exigir que los anada. |
| 7 | Empaquetado | Preparar todo para generar `.vsix` (instalable). |

## 4. Arquitectura

### 4.1 Flujo de usuario

```
1. Usuario abre archivo .py y coloca cursor dentro de una funcion con @triton
2. Ejecuta comando "Triton: Traducir funcion seleccionada"
3. Extension detecta la funcion y sus comentarios
   |-- Si no hay @triton / @in -> mostrar warning y abortar
   +-- Si hay comentarios -> extraer nombres de dims (N, D_in, etc.)
4. Abre Webview Panel lateral con:
   |-- Codigo PyTorch (read-only)
   |-- Formulario de dimensiones (inputs editables)
   +-- Boton "Traducir"
5. Usuario rellena dims y pulsa "Traducir"
   |-- POST /api/v1/translate (timeout 120s)
   |-- Mostrar progreso (spinner)
   +-- Mostrar resultados: codigo Triton + validacion estatica
6. Boton "Validar GPU" se habilita
   |-- POST /api/v1/jobs/{job_id}/gpu-validate (timeout 360s)
   +-- Mostrar: compilation_pass, execution_pass, output_shape, device
7. Boton "Evaluar" se habilita
   |-- POST /api/v1/evaluate (timeout 120s)
   +-- Mostrar: accuracy_pass, max_error, speedup
8. El usuario puede copiar el codigo Triton o abrirlo en un nuevo archivo
```

### 4.2 Componentes

| Componente | Archivo | Responsabilidad |
|------------|---------|---------------|
| **Entry Point** | `src/extension.ts` | Registra el comando `triton.translate`, activa la extension, crea el panel. |
| **Parser** | `src/utils/parser.ts` | Dado un `TextDocument` y una `Position`, extrae la funcion completa (desde `def` hasta el final del bloque indentado). Parsea comentarios encima para detectar `@triton`, `@in`, `@out`. Extrae los nombres de dimensiones (tokens mayusculas en los shapes). |
| **API Client** | `src/api/client.ts` | Clase `TritonClient` con metodos `translate()`, `gpuValidate()`, `evaluate()`. Usa `node-fetch` con `AbortController`. Timeouts: 120s (translate), 360s (gpu-validate), 120s (evaluate). |
| **Webview Panel** | `src/panels/TritonPanel.ts` | Crea `WebviewPanel` con `ViewColumn.Beside`. Maneja `postMessage` entre el JS de la webview y la extension. Coordina la orquestacion de los 3 pasos. |
| **Webview UI** | `media/main.js` | Logica del frontend: renderizar formulario, manejar botones, mostrar spinners, plegar secciones. |
| **Webview Styles** | `media/style.css` | Estilos adaptados al tema de VS Code (usando variables CSS). |

## 5. Especificacion tecnica

### 5.1 `package.json`

```json
{
  "name": "triton-translator",
  "displayName": "Triton Translator",
  "description": "Traduce funciones PyTorch a kernels Triton usando el servicio dockerizado",
  "version": "0.1.0",
  "publisher": "triton-team",
  "engines": {
    "vscode": "^1.74.0"
  },
  "categories": [
    "Other",
    "Machine Learning",
    "Snippets"
  ],
  "activationEvents": [
    "onCommand:triton.translate"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "triton.translate",
        "title": "Triton: Traducir funcion seleccionada",
        "icon": "$(rocket)"
      }
    ],
    "menus": {
      "editor/context": [
        {
          "command": "triton.translate",
          "group": "9_cutcopypaste@5",
          "when": "editorLangId == python"
        }
      ]
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "package": "vsce package"
  },
  "dependencies": {
    "node-fetch": "^3.3.2"
  },
  "devDependencies": {
    "@types/vscode": "^1.74.0",
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0",
    "vsce": "^2.15.0"
  }
}
```

### 5.2 `tsconfig.json`

```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2020",
    "outDir": "out",
    "lib": ["ES2020"],
    "sourceMap": true,
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "exclude": ["node_modules", ".vscode-test"]
}
```

### 5.3 `.vscodeignore`

```
.vscode
test/
node_modules/
out/
src/
tsconfig.json
.gitignore
```

### 5.4 API Client (`src/api/client.ts`)

```typescript
const BASE_URL = 'http://localhost:8000/api/v1';

interface TranslatePayload {
  source_code: string;
  provider?: string;
  model?: string;
  dims?: Record<string, number>;
  gpu_validate?: boolean;
}

interface TranslateResponse {
  job_id: string;
  status: string;
  generated_code: string | null;
  validation: { passed: boolean; errors: string[]; warnings: string[] };
  gpu_validation: any;
  errors: string[];
}

interface GpuValidateResponse {
  compilation_pass: boolean;
  execution_pass: boolean;
  output_shape?: string;
  device?: string;
  errors: string[];
}

interface EvaluateResponse {
  job_id: string;
  accuracy_pass: boolean;
  max_error: number;
  speedup: number;
  errors: string[];
}

class TritonClient {
  async translate(payload: TranslatePayload): Promise<TranslateResponse>
  async gpuValidate(jobId: string): Promise<GpuValidateResponse>
  async evaluate(jobId: string, dims: Record<string, number>): Promise<EvaluateResponse>
}
```

### 5.5 Parser (`src/utils/parser.ts`)

```typescript
interface ParsedFunction {
  sourceCode: string;
  hasTritonAnnotation: boolean;
  dims: string[];
  errors: string[];
}

function parseFunctionAtCursor(document: vscode.TextDocument, position: vscode.Position): ParsedFunction
```

**Logica de extraccion:**
1. Dado `position`, buscar hacia atras la linea que empieza con `def ` (ignorando indentacion).
2. Subir mas para capturar comentarios consecutivos (lineas que empiezan con `#`).
3. Leer desde `def` hacia adelante hasta que se encuentre una linea no vacia con indentacion menor o igual a la de `def`.
4. Parsear comentarios con regex: `@in\s+\w+:\s*\(([^)]+)\)` y `@out\s*\(([^)]+)\)`.
5. De los grupos capturados, extraer tokens que sean mayusculas o contengan `_` (ej: `N`, `D_in`, `D_out`).
6. Deduplicar y devolver como `dims`.

### 5.6 Webview Panel (`src/panels/TritonPanel.ts`)

```typescript
class TritonPanel {
  static createOrShow(extensionUri: vscode.Uri, parsed: ParsedFunction);
  private _updateWebview(parsed: ParsedFunction);
  private _handleMessage(message: any);
  private async _doTranslate(dims: Record<string, number>);
  private async _doGpuValidate(jobId: string);
  private async _doEvaluate(jobId: string, dims: Record<string, number>);
}
```

**Mensajes Webview <-> Extensión (postMessage):**

| Direccion | Tipo | Payload |
|-----------|------|---------|
| Webview -> Ext | `translate` | `{ dims: Record<string, number> }` |
| Webview -> Ext | `gpuValidate` | `{ jobId: string }` |
| Webview -> Ext | `evaluate` | `{ jobId: string, dims: Record<string, number> }` |
| Webview -> Ext | `copyCode` | `{ code: string }` |
| Webview -> Ext | `openInNewFile` | `{ code: string, language: string }` |
| Ext -> Webview | `setResult` | `{ step: 'translate'|'gpu'|'evaluate', data: any, error?: string }` |
| Ext -> Webview | `setProgress` | `{ step: string, active: boolean }` |

### 5.7 Webview UI (`media/main.js`)

**Estado:**
- `jobId`: string | null
- `translateResult`: object | null
- `gpuResult`: object | null
- `evaluateResult`: object | null

**Renderizado:**
1. **Header:** Titulo + estado del job.
2. **Codigo PyTorch:** `<pre>` con el codigo original.
3. **Dimensiones:** Inputs dinamicos generados a partir de `parsed.dims`.
4. **Botones:**
   - `Traducir` (siempre habilitado si hay dims validos).
   - `Validar GPU` (habilitado si `jobId` existe y `translateResult.status === 'completed'`).
   - `Evaluar` (habilitado si `gpuResult.compilation_pass && gpuResult.execution_pass`).
5. **Resultados (3 secciones plegables):**
   - Triton Code: `<pre>` con syntax highlighting basico + botones "Copiar" y "Abrir".
   - GPU Validation: indicadores booleanos con colores (verde/rojo).
   - Evaluation: numeros formateados.

### 5.8 Extension Entry Point (`src/extension.ts`)

```typescript
export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand(
    "triton.translate",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.document.languageId !== "python") {
        vscode.window.showErrorMessage("Abre un archivo Python primero.");
        return;
      }

      const parsed = parseFunctionAtCursor(editor.document, editor.selection.active);
      
      if (!parsed.hasTritonAnnotation) {
        vscode.window.showWarningMessage(
          "La funcion no tiene comentarios @triton/@in/@out. Anadelos antes de continuar."
        );
        return;
      }

      if (parsed.errors.length > 0) {
        vscode.window.showErrorMessage(parsed.errors.join(" "));
        return;
      }

      TritonPanel.createOrShow(context.extensionUri, parsed);
    }
  );

  context.subscriptions.push(disposable);
}
```

## 6. Flujo de empaquetado (.vsix)

1. `cd apps/vscode-extension`
2. `npm install`
3. `npm run compile`
4. `npx vsce package`
5. Archivo generado: `triton-translator-0.1.0.vsix`

## 7. Consideraciones y riesgos

| Riesgo | Mitigacion |
|--------|------------|
| Timeout de 6 min en GPU validation | `AbortController` con signal de 360s. Spinner claro. Boton cancelable. |
| Funciones con indentacion compleja (nested) | Parser sube hasta `def` y lee hasta la primera linea con indentacion <= `def`. Manejar decoradores consecutivos. |
| Servicio no disponible | Health check implicito en el primer `translate`. Si falla, mostrar error claro: "Asegurate de que docker-compose esta corriendo en localhost:8000". |
| Temas de VS Code | Usar variables CSS estandar de VS Code (`--vscode-editor-background`, `--vscode-editor-foreground`, etc.). |
| Tamano de bundle | `node-fetch` es ligero. `vsce package` excluye `node_modules` innecesarios via `.vscodeignore`. |

## 8. Proximos pasos

1. Crear la estructura de carpetas en `apps/vscode-extension/`.
2. Escribir todos los archivos fuente (`package.json`, `tsconfig.json`, `src/`, `media/`).
3. Compilar y verificar sin errores de TypeScript.
4. Preparar instrucciones de empaquetado.
