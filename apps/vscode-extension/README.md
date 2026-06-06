# Triton Translator

Extensión de VS Code para traducir funciones PyTorch a kernels Triton usando el servicio dockerizado.

## Requisitos previos

- **Servicio corriendo**: El servicio FastAPI debe estar activo en `http://localhost:8000`:
  ```bash
  cd /path/to/repo
  docker-compose up -d
  ```
- Verificar que responde:
  ```bash
  curl http://localhost:8000/api/v1/health
  ```

## Instalación

1. En VS Code, ve a **Extensiones** → `...` → **Instalar desde VSIX...**
2. Selecciona:
   ```
   apps/vscode-extension/triton-translator-0.1.0.vsix
   ```
3. Recarga la ventana si te lo pide.

## Cómo usar

1. Abre **cualquier archivo** (no necesitas un `.py` específico).
2. Ejecuta el comando:
   - `Cmd + Shift + P` → **"Triton: Traducir función seleccionada"**
   - O clic derecho en el editor → **"Triton: Traducir función seleccionada"**
3. Se abre una pestaña lateral con:
   - Un **textarea** para pegar tu código PyTorch.
   - Un botón **"Analizar dimensiones"** que detecta automáticamente `N`, `D_in`, etc.
   - Inputs para introducir los **valores numéricos** de cada dimensión.
   - Botones **Traducir → Validar GPU → Evaluar**.

### Formato del código esperado

Tu código debe incluir comentarios `@triton`, `@in` y `@out` con shapes:

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

### Flujo de trabajo

| Paso | Botón | Qué hace | Tiempo estimado |
|------|-------|----------|-----------------|
| 1 | **Analizar** | Detecta dimensiones del textarea | Instantáneo |
| 2 | **Traducir** | Envía código al servicio y genera kernel Triton | 30-90s |
| 3 | **Validar GPU** | Compila y ejecuta el kernel en Modal GPU | 2-5 min |
| 4 | **Evaluar** | Compara precisión y velocidad vs PyTorch | 10-30s |

### Resultados

- **Código Triton**: Puedes copiarlo o abrirlo en un nuevo archivo.
- **Validación estática**: Errores/warnings de sintaxis.
- **Validación GPU**: Éxito/fracaso de compilación y ejecución.
- **Evaluación**: Precisión, error máximo, y speedup.

## Desarrollo (empaquetar)

```bash
cd apps/vscode-extension
npm run compile
npx vsce package
```

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `command not found` | Reinstala el `.vsix` y recarga la ventana |
| Timeout en GPU | Normal en el primer uso. Modal necesita cold-start. |
| No detecta dimensiones | Asegúrate de que los comentarios usan el formato `@in x: (N, D)` |
| Servicio no responde | Verifica `docker-compose up` y `localhost:8000` |
