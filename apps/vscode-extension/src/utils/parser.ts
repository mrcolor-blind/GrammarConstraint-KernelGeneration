import * as vscode from 'vscode';

export interface ParsedFunction {
  sourceCode: string;
  hasTritonAnnotation: boolean;
  dims: string[];
  errors: string[];
}

export function parseFunctionAtCursor(
  document: vscode.TextDocument,
  position: vscode.Position
): ParsedFunction {
  const errors: string[] = [];
  const lineCount = document.lineCount;

  // 1. Encontrar la línea que empieza con 'def ' buscando hacia atrás desde el cursor
  let defLine = position.line;
  while (defLine >= 0) {
    const line = document.lineAt(defLine);
    const text = line.text.trimStart();
    if (text.startsWith('def ')) {
      break;
    }
    defLine--;
  }

  if (defLine < 0) {
    errors.push("No se encontró una definición de función (def) bajo el cursor.");
    return { sourceCode: '', hasTritonAnnotation: false, dims: [], errors };
  }

  const defLineObj = document.lineAt(defLine);
  const defIndent = defLineObj.firstNonWhitespaceCharacterIndex;

  // 2. Leer comentarios encima del def
  const comments: string[] = [];
  let commentLine = defLine - 1;
  while (commentLine >= 0) {
    const line = document.lineAt(commentLine);
    const text = line.text.trim();
    if (text.startsWith('#')) {
      comments.unshift(text);
      commentLine--;
    } else if (text.length === 0) {
      commentLine--;
    } else {
      break;
    }
  }

  const hasTritonAnnotation = comments.some(c => c.includes('@triton'));

  // 3. Extraer dimensiones de @in y @out
  const dimsSet = new Set<string>();
  const dimRegex = /@(?:in|out)\s+(?:\w+:\s*)?\(([^)]+)\)/g;
  for (const comment of comments) {
    let match;
    while ((match = dimRegex.exec(comment)) !== null) {
      const shape = match[1];
      // Extraer tokens que son mayúsculas o contienen guion bajo
      const tokens = shape.split(/[,\s]+/).filter(t => /^[A-Z][A-Z0-9_]*$/.test(t));
      for (const token of tokens) {
        dimsSet.add(token);
      }
    }
  }

  // 4. Leer todo el bloque de la función desde def hasta la siguiente línea con indent <= def
  const lines: string[] = [...comments];
  lines.push(defLineObj.text);

  for (let i = defLine + 1; i < lineCount; i++) {
    const line = document.lineAt(i);
    const text = line.text;
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      lines.push(text);
      continue;
    }
    const indent = line.firstNonWhitespaceCharacterIndex;
    if (indent <= defIndent) {
      break;
    }
    lines.push(text);
  }

  const sourceCode = lines.join('\n') + '\n';

  return {
    sourceCode,
    hasTritonAnnotation,
    dims: Array.from(dimsSet),
    errors,
  };
}