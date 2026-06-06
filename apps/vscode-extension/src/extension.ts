import * as vscode from 'vscode';
import { parseFunctionAtCursor } from './utils/parser';
import { TritonPanel } from './panels/TritonPanel';

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand('triton.translate', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'python') {
      vscode.window.showErrorMessage('Abre un archivo Python primero.');
      return;
    }

    const parsed = parseFunctionAtCursor(editor.document, editor.selection.active);

    if (!parsed.hasTritonAnnotation) {
      vscode.window.showWarningMessage(
        'La función no tiene comentarios @triton/@in/@out. Añádelos antes de continuar.'
      );
      return;
    }

    if (parsed.errors.length > 0) {
      vscode.window.showErrorMessage(parsed.errors.join(' '));
      return;
    }

    if (parsed.dims.length === 0) {
      vscode.window.showWarningMessage(
        'No se detectaron dimensiones en los comentarios @in/@out. Añade comentarios con shapes (ej: # @in x: (N, D_in)).'
      );
    }

    TritonPanel.createOrShow(context.extensionUri, parsed);
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
