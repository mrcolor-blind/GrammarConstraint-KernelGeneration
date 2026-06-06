import * as vscode from 'vscode';
import { TritonPanel } from './panels/TritonPanel';

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand('triton.translate', async () => {
    TritonPanel.createOrShow(context.extensionUri);
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
