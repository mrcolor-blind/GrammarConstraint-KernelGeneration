import * as vscode from 'vscode';
import { TritonPanel } from './panels/TritonPanel';
import { HistoryViewProvider } from './providers/historyViewProvider';

export function activate(context: vscode.ExtensionContext) {
  // Register command to open main webview
  const disposable = vscode.commands.registerCommand('triton.translate', async () => {
    TritonPanel.createOrShow(context.extensionUri);
  });
  context.subscriptions.push(disposable);

  // Register history view provider
  const historyProvider = new HistoryViewProvider(context.extensionUri);
  const treeView = vscode.window.createTreeView('triton.history', {
    treeDataProvider: historyProvider,
  });
  context.subscriptions.push(treeView);

  // Register refresh command
  const refreshDisposable = vscode.commands.registerCommand('triton.refreshHistory', () => {
    historyProvider.refresh();
  });
  context.subscriptions.push(refreshDisposable);

  // Register command to load a run from history
  const loadRunDisposable = vscode.commands.registerCommand('triton.loadRun', async (jobId: string) => {
    TritonPanel.createOrShow(context.extensionUri);
    // Wait a bit for the panel to initialize, then load the run
    setTimeout(() => {
      if (TritonPanel.currentPanel) {
        TritonPanel.currentPanel.loadRun(jobId);
      }
    }, 500);
  });
  context.subscriptions.push(loadRunDisposable);
}

export function deactivate() {}
