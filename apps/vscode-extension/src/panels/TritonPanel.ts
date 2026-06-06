import * as vscode from 'vscode';
import { ParsedFunction } from '../utils/parser';
import TritonClient, { TranslateResponse, GpuValidateResponse, EvaluateResponse } from '../api/client';

export class TritonPanel {
  public static currentPanel: TritonPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposables: vscode.Disposable[] = [];
  private _parsed: ParsedFunction;
  private _jobId: string | null = null;
  private _dims: Record<string, number> = {};

  public static createOrShow(extensionUri: vscode.Uri, parsed: ParsedFunction) {
    const column = vscode.window.activeTextEditor
      ? vscode.ViewColumn.Beside
      : vscode.ViewColumn.One;

    if (TritonPanel.currentPanel) {
      TritonPanel.currentPanel._panel.reveal(column);
      TritonPanel.currentPanel._update(parsed);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'tritonPanel',
      'Triton Translator',
      column,
      {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
      }
    );

    TritonPanel.currentPanel = new TritonPanel(panel, extensionUri, parsed);
  }

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri, parsed: ParsedFunction) {
    this._panel = panel;
    this._extensionUri = extensionUri;
    this._parsed = parsed;

    this._update(parsed);

    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

    this._panel.webview.onDidReceiveMessage(
      async (message) => {
        switch (message.command) {
          case 'translate':
            this._dims = message.dims;
            await this._doTranslate(message.dims);
            break;
          case 'gpuValidate':
            await this._doGpuValidate(message.jobId);
            break;
          case 'evaluate':
            await this._doEvaluate(message.jobId, message.dims);
            break;
          case 'copyCode':
            await vscode.env.clipboard.writeText(message.code);
            vscode.window.showInformationMessage('Código copiado al portapapeles.');
            break;
          case 'openInNewFile':
            const doc = await vscode.workspace.openTextDocument({
              content: message.code,
              language: 'python',
            });
            await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            break;
        }
      },
      null,
      this._disposables
    );
  }

  private _update(parsed: ParsedFunction) {
    this._parsed = parsed;
    this._panel.webview.html = this._getHtmlForWebview();
  }

  private async _doTranslate(dims: Record<string, number>) {
    this._postMessage({ command: 'setProgress', step: 'translate', active: true });
    try {
      const result = await TritonClient.translate({
        source_code: this._parsed.sourceCode,
        provider: 'nvidia-grammar',
        dims,
      });
      this._jobId = result.job_id;
      this._postMessage({ command: 'setResult', step: 'translate', data: result });
    } catch (err: any) {
      this._postMessage({ command: 'setResult', step: 'translate', data: null, error: err.message || 'Error desconocido' });
    } finally {
      this._postMessage({ command: 'setProgress', step: 'translate', active: false });
    }
  }

  private async _doGpuValidate(jobId: string) {
    this._postMessage({ command: 'setProgress', step: 'gpu', active: true });
    try {
      const result = await TritonClient.gpuValidate(jobId);
      this._postMessage({ command: 'setResult', step: 'gpu', data: result });
    } catch (err: any) {
      this._postMessage({ command: 'setResult', step: 'gpu', data: null, error: err.message || 'Error desconocido' });
    } finally {
      this._postMessage({ command: 'setProgress', step: 'gpu', active: false });
    }
  }

  private async _doEvaluate(jobId: string, dims: Record<string, number>) {
    this._postMessage({ command: 'setProgress', step: 'evaluate', active: true });
    try {
      const result = await TritonClient.evaluate(jobId, dims);
      this._postMessage({ command: 'setResult', step: 'evaluate', data: result });
    } catch (err: any) {
      this._postMessage({ command: 'setResult', step: 'evaluate', data: null, error: err.message || 'Error desconocido' });
    } finally {
      this._postMessage({ command: 'setProgress', step: 'evaluate', active: false });
    }
  }

  private _postMessage(message: any) {
    this._panel.webview.postMessage(message);
  }

  private _getHtmlForWebview(): string {
    const scriptPath = vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js');
    const stylePath = vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css');
    const scriptUri = this._panel.webview.asWebviewUri(scriptPath);
    const styleUri = this._panel.webview.asWebviewUri(stylePath);

    const sourceCode = this._escapeHtml(this._parsed.sourceCode);
    const dims = JSON.stringify(this._parsed.dims);

    return `<!DOCTYPE html>
      <html lang="es">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="${styleUri}">
        <title>Triton Translator</title>
      </head>
      <body>
        <div class="container">
          <h1>🚀 Triton Translator</h1>
          <p class="subtitle">Traduce funciones PyTorch a kernels Triton</p>

          <section class="section">
            <h2>Código PyTorch</h2>
            <pre class="code-block"><code>${sourceCode}</code></pre>
          </section>

          <section class="section">
            <h2>Dimensiones</h2>
            <p class="hint">Introduce los valores numéricos para cada dimensión detectada:</p>
            <div id="dims-form"></div>
          </section>

          <section class="section actions">
            <button id="btn-translate" class="btn btn-primary">Traducir</button>
            <button id="btn-gpu" class="btn btn-secondary" disabled>Validar GPU</button>
            <button id="btn-evaluate" class="btn btn-secondary" disabled>Evaluar</button>
          </section>

          <div id="progress" class="progress hidden">
            <div class="spinner"></div>
            <span id="progress-text">Procesando...</span>
          </div>

          <div id="results" class="results"></div>
        </div>

        <script>
          const initialDims = ${dims};
          const initialSourceCode = ${JSON.stringify(sourceCode)};
        </script>
        <script src="${scriptUri}"></script>
      </body>
      </html>`;
  }

  private _escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  public dispose() {
    TritonPanel.currentPanel = undefined;
    this._panel.dispose();
    while (this._disposables.length) {
      const x = this._disposables.pop();
      if (x) {
        x.dispose();
      }
    }
  }
}
