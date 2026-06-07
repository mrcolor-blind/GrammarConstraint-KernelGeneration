import * as vscode from 'vscode';
import TritonClient, { TranslateResponse, GpuValidateResponse, EvaluateResponse, JobListResponse, JobDetail } from '../api/client';

export class TritonPanel {
  public static currentPanel: TritonPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposables: vscode.Disposable[] = [];
  private _jobId: string | null = null;
  private _sourceCode: string = '';
  private _callSiteCode: string = '';
  private _dims: Record<string, number> = {};

  public static createOrShow(extensionUri: vscode.Uri) {
    const column = vscode.window.activeTextEditor
      ? vscode.ViewColumn.Beside
      : vscode.ViewColumn.One;

    if (TritonPanel.currentPanel) {
      TritonPanel.currentPanel._panel.reveal(column);
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

    TritonPanel.currentPanel = new TritonPanel(panel, extensionUri);
  }

  public async loadRun(jobId: string): Promise<void> {
    return this._doLoadRun(jobId);
  }

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._extensionUri = extensionUri;

    this._panel.webview.html = this._getHtmlForWebview();

    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

    // Load history immediately
    this._loadHistory();

    this._panel.webview.onDidReceiveMessage(
      async (message) => {
        switch (message.command) {
          case 'translate':
            this._sourceCode = message.sourceCode;
            this._callSiteCode = message.callSiteCode || '';
            this._dims = message.dims || {};
            await this._doTranslate(message.sourceCode, message.callSiteCode, message.dims, message.useGrammar);
            break;
          case 'gpuValidate':
            await this._doGpuValidate(message.jobId);
            break;
          case 'evaluate':
            await this._doEvaluate(message.jobId, message.dims);
            break;
          case 'loadRun':
            await this._doLoadRun(message.jobId);
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

  private async _loadHistory() {
    try {
      const result = await TritonClient.listRuns(20);
      this._postMessage({ command: 'setHistory', data: result });
    } catch (err: any) {
      this._postMessage({ command: 'setHistory', data: null, error: err.message });
    }
  }

  private async _doLoadRun(jobId: string) {
    this._postMessage({ command: 'setProgress', step: 'load', active: true });
    try {
      const result = await TritonClient.getRun(jobId);
      this._jobId = result.job_id;
      this._postMessage({ command: 'setRunDetail', data: result });
    } catch (err: any) {
      this._postMessage({ command: 'setRunDetail', data: null, error: err.message });
    } finally {
      this._postMessage({ command: 'setProgress', step: 'load', active: false });
    }
  }

  private async _doTranslate(sourceCode: string, callSiteCode: string, dims: Record<string, number>, useGrammar?: boolean) {
    this._postMessage({ command: 'setProgress', step: 'translate', active: true });
    try {
      const provider = useGrammar ? 'nvidia-grammar' : 'nvidia';
      const result = await TritonClient.translate({
        source_code: sourceCode,
        call_site_code: callSiteCode || undefined,
        provider,
        dims,
      });
      this._jobId = result.job_id;
      this._postMessage({ command: 'setResult', step: 'translate', data: result });
      // Refresh history after successful translation
      await this._loadHistory();
    } catch (err: any) {
      this._postMessage({ command: 'setResult', step: 'translate', data: null, error: err.message || 'Error desconocido' });
    } finally {
      this._postMessage({ command: 'setProgress', step: 'translate', active: false });
    }
  }

  private async _doGpuValidate(jobId: string) {
    this._postMessage({ command: 'setProgress', step: 'gpu', active: true });
    try {
      // Step 1: smoke test (compile + execute)
      const gpuResult = await TritonClient.gpuValidate(jobId);
      this._postMessage({ command: 'setResult', step: 'gpu', data: gpuResult });

      // Step 2: accuracy + speedup comparison (always run, even if Triton failed)
      this._postMessage({ command: 'setProgress', step: 'compare', active: true });
      try {
        const compareResult = await TritonClient.compare(jobId);
        this._postMessage({ command: 'setResult', step: 'compare', data: compareResult });
      } catch (cmpErr: any) {
        this._postMessage({ command: 'setResult', step: 'compare', data: null, error: cmpErr.message || 'Error en comparación' });
      } finally {
        this._postMessage({ command: 'setProgress', step: 'compare', active: false });
      }

      await this._loadHistory();
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

    const exampleCode = `import torch

def linear_relu(x, weight, bias):
    z = x @ weight.T + bias
    return torch.relu(z)`;

    const exampleCallSite = `import torch

x = torch.randn(128, 256)
weight = torch.randn(512, 256)
bias = torch.randn(512)
out = linear_relu(x, weight, bias)`;

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

          <section class="section history-section">
            <h2>📜 Historial de Generaciones</h2>
            <div id="history-list">
              <p class="hint">Cargando historial...</p>
            </div>
          </section>

          <section class="section">
            <h2>Definición de la función PyTorch</h2>
            <p class="hint">Pega tu función PyTorch (sin comentarios, puro código):</p>
            <textarea id="code-input" class="code-textarea" rows="8" spellcheck="false">${exampleCode}</textarea>
          </section>

          <section class="section">
            <h2>Código de llamada (call site)</h2>
            <p class="hint">Pega el código que invoca tu función para extraer shapes reales de los tensores:</p>
            <textarea id="call-site-input" class="code-textarea" rows="6" spellcheck="false">${exampleCallSite}</textarea>
          </section>

          <section class="section options-section">
            <div class="option-row">
              <label class="toggle-switch" for="use-grammar">
                <input type="checkbox" id="use-grammar" checked>
                <span class="toggle-slider"></span>
              </label>
              <span class="option-label">Usar gramática restringida</span>
              <span class="option-hint">(generación más estructurada, puede ser más lenta)</span>
            </div>
          </section>

          <section class="section actions">
            <button id="btn-translate" class="btn btn-primary">Traducir</button>
          </section>

          <div id="progress" class="progress hidden">
            <div class="spinner"></div>
            <span id="progress-text">Procesando...</span>
          </div>

          <div id="results" class="results"></div>
        </div>

        <script src="${scriptUri}"></script>
      </body>
      </html>`;
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
