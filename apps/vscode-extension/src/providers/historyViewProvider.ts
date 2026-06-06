import * as vscode from 'vscode';
import TritonClient from '../api/client';
import { TritonPanel } from '../panels/TritonPanel';

export class HistoryItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly jobId?: string,
    public readonly status?: string,
    public readonly command?: vscode.Command
  ) {
    super(label, collapsibleState);
    
    if (jobId) {
      this.iconPath = this._getStatusIcon(status);
      this.description = this._getDateLabel();
    }
    
    this.contextValue = jobId ? 'historyItem' : 'actionItem';
  }

  private _getStatusIcon(status?: string): vscode.ThemeIcon {
    if (status === 'completed') {
      return new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'));
    } else if (status === 'failed') {
      return new vscode.ThemeIcon('error', new vscode.ThemeColor('testing.iconFailed'));
    } else {
      return new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('testing.iconQueued'));
    }
  }

  private _getDateLabel(): string {
    // Simple placeholder - could be enhanced with actual date parsing
    return '';
  }
}

export class HistoryViewProvider implements vscode.TreeDataProvider<HistoryItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<HistoryItem | undefined | null | void> = new vscode.EventEmitter<HistoryItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<HistoryItem | undefined | null | void> = this._onDidChangeTreeData.event;
  
  private _items: HistoryItem[] = [];
  private _extensionUri: vscode.Uri;

  constructor(extensionUri: vscode.Uri) {
    this._extensionUri = extensionUri;
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: HistoryItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: HistoryItem): Promise<HistoryItem[]> {
    if (!element) {
      // Root level: show "New Translation" button + history items
      const items: HistoryItem[] = [];
      
      // Add "New Translation" action item
      const newTranslationItem = new HistoryItem(
        'Nueva traducción',
        vscode.TreeItemCollapsibleState.None,
        undefined,
        undefined,
        {
          command: 'triton.translate',
          title: 'Abrir Triton Translator',
        }
      );
      newTranslationItem.iconPath = new vscode.ThemeIcon('add');
      newTranslationItem.tooltip = 'Abrir panel para nueva traducción';
      items.push(newTranslationItem);
      
      // Add separator
      items.push(new HistoryItem(
        'Historial',
        vscode.TreeItemCollapsibleState.None,
        undefined,
        undefined
      ));
      
      // Load history from API
      try {
        const result = await TritonClient.listRuns(20);
        if (result.items && result.items.length > 0) {
          for (const run of result.items) {
            const date = run.created_at ? new Date(run.created_at).toLocaleString() : '';
            const label = run.function_name || `Job ${run.job_id.substring(0, 8)}`;
            const item = new HistoryItem(
              label,
              vscode.TreeItemCollapsibleState.None,
              run.job_id,
              run.status,
              {
                command: 'triton.loadRun',
                title: 'Cargar generación',
                arguments: [run.job_id]
              }
            );
            item.tooltip = `Job: ${run.job_id}\nEstado: ${run.status}\nFecha: ${date}`;
            item.description = `${run.status} • ${date}`;
            items.push(item);
          }
        } else {
          // No history items
          const emptyItem = new HistoryItem(
            'No hay generaciones previas',
            vscode.TreeItemCollapsibleState.None
          );
          emptyItem.iconPath = new vscode.ThemeIcon('info');
          items.push(emptyItem);
        }
      } catch (err) {
        const errorItem = new HistoryItem(
          'Error cargando historial',
          vscode.TreeItemCollapsibleState.None
        );
        errorItem.iconPath = new vscode.ThemeIcon('warning');
        items.push(errorItem);
      }
      
      return items;
    }
    
    return [];
  }
}
