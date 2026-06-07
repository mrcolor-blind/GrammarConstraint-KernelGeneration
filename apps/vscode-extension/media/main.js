(function () {
  const vscode = acquireVsCodeApi();
  let jobId = null;
  let translateResult = null;
  let gpuResult = null;
  let compareResult = null;
  let evaluateResult = null;
  let runsHistory = [];
  let currentRunDetail = null;

  const codeInput = document.getElementById('code-input');
  const callSiteInput = document.getElementById('call-site-input');
  const btnTranslate = document.getElementById('btn-translate');
  const progressDiv = document.getElementById('progress');
  const progressText = document.getElementById('progress-text');
  const resultsDiv = document.getElementById('results');
  const historyList = document.getElementById('history-list');

  function renderHistory() {
    if (!historyList) return;
    
    if (runsHistory.length === 0) {
      historyList.innerHTML = '<p class="hint">No hay generaciones previas. Realiza una traducción para guardar en el historial.</p>';
      return;
    }

    let html = '<div class="history-list">';
    runsHistory.forEach(run => {
      const statusClass = run.status === 'completed' ? 'success' : (run.status === 'failed' ? 'error' : 'warning');
      const date = run.created_at ? new Date(run.created_at).toLocaleString() : '';
      html += `
        <div class="history-item" data-job-id="${run.job_id}">
          <div class="history-item-main">
            <span class="history-status badge ${statusClass}">${run.status}</span>
            <span class="history-id">${run.job_id}</span>
            <span class="history-date">${date}</span>
          </div>
          <div class="history-item-meta">
            ${run.function_name ? `<span class="history-func">${run.function_name}</span>` : ''}
            <span class="history-model">${run.provider || 'nvidia-grammar'}</span>
          </div>
        </div>
      `;
    });
    html += '</div>';
    historyList.innerHTML = html;

    // Attach click handlers
    document.querySelectorAll('.history-item').forEach(item => {
      item.addEventListener('click', () => {
        const jobId = item.dataset.jobId;
        if (jobId) {
          vscode.postMessage({ command: 'loadRun', jobId });
        }
      });
    });
  }

  function setProgress(step, active) {
    if (active) {
      progressDiv.classList.remove('hidden');
      const texts = {
        translate: 'Traduciendo a Triton...',
        gpu: 'Validando en GPU — compilando y ejecutando (2-5 min)...',
        compare: 'Comparando precisión y velocidad vs PyTorch (2-5 min)...',
        evaluate: 'Evaluando numéricamente...',
        load: 'Cargando detalles...',
      };
      progressText.textContent = texts[step] || 'Procesando...';
    } else {
      progressDiv.classList.add('hidden');
    }
  }

  function renderResults() {
    let html = '';

    // If we have a loaded run detail, show it
    if (currentRunDetail) {
      const data = currentRunDetail;
      html += '<div class="result-section current-run">';
      html += '<h3 class="toggle" data-target="res-current">Generación Actual</h3>';
      html += '<div id="res-current" class="collapsible">';
      html += `<p><strong>Job ID:</strong> ${data.job_id}</p>`;
      html += `<p><strong>Estado:</strong> <span class="badge ${data.status === 'completed' ? 'success' : 'error'}">${data.status}</span></p>`;
      
      if (data.source_code) {
        html += '<p><strong>Código fuente:</strong></p>';
        html += `<pre class="code-block"><code>${escapeHtml(data.source_code)}</code></pre>`;
      }
      
      if (data.generated_code) {
        html += '<p><strong>Código Triton:</strong></p>';
        html += '<div class="code-actions">';
        html += `<button class="btn btn-small" onclick="copyCode('triton')">Copiar</button>`;
        html += `<button class="btn btn-small" onclick="openFile('triton')">Abrir en nuevo archivo</button>`;
        html += '</div>';
        html += `<pre class="code-block"><code>${escapeHtml(data.generated_code)}</code></pre>`;
      }
      
      if (data.validation) {
        html += `<p><strong>Validación estática:</strong> <span class="badge ${data.validation.passed ? 'success' : 'warning'}">${data.validation.passed ? 'OK' : 'FALLO'}</span></p>`;
      }
      
      if (data.gpu_validation) {
        html += `<p><strong>GPU:</strong> <span class="badge ${data.gpu_validation.compilation_pass ? 'success' : 'error'}">Compilación ${data.gpu_validation.compilation_pass ? 'OK' : 'FALLO'}</span></p>`;
      }
      
      html += '</div></div>';
    }

    if (translateResult) {
      html += '<div class="result-section">';
      html += '<h3 class="toggle" data-target="res-translate">Traducción</h3>';
      html += '<div id="res-translate" class="collapsible">';
      if (translateResult.error) {
        html += `<div class="error-box">${escapeHtml(translateResult.error)}</div>`;
      } else {
        const data = translateResult.data;
        html += `<p><strong>Job ID:</strong> ${data.job_id}</p>`;
        html += `<p><strong>Estado:</strong> <span class="badge ${data.status === 'completed' ? 'success' : 'error'}">${data.status}</span></p>`;
        html += `<p><strong>Validación estática:</strong> <span class="badge ${data.validation.passed ? 'success' : 'warning'}">${data.validation.passed ? 'OK' : 'FALLO'}</span></p>`;
        if (data.validation.errors.length > 0) {
          html += '<ul class="error-list">' + data.validation.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('') + '</ul>';
        }
        if (data.validation.warnings.length > 0) {
          html += '<ul class="warn-list">' + data.validation.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('') + '</ul>';
        }
        if (data.generated_code) {
          html += '<div class="code-actions">';
          html += `<button class="btn btn-small" onclick="copyCode('triton')">Copiar</button>`;
          html += `<button class="btn btn-small" onclick="openFile('triton')">Abrir en nuevo archivo</button>`;
          html += '</div>';
          html += `<pre class="code-block"><code>${escapeHtml(data.generated_code)}</code></pre>`;
        }
        // Inline GPU button - only if validation passed and status is completed
        if (data.validation.passed && data.status === 'completed') {
          html += `<div class="inline-action">`;
          html += `<button class="btn btn-gpu-inline" onclick="validateGpu('${data.job_id}')">🚀 Validar GPU</button>`;
          html += `<span class="hint">Compila y ejecuta el kernel en una GPU real vía Modal</span>`;
          html += `</div>`;
        }
      }
      html += '</div></div>';
    }

    if (gpuResult) {
      html += '<div class="result-section">';
      html += '<h3 class="toggle" data-target="res-gpu">Validación GPU</h3>';
      html += '<div id="res-gpu" class="collapsible">';
      if (gpuResult.error) {
        html += `<div class="error-box">${escapeHtml(gpuResult.error)}</div>`;
      } else {
        const data = gpuResult.data;
        html += `<p><strong>Compilación:</strong> <span class="badge ${data.compilation_pass ? 'success' : 'error'}">${data.compilation_pass ? 'OK' : 'FALLO'}</span></p>`;
        html += `<p><strong>Ejecución:</strong> <span class="badge ${data.execution_pass ? 'success' : 'error'}">${data.execution_pass ? 'OK' : 'FALLO'}</span></p>`;
        if (data.output_shape) html += `<p><strong>Output shape:</strong> ${escapeHtml(data.output_shape)}</p>`;
        if (data.device) html += `<p><strong>Dispositivo:</strong> ${escapeHtml(data.device)}</p>`;
        if (data.pytorch_time_ms != null) {
          html += `<p><strong>PyTorch (GPU):</strong> ${data.pytorch_time_ms.toFixed(3)} ms</p>`;
        }
        if (data.errors && data.errors.length > 0) {
          html += '<ul class="error-list">' + data.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('') + '</ul>';
        }
        if (data.logs && data.logs.length > 0) {
          html += `<div class="log-toggle" onclick="this.nextElementSibling.classList.toggle('collapsed')">📋 Ver logs GPU (${data.logs.length} líneas)</div>`;
          html += `<pre class="log-block collapsed"><code>${escapeHtml(data.logs.join('\n'))}</code></pre>`;
        }
      }
      html += '</div></div>';
    }

    if (compareResult) {
      html += '<div class="result-section">';
      html += '<h3 class="toggle" data-target="res-compare">Comparación vs PyTorch</h3>';
      html += '<div id="res-compare" class="collapsible">';
      if (compareResult.error) {
        html += `<div class="error-box">${escapeHtml(compareResult.error)}</div>`;
      } else {
        const d = compareResult.data;
        const strategy = d.strategy || 'user_comparison';
        const isTritonBench = strategy === 'tritonbench';

        html += `<p><strong>Estrategia:</strong> <span class="badge ${isTritonBench ? 'info' : 'default'}">${isTritonBench ? 'TritonBench' : 'Comparación directa PyTorch'}</span></p>`;

        // ── Timing Section ──
        html += '<div class="timing-section">';
        html += '<h4>⏱️ Tiempos de ejecución</h4>';

        // PyTorch (reference) — always show first
        const pytorchTime = d.ref_time_ms != null ? d.ref_time_ms : (gpuResult && gpuResult.data ? gpuResult.data.pytorch_time_ms : null);
        if (pytorchTime != null) {
          html += `<div class="timing-row">`;
          html += `<span class="timing-label">🐍 PyTorch</span>`;
          html += `<span class="timing-value">${pytorchTime.toFixed(3)} ms</span>`;
          html += `</div>`;
        } else {
          html += `<p class="hint">PyTorch: no medido</p>`;
        }

        // Triton (generated) — show second
        if (d.gen_time_ms != null) {
          html += `<div class="timing-row">`;
          html += `<span class="timing-label">⚡ Triton</span>`;
          html += `<span class="timing-value">${d.gen_time_ms.toFixed(3)} ms</span>`;
          html += `</div>`;
        } else {
          html += `<p class="hint">Triton: no medido (posiblemente falló compilación o ejecución)</p>`;
        }

        // Speedup
        if (d.speedup != null) {
          const speedupBadge = d.speedup >= 1.5 ? 'success' : (d.speedup >= 1.0 ? 'warning' : 'error');
          html += `<div class="timing-row speedup">`;
          html += `<span class="timing-label">📈 Speedup</span>`;
          html += `<span class="timing-value badge ${speedupBadge}">${d.speedup.toFixed(2)}x</span>`;
          html += `</div>`;
        }
        html += '</div>';

        // ── Accuracy / Quality Section ──
        html += '<div class="accuracy-section">';
        html += '<h4>🎯 Precisión</h4>';

        if (isTritonBench) {
          // TritonBench specific metrics
          html += `<p><strong>Call accuracy:</strong> <span class="badge ${d.call_accuracy ? 'success' : 'error'}">${d.call_accuracy ? 'OK' : 'FALLO'}</span></p>`;
          html += `<p><strong>Exec accuracy:</strong> <span class="badge ${d.exec_accuracy ? 'success' : 'error'}">${d.exec_accuracy ? 'OK' : 'FALLO'}</span></p>`;
          if (d.operator) html += `<p><strong>Operador:</strong> ${escapeHtml(d.operator)}</p>`;
        } else {
          // User comparison metrics
          html += `<p><strong>Compilación:</strong> <span class="badge ${d.compilation_pass ? 'success' : 'error'}">${d.compilation_pass ? 'OK' : 'FALLO'}</span></p>`;
          html += `<p><strong>Precisión numérica:</strong> <span class="badge ${d.accuracy_pass ? 'success' : 'error'}">${d.accuracy_pass ? 'OK' : 'FALLO'}</span></p>`;
          if (d.max_diff != null) html += `<p><strong>Max diff:</strong> ${d.max_diff.toExponential(3)}</p>`;
        }
        html += '</div>';

        // ── Recommendation ──
        if (d.suggest_replacement) {
          html += `<p><span class="badge success">✅ Recomendado reemplazar PyTorch con este kernel</span></p>`;
        }
        if (d.reason) html += `<p class="hint">${escapeHtml(d.reason)}</p>`;

        // ── Errors ──
        if (d.errors && d.errors.length > 0) {
          html += '<ul class="error-list">' + d.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('') + '</ul>';
        }

        // ── Logs ──
        if (d.logs && d.logs.length > 0) {
          html += `<div class="log-toggle" onclick="this.nextElementSibling.classList.toggle('collapsed')">📋 Ver logs comparación (${d.logs.length} líneas)</div>`;
          html += `<pre class="log-block collapsed"><code>${escapeHtml(d.logs.join('\n'))}</code></pre>`;
        }
      }
      html += '</div></div>';
    }

    if (evaluateResult) {
      html += '<div class="result-section">';
      html += '<h3 class="toggle" data-target="res-evaluate">Evaluación Numérica</h3>';
      html += '<div id="res-evaluate" class="collapsible">';
      if (evaluateResult.error) {
        html += `<div class="error-box">${escapeHtml(evaluateResult.error)}</div>`;
      } else {
        const data = evaluateResult.data;
        html += `<p><strong>Precisión:</strong> <span class="badge ${data.accuracy_pass ? 'success' : 'error'}">${data.accuracy_pass ? 'OK' : 'FALLO'}</span></p>`;
        html += `<p><strong>Error máximo:</strong> ${data.max_error.toExponential(2)}</p>`;
        html += `<p><strong>Speedup:</strong> <span class="badge ${data.speedup > 1 ? 'success' : 'warning'}">${data.speedup.toFixed(2)}x</span></p>`;
        if (data.errors.length > 0) {
          html += '<ul class="error-list">' + data.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('') + '</ul>';
        }
      }
      html += '</div></div>';
    }

    resultsDiv.innerHTML = html;
    attachToggles();
  }

  function attachToggles() {
    document.querySelectorAll('.toggle').forEach(toggle => {
      toggle.addEventListener('click', () => {
        const target = document.getElementById(toggle.dataset.target);
        if (target) {
          target.classList.toggle('collapsed');
          toggle.classList.toggle('collapsed');
        }
      });
    });
  }

  function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function updateButtons() {
    btnTranslate.disabled = progressDiv.classList.contains('hidden') === false;
  }

  btnTranslate.addEventListener('click', () => {
    const sourceCode = codeInput.value;
    if (!sourceCode.trim()) {
      alert('Pega el código de la definición de la función antes de traducir.');
      return;
    }
    const callSiteCode = callSiteInput.value;
    vscode.postMessage({ command: 'translate', sourceCode, callSiteCode, dims: {} });
  });

  window.addEventListener('message', (event) => {
    const message = event.data;
    switch (message.command) {
      case 'setProgress':
        setProgress(message.step, message.active);
        updateButtons();
        break;
      case 'setResult':
        if (message.step === 'translate') {
          translateResult = message;
          if (message.data) jobId = message.data.job_id;
          // Clear previous GPU/compare/evaluate results on new translation
          gpuResult = null;
          compareResult = null;
          evaluateResult = null;
        } else if (message.step === 'gpu') {
          gpuResult = message;
        } else if (message.step === 'compare') {
          compareResult = message;
        } else if (message.step === 'evaluate') {
          evaluateResult = message;
        }
        renderResults();
        updateButtons();
        break;
      case 'setHistory':
        if (message.data && message.data.items) {
          runsHistory = message.data.items;
        } else {
          runsHistory = [];
        }
        renderHistory();
        break;
      case 'setRunDetail':
        if (message.data) {
          currentRunDetail = message.data;
          // Also populate the results
          if (message.data.job_id) {
            jobId = message.data.job_id;
          }
          // Restore call site if available
          if (message.data.call_site_code) {
            callSiteInput.value = message.data.call_site_code;
          }
          // Set up translate/gpu/compare/evaluate results from the detail
          translateResult = {
            data: {
              job_id: message.data.job_id,
              status: message.data.status,
              validation: message.data.validation || { passed: false, errors: [], warnings: [] },
              generated_code: message.data.generated_code,
            }
          };
          if (message.data.gpu_validation) {
            gpuResult = { data: message.data.gpu_validation };
          } else {
            gpuResult = null;
          }
          if (message.data.comparison_json) {
            compareResult = { data: message.data.comparison_json };
          } else {
            compareResult = null;
          }
          // Note: evaluate data is not in JobDetail, would need separate call
          evaluateResult = null;
          renderResults();
        } else {
          alert('Error cargando detalle: ' + (message.error || 'desconocido'));
        }
        updateButtons();
        break;
    }
  });

  window.copyCode = (type) => {
    const code = currentRunDetail?.generated_code || (translateResult?.data?.generated_code);
    if (code) {
      vscode.postMessage({ command: 'copyCode', code });
    }
  };

  window.openFile = (type) => {
    const code = currentRunDetail?.generated_code || (translateResult?.data?.generated_code);
    if (code) {
      vscode.postMessage({ command: 'openInNewFile', code });
    }
  };

  window.validateGpu = (jobId) => {
    vscode.postMessage({ command: 'gpuValidate', jobId });
  };

  window.evaluateRun = (jobId) => {
    vscode.postMessage({ command: 'evaluate', jobId, dims: {} });
  };

  updateButtons();
})();
