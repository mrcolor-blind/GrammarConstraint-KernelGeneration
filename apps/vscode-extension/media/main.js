(function () {
  const vscode = acquireVsCodeApi();
  let jobId = null;
  let translateResult = null;
  let gpuResult = null;
  let evaluateResult = null;

  const dimsForm = document.getElementById('dims-form');
  const btnTranslate = document.getElementById('btn-translate');
  const btnGpu = document.getElementById('btn-gpu');
  const btnEvaluate = document.getElementById('btn-evaluate');
  const progressDiv = document.getElementById('progress');
  const progressText = document.getElementById('progress-text');
  const resultsDiv = document.getElementById('results');

  function renderDimsForm() {
    if (!initialDims || initialDims.length === 0) {
      dimsForm.innerHTML = '<p class="warning">No se detectaron dimensiones. Anade comentarios @in/@out con shapes.</p>';
      return;
    }

    dimsForm.innerHTML = '';
    initialDims.forEach(dim => {
      const label = document.createElement('label');
      label.className = 'dim-label';
      label.innerHTML = `<span>${dim}</span><input type="number" class="dim-input" data-dim="${dim}" value="128" min="1">`;
      dimsForm.appendChild(label);
    });
  }

  function getDims() {
    const inputs = dimsForm.querySelectorAll('.dim-input');
    const dims = {};
    inputs.forEach(input => {
      dims[input.dataset.dim] = parseInt(input.value, 10) || 1;
    });
    return dims;
  }

  function setProgress(step, active) {
    if (active) {
      progressDiv.classList.remove('hidden');
      const texts = {
        translate: 'Traduciendo a Triton...',
        gpu: 'Validando en GPU (puede tardar 2-5 min)...',
        evaluate: 'Evaluando numericamente...',
      };
      progressText.textContent = texts[step] || 'Procesando...';
    } else {
      progressDiv.classList.add('hidden');
    }
  }

  function renderResults() {
    let html = '';

    if (translateResult) {
      html += '<div class="result-section">';
      html += '<h3 class="toggle" data-target="res-translate">Traduccion</h3>';
      html += '<div id="res-translate" class="collapsible">';
      if (translateResult.error) {
        html += `<div class="error-box">${escapeHtml(translateResult.error)}</div>`;
      } else {
        const data = translateResult.data;
        html += `<p><strong>Job ID:</strong> ${data.job_id}</p>`;
        html += `<p><strong>Estado:</strong> <span class="badge ${data.status === 'completed' ? 'success' : 'error'}">${data.status}</span></p>`;
        html += `<p><strong>Validacion estatica:</strong> <span class="badge ${data.validation.passed ? 'success' : 'warning'}">${data.validation.passed ? 'OK' : 'FALLO'}</span></p>`;
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
      }
      html += '</div></div>';
    }

    if (gpuResult) {
      html += '<div class="result-section">';
      html += '<h3 class="toggle" data-target="res-gpu">Validacion GPU</h3>';
      html += '<div id="res-gpu" class="collapsible">';
      if (gpuResult.error) {
        html += `<div class="error-box">${escapeHtml(gpuResult.error)}</div>`;
      } else {
        const data = gpuResult.data;
        html += `<p><strong>Compilacion:</strong> <span class="badge ${data.compilation_pass ? 'success' : 'error'}">${data.compilation_pass ? 'OK' : 'FALLO'}</span></p>`;
        html += `<p><strong>Ejecucion:</strong> <span class="badge ${data.execution_pass ? 'success' : 'error'}">${data.execution_pass ? 'OK' : 'FALLO'}</span></p>`;
        if (data.output_shape) html += `<p><strong>Output shape:</strong> ${escapeHtml(data.output_shape)}</p>`;
        if (data.device) html += `<p><strong>Dispositivo:</strong> ${escapeHtml(data.device)}</p>`;
        if (data.errors.length > 0) {
          html += '<ul class="error-list">' + data.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('') + '</ul>';
        }
      }
      html += '</div></div>';
    }

    if (evaluateResult) {
      html += '<div class="result-section">';
      html += '<h3 class="toggle" data-target="res-evaluate">Evaluacion Numerica</h3>';
      html += '<div id="res-evaluate" class="collapsible">';
      if (evaluateResult.error) {
        html += `<div class="error-box">${escapeHtml(evaluateResult.error)}</div>`;
      } else {
        const data = evaluateResult.data;
        html += `<p><strong>Precision:</strong> <span class="badge ${data.accuracy_pass ? 'success' : 'error'}">${data.accuracy_pass ? 'OK' : 'FALLO'}</span></p>`;
        html += `<p><strong>Error maximo:</strong> ${data.max_error.toExponential(2)}</p>`;
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
    btnGpu.disabled = !jobId || !translateResult || (translateResult.data && translateResult.data.status !== 'completed');
    btnEvaluate.disabled = !jobId || !gpuResult || (gpuResult.data && !(gpuResult.data.compilation_pass && gpuResult.data.execution_pass));
  }

  btnTranslate.addEventListener('click', () => {
    const dims = getDims();
    vscode.postMessage({ command: 'translate', dims });
  });

  btnGpu.addEventListener('click', () => {
    if (!jobId) return;
    vscode.postMessage({ command: 'gpuValidate', jobId });
  });

  btnEvaluate.addEventListener('click', () => {
    if (!jobId) return;
    const dims = getDims();
    vscode.postMessage({ command: 'evaluate', jobId, dims });
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
        } else if (message.step === 'gpu') {
          gpuResult = message;
        } else if (message.step === 'evaluate') {
          evaluateResult = message;
        }
        renderResults();
        updateButtons();
        break;
    }
  });

  window.copyCode = (type) => {
    if (type === 'triton' && translateResult && translateResult.data && translateResult.data.generated_code) {
      vscode.postMessage({ command: 'copyCode', code: translateResult.data.generated_code });
    }
  };

  window.openFile = (type) => {
    if (type === 'triton' && translateResult && translateResult.data && translateResult.data.generated_code) {
      vscode.postMessage({ command: 'openInNewFile', code: translateResult.data.generated_code });
    }
  };

  renderDimsForm();
  updateButtons();
})();
