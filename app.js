/**
 * ============================================================
 * ML Explorer — Frontend Application
 * ============================================================
 * Handles UI interactions, API calls, chart rendering,
 * and the step-by-step wizard flow.
 * ============================================================
 */

// ============================================================
// CONFIG
// ============================================================
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : 'https://ml-explorer-fb3b.onrender.com'; // Live Render backend

// ============================================================
// STATE
// ============================================================
const state = {
  sessionId: null,
  schema: null,
  edaData: null,
  trainResults: null,
  shapData: null,
  currentStep: 1,
  charts: {},
};

// ============================================================
// DOM ELEMENTS
// ============================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initUpload();
  initStepper();
  initButtons();
  initSliders();
  checkApiHealth();
});

// ============================================================
// API HELPERS
// ============================================================
async function apiCall(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const defaultOptions = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };

  try {
    const res = await fetch(url, defaultOptions);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `API error: ${res.status}`);
    }
    return res;
  } catch (err) {
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      throw new Error('Cannot connect to API server. Make sure the backend is running.');
    }
    throw err;
  }
}

async function checkApiHealth() {
  const dot = $('#api-dot');
  const text = $('#api-status-text');
  const urlDisplay = $('#api-url-display');
  urlDisplay.textContent = API_BASE;

  try {
    const res = await apiCall('/api/health');
    const data = await res.json();
    dot.className = 'status-dot connected';
    text.textContent = `Connected · ${data.sessions_active} active sessions`;
  } catch {
    dot.className = 'status-dot disconnected';
    text.textContent = 'API offline — start backend with: uvicorn main:app --reload';
  }
}

// ============================================================
// LOADING & TOASTS
// ============================================================
function showLoading(message = 'Processing...', sub = '', showProgress = false) {
  const overlay = $('#loading-overlay');
  $('#loading-text').textContent = message;
  $('#loading-sub').textContent = sub;
  $('#train-progress').style.display = showProgress ? 'block' : 'none';
  overlay.classList.add('active');
}

function hideLoading() {
  $('#loading-overlay').classList.remove('active');
}

function showToast(message, type = 'info') {
  const container = $('#toast-container');
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(30px)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ============================================================
// STEPPER NAVIGATION
// ============================================================
function initStepper() {
  $$('.step-item').forEach(step => {
    step.addEventListener('click', () => {
      const targetStep = parseInt(step.dataset.step);
      // Only allow going back or to completed steps
      if (targetStep < state.currentStep) {
        goToStep(targetStep);
      }
    });
  });
}

function goToStep(step) {
  // Hide all panels
  $$('.panel').forEach(p => p.classList.remove('active'));
  $(`#panel-${step}`).classList.add('active');

  // Update stepper visuals
  $$('.step-item').forEach(s => {
    const sNum = parseInt(s.dataset.step);
    s.classList.remove('active', 'completed');
    if (sNum === step) s.classList.add('active');
    else if (sNum < step) s.classList.add('completed');
  });

  $$('.step-connector').forEach((c, i) => {
    c.classList.remove('active', 'completed');
    if (i + 1 < step) c.classList.add('completed');
    else if (i + 1 === step) c.classList.add('active');
  });

  state.currentStep = step;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================
// UPLOAD HANDLING
// ============================================================
function initUpload() {
  const zone = $('#upload-zone');
  const fileInput = $('#file-input');

  // Browse trigger
  $('#browse-trigger').addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  zone.addEventListener('click', () => fileInput.click());

  // Drag & drop
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      uploadFile(e.target.files[0]);
    }
  });

  // Sample dataset buttons
  $$('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      uploadSample(btn.dataset.sample);
    });
  });
}

async function uploadFile(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please upload a CSV file', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Max 10MB.', 'error');
    return;
  }

  showLoading('Uploading dataset...', file.name);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    handleUploadSuccess(data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

async function uploadSample(name) {
  showLoading('Loading sample dataset...', name);

  try {
    const res = await apiCall(`/api/upload?sample=${name}`, { method: 'POST' });
    const data = await res.json();
    handleUploadSuccess(data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

function handleUploadSuccess(data) {
  state.sessionId = data.session_id;
  state.schema = data.schema;

  showToast(`Loaded ${data.filename} successfully!`, 'success');
  renderPreview(data);
  checkApiHealth();
}

// ============================================================
// DATA PREVIEW RENDERING
// ============================================================
function renderPreview(data) {
  const { schema, filename } = data;

  $('#file-name-display').textContent = filename;
  $('#row-count').textContent = schema.shape.rows.toLocaleString();
  $('#col-count').textContent = schema.shape.cols;

  // Data table
  const tableWrapper = $('#data-table-wrapper');
  const preview = schema.preview;
  const cols = schema.columns.map(c => c.name);

  let html = '<table class="data-table"><thead><tr>';
  html += '<th>#</th>';
  cols.forEach(c => { html += `<th>${escapeHtml(c)}</th>`; });
  html += '</tr></thead><tbody>';

  preview.forEach((row, i) => {
    html += `<tr><td style="color:var(--text-muted)">${i + 1}</td>`;
    cols.forEach(c => {
      const val = row[c];
      html += `<td>${val === null ? '<span style="color:var(--red);opacity:0.5">null</span>' : escapeHtml(String(val))}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  tableWrapper.innerHTML = html;

  // Schema grid
  const schemaGrid = $('#schema-grid');
  schemaGrid.innerHTML = '';
  schema.columns.forEach(col => {
    const pill = document.createElement('div');
    pill.className = 'schema-pill';
    const missingText = col.missing > 0 ? `${col.missing_pct}% null` : '✓';
    pill.innerHTML = `
      <span class="pill-type ${col.type}">${col.type === 'numeric' ? 'NUM' : 'CAT'}</span>
      <span class="pill-name">${escapeHtml(col.name)}</span>
      <span class="pill-missing">${missingText}</span>
    `;
    schemaGrid.appendChild(pill);
  });

  // Show preview
  $('#preview-container').style.display = 'block';

  // Populate target select in step 3
  const targetSelect = $('#target-select');
  targetSelect.innerHTML = '<option value="">Select target column...</option>';
  schema.columns.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col.name;
    opt.textContent = `${col.name} (${col.type}, ${col.unique} unique)`;
    targetSelect.appendChild(opt);
  });
}

// ============================================================
// EDA
// ============================================================
async function loadEDA() {
  if (state.edaData) {
    renderEDA(state.edaData);
    return;
  }

  showLoading('Generating EDA...', 'Analyzing distributions and correlations');

  try {
    const res = await apiCall('/api/eda', {
      method: 'POST',
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    state.edaData = await res.json();
    renderEDA(state.edaData);
    showToast('EDA generated!', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

function renderEDA(eda) {
  const grid = $('#eda-grid');
  grid.innerHTML = '';

  // Destroy existing charts
  Object.values(state.charts).forEach(c => c.destroy && c.destroy());
  state.charts = {};

  // Distribution charts
  const distEntries = Object.entries(eda.distributions);
  distEntries.forEach(([colName, distData], i) => {
    const card = document.createElement('div');
    card.className = 'card';
    const canvasId = `eda-chart-${i}`;
    card.innerHTML = `
      <div class="card-title">${escapeHtml(colName)}</div>
      <div class="chart-container"><canvas id="${canvasId}"></canvas></div>
    `;
    grid.appendChild(card);

    // Render chart after DOM insert
    requestAnimationFrame(() => {
      const ctx = document.getElementById(canvasId);
      if (!ctx) return;

      if (distData.type === 'numeric') {
        // Histogram
        const labels = distData.bins.slice(0, -1).map((b, j) =>
          `${b.toFixed(1)}–${distData.bins[j + 1].toFixed(1)}`
        );
        state.charts[canvasId] = new Chart(ctx, {
          type: 'bar',
          data: {
            labels,
            datasets: [{
              label: 'Frequency',
              data: distData.values,
              backgroundColor: 'rgba(102, 126, 234, 0.4)',
              borderColor: 'rgba(102, 126, 234, 0.8)',
              borderWidth: 1,
              borderRadius: 3,
            }],
          },
          options: chartDefaults('Histogram'),
        });
      } else {
        // Bar chart for categorical
        const colors = generateColors(distData.labels.length);
        state.charts[canvasId] = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: distData.labels,
            datasets: [{
              data: distData.values,
              backgroundColor: colors,
              borderWidth: 0,
            }],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'right',
                labels: { color: '#8b92a8', font: { size: 11, family: 'Inter' }, padding: 10 },
              },
            },
          },
        });
      }
    });
  });

  // Correlation heatmap
  if (eda.correlations) {
    const card = document.createElement('div');
    card.className = 'card full-width';
    card.innerHTML = `<div class="card-title">📈 Correlation Matrix</div><div class="heatmap-container" id="corr-heatmap"></div>`;
    grid.appendChild(card);

    requestAnimationFrame(() => {
      renderCorrelationHeatmap(eda.correlations);
    });
  }

  // Missing data summary
  if (Object.keys(eda.missing_summary).length > 0) {
    const card = document.createElement('div');
    card.className = 'card';
    let missingHtml = '<div class="card-title">⚠️ Missing Values</div>';
    Object.entries(eda.missing_summary).forEach(([col, info]) => {
      const pct = info.pct;
      const color = pct > 30 ? 'var(--red)' : pct > 10 ? 'var(--amber)' : 'var(--green)';
      missingHtml += `
        <div class="shap-bar">
          <span class="feature-name">${escapeHtml(col)}</span>
          <div class="bar-track">
            <div class="bar-fill positive" style="width:${Math.min(pct, 100)}%; background:${color};"></div>
          </div>
          <span class="importance-val">${pct}%</span>
        </div>
      `;
    });
    card.innerHTML = missingHtml;
    grid.appendChild(card);
  }
}

function renderCorrelationHeatmap(corrData) {
  const container = document.getElementById('corr-heatmap');
  if (!container) return;

  const { columns, values } = corrData;
  let html = '<table class="heatmap-table"><thead><tr><th></th>';
  columns.forEach(c => {
    const short = c.length > 12 ? c.substring(0, 10) + '…' : c;
    html += `<th title="${escapeHtml(c)}">${escapeHtml(short)}</th>`;
  });
  html += '</tr></thead><tbody>';

  values.forEach((row, i) => {
    const short = columns[i].length > 12 ? columns[i].substring(0, 10) + '…' : columns[i];
    html += `<tr><th style="writing-mode:horizontal-tb; text-align:right; white-space:nowrap;" title="${escapeHtml(columns[i])}">${escapeHtml(short)}</th>`;
    row.forEach(val => {
      const bg = correlationColor(val);
      const textColor = Math.abs(val) > 0.5 ? '#fff' : '#8b92a8';
      html += `<td style="background:${bg}; color:${textColor};">${val.toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function correlationColor(val) {
  const abs = Math.abs(val);
  if (val > 0) {
    return `rgba(59, 130, 246, ${abs * 0.7 + 0.05})`;
  } else {
    return `rgba(239, 68, 68, ${abs * 0.7 + 0.05})`;
  }
}

// ============================================================
// TRAINING
// ============================================================
async function trainModels() {
  const targetCol = $('#target-select').value;
  if (!targetCol) {
    showToast('Please select a target column', 'error');
    return;
  }

  const selectedModels = [];
  $$('input[name="model"]:checked').forEach(cb => selectedModels.push(cb.value));
  if (selectedModels.length === 0) {
    showToast('Please select at least one model', 'error');
    return;
  }

  const testSize = parseInt($('#test-split').value) / 100;

  showLoading('Training models...', `${selectedModels.length} models · ${targetCol}`, true);

  try {
    const res = await apiCall('/api/train', {
      method: 'POST',
      body: JSON.stringify({
        session_id: state.sessionId,
        target_column: targetCol,
        test_size: testSize,
        models: selectedModels,
      }),
    });

    state.trainResults = await res.json();
    renderResults(state.trainResults);
    goToStep(4);
    showToast('Models trained successfully!', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

// ============================================================
// RESULTS RENDERING
// ============================================================
function renderResults(results) {
  const { task_type, models, best_model } = results;

  // Best model badge
  if (best_model) {
    $('#best-badge').style.display = 'inline-flex';
    $('#best-model-name').textContent = best_model;
  }

  // Metric cards for best model
  const bestMetrics = models[best_model] || {};
  const metricsRow = $('#metrics-row');
  metricsRow.innerHTML = '';

  const metricConfig = task_type === 'classification'
    ? [
        { key: 'accuracy', label: 'Accuracy', fmt: v => (v * 100).toFixed(1) + '%' },
        { key: 'f1', label: 'F1 Score', fmt: v => v.toFixed(3) },
        { key: 'precision', label: 'Precision', fmt: v => (v * 100).toFixed(1) + '%' },
        { key: 'recall', label: 'Recall', fmt: v => (v * 100).toFixed(1) + '%' },
        { key: 'auc_roc', label: 'AUC-ROC', fmt: v => v ? v.toFixed(3) : 'N/A' },
      ]
    : [
        { key: 'r2', label: 'R² Score', fmt: v => v.toFixed(4) },
        { key: 'rmse', label: 'RMSE', fmt: v => v.toFixed(3) },
        { key: 'mae', label: 'MAE', fmt: v => v.toFixed(3) },
        { key: 'mape', label: 'MAPE', fmt: v => v.toFixed(1) + '%' },
      ];

  metricConfig.forEach(mc => {
    const val = bestMetrics[mc.key];
    if (val === undefined) return;
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <span class="metric-value">${mc.fmt(val)}</span>
      <span class="metric-label">${mc.label}</span>
    `;
    metricsRow.appendChild(card);
  });

  // Comparison table
  const wrapper = $('#comparison-table-wrapper');
  const metricKeys = metricConfig.map(m => m.key);
  let tableHtml = '<table class="comparison-table"><thead><tr><th>Model</th>';
  metricConfig.forEach(m => { tableHtml += `<th>${m.label}</th>`; });
  tableHtml += '</tr></thead><tbody>';

  Object.entries(models).forEach(([name, metrics]) => {
    if (metrics.error) return;
    const isBest = name === best_model;
    tableHtml += `<tr class="${isBest ? 'best-row' : ''}">`;
    tableHtml += `<td class="model-name-cell">${escapeHtml(name)}${isBest ? '<span class="best-star">⭐</span>' : ''}</td>`;
    metricConfig.forEach(mc => {
      const val = metrics[mc.key];
      tableHtml += `<td>${val !== undefined && val !== null ? mc.fmt(val) : '—'}</td>`;
    });
    tableHtml += '</tr>';
  });
  tableHtml += '</tbody></table>';
  wrapper.innerHTML = tableHtml;

  // Charts
  const chartsContainer = $('#results-charts');
  chartsContainer.innerHTML = '';

  // Destroy old charts
  ['results-bar', 'results-radar', 'results-cm'].forEach(k => {
    if (state.charts[k]) { state.charts[k].destroy(); delete state.charts[k]; }
  });

  // Bar chart comparing models
  const barCard = document.createElement('div');
  barCard.className = 'card';
  barCard.innerHTML = `<div class="card-title">📊 Model Comparison</div><div class="chart-container"><canvas id="results-bar-chart"></canvas></div>`;
  chartsContainer.appendChild(barCard);

  requestAnimationFrame(() => {
    const modelNames = Object.keys(models).filter(n => !models[n].error);
    const primaryMetric = task_type === 'classification' ? 'f1' : 'r2';
    const secondaryMetric = task_type === 'classification' ? 'accuracy' : 'mae';

    const ctx = document.getElementById('results-bar-chart');
    if (!ctx) return;

    state.charts['results-bar'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: modelNames,
        datasets: [
          {
            label: task_type === 'classification' ? 'F1 Score' : 'R²',
            data: modelNames.map(n => models[n][primaryMetric] || 0),
            backgroundColor: 'rgba(102, 126, 234, 0.5)',
            borderColor: 'rgba(102, 126, 234, 0.9)',
            borderWidth: 1,
            borderRadius: 5,
          },
          {
            label: task_type === 'classification' ? 'Accuracy' : 'MAE',
            data: modelNames.map(n => models[n][secondaryMetric] || 0),
            backgroundColor: 'rgba(0, 212, 255, 0.4)',
            borderColor: 'rgba(0, 212, 255, 0.8)',
            borderWidth: 1,
            borderRadius: 5,
          },
        ],
      },
      options: chartDefaults('Score'),
    });
  });

  // Radar chart (classification only)
  if (task_type === 'classification') {
    const radarCard = document.createElement('div');
    radarCard.className = 'card';
    radarCard.innerHTML = `<div class="card-title">🕸️ Performance Radar</div><div class="chart-container"><canvas id="results-radar-chart"></canvas></div>`;
    chartsContainer.appendChild(radarCard);

    requestAnimationFrame(() => {
      const radarCtx = document.getElementById('results-radar-chart');
      if (!radarCtx) return;

      const modelNames = Object.keys(models).filter(n => !models[n].error);
      const radarLabels = ['Accuracy', 'F1', 'Precision', 'Recall'];
      const radarKeys = ['accuracy', 'f1', 'precision', 'recall'];
      const colors = generateColors(modelNames.length);

      state.charts['results-radar'] = new Chart(radarCtx, {
        type: 'radar',
        data: {
          labels: radarLabels,
          datasets: modelNames.map((name, i) => ({
            label: name,
            data: radarKeys.map(k => models[name][k] || 0),
            borderColor: colors[i],
            backgroundColor: colors[i].replace('1)', '0.1)'),
            borderWidth: 2,
            pointRadius: 3,
          })),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            r: {
              beginAtZero: true,
              max: 1,
              ticks: { display: false },
              grid: { color: 'rgba(255,255,255,0.05)' },
              angleLines: { color: 'rgba(255,255,255,0.05)' },
              pointLabels: { color: '#8b92a8', font: { size: 11, family: 'Inter' } },
            },
          },
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: '#8b92a8', font: { size: 11, family: 'Inter' }, padding: 15 },
            },
          },
        },
      });
    });
  }

  // Confusion matrix
  if (task_type === 'classification' && bestMetrics.confusion_matrix) {
    const cmCard = document.createElement('div');
    cmCard.className = 'card';
    cmCard.innerHTML = `<div class="card-title">🔢 Confusion Matrix — ${escapeHtml(best_model)}</div><div id="cm-container" style="text-align:center;"></div>`;
    chartsContainer.appendChild(cmCard);

    requestAnimationFrame(() => {
      renderConfusionMatrix(bestMetrics.confusion_matrix);
    });
  }
}

function renderConfusionMatrix(cm) {
  const container = document.getElementById('cm-container');
  if (!container) return;

  const n = cm.length;
  const maxVal = Math.max(...cm.flat());

  let html = '<table class="cm-table"><thead><tr><th></th>';
  for (let j = 0; j < n; j++) html += `<th>Pred ${j}</th>`;
  html += '</tr></thead><tbody>';

  for (let i = 0; i < n; i++) {
    html += `<tr><th>Actual ${i}</th>`;
    for (let j = 0; j < n; j++) {
      const val = cm[i][j];
      const intensity = val / maxVal;
      const bg = i === j
        ? `rgba(16, 185, 129, ${intensity * 0.6 + 0.05})`
        : `rgba(239, 68, 68, ${intensity * 0.5 + 0.02})`;
      html += `<td style="background:${bg}; font-weight:700;">${val}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

// ============================================================
// SHAP EXPLANATIONS
// ============================================================
async function loadExplanations() {
  if (state.shapData) {
    renderShap(state.shapData);
    return;
  }

  showLoading('Computing SHAP values...', 'This may take 30-60 seconds');

  try {
    const res = await apiCall('/api/explain', {
      method: 'POST',
      body: JSON.stringify({ session_id: state.sessionId, max_samples: 100 }),
    });

    state.shapData = await res.json();

    if (state.shapData.error) {
      showToast(state.shapData.error, 'error');
      return;
    }

    renderShap(state.shapData);
    showToast('SHAP explanations computed!', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

function renderShap(shap) {
  const section = $('#shap-section');
  section.innerHTML = '';

  // Feature importance bar chart
  const impCard = document.createElement('div');
  impCard.className = 'card full-width';
  impCard.innerHTML = `
    <div class="card-title">📊 Global Feature Importance (|SHAP|) — ${escapeHtml(shap.model_name)}</div>
    <div id="shap-importance"></div>
  `;
  section.appendChild(impCard);

  requestAnimationFrame(() => {
    const container = document.getElementById('shap-importance');
    if (!container || !shap.feature_importance) return;

    const maxImp = shap.feature_importance[0]?.importance || 1;
    shap.feature_importance.forEach(f => {
      const pct = (f.importance / maxImp) * 100;
      container.innerHTML += `
        <div class="shap-bar">
          <span class="feature-name" title="${escapeHtml(f.feature)}">${escapeHtml(truncate(f.feature, 25))}</span>
          <div class="bar-track">
            <div class="bar-fill positive" style="width:${pct}%;"></div>
          </div>
          <span class="importance-val">${f.importance.toFixed(4)}</span>
        </div>
      `;
    });
  });

  // Individual explanations
  if (shap.individual_explanations && shap.individual_explanations.length > 0) {
    const indCard = document.createElement('div');
    indCard.className = 'card full-width';
    let indHtml = `<div class="card-title">🔬 Individual Prediction Explanations</div>`;

    shap.individual_explanations.forEach((sample, idx) => {
      indHtml += `<div style="margin-bottom:24px;"><strong style="color:var(--text-accent); font-size:0.85rem;">Sample #${sample.sample_index}</strong>`;
      sample.contributions.forEach(c => {
        const isPositive = c.shap_value >= 0;
        indHtml += `
          <div class="waterfall-item ${isPositive ? 'positive' : 'negative'}">
            <span class="waterfall-feature">${escapeHtml(truncate(c.feature, 22))}</span>
            <span class="waterfall-value ${isPositive ? 'pos' : 'neg'}">${isPositive ? '+' : ''}${c.shap_value.toFixed(4)}</span>
          </div>
        `;
      });
      indHtml += '</div>';
    });

    indCard.innerHTML = indHtml;
    section.appendChild(indCard);
  }

  // SHAP chart
  if (shap.feature_importance) {
    const chartCard = document.createElement('div');
    chartCard.className = 'card full-width';
    chartCard.innerHTML = `<div class="card-title">📈 Feature Importance Chart</div><div class="chart-container" style="height:350px;"><canvas id="shap-chart"></canvas></div>`;
    section.appendChild(chartCard);

    requestAnimationFrame(() => {
      const ctx = document.getElementById('shap-chart');
      if (!ctx) return;

      if (state.charts['shap-chart']) state.charts['shap-chart'].destroy();

      const topFeatures = shap.feature_importance.slice(0, 10).reverse();
      state.charts['shap-chart'] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: topFeatures.map(f => truncate(f.feature, 20)),
          datasets: [{
            label: 'Mean |SHAP|',
            data: topFeatures.map(f => f.importance),
            backgroundColor: topFeatures.map((_, i) => {
              const ratio = i / topFeatures.length;
              return `rgba(${Math.round(102 + ratio * 50)}, ${Math.round(126 - ratio * 30)}, 234, 0.6)`;
            }),
            borderColor: 'rgba(102, 126, 234, 0.9)',
            borderWidth: 1,
            borderRadius: 4,
          }],
        },
        options: {
          ...chartDefaults('Mean |SHAP|'),
          indexAxis: 'y',
        },
      });
    });
  }
}

// ============================================================
// DOWNLOAD PREDICTIONS
// ============================================================
async function downloadPredictions() {
  showLoading('Generating predictions...', 'Creating CSV download');

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Download failed');
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${state.sessionId}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Predictions downloaded!', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
}

// ============================================================
// BUTTON HANDLERS
// ============================================================
function initButtons() {
  // Step navigation
  $('#btn-to-eda').addEventListener('click', () => {
    goToStep(2);
    loadEDA();
  });

  $('#btn-to-config').addEventListener('click', () => goToStep(3));
  $('#btn-to-explain').addEventListener('click', () => {
    goToStep(5);
    loadExplanations();
  });

  // Back buttons
  $('#btn-back-1').addEventListener('click', () => goToStep(1));
  $('#btn-back-2').addEventListener('click', () => goToStep(2));
  $('#btn-back-3').addEventListener('click', () => goToStep(3));
  $('#btn-back-4').addEventListener('click', () => goToStep(4));

  // Train
  $('#btn-train').addEventListener('click', trainModels);

  // Download
  $('#btn-download').addEventListener('click', downloadPredictions);
  $('#btn-download-2').addEventListener('click', downloadPredictions);

  // Target select enables train button
  $('#target-select').addEventListener('change', (e) => {
    $('#btn-train').disabled = !e.target.value;
    if (e.target.value) {
      const col = state.schema.columns.find(c => c.name === e.target.value);
      if (col) {
        const taskType = col.type === 'numeric' && col.unique > 10 ? 'Regression' : 'Classification';
        $('#target-info').innerHTML = `
          <strong style="color:var(--text-accent);">${taskType}</strong> task detected
          · ${col.unique} unique values · ${col.missing} missing
        `;
        // Update model names based on task type
        const logRegLabel = $$('input[name="model"][value="Logistic Regression"]');
        if (logRegLabel.length) {
          const label = logRegLabel[0].closest('.model-checkbox').querySelector('.model-name');
          label.textContent = taskType === 'Regression' ? 'Linear Regression' : 'Logistic Regression';
        }
      }
    }
  });
}

// ============================================================
// SLIDERS
// ============================================================
function initSliders() {
  $('#test-split').addEventListener('input', (e) => {
    $('#split-val').textContent = e.target.value + '%';
  });
}

// ============================================================
// CHART DEFAULTS
// ============================================================
function chartDefaults(yLabel = '') {
  return {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        ticks: { color: '#8b92a8', font: { size: 10, family: 'Inter' }, maxRotation: 45 },
        grid: { color: 'rgba(255,255,255,0.03)' },
      },
      y: {
        ticks: { color: '#8b92a8', font: { size: 10, family: 'Inter' } },
        grid: { color: 'rgba(255,255,255,0.03)' },
        title: yLabel ? { display: true, text: yLabel, color: '#555d75', font: { size: 11 } } : {},
      },
    },
    plugins: {
      legend: {
        labels: { color: '#8b92a8', font: { size: 11, family: 'Inter' }, padding: 15 },
      },
      tooltip: {
        backgroundColor: 'rgba(12, 14, 26, 0.95)',
        titleColor: '#e8ecf4',
        bodyColor: '#8b92a8',
        borderColor: 'rgba(99, 102, 241, 0.2)',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
      },
    },
  };
}

// ============================================================
// UTILITIES
// ============================================================
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, len) {
  return str.length > len ? str.substring(0, len - 1) + '…' : str;
}

function generateColors(n) {
  const baseColors = [
    'rgba(102, 126, 234, 1)',
    'rgba(0, 212, 255, 1)',
    'rgba(118, 75, 162, 1)',
    'rgba(16, 185, 129, 1)',
    'rgba(245, 158, 11, 1)',
    'rgba(239, 68, 68, 1)',
    'rgba(59, 130, 246, 1)',
    'rgba(168, 85, 247, 1)',
    'rgba(236, 72, 153, 1)',
    'rgba(20, 184, 166, 1)',
  ];
  const colors = [];
  for (let i = 0; i < n; i++) {
    colors.push(baseColors[i % baseColors.length]);
  }
  return colors;
}
