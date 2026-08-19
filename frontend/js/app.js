/**
 * RetainAI - Customer Churn Prediction & Explainable AI Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initFormControls();
  initPresets();
  initFormSubmission();
  initDashboardCharts();
  fetchInitialStats();
});

/* ==========================================================================
   Navigation & Tabs
   ========================================================================== */
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const pageTitle = document.getElementById('page-title');
  const pageSubtitle = document.getElementById('page-subtitle');
  const quickPredictBtn = document.getElementById('btn-quick-predict');
  const menuToggle = document.getElementById('menu-toggle');
  const sidebar = document.querySelector('.sidebar');

  const titles = {
    dashboard: {
      title: 'Executive Churn Dashboard',
      subtitle: 'Real-time telecommunication subscriber attrition analytics & predictive insights'
    },
    prediction: {
      title: 'AI Churn Predictor & Explainability',
      subtitle: 'Simulate customer attributes to generate risk scoring and SHAP feature attributions'
    },
    models: {
      title: 'ML Model Performance & Evaluation Matrix',
      subtitle: 'Hyperparameter-tuned model benchmarks, cross-validation metrics, and SMOTE balancing'
    }
  };

  function switchTab(tabId) {
    navItems.forEach(item => {
      item.classList.toggle('active', item.getAttribute('data-tab') === tabId);
    });

    tabPanes.forEach(pane => {
      pane.classList.toggle('active', pane.id === `tab-${tabId}`);
    });

    if (titles[tabId]) {
      pageTitle.textContent = titles[tabId].title;
      pageSubtitle.textContent = titles[tabId].subtitle;
    }

    if (sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  }

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabId = item.getAttribute('data-tab');
      switchTab(tabId);
    });
  });

  if (quickPredictBtn) {
    quickPredictBtn.addEventListener('click', () => {
      switchTab('prediction');
    });
  }

  if (menuToggle) {
    menuToggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
}

/* ==========================================================================
   Form Controls & Real-Time Sync
   ========================================================================== */
function initFormControls() {
  const tenureSlider = document.getElementById('tenure');
  const tenureNum = document.getElementById('tenure-num');
  const tenureVal = document.getElementById('tenure-val');
  const monthlyInput = document.getElementById('MonthlyCharges');
  const totalInput = document.getElementById('TotalCharges');

  function updateTenure(val) {
    const tenure = Math.max(0, Math.min(72, parseInt(val, 10) || 0));
    tenureSlider.value = tenure;
    tenureNum.value = tenure;
    tenureVal.textContent = `${tenure} mo`;
    
    // Auto calculate suggested Total Charges
    const monthly = parseFloat(monthlyInput.value) || 0;
    if (totalInput && !totalInput.dataset.manualEdit) {
      totalInput.value = (monthly * tenure).toFixed(2);
    }
  }

  tenureSlider.addEventListener('input', (e) => updateTenure(e.target.value));
  tenureNum.addEventListener('input', (e) => updateTenure(e.target.value));

  monthlyInput.addEventListener('input', () => {
    const monthly = parseFloat(monthlyInput.value) || 0;
    const tenure = parseInt(tenureSlider.value, 10) || 0;
    if (totalInput && !totalInput.dataset.manualEdit) {
      totalInput.value = (monthly * tenure).toFixed(2);
    }
  });

  totalInput.addEventListener('input', () => {
    totalInput.dataset.manualEdit = "true";
  });
}

/* ==========================================================================
   Preset Customer Profiles
   ========================================================================== */
const PROFILES = {
  high_risk: {
    gender: "Female",
    SeniorCitizen: "0",
    Partner: "No",
    Dependents: "No",
    tenure: 2,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 95.50,
    TotalCharges: 191.00
  },
  loyal_customer: {
    gender: "Male",
    SeniorCitizen: "0",
    Partner: "Yes",
    Dependents: "Yes",
    tenure: 68,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "Yes",
    StreamingMovies: "No",
    Contract: "Two year",
    PaperlessBilling: "No",
    PaymentMethod: "Credit card (automatic)",
    MonthlyCharges: 64.20,
    TotalCharges: 4365.60
  },
  moderate_risk: {
    gender: "Male",
    SeniorCitizen: "1",
    Partner: "No",
    Dependents: "No",
    tenure: 14,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "Yes",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Bank transfer (automatic)",
    MonthlyCharges: 79.85,
    TotalCharges: 1117.90
  }
};

function populateForm(profile) {
  const form = document.getElementById('churn-form');
  const totalInput = document.getElementById('TotalCharges');
  if (totalInput) delete totalInput.dataset.manualEdit;

  for (const [key, value] of Object.entries(profile)) {
    const el = form.elements[key];
    if (el) {
      el.value = value;
      if (key === 'tenure') {
        document.getElementById('tenure-num').value = value;
        document.getElementById('tenure-val').textContent = `${value} mo`;
      }
    }
  }
}

function initPresets() {
  document.getElementById('btn-preset-high').addEventListener('click', () => {
    populateForm(PROFILES.high_risk);
    showToast('Loaded High Risk Customer Profile', 'info');
  });

  document.getElementById('btn-preset-loyal').addEventListener('click', () => {
    populateForm(PROFILES.loyal_customer);
    showToast('Loaded Loyal Long-Term Customer Profile', 'success');
  });

  document.getElementById('btn-preset-moderate').addEventListener('click', () => {
    populateForm(PROFILES.moderate_risk);
    showToast('Loaded Moderate Risk Customer Profile', 'info');
  });

  document.getElementById('btn-form-reset').addEventListener('click', () => {
    document.getElementById('churn-form').reset();
    document.getElementById('tenure-val').textContent = '12 mo';
    document.getElementById('tenure-num').value = 12;
    showToast('Reset form fields to default values', 'info');
  });
}

/* ==========================================================================
   Prediction Submission & Result Rendering
   ========================================================================== */
function initFormSubmission() {
  const form = document.getElementById('churn-form');
  const submitBtn = document.getElementById('btn-predict-submit');
  const placeholder = document.getElementById('result-placeholder');
  const loading = document.getElementById('result-loading');
  const resultCard = document.getElementById('result-card');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Extract form data
    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      if (['SeniorCitizen'].includes(key)) {
        payload[key] = parseInt(value, 10);
      } else if (['tenure', 'MonthlyCharges', 'TotalCharges'].includes(key)) {
        payload[key] = parseFloat(value) || 0;
      } else {
        payload[key] = value;
      }
    }

    // UI Loading state
    placeholder.classList.add('hidden');
    resultCard.classList.add('hidden');
    loading.classList.remove('hidden');
    submitBtn.disabled = true;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Prediction failed.');
      }

      renderPredictionResult(data);
      showToast(`Prediction generated: ${data.risk_level.toUpperCase()} RISK`, data.risk_level === 'High' ? 'error' : 'success');

    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message}`, 'error');
      placeholder.classList.remove('hidden');
    } finally {
      loading.classList.add('hidden');
      submitBtn.disabled = false;
    }
  });
}

function renderPredictionResult(data) {
  const resultCard = document.getElementById('result-card');
  const riskBadge = document.getElementById('res-risk-badge');
  const modelTag = document.getElementById('res-model-tag');
  const probaValue = document.getElementById('res-proba-value');
  const outcomeText = document.getElementById('res-outcome-text');
  const outcomeDesc = document.getElementById('res-outcome-desc');
  const progressCircle = document.getElementById('progress-ring-circle');
  const riskFill = document.getElementById('res-risk-fill');
  const factorsList = document.getElementById('res-factors-list');
  const recsList = document.getElementById('res-recs-list');

  const proba = data.probability;
  const probaPct = Math.round(proba * 100);
  const risk = data.risk_level;

  // Set Model Tag
  modelTag.textContent = `Model: ${data.model_used || 'XGBoost'}`;

  // Set Risk Badge & Color Scheme
  riskBadge.className = `risk-badge ${risk.toLowerCase()}`;
  riskBadge.textContent = `${risk.toUpperCase()} RISK`;

  // Outcome text
  if (risk === 'High') {
    outcomeText.textContent = 'LIKELY TO CHURN';
    outcomeText.style.color = 'var(--danger-light)';
    outcomeDesc.textContent = 'This customer exhibits immediate flight risk indicators and requires priority retention outreach.';
    riskFill.style.background = 'var(--danger)';
    progressCircle.style.stroke = 'var(--danger)';
  } else if (risk === 'Medium') {
    outcomeText.textContent = 'MODERATE CHURN RISK';
    outcomeText.style.color = 'var(--warning-light)';
    outcomeDesc.textContent = 'Customer demonstrates borderline retention signals. Targeted engagement is recommended.';
    riskFill.style.background = 'var(--warning)';
    progressCircle.style.stroke = 'var(--warning)';
  } else {
    outcomeText.textContent = 'LOYAL SUBSCRIBER';
    outcomeText.style.color = 'var(--success-light)';
    outcomeDesc.textContent = 'Customer shows strong loyalty markers with minimal attrition propensity.';
    riskFill.style.background = 'var(--success)';
    progressCircle.style.stroke = 'var(--success)';
  }

  // Animate Gauge Ring (Circumference = 2 * PI * 68 = 427.26)
  const circumference = 2 * Math.PI * 68;
  const offset = circumference - (probaPct / 100) * circumference;
  progressCircle.style.strokeDashoffset = offset;
  probaValue.textContent = `${probaPct}%`;
  riskFill.style.width = `${probaPct}%`;

  // Render SHAP Factors
  factorsList.innerHTML = '';
  const factorDetails = data.factor_details || [];

  if (factorDetails.length > 0) {
    factorDetails.forEach(item => {
      const factorDiv = document.createElement('div');
      factorDiv.className = `factor-item ${item.is_risk_driver ? 'risk-driver' : 'risk-reducer'}`;
      factorDiv.innerHTML = `
        <span class="factor-name">${item.factor}</span>
        <span class="factor-tag ${item.is_risk_driver ? 'risk' : 'safe'}">
          ${item.is_risk_driver ? '+' : '-'}${Math.abs(Math.round(item.weight * 100))}% Impact
        </span>
      `;
      factorsList.appendChild(factorDiv);
    });
  } else {
    (data.top_factors || []).forEach(factor => {
      const factorDiv = document.createElement('div');
      factorDiv.className = 'factor-item risk-driver';
      factorDiv.innerHTML = `
        <span class="factor-name">${factor}</span>
        <span class="factor-tag risk">Primary Driver</span>
      `;
      factorsList.appendChild(factorDiv);
    });
  }

  // Render Retention Recommendations
  recsList.innerHTML = '';
  const recs = data.recommendations || [];
  if (recs.length > 0) {
    recs.forEach(rec => {
      const recDiv = document.createElement('div');
      recDiv.className = 'rec-item';
      recDiv.innerHTML = `
        <div class="rec-title"><i class="fa-solid fa-arrow-right"></i> ${rec.action}</div>
        <div class="rec-desc">${rec.detail}</div>
      `;
      recsList.appendChild(recDiv);
    });
  }

  resultCard.classList.remove('hidden');
}

/* ==========================================================================
   Dashboard Charts & Visualizations (Chart.js)
   ========================================================================== */
function initDashboardCharts() {
  // 1. Churn Ratio Doughnut Chart
  const ctxRatio = document.getElementById('chartChurnRatio');
  if (ctxRatio) {
    new Chart(ctxRatio, {
      type: 'doughnut',
      data: {
        labels: ['Retained Customers', 'Churned Customers'],
        datasets: [{
          data: [5174, 1869],
          backgroundColor: ['#10b981', '#ef4444'],
          borderColor: '#111827',
          borderWidth: 3,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '72%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
          }
        }
      }
    });
  }

  // 2. Contract Risk Bar Chart
  const ctxContract = document.getElementById('chartContractRisk');
  if (ctxContract) {
    new Chart(ctxContract, {
      type: 'bar',
      data: {
        labels: ['Month-to-Month', 'One Year', 'Two Year'],
        datasets: [
          {
            label: 'Churn Rate (%)',
            data: [42.7, 11.3, 2.8],
            backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 50,
            ticks: { color: '#94a3b8', callback: val => `${val}%` },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          x: {
            ticks: { color: '#94a3b8' },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // 3. Tenure vs. Churn Hazard Line Chart
  const ctxTenure = document.getElementById('chartTenureChurn');
  if (ctxTenure) {
    new Chart(ctxTenure, {
      type: 'line',
      data: {
        labels: ['0-6m', '7-12m', '13-24m', '25-36m', '37-48m', '49-60m', '61-72m'],
        datasets: [{
          label: 'Churn Hazard Rate (%)',
          data: [52.1, 41.3, 29.5, 21.0, 15.2, 9.8, 4.3],
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          fill: true,
          tension: 0.35,
          pointBackgroundColor: '#818cf8',
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: { color: '#94a3b8', callback: val => `${val}%` },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          x: {
            ticks: { color: '#94a3b8' },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // 4. Global SHAP Feature Importance Drivers Horizontal Bar
  const ctxDrivers = document.getElementById('chartFeatureDrivers');
  if (ctxDrivers) {
    new Chart(ctxDrivers, {
      type: 'bar',
      data: {
        labels: [
          'Month-to-Month Contract',
          'Tenure Duration',
          'Fiber Optic Internet',
          'Total/Monthly Charges',
          'No Tech Support',
          'Electronic Check Payment'
        ],
        datasets: [{
          label: 'SHAP Feature Importance (|Mean SHAP Value|)',
          data: [0.88, 0.74, 0.58, 0.49, 0.41, 0.32],
          backgroundColor: [
            '#ef4444',
            '#6366f1',
            '#f59e0b',
            '#06b6d4',
            '#8b5cf6',
            '#3b82f6'
          ],
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            beginAtZero: true,
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y: {
            ticks: { color: '#94a3b8', font: { size: 11 } },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }
}

/* ==========================================================================
   Backend Data Fetching
   ========================================================================== */
async function fetchInitialStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    const data = await res.json();

    if (data.dataset_stats) {
      const stats = data.dataset_stats;
      document.getElementById('kpi-total-customers').textContent = stats.total_customers.toLocaleString();
      document.getElementById('kpi-churned-customers').textContent = stats.churned_customers.toLocaleString();
      document.getElementById('kpi-churn-rate').textContent = `${stats.churn_rate}%`;
      document.getElementById('kpi-retained-customers').textContent = stats.retained_customers.toLocaleString();
    }

    if (data.best_model) {
      document.getElementById('sidebar-model-name').textContent = data.best_model;
    }
  } catch (e) {
    console.warn('Initial stats fetch notice:', e);
  }
}

/* ==========================================================================
   Toast Notification System
   ========================================================================== */
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const iconClass = type === 'success' ? 'fa-circle-check' : (type === 'error' ? 'fa-triangle-exclamation' : 'fa-circle-info');
  toast.innerHTML = `
    <i class="fa-solid ${iconClass}"></i>
    <span>${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
