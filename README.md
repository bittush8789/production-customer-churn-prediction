# RetainAI: Production Customer Churn Prediction & Explainable AI Web Application

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.0](https://img.shields.io/badge/backend-Flask%203.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20LogReg-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application that predicts telecommunication customer churn probability, classifies subscribers into risk tiers (Low, Medium, High), provides individual customer SHAP explainability factors, and delivers prescriptive retention actions through a SaaS analytics dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Dataset Overview](#5-dataset-overview)
6. [Feature Engineering](#6-feature-engineering)
7. [ML Pipeline](#7-ml-pipeline)
8. [Model Comparison & Evaluation](#8-model-comparison--evaluation)
9. [Evaluation Metrics & Selection Strategy](#9-evaluation-metrics--selection-strategy)
10. [SHAP Explainability Layer](#10-shap-explainability-layer)
11. [Flask REST API Reference](#11-flask-rest-api-reference)
12. [Frontend Architecture & Visualizations](#12-frontend-architecture--visualizations)
13. [Project Directory Structure](#13-project-directory-structure)
14. [Installation Guide](#14-installation-guide)
15. [Model Training Instructions](#15-model-training-instructions)
16. [Running the Application](#16-running-the-application)
17. [Example Prediction Walkthrough](#17-example-prediction-walkthrough)
18. [Testing & Quality Assurance](#18-testing--quality-assurance)
19. [Future Improvements](#19-future-improvements)

---

## 1. Project Overview

RetainAI is an enterprise-grade Machine Learning solution designed to identify telecom subscribers who are likely to cancel their subscriptions. The application pairs predictive modeling (XGBoost, Random Forest, Logistic Regression) with explainable AI (SHAP) and a responsive frontend interface.

---

## 2. Business Problem

Customer acquisition costs (CAC) in the telecom industry are typically 5 to 7 times higher than customer retention costs. Preventing subscriber attrition requires:
* Early detection of high-risk accounts.
* Clear visibility into the exact drivers causing churn (e.g., month-to-month contracts, lack of technical support, fiber optic pricing pressure).
* Automated retention recommendations tailored to individual customer risk profiles.

---

## 3. System Architecture

```text
┌────────────────────────────────────────────────────────┐
│             HTML5 / CSS3 / Vanilla JS                  │
│       (Overview Dashboard, Predictor, Benchmarks)      │
└───────────────────────────┬────────────────────────────┘
                            │ JSON over HTTP (fetch)
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Flask REST API (app.py)                │
│    • Input Validation   • CORS   • Error Handlers      │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│           Inference Engine (predictor.py)              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Preprocessing Pipeline (preprocessing.pkl)    │  │
│  │    • Feature Engineering                         │  │
│  │    • Median Imputation & StandardScaler          │  │
│  │    • OneHotEncoder (Categoricals)                │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Production ML Classifier (churn_model.pkl)    │  │
│  │    • Calibrated Churn Probability                │  │
│  │    • Risk Tiering (Low < 35%, Med, High >= 65%)  │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 3. SHAP Explainability Engine                    │  │
│  │    • TreeExplainer Feature Attributions          │  │
│  │    • Dynamic Positive & Negative Risk Drivers    │  │
│  │    • Prescriptive Retention Recommendations      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack & Latest Verified Versions

| Category | Technology | Latest Tested Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.13.3` / `3.11+` | Core execution environment |
| **Backend Framework** | Flask | `3.1.1` | REST API routing and static asset delivery |
| **CORS Middleware** | Flask-CORS | `6.0.2` | Cross-origin resource sharing |
| **Machine Learning** | Scikit-Learn | `1.6.1` | Preprocessing pipelines, metrics, CV |
| **Gradient Boosting** | XGBoost | `3.2.0` | Production churn classification |
| **Class Imbalance** | Imbalanced-Learn | `0.14.2` | SMOTE synthetic oversampling |
| **Explainable AI** | SHAP | `0.51.0` | TreeExplainer & LinearExplainer attributions |
| **Data Manipulation** | Pandas | `2.2.3` | DataFrame operations and schema validation |
| **Numerical Computing** | NumPy | `2.4.4` | Vectorized math operations |
| **Serialization** | Joblib | `1.4.2` | Model & pipeline artifact persistence |
| **Visualization** | Chart.js | `4.4.1` | Dashboard analytics graphs |
| **Styling & Structure** | HTML5 / CSS3 | ES6+ / CSS3 Tokens | Responsive glassmorphism interface |
| **Testing** | Pytest | `8.0.0` | Automated unit & integration testing |

---

## 5. Dataset Overview

Based on the standard IBM/Telco Customer Churn dataset containing 7,043 customer records across 21 features:

| Feature Category | Features | Description |
| :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Basic customer background |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | Subscription terms and payment mode |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Subscribed product bundle |
| **Financials** | `MonthlyCharges`, `TotalCharges` | Billing amounts |
| **Target** | `Churn` | `Yes` (26.5%) / `No` (73.5%) |

---

## 6. Feature Engineering

The `FeatureEngineer` transformer (`ml/src/feature_engineering.py`) derives domain-specific signals:

1. `tenure_group`: Segments tenure into buckets (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `>60m`).
   * *Rationale*: Early lifecycle months exhibit significantly higher hazard rates.
2. `num_services`: Sum of active services (Phone, Internet, Security, Backup, Streaming).
   * *Rationale*: Greater product adoption increases customer switching costs.
3. `charge_per_service`: `MonthlyCharges / (num_services + 1)`.
   * *Rationale*: Measures perceived cost burden relative to value received.
4. `has_streaming`: Flag for `StreamingTV` or `StreamingMovies`.
5. `has_security`: Flag for `OnlineSecurity`, `TechSupport`, `DeviceProtection`, or `OnlineBackup`.
   * *Rationale*: Security-bundled accounts have lower churn rates.
6. `contract_risk`: Risk ranking of contract types (`Month-to-month` = High, `One year` = Medium, `Two year` = Low).
7. `charges_to_tenure_ratio`: `MonthlyCharges / (tenure + 1)`.

---

## 7. ML Pipeline

1. **Data Cleaning**: String conversion and missing value resolution for `TotalCharges`.
2. **ColumnTransformer Pipeline**:
   * Numerical features $\to$ `SimpleImputer(strategy='median')` $\to$ `StandardScaler()`.
   * Categorical features $\to$ `SimpleImputer(strategy='most_frequent')` $\to$ `OneHotEncoder(handle_unknown='ignore')`.
3. **Class Imbalance Handling**: Applied **SMOTE** on the training set to address the 73.5% / 26.5% raw imbalance.
4. **Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation using GridSearchCV across:
   * Logistic Regression ($C$, solver)
   * Random Forest (n_estimators, max_depth, min_samples_split)
   * XGBoost (learning_rate, max_depth, n_estimators, scale_pos_weight, subsample)

---

## 8. Model Comparison & Evaluation

Held-out test set evaluation results ($N = 1,409$):

| Model | Accuracy | Precision | Recall (Churn) | F1-Score | ROC-AUC | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Selected)** | **74.80%** | **51.61%** | **81.28%** | **63.14%** | **84.53%** | **0.7714** |
| Random Forest Classifier | 74.52% | 51.29% | 79.68% | 62.41% | 84.25% | 0.7633 |
| Logistic Regression | 73.81% | 50.44% | 77.01% | 60.95% | 84.58% | 0.7522 |

*Composite Score formula: $0.40 \times \text{ROC-AUC} + 0.30 \times \text{Recall} + 0.30 \times \text{F1-Score}$.*

---

## 9. Evaluation Metrics & Selection Strategy

In customer churn prediction, **false negatives** (failing to identify a churning customer) are significantly more expensive than false positives (offering a retention incentive to an already loyal customer). 

Therefore, our selection criteria prioritized:
* **Recall (81.28%)**: Capturing >81% of all churning subscribers.
* **ROC-AUC (84.53%)**: High discriminative separation across probability thresholds.
* **F1-Score (63.14%)**: Balance between precision and recall on imbalanced targets.

---

## 10. SHAP Explainability Layer

Using **TreeExplainer**, every prediction generates localized Shapley values indicating how each attribute pushed the churn probability up or down:

```text
Prediction: HIGH CHURN RISK (87% Churn Probability)

Top Influencing Factors (SHAP):
1. Month-to-month contract (High churn volatility)       [+28% Risk Impact]
2. Low customer tenure (2 months)                        [+22% Risk Impact]
3. Fiber optic broadband subscription                    [+16% Risk Impact]
4. Absence of Tech Support assistance                    [+12% Risk Impact]
5. Electronic Check payment method                       [+8%  Risk Impact]
```

---

## 11. Flask REST API Reference

### 1. Health Check
* **`GET /health`** or **`GET /api/health`**
* **Response**:
  ```json
  {
    "service": "Customer Churn Prediction API",
    "status": "healthy",
    "version": "1.0.0"
  }
  ```

### 2. Predict Customer Churn
* **`POST /predict`** or **`POST /api/predict`**
* **Headers**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 191.0
  }
  ```
* **Success Response (200 OK)**:
  ```json
  {
    "prediction": "Churn",
    "probability": 0.8732,
    "probability_pct": 87.3,
    "risk_level": "High",
    "top_factors": [
      "Month-to-month contract (High churn volatility)",
      "Low customer tenure (2 months - onboarding vulnerability)",
      "Fiber optic broadband subscription (Higher churn sensitivity)"
    ],
    "factor_details": [
      {
        "factor": "Month-to-month contract (High churn volatility)",
        "impact": "Increases Risk",
        "is_risk_driver": true,
        "weight": 0.28
      }
    ],
    "recommendations": [
      {
        "action": "Offer Annual Contract Incentive",
        "detail": "Provide a 15% discount on a 1-year or 2-year commitment to eliminate month-to-month churn risk.",
        "priority": "High"
      }
    ],
    "model_used": "XGBoost"
  }
  ```

### 3. Client Code Examples

#### Python (Requests)
```python
import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 191.0
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Prediction: {result['prediction']} ({result['risk_level']} Risk)")
print(f"Probability: {result['probability_pct']}%")
print("Top Contributing Factors:")
for factor in result['top_factors']:
    print(f"  • {factor}")
```

#### cURL (Terminal / Bash)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 191.0
  }'
```

#### JavaScript (Fetch API)
```javascript
const customerData = {
  gender: "Female",
  SeniorCitizen: 0,
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
  MonthlyCharges: 95.5,
  TotalCharges: 191.0
};

fetch('http://127.0.0.1:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(customerData)
})
  .then(res => res.json())
  .then(data => {
    console.log(`Risk Level: ${data.risk_level} (${data.probability_pct}%)`);
    console.log('Top Factors:', data.top_factors);
    console.log('Retention Recommendations:', data.recommendations);
  })
  .catch(err => console.error('Error:', err));
```

---

## 12. Frontend Architecture & Visualizations

* **KPI Cards**: Real-time display of Total Customers (7,043), Churned Customers (1,869), Churn Rate (26.5%), and Retained Customer Base (5,174).
* **Interactive Charts**:
  1. *Churn Distribution*: Doughnut chart of retained vs churned users.
  2. *Contract Risk Analysis*: Bar chart comparing month-to-month vs 1-year/2-year rates.
  3. *Tenure vs Hazard Curve*: Line chart mapping churn risk across subscriber lifecycle.
  4. *Global Feature Importance*: Horizontal bar chart ranking primary drivers.
* **Instant Profile Presets**: 1-click loading for *High Risk*, *Loyal Customer*, *Moderate Risk*, and *Reset*.
* **Animated Probability Gauge**: SVG circular progress meter displaying exact probability percentage.
* **Dynamic Risk Badges**: `HIGH RISK` (Red), `MEDIUM RISK` (Amber), `LOW RISK` (Emerald).

---

## 13. Project Directory Structure

```text
customer-churn-prediction/
│
├── frontend/
│   ├── index.html          # Responsive single-page application UI
│   ├── css/
│   │   └── style.css       # Design tokens, glassmorphism, animations
│   └── js/
│       └── app.js          # API client, Chart.js graphs, form controls
│
├── backend/
│   ├── app.py              # Flask server entrypoint & static routing
│   ├── routes.py           # REST endpoints, validation & sample profiles
│   └── predictor.py        # ML inference, SHAP explainer & recommendations
│
├── ml/
│   ├── data/
│   │   └── customer_churn.csv      # Standard Telco dataset
│   ├── notebooks/
│   │   ├── 01_eda.ipynb            # Exploratory data analysis
│   │   ├── 02_preprocessing.ipynb  # Pipeline & feature engineering
│   │   ├── 03_training.ipynb       # Model training & CV
│   │   └── 04_evaluation.ipynb     # ROC curves & SHAP analysis
│   ├── src/
│   │   ├── __init__.py
│   │   ├── load_data.py            # Dataset loader / synthetic generator
│   │   ├── feature_engineering.py  # FeatureEngineer custom transformer
│   │   ├── preprocessing.py        # ColumnTransformer pipeline
│   │   ├── train.py                # Model training, SMOTE & selection
│   │   └── evaluate.py             # Evaluation reports & metrics
│   └── models/
│       ├── churn_model.pkl         # Production XGBoost classifier
│       ├── preprocessing.pkl       # Fitted preprocessing pipeline
│       └── model_meta.json         # Benchmark metrics & feature names
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py       # Pipeline & feature transformation tests
│   ├── test_model.py               # Model loading, bounds & risk tests
│   └── test_api.py                 # REST API & payload validation tests
│
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Non-root secure Docker container config
├── docker-compose.yml              # Multi-container orchestration config
├── .dockerignore                   # Docker build ignore rules
├── README.md                       # Complete documentation
└── .gitignore                      # Git ignore rules
```

---

## 14. Installation Guide

### Prerequisites
* Python 3.11 or higher
* pip package manager

### Steps
1. Clone or navigate to the project directory:
   ```bash
   cd customer-churn-prediction
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 15. Model Training Instructions

To train and benchmark all models from scratch:

```bash
python ml/src/train.py
```

**Expected Output**:
```text
--- [1/6] Loading & Cleaning Dataset ---
--- [2/6] Stratified Train/Test Split ---
--- [3/6] Preprocessing Pipeline Fitting ---
--- [4/6] Handling Class Imbalance with SMOTE ---
--- [5/6] Training & Tuning Models ---
--- [6/6] Model Selection & Export ---
WINNER: XGBoost with Composite Score = 0.7714
Artifacts successfully saved to ml/models/
```

---

## 16. Running the Application

### Option A: Local Python Environment
Start the Flask application server:
```bash
python backend/app.py
```

### Option B: Docker (Non-Root User)
Build and run using the secure non-root Dockerfile:
```bash
# Build the Docker image
docker build -t retainai-churn-prediction .

# Run the container
docker run -d --name retainai_app -p 5000:5000 retainai-churn-prediction
```

### Option C: Docker Compose
Start with a single command:
```bash
# Start container in detached mode
docker-compose up -d --build

# View container logs
docker-compose logs -f

# Stop and remove containers
docker-compose down
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 17. Example Prediction Walkthrough

1. Open the **Churn Predictor** tab in the web interface.
2. Click **High Risk Customer** from the preset profile toolbar.
3. Observe fields populating with: Month-to-month contract, Fiber optic internet, 2 months tenure, no tech support, $95.50/mo.
4. Click **PREDICT CHURN PROBABILITY**.
5. View the animated gauge displaying **87% Churn Probability**, glowing **HIGH RISK** badge, top SHAP drivers, and automated retention recommendations.

---

## 18. Testing & Quality Assurance

Run the automated Pytest test suite:

```bash
pytest -v
```

Tests include:
* Preprocessing pipeline transformations & missing value handling
* Feature engineering calculations
* Model loading & probability range $[0, 1]$ validation
* Risk category mapping
* Flask REST API status codes (`200 OK`, `400 Bad Request`)

---

## 19. Future Improvements

* **Multi-Language Support**: Expand UI localization.
* **Customer Segment Clustering**: Unsupervised K-Means clustering overlay for cohort identification.
* **Automated Email Dispatch**: Webhook integrations to trigger customer retention campaigns directly from the dashboard.
