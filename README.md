# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-brightgreen?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning system that predicts whether a bank customer will churn,
built with a production-ready FastAPI deployment, Dockerized API, and business-optimized
threshold tuning.

---

## Demo


[![Bank Customer Churn Prediction Demo](https://img.youtube.com/vi/JZmN-7Ldkm0/hqdefault.jpg)](https://www.youtube.com/watch?v=JZmN-7Ldkm0)

> Click the thumbnail above to watch the demo.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Results](#model-results)
- [Threshold Tuning](#threshold-tuning)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Docker Deployment](#docker-deployment)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Technologies Used](#technologies-used)

---

## Overview

Customer churn is one of the biggest challenges banks face — losing a customer costs
5–7x more than retaining one. This project builds an end-to-end ML pipeline that:

- Predicts whether a customer will churn (leave the bank)
- Compares Logistic Regression, Random Forest, and XGBoost across different ML families
- Applies **threshold tuning** to optimize for business impact (maximizing churn recall)
- Deploys the model as a **production-ready FastAPI** with Pydantic validation and a 3-tier
  risk categorization system
- Ships as a **Dockerized API** deployable anywhere

**Target variable:** `0` → Customer stays · `1` → Customer churns

---

## Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Bank Customer Churn Modelling](https://www.kaggle.com/datasets/shubh0799/churn-modelling) |
| Total rows | 10,000 customers |
| Features | 10 input features (demographic + account) |
| Target | `Exited` — binary churn label |
| Class distribution | ~80% stay · ~20% churn (imbalanced) |

### Features

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | int | Customer credit score (300–900) |
| `Geography` | categorical | France / Spain / Germany |
| `Gender` | categorical | Male / Female |
| `Age` | int | Customer age (18–100) |
| `Tenure` | int | Years with the bank (0–10) |
| `Balance` | float | Account balance |
| `NumOfProducts` | int | Number of bank products held (1–4) |
| `HasCrCard` | binary | Has credit card (yes/no) |
| `IsActiveMember` | binary | Is an active member (yes/no) |
| `EstimatedSalary` | float | Estimated annual salary |

> ⚠️ Note: The dataset is class-imbalanced (~20% churners). Accuracy alone is a misleading
> metric — Precision, Recall, and F1-score are used instead.

---

## ML Pipeline

The full pipeline is packaged as a single `churn_pipeline.pkl` using sklearn's `Pipeline`:

```
Raw Input (10 features)
        │
        ▼
Preprocessing
  ├── StandardScaler       → CreditScore, Age, Balance, EstimatedSalary, Tenure, Salary
  └── OneHotEncoder        → Geography, Gender
        │
        ▼
Feature Engineering
  ├── AgeGroup             → bins Age into Young / Adult / MidAge / Senior
  ├── ZeroBalance          → binary flag for Balance == 0
  └── EngagementScore      → NumOfProducts + IsActiveMember
        │
        ▼
Model Training
  ├── Logistic Regression  → baseline linear model
  ├── Random Forest        → bagging ensemble
  └── XGBoost              → boosting ensemble  ← best model
        │
        ▼
Threshold Tuning @ 0.35   → churn-optimized predictions
        │
        ▼
Output: prediction + churn_probability + risk_category
```

**Training strategy:** Three models from different ML families are compared —
Logistic Regression as the interpretable baseline, Random Forest for bagging,
and XGBoost for sequential boosting. XGBoost with threshold tuning achieves the
best churn detection performance.

---

## Feature Engineering

Three engineered features were added on top of the raw inputs:

| Feature | Logic | Why |
|---|---|---|
| `AgeGroup` | Bins Age → Young / Adult / MidAge / Senior | Churn behavior differs significantly across life stages |
| `ZeroBalance` | `1` if Balance == 0, else `0` | Zero-balance accounts are a strong disengagement signal |
| `EngagementScore` | `NumOfProducts + IsActiveMember` | Combines two loyalty signals into a single engagement proxy |

---


## Model Results

### Overall comparison (default threshold = 0.50)

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Logistic Regression | 73.6% | 40.1% | 69.5% | 51.0% |
| Random Forest | 84.7% | 59.9% | 66.2% | 62.9% |
| **XGBoost** | **84.9%** | **59.7%** | **70.2%** | **64.5%** |

### Full per-class breakdown

#### Logistic Regression (TN=1199, FP=408, FN=120, TP=273)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — stays | 90.9% | 74.6% | 82.0% | 1607 |
| 1 — churns | 40.1% | 69.5% | 51.0% | 393 |
| **Accuracy** | — | — | **73.6%** | **2000** |

#### Random Forest (TN=1433, FP=174, FN=133, TP=260)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — stays | 91.5% | 89.2% | 90.3% | 1607 |
| 1 — churns | 59.9% | 66.2% | 62.9% | 393 |
| **Accuracy** | — | — | **84.7%** | **2000** |

#### XGBoost @ default 0.50 (TN=1421, FP=186, FN=117, TP=276)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — stays | 92.4% | 88.4% | 90.4% | 1607 |
| 1 — churns | 59.7% | 70.2% | 64.5% | 393 |
| **Accuracy** | — | — | **84.9%** | **2000** |

> **Context:** Accuracy alone is misleading for imbalanced data (~20% churners).
> Logistic Regression hits 73.6% accuracy but recall on churners (69.5%) comes at the
> cost of terrible precision (40.1%) — it flags almost half the non-churners as churn.
> XGBoost achieves the best balance — 84.9% accuracy with 70.2% churn recall before
> threshold tuning, which climbs to **80%** after tuning @ 0.35.

## Threshold Tuning

By default, models classify at probability ≥ 0.5. In banking, **missing a churner is more
expensive** than falsely flagging a loyal customer. Lowering the threshold catches more
churners at the cost of some false alarms — a deliberate business trade-off.

```python
y_prob = best_model_xgb.predict_proba(x_test_scaled)[:, 1]
y_pred = (y_prob > 0.35).astype(int)
```

### Impact of threshold tuning (XGBoost @ 0.35 vs default 0.50)

| Metric | Default (0.50) | Tuned (0.35) | Improvement |
|---|---|---|---|
| Recall (churners) | 66% | **80%** | +14 pp |
| Missed churners (FN) | 129 | **78** | −51 customers saved |
| True detections (TP) | 264 | **315** | +51 more caught |

The threshold of 0.35 was chosen by analyzing the Precision-Recall curve and minimizing
total business cost: `(FN_cost × FN_count) + (FP_cost × FP_count)`.

---

## API Reference

The FastAPI app (`app.py`) exposes two endpoints:

### `GET /`
Health check — confirms the API is running.

```json
{ "message": "Bank Customer Churn Prediction API is running 🚀" }
```

### `POST /predict`

Predict churn for a single customer.

**Request body:**

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 60000.0,
  "NumOfProducts": 2,
  "HasCrCard": "yes",
  "IsActiveMember": "yes",
  "EstimatedSalary": 70000.0
}
```

**Response:**

```json
{
  "prediction": "Customer will not churn",
  "churn_probability": 0.2134,
  "risk_category": "Low Risk"
}
```

### Risk categories

| Category | Condition | Recommended Action |
|---|---|---|
| 🔴 High Risk | prob ≥ 0.75 | Immediate intervention — dedicated manager call, personalized offer |
| 🟡 Medium Risk | 0.40 ≤ prob < 0.75 | Targeted communication, loyalty rewards, cross-sell |
| 🟢 Low Risk | prob < 0.40 | Routine engagement only |

**Pydantic validation** — all fields are validated automatically. Invalid inputs return
HTTP 422 with a clear error message before the model is ever called:

| Field | Constraint |
|---|---|
| `CreditScore` | int, 300–900 |
| `Geography` | "France" / "Spain" / "Germany" |
| `Gender` | "Male" / "Female" |
| `Age` | int, 18–100 |
| `Tenure` | int, 0–10 |
| `NumOfProducts` | int, 1–4 |
| `HasCrCard` | "yes" / "no" |
| `IsActiveMember` | "yes" / "no" |
| `Balance` | float, ≥ 0 |
| `EstimatedSalary` | float, ≥ 0 |

The interactive Swagger UI (auto-generated by FastAPI) is available at `/docs` when the
API is running.

---

## Project Structure

```
Churn_Prediction/
│
├── BankCustomerChurn.ipynb     # Main notebook — EDA, training, evaluation
├── Churn_Modelling.csv         # Dataset (10,000 rows)
├── app.py                      # FastAPI application
├── churn_pipeline.pkl          # Trained sklearn pipeline (preprocessing + XGBoost)
├── Dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- Docker (optional — for containerized deployment)

### Install dependencies

```bash
git clone https://github.com/itsShivam73/Churn_Prediction.git
cd Churn_Prediction
pip install -r requirements.txt
```

---

## How to Run

### Run the API locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser at:
- **API root:** `http://localhost:8000`
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Test the API with curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 3,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": "yes",
    "IsActiveMember": "no",
    "EstimatedSalary": 50000
  }'
```

### Run the notebook

Open `BankCustomerChurn.ipynb` in Jupyter or Google Colab and run cells top to bottom.
The notebook covers EDA, preprocessing, model training, evaluation, and threshold tuning.

---

## Docker Deployment

### Pull from Docker Hub

```bash
docker pull itsshivaam/churn-prediction
docker run -p 8000:8000 itsshivaam/churn-prediction
```

> Docker Hub: [hub.docker.com/u/itsshivaam](https://hub.docker.com/u/itsshivaam)

### Build locally

```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

The Dockerfile uses `python:3.11-slim` as the base image (~50MB vs ~900MB for full Python)
and serves the API with `uvicorn` on port 8000.

---

## Key Design Decisions

### Why XGBoost over Logistic Regression and Random Forest?
XGBoost's sequential boosting trains each tree to correct the previous tree's errors —
more effective than Logistic Regression's linear boundary and Random Forest's parallel
bagging for this tabular banking dataset. With threshold tuning, it achieves the best
churn recall across all three models tested.

### Why threshold 0.35 instead of 0.50?
In banking, the asymmetric cost of errors justifies a lower threshold. Missing a churner
(False Negative) means losing a customer permanently. Falsely flagging a loyal customer
(False Positive) costs only a wasted retention offer. The 0.35 threshold was selected by
minimizing total business cost on the Precision-Recall curve — it reduced missed churners
from 129 to 78.

### Why a sklearn Pipeline for churn_pipeline.pkl?
Packaging preprocessing and the model together in a single Pipeline object eliminates
training-serving skew — the API cannot accidentally apply different transformations than
the training code used. The pipeline handles StandardScaler + OneHotEncoder + XGBoost as
a single `predict_proba()` call.

### Why StandardScaler for this dataset?
Logistic Regression is sensitive to feature scale — `CreditScore` (300–900), `Balance`
(0–250K), and `EstimatedSalary` (0–200K) differ vastly in magnitude. StandardScaler
normalizes to mean=0, std=1. Tree-based models (Random Forest, XGBoost) are
scale-invariant, but scaling is applied consistently across all three for fair comparison.

### Why python:3.11-slim in the Dockerfile?
The slim image reduces the final Docker image size dramatically (~50MB vs ~900MB) by
omitting OS packages not needed for running Python. The `requirements.txt` layer is copied
before `app.py` to leverage Docker's layer caching — pip install only re-runs when
dependencies change, not on every code change.

---

## Known Limitations

- Dataset is class-imbalanced (~20% churners) — techniques like SMOTE or class weighting
  could further improve minority class performance
- Model trained on a public benchmark dataset — real bank data would require retraining
  and likely different feature engineering
- No model monitoring in place — input feature distributions may drift in production
  over time without detection

---

## Future Improvements

- [ ] **SHAP values** — add explainability to show which features drove each individual prediction; critical for banking regulatory compliance
- [ ] **Class weighting** — pass `scale_pos_weight` to XGBoost to further improve churn class recall
- [ ] **SMOTE** — oversample the minority class during training to complement threshold tuning
- [ ] **Model monitoring** — track input feature drift and prediction distribution shift in production using Evidently AI
- [ ] **A/B testing framework** — measure whether retention campaigns triggered by model predictions actually reduce churn
- [ ] **CI/CD pipeline** — automate model retraining and Docker image rebuild on new data with GitHub Actions
- [ ] **Richer features** — incorporate transaction frequency, login history, and customer service call data if available

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| scikit-learn | Preprocessing, pipeline, Logistic Regression, Random Forest |
| XGBoost | Best-performing classifier |
| pandas / numpy | Data manipulation |
| FastAPI | Production REST API |
| Pydantic | Input validation and schema enforcement |
| uvicorn | ASGI server for FastAPI |
| joblib | Model serialization (`churn_pipeline.pkl`) |
| Docker | Containerization and portable deployment |
| Jupyter Notebook | EDA, training, and evaluation |

---

## Author

**Shivam Pandey**
Data Science Student | Machine Learning Enthusiast

---

## License

This project is licensed under the MIT License.
Dataset sourced from Kaggle under public license.
