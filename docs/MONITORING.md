# Model Degradation Detection & Redeployment Strategy

## Overview

In production, model performance can degrade over time due to changes in input data distribution, evolving language patterns, or shifts in user behavior. This document outlines a strategy to detect degradation and automatically trigger retraining and redeployment.

---

## 1. Degradation Detection

### 1.1 Data Drift Monitoring

Data drift occurs when the distribution of incoming data diverges from the training data.

**Approach:**
- Log all incoming review texts and their TF-IDF feature vectors
- Periodically compare the feature distribution of recent predictions against the training set baseline
- **Statistical tests:**
  - **Population Stability Index (PSI)** on TF-IDF feature distributions — PSI > 0.2 indicates significant drift
  - **Kolmogorov-Smirnov (KS) test** on individual high-importance features
- **Text-level signals:**
  - Average review length shifting significantly
  - New vocabulary appearing that is out-of-vocabulary for the TF-IDF vectorizer
  - Language distribution shift (e.g., sudden increase in JA vs EN input)

### 1.2 Prediction Drift Monitoring

Even without labeled data, shifts in prediction patterns can indicate problems.

**Approach:**
- Track the distribution of confidence scores over time
- Alert if:
  - Mean confidence drops below a threshold (e.g., < 0.7)
  - The positive/negative prediction ratio shifts beyond expected bounds
  - The proportion of low-confidence predictions (< 0.6) increases significantly

### 1.3 Ground Truth Feedback Loop

When ground truth labels become available (e.g., through user feedback or manual review):

**Approach:**
- Compute rolling accuracy, F1, and ROC-AUC on labeled samples
- Alert if any metric drops below a threshold (e.g., F1 < 0.85)
- Compare against the model's validation set performance as the baseline

### 1.4 Latency and Throughput Anomalies

Performance degradation can also manifest as infrastructure issues.

**Approach:**
- Monitor API response times (p50, p95, p99)
- Track request throughput and error rates
- Alert on sudden spikes in latency or error rates

---

## 2. Monitoring Implementation

### 2.1 Logging Pipeline

```
API Request → Prediction → Log to database/file
                               ↓
                         Async batch job (hourly/daily)
                               ↓
                         Compute drift metrics
                               ↓
                         Alert if thresholds exceeded
```

**What to log per prediction:**
- Timestamp
- Raw input text (or hash for privacy)
- Detected language
- Predicted sentiment and confidence score
- Preprocessing and inference latency
- Model version

### 2.2 Metrics Dashboard

A Prometheus + Grafana stack is recommended for real-time monitoring:

| Metric | Type | Alert Threshold |
|--------|------|-----------------|
| `prediction_confidence_mean` | Gauge | < 0.7 |
| `prediction_positive_ratio` | Gauge | Outside [0.3, 0.7] |
| `data_drift_psi` | Gauge | > 0.2 |
| `api_latency_p95` | Histogram | > 500ms |
| `api_error_rate` | Counter | > 1% |
| `model_accuracy_rolling` | Gauge | < 0.85 (if labels available) |

---

## 3. Automated Retraining Pipeline

### 3.1 Trigger Conditions

Retraining is triggered when any of the following conditions are met:

1. **Data drift detected:** PSI > 0.2 on key features for 3 consecutive measurement windows
2. **Performance drop:** Rolling F1 drops below 0.85 (requires labeled data)
3. **Prediction drift:** Mean confidence below 0.7 for 24+ hours
4. **Scheduled:** Monthly retrain regardless of drift (defensive measure)

### 3.2 Retraining Pipeline

```
Trigger
  ↓
Collect new data (recent predictions + any available labels)
  ↓
Merge with original training data
  ↓
Preprocess (same pipeline as initial training)
  ↓
Train candidate model (same architecture, same hyperparameters)
  ↓
Evaluate on held-out validation set
  ↓
Compare against current production model
  ↓
If better → Promote to production
If worse → Alert team, keep current model
```

**Key principles:**
- Never deploy a model that performs worse than the current one
- Always evaluate on a consistent held-out set
- Log all training runs for reproducibility

### 3.3 Model Versioning

- Each model is tagged with a version (e.g., `v1.0.0`, `v1.1.0`)
- Store all model artifacts in versioned storage (e.g., S3 with version prefixes, or MLflow Model Registry)
- Keep at least the last 3 model versions available for rollback

---

## 4. Deployment Strategy

### 4.1 Blue/Green Deployment

1. **Blue (current):** Serving production traffic
2. **Green (candidate):** New model deployed to a staging environment
3. Run validation tests against the green deployment
4. If tests pass, swap traffic from blue to green
5. Keep blue available for instant rollback

### 4.2 Canary Deployment (Alternative)

1. Route 5-10% of traffic to the new model
2. Monitor metrics for the canary vs. the stable deployment
3. If canary performs well over 24 hours, gradually increase to 100%
4. If canary degrades, route all traffic back to the stable deployment

### 4.3 Rollback

- Automated rollback if the new model's error rate exceeds the previous model's by > 2%
- Manual rollback available via a single command/API call
- Rollback target: the last known-good model version

---

## 5. Architecture Diagram

```
                    ┌─────────────────────────────┐
                    │       Incoming Requests       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Load Balancer /         │
                    │      API Gateway             │
                    └──────┬───────────────┬──────┘
                           │               │
                   ┌───────▼──────┐ ┌──────▼───────┐
                   │  Blue (v1.0) │ │ Green (v1.1) │
                   │  (stable)    │ │ (candidate)  │
                   └───────┬──────┘ └──────┬───────┘
                           │               │
                    ┌──────▼───────────────▼──────┐
                    │      Prediction Logger       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Monitoring Service       │
                    │  (drift detection, alerting) │
                    └──────────────┬──────────────┘
                                   │
                           ┌───────▼───────┐
                           │  Alert / Trigger│
                           │  Retrain?       │
                           └───────┬───────┘
                                   │ Yes
                    ┌──────────────▼──────────────┐
                    │    Retraining Pipeline        │
                    │  (collect → train → evaluate) │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Model Registry (versioned)  │
                    └──────────────────────────────┘
```

---

## 6. Tools & Technologies (Recommended)

| Component | Tool | Notes |
|-----------|------|-------|
| Prediction logging | Structured JSON logs → ELK or CloudWatch | Append-only, queryable |
| Drift detection | Custom Python job (scipy, numpy) | Runs hourly/daily via cron or Airflow |
| Metrics | Prometheus | Exposes `/metrics` endpoint from the API |
| Dashboards | Grafana | Visualize drift, confidence, latency |
| Alerting | Grafana Alerts or PagerDuty | Notify on threshold breaches |
| Model registry | MLflow or S3 versioned paths | Track experiments and artifacts |
| Orchestration | Airflow or GitHub Actions | Trigger retraining pipeline |
| Deployment | Docker + Kubernetes or ECS | Blue/green via service mesh or ALB |
