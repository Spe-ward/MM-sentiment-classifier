# Model Degradation Detection & Redeployment Strategy

## The Problem

The deployed SVM model was trained on the IMDb 50K dataset — English-language movie reviews from a fixed point in time. In production, the model will encounter reviews that differ from this training distribution: new slang, different review styles, Japanese input (translated before prediction), or entirely different domains. Without monitoring, these shifts silently degrade prediction quality with no alert to the team.

This document answers two questions:
1. **How do we detect that the model is degrading?**
2. **How do we automatically retrain and redeploy if it is?**

---

## What We Already Have

The API logs every prediction to `logs/predictions.jsonl`:

```json
{
  "timestamp": "2026-03-30T12:00:00+00:00",
  "review": "This movie was amazing!",
  "language_detected": "en",
  "translated": false,
  "sentiment": "positive",
  "confidence": 0.9884,
  "model": "svm",
  "response_time_ms": 12.34
}
```

Aggregate stats are available live at `/api/stats` and snapshotted to `logs/stats_snapshots.jsonl` every 100 predictions. This is the foundation for everything below.

---

## Detecting Degradation

### Strategy: Confidence-Based Drift Detection

We use **prediction confidence distribution** as the primary degradation signal. This works without ground truth labels, which is critical because in production we rarely get immediate feedback on whether a prediction was correct.

**How it works:**

A well-calibrated model produces high-confidence predictions on data similar to its training set. When input data drifts — new vocabulary, different sentence structures, domain shift — the model becomes less certain. This manifests as a drop in average confidence and an increase in predictions falling below a confidence threshold.

**Implementation:**

A daily batch job reads `logs/predictions.jsonl` and computes:

| Metric | How to compute | Alert threshold |
|--------|---------------|-----------------|
| **Mean confidence** | Average of `confidence` field over last 24h | < 0.75 |
| **Low-confidence rate** | % of predictions with confidence < 0.6 | > 15% |
| **Positive/negative ratio** | `positive_count / total` over last 24h | Outside [0.3, 0.7] |

**Why these thresholds:** The deployed SVM produces mean confidence of ~0.93 on the validation set. A drop to 0.75 represents a significant shift. The positive/negative ratio on balanced review data should hover near 0.5 — a persistent skew suggests the model is systematically misclassifying one class.

**What this catches:**
- Vocabulary drift (new words the TF-IDF vectorizer hasn't seen → lower confidence)
- Domain shift (non-movie reviews hitting the endpoint)
- Translation quality issues (poorly translated JA input producing garbled English)
- Data quality problems (empty reviews, spam, non-review text)

**What this does not catch:**
- Subtle accuracy loss where the model remains confident but wrong (requires labeled data)
- Gradual drift that stays within thresholds

For the second case, if a ground truth feedback mechanism is added later (e.g., users can flag incorrect predictions), rolling F1 computed against those labels replaces confidence monitoring as the primary signal, with an alert threshold of F1 < 0.85 (compared to the model's validation F1 of 0.9178).

---

## Retraining Pipeline

When degradation is detected, the following automated pipeline runs:

```
Alert triggered (confidence drift or F1 drop)
  │
  ▼
Collect data
  │  Pull recent predictions from logs/predictions.jsonl
  │  If labeled feedback exists, include those labels
  │  Merge with original IMDb training data
  │
  ▼
Preprocess
  │  Apply the same clean_review() pipeline
  │  Fit a new TF-IDF vectorizer on the combined corpus
  │
  ▼
Train candidate model
  │  Train SVM (LinearSVC) with identical hyperparameters
  │  Same 80/10/10 split strategy on the combined dataset
  │
  ▼
Evaluate
  │  Compute F1, accuracy, ROC-AUC on held-out validation set
  │  Compare against current production model (F1 = 0.9178)
  │
  ▼
Decision gate
  │  If candidate F1 > production F1 → promote
  │  If candidate F1 ≤ production F1 → alert team, keep current model
  │
  ▼
Deploy (if promoted)
  │  Save new model artifacts (svm.joblib, tfidf_vectorizer.joblib)
  │  Build new Docker image tagged with model version
  │  Deploy via blue/green swap
```

**Trigger conditions (any one is sufficient):**
- Mean confidence < 0.75 for 3 consecutive daily checks
- Low-confidence rate > 15% for 3 consecutive daily checks
- Rolling F1 < 0.85 (if labeled data available)
- Monthly scheduled retrain regardless of metrics (defensive)

**Safeguard:** A candidate model is never deployed if it scores lower than the current production model on the held-out set. This prevents a bad retrain from making things worse.

---

## Redeployment Strategy: Blue/Green

```
                ┌──────────────────────┐
                │   Load Balancer       │
                └──────┬───────────────┘
                       │
            ┌──────────▼──────────┐
            │  Traffic routing     │
            └───┬─────────────┬───┘
                │             │
        ┌───────▼──────┐ ┌───▼────────┐
        │  Blue (v1.0) │ │ Green(v1.1)│
        │  ACTIVE      │ │ STANDBY    │
        └──────────────┘ └────────────┘
```

1. The current model runs in the **blue** container (active, serving all traffic)
2. After retraining, the candidate model is deployed to the **green** container
3. Automated validation tests run against green: health check, sample predictions, latency check
4. If green passes validation, the load balancer swaps traffic from blue to green
5. Blue stays running for 24 hours as a rollback target
6. If green shows degraded metrics within 24 hours, traffic routes back to blue automatically

**Rollback trigger:** If the newly deployed model's mean confidence drops below the previous model's by more than 5%, or if error rate exceeds 1%, traffic is automatically routed back to the previous version.

**Versioning:** Each model is tagged (`v1.0.0`, `v1.1.0`, etc.) and stored alongside its Docker image. At least 3 previous versions are retained for rollback.

---

## End-to-End Flow

```
  Predictions arrive
        │
        ▼
  API serves response ──→ logs/predictions.jsonl
        │
        ▼
  Dashboard (live)        Daily batch job
  /api/stats              reads logs
                                │
                                ▼
                          Compute metrics
                          (confidence mean,
                           low-conf rate,
                           pos/neg ratio)
                                │
                          Threshold exceeded?
                           │           │
                          No          Yes
                           │           │
                        (no action)    ▼
                                  Retrain pipeline
                                  (collect → train → evaluate)
                                       │
                                  Candidate better?
                                   │          │
                                  No         Yes
                                   │          │
                              Alert team   Blue/green deploy
                                           │
                                       Monitor 24h
                                       │        │
                                    Stable    Degraded
                                       │        │
                                  Keep new   Rollback
```
