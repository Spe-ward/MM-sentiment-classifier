# API Documentation

## Base URL

```
http://localhost:8000
```

## Interactive Docs

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Web Dashboard

Open http://localhost:8000 to access the dashboard with model stats, live prediction tracking, and documentation.

---

## Prediction Endpoints

### POST /predict

Classify the sentiment of a single movie review. Accepts English or Japanese text.

**Request:**
```json
{
  "review": "This movie was absolutely fantastic!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.9022,
  "language_detected": "en"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic!"}'
```

**Japanese input:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "この映画は本当に素晴らしかった！"}'
```

---

### POST /predict/batch

Classify multiple reviews in a single request. Maximum 100 reviews per request.

**Request:**
```json
{
  "reviews": [
    "Great film!",
    "Terrible waste of time.",
    "この映画は最悪でした。"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "review": "Great film!",
      "sentiment": "positive",
      "confidence": 0.9994,
      "language_detected": "en"
    },
    {
      "review": "Terrible waste of time.",
      "sentiment": "negative",
      "confidence": 1.0,
      "language_detected": "en"
    },
    {
      "review": "この映画は最悪でした。",
      "sentiment": "negative",
      "confidence": 0.8721,
      "language_detected": "ja"
    }
  ]
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great film!", "Terrible waste of time."]}'
```

---

## System Endpoints

### GET /health

Health check and readiness probe.

**Response:**
```json
{
  "status": "ok",
  "model": "svm"
}
```

---

### GET /

Redirects to the web dashboard at `/static/index.html`.

---

## Dashboard API Endpoints

These endpoints power the web dashboard and are also available for programmatic use.

### GET /api/models

Returns all trained model metrics and indicates which model is currently active.

**Response:**
```json
{
  "active_model": "svm",
  "models": [
    {
      "model": "distilbert",
      "accuracy": 0.9186,
      "precision": 0.9164,
      "recall": 0.9212,
      "f1": 0.9188,
      "roc_auc": 0.9754,
      "train_time_sec": 2910.33,
      "inference_time_sec": 38.8515,
      "model_size_mb": 256.1,
      "is_active": false
    },
    {
      "model": "svm",
      "accuracy": 0.9172,
      "precision": 0.9113,
      "recall": 0.9244,
      "f1": 0.9178,
      "roc_auc": 0.9731,
      "train_time_sec": 3.51,
      "inference_time_sec": 0.0166,
      "model_size_mb": 1.91,
      "is_active": true
    }
  ]
}
```

---

### GET /api/stats

Returns live prediction statistics since server startup.

**Response:**
```json
{
  "total_predictions": 42,
  "avg_confidence": 0.9231,
  "positive_count": 25,
  "negative_count": 17,
  "language_breakdown": {"en": 38, "ja": 4},
  "uptime_seconds": 3600.5
}
```

---

### GET /api/predictions?limit=20

Returns the most recent predictions from the persistent log file. The `limit` parameter controls how many to return (1-200, default 20).

**Response:**
```json
[
  {
    "timestamp": "2026-03-30T12:00:00.000000+00:00",
    "review": "This movie was amazing!",
    "language_detected": "en",
    "translated": false,
    "sentiment": "positive",
    "confidence": 0.9884,
    "model": "svm",
    "response_time_ms": 12.34
  }
]
```

---

### GET /api/docs/{name}

Returns raw markdown content for project documentation. Valid names: `readme`, `api`, `monitoring`.

**Response:** Plain text (markdown).

Returns 404 for unknown document names.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (e.g., batch size > 100) |
| 404 | Not found (e.g., unknown doc name) |
| 422 | Validation error (e.g., empty review string) |
| 500 | Internal server error |

**Example 422 response:**
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "review"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `sentiment` | string | `"positive"` or `"negative"` |
| `confidence` | float | Model confidence score (0.0 to 1.0) |
| `language_detected` | string | ISO 639-1 language code of the input (e.g., `"en"`, `"ja"`) |

## Logging

Every prediction is logged to `logs/predictions.jsonl` with timestamp, review text (first 200 chars), detected language, sentiment, confidence, model name, and response latency. Stats snapshots are written to `logs/stats_snapshots.jsonl` every 100 predictions. Application events are logged to `logs/api.log`.
