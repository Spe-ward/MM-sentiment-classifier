# API Documentation

## Base URL

```
http://localhost:8000
```

## Interactive Docs

FastAPI auto-generates interactive documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Endpoints

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

**Example (Japanese input):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "この映画は本当に素晴らしかった！"}'
```

---

### POST /predict/batch

Classify the sentiment of multiple reviews in a single request. Maximum 100 reviews per request.

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

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (e.g., batch size > 100) |
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
