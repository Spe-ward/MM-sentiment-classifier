"""Tests for the dashboard API endpoints and logging."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app
from src.predict import SentimentPredictor
from src.dashboard import LOG_DIR
import src.api as api_module

# Load predictor for tests
api_module.predictor = SentimentPredictor()

client = TestClient(app)


def test_root_redirects():
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert "/static/index.html" in resp.headers["location"]


def test_get_models():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_model" in data
    assert "models" in data
    assert len(data["models"]) >= 1
    for m in data["models"]:
        assert "model" in m
        assert "f1" in m
        assert "is_active" in m


def test_get_stats():
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "avg_confidence" in data
    assert "positive_count" in data
    assert "negative_count" in data
    assert "language_breakdown" in data
    assert "uptime_seconds" in data


def test_get_doc_readme():
    resp = client.get("/api/docs/readme")
    assert resp.status_code == 200
    assert "Sentiment Classification" in resp.text


def test_get_doc_api():
    resp = client.get("/api/docs/api")
    assert resp.status_code == 200
    assert "/predict" in resp.text


def test_get_doc_monitoring():
    resp = client.get("/api/docs/monitoring")
    assert resp.status_code == 200
    assert "Degradation" in resp.text


def test_get_doc_not_found():
    resp = client.get("/api/docs/nonexistent")
    assert resp.status_code == 404


def test_static_index():
    resp = client.get("/static/index.html")
    assert resp.status_code == 200
    assert "Sentiment Classifier" in resp.text


def test_tracker_updates_on_predict():
    stats_before = client.get("/api/stats").json()
    before_count = stats_before["total_predictions"]

    client.post("/predict", json={"review": "Great movie!"})

    stats_after = client.get("/api/stats").json()
    assert stats_after["total_predictions"] == before_count + 1


def test_prediction_logged_to_file():
    """Verify predictions are written to the JSONL log file."""
    client.post("/predict", json={"review": "Logging test review"})

    log_path = LOG_DIR / "predictions.jsonl"
    assert log_path.exists(), "predictions.jsonl should exist after a prediction"

    # Read last line
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    last = json.loads(lines[-1])
    assert "timestamp" in last
    assert "sentiment" in last
    assert "confidence" in last
    assert "review" in last
    assert "response_time_ms" in last
    assert "model" in last
    assert "language_detected" in last


def test_get_predictions_endpoint():
    # Ensure at least one prediction exists
    client.post("/predict", json={"review": "Test for predictions endpoint"})

    resp = client.get("/api/predictions?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "timestamp" in data[-1]
    assert "sentiment" in data[-1]


def test_api_log_exists():
    """Verify the api.log file is created."""
    log_path = LOG_DIR / "api.log"
    assert log_path.exists(), "api.log should exist after app startup"
