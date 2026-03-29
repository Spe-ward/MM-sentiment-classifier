"""Tests for the sentiment classification API."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.predict import SentimentPredictor
import src.api as api_module

# Load predictor once for tests (lifespan doesn't run with TestClient)
api_module.predictor = SentimentPredictor()

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_predict_positive():
    resp = client.post("/predict", json={"review": "This movie was absolutely fantastic!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] in ("positive", "negative")
    assert 0 <= data["confidence"] <= 1
    assert "language_detected" in data


def test_predict_negative():
    resp = client.post("/predict", json={"review": "Terrible movie, complete waste of time."})
    assert resp.status_code == 200
    assert resp.json()["sentiment"] == "negative"


def test_predict_empty_review():
    resp = client.post("/predict", json={"review": ""})
    assert resp.status_code == 422


def test_predict_batch():
    resp = client.post("/predict/batch", json={
        "reviews": ["Great film!", "Awful movie."]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "sentiment" in pred
        assert "confidence" in pred
        assert "language_detected" in pred


def test_predict_batch_empty_list():
    resp = client.post("/predict/batch", json={"reviews": []})
    assert resp.status_code == 422
