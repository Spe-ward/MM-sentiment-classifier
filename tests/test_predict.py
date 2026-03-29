"""Tests for the prediction module."""

from src.predict import SentimentPredictor
from src.preprocess import clean_review


def test_clean_review_html():
    text = "Great movie!<br /><br />Loved it."
    cleaned = clean_review(text)
    assert "<br" not in cleaned
    assert "great movie" in cleaned


def test_clean_review_lowercase():
    assert clean_review("HELLO WORLD") == "hello world"


def test_predictor_loads():
    p = SentimentPredictor()
    assert p.model is not None
    assert p.vectorizer is not None


def test_predict_returns_valid_result():
    p = SentimentPredictor()
    result = p.predict("This movie was great!")
    assert result["sentiment"] in ("positive", "negative")
    assert 0 <= result["confidence"] <= 1


def test_predict_batch():
    p = SentimentPredictor()
    results = p.predict_batch(["Good movie", "Bad movie"])
    assert len(results) == 2
    for r in results:
        assert "review" in r
        assert "sentiment" in r
        assert "confidence" in r
