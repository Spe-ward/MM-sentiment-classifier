"""Prediction logic and model loading for the sentiment API."""

import joblib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.preprocess import clean_review

MODELS_DIR = Path("models")


class SentimentPredictor:
    """Loads a trained model and vectorizer for sentiment prediction."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self._select_best_model()
        self._load_model()

    def _select_best_model(self) -> str:
        """Select the best traditional model based on saved results."""
        results_path = MODELS_DIR / "traditional_results.csv"
        if results_path.exists():
            import pandas as pd
            df = pd.read_csv(results_path)
            return df.sort_values("f1", ascending=False).iloc[0]["model"]
        # Fallback
        return "logistic_regression"

    def _load_model(self):
        """Load the model and TF-IDF vectorizer from disk."""
        model_path = MODELS_DIR / f"{self.model_name}.joblib"
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single review."""
        cleaned = clean_review(text)
        features = self.vectorizer.transform([cleaned])
        proba = self.model.predict_proba(features)[0]
        label = "positive" if proba[1] >= 0.5 else "negative"
        confidence = float(max(proba))

        return {
            "sentiment": label,
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict sentiment for multiple reviews."""
        cleaned = [clean_review(t) for t in texts]
        features = self.vectorizer.transform(cleaned)
        probas = self.model.predict_proba(features)

        results = []
        for text, proba in zip(texts, probas):
            label = "positive" if proba[1] >= 0.5 else "negative"
            confidence = float(max(proba))
            results.append({
                "review": text,
                "sentiment": label,
                "confidence": round(confidence, 4),
            })
        return results
