"""Training script for traditional ML models and transformer."""

import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier
from tqdm import tqdm

from src.preprocess import clean_review


RANDOM_SEED = 42
DATA_PATH = Path("data/IMDB Dataset.csv")
MODELS_DIR = Path("models")

# Models that don't natively support predict_proba need calibration
TRADITIONAL_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "naive_bayes": MultinomialNB(),
    "svm": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_SEED)),
    "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED),
    "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=RANDOM_SEED, verbose=-1),
    "ridge": CalibratedClassifierCV(RidgeClassifier(random_state=RANDOM_SEED)),
}


def load_and_preprocess():
    """Load dataset and apply text cleaning."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Cleaning text...")
    df["clean_review"] = [clean_review(r) for r in tqdm(df["review"], desc="Preprocessing")]

    # Encode labels: positive=1, negative=0
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


def split_data(df):
    """Split into train/val/test (80/10/10) with stratification."""
    X = df["clean_review"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize(X_train, X_val, X_test):
    """Fit TF-IDF vectorizer on training data and transform all splits."""
    print("Fitting TF-IDF vectorizer (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # Save vectorizer
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    print(f"Vectorizer saved to {MODELS_DIR / 'tfidf_vectorizer.joblib'}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def evaluate_model(model, X, y, model_name):
    """Evaluate a model and return metrics dict."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    return metrics


def train_traditional_models(X_train, X_val, y_train, y_val):
    """Train all traditional ML models and return results."""
    results = []

    for name, model in TRADITIONAL_MODELS.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate on validation set
        metrics = evaluate_model(model, X_val, y_val, name)
        metrics["train_time_sec"] = round(train_time, 2)

        # Inference speed (on validation set)
        start = time.time()
        model.predict(X_val)
        metrics["inference_time_sec"] = round(time.time() - start, 4)

        # Save model
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        metrics["model_size_mb"] = round(model_size_mb, 2)

        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
              f"ROC-AUC: {metrics['roc_auc']:.4f} | Time: {train_time:.1f}s | "
              f"Size: {model_size_mb:.1f}MB")

        results.append(metrics)

    return results


def main():
    """Full training pipeline for traditional models."""
    df = load_and_preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = vectorize(X_train, X_val, X_test)

    # Train and evaluate traditional models on validation set
    results = train_traditional_models(X_train_tfidf, X_val_tfidf, y_train, y_val)

    # Save results
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(MODELS_DIR / "traditional_results.csv", index=False)
    print(f"\n{'='*60}")
    print("Traditional Model Results (Validation Set):")
    print(results_df.to_string(index=False))

    # Save test data for later evaluation
    test_data = {
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
    }
    joblib.dump(test_data, MODELS_DIR / "test_data.joblib")
    joblib.dump(X_test_tfidf, MODELS_DIR / "X_test_tfidf.joblib")
    print(f"\nTest data saved for Phase 3 evaluation.")


if __name__ == "__main__":
    main()
