"""Train a logistic regression classifier on sentence-transformer embeddings."""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.preprocess import clean_review

RANDOM_SEED = 42
MODELS_DIR = Path("models")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast, strong for classification


def main():
    # Load and preprocess (same pipeline as traditional models)
    print("Loading dataset...")
    df = pd.read_csv("data/IMDB Dataset.csv")

    print("Cleaning text...")
    df["clean_review"] = [clean_review(r) for r in tqdm(df["review"], desc="Preprocessing")]
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # Same split as all other models (80/10/10, seed=42)
    X = df["clean_review"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Encode with sentence-transformer
    print(f"\nLoading sentence-transformer: {EMBEDDING_MODEL}...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding training set...")
    start_encode = time.time()
    X_train_emb = encoder.encode(X_train.tolist(), show_progress_bar=True, batch_size=128)
    encode_time = time.time() - start_encode
    print(f"Encoding time (train): {encode_time:.1f}s")

    print("Encoding validation set...")
    X_val_emb = encoder.encode(X_val.tolist(), show_progress_bar=True, batch_size=128)

    print(f"Embedding dimension: {X_train_emb.shape[1]}")

    # Train logistic regression on embeddings
    print("\nTraining LogisticRegression on sentence embeddings...")
    start_train = time.time()
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_emb, y_train)
    train_time = time.time() - start_train
    total_time = time.time() - start_encode  # includes encoding

    # Evaluate on validation set
    y_pred = model.predict(X_val_emb)
    y_proba = model.predict_proba(X_val_emb)[:, 1]

    # Inference time (encoding + prediction on val set)
    start_inference = time.time()
    X_val_emb_fresh = encoder.encode(X_val.tolist(), show_progress_bar=False, batch_size=128)
    model.predict(X_val_emb_fresh)
    inference_time = time.time() - start_inference

    metrics = {
        "model": "sbert_logreg",
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "train_time_sec": round(total_time, 2),
        "inference_time_sec": round(inference_time, 4),
    }

    print(f"\n{'='*60}")
    print(f"SBERT Embeddings + LogReg Results (Validation Set):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Total time (encode+train): {total_time:.1f}s")
    print(f"  Inference time (val set):  {inference_time:.2f}s")

    # Save model and encoder name
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "sbert_logreg.joblib"
    joblib.dump({"classifier": model, "encoder_name": EMBEDDING_MODEL}, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    metrics["model_size_mb"] = round(model_size_mb, 2)
    print(f"  Model size (classifier only): {model_size_mb:.2f} MB")
    print(f"  + Encoder ({EMBEDDING_MODEL}): ~80 MB")

    # Append to traditional results CSV for unified comparison
    results_path = MODELS_DIR / "traditional_results.csv"
    if results_path.exists():
        existing = pd.read_csv(results_path)
        # Remove old sbert_logreg row if re-running
        existing = existing[existing["model"] != "sbert_logreg"]
        updated = pd.concat([existing, pd.DataFrame([metrics])], ignore_index=True)
    else:
        updated = pd.DataFrame([metrics])

    updated = updated.sort_values("f1", ascending=False)
    updated.to_csv(results_path, index=False)
    print(f"\nResults appended to {results_path}")
    print(updated.to_string(index=False))


if __name__ == "__main__":
    main()
