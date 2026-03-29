"""Fine-tune DistilBERT for sentiment classification (CPU-compatible)."""

import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.preprocess import clean_review


RANDOM_SEED = 42
MODELS_DIR = Path("models")
DEVICE = torch.device("cpu")

# Hyperparameters (tuned for CPU feasibility)
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5


class ReviewDataset(Dataset):
    """PyTorch dataset for tokenized reviews."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, scheduler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load the same test split used by traditional models
    test_data = joblib.load(MODELS_DIR / "test_data.joblib")
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    # Load full dataset and reconstruct training split
    print("Loading and preprocessing data...")
    df = pd.read_csv("data/IMDB Dataset.csv")
    df["clean_review"] = [clean_review(r) for r in tqdm(df["review"], desc="Preprocessing")]
    df["label"] = (df["sentiment"] == "positive").astype(int)

    from sklearn.model_selection import train_test_split
    X = df["clean_review"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )

    X_train = X_train.tolist()
    X_val = X_val.tolist()
    y_train = y_train.tolist()
    y_val = y_val.tolist()

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Tokenizer and model
    print("Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(DEVICE)

    # Datasets
    train_dataset = ReviewDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = ReviewDataset(X_val, y_val, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    print(f"\nTraining DistilBERT for {EPOCHS} epochs on {DEVICE}...")
    print(f"Batch size: {BATCH_SIZE} | Max length: {MAX_LEN} | LR: {LEARNING_RATE}")
    print(f"Total steps: {total_steps:,}")

    best_f1 = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_metrics = evaluate(model, val_loader)
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            # Save best model
            save_dir = MODELS_DIR / "distilbert"
            save_dir.mkdir(exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  Best model saved (F1: {best_f1:.4f})")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save results
    results = {
        "model": "distilbert",
        "accuracy": val_metrics["accuracy"],
        "f1": val_metrics["f1"],
        "roc_auc": val_metrics["roc_auc"],
        "train_time_sec": round(total_time, 2),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
    }

    with open(MODELS_DIR / "transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal results: {results}")


if __name__ == "__main__":
    main()
