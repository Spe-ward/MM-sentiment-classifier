"""Dashboard API endpoints, prediction tracking, and persistent logging."""

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import time

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

MODELS_DIR = Path("models")
LOG_DIR = Path("logs")

router = APIRouter(prefix="/api", tags=["Dashboard"])


# --- Prediction Logger (persistent) ---

class PredictionLogger:
    """Writes prediction records and stats snapshots to JSONL files in logs/."""

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.predictions_path = self.log_dir / "predictions.jsonl"
        self.snapshots_path = self.log_dir / "stats_snapshots.jsonl"

    def log_prediction(self, record: dict):
        """Append a single prediction record to predictions.jsonl."""
        with open(self.predictions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_stats_snapshot(self, stats: dict):
        """Append a stats snapshot to stats_snapshots.jsonl."""
        snapshot = {"timestamp": datetime.now(timezone.utc).isoformat(), **stats}
        with open(self.snapshots_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Read the last N predictions from the log file."""
        if not self.predictions_path.exists():
            return []
        # Read last N lines efficiently
        lines = deque(maxlen=limit)
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        results = []
        for line in lines:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return results


logger = PredictionLogger()


# --- Prediction Tracker (in-memory + persistent) ---

SNAPSHOT_INTERVAL = 100  # write stats snapshot every N predictions

@dataclass
class PredictionTracker:
    """In-memory tracker for live prediction statistics with persistent logging."""

    start_time: float = field(default_factory=time)
    total: int = 0
    confidence_sum: float = 0.0
    positive: int = 0
    negative: int = 0
    languages: dict = field(default_factory=dict)

    def record(
        self,
        sentiment: str,
        confidence: float,
        language: str,
        review: str = "",
        model: str = "",
        response_time_ms: float = 0.0,
        translated: bool = False,
    ):
        # Update in-memory counters
        self.total += 1
        self.confidence_sum += confidence
        if sentiment == "positive":
            self.positive += 1
        else:
            self.negative += 1
        self.languages[language] = self.languages.get(language, 0) + 1

        # Write persistent log
        logger.log_prediction({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "review": review[:200],
            "language_detected": language,
            "translated": translated,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "model": model,
            "response_time_ms": round(response_time_ms, 2),
        })

        # Periodic stats snapshot
        if self.total % SNAPSHOT_INTERVAL == 0:
            logger.log_stats_snapshot(self.summary())

    def summary(self) -> dict:
        return {
            "total_predictions": self.total,
            "avg_confidence": round(self.confidence_sum / self.total, 4) if self.total > 0 else 0,
            "positive_count": self.positive,
            "negative_count": self.negative,
            "language_breakdown": self.languages,
            "uptime_seconds": round(time() - self.start_time, 1),
        }


tracker = PredictionTracker()


# --- Endpoints ---

@router.get("/models")
async def get_models():
    """Return all model metrics and indicate the active model."""
    from src.api import predictor

    active_name = predictor.model_name if predictor else "unknown"

    models = []

    # Traditional models + SBERT
    results_path = MODELS_DIR / "traditional_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        for _, row in df.iterrows():
            entry = row.to_dict()
            entry["is_active"] = entry["model"] == active_name
            models.append(entry)

    # Transformer
    transformer_path = MODELS_DIR / "transformer_results.json"
    if transformer_path.exists():
        with open(transformer_path) as f:
            t = json.load(f)
        models.append({
            "model": t["model"],
            "accuracy": t["accuracy"],
            "precision": t["precision"],
            "recall": t["recall"],
            "f1": t["f1"],
            "roc_auc": t["roc_auc"],
            "train_time_sec": t["train_time_sec"],
            "inference_time_sec": t["inference_time_sec"],
            "model_size_mb": t["model_size_mb"],
            "is_active": t["model"] == active_name,
        })

    # Sort by F1 descending
    models.sort(key=lambda m: m.get("f1", 0), reverse=True)

    return {"active_model": active_name, "models": models}


@router.get("/stats")
async def get_stats():
    """Return live prediction statistics."""
    return tracker.summary()


@router.get("/predictions")
async def get_predictions(limit: int = Query(default=20, ge=1, le=200)):
    """Return the most recent predictions from the persistent log."""
    return logger.get_recent(limit)


DOC_MAP = {
    "readme": Path("README.md"),
    "api": Path("docs/API.md"),
    "monitoring": Path("docs/MONITORING.md"),
}


@router.get("/docs/{name}", response_class=PlainTextResponse)
async def get_doc(name: str):
    """Return raw markdown content for a named document."""
    path = DOC_MAP.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found")
    return path.read_text(encoding="utf-8")
