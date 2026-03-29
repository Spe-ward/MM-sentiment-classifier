# Sentiment Classification for Movie Reviews (EN/JA)

Binary sentiment classification (positive/negative) for movie reviews, with support for both English and Japanese input.

## Overview

Compares eight ML models on the [IMDb 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) dataset and serves the best one via a REST API with a web dashboard. Japanese reviews are automatically detected and translated offline before prediction.

## Model Results

| Model | Accuracy | F1 | ROC-AUC | Train Time | Size |
|-------|----------|-----|---------|------------|------|
| **DistilBERT** | **91.86%** | **0.9188** | **0.9754** | 2910s (GPU) | 256 MB |
| SVM (LinearSVC) | 91.72% | 0.9178 | 0.9731 | 3.5s | 1.9 MB |
| Ridge Classifier | 91.68% | 0.9175 | 0.9727 | 5.2s | 1.9 MB |
| Logistic Regression | 90.66% | 0.9078 | 0.9703 | 1.3s | 0.4 MB |
| LightGBM | 89.64% | 0.8973 | 0.9628 | 190.7s | 2.8 MB |
| Naive Bayes | 88.30% | 0.8845 | 0.9528 | 0.1s | 1.5 MB |
| Random Forest | 86.58% | 0.8669 | 0.9396 | 96.1s | 197.1 MB |
| SBERT + LogReg | 83.34% | 0.8332 | 0.9119 | 2157s | ~80 MB |

*Validation set (5,000 samples). Full analysis in [notebooks/03_evaluation.ipynb](notebooks/03_evaluation.ipynb).*

**Deployed model: SVM (LinearSVC)** — 0.1% F1 behind DistilBERT but 130x smaller and 2300x faster inference. No GPU needed.

## How to Run

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Local Setup

```bash
git clone https://github.com/Spe-ward/MM-sentiment-classifier.git
cd MM-sentiment-classifier

python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# source venv/bin/activate      # macOS/Linux

# Install runtime dependencies only (for serving)
pip install -r requirements.txt

# OR install everything (training, EDA, testing)
pip install -r requirements-dev.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Download dataset (requires kaggle in requirements-dev.txt)
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p data/
unzip data/imdb-dataset-of-50k-movie-reviews.zip -d data/

# Train models (requires requirements-dev.txt)
python -m src.train

# Start the server (only requires requirements.txt)
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### Web Dashboard

Open http://localhost:8000 to access the dashboard with:
- Live model metrics and health status
- Prediction statistics and recent prediction log
- Full project documentation

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Single prediction (English)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic!"}'

# Single prediction (Japanese — auto-translated)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "この映画は本当に素晴らしかった！"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great film!", "Terrible waste of time."]}'
```

Interactive Swagger docs at http://localhost:8000/docs when the server is running.

## Project Structure

```
MM-sentiment-classifier/
├── README.md
├── requirements.txt              # Runtime deps only (API serving)
├── requirements-dev.txt          # All deps (training, EDA, testing)
├── Dockerfile
├── .dockerignore
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 03_evaluation.ipynb             # Model comparison and evaluation
│   └── colab_train_distilbert.ipynb    # DistilBERT training (Colab GPU)
├── src/
│   ├── api.py                          # FastAPI application
│   ├── dashboard.py                    # Dashboard endpoints and prediction logging
│   ├── predict.py                      # Model loading and prediction logic
│   ├── preprocess.py                   # Text preprocessing pipeline
│   ├── translate.py                    # Offline JA→EN translation layer
│   ├── train.py                        # Traditional model training
│   ├── train_embeddings.py             # SBERT embeddings + LogReg training
│   ├── train_transformer.py            # DistilBERT fine-tuning (CPU)
│   └── static/                         # Dashboard HTML/CSS
├── models/                             # Serialized models (gitignored)
├── data/                               # Dataset (gitignored)
├── logs/                               # Prediction logs (gitignored, created at runtime)
├── docs/
│   ├── API.md                          # API endpoint documentation
│   └── MONITORING.md                   # Model degradation detection strategy
└── tests/
    ├── test_api.py
    ├── test_dashboard.py
    └── test_predict.py
```

## Documentation

- [API Reference](docs/API.md) — endpoints, request/response formats, curl examples
- [Monitoring Strategy](docs/MONITORING.md) — data drift detection, retraining pipeline, deployment strategy
- [EDA Notebook](notebooks/01_eda.ipynb) — dataset exploration and visualizations
- [Evaluation Notebook](notebooks/03_evaluation.ipynb) — full 8-model comparison with charts

## Logging

All predictions are logged to `logs/predictions.jsonl` (one JSON line per prediction with timestamp, review text, sentiment, confidence, language, and latency). Stats snapshots are written to `logs/stats_snapshots.jsonl` every 100 predictions. Application events are logged to `logs/api.log`.
