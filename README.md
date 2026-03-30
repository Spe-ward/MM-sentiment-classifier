# Sentiment Classification for Movie Reviews (EN/JA)

Binary sentiment classification for movie reviews using the [IMDb 50K dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Serves predictions via REST API with a web dashboard. Supports English and Japanese input.

## How It Works

1. Text is cleaned (HTML removal, lowercasing, punctuation stripping)
2. Japanese input is detected and translated to English offline via `argostranslate`
3. The review is vectorized with TF-IDF (unigrams + bigrams, 50K features)
4. An SVM classifier predicts positive or negative sentiment with a confidence score

## Model Comparison

Eight models were trained on 40,000 reviews and evaluated on a 5,000-sample validation set:

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

**Deployed model: SVM** — within 0.1% F1 of DistilBERT, 130x smaller, no GPU required. Full analysis in the [evaluation notebook](notebooks/03_evaluation.ipynb).

## Quick Start

### Docker (recommended)

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

Open http://localhost:8000 for the web dashboard, or http://localhost:8000/docs for the Swagger UI.

### Local

```bash
git clone https://github.com/Spe-ward/MM-sentiment-classifier.git
cd MM-sentiment-classifier

python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# source venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

> **Note:** `requirements.txt` contains runtime dependencies only. To retrain models or run notebooks, install `requirements-dev.txt` instead and see [Training](#training) below.

## API Examples

```bash
# Health check
curl http://localhost:8000/health

# Predict sentiment (English)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic!"}'

# Predict sentiment (Japanese — auto-translated)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "この映画は本当に素晴らしかった！"}'

# Batch prediction (up to 100 reviews)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great film!", "Terrible waste of time."]}'
```

See [docs/API.md](docs/API.md) for full endpoint reference including dashboard API endpoints.

## Training

To retrain models from scratch:

```bash
pip install -r requirements-dev.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Download dataset (requires Kaggle API credentials in ~/.kaggle/kaggle.json)
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p data/
unzip data/imdb-dataset-of-50k-movie-reviews.zip -d data/

# Train traditional models (~5 min)
python -m src.train

# Train SBERT embeddings model (~35 min on CPU)
python -m src.train_embeddings

# Train DistilBERT (use the Colab notebook for GPU — see notebooks/colab_train_distilbert.ipynb)
```

## Project Structure

```
├── requirements.txt            Runtime dependencies (API serving)
├── requirements-dev.txt        Full dependencies (training, notebooks, testing)
├── Dockerfile
│
├── src/
│   ├── api.py                  FastAPI application and prediction endpoints
│   ├── dashboard.py            Dashboard API, prediction tracking, and logging
│   ├── predict.py              Model loading and inference
│   ├── preprocess.py           Text cleaning pipeline
│   ├── translate.py            Offline JA→EN translation (argostranslate)
│   ├── train.py                Traditional model training (6 models)
│   ├── train_embeddings.py     SBERT embeddings + LogReg
│   ├── train_transformer.py    DistilBERT fine-tuning
│   └── static/                 Web dashboard (HTML/CSS)
│
├── notebooks/
│   ├── 01_eda.ipynb            Exploratory data analysis
│   ├── 03_evaluation.ipynb     8-model comparison and error analysis
│   └── colab_train_distilbert.ipynb
│
├── docs/
│   ├── API.md                  API endpoint documentation
│   └── MONITORING.md           Degradation detection and redeployment strategy
│
├── tests/                      pytest test suite (23 tests)
├── models/                     Serialized model artifacts (gitignored)
├── data/                       IMDb dataset (gitignored)
└── logs/                       Prediction logs (gitignored, created at runtime)
```

## Logging

All predictions are persistently logged to `logs/predictions.jsonl` with timestamp, review text, detected language, sentiment, confidence, model name, and response latency. Aggregate snapshots are written to `logs/stats_snapshots.jsonl` every 100 predictions. The web dashboard at http://localhost:8000 displays these statistics live.

## Documentation

- [API Reference](docs/API.md) — all endpoints, request/response schemas, curl examples
- [Monitoring Strategy](docs/MONITORING.md) — degradation detection, automated retraining, redeployment
- [EDA Notebook](notebooks/01_eda.ipynb) — dataset exploration and visualizations
- [Evaluation Notebook](notebooks/03_evaluation.ipynb) — model comparison with confusion matrices, ROC curves, and error analysis
