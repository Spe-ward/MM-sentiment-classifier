# Sentiment Classification for Movie Reviews (EN/JA)

Binary sentiment classification (positive/negative) for movie reviews, with support for both English and Japanese input.

## Overview

This project builds and compares multiple machine learning models for movie review sentiment classification using the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The best-performing model is served via a RESTful API and containerized with Docker.

Japanese reviews are supported through an offline translation layer that detects the input language and translates JA→EN before prediction.

## Approach

### Data
- 50,000 movie reviews, perfectly balanced (25K positive / 25K negative)
- Split: 80% train (40,000) / 10% validation (5,000) / 10% test (5,000), stratified, seed=42
- Text preprocessing: HTML tag removal, lowercasing, punctuation stripping, whitespace normalization

### Models
Eight models were trained and compared — six traditional ML models using TF-IDF features (unigrams + bigrams, 50K vocabulary), one sentence-transformer embedding approach, and one fine-tuned transformer:

| Model | Accuracy | F1 | ROC-AUC | Train Time | Model Size |
|-------|----------|-----|---------|------------|------------|
| **DistilBERT** | **91.86%** | **0.9188** | **0.9754** | 2910s (GPU) | 256 MB |
| SVM (LinearSVC) | 91.72% | 0.9178 | 0.9731 | 3.5s | 1.9 MB |
| Ridge Classifier | 91.68% | 0.9175 | 0.9727 | 5.2s | 1.9 MB |
| Logistic Regression | 90.66% | 0.9078 | 0.9703 | 1.3s | 0.4 MB |
| LightGBM | 89.64% | 0.8973 | 0.9628 | 190.7s | 2.8 MB |
| Naive Bayes | 88.30% | 0.8845 | 0.9528 | 0.1s | 1.5 MB |
| Random Forest | 86.58% | 0.8669 | 0.9396 | 96.1s | 197.1 MB |
| SBERT + LogReg | 83.34% | 0.8332 | 0.9119 | 2157s | ~80 MB |

*Results on the validation set (5,000 samples). See [evaluation notebook](notebooks/03_evaluation.ipynb) for full analysis.*

### Model Selection: SVM (LinearSVC)

SVM was selected for API deployment over DistilBERT because:
- Only 0.1% F1 behind DistilBERT (0.9178 vs 0.9188) — statistically negligible
- **130x smaller** model size (1.9 MB vs 256 MB)
- **2300x faster** inference
- No GPU required at inference time
- Simpler deployment: single joblib file + TF-IDF vectorizer

### EN/JA Support
The IMDb dataset is English-only. Japanese input is handled at inference time via an offline translation layer using `argostranslate` (JA→EN) with `langdetect` for language detection.

## Implementation Steps

1. **Data Analysis** — Explored class distribution, review length, noise patterns (HTML tags in 58% of reviews), and word frequencies. See [EDA notebook](notebooks/01_eda.ipynb).
2. **Preprocessing** — Built a cleaning pipeline (HTML removal, lowercasing, punctuation stripping) shared across all models.
3. **Traditional Models** — Trained 6 models with TF-IDF features + 1 sentence-transformer approach. SVM and Ridge tied for best traditional performance.
4. **Transformer** — Fine-tuned DistilBERT on Google Colab GPU (3 epochs, 48 min). Achieved highest accuracy but marginal gain over SVM.
5. **API** — Built FastAPI service with `/predict`, `/predict/batch`, and `/health` endpoints. Includes automatic JA→EN translation.
6. **Docker** — Containerized the API with a lightweight Python 3.11 image.

## How to Run

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Kaggle API credentials (`~/.kaggle/kaggle.json`) for dataset download

### Local Development

```bash
# Clone the repository
git clone https://github.com/Spe-ward/MM-sentiment-classifier.git
cd MM-sentiment-classifier

# Create and activate virtual environment (Python 3.11)
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Download the dataset
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p data/
unzip data/imdb-dataset-of-50k-movie-reviews.zip -d data/

# Train traditional models (~5 min)
python -m src.train

# Train transformer (optional, requires GPU — see notebooks/colab_train_distilbert.ipynb)
# python -m src.train_transformer

# Run the API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build the image
docker build -t sentiment-api .

# Run the container
docker run -p 8000:8000 sentiment-api
```

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic!"}'

# Japanese input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "この映画は本当に素晴らしかった！"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great film!", "Terrible waste of time."]}'
```

## Project Structure

```
MM-sentiment-classifier/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 03_evaluation.ipynb             # Model comparison and evaluation
│   └── colab_train_distilbert.ipynb    # DistilBERT training (Colab GPU)
├── src/
│   ├── __init__.py
│   ├── api.py                          # FastAPI application
│   ├── predict.py                      # Prediction logic / model loading
│   ├── preprocess.py                   # Text preprocessing pipeline
│   ├── translate.py                    # JA→EN translation layer
│   ├── train.py                        # Traditional model training
│   ├── train_embeddings.py             # SBERT embeddings + LogReg
│   └── train_transformer.py            # DistilBERT fine-tuning (CPU)
├── models/                             # Serialized models (gitignored)
├── data/                               # Dataset (gitignored)
├── docs/
│   ├── API.md                          # API endpoint documentation
│   └── MONITORING.md                   # Model degradation strategy
└── tests/
    ├── test_api.py
    └── test_predict.py
```

## API Documentation

See [docs/API.md](docs/API.md) for full endpoint documentation with request/response examples.

Interactive docs available at http://localhost:8000/docs (Swagger UI) when the server is running.

## Model Degradation & Monitoring (Bonus)

See [docs/MONITORING.md](docs/MONITORING.md) for the proposed strategy covering:
- Data drift detection (PSI, KS-test)
- Prediction confidence monitoring
- Automated retraining pipeline
- Blue/green deployment and rollback
