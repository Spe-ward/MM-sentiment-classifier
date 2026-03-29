# Sentiment Classification for Movie Reviews (EN/JA)

Binary sentiment classification (positive/negative) for movie reviews, with support for both English and Japanese input.

## Overview

This project builds and compares multiple machine learning models for movie review sentiment classification using the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The best-performing model is served via a RESTful API and containerized with Docker.

Japanese reviews are supported through an offline translation layer that detects the input language and translates JA→EN before prediction.

## Approach

### Data
- 50,000 movie reviews, perfectly balanced (25K positive / 25K negative)
- Split: 80% train (40,000) / 10% validation (5,000) / 10% test (5,000)
- Text preprocessing: HTML tag removal, lowercasing, punctuation stripping, whitespace normalization

### Models
Seven models were trained and compared — six traditional ML models using TF-IDF features (unigrams + bigrams, 50K vocabulary), plus one fine-tuned transformer:

| Model | Accuracy | F1 | ROC-AUC | Train Time | Model Size |
|-------|----------|-----|---------|------------|------------|
| **SVM (LinearSVC)** | **91.72%** | **0.9178** | **0.9731** | 3.5s | 1.9 MB |
| Ridge Classifier | 91.68% | 0.9175 | 0.9727 | 5.2s | 1.9 MB |
| Logistic Regression | 90.66% | 0.9078 | 0.9703 | 1.3s | 0.4 MB |
| LightGBM | 89.64% | 0.8973 | 0.9628 | 190.7s | 2.8 MB |
| Naive Bayes | 88.30% | 0.8845 | 0.9528 | 0.1s | 1.5 MB |
| Random Forest | 86.58% | 0.8669 | 0.9396 | 96.1s | 197.1 MB |
| DistilBERT | *training in progress* | — | — | — | ~260 MB |

*Results shown on the validation set. Final model selection and test set evaluation pending.*

### EN/JA Support
The IMDb dataset is English-only. Japanese input is handled at inference time via an offline translation layer using `argostranslate` (JA→EN) with `langdetect` for language detection.

## Current Status

- [x] Phase 0: Project setup
- [x] Phase 1: Exploratory data analysis ([notebook](notebooks/01_eda.ipynb))
- [x] Phase 2a: Traditional ML models trained and evaluated
- [ ] Phase 2b: DistilBERT transformer (training in progress)
- [ ] Phase 3: Model evaluation and comparison
- [ ] Phase 4: REST API with JA translation
- [ ] Phase 5: Docker containerization
- [ ] Phase 6: README finalization
- [ ] Phase 7: Model degradation detection strategy (bonus)

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

# Train traditional models
python -m src.train

# Train transformer (CPU, takes ~1hr)
python -m src.train_transformer

# Run the API (coming soon)
# uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Docker

<!-- TODO: Fill in after Dockerfile is created -->

## Project Structure

```
MM-sentiment-classifier/
├── README.md
├── requirements.txt
├── Dockerfile                    # (coming soon)
├── .dockerignore                 # (coming soon)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_model_development.ipynb # (coming soon)
│   └── 03_evaluation.ipynb       # (coming soon)
├── src/
│   ├── __init__.py
│   ├── api.py                    # (coming soon) FastAPI application
│   ├── predict.py                # (coming soon) Prediction logic
│   ├── preprocess.py             # Text preprocessing pipeline
│   ├── train.py                  # Traditional model training
│   └── train_transformer.py      # DistilBERT fine-tuning
├── models/                       # Serialized models (gitignored)
├── data/                         # Dataset (gitignored)
├── docs/
│   ├── API.md                    # (coming soon)
│   └── MONITORING.md             # (coming soon)
└── tests/
    ├── test_api.py               # (coming soon)
    └── test_predict.py           # (coming soon)
```

## API Documentation

See [docs/API.md](docs/API.md) for full endpoint documentation with examples. *(coming soon)*

## Model Degradation & Monitoring (Bonus)

See [docs/MONITORING.md](docs/MONITORING.md) for the proposed monitoring and redeployment strategy. *(coming soon)*
