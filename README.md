# Sentiment Classification for Movie Reviews (EN/JA)

Binary sentiment classification (positive/negative) for movie reviews, with support for both English and Japanese input.

## Overview

This project builds and compares multiple machine learning models for movie review sentiment classification using the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The best-performing model is served via a RESTful API and containerized with Docker.

Japanese reviews are supported through an offline translation layer that detects the input language and translates JA→EN before prediction.

## Approach

<!-- TODO: Fill in after model development and evaluation -->

## Implementation Steps

<!-- TODO: Fill in after all phases complete -->

## How to Run

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Kaggle API credentials (`~/.kaggle/kaggle.json`) for dataset download

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd Macromill

# Create and activate virtual environment (Python 3.11)
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download the dataset
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p data/
unzip data/imdb-dataset-of-50k-movie-reviews.zip -d data/

# Run the API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Docker

<!-- TODO: Fill in after Dockerfile is created -->

## Project Structure

```
Macromill/
├── README.md
├── PLAN.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── models/
├── data/
├── docs/
│   ├── API.md
│   └── MONITORING.md
└── tests/
    ├── test_api.py
    └── test_predict.py
```

## API Documentation

See [docs/API.md](docs/API.md) for full endpoint documentation with examples.

## Model Degradation & Monitoring (Bonus)

See [docs/MONITORING.md](docs/MONITORING.md) for the proposed monitoring and redeployment strategy.
