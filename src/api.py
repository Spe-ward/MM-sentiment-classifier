"""FastAPI application for sentiment classification."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import SentimentPredictor
from src.translate import translate_if_needed


# Global predictor instance
predictor: SentimentPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    predictor = SentimentPredictor()
    yield


app = FastAPI(
    title="Sentiment Classification API",
    description=(
        "Binary sentiment classification (positive/negative) for movie reviews. "
        "Supports English and Japanese input. Japanese reviews are automatically "
        "translated to English before prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request / Response schemas ---

class PredictRequest(BaseModel):
    review: str = Field(..., min_length=1, description="Movie review text (EN or JA)")

class PredictResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    language_detected: str = Field(..., description="Detected input language (ISO 639-1)")

class BatchPredictRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, description="List of movie review texts")

class BatchPrediction(BaseModel):
    review: str
    sentiment: str
    confidence: float
    language_detected: str

class BatchPredictResponse(BaseModel):
    predictions: list[BatchPrediction]

class HealthResponse(BaseModel):
    status: str
    model: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Returns the loaded model name."""
    return HealthResponse(
        status="ok",
        model=predictor.model_name if predictor else "not loaded",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictRequest):
    """Classify the sentiment of a single movie review.

    Accepts English or Japanese text. Japanese input is automatically
    detected and translated to English before prediction.
    """
    translated_text, lang = translate_if_needed(request.review)
    result = predictor.predict(translated_text)

    return PredictResponse(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        language_detected=lang,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """Classify the sentiment of multiple movie reviews.

    Each review is independently language-detected and translated if needed.
    """
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 reviews per batch request")

    predictions = []
    for review_text in request.reviews:
        translated_text, lang = translate_if_needed(review_text)
        result = predictor.predict(translated_text)
        predictions.append(BatchPrediction(
            review=review_text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            language_detected=lang,
        ))

    return BatchPredictResponse(predictions=predictions)
