"""FastAPI application for sentiment classification."""

import logging
import time as time_module
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import SentimentPredictor
from src.translate import translate_if_needed
from src.dashboard import router as dashboard_router, tracker, LOG_DIR

# --- Logging setup ---
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler(),
    ],
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# Global predictor instance
predictor: SentimentPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    start = time_module.perf_counter()
    predictor = SentimentPredictor()
    load_time = time_module.perf_counter() - start
    log.info("Model loaded: %s (%.2fs)", predictor.model_name, load_time)
    yield
    log.info("Application shutting down")


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

# Include dashboard API routes
app.include_router(dashboard_router)


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

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/static/index.html")


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
    start = time_module.perf_counter()
    translated_text, lang = translate_if_needed(request.review)
    result = predictor.predict(translated_text)
    elapsed_ms = (time_module.perf_counter() - start) * 1000

    tracker.record(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        language=lang,
        review=request.review,
        model=predictor.model_name,
        response_time_ms=elapsed_ms,
        translated=(lang != "en"),
    )

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
        start = time_module.perf_counter()
        translated_text, lang = translate_if_needed(review_text)
        result = predictor.predict(translated_text)
        elapsed_ms = (time_module.perf_counter() - start) * 1000

        tracker.record(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            language=lang,
            review=review_text,
            model=predictor.model_name,
            response_time_ms=elapsed_ms,
            translated=(lang != "en"),
        )
        predictions.append(BatchPrediction(
            review=review_text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            language_detected=lang,
        ))

    return BatchPredictResponse(predictions=predictions)


# Mount static files AFTER all routes are registered
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
