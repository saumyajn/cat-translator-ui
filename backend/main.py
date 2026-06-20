"""FastAPI app for likely cat intent classification."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from audio_utils import AudioDecoderUnavailableError, AudioProcessingError, load_audio_file
from model_loader import ModelBundle, load_model_and_labels, predict_intent


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cat_model.keras"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

INTENT_MESSAGES = {
    "Food": "Your cat is probably asking for food.",
    "Isolation": "Your cat may be seeking attention or reacting to being alone.",
    "Brushing": "Your cat may be expressing happy contact or brushing-related comfort.",
}


def cors_origins() -> list[str]:
    """Return allowed CORS origins for local dev and deployed frontend."""
    origins = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ]
    frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()

    if frontend_origin and frontend_origin not in origins:
        origins.append(frontend_origin)

    return origins


def prediction_message(intent: str, top_guess: str) -> str:
    """Return careful user-facing copy for a likely intent prediction."""
    if intent == "Uncertain":
        uncertain_messages = {
            "Food": "The model is not fully sure, but your cat may be asking for food.",
            "Isolation": (
                "The model is not fully sure, but your cat may be seeking attention "
                "or reacting to being alone."
            ),
            "Brushing": (
                "The model is not fully sure, but your cat may be expressing happy "
                "contact or brushing-related comfort."
            ),
        }
        return uncertain_messages.get(
            top_guess,
            "The model is not fully sure, but it found a possible cat intent pattern.",
        )

    if intent == "Unknown":
        return "The model could not identify a clear intent. Try a clearer recording with less background noise."

    return INTENT_MESSAGES.get(
        intent,
        "Your cat's sound was classified as a likely intent, not a literal translation.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_and_labels(MODEL_PATH, LABEL_MAP_PATH)
    yield


app = FastAPI(
    title="Cat Intent Translator API",
    description="Predicts likely cat vocal intent. It does not translate exact cat language.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health."""
    return {"status": "ok"}


@app.post("/translate")
async def translate_cat_audio(file: UploadFile = File(...)) -> dict:
    """Predict likely cat intent from an uploaded audio file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file was uploaded.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio file type. Please upload a WAV or another common "
                "browser audio format."
            ),
        )

    model_bundle: ModelBundle | None = getattr(app.state, "model_bundle", None)
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)

        waveform = load_audio_file(temp_path, sample_rate=16_000, duration_seconds=3.0)
        prediction = predict_intent(model_bundle, waveform)

        return {
            "intent": prediction["intent"],
            "top_guess": prediction["top_guess"],
            "confidence": prediction["confidence"],
            "all_predictions": prediction["all_predictions"],
            "message": prediction_message(prediction["intent"], prediction["top_guess"]),
            "disclaimer": "This predicts likely intent only; it is not a literal cat-language translation.",
        }
    except AudioDecoderUnavailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AudioProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again with a clear short cat audio clip.",
        ) from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
