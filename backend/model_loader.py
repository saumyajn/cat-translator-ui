"""Model loading and prediction helpers for the cat intent classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
CONFIDENT_THRESHOLD = 0.75
UNCERTAIN_THRESHOLD = 0.45


@dataclass(frozen=True)
class ModelBundle:
    """Loaded classifier, YAMNet embedding model, and label map."""

    classifier: tf.keras.Model
    yamnet: object
    label_map: dict[int, str]


def load_model_and_labels(
    model_path: str | Path = "cat_model.keras",
    label_map_path: str | Path = "label_map.json",
) -> ModelBundle:
    """Load the Keras classifier, TensorFlow Hub YAMNet, and label map."""
    model_path = Path(model_path)
    label_map_path = Path(label_map_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")

    classifier = tf.keras.models.load_model(model_path)
    yamnet = hub.load(YAMNET_HANDLE)

    with label_map_path.open("r", encoding="utf-8") as file:
        raw_label_map = json.load(file)

    label_map = {int(index): str(label) for index, label in raw_label_map.items()}

    if not label_map:
        raise ValueError("Label map is empty.")

    return ModelBundle(classifier=classifier, yamnet=yamnet, label_map=label_map)


def _waveform_to_mean_embedding(bundle: ModelBundle, waveform: np.ndarray) -> np.ndarray:
    """Convert one waveform into one averaged YAMNet embedding vector."""
    scores, embeddings, spectrogram = bundle.yamnet(waveform.astype(np.float32))
    del scores, spectrogram
    mean_embedding = tf.reduce_mean(embeddings, axis=0)
    return mean_embedding.numpy().astype(np.float32)


def predict_intent(bundle: ModelBundle, waveform: np.ndarray) -> dict:
    """Predict likely cat intent from a preprocessed waveform."""
    embedding = _waveform_to_mean_embedding(bundle, waveform)
    model_input = np.expand_dims(embedding, axis=0)
    print(f"Model input shape before prediction: {model_input.shape}")
    probabilities = bundle.classifier.predict(
        model_input,
        verbose=0,
    )[0]

    probabilities = np.asarray(probabilities, dtype=np.float32)
    top_index = int(np.argmax(probabilities))
    top_label = bundle.label_map.get(top_index, f"Class {top_index}")
    top_confidence = float(probabilities[top_index])

    all_predictions = [
        {
            "label": bundle.label_map.get(index, f"Class {index}"),
            "confidence": round(float(probabilities[index]), 4),
        }
        for index in range(len(probabilities))
    ]
    all_predictions.sort(key=lambda item: item["confidence"], reverse=True)

    if top_confidence >= CONFIDENT_THRESHOLD:
        intent = top_label
    elif top_confidence >= UNCERTAIN_THRESHOLD:
        intent = "Uncertain"
    else:
        intent = "Unknown"

    return {
        "intent": intent,
        "top_guess": top_label,
        "confidence": round(top_confidence, 4),
        "all_predictions": all_predictions,
    }
