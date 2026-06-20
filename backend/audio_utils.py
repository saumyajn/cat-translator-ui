"""Audio preprocessing helpers for cat intent inference."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np


class AudioProcessingError(ValueError):
    """Raised when an uploaded audio file cannot be processed."""


class AudioDecoderUnavailableError(RuntimeError):
    """Raised when the server is missing the required audio decoder."""


def pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or trim audio to exactly target_samples."""
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")

    if audio.ndim != 1:
        audio = np.asarray(audio).reshape(-1)

    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
    else:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


def convert_to_wav(
    input_path: str | Path,
    sample_rate: int = 16_000,
) -> Path:
    """Convert uploaded browser audio to mono WAV with ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise AudioDecoderUnavailableError(
            "FFmpeg is not installed or is not available on PATH. Install FFmpeg "
            "and restart the backend so browser-recorded audio can be decoded."
        )

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = Path(output_file.name)
    output_file.close()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        output_path.unlink(missing_ok=True)
        raise AudioProcessingError("Audio decoding timed out. Try a shorter recording.") from exc

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        error_text = (result.stderr or result.stdout or "").strip()
        detail = error_text.splitlines()[-1] if error_text else "unknown ffmpeg error"
        raise AudioProcessingError(
            f"Could not decode the uploaded audio file with FFmpeg: {detail}"
        )

    return output_path


def load_audio_file(
    path: str | Path,
    sample_rate: int = 16_000,
    duration_seconds: float = 3.0,
) -> np.ndarray:
    """Convert input to mono WAV, then load and standardize it to 3 seconds."""
    target_samples = int(sample_rate * duration_seconds)
    wav_path: Path | None = None

    try:
        wav_path = convert_to_wav(path, sample_rate=sample_rate)
        audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    except Exception as exc:
        if isinstance(exc, (AudioProcessingError, AudioDecoderUnavailableError)):
            raise

        raise AudioProcessingError(
            "Could not read the uploaded audio file. Please upload a valid audio file."
        ) from exc
    finally:
        if wav_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)

    if audio.size == 0:
        raise AudioProcessingError("The uploaded audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise AudioProcessingError("The uploaded audio contains invalid numeric values.")

    return pad_or_trim(audio, target_samples)
