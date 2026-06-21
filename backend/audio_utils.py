"""Audio preprocessing helpers for cat intent inference."""

from __future__ import annotations

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
    """Convert any uploaded audio to mono 16 kHz WAV with imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise AudioDecoderUnavailableError(
            "imageio-ffmpeg is not installed. Run pip install -r requirements.txt "
            "and restart the backend."
        ) from exc

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = Path(output_file.name)
    output_file.close()
    print(f"Temporary converted wav path: {output_path}")

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
        detail_lines = error_text.splitlines()
        detail = "\n".join(detail_lines[-5:]) if detail_lines else "unknown ffmpeg error"
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
        print(f"Decoded audio shape before pad/trim: {audio.shape}")
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

    processed_audio = pad_or_trim(audio, target_samples)
    print(f"Audio array shape after pad/trim: {processed_audio.shape}")
    return processed_audio
