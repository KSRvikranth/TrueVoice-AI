"""Audio conversion and validation helpers (portable, no hardcoded paths)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts import config


class AudioValidationError(ValueError):
    """The audio is empty, corrupted, too short/long, or silent.

    Subclasses ValueError so existing `except ValueError` handlers
    (HTTP 422 in app.py) keep working unchanged.
    """


def convert_to_wav(input_path, output_path, sample_rate: int = 16000) -> None:
    """Convert any FFmpeg-readable file to mono 16 kHz PCM WAV."""
    ffmpeg = config.find_ffmpeg()
    command = [
        ffmpeg, "-nostdin", "-y", "-v", "error",
        "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(output_path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True,
        timeout=config.FFMPEG_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise AudioValidationError(
            "Audio conversion failed. The file may be corrupted or use an "
            "unsupported codec."
        )
    out = Path(output_path)
    if not out.exists() or out.stat().st_size == 0:
        raise AudioValidationError("Converted WAV file was not created.")


def validate_wav(wav_path) -> dict:
    """Check a WAV file is readable, long enough and not silent.

    Returns basic stats; raises AudioValidationError with a clear message.
    """
    try:
        info = sf.info(str(wav_path))
    except Exception as exc:  # soundfile raises RuntimeError / LibsndfileError
        raise AudioValidationError(
            "The audio file could not be read. It may be corrupted."
        ) from exc

    if info.frames == 0 or info.samplerate <= 0:
        raise AudioValidationError("The audio file is empty.")

    duration = info.frames / info.samplerate
    if duration < config.MIN_AUDIO_SECONDS:
        raise AudioValidationError(
            f"Audio is too short ({duration:.2f}s). Minimum is "
            f"{config.MIN_AUDIO_SECONDS:g}s."
        )
    if duration > config.MAX_AUDIO_SECONDS:
        raise AudioValidationError(
            f"Audio is too long ({duration:.0f}s). Maximum is "
            f"{config.MAX_AUDIO_SECONDS}s."
        )

    data, _ = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if not np.isfinite(data).all():
        raise AudioValidationError("Audio contains invalid sample values.")
    rms = float(np.sqrt(np.mean(np.square(data))))
    if rms < config.SILENCE_RMS_THRESHOLD:
        raise AudioValidationError(
            "The audio is silent. No speech could be analysed."
        )

    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "rms": rms,
    }
