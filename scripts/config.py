"""Central, portable configuration for VoxShield AI.

All paths are resolved relative to the project root (never the current
working directory) and every environment-specific value can be overridden
with an environment variable or a `.env` file. No third-party dependency
is needed: a tiny `.env` parser is included.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# ----- Paths -----
MODELS_DIR = _path_env("VOXSHIELD_MODELS_DIR", PROJECT_ROOT / "models")
RESULTS_DIR = _path_env("VOXSHIELD_RESULTS_DIR", PROJECT_ROOT / "results")
MODEL_PATH = _path_env(
    "VOXSHIELD_MODEL_PATH", MODELS_DIR / "voice_cnn_multisource.pth"
)

# ----- Upload limits -----
MAX_UPLOAD_MB = _int_env("VOXSHIELD_MAX_UPLOAD_MB", 25)
MAX_UPLOAD_SIZE = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".mpeg", ".ogg", ".flac", ".m4a", ".aac", ".webm",
}

# ----- Audio validation -----
MIN_AUDIO_SECONDS = float(os.environ.get("VOXSHIELD_MIN_AUDIO_SECONDS", "0.5"))
MAX_AUDIO_SECONDS = _int_env("VOXSHIELD_MAX_AUDIO_SECONDS", 600)
SILENCE_RMS_THRESHOLD = float(os.environ.get("VOXSHIELD_SILENCE_RMS", "1e-4"))
FFMPEG_TIMEOUT_SECONDS = _int_env("VOXSHIELD_FFMPEG_TIMEOUT", 120)

# ----- Decision thresholds (defaults reproduce the original behaviour) -----
# prediction = "fake" when P(fake) >= FAKE_THRESHOLD (0.5 == original rule)
FAKE_THRESHOLD = float(os.environ.get("VOXSHIELD_FAKE_THRESHOLD", "0.5"))
# confidence below this is reported as UNCERTAIN / verdict "uncertain"
UNCERTAIN_BELOW = float(os.environ.get("VOXSHIELD_UNCERTAIN_BELOW", "0.65"))
MODERATE_BELOW = float(os.environ.get("VOXSHIELD_MODERATE_BELOW", "0.80"))
EVALUATION_DIR = _path_env("VOXSHIELD_EVALUATION_DIR", RESULTS_DIR / "evaluation")

# ----- Whisper -----
WHISPER_MODEL_NAME = os.environ.get("VOXSHIELD_WHISPER_MODEL", "base")


class FFmpegNotFoundError(RuntimeError):
    """Raised when no usable FFmpeg executable can be located."""


def find_ffmpeg() -> str:
    """Locate FFmpeg: FFMPEG_PATH env var first, then the system PATH."""
    configured = os.environ.get("FFMPEG_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            candidate = candidate / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if candidate.is_file():
            return str(candidate)
        raise FFmpegNotFoundError(
            f"FFMPEG_PATH is set to '{configured}' but no ffmpeg executable "
            "was found there."
        )
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FFmpegNotFoundError(
        "FFmpeg was not found. Install it and add it to PATH, or set the "
        "FFMPEG_PATH environment variable (see .env.example)."
    )
