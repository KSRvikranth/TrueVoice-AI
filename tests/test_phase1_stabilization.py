"""Phase 1 tests: portability, audio validation, model caching, API hardening."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import config
from scripts.audio_utils import AudioValidationError, convert_to_wav, validate_wav

ROOT = Path(__file__).resolve().parent.parent
HAS_FFMPEG = bool(__import__("shutil").which("ffmpeg"))


# ---------- portability ----------
def test_no_machine_specific_paths_in_source():
    offenders = []
    for path in list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if "C:\\Users" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(path.name)
    assert not offenders, offenders


def test_requirements_is_utf8_and_has_whisper():
    raw = (ROOT / "requirements.txt").read_bytes()
    assert not raw.startswith((b"\xff\xfe", b"\xfe\xff")), "must not be UTF-16"
    text = raw.decode("utf-8")
    assert "openai-whisper" in text


def test_model_path_independent_of_cwd(tmp_path):
    code = (
        "from scripts.voice_inference import MODEL_PATH, get_model;"
        "assert MODEL_PATH.is_absolute() and MODEL_PATH.exists();"
        "get_model(); print('ok')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert res.returncode == 0, res.stderr[-400:]


def test_ffmpeg_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FFMPEG_PATH", str(tmp_path / "missing" / "ffmpeg"))
    with pytest.raises(config.FFmpegNotFoundError):
        config.find_ffmpeg()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_ffmpeg_found_on_path(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    assert Path(config.find_ffmpeg()).name.startswith("ffmpeg")


# ---------- audio validation ----------
def test_valid_audio_passes(make_wav):
    info = validate_wav(make_wav(seconds=2))
    assert info["duration_seconds"] == pytest.approx(2.0, abs=0.01)


def test_empty_audio_rejected(tmp_path):
    import numpy as np, soundfile as sf
    p = tmp_path / "e.wav"
    sf.write(str(p), np.zeros(0, dtype="float32"), 16000)
    with pytest.raises(AudioValidationError, match="empty"):
        validate_wav(p)


def test_silent_audio_rejected(make_wav):
    with pytest.raises(AudioValidationError, match="silent"):
        validate_wav(make_wav(kind="silence"))


def test_too_short_audio_rejected(make_wav):
    with pytest.raises(AudioValidationError, match="too short"):
        validate_wav(make_wav(seconds=0.1))


def test_corrupted_audio_rejected(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"This is not valid WAV audio.")
    with pytest.raises(AudioValidationError, match="corrupt"):
        validate_wav(p)


def test_validation_error_is_valueerror():
    assert issubclass(AudioValidationError, ValueError)  # keeps HTTP 422 mapping


# ---------- inference ----------
def test_predict_voice_output_format_unchanged(make_wav):
    from scripts.voice_inference import predict_voice
    res = predict_voice(make_wav(seconds=6, kind="noise"))
    # original keys must all remain; new keys may only be added
    assert {"prediction", "real_probability", "fake_probability",
            "confidence", "confidence_status"} <= set(res)
    assert res["prediction"] in {"real", "fake"}
    assert res["real_probability"] + res["fake_probability"] == pytest.approx(1.0, abs=1e-3)


def test_predict_voice_rejects_silence(make_wav):
    from scripts.voice_inference import predict_voice
    with pytest.raises(ValueError):
        predict_voice(make_wav(kind="silence"))


def test_model_loaded_only_once():
    from scripts.voice_inference import get_model
    assert get_model() is get_model()


def test_stereo_and_other_sample_rates(make_wav):
    from scripts.voice_inference import predict_voice
    for sr, ch in [(8000, 1), (44100, 2), (22050, 1)]:
        res = predict_voice(make_wav(f"x{sr}{ch}.wav", seconds=3, sr=sr, kind="noise", channels=ch))
        assert res["prediction"] in {"real", "fake"}


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("ext", ["mp3", "flac", "ogg"])
def test_conversion_of_other_formats(make_wav, tmp_path, ext):
    src = make_wav(seconds=2)
    out = tmp_path / f"o.{ext}"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), str(out)], check=True)
    wav = tmp_path / "back.wav"
    convert_to_wav(out, wav)
    assert validate_wav(wav)["sample_rate"] == 16000


# ---------- API ----------
@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    from scripts import analysis_history, analysis_report_store

    # isolate results written by the API into a temp dir
    monkeypatch.setattr(analysis_history, "RESULTS_FOLDER", tmp_path)
    monkeypatch.setattr(analysis_history, "HISTORY_FILE", tmp_path / "h.csv")
    monkeypatch.setattr(analysis_report_store, "REPORTS_FOLDER", tmp_path / "reports")

    class FakeWhisper:  # test double for TRANSCRIPTION ONLY (CNN stays real)
        def transcribe(self, path, **kw):
            return {"text": "I am calling from your bank. Share your OTP immediately."}

    monkeypatch.setattr(app_module, "get_whisper_model", lambda: FakeWhisper())
    return TestClient(app_module.app), tmp_path


def test_api_health_and_home(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "running"}
    assert c.get("/").status_code == 200


def test_api_missing_file_422(client):
    assert client[0].post("/analyze").status_code == 422


def test_api_unsupported_extension_400(client):
    r = client[0].post("/analyze", files={"file": ("notes.txt", b"x", "text/plain")})
    assert r.status_code == 400


def test_api_corrupted_wav_422(client):
    r = client[0].post("/analyze", files={"file": ("broken.wav", b"not audio", "audio/wav")})
    assert r.status_code == 422


def test_api_empty_file_422(client):
    r = client[0].post("/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 422


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_api_silent_audio_422(client, make_wav):
    c, _ = client
    with open(make_wav(kind="silence"), "rb") as f:
        r = c.post("/analyze", files={"file": ("s.wav", f, "audio/wav")})
    assert r.status_code == 422 and "silent" in r.json()["detail"].lower()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_api_full_pipeline_backward_compatible_and_single_history_row(client, make_wav):
    c, tmp = client
    with open(make_wav(seconds=6, kind="noise"), "rb") as f:
        r = c.post("/analyze", files={"file": ("call.wav", f, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ["audio_file", "voice_analysis", "transcription", "text_analysis",
                "risk_analysis", "explanation", "alert", "report_id"]:
        assert key in body
    # regression: history used to be written twice per request
    rows = (tmp / "h.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2  # header + exactly one record
    assert c.get("/history").json()["history"][0]["audio_file"] == "call.wav"
    assert c.get(f"/history-reports/{body['report_id']}").status_code == 200


def test_api_streaming_size_limit(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "MAX_UPLOAD_SIZE", 1024)
    r = client[0].post("/analyze", files={"file": ("big.wav", b"0" * 5000, "audio/wav")})
    assert r.status_code == 413


def test_api_path_traversal_report_id(client):
    c, _ = client
    assert c.get("/history-reports/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert c.get("/history-reports/nothex").status_code == 404
