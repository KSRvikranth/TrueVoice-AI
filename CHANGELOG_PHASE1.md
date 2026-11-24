# Phase 1 – Stabilization (branch `voxshield-upgrade`)

## Baseline (before changes, measured)
- `test_detector_regression.py`: 5/5 PASS; `test_risk_scoring.py`: 5/5 PASS (outputs byte-identical after changes)
- `test_edge_cases.py`: prints scenarios only (no PASS/FAIL assertions); API scripts need a live server
- All 3 `.pth` checkpoints load into `VoiceCNN`
- Empty and silent audio returned **fake @ 100% confidence** (bug)
- Inference crashed when started outside the project folder (relative `MODEL_PATH`)

## Bugs fixed
1. Hardcoded `C:\Users\sagas\...` FFmpeg path in `app.py` → `FFMPEG_PATH` env var, then `PATH`; clear 503 if missing
2. Hardcoded Windows audio paths in `test_api_pipeline.py`, `scripts/test_short_audio_windows.py`
3. Relative `MODEL_PATH` → absolute, configurable
4. CNN reloaded from disk on every request → cached, thread-safe
5. No audio validation → `AudioValidationError` (HTTP 422) for empty/corrupt/short/long/silent
6. Upload limit trusted `Content-Length` only → enforced while streaming (HTTP 413)
7. `/analyze` saved each analysis to history twice → once
8. `/history` and `/history/export` read a cwd-relative file different from where history is written
9. `whisper` missing from `requirements.txt`; imported at module top (API could not start without it) → added, lazy import, clear 503
10. `requirements.txt` was UTF-16 → UTF-8

## Preserved
All endpoints, response keys, risk scoring, text analysis, trusted-contact flow, templates, all model files.

## New files
`scripts/config.py`, `scripts/audio_utils.py`, `.env.example`, `.gitattributes`, `requirements-dev.txt`, `tests/` (27 tests)

## Not verified here (honest limits)
- Whisper transcription was **not** run in my sandbox (package/model download unavailable); API tests replace *only* transcription with a stub. The CNN, validation, FFmpeg, risk and history code run for real.
- No model accuracy is claimed. Real evaluation (Phase 2) needs ASVspoof/WaveFake data, which is not in the repo.
- Observed risk to address in Phase 2/4: the CNN outputs ~100% "fake" for synthetic tones/noise, so its probabilities are not calibrated evidence.
