# Phase 2 – Model evaluation, uncertainty handling, measured performance

## Headline (read this first)
**The model's accuracy has NOT been measured, and this report does not claim any.**
No dataset ships with the repository (`dataset/`, `results/*.csv` are git-ignored) and
ASVspoof / WaveFake could not be downloaded from my sandbox (network restricted).
What *is* delivered and verified: a complete evaluation pipeline (51 passing tests,
metric formulas checked against scikit-learn and an independent brute-force EER), the
single command to produce real numbers on your machine, uncertain-prediction handling,
and real before/after latency measurements.
The generated report currently says **NOT_EVALUATED**, with run instructions
(`results/evaluation/evaluation_report.md`) – by design, never a placeholder score.

## What was added
| Item | File |
|---|---|
| Metrics: accuracy, precision, recall, F1, confusion matrix, FPR, FNR, ROC-AUC, EER (+threshold) | `scripts/evaluation.py` |
| 95 % bootstrap confidence intervals, per-source/dataset/language breakdown | same |
| Operating points: threshold for 1 % / 5 % / 10 % false-positive budgets | same |
| Data-leakage check (files, real speakers, fake synthesis sources) | same |
| Report generator (JSON + Markdown + per-file predictions CSV); status EVALUATED / NOT_EVALUATED / SMOKE_TEST | same |
| Evaluation CLI using the **production** preprocessing; optional trim vs no-trim comparison | `scripts/evaluate_model.py` |
| ASVspoof 2019 LA and WaveFake manifest adapters (keeps speaker + synthesis-system IDs for speaker/system-disjoint analysis) | `scripts/prepare_eval_manifest.py` |
| Latency benchmark | `scripts/benchmark_inference.py` |
| `GET /evaluation` endpoint (dashboard hook; returns NOT_EVALUATED until a run exists) | `app.py` |

## Uncertain-prediction handling (behaviour change – please review)
* `predict_voice` gains an **additive** field `verdict`: `genuine` / `synthetic` / `uncertain`
  (uncertain = confidence < 0.65, same cut-off the code already used for `UNCERTAIN`).
  `prediction`, `confidence` and all original keys are unchanged.
* Risk scoring: a *fake* prediction with `UNCERTAIN` confidence now adds **5** points
  (signal `voice_model_uncertain`) instead of 20. Confident predictions and inputs without
  a confidence field score exactly as before (existing test outputs are byte-identical).
  An uncertain voice result can never push a call to HIGH on its own.
* Thresholds are configurable (`VOXSHIELD_FAKE_THRESHOLD`, `VOXSHIELD_UNCERTAIN_BELOW`);
  defaults reproduce the original behaviour. Pick a non-default threshold only from the
  validation-set operating points the report produces, never from the test set.

## Finding to act on
`scripts/evaluate_multisource_cnn.py` does **not** trim silence, whereas production
inference does (`top_db=30`). If training used no trimming, production preprocessing may
be mismatched to the model. Run `--compare-preprocessing` on your data: it scores the same
files both ways and tells you which is better, with numbers.

## Measured results
* Tests: **51 passed** (27 Phase 1 + 24 Phase 2); original `test_detector_regression.py`,
  `test_risk_scoring.py`, `test_edge_cases.py` output byte-identical to the baseline.
* Latency (1 vCPU sandbox, CPU): see `results/benchmarks/latency_comparison.md`.
  Single-request mean 12.45 → 11.89 ms (noise-level); 4-thread throughput 66.9 → 103.0 req/s (+54 %).
* Earlier observation, still true: the CNN returns ~100 % "fake" for synthetic tones and
  noise – another reason to evaluate on real corpora before trusting its probabilities.

## Run it on your data (Windows PowerShell, project root, venv active)
```powershell
# Your existing split (created by create_multisource_split.py) – no download needed:
python -m scripts.evaluate_model --manifest results\multisource_test.csv `
       --train-manifest results\multisource_train.csv `
       --validation-manifest results\multisource_validation.csv --compare-preprocessing

# Independent corpus (ASVspoof 2019 LA eval):
python -m scripts.prepare_eval_manifest asvspoof --protocol <protocol.txt> --audio-dir <flac_dir> --out results\asvspoof_eval.csv
python -m scripts.evaluate_model --manifest results\asvspoof_eval.csv --name asvspoof2019_la_eval --compare-preprocessing
```
Output: `results\evaluation\evaluation_<name>.md/.json` + `_predictions.csv`.
Only an independent corpus (unseen speakers *and* unseen generators) says anything about
generalisation; your own split mostly measures in-distribution performance – the leakage
section of the report states which one you have.

## Still open (later phases)
AASIST / Wav2Vec2 comparison (only worthwhile once the baseline numbers exist), overlapping
chunking for long audio, real-time mode, speaker verification, real notifications, dashboard.
