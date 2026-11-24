# VoxShield AI - Model Evaluation Report
*Generated:* 2026-11-22T05:11:29+00:00  
*Status:* **NOT_EVALUATED**

## Evaluation has NOT been completed

Evaluation manifest not found: /home/claude/voice-cloning-detection-main/results/multisource_test.csv

### How to run it
```
# 1. Put your evaluation data on disk (e.g. ASVspoof 2019 LA eval, WaveFake) and build a manifest:
python -m scripts.prepare_eval_manifest asvspoof --protocol <ASVspoof protocol .txt> --audio-dir <flac dir> --out results/asvspoof_eval.csv
python -m scripts.prepare_eval_manifest wavefake --wavefake-root <dir> --real-dir <LJSpeech wavs> --out results/wavefake_eval.csv
# 2. Evaluate:
python -m scripts.evaluate_model --manifest results/asvspoof_eval.csv --name asvspoof2019_la_eval --compare-preprocessing
```
