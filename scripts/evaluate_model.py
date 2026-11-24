"""Evaluate the production CNN on a labelled manifest and write a report.

Usage (from the project root):
    python -m scripts.evaluate_model
    python -m scripts.evaluate_model --manifest results/multisource_test.csv \
        --train-manifest results/multisource_train.csv --compare-preprocessing

Manifest CSV columns: file,label[,dataset,language,speaker,source]
  label is "real" or "fake"; `file` is relative to the project root or absolute.
Uses the SAME preprocessing as production (`voice_inference.prepare_audio`).
If the manifest or audio is missing, a NOT_EVALUATED report is written - no
numbers are ever invented.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from scripts import config
from scripts import evaluation as ev

DEFAULT_MANIFEST = config.RESULTS_DIR / "multisource_test.csv"
DEFAULT_TRAIN = config.RESULTS_DIR / "multisource_train.csv"
DEFAULT_VALIDATION = config.RESULTS_DIR / "multisource_validation.csv"

HOW_TO = [
    "# 1. Put your evaluation data on disk (e.g. ASVspoof 2019 LA eval, WaveFake) and build a manifest:",
    "python -m scripts.prepare_eval_manifest asvspoof --protocol <ASVspoof protocol .txt> --audio-dir <flac dir> --out results/asvspoof_eval.csv",
    "python -m scripts.prepare_eval_manifest wavefake --wavefake-root <dir> --real-dir <LJSpeech wavs> --out results/wavefake_eval.csv",
    "# 2. Evaluate:",
    "python -m scripts.evaluate_model --manifest results/asvspoof_eval.csv --name asvspoof2019_la_eval --compare-preprocessing",
]


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for needed in ("file", "label"):
        if rows and needed not in rows[0]:
            raise ValueError(f"Manifest {path} is missing required column '{needed}'.")
    bad = {r["label"] for r in rows} - {"real", "fake"}
    if bad:
        raise ValueError(f"Unknown labels in manifest: {sorted(bad)} (use 'real' or 'fake').")
    return rows


def resolve(file_value):
    p = Path(file_value.replace("\\", "/"))
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def score_files(rows, trim=True, batch_size=32):
    """Return (kept_rows, y_true, fake_scores, skipped) using production preprocessing."""
    import torch
    from scripts.audio_utils import validate_wav
    from scripts.voice_inference import DEVICE, get_model, prepare_audio

    model = get_model()
    kept, y_true, scores, skipped = [], [], [], []
    batch, batch_rows = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            probs = torch.softmax(model(torch.cat(batch).to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        for r, p in zip(batch_rows, probs):
            kept.append(r); y_true.append(1 if r["label"] == "fake" else 0); scores.append(float(p))
        batch.clear(); batch_rows.clear()

    for i, row in enumerate(rows, 1):
        path = resolve(row["file"])
        try:
            # evaluation reads raw files (wav/flac); validate like production does
            validate_wav(path)
            batch.append(prepare_audio(path, trim=trim)); batch_rows.append(row)
        except Exception as exc:  # recorded, never silently dropped
            skipped.append(f"{row['file']}: {exc}")
        if len(batch) >= batch_size:
            flush()
        if i % 200 == 0:
            print(f"  scored {i}/{len(rows)}", flush=True)
    flush()
    return kept, np.array(y_true), np.array(scores), skipped


def build_report(name, rows, y, s, skipped, threshold, train_rows=None, extra=None, status=ev.STATUS_EVALUATED):
    metrics = ev.compute_metrics(y, s, threshold)
    report = {
        "status": status,
        "generated_at": ev.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": {"evaluation_set": name, "model": str(config.MODEL_PATH.name), "threshold": threshold,
                "files_scored": int(len(y)), "files_in_manifest": len(rows) + len(skipped)},
        "metrics": metrics,
        "confidence_intervals": ev.bootstrap_ci(y, s, threshold),
        "uncertain_handling": ev.selective_metrics(
            y, s, low=1 - config.UNCERTAIN_BELOW, high=config.UNCERTAIN_BELOW, threshold=threshold),
        "operating_points": [ev.operating_point(y, s, t) for t in (0.01, 0.05, 0.10)],
        "by_source": ev.group_breakdown(rows, y, s, "source", threshold),
        "by_dataset": ev.group_breakdown(rows, y, s, "dataset", threshold),
        "by_language": ev.group_breakdown(rows, y, s, "language", threshold),
        "leakage_check": ev.check_leakage(train_rows, rows) if train_rows else
            {"checked": False, "reason": "No training manifest supplied (--train-manifest)."},
        "skipped_files": {"count": len(skipped), "examples": skipped[:3]} if skipped else None,
    }
    if extra:
        report.update(extra)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--train-manifest", type=Path, default=None)
    ap.add_argument("--validation-manifest", type=Path, default=None,
                    help="Optional: operating thresholds are then chosen on THIS set, not the test set.")
    ap.add_argument("--name", default=None, help="Label for the report/filenames.")
    ap.add_argument("--threshold", type=float, default=config.FAKE_THRESHOLD)
    ap.add_argument("--compare-preprocessing", action="store_true",
                    help="Also score with silence-trimming disabled and report both.")
    ap.add_argument("--out-dir", type=Path, default=config.EVALUATION_DIR)
    ap.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    name = args.name or args.manifest.stem
    stem = "evaluation_report" if args.name is None else f"evaluation_{args.name}"

    if not args.manifest.exists():
        rep = ev.not_evaluated_report(f"Evaluation manifest not found: {args.manifest}", HOW_TO)
        j, m = ev.write_report(rep, args.out_dir, stem)
        print(f"NOT EVALUATED - manifest missing: {args.manifest}\nWrote {m}")
        return 2

    rows_all = read_manifest(args.manifest)
    if not rows_all:
        print("Manifest is empty."); return 2

    train_rows = read_manifest(args.train_manifest) if args.train_manifest and args.train_manifest.exists() else None
    print(f"Scoring {len(rows_all)} files with {config.MODEL_PATH.name} (production preprocessing)...")
    t0 = time.time()
    rows, y, s, skipped = score_files(rows_all, trim=True)
    if len(y) == 0:
        rep = ev.not_evaluated_report("No audio file in the manifest could be loaded. " + "; ".join(skipped[:3]), HOW_TO)
        ev.write_report(rep, args.out_dir, stem); print("NOT EVALUATED - no loadable audio."); return 2

    extra = {}
    if args.compare_preprocessing:
        rows2, y2, s2, _ = score_files(rows_all, trim=False)
        extra["preprocessing_comparison"] = {
            "production (trim silence top_db=30)": ev.compute_metrics(y, s, args.threshold),
            "no trim": ev.compute_metrics(y2, s2, args.threshold)}

    report = build_report(name, rows, y, s, skipped, args.threshold, train_rows, extra,
                          status=ev.STATUS_SMOKE_TEST if args.smoke_test else ev.STATUS_EVALUATED)
    report["run"]["elapsed_seconds"] = round(time.time() - t0, 1)

    if args.validation_manifest and args.validation_manifest.exists():
        vr, vy, vs, _ = score_files(read_manifest(args.validation_manifest))
        report["operating_points_from_validation"] = [ev.operating_point(vy, vs, t) for t in (0.01, 0.05, 0.10)]

    # per-sample predictions so every number can be reproduced/audited
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / f"{stem}_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["file", "label", "fake_probability"])
        for r, sc in zip(rows, s):
            w.writerow([r["file"], r["label"], round(float(sc), 6)])

    j, m = ev.write_report(report, args.out_dir, stem)
    mt = report["metrics"]
    print(f"\nAccuracy {mt['accuracy']}  F1 {mt['f1_fake']}  AUC {mt['roc_auc']}  EER {mt['eer']}  "
          f"FPR {mt['false_positive_rate']}  FNR {mt['false_negative_rate']}")
    print(f"Report: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
