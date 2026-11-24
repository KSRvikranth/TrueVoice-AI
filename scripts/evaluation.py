"""Model-evaluation metrics, leakage checks and report generation.

Pure numpy/scikit-learn (no torch) so it is fast to import and easy to test.
Conventions:  label 1 = FAKE (synthetic/cloned, the "positive" class),
              label 0 = REAL (bona fide).  score = P(fake).

Honesty rules enforced here:
  * Every report carries a `status`: EVALUATED, NOT_EVALUATED or SMOKE_TEST.
  * Metrics are only ever computed from supplied predictions - nothing is
    defaulted, estimated or hardcoded.
  * Single-class data yields `None` (not a made-up value) for AUC / EER.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

STATUS_EVALUATED = "EVALUATED"
STATUS_NOT_EVALUATED = "NOT_EVALUATED"
STATUS_SMOKE_TEST = "SMOKE_TEST"


def _clean(value):
    """JSON-safe float (None for NaN/inf)."""
    if value is None:
        return None
    value = float(value)
    return round(value, 6) if math.isfinite(value) else None


# ---------------------------------------------------------------- metrics
def compute_eer(y_true, scores):
    """Equal Error Rate and the score threshold where FPR ~= FNR."""
    y_true = np.asarray(y_true)
    if len(set(y_true.tolist())) < 2:
        return None, None
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2), float(thr[idx])


def operating_point(y_true, scores, target_fpr):
    """Best-recall threshold whose false-positive rate is <= target_fpr."""
    y_true = np.asarray(y_true)
    if len(set(y_true.tolist())) < 2:
        return None
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = np.where(fpr <= target_fpr)[0]
    i = int(ok[-1])
    return {
        "target_fpr": target_fpr,
        "threshold": _clean(min(float(thr[i]), 1.0)),
        "fpr": _clean(fpr[i]),
        "recall_fake": _clean(tpr[i]),
    }


def compute_metrics(y_true, scores, threshold=0.5):
    """Full metric set at a decision threshold on P(fake)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.size == 0:
        raise ValueError("No samples to evaluate.")
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_classes = len(set(y_true.tolist())) == 2
    eer, eer_thr = compute_eer(y_true, scores)

    return {
        "n_samples": int(y_true.size),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "threshold": threshold,
        "accuracy": _clean(accuracy_score(y_true, y_pred)),
        "precision_fake": _clean(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": _clean(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": _clean(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": _clean(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": _clean(fn / (fn + tp)) if (fn + tp) else None,
        "roc_auc": _clean(roc_auc_score(y_true, scores)) if two_classes else None,
        "eer": _clean(eer),
        "eer_threshold": _clean(eer_thr),
    }


def selective_metrics(y_true, scores, low=0.35, high=0.65, threshold=0.5):
    """Metrics when low-confidence predictions are reported as UNCERTAIN.

    The uncertain band is fake-probability in (low, high), matching the
    runtime rule `confidence < 0.65` for a 2-class softmax.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    keep = (scores <= low) | (scores >= high)
    out = {"uncertain_band": [low, high],
           "coverage": _clean(keep.mean()),
           "n_uncertain": int((~keep).sum())}
    if keep.sum() == 0:
        out["metrics_on_confident"] = None
    else:
        out["metrics_on_confident"] = compute_metrics(y_true[keep], scores[keep], threshold)
    return out


def bootstrap_ci(y_true, scores, threshold=0.5, n_boot=500, seed=0, alpha=0.05):
    """Percentile bootstrap CIs for accuracy, F1, ROC-AUC and EER."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(y_true.tolist())) < 2 or y_true.size < 20:
        return None
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(n_boot):
        idx = rng.integers(0, y_true.size, y_true.size)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        m = compute_metrics(y_true[idx], scores[idx], threshold)
        for k in ("accuracy", "f1_fake", "roc_auc", "eer"):
            if m[k] is not None:
                draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [_clean(np.percentile(v, lo)), _clean(np.percentile(v, hi))]
            for k, v in draws.items() if v} | {"n_bootstrap": n_boot, "confidence": 1 - alpha}


def group_breakdown(rows, y_true, scores, key, threshold=0.5):
    """Per-group (dataset / synthesis source / language) error analysis."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row.get(key) or "unknown")].append(i)
    out = []
    for name, idx in sorted(groups.items()):
        yt = np.asarray(y_true)[idx]
        pred = (np.asarray(scores)[idx] >= threshold).astype(int)
        n_fake = int((yt == 1).sum())
        out.append({
            "group": name, "n": len(idx), "n_real": len(idx) - n_fake, "n_fake": n_fake,
            "accuracy": _clean((pred == yt).mean()),
            "false_positive_rate": _clean(((pred == 1) & (yt == 0)).sum() / max(1, (yt == 0).sum())) if (yt == 0).any() else None,
            "false_negative_rate": _clean(((pred == 0) & (yt == 1)).sum() / max(1, (yt == 1).sum())) if (yt == 1).any() else None,
        })
    return out


# ---------------------------------------------------------------- leakage
def _norm_path(p):
    return str(p).replace("\\", "/").strip().lower()


def check_leakage(train_rows, eval_rows):
    """Detect train/test contamination by file, speaker and synthesis source."""
    def vals(rows, col, only_label=None):
        out = set()
        for r in rows:
            if only_label and r.get("label") != only_label:
                continue
            v = (r.get(col) or "").strip().lower()
            if v and v not in {"unknown", "none", "n/a", "-"}:
                out.add(v)
        return out

    files = {_norm_path(r["file"]) for r in train_rows} & {_norm_path(r["file"]) for r in eval_rows}
    speakers = vals(train_rows, "speaker", "real") & vals(eval_rows, "speaker", "real")
    sources = vals(train_rows, "source", "fake") & vals(eval_rows, "source", "fake")
    return {
        "checked": True,
        "overlapping_files": len(files),
        "overlapping_real_speakers": len(speakers),
        "overlapping_fake_sources": len(sources),
        "examples": {"files": sorted(files)[:5], "speakers": sorted(speakers)[:5],
                     "sources": sorted(sources)[:5]},
        "leakage_free": not (files or speakers or sources),
        "note": ("Overlapping fake sources mean the same synthesis system appears in "
                 "train and test; this measures in-distribution, not unseen-generator, "
                 "performance."),
    }


# ---------------------------------------------------------------- reports
def not_evaluated_report(reason, instructions):
    return {
        "status": STATUS_NOT_EVALUATED,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reason": reason,
        "how_to_run": instructions,
        "metrics": None,
    }


def render_markdown(report):
    lines = [f"# VoxShield AI - Model Evaluation Report",
             f"*Generated:* {report.get('generated_at')}  ",
             f"*Status:* **{report['status']}**", ""]
    if report["status"] == STATUS_NOT_EVALUATED:
        lines += ["## Evaluation has NOT been completed", "", report.get("reason", ""), "",
                  "### How to run it", "```", *[str(x) for x in report.get("how_to_run", [])], "```"]
        return "\n".join(lines) + "\n"

    if report["status"] == STATUS_SMOKE_TEST:
        lines += ["> **SMOKE TEST ONLY.** Synthetic placeholder data was used to verify the "
                  "pipeline runs. These numbers say NOTHING about real-world accuracy.", ""]

    info = report.get("run", {})
    lines += ["## Run", *[f"- **{k}:** {v}" for k, v in info.items()], ""]
    m = report["metrics"]
    cm = m["confusion_matrix"]
    lines += [
        "## Overall metrics (fake = positive class)", "",
        "| Metric | Value |", "|---|---|",
        f"| Samples (real / fake) | {m['n_samples']} ({m['n_real']} / {m['n_fake']}) |",
        f"| Threshold on P(fake) | {m['threshold']} |",
        f"| Accuracy | {m['accuracy']} |", f"| Precision | {m['precision_fake']} |",
        f"| Recall | {m['recall_fake']} |", f"| F1 | {m['f1_fake']} |",
        f"| False Positive Rate (real flagged fake) | {m['false_positive_rate']} |",
        f"| False Negative Rate (fake missed) | {m['false_negative_rate']} |",
        f"| ROC-AUC | {m['roc_auc']} |",
        f"| EER (threshold) | {m['eer']} ({m['eer_threshold']}) |", "",
        "### Confusion matrix", "", "| | Pred real | Pred fake |", "|---|---|---|",
        f"| **True real** | {cm['tn']} | {cm['fp']} |", f"| **True fake** | {cm['fn']} | {cm['tp']} |", ""]

    ci = report.get("confidence_intervals")
    if ci:
        lines += ["### 95% bootstrap confidence intervals", ""]
        lines += [f"- {k}: {v[0]} - {v[1]}" for k, v in ci.items() if isinstance(v, list)] + [""]

    sel = report.get("uncertain_handling")
    if sel:
        lines += ["## Uncertain-prediction handling", "",
                  f"Band: fake-probability in {sel['uncertain_band']} -> UNCERTAIN. "
                  f"Coverage {sel['coverage']} ({sel['n_uncertain']} samples abstained).", ""]
        c = sel.get("metrics_on_confident")
        if c:
            lines += [f"On confident samples only: accuracy {c['accuracy']}, FPR "
                      f"{c['false_positive_rate']}, FNR {c['false_negative_rate']}, F1 {c['f1_fake']}.", ""]

    ops = report.get("operating_points")
    if ops:
        lines += ["## Operating points (choose the threshold for your FPR budget)", "",
                  "| Target FPR | Threshold | Actual FPR | Fake recall |", "|---|---|---|---|"]
        lines += [f"| {o['target_fpr']} | {o['threshold']} | {o['fpr']} | {o['recall_fake']} |"
                  for o in ops if o] + [""]

    for key, title in (("by_source", "By synthesis source / real corpus"),
                       ("by_dataset", "By dataset"), ("by_language", "By language")):
        rows = report.get(key)
        if rows:
            lines += [f"## {title}", "", "| Group | n | Accuracy | FPR | FNR |", "|---|---|---|---|---|"]
            lines += [f"| {r['group']} | {r['n']} | {r['accuracy']} | {r['false_positive_rate']} | "
                      f"{r['false_negative_rate']} |" for r in rows] + [""]

    pp = report.get("preprocessing_comparison")
    if pp:
        lines += ["## Preprocessing comparison (same model, same files)", "",
                  "| Preprocessing | Accuracy | F1 | ROC-AUC | EER |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {v['accuracy']} | {v['f1_fake']} | {v['roc_auc']} | {v['eer']} |"
                  for k, v in pp.items()] + [""]

    lk = report.get("leakage_check")
    if lk:
        lines += ["## Data-leakage check", ""]
        if lk.get("checked"):
            lines += [f"- Overlapping files: {lk['overlapping_files']}",
                      f"- Overlapping real speakers: {lk['overlapping_real_speakers']}",
                      f"- Overlapping fake sources: {lk['overlapping_fake_sources']}",
                      f"- Leakage-free: **{lk['leakage_free']}**", f"- {lk['note']}", ""]
        else:
            lines += [f"Not checked: {lk.get('reason')}", ""]

    if report.get("skipped_files"):
        lines += ["## Files skipped", "", f"{report['skipped_files']['count']} file(s) failed to load "
                  "and are excluded from the metrics (first few: "
                  f"{report['skipped_files']['examples']}).", ""]
    return "\n".join(lines) + "\n"


def write_report(report, out_dir, stem="evaluation_report"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    return out_dir / f"{stem}.json", out_dir / f"{stem}.md"
