"""Phase 2 tests: metric correctness (known answers), leakage, reports, CLI, uncertainty."""
import csv
import itertools
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts import config, evaluation as ev
from scripts.risk_scoring import calculate_risk


# ---------- metric known-answer tests ----------
def test_confusion_matrix_hand_computed():
    m = ev.compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == 0.5
    assert m["false_positive_rate"] == 0.5 and m["false_negative_rate"] == 0.5
    assert m["precision_fake"] == 0.5 and m["recall_fake"] == 0.5 and m["f1_fake"] == 0.5


def test_perfect_separation():
    m = ev.compute_metrics([0, 0, 1, 1], [0.01, 0.2, 0.8, 0.99])
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["eer"] == 0.0


def test_auc_matches_sklearn_on_random_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300); s = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    assert ev.compute_metrics(y, s)["roc_auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-6)


def _brute_force_eer(y, s):
    best = None
    for t in sorted(set(s)) + [max(s) + 1]:
        pred = np.asarray(s) >= t
        fpr = np.mean(pred[np.asarray(y) == 0]); fnr = np.mean(~pred[np.asarray(y) == 1])
        gap = abs(fpr - fnr)
        if best is None or gap < best[0]:
            best = (gap, (fpr + fnr) / 2)
    return best[1]


def test_eer_matches_independent_brute_force():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400); s = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.25, 400), 0, 1)
    eer, _ = ev.compute_eer(y, s)
    assert eer == pytest.approx(_brute_force_eer(y, s), abs=0.01)


def test_single_class_gives_none_not_invented_numbers():
    m = ev.compute_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert m["roc_auc"] is None and m["eer"] is None and m["false_positive_rate"] is None


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        ev.compute_metrics([], [])


def test_operating_point_respects_fpr_budget():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000); s = np.clip(y * 0.35 + rng.random(2000) * 0.65, 0, 1)
    for target in (0.01, 0.05, 0.1):
        op = ev.operating_point(y, s, target)
        pred = s >= op["threshold"]
        assert pred[y == 0].mean() <= target + 1e-9


def test_selective_metrics_abstains_in_band():
    s = np.array([0.05, 0.4, 0.5, 0.6, 0.95]); y = np.array([0, 0, 1, 0, 1])
    sel = ev.selective_metrics(y, s, 0.35, 0.65)
    assert sel["n_uncertain"] == 3 and sel["coverage"] == pytest.approx(0.4)


def test_bootstrap_ci_is_seeded_and_brackets_estimate():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 500); s = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    a = ev.bootstrap_ci(y, s, seed=1, n_boot=200); b = ev.bootstrap_ci(y, s, seed=1, n_boot=200)
    assert a == b
    auc = ev.compute_metrics(y, s)["roc_auc"]
    assert a["roc_auc"][0] <= auc <= a["roc_auc"][1]


# ---------- leakage ----------
def test_leakage_detected_and_clean_cases():
    train = [{"file": "a/1.wav", "label": "real", "speaker": "s1", "source": ""},
             {"file": "a/2.wav", "label": "fake", "speaker": "", "source": "gan_a"}]
    leaky = [{"file": "A\\1.wav".replace("A\\", "a/"), "label": "real", "speaker": "S1", "source": ""},
             {"file": "b/9.wav", "label": "fake", "speaker": "", "source": "GAN_A"}]
    clean = [{"file": "b/3.wav", "label": "real", "speaker": "s2", "source": ""},
             {"file": "b/4.wav", "label": "fake", "speaker": "", "source": "gan_b"}]
    r = ev.check_leakage(train, leaky)
    assert (r["overlapping_files"], r["overlapping_real_speakers"], r["overlapping_fake_sources"]) == (1, 1, 1)
    assert not r["leakage_free"]
    assert ev.check_leakage(train, clean)["leakage_free"]


# ---------- reports ----------
def test_not_evaluated_report_has_no_metrics():
    r = ev.not_evaluated_report("no data", ["cmd"])
    assert r["status"] == ev.STATUS_NOT_EVALUATED and r["metrics"] is None
    assert "NOT been completed" in ev.render_markdown(r)


def test_smoke_report_is_loudly_marked():
    from scripts.evaluate_model import build_report
    y = np.array([0, 1] * 15); s = np.where(y == 1, 0.8, 0.2)
    rows = [{"file": f"{i}.wav", "label": "fake" if v else "real"} for i, v in enumerate(y)]
    md = ev.render_markdown(build_report("t", rows, y, s, [], 0.5, status=ev.STATUS_SMOKE_TEST))
    assert "SMOKE TEST ONLY" in md and "NOTHING about real-world accuracy" in md


# ---------- manifest adapters ----------
def test_asvspoof_adapter(tmp_path):
    from scripts.prepare_eval_manifest import asvspoof_rows
    (tmp_path / "audio").mkdir()
    for fid in ("LA_E_1", "LA_E_2", "LA_E_3"):
        (tmp_path / "audio" / f"{fid}.flac").write_bytes(b"x")
    proto = tmp_path / "p.txt"
    proto.write_text("LA_0001 LA_E_1 - - bonafide\nLA_0002 LA_E_2 - A07 spoof\nLA_0003 LA_E_9 - A08 spoof\n")
    rows, missing = asvspoof_rows(proto, tmp_path / "audio")
    assert missing == 1 and len(rows) == 2
    real = next(r for r in rows if r["label"] == "real"); fake = next(r for r in rows if r["label"] == "fake")
    assert real["speaker"] == "LA_0001" and fake["source"] == "A07" and fake["speaker"] == ""


# ---------- CLI end-to-end (pipeline smoke test; NOT a real accuracy result) ----------
def test_cli_not_evaluated_when_manifest_missing(tmp_path):
    from scripts.evaluate_model import main
    rc = main(["--manifest", str(tmp_path / "none.csv"), "--out-dir", str(tmp_path / "out")])
    assert rc == 2
    rep = json.loads((tmp_path / "out" / "evaluation_report.json").read_text())
    assert rep["status"] == "NOT_EVALUATED"


def test_cli_end_to_end_with_skipped_and_leakage(tmp_path, make_wav):
    from scripts.evaluate_model import main
    rows = []
    for i in range(12):
        kind = "tone" if i % 2 else "noise"
        p = make_wav(f"f{i}.wav", seconds=6, kind=kind)
        rows.append({"file": str(p), "dataset": "synthetic", "label": "fake" if i % 2 else "real",
                     "language": "en", "speaker": f"sp{i}" if i % 2 == 0 else "", "source": f"gen{i%4}" if i % 2 else "noise"})
    bad = tmp_path / "bad.wav"; bad.write_bytes(b"junk")
    rows.append({"file": str(bad), "dataset": "synthetic", "label": "real", "language": "en", "speaker": "x", "source": "noise"})
    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    train = tmp_path / "t.csv"
    with open(train, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerow(rows[0])

    rc = main(["--manifest", str(man), "--train-manifest", str(train), "--out-dir", str(tmp_path / "o"),
               "--name", "smoke", "--compare-preprocessing", "--smoke-test"])
    assert rc == 0
    rep = json.loads((tmp_path / "o" / "evaluation_smoke.json").read_text())
    assert rep["status"] == "SMOKE_TEST"
    assert rep["metrics"]["n_samples"] == 12                      # corrupted file excluded...
    assert rep["skipped_files"]["count"] == 1                      # ...and reported, not hidden
    assert rep["leakage_check"]["overlapping_files"] == 1         # train row 0 is in the test set
    assert set(rep["preprocessing_comparison"]) == {"production (trim silence top_db=30)", "no trim"}
    preds = list(csv.DictReader(open(tmp_path / "o" / "evaluation_smoke_predictions.csv")))
    assert len(preds) == 12 and all(0 <= float(r["fake_probability"]) <= 1 for r in preds)


# ---------- uncertain handling / risk scoring ----------
@pytest.mark.parametrize("fake_p,expected_status,expected_verdict", [
    (0.99, "HIGH_MODEL_CONFIDENCE", "synthetic"),
    (0.70, "MODERATE", "synthetic"),
    (0.55, "UNCERTAIN", "uncertain"),
    (0.45, "UNCERTAIN", "uncertain"),
    (0.20, "HIGH_MODEL_CONFIDENCE", "genuine"),
])
def test_decision_logic_with_controlled_logits(monkeypatch, make_wav, fake_p, expected_status, expected_verdict):
    import torch
    import scripts.voice_inference as vi

    class Stub(torch.nn.Module):  # test double for the NETWORK only; decision rules are real
        def forward(self, x):
            return torch.log(torch.tensor([[1 - fake_p, fake_p]])).repeat(x.shape[0], 1)

    monkeypatch.setattr(vi, "get_model", lambda: Stub())
    r = vi.predict_voice(make_wav(seconds=2, kind="noise"))
    assert r["confidence_status"] == expected_status and r["verdict"] == expected_verdict
    assert r["prediction"] == ("fake" if fake_p >= 0.5 else "real")   # original rule preserved


NO_TEXT = {"detected_categories": [], "strong_text_signals": []}


def test_uncertain_fake_gets_minimal_weight():
    uncertain = {"prediction": "fake", "fake_probability": 0.55, "confidence_status": "UNCERTAIN"}
    r = calculate_risk(uncertain, NO_TEXT)
    assert r["risk_score"] == 5 and r["risk_level"] == "LOW" and "voice_model_uncertain" in r["signals"]


def test_confident_fake_and_legacy_input_unchanged():
    assert calculate_risk({"prediction": "fake", "fake_probability": 0.99,
                           "confidence_status": "HIGH_MODEL_CONFIDENCE"}, NO_TEXT)["risk_score"] == 20
    assert calculate_risk({"prediction": "fake", "fake_probability": 0.99}, NO_TEXT)["risk_score"] == 20


def test_uncertain_voice_never_pushes_to_high_alone():
    scam = {"detected_categories": ["otp_request"], "strong_text_signals": ["otp_request"]}
    r = calculate_risk({"prediction": "fake", "confidence_status": "UNCERTAIN"}, scam)
    assert r["risk_level"] != "HIGH"


# ---------- API ----------
def test_evaluation_endpoint(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    monkeypatch.setattr(config, "EVALUATION_DIR", tmp_path / "ev")
    c = TestClient(app_module.app)
    assert c.get("/evaluation").json()["status"] == "NOT_EVALUATED"
    from scripts.evaluate_model import build_report
    y = np.array([0, 1] * 15); sc = np.where(y == 1, 0.8, 0.2)
    rows = [{"file": f"{i}.wav", "label": "fake" if v else "real"} for i, v in enumerate(y)]
    ev.write_report(build_report("t", rows, y, sc, [], 0.5), tmp_path / "ev")
    body = c.get("/evaluation").json()
    assert body["status"] == "EVALUATED" and body["metrics"]["accuracy"] == 1.0
