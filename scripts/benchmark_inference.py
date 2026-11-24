"""Measure end-to-end CNN inference latency (CPU/GPU) on a fixed 6 s clip.

    python -m scripts.benchmark_inference --out results/benchmarks/latency_after.json
    python -m scripts.benchmark_inference --root <other checkout> --label before

`--root` lets the same benchmark run against an older checkout for a fair
before/after comparison.
"""
from __future__ import annotations

import argparse, json, os, platform, statistics, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    root = args.root.resolve()
    sys.path.insert(0, str(root)); os.chdir(root)
    import numpy as np, soundfile as sf, torch
    from scripts.voice_inference import predict_voice, prepare_audio

    # deterministic clip: speech-like modulated harmonics + noise, 6 s @ 16 kHz
    sr, t = 16000, np.arange(16000 * 6) / 16000
    rng = np.random.default_rng(0)
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1))
    x = (0.2 * x * (1 + 0.6 * np.sin(2 * np.pi * 4 * t)) + 0.01 * rng.standard_normal(t.size)).astype("float32")
    wav = Path(tempfile.mkdtemp()) / "bench.wav"; sf.write(str(wav), x, sr)

    t0 = time.perf_counter(); predict_voice(wav); cold = (time.perf_counter() - t0) * 1000
    for _ in range(3): predict_voice(wav)

    total, prep = [], []
    for _ in range(args.runs):
        t0 = time.perf_counter(); prepare_audio(wav); prep.append((time.perf_counter() - t0) * 1000)
    for _ in range(args.runs):
        t0 = time.perf_counter(); predict_voice(wav); total.append((time.perf_counter() - t0) * 1000)

    n_conc = args.runs
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(lambda _: predict_voice(wav), range(n_conc)))
    conc_wall = time.perf_counter() - t0

    result = {
        "label": args.label, "root": str(root), "runs": args.runs,
        "machine": {"python": platform.python_version(), "torch": torch.__version__,
                    "cpu_count": os.cpu_count(), "device": "cuda" if torch.cuda.is_available() else "cpu"},
        "first_call_ms": round(cold, 1),
        "preprocessing_only_ms": {"mean": round(statistics.mean(prep), 2), "p50": round(pct(prep, 50), 2), "p95": round(pct(prep, 95), 2)},
        "predict_voice_ms": {"mean": round(statistics.mean(total), 2), "p50": round(pct(total, 50), 2),
                             "p95": round(pct(total, 95), 2), "max": round(max(total), 2)},
        f"concurrent_{args.threads}_threads": {"requests": n_conc, "wall_seconds": round(conc_wall, 2),
                                              "throughput_per_s": round(n_conc / conc_wall, 1)},
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
