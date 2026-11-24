# Inference latency – before vs after (measured)

Method: identical benchmark (`scripts/benchmark_inference.py`) run against the original
checkout (git `main`) and the upgraded code. 3 independent runs × 100 calls per version,
same 6-second clip, CPU only. Machine: sandbox with **1 vCPU**, Python 3.12.3, torch 2.15.0.
Values are the median of the three run-means (individual runs in `latency_comparison.json`).
Your Windows PC will show different absolute numbers; re-run the script to get yours.

| Measure | Before | After | Change |
|---|---|---|---|
| `predict_voice` mean latency (ms) | 12.45 | 11.89 | -4.5 % (within run-to-run noise) |
| `predict_voice` p95 latency (ms) | 14.60 | 13.91 | -4.7 % (within noise) |
| Preprocessing only, mean (ms) | 5.15 | 5.33 | no change (unchanged code) |
| Throughput, 4 concurrent threads (req/s) | 66.9 | 103.0 | **+54 %** (consistent in all 3 runs) |
| First call after start (ms) | 1202 | 1140 | one-off warm-up cost (library start-up) |

Interpretation: the CNN is tiny, so reloading it per request was cheap; caching it mainly
removes per-request disk I/O and lock-free repeated loading under concurrency. Single-request
latency is dominated by audio preprocessing (~5 ms of ~12 ms), and that did not change.
The new input validation costs a few milliseconds but is offset by the cached model.
