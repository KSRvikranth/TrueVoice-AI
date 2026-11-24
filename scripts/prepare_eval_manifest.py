"""Convert public deepfake-audio corpora into the evaluation manifest format.

Datasets are NOT downloaded by this project; obtain them yourself and respect
their licences (ASVspoof 2019: ODC-By via Edinburgh DataShare; WaveFake: Zenodo,
CC BY-SA 4.0; LJSpeech: public domain).

ASVspoof 2019 LA protocol line:  SPEAKER_ID FILE_ID - SYSTEM_ID KEY(bonafide|spoof)
Speaker and synthesis-system IDs are kept so evaluation can group by them.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = ["file", "dataset", "label", "language", "speaker", "source"]


def asvspoof_rows(protocol, audio_dir, ext=".flac", dataset="asvspoof2019_la"):
    rows, missing = [], 0
    for line in Path(protocol).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        speaker, file_id, _, system, key = parts[:5]
        path = Path(audio_dir) / f"{file_id}{ext}"
        if not path.exists():
            missing += 1
            continue
        real = key.lower() == "bonafide"
        rows.append({"file": str(path), "dataset": dataset, "label": "real" if real else "fake",
                     "language": "en", "speaker": speaker if real else "",
                     "source": "bonafide" if real else system})
    return rows, missing


def wavefake_rows(wavefake_root, real_dir, dataset="wavefake"):
    rows = []
    for gen_dir in sorted(p for p in Path(wavefake_root).iterdir() if p.is_dir()):
        for wav in sorted(gen_dir.glob("*.wav")):
            rows.append({"file": str(wav), "dataset": dataset, "label": "fake", "language": "en",
                         "speaker": "", "source": gen_dir.name})
    for wav in sorted(Path(real_dir).glob("*.wav")):
        rows.append({"file": str(wav), "dataset": dataset, "label": "real", "language": "en",
                     "speaker": "ljspeech", "source": "ljspeech"})
    return rows


def write(rows, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    n_fake = sum(r["label"] == "fake" for r in rows)
    print(f"Wrote {len(rows)} rows ({len(rows)-n_fake} real / {n_fake} fake) to {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("asvspoof")
    a.add_argument("--protocol", required=True); a.add_argument("--audio-dir", required=True)
    a.add_argument("--out", required=True); a.add_argument("--ext", default=".flac")
    w = sub.add_parser("wavefake")
    w.add_argument("--wavefake-root", required=True); w.add_argument("--real-dir", required=True)
    w.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.kind == "asvspoof":
        rows, missing = asvspoof_rows(args.protocol, args.audio_dir, args.ext)
        if missing:
            print(f"WARNING: {missing} protocol entries had no audio file and were skipped.")
    else:
        rows = wavefake_rows(args.wavefake_root, args.real_dir)
    if not rows:
        raise SystemExit("No usable rows found - check the paths.")
    write(rows, args.out)


if __name__ == "__main__":
    main()
