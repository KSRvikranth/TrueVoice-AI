import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def make_wav(tmp_path):
    def _make(name="a.wav", seconds=3.0, sr=16000, kind="tone", channels=1):
        n = int(seconds * sr)
        t = np.arange(n) / sr
        if kind == "tone":
            x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
        elif kind == "silence":
            x = np.zeros(n)
        elif kind == "noise":
            x = 0.1 * np.random.default_rng(0).standard_normal(n)
        else:
            raise ValueError(kind)
        x = x.astype("float32")
        if channels == 2:
            x = np.stack([x, x * 0.5], axis=1)
        path = tmp_path / name
        sf.write(str(path), x, sr)
        return path
    return _make
