"""A spread-spectrum watermark for generated speech.

A cloned voice can be passed off as the real speaker. Every clip this model
clones carries a mark that :func:`detect` finds again, so a platform, or the
person whose voice it is, can check whether a recording came from here.

The mark is a pseudo-random +/-1 sequence, derived from a key, tiled across the
clip and added at a small fraction of the clip's own loudness. Detection
correlates a recording with the same sequence: in unmarked audio the
correlation is noise around zero; in marked audio it grows with the square root
of the clip's length, so a few seconds give a clear margin. A change of volume
scales signal and mark together and leaves the score unchanged.

What it is not: cryptographic, or robust to resampling, time stretching or a
lossy re-encode that removes the band the mark sits in. It is a deterrent and a
provenance check, not a guarantee, and it is described that way to callers.
"""

import hashlib

import numpy as np

DEFAULT_KEY = "framerai-voice"
PERIOD = 8192
STRENGTH = 0.015
THRESHOLD = 4.0


def _sequence(key: str, length: int = PERIOD) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    pn = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=length)
    # High-pass by first difference: the mark sits above the band where most
    # of a voice's energy is, which keeps it quieter to the ear and less
    # correlated with speech under detection.
    pn = np.diff(pn, prepend=pn[-1]) / 2.0
    return pn.astype(np.float32)


def _tiled(key: str, n: int) -> np.ndarray:
    pn = _sequence(key)
    reps = int(np.ceil(n / len(pn)))
    return np.tile(pn, reps)[:n]


def embed(waveform, key: str = DEFAULT_KEY, strength: float = STRENGTH) -> np.ndarray:
    """``waveform`` with the mark added at ``strength`` times its RMS."""
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
    if rms == 0.0:
        return wave.copy()
    marked = wave + strength * rms * _tiled(key, wave.size)
    peak = float(np.max(np.abs(marked)))
    return marked / peak if peak > 1.0 else marked


def score(waveform, key: str = DEFAULT_KEY) -> float:
    """Normalised correlation with the key's sequence; about N(0, 1) when unmarked."""
    wave = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if wave.size == 0:
        return 0.0
    pn = _tiled(key, wave.size).astype(np.float64)
    # Correlate the high-passed signal, so a voice's low-frequency energy does
    # not swamp the statistic.
    hp = np.diff(wave, prepend=wave[0])
    num = float(np.dot(hp, pn))
    den = float(np.sqrt(np.dot(hp**2, pn**2)))
    return num / den if den > 0 else 0.0


def detect(waveform, key: str = DEFAULT_KEY, threshold: float = THRESHOLD) -> dict:
    """Whether ``waveform`` carries the mark, with the score that decided it."""
    s = score(waveform, key)
    return {"watermarked": s >= threshold, "score": round(s, 3), "threshold": threshold}
