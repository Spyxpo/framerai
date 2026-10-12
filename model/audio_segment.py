"""Splitting long audio at pauses, for transcription.

The audio encoder reads a bounded window (``audio_max_frames`` hops), so a
recording longer than that lost everything past the first window. Here the
recording is cut into windows that fit, each cut placed at the quietest point
near the end of its window, so a cut lands in a pause rather than mid-word.
"""

import torch

FRAME = 400  # 25 ms at 16 kHz
SEARCH = 0.25  # look for a pause in the last quarter of each window


def _energy(wav: torch.Tensor, frame: int) -> torch.Tensor:
    n = wav.numel() // frame
    if n == 0:
        return wav.new_zeros(0)
    return wav[: n * frame].view(n, frame).pow(2).mean(dim=1)


def split_on_silence(wav: torch.Tensor, max_samples: int, frame: int = FRAME,
                     search: float = SEARCH) -> list[torch.Tensor]:
    """Windows of at most ``max_samples``, cut at the quietest nearby frame."""
    wav = wav.reshape(-1)
    if wav.numel() <= max_samples:
        return [wav]
    pieces, start = [], 0
    while wav.numel() - start > max_samples:
        window = wav[start:start + max_samples]
        energy = _energy(window, frame)
        lo = int(len(energy) * (1 - search))
        if len(energy) > lo + 1:
            cut = (lo + int(torch.argmin(energy[lo:]))) * frame + frame // 2
        else:
            cut = max_samples
        cut = max(frame, min(cut, max_samples))
        pieces.append(wav[start:start + cut])
        start += cut
    pieces.append(wav[start:])
    return [p for p in pieces if p.numel()]
