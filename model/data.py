"""Local-corpus datasets for from-scratch FramerAI training.

Reads training data from a local directory instead of any external teacher
model. Three modalities are supported:

- text: ``.txt`` files (split on blank lines) and ``.jsonl`` records with a
  ``text`` field, a ``prompt``/``response`` pair, or an ``instruction``/``output``
  pair. Image and audio caption records also contribute their caption text.
- image-caption: ``.jsonl`` records with ``image`` (path) and ``caption``.
- audio-caption: ``.jsonl`` records with ``audio`` (path) and ``text``.

Paths inside ``.jsonl`` records are resolved relative to the file that contains
them.
"""

import glob
import json
import os

import torch
from torch.utils.data import Dataset

from .modules.audio_encoder import AudioFrontEnd


# ---------------------------------------------------------------------------
# Text records
# ---------------------------------------------------------------------------

def _record_to_text(record: dict) -> str:
    """Turn a JSONL record into a training string, or '' if none applies."""
    if "text" in record:
        return str(record["text"]).strip()
    if "prompt" in record and "response" in record:
        return f"<user>{record['prompt']}<assistant>{record['response']}".strip()
    if "instruction" in record and "output" in record:
        instr = record["instruction"]
        if record.get("input"):
            instr = f"{instr}\n{record['input']}"
        return f"<user>{instr}<assistant>{record['output']}".strip()
    if "caption" in record:
        return str(record["caption"]).strip()
    return ""


def iter_text_records(data_dir: str):
    """Yield training strings from all text and caption sources in a directory."""
    if not data_dir or not os.path.isdir(data_dir):
        return

    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)):
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        for chunk in content.split("\n\n"):
            chunk = chunk.strip()
            if chunk:
                yield chunk

    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _record_to_text(record)
                if text:
                    yield text


class TextCorpusDataset(Dataset):
    """Next-token dataset over a local text corpus."""

    def __init__(self, texts: list, tokenizer, max_len: int = 512):
        self.samples = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special=True)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens += [tokenizer.pad_id] * (max_len - len(tokens))
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def build_text_dataset(data_dir: str, tokenizer, max_len: int = 512):
    """Build a TextCorpusDataset from a directory, or None if it has no text."""
    texts = list(iter_text_records(data_dir))
    if not texts:
        return None
    return TextCorpusDataset(texts, tokenizer, max_len=max_len)


# ---------------------------------------------------------------------------
# Modality pairs (image / audio caption)
# ---------------------------------------------------------------------------

def _iter_pairs(data_dir: str, path_key: str, text_keys: tuple):
    """Yield (resolved_path, caption) tuples from JSONL records with path_key."""
    if not data_dir or not os.path.isdir(data_dir):
        return
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)):
        base = os.path.dirname(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if path_key not in record:
                    continue
                media = record[path_key]
                media_path = media if os.path.isabs(media) else os.path.join(base, media)
                caption = ""
                for key in text_keys:
                    if key in record:
                        caption = str(record[key])
                        break
                if os.path.exists(media_path):
                    yield media_path, caption


def _encode_caption(tokenizer, caption: str, max_len: int) -> torch.Tensor:
    tokens = tokenizer.encode(caption, add_special=True)[:max_len]
    tokens += [tokenizer.pad_id] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


class ImageCaptionDataset(Dataset):
    """Image-caption pairs for diffusion training."""

    def __init__(self, data_dir: str, tokenizer, resolution: int = 256, caption_len: int = 64):
        self.pairs = list(_iter_pairs(data_dir, "image", ("caption", "text")))
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.caption_len = caption_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image

        path, caption = self.pairs[idx]
        img = Image.open(path).convert("RGB").resize((self.resolution, self.resolution))
        tensor = torch.from_numpy(_image_to_array(img)).permute(2, 0, 1).float()
        tensor = tensor / 127.5 - 1.0
        return {
            "input_ids": _encode_caption(self.tokenizer, caption, self.caption_len),
            "target_images": tensor,
        }


def _image_to_array(img):
    import numpy as np

    return np.asarray(img, dtype="float32")


class AudioCaptionDataset(Dataset):
    """Audio-caption pairs for audio-generation training."""

    def __init__(self, data_dir: str, tokenizer, config, caption_len: int = 64):
        self.pairs = list(_iter_pairs(data_dir, "audio", ("text", "caption")))
        self.tokenizer = tokenizer
        self.caption_len = caption_len
        self.config = config
        self.frontend = AudioFrontEnd(
            config.audio_sample_rate, config.audio_n_fft,
            config.audio_hop_length, config.audio_n_mels,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, caption = self.pairs[idx]
        waveform = load_waveform(path, self.config.audio_sample_rate)
        mel = self.frontend(waveform.unsqueeze(0))[0]  # (n_mels, frames)
        mel = _normalize_mel(mel)
        mel = _fit_frames(mel, self.config.audio_gen_frames).unsqueeze(0)  # (1, n_mels, frames)
        return {
            "input_ids": _encode_caption(self.tokenizer, caption, self.caption_len),
            "target_audio": mel,
        }


def _normalize_mel(mel: torch.Tensor) -> torch.Tensor:
    """Map a log-mel spectrogram to roughly [-1, 1] to match the generator."""
    return ((mel + 4.0) / 4.0).clamp(-1.0, 1.0)


def _fit_frames(mel: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Pad or truncate a (n_mels, T) mel to exactly n_frames along time."""
    t = mel.shape[1]
    if t == n_frames:
        return mel
    if t > n_frames:
        return mel[:, :n_frames]
    pad = torch.full((mel.shape[0], n_frames - t), -1.0, dtype=mel.dtype)
    return torch.cat([mel, pad], dim=1)


def load_waveform(path: str, target_sr: int) -> torch.Tensor:
    """Load an audio file as a mono waveform tensor at target_sr.

    Uses soundfile if available, otherwise torchaudio. Raises a clear error if
    neither is installed.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)  # (num_samples,)
    except ImportError:
        try:
            import torchaudio

            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0)
        except ImportError as exc:
            raise ImportError(
                "Loading audio files needs 'soundfile' or 'torchaudio'. "
                "Install them with: pip install soundfile torchaudio"
            ) from exc

    if sr != target_sr:
        wav = _resample(wav, sr, target_sr)
    return wav


def _resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Linear resampling to avoid a hard torchaudio dependency."""
    if orig_sr == target_sr:
        return wav
    duration = wav.shape[0] / orig_sr
    new_len = int(duration * target_sr)
    if new_len <= 1:
        return wav
    idx = torch.linspace(0, wav.shape[0] - 1, new_len)
    left = idx.floor().long().clamp(max=wav.shape[0] - 1)
    right = (left + 1).clamp(max=wav.shape[0] - 1)
    frac = idx - left.float()
    return wav[left] * (1 - frac) + wav[right] * frac
