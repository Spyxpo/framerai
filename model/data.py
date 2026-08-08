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

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

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
# Streaming, packed token shards (scales past in-memory tokenization)
# ---------------------------------------------------------------------------

SHARD_META = "meta.json"


def _shard_dtype(vocab_size: int):
    """uint16 fits vocabularies up to 65536; uint32 otherwise."""
    return np.uint16 if vocab_size <= 65536 else np.uint32


def prepare_shards(data_dir: str, tokenizer, out_dir: str, shard_tokens: int = 1_000_000):
    """Tokenize a corpus into memory-mappable ``.bin`` shards + a meta.json.

    Streams text records (no full-corpus buffering), separating documents with
    the EOS token so packed blocks respect document boundaries. Returns the meta
    dict. Run once offline, then train from the shard directory via
    :class:`PackedTokenDataset`.
    """
    os.makedirs(out_dir, exist_ok=True)
    dtype = _shard_dtype(tokenizer.vocab_size)
    shards, buf, total = [], [], 0

    def flush():
        if not buf:
            return
        idx = len(shards)
        path = os.path.join(out_dir, f"shard_{idx:05d}.bin")
        np.asarray(buf, dtype=dtype).tofile(path)
        shards.append(os.path.basename(path))
        buf.clear()

    for text in iter_text_records(data_dir):
        ids = tokenizer.encode(text, add_special=True)  # wraps with sos/eos
        buf.extend(ids)
        total += len(ids)
        while len(buf) >= shard_tokens:
            carry = buf[shard_tokens:]
            del buf[shard_tokens:]
            flush()
            buf.extend(carry)
    flush()

    meta = {
        "dtype": np.dtype(dtype).name,
        "shards": shards,
        "total_tokens": total,
        "vocab_size": tokenizer.vocab_size,
    }
    with open(os.path.join(out_dir, SHARD_META), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class PackedTokenDataset(IterableDataset):
    """Stream fixed-length next-token blocks from memory-mapped token shards.

    Blocks are non-overlapping windows of ``seq_len`` tokens (label shifted by
    one), packed with no padding. Work is split deterministically across
    distributed ranks and DataLoader workers, so every (rank, worker) sees a
    disjoint slice of blocks. An optional shuffle buffer adds ordering noise.
    """

    def __init__(self, shard_dir: str, seq_len: int, rank: int = 0, world_size: int = 1,
                 shuffle_buffer: int = 0, seed: int = 0):
        with open(os.path.join(shard_dir, SHARD_META)) as f:
            self.meta = json.load(f)
        self.shard_dir = shard_dir
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = max(1, world_size)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.dtype = np.dtype(self.meta["dtype"])

    def _blocks(self, consumer_id: int, total_consumers: int):
        step = self.seq_len
        gid = 0
        for shard in self.meta["shards"]:
            arr = np.memmap(os.path.join(self.shard_dir, shard), dtype=self.dtype, mode="r")
            n_blocks = (len(arr) - 1) // step
            for b in range(n_blocks):
                if gid % total_consumers == consumer_id:
                    start = b * step
                    chunk = np.asarray(arr[start:start + step + 1], dtype=np.int64)
                    yield {
                        "input_ids": torch.from_numpy(chunk[:-1]),
                        "labels": torch.from_numpy(chunk[1:]),
                    }
                gid += 1

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        consumer_id = self.rank * num_workers + worker_id
        total_consumers = self.world_size * num_workers

        stream = self._blocks(consumer_id, total_consumers)
        if self.shuffle_buffer <= 1:
            yield from stream
            return

        import random

        rng = random.Random(self.seed + consumer_id)
        buffer = []
        for item in stream:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                yield buffer.pop(rng.randrange(len(buffer)))
        rng.shuffle(buffer)
        yield from buffer


def build_packed_dataset(shard_dir: str, seq_len: int, rank: int = 0, world_size: int = 1,
                         shuffle_buffer: int = 1024):
    """Return a PackedTokenDataset if a prepared shard dir exists, else None."""
    if not os.path.isfile(os.path.join(shard_dir, SHARD_META)):
        return None
    return PackedTokenDataset(shard_dir, seq_len, rank=rank, world_size=world_size,
                              shuffle_buffer=shuffle_buffer)


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


class InterleavedSequenceBuilder:
    """Build token sequences with modality placeholders in the right places.

    Interleaved placement needs the sequence to reserve exactly one token per
    modality embedding. Getting that count wrong corrupts the sequence, so the
    builder that emits the placeholders and the encoder that fills them have to
    agree; this class owns the emitting half and reports how many slots it
    reserved so the caller can check.

    A run looks like ``<img> <img_patch> x N <img_end>``: the markers stay so
    the model can tell where a modality started and stopped, and only the inner
    run is replaced by embeddings.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_patch = tokenizer.reserved_tokens["<img_patch>"]
        self.audio_frame = tokenizer.reserved_tokens["<audio_frame>"]
        self.specials = tokenizer.special_tokens

    def _run(self, open_token: str, placeholder: int, end_token: str, count: int) -> list:
        return [self.specials[open_token]] + [placeholder] * count + [self.specials[end_token]]

    def image_run(self, n_tokens: int) -> list:
        return self._run("<img>", self.image_patch, "<img_end>", n_tokens)

    def audio_run(self, n_frames: int) -> list:
        return self._run("<audio>", self.audio_frame, "<audio_end>", n_frames)

    def build(self, segments: list) -> dict:
        """Assemble a sequence from ``("text"|"image"|"audio", value)`` segments.

        ``value`` is a string for text and a token count for a modality.
        Returns the ids plus the number of slots reserved per placeholder, which
        is what the caller checks against the encoder's output.
        """
        ids, counts = [], {self.image_patch: 0, self.audio_frame: 0}
        for kind, value in segments:
            if kind == "text":
                ids.extend(self.tokenizer.encode(value, add_special=False))
            elif kind == "image":
                ids.extend(self.image_run(value))
                counts[self.image_patch] += value
            elif kind == "audio":
                ids.extend(self.audio_run(value))
                counts[self.audio_frame] += value
            else:
                raise ValueError(f"Unknown segment kind '{kind}'")
        return {"input_ids": ids, "placeholder_counts": counts}
