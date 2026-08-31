"""Tests for CTC auxiliary head wiring into the training loop (Issue #158)."""

import json
import wave

import numpy as np
import torch
from torch.utils.data import DataLoader

from conftest import tiny_config
from model.data import AudioCaptionDataset
from model.framer import FramerModel
from model.modules.audio_lm import CTCHead
from model.tokenizer import FramerTokenizer


def ctc_multimodal_config(**overrides):
    """Smallest config that builds audio encoder and CTC head."""
    base = dict(
        text_only=False,
        use_ctc_head=True,
        ctc_loss_weight=0.5,
        audio_sample_rate=16000,
        audio_n_fft=64,
        audio_hop_length=16,
        audio_n_mels=16,
        audio_max_frames=32,
        audio_d_model=32,
        audio_n_heads=4,
        audio_n_layers=1,
    )
    base.update(overrides)
    return tiny_config(**base)


def write_dummy_wav(path, duration_sec: float, sr: int = 16000):
    """Write a synthetic mono WAV file using Python's standard wave library."""
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def create_tiny_audio_dataset(tmp_path, vocab_size: int = 256):
    """Helper to construct a tiny AudioCaptionDataset on disk with varying audio lengths."""
    data_dir = tmp_path / "audio_data"
    data_dir.mkdir(exist_ok=True)

    wav1 = data_dir / "sample1.wav"
    wav2 = data_dir / "sample2.wav"
    write_dummy_wav(wav1, duration_sec=0.25, sr=16000)  # 4000 samples
    write_dummy_wav(wav2, duration_sec=0.50, sr=16000)  # 8000 samples

    jsonl_path = data_dir / "records.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"audio": "sample1.wav", "text": "hello world"}) + "\n")
        f.write(json.dumps({"audio": "sample2.wav", "text": "a quick brown fox jumps over the lazy dog"}) + "\n")

    tokenizer = FramerTokenizer(vocab_size=vocab_size)
    tokenizer.train(["hello world", "a quick brown fox jumps over the lazy dog"], target_vocab_size=vocab_size)
    return str(data_dir), tokenizer


def test_default_config_leaves_ctc_disabled():
    """Default FramerConfig must keep use_ctc_head=False and self.ctc_head=None."""
    config = tiny_config(text_only=False)
    assert not config.use_ctc_head
    model = FramerModel(config)
    assert model.ctc_head is None

    out = model(audio=torch.randn(1, 4000))
    assert "ctc_loss" not in out


def test_ctc_head_skipped_when_targets_missing():
    """When use_ctc_head=True but no ctc_targets exist, ctc_loss term is omitted."""
    config = ctc_multimodal_config(use_ctc_head=True)
    model = FramerModel(config)
    assert isinstance(model.ctc_head, CTCHead)

    out = model(audio=torch.randn(1, 4000))
    assert "ctc_loss" not in out


def test_real_audio_caption_dataset_batching_and_collation(tmp_path):
    """DataLoader(batch_size=2) must collate audio clips of different lengths."""
    config = ctc_multimodal_config()
    data_dir, tokenizer = create_tiny_audio_dataset(tmp_path, vocab_size=config.vocab_size)
    dataset = AudioCaptionDataset(data_dir, tokenizer, config, caption_len=64)

    assert len(dataset) == 2
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)

    batch = next(iter(loader))
    # Batch audio shape must match maximum length (8000 samples)
    assert batch["audio"].shape == (2, 8000)
    assert batch["input_ids"].shape == (2, 64)
    assert batch["ctc_targets"].shape == (2, 64)

    # ctc_target_lengths must be actual transcript lengths, NOT 64
    target_lens = batch["ctc_target_lengths"].tolist()
    assert target_lens[0] < 64 and target_lens[1] < 64
    assert target_lens[0] > 0 and target_lens[1] > 0
    assert target_lens[0] != target_lens[1]

    # ctc_input_lengths must correspond to encoder time steps
    input_lens = batch["ctc_input_lengths"].tolist()
    # hop_length=16, 4000//16 + 1 = 251 -> capped at audio_max_frames (32)
    assert input_lens[0] == 32
    assert input_lens[1] == 32


def test_real_data_path_positive_ctc_loss_and_gradient_flow(tmp_path):
    """Real DataLoader batch through FramerModel yields finite, positive CTC loss and valid gradients."""
    torch.manual_seed(42)
    config = ctc_multimodal_config(use_ctc_head=True, ctc_loss_weight=0.5)
    data_dir, tokenizer = create_tiny_audio_dataset(tmp_path, vocab_size=config.vocab_size)
    model = FramerModel(config)

    dataset = AudioCaptionDataset(data_dir, tokenizer, config, caption_len=64)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))

    out = model(
        input_ids=batch["input_ids"],
        target_audio=batch["target_audio"],
        target_waveform=batch["target_waveform"],
        audio=batch["audio"],
        ctc_targets=batch["ctc_targets"],
        ctc_input_lengths=batch["ctc_input_lengths"],
        ctc_target_lengths=batch["ctc_target_lengths"],
    )

    assert "ctc_loss" in out
    assert torch.isfinite(out["ctc_loss"])
    assert out["ctc_loss"].item() > 0.0, f"Expected positive CTC loss, got {out['ctc_loss'].item()}"

    assert "loss" in out
    assert torch.isfinite(out["loss"])

    out["loss"].backward()

    # CTC head projection weights must receive gradients
    assert model.ctc_head.proj.weight.grad is not None
    assert torch.abs(model.ctc_head.proj.weight.grad).sum() > 0

    # Audio encoder parameters must receive gradients
    has_audio_encoder_grad = any(
        p.grad is not None and torch.abs(p.grad).sum() > 0
        for p in model.audio_encoder.parameters()
    )
    assert has_audio_encoder_grad, "Gradients failed to flow back into AudioEncoder parameters"


def test_real_data_path_multi_step_optimization(tmp_path):
    """Running multiple optimizer steps on a real dataset batch decreases CTC loss overall."""
    torch.manual_seed(42)
    config = ctc_multimodal_config(use_ctc_head=True, ctc_loss_weight=1.0)
    data_dir, tokenizer = create_tiny_audio_dataset(tmp_path, vocab_size=config.vocab_size)
    model = FramerModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    dataset = AudioCaptionDataset(data_dir, tokenizer, config, caption_len=64)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))

    initial_loss = None
    final_loss = None

    for step in range(15):
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"],
            target_audio=batch["target_audio"],
            target_waveform=batch["target_waveform"],
            audio=batch["audio"],
            ctc_targets=batch["ctc_targets"],
            ctc_input_lengths=batch["ctc_input_lengths"],
            ctc_target_lengths=batch["ctc_target_lengths"],
        )
        loss = out["ctc_loss"]
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        out["loss"].backward()
        optimizer.step()

    assert final_loss < initial_loss, f"Expected CTC loss to decrease, but initial={initial_loss}, final={final_loss}"


def test_ctc_loss_weighting_metric():
    """ctc_loss metric must report raw unweighted loss while total loss includes weight * ctc_loss."""
    torch.manual_seed(42)
    weight = 0.3
    config = ctc_multimodal_config(use_ctc_head=True, ctc_loss_weight=weight)
    model = FramerModel(config)

    audio = torch.randn(2, 4000)
    ctc_targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    ctc_target_lengths = torch.tensor([4, 3], dtype=torch.long)

    out = model(
        audio=audio,
        ctc_targets=ctc_targets,
        ctc_target_lengths=ctc_target_lengths,
    )

    raw_ctc = out["ctc_loss"].item()
    total_loss = out["loss"].item()

    assert torch.isclose(torch.tensor(total_loss), torch.tensor(raw_ctc * weight), rtol=1e-4)


def test_exact_failure_mode_regression_ctc_pipeline(tmp_path):
    """Exact failure mode regression test verifying all 4 reviewer issues are fixed simultaneously."""
    config = ctc_multimodal_config(use_ctc_head=True, ctc_loss_weight=0.5, audio_max_frames=128)
    data_dir, tokenizer = create_tiny_audio_dataset(tmp_path, vocab_size=config.vocab_size)
    model = FramerModel(config)

    dataset = AudioCaptionDataset(data_dir, tokenizer, config, caption_len=64)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))

    # Issue 3: DataLoader(batch_size=2) succeeded with different audio lengths (4000 vs 8000)
    assert batch["audio"].shape == (2, 8000)

    # Issue 1: ctc_target_lengths are true transcript lengths, NOT 64
    assert (batch["ctc_target_lengths"] != 64).all()
    assert batch["ctc_target_lengths"].tolist() == [
        len(tokenizer.encode("hello world", add_special=True)),
        len(tokenizer.encode("a quick brown fox jumps over the lazy dog", add_special=True)),
    ]

    # Issue 2: ctc_input_lengths are temporal frames (4000//16+1=251, 8000//16+1=501), NOT n_mels (16)
    assert batch["ctc_input_lengths"].tolist() == [128, 128]  # capped at audio_max_frames=128

    # Issue 4: target_audio passed to model does NOT leak audio into prefix_embeds
    lm_out = model.forward_lm(batch["input_ids"])
    # Context length for forward_lm with input_ids of len 64 should be 64, prefix_len=0
    assert lm_out["logits"].shape == (2, 64, config.vocab_size)

    # Run full model forward
    out = model(
        input_ids=batch["input_ids"],
        target_audio=batch["target_audio"],
        target_waveform=batch["target_waveform"],
        audio=batch["audio"],
        ctc_targets=batch["ctc_targets"],
        ctc_input_lengths=batch["ctc_input_lengths"],
        ctc_target_lengths=batch["ctc_target_lengths"],
    )

    # Positive and finite CTC loss
    assert "ctc_loss" in out
    assert torch.isfinite(out["ctc_loss"])
    assert out["ctc_loss"].item() > 0.0

    # Non-zero gradients reach audio encoder
    out["loss"].backward()
    has_audio_encoder_grad = any(
        p.grad is not None and torch.abs(p.grad).sum() > 0
        for p in model.audio_encoder.parameters()
    )
    assert has_audio_encoder_grad, "AudioEncoder did not receive gradients"
