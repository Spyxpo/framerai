"""Unit tests for SFT (Supervised Fine-Tuning) pipeline."""

import os

import torch
from torch.utils.data import DataLoader

from model.configs import FramerConfig
from model.data import SFTDataset
from model.framer import FramerModel
from model.tokenizer import FramerTokenizer
from model.training.sft import train_sft


def test_sft_training_pass(tmp_path):
    config = FramerConfig.from_preset("framer-tiny")
    config.max_steps = 2
    config.batch_size = 1

    tokenizer = FramerTokenizer(vocab_size=config.vocab_size)

    sft_path = tmp_path / "sft_data.jsonl"
    sft_path.write_text(
        '{"prompt": "What is Python?", "response": "Python is a programming language."}\n'
        '{"prompt": "Count to 3.", "response": "1, 2, 3."}\n'
    )

    dataset = SFTDataset(str(sft_path), tokenizer, max_len=64)
    loader = DataLoader(dataset, batch_size=1)

    model = FramerModel(config)
    device = torch.device("cpu")

    final_step = train_sft(
        config=config,
        model=model,
        dataloader=loader,
        device=device,
        output_dir=str(tmp_path / "output"),
        log_interval=1,
        save_interval=2,
    )

    assert final_step == 2
    assert os.path.exists(tmp_path / "output" / "model_final.pt")
