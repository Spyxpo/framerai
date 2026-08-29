"""Unit tests for SFT (Supervised Fine-Tuning) pipeline and loss mask alignment."""

import os

import torch
from torch.utils.data import DataLoader

from model.configs import FramerConfig
from model.data import SFTDataset
from model.framer import FramerModel
from model.tokenizer import ChatTemplate, FramerTokenizer
from model.training.sft import train_sft


def test_sft_dataset_loss_mask_and_label_alignment(tmp_path):
    """Regression test ensuring SFT dataset pre-shifts labels for next-token prediction and masks prompt tokens."""
    tokenizer = FramerTokenizer(vocab_size=400)
    sft_path = tmp_path / "sft_data.jsonl"
    sft_path.write_text(
        '{"prompt": "What is Python?", "response": "Python is a programming language."}\n'
    )

    dataset = SFTDataset(str(sft_path), tokenizer, max_len=64)
    item = dataset[0]

    input_ids = item["input_ids"]
    labels = item["labels"]

    assert len(input_ids) == len(labels)

    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    template = ChatTemplate("v1")
    encoded = template.encode_conversation(messages, tokenizer, max_len=64)

    assert torch.equal(input_ids, encoded["input_ids"])
    assert torch.equal(labels, encoded["labels"])

    # 1. Find the first non-masked label index
    non_masked_indices = (labels != -100).nonzero(as_tuple=True)[0]
    assert len(non_masked_indices) > 0, "Labels must contain non-masked assistant target tokens"

    first_target_idx = non_masked_indices[0].item()

    # 2. All prompt positions (0 .. first_target_idx - 1) in labels MUST be masked with -100
    for i in range(first_target_idx):
        assert labels[i].item() == -100, f"Prompt position at index {i} must be masked with -100"

    # 3. The first non-masked label MUST correspond to the first assistant token
    ast_ids = tokenizer.encode("<assistant>Python is a programming language.", add_special=False)
    assert labels[first_target_idx].item() == ast_ids[0], (
        f"Position {first_target_idx} label must predict first assistant token {ast_ids[0]}, "
        f"got {labels[first_target_idx].item()}"
    )

    # 4. Verify next-token pre-shifting: input_ids[first_target_idx] (last prompt token) != labels[first_target_idx]
    # (This proves next-token prediction alignment, which would FAIL under the old same-index copy objective!)
    assert input_ids[first_target_idx].item() != labels[first_target_idx].item()

    # 5. Verify assistant target token sequence alignment
    for j in range(len(ast_ids)):
        idx = first_target_idx + j
        assert labels[idx].item() == ast_ids[j], f"Label at {idx} must be assistant token ast_ids[{j}]"

    # 6. Final label must be EOS token ID
    assert labels[first_target_idx + len(ast_ids)].item() == tokenizer.eos_id


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
