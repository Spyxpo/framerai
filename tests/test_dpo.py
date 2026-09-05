"""Unit tests for DPO (Direct Preference Optimization) pipeline."""

import os

import torch
from torch.utils.data import DataLoader

from model.configs import FramerConfig
from model.data import DPODataset
from model.framer import FramerModel
from model.tokenizer import FramerTokenizer
from model.training.dpo import compute_dpo_loss, get_batch_logps, train_dpo


def test_compute_dpo_loss():
    pi_chosen = torch.tensor([-1.5, -2.0])
    pi_rejected = torch.tensor([-3.0, -3.5])
    ref_chosen = torch.tensor([-1.8, -2.1])
    ref_rejected = torch.tensor([-2.5, -2.8])

    loss, chosen_r, rejected_r = compute_dpo_loss(
        pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1
    )

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    assert chosen_r.shape == (2,)
    assert rejected_r.shape == (2,)


def test_get_batch_logps_masking():
    logits = torch.randn(2, 5, 10)
    labels = torch.tensor([[1, 2, 3, 4, 5], [1, 2, -100, -100, -100]])

    logps = get_batch_logps(logits, labels, ignore_index=-100)
    assert logps.shape == (2,)


def test_dpo_training_pass(tmp_path):
    config = FramerConfig.from_preset("framer-tiny")
    config.max_steps = 2
    config.batch_size = 1

    tokenizer = FramerTokenizer(vocab_size=config.vocab_size)

    dpo_path = tmp_path / "dpo_data.jsonl"
    dpo_path.write_text(
        '{"prompt": "Hello", "chosen": "Hi there!", "rejected": "Go away."}\n'
        '{"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"}\n'
    )

    dataset = DPODataset(str(dpo_path), tokenizer, max_len=64)
    loader = DataLoader(dataset, batch_size=1)

    device = torch.device("cpu")
    policy_model = FramerModel(config).to(device)

    # Construct reference model using from_config_meta
    ref_model = FramerModel.from_config_meta(config)
    ref_model.to_empty(device=device)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Assert reference model is frozen
    assert all(not p.requires_grad for p in ref_model.parameters())

    final_step = train_dpo(
        config=config,
        policy_model=policy_model,
        ref_model=ref_model,
        dataloader=loader,
        device=device,
        output_dir=str(tmp_path / "output"),
        beta=0.1,
        log_interval=1,
        save_interval=2,
    )

    assert final_step == 2
    assert os.path.exists(tmp_path / "output" / "model_dpo_final.pt")


def test_dpo_attention_mask_propagation(tmp_path):
    config = FramerConfig.from_preset("framer-tiny")
    config.max_steps = 1
    config.batch_size = 1
    config.gradient_accumulation_steps = 1

    tokenizer = FramerTokenizer(vocab_size=config.vocab_size)

    dpo_path = tmp_path / "dpo_mask_data.jsonl"
    dpo_path.write_text(
        '{"prompt": "Short", "chosen": "Good response", "rejected": "Bad response"}\n'
    )

    max_len = 32
    dataset = DPODataset(str(dpo_path), tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=1)

    # Check sample keys and shapes
    sample = dataset[0]
    assert "chosen_attention_mask" in sample
    assert "rejected_attention_mask" in sample
    assert sample["chosen_attention_mask"].shape == (max_len,)
    assert sample["rejected_attention_mask"].shape == (max_len,)

    # Verify padding positions in mask are 0 and token positions are 1
    chosen_tokens = sample["chosen_input_ids"]
    chosen_mask = sample["chosen_attention_mask"]
    pad_count = (chosen_tokens == tokenizer.pad_id).sum().item()
    assert pad_count > 0, "Test sample should be padded to test attention mask"
    assert (chosen_mask[chosen_tokens == tokenizer.pad_id] == 0).all()
    assert (chosen_mask[chosen_tokens != tokenizer.pad_id] == 1).all()

    device = torch.device("cpu")
    policy_model = FramerModel(config).to(device)
    ref_model = FramerModel.from_config_meta(config)
    ref_model.to_empty(device=device)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()

    policy_forward_masks = []
    ref_forward_masks = []

    orig_policy_forward = policy_model.forward

    def policy_forward_wrapper(*args, **kwargs):
        policy_forward_masks.append(kwargs.get("attention_mask"))
        return orig_policy_forward(*args, **kwargs)

    orig_ref_forward = ref_model.forward

    def ref_forward_wrapper(*args, **kwargs):
        ref_forward_masks.append(kwargs.get("attention_mask"))
        return orig_ref_forward(*args, **kwargs)

    policy_model.forward = policy_forward_wrapper
    ref_model.forward = ref_forward_wrapper

    train_dpo(
        config=config,
        policy_model=policy_model,
        ref_model=ref_model,
        dataloader=loader,
        device=device,
        output_dir=str(tmp_path / "output"),
        beta=0.1,
        log_interval=1,
        save_interval=2,
    )

    # Each step makes 2 forward calls for policy (chosen, rejected) and 2 for ref (chosen, rejected)
    assert len(policy_forward_masks) == 2
    assert len(ref_forward_masks) == 2

    # Check chosen attention mask passed to policy and ref
    assert policy_forward_masks[0] is not None
    assert ref_forward_masks[0] is not None
    assert policy_forward_masks[0].shape == (1, max_len)
    assert ref_forward_masks[0].shape == (1, max_len)
    assert torch.equal(policy_forward_masks[0], ref_forward_masks[0])

    # Check rejected attention mask passed to policy and ref
    assert policy_forward_masks[1] is not None
    assert ref_forward_masks[1] is not None
    assert policy_forward_masks[1].shape == (1, max_len)
    assert ref_forward_masks[1].shape == (1, max_len)
    assert torch.equal(policy_forward_masks[1], ref_forward_masks[1])
