"""
Knowledge distillation trainer for FramerAI.

Trains the FramerAI student model using a locally-loaded open-source teacher
model's logits and hidden states. Supports:

- True soft-label distillation (KL divergence on teacher logits)
- Hard label loss (cross-entropy on teacher text)
- Feature distillation (aligning hidden representations)
- Progressive sequence length training (curriculum learning)
- RoPE scaling for extended context (up to 1M+ tokens)
- Vocabulary projection (handles teacher/student vocab mismatch)
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..configs import FramerConfig
from ..framer import FramerModel
from ..tokenizer import FramerTokenizer
from ..utils import get_device, count_parameters, save_checkpoint
from .teacher_models import TeacherModel

logger = logging.getLogger("framerai.distillation")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""
    # Data
    data_path: str = "data/distillation/distillation_data.jsonl"
    output_dir: str = "checkpoints"

    # Training
    max_steps: int = 50000
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Distillation
    temperature: float = 2.0
    alpha_hard: float = 0.5          # Weight for hard label loss (cross-entropy on text)
    alpha_soft: float = 0.5          # Weight for soft label loss (KL div on teacher logits)
    alpha_feature: float = 0.0       # Weight for hidden state alignment loss

    # Progressive training
    progressive_seq_len: bool = True
    initial_seq_len: int = 512
    target_seq_len: int = 2048
    seq_len_warmup_steps: int = 10000

    # Context extension
    rope_scaling_factor: float = 1.0
    rope_scaling_type: str = "linear"   # linear or dynamic (NTK-aware)

    # Teacher
    teacher_batch_size: int = 2         # Smaller batch for teacher (uses more VRAM)

    # Logging & saving
    log_interval: int = 10
    save_interval: int = 1000

    # Device
    device: str = "auto"
    mixed_precision: bool = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DistillationDataset(Dataset):
    """Dataset that loads teacher-generated text for distillation training."""

    def __init__(
        self,
        data_path: str,
        student_tokenizer: FramerTokenizer,
        max_seq_len: int = 2048,
    ):
        self.student_tokenizer = student_tokenizer
        self.max_seq_len = max_seq_len
        self.current_max_len = max_seq_len
        self.samples = []

        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} distillation samples")

    def set_max_len(self, max_len: int):
        self.current_max_len = min(max_len, self.max_seq_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")

        # For conversation data, flatten
        if "conversation" in sample:
            parts = []
            for msg in sample["conversation"]:
                role_tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
                parts.append(f"{role_tag}{msg['content']}")
            text = "".join(parts) + "<|end|>"
        else:
            text = f"<|user|>{prompt}<|assistant|>{response}<|end|>"

        tokens = self.student_tokenizer.encode(text, add_special=True)

        # Find where response starts (for label masking)
        if "conversation" not in sample:
            prompt_text = f"<|user|>{prompt}<|assistant|>"
            prompt_tokens = self.student_tokenizer.encode(prompt_text, add_special=True)
            prompt_len = len(prompt_tokens)
        else:
            prompt_len = 0  # Train on full conversation

        max_len = self.current_max_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        pad_len = max_len - len(tokens)
        tokens += [self.student_tokenizer.pad_id] * pad_len

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Mask prompt in labels
        if prompt_len > 1:
            labels[:min(prompt_len - 1, len(labels))] = -100

        # Also store the raw text for teacher tokenization
        return {
            "input_ids": input_ids,
            "labels": labels,
            "text": text,
        }


# ---------------------------------------------------------------------------
# Vocabulary Projector
# ---------------------------------------------------------------------------

class VocabProjector(nn.Module):
    """
    Projects teacher logits (teacher vocab) to student vocab space.

    Since teacher and student have different tokenizers and vocab sizes,
    we learn a lightweight projection to align them.
    """

    def __init__(self, teacher_vocab_size: int, student_vocab_size: int, hidden_dim: int = 1024):
        super().__init__()
        # Use a small MLP to project between vocab spaces
        self.proj = nn.Sequential(
            nn.Linear(teacher_vocab_size, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, student_vocab_size, bias=False),
        )

    def forward(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        return self.proj(teacher_logits)


class FeatureProjector(nn.Module):
    """Projects teacher hidden states to student hidden dimension."""

    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(teacher_hidden)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """
    Trains FramerAI by distilling knowledge from a local open-source teacher model.

    The teacher model runs inference to produce logits and hidden states,
    and the student (FramerAI) is trained to match them.
    """

    def __init__(
        self,
        student: FramerModel,
        student_tokenizer: FramerTokenizer,
        teacher: TeacherModel,
        config: DistillationConfig,
    ):
        self.student = student
        self.student_tokenizer = student_tokenizer
        self.teacher = teacher
        self.config = config
        self.device = get_device(config.device)

        self.student = self.student.to(self.device)

        # Vocab projector (teacher vocab -> student vocab)
        self.vocab_projector = VocabProjector(
            teacher.vocab_size, student.config.vocab_size,
            hidden_dim=min(1024, teacher.vocab_size // 4, student.config.vocab_size // 4),
        ).to(self.device)

        # Feature projector (optional, for hidden state alignment)
        self.feature_projector = None
        if config.alpha_feature > 0:
            self.feature_projector = FeatureProjector(
                teacher.hidden_size, student.config.d_model,
            ).to(self.device)

        # Apply RoPE scaling for extended context
        if config.rope_scaling_factor > 1.0:
            self._apply_rope_scaling(config.rope_scaling_factor, config.rope_scaling_type)

    def _apply_rope_scaling(self, factor: float, scaling_type: str):
        """Scale RoPE embeddings for extended context length."""
        logger.info(f"Applying RoPE scaling: {factor}x ({scaling_type})")
        for module in self.student.modules():
            if hasattr(module, "inv_freq"):
                if scaling_type == "linear":
                    module.inv_freq = module.inv_freq / factor
                elif scaling_type == "dynamic":
                    base = 10000.0
                    dim = module.inv_freq.shape[0] * 2
                    new_base = base * (factor ** (dim / (dim - 2)))
                    module.inv_freq = 1.0 / (
                        new_base ** (torch.arange(0, dim, 2, device=module.inv_freq.device).float() / dim)
                    )
                # Rebuild cos/sin cache
                max_seq_len = int(self.student.config.max_seq_len * factor)
                t = torch.arange(max_seq_len, device=module.inv_freq.device).float()
                freqs = torch.einsum("i,j->ij", t, module.inv_freq)
                emb = torch.cat([freqs, freqs], dim=-1)
                if hasattr(module, "cos_cached"):
                    module.cos_cached = emb.cos()
                    module.sin_cached = emb.sin()

    def _get_current_seq_len(self, step: int) -> int:
        if not self.config.progressive_seq_len:
            return self.config.target_seq_len
        if step >= self.config.seq_len_warmup_steps:
            return self.config.target_seq_len
        progress = step / self.config.seq_len_warmup_steps
        current = int(
            self.config.initial_seq_len
            + (self.config.target_seq_len - self.config.initial_seq_len) * progress
        )
        return max(self.config.initial_seq_len, (current // 64) * 64)

    @torch.no_grad()
    def _get_teacher_outputs(self, texts: list, max_len: int):
        """Run teacher model on texts to get logits (and optionally hidden states)."""
        teacher_tok = self.teacher.tokenizer
        encoded = teacher_tok(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_len,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if self.config.alpha_feature > 0:
            logits, hidden = self.teacher.get_logits_and_hidden(input_ids, attention_mask)
            return logits, hidden
        else:
            logits = self.teacher.get_logits(input_ids, attention_mask)
            return logits, None

    def _compute_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor = None,
        student_hidden: torch.Tensor = None,
        teacher_hidden: torch.Tensor = None,
    ) -> dict:
        """Compute combined distillation loss."""
        losses = {}
        cfg = self.config

        # Hard label loss
        if cfg.alpha_hard > 0:
            hard_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            losses["hard"] = hard_loss

        # Soft label loss (KL divergence on projected teacher logits)
        if teacher_logits is not None and cfg.alpha_soft > 0:
            T = cfg.temperature
            # Project teacher logits to student vocab space
            # Truncate/pad sequence dimension to match
            seq_len = min(student_logits.size(1), teacher_logits.size(1))
            s_logits = student_logits[:, :seq_len, :]
            t_logits = teacher_logits[:, :seq_len, :].to(self.device)

            # Project teacher vocab -> student vocab
            t_projected = self.vocab_projector(t_logits)

            student_log_probs = F.log_softmax(s_logits / T, dim=-1)
            teacher_probs = F.softmax(t_projected / T, dim=-1)

            # Mask padding
            mask = (labels[:, :seq_len] != -100).unsqueeze(-1).float()
            soft_loss = F.kl_div(
                student_log_probs * mask,
                teacher_probs * mask,
                reduction="batchmean",
            ) * (T * T)
            losses["soft"] = soft_loss

        # Feature alignment loss
        if (teacher_hidden is not None and student_hidden is not None
                and self.feature_projector is not None and cfg.alpha_feature > 0):
            seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
            s_feat = student_hidden[:, :seq_len, :]
            t_feat = teacher_hidden[:, :seq_len, :].to(self.device)
            t_projected = self.feature_projector(t_feat)

            feature_loss = F.mse_loss(s_feat, t_projected)
            losses["feature"] = feature_loss

        # Weighted total
        total = torch.tensor(0.0, device=self.device)
        if "hard" in losses:
            total = total + cfg.alpha_hard * losses["hard"]
        if "soft" in losses:
            total = total + cfg.alpha_soft * losses["soft"]
        if "feature" in losses:
            total = total + cfg.alpha_feature * losses["feature"]
        losses["total"] = total

        return losses

    def train(self, resume_from: str = None):
        """Run the full distillation training loop."""
        cfg = self.config

        # Dataset
        dataset = DistillationDataset(
            cfg.data_path, self.student_tokenizer,
            max_seq_len=cfg.target_seq_len,
        )

        def collate_fn(batch):
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "texts": [b["text"] for b in batch],
            }

        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collate_fn,
        )

        # Optimizer (includes vocab projector + feature projector params)
        all_params = list(self.student.parameters()) + list(self.vocab_projector.parameters())
        if self.feature_projector:
            all_params += list(self.feature_projector.parameters())

        no_decay = ["bias", "norm", "LayerNorm"]
        param_groups = [
            {
                "params": [p for n, p in self.student.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
            {
                "params": list(self.vocab_projector.parameters())
                          + (list(self.feature_projector.parameters()) if self.feature_projector else []),
                "weight_decay": 0.0,
                "lr": cfg.learning_rate * 3,  # Projectors learn faster
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate, betas=(0.9, 0.95))

        # LR schedule: warmup + cosine
        def lr_schedule(step):
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

        # Mixed precision
        scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision and self.device.type == "cuda" else None

        # Resume
        step = 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
            self.student.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "vocab_projector_state_dict" in ckpt:
                self.vocab_projector.load_state_dict(ckpt["vocab_projector_state_dict"])
            step = ckpt.get("step", 0)
            logger.info(f"Resumed from {resume_from} at step {step}")

        # Log config
        logger.info("=" * 60)
        logger.info("Knowledge Distillation (Local Open-Source Teacher)")
        logger.info(f"  Teacher: {self.teacher.name} ({self.teacher.vocab_size} vocab)")
        logger.info(f"  Student: FramerAI ({count_parameters(self.student):,} params, {self.student.config.vocab_size} vocab)")
        logger.info(f"  Data: {cfg.data_path} ({len(dataset)} samples)")
        logger.info(f"  Steps: {cfg.max_steps} | Batch: {cfg.batch_size}x{cfg.gradient_accumulation_steps}")
        logger.info(f"  LR: {cfg.learning_rate} | Temperature: {cfg.temperature}")
        logger.info(f"  Loss weights — hard: {cfg.alpha_hard}, soft: {cfg.alpha_soft}, feature: {cfg.alpha_feature}")
        if cfg.progressive_seq_len:
            logger.info(f"  Progressive seq len: {cfg.initial_seq_len} -> {cfg.target_seq_len}")
        if cfg.rope_scaling_factor > 1.0:
            logger.info(f"  RoPE scaling: {cfg.rope_scaling_factor}x ({cfg.rope_scaling_type})")
        logger.info("=" * 60)

        self.student.train()
        self.vocab_projector.train()
        if self.feature_projector:
            self.feature_projector.train()

        total_loss = 0
        loss_components = {"hard": 0, "soft": 0, "feature": 0}
        start_time = time.time()

        while step < cfg.max_steps:
            if cfg.progressive_seq_len:
                new_len = self._get_current_seq_len(step)
                dataset.set_max_len(new_len)

            for batch in loader:
                if step >= cfg.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                texts = batch["texts"]

                # Get teacher outputs
                max_len = input_ids.size(1) + 1
                teacher_logits, teacher_hidden = self._get_teacher_outputs(texts, max_len)

                # Student forward
                if scaler:
                    with torch.amp.autocast("cuda"):
                        results = self.student(input_ids=input_ids, labels=labels)
                        student_logits = results["logits"]
                        losses = self._compute_loss(
                            student_logits, labels,
                            teacher_logits=teacher_logits,
                            teacher_hidden=teacher_hidden,
                        )
                        loss = losses["total"] / cfg.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    results = self.student(input_ids=input_ids, labels=labels)
                    student_logits = results["logits"]
                    losses = self._compute_loss(
                        student_logits, labels,
                        teacher_logits=teacher_logits,
                        teacher_hidden=teacher_hidden,
                    )
                    loss = losses["total"] / cfg.gradient_accumulation_steps
                    loss.backward()

                total_loss += loss.item()
                for k in loss_components:
                    if k in losses:
                        loss_components[k] += losses[k].item()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            list(self.student.parameters()) + list(self.vocab_projector.parameters()),
                            cfg.max_grad_norm,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(
                            list(self.student.parameters()) + list(self.vocab_projector.parameters()),
                            cfg.max_grad_norm,
                        )
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                step += 1

                if step % cfg.log_interval == 0:
                    n = cfg.log_interval
                    avg = total_loss / n
                    elapsed = time.time() - start_time
                    speed = step / elapsed if elapsed > 0 else 0
                    lr = scheduler.get_last_lr()[0]
                    seq_len = self._get_current_seq_len(step) if cfg.progressive_seq_len else cfg.target_seq_len
                    parts = [f"Step {step}/{cfg.max_steps}", f"Loss: {avg:.4f}"]
                    for k in ("hard", "soft", "feature"):
                        if loss_components[k] > 0:
                            parts.append(f"{k}: {loss_components[k] / n:.4f}")
                    parts += [f"LR: {lr:.2e}", f"SeqLen: {seq_len}", f"{speed:.1f} steps/s"]
                    logger.info(" | ".join(parts))
                    total_loss = 0
                    loss_components = {k: 0 for k in loss_components}

                if step % cfg.save_interval == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"distill_ckpt_{step}.pt")
                    torch.save({
                        "model_state_dict": self.student.state_dict(),
                        "vocab_projector_state_dict": self.vocab_projector.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "teacher_model": self.teacher.name,
                    }, ckpt_path)
                    logger.info(f"Saved: {ckpt_path}")

        # Final save
        final_path = os.path.join(cfg.output_dir, "distill_final.pt")
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "config": vars(self.student.config) if hasattr(self.student.config, "__dict__") else {},
            "step": step,
            "teacher_model": self.teacher.name,
        }, final_path)
        logger.info(f"Distillation complete. Final model: {final_path}")
        return self.student
