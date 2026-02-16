"""Generation utilities for FramerAI inference."""

import torch
import torch.nn.functional as F
from typing import Optional
from PIL import Image
import numpy as np

from .framer import FramerModel
from .configs import FramerConfig
from .tokenizer import FramerTokenizer


class FramerGenerator:
    """High-level generation interface for FramerAI."""

    def __init__(self, model: FramerModel, tokenizer: FramerTokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str, device: str = "auto"):
        """Load model from checkpoint."""
        from .utils import get_device
        device = get_device(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", FramerConfig())
        if isinstance(config, dict):
            config = FramerConfig(**config)

        model = FramerModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer = FramerTokenizer.load(tokenizer_path)
        return cls(model, tokenizer, str(device))

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        image: torch.Tensor = None,
    ) -> str:
        """Generate text given a prompt, optionally conditioned on an image."""
        tokens = self.tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([tokens], device=self.device)

        image_embeds = None
        if image is not None:
            image_embeds = self.model.forward_vision(image.unsqueeze(0).to(self.device))

        generated = list(tokens)
        for _ in range(max_new_tokens):
            seq = torch.tensor([generated[-self.model.config.max_seq_len:]], device=self.device)
            logits = self.model.forward_text(seq, image_embeds=image_embeds)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == self.tokenizer.eos_id:
                break

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        resolution: int = 256,
    ) -> list:
        """Generate images from text prompt."""
        tokens = self.tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([tokens], device=self.device)

        # Get text context
        text_embeds = self.model.token_embed(input_ids)
        for layer in self.model.layers:
            text_embeds = layer(text_embeds)
        context = self.model.norm(text_embeds)
        context = context.expand(num_images, -1, -1)

        # Generate via diffusion
        shape = (num_images, 3, resolution, resolution)
        images = self.model.diffusion.sample(shape, context, self.device)

        # Convert to PIL images
        results = []
        for i in range(num_images):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            results.append(Image.fromarray(img))

        return results

    @torch.no_grad()
    def generate_video(
        self,
        prompt: str,
        num_frames: int = 16,
    ) -> list:
        """Generate video frames from text prompt."""
        tokens = self.tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([tokens], device=self.device)

        # Get text context
        text_embeds = self.model.token_embed(input_ids)
        for layer in self.model.layers:
            text_embeds = layer(text_embeds)
        context = self.model.norm(text_embeds)

        # Generate video
        video = self.model.video_gen.sample(1, context, self.device)

        # Convert to PIL frames
        frames = []
        video_np = video[0].cpu().permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        for i in range(video_np.shape[0]):
            frame = ((video_np[i] + 1) * 127.5).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))

        return frames

    @torch.no_grad()
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate code with lower temperature for more deterministic output."""
        code_prompt = f"<code>Write {language} code: {prompt}\n```{language}\n"
        return self.generate_text(
            code_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.95,
        )
