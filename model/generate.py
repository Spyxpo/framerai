"""Generation utilities for FramerAI inference."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .configs import FramerConfig
from .framer import FramerModel
from .tokenizer import FramerTokenizer
from .utils.image_request import resolve_image_request


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k then nucleus (top-p) filtering to a (1, V) logits row."""
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        idx_remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(idx_remove, float("-inf"))
    return logits


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
            config = FramerConfig.from_dict(config)

        model = FramerModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer = FramerTokenizer.load(tokenizer_path)
        return cls(model, tokenizer, str(device))

    def _text_context(self, prompt: str) -> torch.Tensor:
        """Encode a prompt to the LM's hidden states for conditioning generators."""
        tokens = self.tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([tokens], device=self.device)
        return self.model.forward_lm(input_ids)["hidden"]

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        image: torch.Tensor = None,
        audio: torch.Tensor = None,
    ) -> str:
        """Generate text, optionally conditioned on an image or audio.

        Uses an incremental KV cache: the prompt is encoded once, then each new
        token attends to the cache in O(1) rather than re-running the whole
        prefix every step.
        """
        tokens = self.tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([tokens], device=self.device)

        prefix_parts = []
        if image is not None:
            prefix_parts.append(self.model.forward_vision(image.unsqueeze(0).to(self.device)))
        if audio is not None:
            prefix_parts.append(self.model.forward_audio(audio.unsqueeze(0).to(self.device)))
        prefix_embeds = torch.cat(prefix_parts, dim=1) if prefix_parts else None

        # Prefill the prompt (+ modality prefix) and seed the cache.
        out = self.model.forward_lm(input_ids, prefix_embeds=prefix_embeds, use_cache=True)
        past = out["past_kvs"]
        logits = out["logits"][:, -1, :]

        generated = list(tokens)
        for _ in range(max_new_tokens):
            step = logits / max(temperature, 1e-6)
            step = _filter_logits(step, top_k, top_p)
            probs = F.softmax(step, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if next_token == self.tokenizer.eos_id:
                break
            nxt = torch.tensor([[next_token]], device=self.device)
            out = self.model.forward_lm(nxt, past_kvs=past, use_cache=True)
            past = out["past_kvs"]
            logits = out["logits"][:, -1, :]

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        width: int = None,
        height: int = None,
        aspect: str = None,
        tier: int = None,
        seed: int = None,
        resolution: int = None,
        return_request: bool = False,
    ):
        """Generate images from a text prompt at a requested size.

        Size comes from explicit ``width``/``height``, then an ``aspect`` at a
        size ``tier``, then whatever the prompt itself asks for ("make it 16:9",
        "1024x1024", "portrait"), then the configured default. ``resolution`` is
        the old square-only parameter, still accepted.

        Set ``return_request`` to also receive the resolved
        :class:`~model.utils.image_request.ImageRequest`, which records what was
        understood and where it came from.
        """
        if resolution is not None and width is None and height is None:
            width = height = int(resolution)

        request = resolve_image_request(
            prompt,
            width=width, height=height, aspect=aspect, tier=tier,
            num_images=num_images, seed=seed, config=self.model.config,
        )
        self._require_size_support(request)

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(request.seed))

        context = self._text_context(request.prompt).expand(request.num_images, -1, -1)
        images = self._sample_images(request, context, generator)

        results = []
        for i in range(request.num_images):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            results.append(Image.fromarray(img))
        return (results, request) if return_request else results

    def _require_size_support(self, request):
        """The pixel U-Net only ever supported square output at small sizes."""
        if self.model.config.image_gen_arch == "latent_dit":
            return
        if request.width != request.height:
            raise ValueError(
                f"image_gen_arch='unet' generates square images only; "
                f"{request.width}x{request.height} was requested. "
                "Set image_gen_arch='latent_dit' for arbitrary aspect ratios."
            )

    def _sample_images(self, request, context, generator):
        """Sample, passing seed and guidance through when the decoder takes them."""
        sampler = self.model.diffusion.sample
        try:
            return sampler(request.shape, context, self.device, generator=generator)
        except TypeError:
            # The pixel U-Net's sampler has no seed parameter.
            return sampler(request.shape, context, self.device)

    @torch.no_grad()
    def generate_video(
        self,
        prompt: str,
        num_frames: int = None,
        width: int = None,
        height: int = None,
        aspect: str = None,
        tier: int = None,
        fps: int = None,
        seed: int = None,
        return_request: bool = False,
    ):
        """Generate video frames from a text prompt at a requested size.

        Size resolves exactly as it does for images, reusing the same buckets and
        prompt parsing; ``num_frames`` and ``fps`` control duration separately.
        """
        config = self.model.config
        request = resolve_image_request(
            prompt, width=width, height=height, aspect=aspect, tier=tier, seed=seed, config=config
        )
        frames_requested = int(num_frames or config.video_frames)
        self._require_video_size_support(request, frames_requested)

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(request.seed))

        context = self._text_context(request.prompt)
        video = self._sample_video(request, frames_requested, fps, context, generator)

        frames = []
        video_np = video[0].cpu().permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        for i in range(video_np.shape[0]):
            frame = ((video_np[i] + 1) * 127.5).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))
        return (frames, request) if return_request else frames

    def _require_video_size_support(self, request, frames):
        """The 3D U-Net was built for one fixed square resolution."""
        config = self.model.config
        if config.video_gen_arch == "spacetime_dit":
            return
        if request.width != request.height or request.width != config.video_resolution:
            raise ValueError(
                f"video_gen_arch='unet3d' generates {config.video_resolution}x"
                f"{config.video_resolution} only; {request.width}x{request.height} was "
                "requested. Set video_gen_arch='spacetime_dit' for arbitrary sizes."
            )
        if frames != config.video_frames:
            raise ValueError(
                f"video_gen_arch='unet3d' generates {config.video_frames} frames only. "
                "Set video_gen_arch='spacetime_dit' for variable duration."
            )

    def _sample_video(self, request, frames, fps, context, generator):
        """Sample, passing duration and size through when the decoder takes them."""
        sampler = self.model.video_gen.sample
        try:
            return sampler(
                1, context, self.device,
                frames=frames, height=request.height, width=request.width,
                fps=fps, generator=generator,
            )
        except TypeError:
            # The 3D U-Net's sampler takes only (batch_size, context, device).
            return sampler(1, context, self.device)

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

    @torch.no_grad()
    def generate_audio(self, prompt: str) -> tuple:
        """Generate an audio waveform from a text prompt.

        Returns a tuple of (waveform numpy array, sample_rate).
        """
        context = self._text_context(prompt)
        mel = self.model.audio_gen.sample(context, self.device)
        waveform = self.model.audio_gen.mel_to_waveform(mel)
        return waveform.cpu().numpy(), self.model.audio_gen.sample_rate

    @torch.no_grad()
    def transcribe(
        self,
        audio: torch.Tensor,
        prompt: str = "<audio><audio_end>Transcribe the audio:",
        max_new_tokens: int = 256,
    ) -> str:
        """Transcribe or describe audio (audio understanding -> text)."""
        return self.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_k=40,
            top_p=0.95,
            audio=audio,
        )
