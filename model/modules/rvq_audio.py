"""Token-based audio generation.

A drop-in sibling of :class:`~model.modules.audio_generator.AudioGenerator`,
exposing the same ``forward(target, context) -> loss`` and
``sample(context, device)`` surface so ``FramerModel`` swaps between them
through a factory.

The mechanism is different in kind, not degree. The mel-diffusion path treats a
spectrogram as a single-channel image, denoises it, and guesses the phase back
with Griffin-Lim. This one encodes audio into discrete acoustic tokens, predicts
them as a classification problem conditioned on text, and decodes them through a
learned codec that reconstructs phase along with magnitude.
"""

import torch
import torch.nn as nn

from .audio_codec import AudioCodec
from .audio_lm import AudioTokenEmbedding, AudioTokenHead, SpeakerEncoder
from .transformer import RMSNorm, TransformerBlock
from .vocoder import NeuralVocoder


class RVQAudioGenerator(nn.Module):
    """Text-conditioned acoustic token prediction with a neural codec decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sample_rate = config.codec_sample_rate
        self.n_quantizers = config.rvq_n_quantizers
        self.codebook_size = config.rvq_codebook_size

        self.codec = AudioCodec(
            base_channels=config.codec_base_channels,
            latent_dim=config.rvq_codebook_dim,
            hop=config.codec_hop,
            n_quantizers=config.rvq_n_quantizers,
            codebook_size=config.rvq_codebook_size,
            sample_rate=config.codec_sample_rate,
            quantizer_dropout=config.rvq_quantizer_dropout,
        )

        d_model = config.audio_lm_d_model
        self.token_embed = AudioTokenEmbedding(
            config.rvq_n_quantizers, config.rvq_codebook_size, d_model
        )
        self.context_proj = nn.Linear(config.d_model, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, config.audio_lm_n_heads, d_model * 4,
                max_seq_len=config.audio_gen_frames * 4, dropout=config.dropout,
            )
            for _ in range(config.audio_lm_n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = AudioTokenHead(d_model, config.rvq_n_quantizers, config.rvq_codebook_size)

        self.speaker_encoder = (
            SpeakerEncoder(config.rvq_codebook_dim, config.speaker_embed_dim)
            if config.use_speaker_conditioning
            else None
        )
        self.speaker_proj = (
            nn.Linear(config.speaker_embed_dim, d_model)
            if config.use_speaker_conditioning
            else None
        )

        self.vocoder = (
            NeuralVocoder(
                in_channels=config.rvq_codebook_dim,
                d_model=config.vocoder_d_model,
                n_layers=config.vocoder_n_layers,
                n_fft=config.audio_n_fft * 2,
                hop_length=config.codec_hop,
            )
            if config.vocoder_arch == "istft"
            else None
        )

    # ------------------------------------------------------------------

    def _run_lm(self, tokens, text_context, speaker=None):
        """Prefix the conditioning, run the stack, return per-frame hidden states."""
        x = self.token_embed(tokens)
        prefix = []
        if text_context is not None:
            prefix.append(self.context_proj(text_context))
        if speaker is not None and self.speaker_proj is not None:
            prefix.append(self.speaker_proj(speaker).unsqueeze(1))

        prefix_len = 0
        if prefix:
            prefix_embeds = torch.cat(prefix, dim=1)
            prefix_len = prefix_embeds.shape[1]
            x = torch.cat([prefix_embeds, x], dim=1)

        for block in self.blocks:
            x = block(x)["x"]

        return self.norm(x)[:, prefix_len:]

    def forward(self, target, context: torch.Tensor = None) -> torch.Tensor:
        """Training loss.

        ``target`` is a waveform ``(B, N)``. Codes are produced under ``no_grad``
        by the codec, which is trained separately by :meth:`codec_loss`, and the
        language model is trained to predict them.
        """
        if target.dim() == 3:
            # A mel spectrogram from the legacy pipeline; nothing to tokenize.
            raise ValueError(
                "audio_gen_arch='rvq_lm' trains on waveforms, not mel spectrograms"
            )

        with torch.no_grad():
            codes = self.codec.encode(target)

        speaker = None
        if self.speaker_encoder is not None:
            with torch.no_grad():
                latent = self.codec.encode_latent(target)
            speaker = self.speaker_encoder(latent)

        # Teacher forcing: predict frame t from frames < t.
        inputs = codes[:, :, :-1]
        labels = codes[:, :, 1:]
        hidden = self._run_lm(inputs, context, speaker)
        return self.head.loss(hidden, labels)

    def codec_loss(self, waveform: torch.Tensor, commit_weight: float = 1.0) -> torch.Tensor:
        """Reconstruction plus commitment, for training the codec on its own."""
        recon, commit_loss, _ = self.codec(waveform)
        length = min(recon.shape[-1], waveform.shape[-1])
        recon_loss = torch.nn.functional.mse_loss(
            recon[..., :length], waveform[..., :length]
        )
        return recon_loss + commit_weight * commit_loss

    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor = None,
        device: str = "cpu",
        batch_size: int = 1,
        n_frames: int = None,
        temperature: float = 0.9,
        speaker: torch.Tensor = None,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """Autoregressively generate acoustic tokens, then decode a waveform."""
        n_frames = n_frames or self.config.audio_gen_frames
        codes = torch.zeros(
            batch_size, self.n_quantizers, 1, dtype=torch.long, device=device
        )

        for _ in range(n_frames):
            hidden = self._run_lm(codes, context, speaker)
            logits = self.head(hidden)[:, :, -1] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            flat = probs.reshape(-1, self.codebook_size)
            picked = torch.multinomial(flat, 1, generator=generator)
            codes = torch.cat(
                [codes, picked.view(batch_size, self.n_quantizers, 1)], dim=-1
            )

        return self.decode(codes[:, :, 1:])

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Acoustic tokens to a waveform, through the vocoder when configured."""
        if self.vocoder is not None:
            latent = self.codec.quantizer.decode(codes)
            return self.vocoder(latent)
        return self.codec.decode(codes)


def build_audio_generator(config):
    """Return the audio generator the config selects."""
    from .audio_generator import AudioGenerator

    if config.audio_gen_arch == "rvq_lm":
        return RVQAudioGenerator(config)
    return AudioGenerator(
        n_mels=config.audio_n_mels,
        n_frames=config.audio_gen_frames,
        base_channels=config.audio_gen_channels,
        context_dim=config.d_model,
        num_steps=config.diffusion_steps,
        sample_rate=config.audio_sample_rate,
        n_fft=config.audio_n_fft,
        hop_length=config.audio_hop_length,
        vocoder_arch=config.vocoder_arch,
        vocoder_d_model=config.vocoder_d_model,
        vocoder_n_layers=config.vocoder_n_layers,
    )
