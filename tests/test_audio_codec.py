"""RVQ audio codec, ISTFT vocoder, and token-based audio generation.

Mel diffusion plus Griffin-Lim discards phase and then guesses it back, which
caps output quality no matter how well the model trains. These tests cover the
replacement: residual quantization into discrete acoustic tokens, a decoder that
reconstructs phase, and the heads that make audio controllable rather than
merely generated.
"""

import pytest
import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.modules.audio_codec import AudioCodec, CausalConv1d, _strides_for
from model.modules.audio_lm import AudioTokenEmbedding, AudioTokenHead, CTCHead, SpeakerEncoder
from model.modules.rvq import ResidualVQ, VectorQuantizer
from model.modules.rvq_audio import RVQAudioGenerator, build_audio_generator
from model.modules.vocoder import ISTFTHead, NeuralVocoder


def audio_config(**overrides):
    base = dict(
        text_only=False,
        audio_gen_arch="rvq_lm",
        image_size=32,
        patch_size=16,
        vision_d_model=32,
        vision_n_heads=4,
        vision_n_layers=1,
        audio_n_fft=64,
        audio_hop_length=16,
        audio_n_mels=16,
        audio_max_frames=32,
        audio_d_model=32,
        audio_n_heads=4,
        audio_n_layers=1,
        diffusion_steps=10,
        diffusion_channels=64,
        video_frames=2,
        video_resolution=16,
        audio_gen_frames=8,
        audio_gen_channels=32,
        codec_base_channels=8,
        codec_hop=32,
        rvq_n_quantizers=2,
        rvq_codebook_size=16,
        rvq_codebook_dim=16,
        audio_lm_d_model=32,
        audio_lm_n_layers=1,
        audio_lm_n_heads=4,
        vocoder_d_model=32,
        vocoder_n_layers=1,
    )
    base.update(overrides)
    return tiny_config(**base)


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------


def test_quantizer_returns_codebook_entries():
    quantizer = VectorQuantizer(codebook_size=8, codebook_dim=4).eval()
    z = torch.randn(2, 4, 5)
    quantized, indices, commit = quantizer(z)

    assert quantized.shape == z.shape
    assert indices.shape == (2, 5)
    assert indices.min() >= 0 and indices.max() < 8
    assert commit.ndim == 0 and torch.isfinite(commit)


def test_quantization_is_nearest_neighbour():
    quantizer = VectorQuantizer(codebook_size=4, codebook_dim=3).eval()
    with torch.no_grad():
        quantizer.codebook.copy_(torch.eye(4, 3))
    # A vector sitting exactly on entry 2 must select entry 2.
    z = quantizer.codebook[2].view(1, 3, 1)
    assert quantizer(z)[1].item() == 2


def test_straight_through_passes_gradient_to_the_encoder():
    quantizer = VectorQuantizer(codebook_size=8, codebook_dim=4).eval()
    z = torch.randn(1, 4, 6, requires_grad=True)
    quantizer(z)[0].sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_ema_moves_the_codebook_only_in_training():
    quantizer = VectorQuantizer(codebook_size=8, codebook_dim=4)
    z = torch.randn(4, 4, 16)

    quantizer.eval()
    before = quantizer.codebook.clone()
    quantizer(z)
    assert torch.equal(quantizer.codebook, before)

    quantizer.train()
    quantizer(z)
    assert not torch.equal(quantizer.codebook, before)


def test_residual_stages_reduce_the_residual():
    """Each stage encodes what the previous ones missed."""
    torch.manual_seed(42)
    rvq = ResidualVQ(n_quantizers=4, codebook_size=64, codebook_dim=8).eval()
    z = torch.randn(2, 8, 12)


    errors = []
    for n in (1, 2, 4):
        quantized, _, _ = rvq(z, n_quantizers=n)
        errors.append((z - quantized).pow(2).mean().item())

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_codes_roundtrip_through_decode():
    rvq = ResidualVQ(n_quantizers=3, codebook_size=16, codebook_dim=8).eval()
    z = torch.randn(2, 8, 7)
    quantized, codes, _ = rvq(z)

    assert codes.shape == (2, 3, 7)
    assert torch.allclose(rvq.decode(codes), quantized, atol=1e-5)


def test_partial_decode_stays_valid():
    """Decoding the first k codes must work, which is what variable bitrate needs."""
    rvq = ResidualVQ(n_quantizers=4, codebook_size=16, codebook_dim=8).eval()
    codes = rvq.encode(torch.randn(1, 8, 5))
    assert rvq.decode(codes[:, :2]).shape == (1, 8, 5)


def test_the_quantizer_is_meta_constructible():
    with torch.device("meta"):
        rvq = ResidualVQ(n_quantizers=4, codebook_size=256, codebook_dim=64)
    assert sum(p.numel() for p in rvq.parameters()) == 0  # codebooks are buffers


# --------------------------------------------------------------------------
# Codec
# --------------------------------------------------------------------------


def test_hop_factors_into_strides():
    """The strides must multiply to the hop, in moderate steps rather than one."""
    import math

    for hop in (32, 64, 320, 512):
        strides = _strides_for(hop)
        assert math.prod(strides) == hop, (hop, strides)
        assert all(2 <= s <= 8 for s in strides), (hop, strides)

    with pytest.raises(ValueError, match="codec_hop"):
        _strides_for(7)


def test_the_codec_encoder_is_causal():
    conv = CausalConv1d(1, 2, kernel_size=4).eval()
    x = torch.randn(1, 1, 32)
    with torch.no_grad():
        baseline = conv(x)
        perturbed = x.clone()
        perturbed[..., 20:] += 10.0
        after = conv(perturbed)
    assert torch.allclose(baseline[..., :16], after[..., :16], atol=1e-6)


def test_the_codec_reaches_the_configured_frame_rate():
    codec = AudioCodec(base_channels=8, latent_dim=16, hop=32, n_quantizers=2,
                       codebook_size=16).eval()
    with torch.no_grad():
        codes = codec.encode(torch.randn(1, 320))
    assert codes.shape[1] == 2
    assert codes.shape[2] == 10  # 320 samples / hop 32


def test_the_codec_roundtrips_to_a_waveform():
    codec = AudioCodec(base_channels=8, latent_dim=16, hop=32, n_quantizers=2,
                       codebook_size=16).eval()
    waveform = torch.randn(1, 320)
    with torch.no_grad():
        recon, commit, codes = codec(waveform)
    assert recon.dim() == 2 and recon.shape[0] == 1
    assert torch.isfinite(recon).all()
    assert commit.ndim == 0 and torch.isfinite(commit)
    assert codes.dtype == torch.long


# --------------------------------------------------------------------------
# Vocoder
# --------------------------------------------------------------------------


def test_istft_head_produces_a_waveform():
    head = ISTFTHead(channels=16, n_fft=64, hop_length=16).eval()
    with torch.no_grad():
        waveform = head(torch.randn(1, 16, 10))
    assert waveform.dim() == 2
    assert torch.isfinite(waveform).all()


def test_istft_head_honours_a_requested_length():
    head = ISTFTHead(channels=16, n_fft=64, hop_length=16).eval()
    with torch.no_grad():
        assert head(torch.randn(1, 16, 10), length=128).shape[-1] == 128


def test_the_vocoder_runs_end_to_end():
    vocoder = NeuralVocoder(in_channels=16, d_model=32, n_layers=2,
                            n_fft=64, hop_length=16).eval()
    with torch.no_grad():
        waveform = vocoder(torch.randn(1, 16, 12))
    assert waveform.dim() == 2 and torch.isfinite(waveform).all()


def test_the_vocoder_is_deterministic_in_eval():
    vocoder = NeuralVocoder(in_channels=16, d_model=32, n_layers=2,
                            n_fft=64, hop_length=16).eval()
    features = torch.randn(1, 16, 12)
    with torch.no_grad():
        assert torch.allclose(vocoder(features), vocoder(features))


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------


def test_token_embedding_separates_quantizer_stages():
    """Code 5 from stage 0 and code 5 from stage 1 must be different entries."""
    embed = AudioTokenEmbedding(n_quantizers=2, codebook_size=8, d_model=16).eval()
    first = embed(torch.tensor([[[5], [0]]]))
    second = embed(torch.tensor([[[0], [5]]]))
    assert not torch.allclose(first, second)


def test_token_embedding_shape():
    embed = AudioTokenEmbedding(n_quantizers=3, codebook_size=8, d_model=16).eval()
    assert embed(torch.zeros(2, 3, 7, dtype=torch.long)).shape == (2, 7, 16)


def test_token_head_scores_every_quantizer():
    head = AudioTokenHead(d_model=16, n_quantizers=3, codebook_size=8).eval()
    logits = head(torch.randn(2, 5, 16))
    assert logits.shape == (2, 3, 5, 8)

    loss = head.loss(torch.randn(2, 5, 16), torch.zeros(2, 3, 5, dtype=torch.long))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_speaker_embedding_is_fixed_length_and_normalized():
    encoder = SpeakerEncoder(in_channels=16, d_model=32, n_layers=2).eval()
    with torch.no_grad():
        short = encoder(torch.randn(1, 16, 20))
        long = encoder(torch.randn(1, 16, 80))
    assert short.shape == long.shape == (1, 32)
    assert torch.allclose(short.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_speaker_embedding_distinguishes_references():
    encoder = SpeakerEncoder(in_channels=16, d_model=32, n_layers=2).eval()
    with torch.no_grad():
        a = encoder(torch.randn(1, 16, 40))
        b = encoder(torch.randn(1, 16, 40) * 5 + 3)
    assert not torch.allclose(a, b)


def test_ctc_head_produces_a_finite_loss():
    head = CTCHead(d_model=16, vocab_size=10)
    hidden = torch.randn(2, 12, 16)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    loss = head.loss(
        hidden, targets,
        input_lengths=torch.tensor([12, 12]),
        target_lengths=torch.tensor([3, 3]),
    )
    assert loss.ndim == 0 and torch.isfinite(loss)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_the_generator_trains_on_waveforms():
    config = audio_config()
    generator = RVQAudioGenerator(config)
    loss = generator(torch.randn(1, 640), context=torch.randn(1, 4, config.d_model))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_the_generator_rejects_a_mel_spectrogram():
    """The token path trains on waveforms; a silent reinterpretation would be worse."""
    generator = RVQAudioGenerator(audio_config())
    with pytest.raises(ValueError, match="waveforms"):
        generator(torch.randn(1, 16, 16))


def test_the_generator_samples_a_waveform():
    config = audio_config()
    generator = RVQAudioGenerator(config).eval()
    waveform = generator.sample(
        context=torch.randn(1, 4, config.d_model), n_frames=4
    )
    assert waveform.dim() == 2 and waveform.numel() > 0
    assert torch.isfinite(waveform).all()


def test_the_codec_trains_separately():
    generator = RVQAudioGenerator(audio_config())
    loss = generator.codec_loss(torch.randn(1, 640))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_speaker_conditioning_is_optional_and_wired():
    plain = RVQAudioGenerator(audio_config())
    assert plain.speaker_encoder is None

    conditioned = RVQAudioGenerator(audio_config(use_speaker_conditioning=True))
    assert conditioned.speaker_encoder is not None
    loss = conditioned(torch.randn(1, 640), context=torch.randn(1, 4, 64))
    assert torch.isfinite(loss)


def test_the_vocoder_is_used_when_configured():
    assert RVQAudioGenerator(audio_config()).vocoder is None
    assert RVQAudioGenerator(audio_config(vocoder_arch="istft")).vocoder is not None


def test_the_factory_honours_the_arch_switch():
    from model.modules.audio_generator import AudioGenerator

    assert isinstance(build_audio_generator(audio_config()), RVQAudioGenerator)
    assert isinstance(
        build_audio_generator(audio_config(audio_gen_arch="mel_diffusion")), AudioGenerator
    )


def test_the_model_swaps_audio_decoders_without_other_change():
    config = audio_config()
    model = FramerModel(config)
    assert isinstance(model.audio_gen, RVQAudioGenerator)

    out = model(
        input_ids=torch.randint(0, config.vocab_size, (1, 8)),
        target_audio=torch.randn(1, 640),
    )
    assert torch.isfinite(out["audio_loss"])


def test_validate_rejects_a_bad_audio_config():
    with pytest.raises(ValueError, match="audio_gen_arch"):
        audio_config(audio_gen_arch="codec").validate()
    with pytest.raises(ValueError, match="vocoder_arch"):
        audio_config(vocoder_arch="hifigan").validate()
    with pytest.raises(ValueError, match="audio_lm_d_model"):
        audio_config(audio_lm_d_model=30, audio_lm_n_heads=4).validate()
