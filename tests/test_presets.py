"""Preset registry + parameter estimator tests (incl. the 1T target)."""


from conftest import tiny_config
from model.configs import FramerConfig, list_presets, resolve_preset_name
from model.framer import FramerModel
from model.utils.helpers import estimate_params


def test_all_presets_build_configs():
    for name in list_presets():
        cfg = FramerConfig.from_preset(name)
        assert cfg.d_model > 0 and cfg.n_layers > 0
        assert cfg.preset == name


def test_size_aliases():
    assert resolve_preset_name("tiny") == "framer-tiny"
    assert resolve_preset_name("large") == "framer-large"
    assert FramerConfig.from_preset("small").preset == "framer-small"


def test_trillion_preset_is_about_1t_without_instantiation():
    cfg = FramerConfig.from_preset("framer-1t-a32b")
    est = estimate_params(cfg)
    # ~1T total, far smaller active budget — the whole point of MoE.
    assert 8e11 < est["total"] < 1.2e12
    assert est["active"] < est["total"] / 10


def test_moe_active_less_than_total():
    dense = estimate_params(FramerConfig.from_preset("framer-8b"))
    assert dense["active"] == dense["total"]  # dense: active == total
    moe = estimate_params(FramerConfig.from_preset("framer-30b-a3b"))
    assert moe["active"] < moe["total"]


def test_estimator_matches_instantiated_text_only_model():
    cfg = tiny_config(vocab_size=256, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2, d_ff=128)
    model = FramerModel(cfg)
    actual = sum(p.numel() for p in model.parameters())
    est = estimate_params(cfg)["total"]
    # Tied embedding is counted once in both; allow a small relative gap.
    assert abs(actual - est) / actual < 0.02
