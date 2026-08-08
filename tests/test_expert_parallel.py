"""Expert-parallel placement for Mixture-of-Experts layers.

`framer-2t-a49b` has 384 experts per layer across 80 MoE layers. Every rank
holding every expert is what makes the model 3.6 TiB per host. These tests cover
the placement arithmetic and, more importantly, that the single-rank path is
completely unchanged: expert parallelism must be invisible until it is switched
on.
"""

import pytest
import torch

from conftest import tiny_config, tiny_moe_config
from model.configs import FramerConfig
from model.framer import FramerModel
from model.modules.moe import MoEFeedForward
from model.training.expert_parallel import (
    ExpertParallelPlan,
    plan_from_environment,
    shard_experts,
    shard_model_experts,
)


def moe_layer(n_experts=8, d_model=32, expert_d_ff=64, top_k=2):
    return MoEFeedForward(
        d_model=d_model, expert_d_ff=expert_d_ff, n_experts=n_experts,
        n_experts_per_tok=top_k, n_shared_experts=1, dropout=0.0,
    )


# --------------------------------------------------------------------------
# Placement arithmetic
# --------------------------------------------------------------------------


def test_a_disabled_plan_owns_everything():
    plan = ExpertParallelPlan()
    assert not plan.enabled
    assert plan.local_experts(384) == (0, 384)


@pytest.mark.parametrize("ep_world", [2, 4, 8, 16, 32, 64, 128])
def test_the_flagship_expert_count_divides_every_common_mesh(ep_world):
    """384 was chosen for exactly this; the preset comment claims it, so test it."""
    n_experts = FramerConfig.from_preset("framer-2t-a49b").n_experts
    ranges = [
        ExpertParallelPlan(ep_world=ep_world, ep_rank=r).local_experts(n_experts)
        for r in range(ep_world)
    ]
    # Contiguous, non-overlapping, and covering every expert exactly once.
    assert ranges[0][0] == 0
    assert ranges[-1][1] == n_experts
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    assert sum(end - start for start, end in ranges) == n_experts


def test_an_indivisible_expert_count_is_rejected():
    with pytest.raises(ValueError, match="must divide"):
        ExpertParallelPlan(ep_world=5).local_experts(384)


def test_owner_lookup_matches_the_local_ranges():
    plan = ExpertParallelPlan(ep_world=4, ep_rank=0)
    for expert_id in range(384):
        owner = plan.owner_of(expert_id, 384)
        start, end = ExpertParallelPlan(ep_world=4, ep_rank=owner).local_experts(384)
        assert start <= expert_id < end


def test_a_plan_built_without_distributed_is_disabled():
    assert not plan_from_environment().enabled


# --------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------


def test_sharding_keeps_only_the_local_experts():
    layer = shard_experts(moe_layer(n_experts=8), ExpertParallelPlan(ep_world=4, ep_rank=2))
    assert len(layer.experts) == 2
    assert layer.expert_offset == 4
    # n_experts stays global: the router still scores all of them.
    assert layer.n_experts == 8
    assert layer.router.out_features == 8


def test_sharding_is_a_no_op_when_disabled():
    layer = moe_layer(n_experts=8)
    before = len(layer.experts)
    shard_experts(layer, ExpertParallelPlan())
    assert len(layer.experts) == before
    assert layer.expert_offset == 0


def test_sharding_a_model_reaches_every_moe_layer():
    model = FramerModel(tiny_moe_config(n_experts=4, first_dense_layers=0))
    shard_model_experts(model, ExpertParallelPlan(ep_world=2, ep_rank=1))
    # Matched by type: TransformerBlock also carries an is_moe flag.
    moe_layers = [m for m in model.modules() if isinstance(m, MoEFeedForward)]
    assert moe_layers
    for layer in moe_layers:
        assert len(layer.experts) == 2
        assert layer.expert_offset == 2


def test_shards_partition_the_experts_exactly():
    """Every expert is held by exactly one rank, and the union is complete."""
    held = []
    for rank in range(4):
        layer = shard_experts(moe_layer(n_experts=8), ExpertParallelPlan(ep_world=4, ep_rank=rank))
        held.extend(range(layer.expert_offset, layer.expert_offset + len(layer.experts)))
    assert sorted(held) == list(range(8))


def test_sharding_on_meta_allocates_nothing():
    """Sharding happens before materialization, so unheld weights never exist."""
    with torch.device("meta"):
        layer = moe_layer(n_experts=8, d_model=64, expert_d_ff=128)
    shard_experts(layer, ExpertParallelPlan(ep_world=4, ep_rank=0))
    assert len(layer.experts) == 2
    assert all(p.is_meta for p in layer.parameters())


# --------------------------------------------------------------------------
# The single-rank path must be untouched
# --------------------------------------------------------------------------


def test_the_offset_loop_matches_the_original_dispatch():
    """Indexing by global expert id must be identical at offset zero."""
    torch.manual_seed(0)
    layer = moe_layer(n_experts=4).eval()
    x = torch.randn(2, 5, 32)

    with torch.no_grad():
        baseline, aux = layer(x)

    # Re-run with the offset explicitly zero, which is the shipped default.
    layer.expert_offset = 0
    with torch.no_grad():
        again, aux_again = layer(x)

    assert torch.equal(baseline, again)
    assert torch.equal(aux, aux_again)


def test_a_sharded_layer_computes_only_its_own_experts():
    """The partial outputs must sum to the unsharded one, which is what the
    all-to-all combine reassembles at runtime."""
    torch.manual_seed(0)
    full = moe_layer(n_experts=4).eval()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        expected, _ = full(x)

    # Two shards over the same weights, with shared experts counted once.
    partials = []
    for rank in range(2):
        import copy

        shard = shard_experts(copy.deepcopy(full), ExpertParallelPlan(ep_world=2, ep_rank=rank))
        shard.eval()
        if rank > 0:
            shard.shared_experts = torch.nn.ModuleList()
        with torch.no_grad():
            partials.append(shard(x)[0])

    assert torch.allclose(sum(partials), expected, atol=1e-5)


def test_the_model_is_unchanged_without_a_plan():
    config = tiny_moe_config()
    torch.manual_seed(0)
    model = FramerModel(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        before = model(input_ids=ids)["logits"]

    shard_model_experts(model, ExpertParallelPlan())
    with torch.no_grad():
        after = model(input_ids=ids)["logits"]
    assert torch.equal(before, after)


def test_dense_layers_are_left_alone():
    model = FramerModel(tiny_config())
    shard_model_experts(model, ExpertParallelPlan(ep_world=2, ep_rank=0))
    assert not any(isinstance(m, MoEFeedForward) for m in model.modules())


def test_a_transformer_block_is_not_mistaken_for_an_expert_layer():
    """TransformerBlock reports is_moe to describe its FFN, not to be sharded."""
    from model.modules.transformer import TransformerBlock

    model = FramerModel(tiny_moe_config(n_experts=4, first_dense_layers=0))
    shard_model_experts(model, ExpertParallelPlan(ep_world=2, ep_rank=0))
    for block in model.modules():
        if isinstance(block, TransformerBlock):
            assert not hasattr(block, "expert_offset")
