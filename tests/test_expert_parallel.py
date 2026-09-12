"""Expert-parallel placement for Mixture-of-Experts layers.

`framer-2t-a49b` has 384 experts per layer across 80 MoE layers. Every rank
holding every expert is what makes the model 3.6 TiB per host. These tests cover
the placement arithmetic and, more importantly, that the single-rank path is
completely unchanged: expert parallelism must be invisible until it is switched
on.
"""

import copy
import os
import queue
import sys
import uuid

# The 2-rank tests below spawn fresh interpreters, which re-import this module to
# unpickle their worker functions. Those children run without pytest, so
# `conftest` resolves against sys.path alone and finds the root conftest.py
# instead of the one beside this file. Putting the tests directory first keeps
# the parent and the spawned children on tests/conftest.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from conftest import tiny_config, tiny_moe_config
from model.configs import FramerConfig
from model.framer import FramerModel
from model.modules.moe import MoEFeedForward
from model.training.expert_parallel import (
    ExpertParallelPlan,
    all_to_all_combine,
    all_to_all_dispatch,
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


# --------------------------------------------------------------------------
# Issue #230: Distributed expert dispatch & combine
# --------------------------------------------------------------------------


def _init_distributed(rank: int, world_size: int, init_file: str):
    """Safely initialize Gloo process group for 2-rank CPU testing."""
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _cleanup_distributed():
    """Ensure process group is cleanly destroyed."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker_wrapper(rank, world_size, init_file, error_queue, worker_fn):
    """Wrapper that catches any unhandled child exception and puts it on error_queue."""
    try:
        worker_fn(rank, world_size, init_file)
    except BaseException:
        import traceback

        if error_queue is not None:
            try:
                error_queue.put((rank, traceback.format_exc()))
            except Exception:
                pass
        raise


def _run_distributed_test(worker_fn, tmp_path, test_name: str):
    """Execute a 2-rank multi-process test safely with process isolation and error reporting."""
    init_file = os.path.abspath(str(tmp_path / f"dist_init_{test_name}_{uuid.uuid4().hex}"))
    if os.path.exists(init_file):
        os.remove(init_file)

    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()

    try:
        mp.spawn(
            _distributed_worker_wrapper,
            args=(2, init_file, error_queue, worker_fn),
            nprocs=2,
            join=True,
        )
    except Exception as exc:
        # A child that fails puts its traceback on the queue and then dies, so the
        # feeder thread may not have flushed it yet when we get here. Draining with
        # `empty()` races that flush and reports a bare exit code instead of the
        # reason, so wait briefly for one entry per rank.
        errors = []
        for _ in range(2):
            try:
                errors.append(error_queue.get(timeout=5))
            except queue.Empty:
                break
        if errors:
            formatted = "\n\n".join(
                f"--- Error from Rank {r} ---\n{tb}" for r, tb in sorted(errors)
            )
            raise RuntimeError(
                f"Distributed test '{test_name}' failed in worker process:\n{formatted}"
            ) from exc
        raise
    finally:
        if os.path.exists(init_file):
            try:
                os.remove(init_file)
            except OSError:
                pass


def _ep_dispatch_combine_worker(rank, world_size, init_file):
    _init_distributed(rank, world_size, init_file)
    try:
        plan = ExpertParallelPlan(ep_world=world_size, ep_rank=rank)
        if rank == 0:
            counts = torch.tensor([1, 2], dtype=torch.long)
            tokens = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        else:
            counts = torch.tensor([2, 1], dtype=torch.long)
            tokens = torch.tensor([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])

        received, recv_counts = all_to_all_dispatch(tokens, counts, plan)
        if rank == 0:
            assert recv_counts.tolist() == [1, 2]
            assert received.shape == (3, 2)
        else:
            assert recv_counts.tolist() == [2, 1]
            assert received.shape == (3, 2)

        combined = all_to_all_combine(received * 2.0, counts, recv_counts, plan)
        assert combined.shape == tokens.shape
        assert torch.equal(combined, tokens * 2.0)
    finally:
        _cleanup_distributed()


def _ep_equivalence_worker(rank, world_size, init_file):
    _init_distributed(rank, world_size, init_file)
    try:
        torch.manual_seed(42)
        moe_full = MoEFeedForward(
            d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2, n_shared_experts=1, dropout=0.0
        ).eval()

        plan = ExpertParallelPlan(ep_world=world_size, ep_rank=rank)
        moe_sharded = shard_experts(copy.deepcopy(moe_full), plan).eval()

        torch.manual_seed(100 + rank)
        x = torch.randn(2, 4, 32)

        with torch.no_grad():
            ref_out, ref_aux = moe_full(x)
            ep_out, ep_aux = moe_sharded(x)

        torch.testing.assert_close(ep_out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ep_aux, ref_aux, rtol=1e-5, atol=1e-5)
    finally:
        _cleanup_distributed()


def _ep_non_local_routing_worker(rank, world_size, init_file):
    _init_distributed(rank, world_size, init_file)
    try:
        torch.manual_seed(42)
        moe_full = MoEFeedForward(
            d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2, n_shared_experts=0, dropout=0.0
        ).eval()

        # Explicit cross-rank routing:
        # Rank 0 (owns experts 0, 1) routes strictly to experts 2, 3 (owned by Rank 1).
        # Rank 1 (owns experts 2, 3) routes strictly to experts 0, 1 (owned by Rank 0).
        def cross_router(x_flat):
            N = x_flat.shape[0]
            logits = torch.zeros(N, 4)
            if rank == 0:
                logits[:, 2] = 10.0
                logits[:, 3] = 5.0
            else:
                logits[:, 0] = 10.0
                logits[:, 1] = 5.0
            return logits

        object.__setattr__(moe_full, "router", cross_router)

        plan = ExpertParallelPlan(ep_world=world_size, ep_rank=rank)
        moe_sharded = shard_experts(copy.deepcopy(moe_full), plan).eval()
        object.__setattr__(moe_sharded, "router", cross_router)

        x = torch.randn(2, 6, 32)
        with torch.no_grad():
            ref_out, ref_aux = moe_full(x)
            ep_out, ep_aux = moe_sharded(x)

        torch.testing.assert_close(ep_out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ep_aux, ref_aux, rtol=1e-5, atol=1e-5)
    finally:
        _cleanup_distributed()


def _ep_topk_accumulation_worker(rank, world_size, init_file):
    _init_distributed(rank, world_size, init_file)
    try:
        torch.manual_seed(42)
        moe_full = MoEFeedForward(
            d_model=32, expert_d_ff=64, n_experts=8, n_experts_per_tok=3, n_shared_experts=1, dropout=0.0
        ).eval()

        plan = ExpertParallelPlan(ep_world=world_size, ep_rank=rank)
        moe_sharded = shard_experts(copy.deepcopy(moe_full), plan).eval()

        torch.manual_seed(200 + rank)
        x = torch.randn(3, 8, 32)

        with torch.no_grad():
            ref_out, ref_aux = moe_full(x)
            ep_out, ep_aux = moe_sharded(x)

        torch.testing.assert_close(ep_out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ep_aux, ref_aux, rtol=1e-5, atol=1e-5)
    finally:
        _cleanup_distributed()


def _ep_backward_worker(rank, world_size, init_file):
    _init_distributed(rank, world_size, init_file)
    try:
        torch.manual_seed(42)
        # 4 experts: rank 0 owns 0, 1; rank 1 owns 2, 3
        moe_full = MoEFeedForward(
            d_model=16, expert_d_ff=32, n_experts=4, n_experts_per_tok=2, n_shared_experts=0, dropout=0.0
        )

        # Controlled router routing tokens deterministically:
        # Token 0: expert 0 (local to rank 0) & expert 2 (non-local to rank 0, local to rank 1)
        # Token 1: expert 1 (local to rank 0) & expert 3 (non-local to rank 0, local to rank 1)
        # Token 2: expert 2 (local to rank 1) & expert 1 (non-local to rank 1, local to rank 0)
        # Token 3: expert 3 (local to rank 1) & expert 0 (non-local to rank 1, local to rank 0)
        def controlled_router(x_flat):
            N = x_flat.shape[0]
            logits = torch.zeros(N, 4, device=x_flat.device, dtype=x_flat.dtype)
            for i in range(N):
                if i % 4 == 0:
                    logits[i, 0] = 3.0
                    logits[i, 2] = 2.0
                elif i % 4 == 1:
                    logits[i, 1] = 3.0
                    logits[i, 3] = 2.0
                elif i % 4 == 2:
                    logits[i, 2] = 3.0
                    logits[i, 1] = 2.0
                else:
                    logits[i, 3] = 3.0
                    logits[i, 0] = 2.0
            return logits

        object.__setattr__(moe_full, "router", controlled_router)

        torch.manual_seed(100)
        x0 = torch.randn(1, 4, 16)
        x1 = torch.randn(1, 4, 16)
        x_input = x0 if rank == 0 else x1

        # Reference unsharded forward & backward
        x0_ref = x0.clone().detach().requires_grad_(True)
        x1_ref = x1.clone().detach().requires_grad_(True)
        ref_out0, _ = moe_full(x0_ref)
        ref_out1, _ = moe_full(x1_ref)
        ref_loss = ref_out0.sum() + ref_out1.sum()
        ref_loss.backward()

        ref_input_grad = x0_ref.grad.clone() if rank == 0 else x1_ref.grad.clone()
        ref_out = ref_out0 if rank == 0 else ref_out1
        ref_expert_grads = [
            (e.w1.weight.grad.clone(), e.w2.weight.grad.clone(), e.w3.weight.grad.clone())
            for e in moe_full.experts
        ]

        # Sharded EP model
        plan = ExpertParallelPlan(ep_world=world_size, ep_rank=rank)
        moe_sharded = shard_experts(copy.deepcopy(moe_full), plan)
        object.__setattr__(moe_sharded, "router", controlled_router)

        x_ep = x_input.clone().detach().requires_grad_(True)
        ep_out, _ = moe_sharded(x_ep)
        ep_loss = ep_out.sum()
        ep_loss.backward()

        # 1. Forward equivalence
        torch.testing.assert_close(ep_out, ref_out, rtol=1e-5, atol=1e-5)

        # 2. Input activation gradient equivalence (proves backward dispatch & combine)
        torch.testing.assert_close(x_ep.grad, ref_input_grad, rtol=1e-5, atol=1e-5)

        # 3. Local expert parameter gradient equivalence (proves accumulation from off-rank tokens)
        start, end = plan.local_experts(4)
        for local_e, global_e in enumerate(range(start, end)):
            sharded_exp = moe_sharded.experts[local_e]
            w1_grad, w2_grad, w3_grad = ref_expert_grads[global_e]
            torch.testing.assert_close(sharded_exp.w1.weight.grad, w1_grad, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(sharded_exp.w2.weight.grad, w2_grad, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(sharded_exp.w3.weight.grad, w3_grad, rtol=1e-5, atol=1e-5)
    finally:
        _cleanup_distributed()


def test_all_to_all_helpers_disabled_plan():
    """all_to_all_dispatch and combine are no-ops when plan is disabled."""
    plan = ExpertParallelPlan()
    tokens = torch.randn(4, 16)
    counts = torch.tensor([4])
    recv_tok, recv_counts = all_to_all_dispatch(tokens, counts, plan)
    assert torch.equal(recv_tok, tokens)
    assert torch.equal(recv_counts, counts)
    combined = all_to_all_combine(tokens, counts, recv_counts, plan)
    assert torch.equal(combined, tokens)


def test_invalid_expert_offset_without_plan_raises():
    """Setting expert_offset without an enabled ep_plan must fail loudly (Issue #230)."""
    layer = moe_layer(n_experts=4)
    layer.expert_offset = 2
    x = torch.randn(1, 4, 32)
    with pytest.raises(RuntimeError, match="Invalid expert-parallel configuration"):
        layer(x)

    # Disabled plan (ep_world=1) with non-zero offset also fails loudly
    layer.ep_plan = ExpertParallelPlan(ep_world=1)
    with pytest.raises(RuntimeError, match="Invalid expert-parallel configuration"):
        layer(x)


def test_single_rank_plan_matches_unsharded():
    """At ep_world=1, forward matches the unsharded baseline exactly."""
    torch.manual_seed(42)
    layer = moe_layer(n_experts=4).eval()
    x = torch.randn(2, 4, 32)
    with torch.no_grad():
        baseline_out, baseline_aux = layer(x)

    shard_experts(layer, ExpertParallelPlan(ep_world=1, ep_rank=0))
    with torch.no_grad():
        out, aux = layer(x)

    assert torch.equal(baseline_out, out)
    assert torch.equal(baseline_aux, aux)


def test_all_to_all_dispatch_combine_2rank(tmp_path):
    """Multi-process test of all_to_all_dispatch and all_to_all_combine."""
    _run_distributed_test(_ep_dispatch_combine_worker, tmp_path, "dispatch_combine")


def test_two_rank_expert_parallel_equivalence(tmp_path):
    """Issue #230: 2-rank expert-parallel forward matches unsharded reference."""
    _run_distributed_test(_ep_equivalence_worker, tmp_path, "equivalence")


def test_two_rank_explicit_non_local_routing(tmp_path):
    """Issue #230: Tokens routed strictly to off-rank experts are correctly computed."""
    _run_distributed_test(_ep_non_local_routing_worker, tmp_path, "non_local_routing")


def test_two_rank_topk_accumulation(tmp_path):
    """Issue #230: Top-k > 1 accumulation across ranks matches unsharded reference."""
    _run_distributed_test(_ep_topk_accumulation_worker, tmp_path, "topk_accumulation")


def test_two_rank_backward_gradient_equivalence(tmp_path):
    """Issue #230: 2-rank expert-parallel backward matches unsharded reference."""
    _run_distributed_test(_ep_backward_worker, tmp_path, "backward_gradient")
