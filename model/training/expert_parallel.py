"""Expert-parallel placement for Mixture-of-Experts layers.

`framer-2t-a49b` has 384 experts per MoE layer across 80 MoE layers. Every rank
holding every expert is what makes the model 3.6 TiB per host; expert
parallelism instead gives each rank ``n_experts // ep_world`` of them, so the
expert weights divide by the mesh size while the router and attention stay
replicated.

The routing itself is unchanged. What changes is where the expert lives: a token
routed to an expert on another rank is sent there, computed, and sent back,
which is the all-to-all dispatch and combine below.

At ``ep_world == 1`` every path here is a no-op and ``MoEFeedForward`` behaves
exactly as it did before, so single-device runs and the test suite are
untouched.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ExpertParallelPlan:
    """Which experts a rank owns, and over what process group."""

    ep_world: int = 1
    ep_rank: int = 0
    group: object = None

    @property
    def enabled(self) -> bool:
        return self.ep_world > 1

    def local_experts(self, n_experts: int) -> tuple:
        """The half-open range of global expert ids this rank owns."""
        if n_experts % self.ep_world:
            raise ValueError(
                f"n_experts ({n_experts}) must divide the expert-parallel world "
                f"({self.ep_world}); 384 was chosen to divide 8/16/32/64/128"
            )
        per_rank = n_experts // self.ep_world
        start = self.ep_rank * per_rank
        return start, start + per_rank

    def owner_of(self, expert_id: int, n_experts: int) -> int:
        """Which rank holds a given global expert."""
        return expert_id // (n_experts // self.ep_world)


def plan_from_environment(ep_size: int = None, group=None) -> ExpertParallelPlan:
    """Build a plan from the current process group, or a disabled one."""
    if not (dist.is_available() and dist.is_initialized()):
        return ExpertParallelPlan()

    world = dist.get_world_size(group) if group is not None else dist.get_world_size()
    rank = dist.get_rank(group) if group is not None else dist.get_rank()
    return ExpertParallelPlan(ep_world=ep_size or world, ep_rank=rank, group=group)


def build_device_mesh(ep_size: int):
    """A 2D (dp, ep) mesh so FSDP and expert parallelism do not collide.

    FSDP shards parameters over the ``dp`` dimension. Expert weights are already
    split over ``ep``, so sharding them again would divide them twice and leave
    each rank with a fragment of a fragment. Giving each its own dimension keeps
    the two composable.
    """
    from torch.distributed.device_mesh import init_device_mesh

    world = dist.get_world_size()
    if world % ep_size:
        raise ValueError(f"world size ({world}) must be divisible by ep_size ({ep_size})")
    return init_device_mesh("cuda", (world // ep_size, ep_size), mesh_dim_names=("dp", "ep"))


def shard_experts(moe, plan: ExpertParallelPlan):
    """Drop the experts this rank does not own, in place.

    Called on a meta-device module before materialization, so the weights this
    rank will never hold are never allocated in the first place.
    """
    if not plan.enabled:
        return moe

    start, end = plan.local_experts(moe.n_experts)
    moe.experts = torch.nn.ModuleList(list(moe.experts)[start:end])
    moe.ep_plan = plan
    moe.expert_offset = start
    return moe


def shard_model_experts(model, plan: ExpertParallelPlan):
    """Apply :func:`shard_experts` to every MoE layer in a model.

    Matched by type rather than by an ``is_moe`` attribute: ``TransformerBlock``
    also carries that flag, to report whether its FFN is sparse, and sharding a
    block is not a meaningful operation.
    """
    if not plan.enabled:
        return model

    from ..modules.moe import MoEFeedForward

    for module in model.modules():
        if isinstance(module, MoEFeedForward):
            shard_experts(module, plan)
    return model


def all_to_all_dispatch(tokens: torch.Tensor, counts: torch.Tensor, plan: ExpertParallelPlan):
    """Send each rank the tokens routed to the experts it owns.

    Returns the received tokens and the per-rank counts needed to send the
    results back, which the combine step reverses.
    """
    if not plan.enabled:
        return tokens, counts

    recv_counts = torch.empty_like(counts)
    dist.all_to_all_single(recv_counts, counts, group=plan.group)

    send_splits = counts.tolist()
    recv_splits = recv_counts.tolist()
    received = tokens.new_empty((sum(recv_splits), tokens.shape[-1]))
    dist.all_to_all_single(
        received, tokens, output_split_sizes=recv_splits,
        input_split_sizes=send_splits, group=plan.group,
    )
    return received, recv_counts


def all_to_all_combine(
    outputs: torch.Tensor, send_counts: torch.Tensor, recv_counts: torch.Tensor,
    plan: ExpertParallelPlan,
):
    """Return each expert's outputs to the rank whose tokens they were."""
    if not plan.enabled:
        return outputs

    combined = outputs.new_empty((int(send_counts.sum()), outputs.shape[-1]))
    dist.all_to_all_single(
        combined, outputs,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
        group=plan.group,
    )
    return combined
