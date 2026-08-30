"""Few-step sampling: fewer steps, and one forward per step instead of two.

The rectified-flow sampler runs 20-50 steps and classifier-free guidance
doubles the batch on every one, so an image costs 40 to 100 denoiser forwards.
Lowering the step count alone does not help, because below roughly twenty the
undistilled solve degrades visibly: that trades the measured metric for speed
rather than moving the frontier.

These tests pin the two savings separately, that a distilled sampler does not
apply guidance twice, and that the student's objective is the teacher's
endpoint rather than something easier to hit.
"""

import pytest
import torch

from model.configs import FramerConfig
from model.modules.flow import FlowDistiller, ODESampler, sampling_cost


def constant_velocity(value=1.0):
    """A field with a known analytic solution, so a solver can be checked."""
    def velocity_fn(x, t, context=None):
        return torch.full_like(x, value)
    return velocity_fn


# ── What a sample costs ───────────────────────────────────────────────────

def test_the_two_savings_are_visible_separately():
    baseline = sampling_cost(50, "euler", guidance=True)
    fewer_steps = sampling_cost(4, "euler", guidance=True)
    both = sampling_cost(4, "euler", guidance=True, guidance_distilled=True)

    assert baseline == 100
    assert fewer_steps == 8, "fewer steps alone still pays for the guidance pair"
    assert both == 4
    assert baseline / both == 25


def test_heun_is_counted_as_the_two_calls_it_makes():
    assert sampling_cost(10, "heun", guidance=False) == 20
    assert sampling_cost(10, "euler", guidance=False) == 10


def test_an_unguided_sample_never_paid_the_pair():
    assert sampling_cost(20, "euler", guidance=False) == 20


# ── Guidance ──────────────────────────────────────────────────────────────

def test_a_distilled_sampler_does_not_guide_a_second_time():
    calls = []

    def velocity_fn(x, t, context=None):
        calls.append(x.shape[0])
        return torch.zeros_like(x)

    batch = torch.zeros(2, 4, 4, 4)
    context, null = torch.randn(2, 5, 16), torch.zeros(1, 1, 16)

    ODESampler(guidance_distilled=True)._guided(velocity_fn, batch, torch.zeros(2), context, null, 3.0)
    assert calls == [2], "a distilled student already produces the guided field"

    calls.clear()
    ODESampler()._guided(velocity_fn, batch, torch.zeros(2), context, null, 3.0)
    assert calls == [4], "the teacher pays for the conditional and unconditional pair"


def test_guidance_still_extrapolates_when_it_is_not_distilled():
    def velocity_fn(x, t, context):
        half = x.shape[0] // 2
        return torch.cat([torch.ones_like(x[:half]), torch.zeros_like(x[half:])], dim=0)

    guided = ODESampler()._guided(
        velocity_fn, torch.zeros(2, 4, 4, 4), torch.zeros(2),
        torch.randn(2, 5, 16), torch.zeros(1, 1, 16), cfg_scale=3.0,
    )
    assert torch.allclose(guided, torch.full((2, 4, 4, 4), 3.0))


# ── The distiller ─────────────────────────────────────────────────────────

def test_segments_span_the_whole_trajectory():
    distiller = FlowDistiller()
    boundaries = distiller.segments(4)
    assert len(boundaries) == 5
    assert float(boundaries[0]) == 0.0 and float(boundaries[-1]) == 1.0


def test_a_student_of_no_steps_is_refused():
    with pytest.raises(ValueError, match="student_steps must be positive"):
        FlowDistiller().segments(0)
    with pytest.raises(ValueError, match="teacher_substeps must be positive"):
        FlowDistiller(teacher_substeps=0)


def test_the_teacher_walks_the_segment_it_was_given():
    # With a constant field the endpoint is analytic: x + (t_end - t_start) * v.
    distiller = FlowDistiller(teacher_substeps=8)
    x = torch.zeros(2, 4, 4, 4)

    endpoint = distiller.teacher_endpoint(constant_velocity(2.0), x, 0.0, 0.5)
    assert torch.allclose(endpoint, torch.full_like(x, 1.0), atol=1e-5)


def test_a_perfect_student_has_no_loss_to_pay():
    distiller = FlowDistiller(teacher_substeps=4)
    x = torch.zeros(2, 4, 4, 4)

    target = distiller.teacher_endpoint(constant_velocity(2.0), x, 0.0, 0.5)
    perfect = torch.full_like(x, 2.0)
    assert float(distiller.loss(perfect, x, 0.0, 0.5, target)) < 1e-8


def test_a_student_that_misses_is_charged_for_it():
    distiller = FlowDistiller(teacher_substeps=4)
    x = torch.zeros(2, 4, 4, 4)

    target = distiller.teacher_endpoint(constant_velocity(2.0), x, 0.0, 0.5)
    wrong = torch.full_like(x, -2.0)
    assert float(distiller.loss(wrong, x, 0.0, 0.5, target)) > 1.0


def test_the_student_learns_the_guided_field_not_the_bare_one():
    # The teacher's endpoint is produced with guidance on, so what the student
    # is asked to reproduce already carries it. That is what lets sampling drop
    # the pair.
    def velocity_fn(x, t, context=None):
        if context is None:
            return torch.zeros_like(x)
        half = x.shape[0] // 2
        if x.shape[0] > 2:
            return torch.cat([torch.ones_like(x[:half]), torch.zeros_like(x[half:])], dim=0)
        return torch.ones_like(x)

    distiller = FlowDistiller(teacher_substeps=2)
    x = torch.zeros(2, 4, 4, 4)
    guided = distiller.teacher_endpoint(
        velocity_fn, x, 0.0, 1.0,
        context=torch.randn(2, 5, 16), null_context=torch.zeros(1, 1, 16), cfg_scale=3.0,
    )
    unguided = distiller.teacher_endpoint(velocity_fn, x, 0.0, 1.0)
    assert not torch.allclose(guided, unguided)


# ── Configuration ─────────────────────────────────────────────────────────

def test_a_distilled_config_selects_both_savings_together():
    config = FramerConfig(flow_distilled=True, flow_distilled_steps=4).validate()
    assert config.flow_distilled and config.flow_distilled_steps == 4


def test_a_step_count_of_nothing_is_rejected():
    with pytest.raises(ValueError, match="flow_distilled_steps"):
        FramerConfig(flow_distilled_steps=0).validate()


def test_the_default_is_still_the_teacher():
    config = FramerConfig()
    assert config.flow_distilled is False, "distillation is opt-in until a student exists"
