"""Rectified flow objective and ODE sampler.

The ancestral DDPM sampler this replaces walks 1000 steps, each a full denoiser
forward. Rectified flow trains the model to predict a straight-line velocity
between noise and data, which makes the sampling trajectory nearly straight and
solvable in 20-50 Euler or Heun steps - a 20-50x reduction in sampling cost for
comparable output.

Classifier-free guidance lives here too, because it is a property of how the
field is evaluated rather than of the denoiser: the conditional and
unconditional predictions are extrapolated apart at each step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RectifiedFlow(nn.Module):
    """Linear interpolation between noise and data, with a velocity target.

    ``x_t = (1 - t) * noise + t * data`` and the target velocity is
    ``data - noise``, constant along each path. ``t`` runs 0 (pure noise) to
    1 (data).
    """

    def __init__(self, logit_normal_sampling: bool = True):
        super().__init__()
        # Timestep sampling concentrated near the middle of the trajectory,
        # where the prediction problem is hardest. Uniform sampling wastes
        # capacity on the near-noise and near-data ends.
        self.logit_normal_sampling = logit_normal_sampling

    def sample_t(self, batch_size: int, device=None, generator=None) -> torch.Tensor:
        if not self.logit_normal_sampling:
            return torch.rand(batch_size, device=device, generator=generator)
        normal = torch.randn(batch_size, device=device, generator=generator)
        return torch.sigmoid(normal)

    @staticmethod
    def interpolate(noise: torch.Tensor, data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *([1] * (data.dim() - 1)))
        return (1 - t) * noise + t * data

    @staticmethod
    def target(noise: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return data - noise

    @staticmethod
    def loss(velocity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(velocity, target)


class ODESampler(nn.Module):
    """Solve the flow ODE from noise to data.

    ``euler`` costs one denoiser call per step; ``heun`` costs two but is second
    order, so it usually reaches the same quality in fewer total calls.
    """

    def __init__(self, steps: int = 50, method: str = "euler",
                 guidance_distilled: bool = False):
        super().__init__()
        if method not in ("euler", "heun"):
            raise ValueError(f"Unknown ODE method '{method}'. Expected 'euler' or 'heun'")
        self.steps = steps
        self.method = method
        # A student trained with guidance folded in already produces the guided
        # field, so evaluating it twice and extrapolating would apply guidance
        # twice. Skipping the pair halves the cost of every step.
        self.guidance_distilled = guidance_distilled

    def _guided(self, velocity_fn, x, t, context, null_context, cfg_scale):
        """One velocity evaluation, with classifier-free guidance when asked.

        The conditional and unconditional fields are evaluated in a single
        batch-doubled forward, then extrapolated apart. ``cfg_scale`` of 1.0 is
        the unguided field, so the doubling is skipped entirely.
        """
        if self.guidance_distilled:
            return velocity_fn(x, t, context)
        if null_context is None or cfg_scale is None or cfg_scale == 1.0 or context is None:
            return velocity_fn(x, t, context)

        batch = x.shape[0]
        x_pair = torch.cat([x, x], dim=0)
        t_pair = torch.cat([t, t], dim=0)
        context_pair = torch.cat([context, null_context.expand_as(context)], dim=0)

        velocity = velocity_fn(x_pair, t_pair, context_pair)
        conditional, unconditional = velocity[:batch], velocity[batch:]
        return unconditional + cfg_scale * (conditional - unconditional)

    @torch.no_grad()
    def sample(
        self,
        velocity_fn,
        shape: tuple,
        context: torch.Tensor = None,
        null_context: torch.Tensor = None,
        cfg_scale: float = None,
        device="cpu",
        steps: int = None,
        generator: torch.Generator = None,
        clamp_fn=None,
    ) -> torch.Tensor:
        """Solve from noise to data.

        ``clamp_fn(x, t)`` runs after each step when given. It is how a window
        of video is held to the frames that came before it: without a hook the
        solver has no way to say "these positions are already decided", and a
        long clip has to be generated as independent pieces that do not join.
        """
        steps = steps or self.steps
        x = torch.randn(shape, device=device, generator=generator)
        timeline = torch.linspace(0.0, 1.0, steps + 1, device=device)

        for i in range(steps):
            t_now, t_next = timeline[i], timeline[i + 1]
            dt = t_next - t_now
            t_batch = t_now.expand(shape[0])

            velocity = self._guided(velocity_fn, x, t_batch, context, null_context, cfg_scale)

            if self.method == "euler":
                x = x + dt * velocity
            else:
                x_predicted = x + dt * velocity
                velocity_next = self._guided(
                    velocity_fn, x_predicted, t_next.expand(shape[0]),
                    context, null_context, cfg_scale,
                )
                x = x + dt * 0.5 * (velocity + velocity_next)

            if clamp_fn is not None:
                x = clamp_fn(x, t_next)

        return x


class FlowDistiller(nn.Module):
    """Train a few-step student against a many-step guided teacher.

    The rectified-flow sampler is 20-50 steps, and classifier-free guidance
    doubles the batch on every one of them, so a sampled image costs 40 to 100
    denoiser forwards. Lowering the step count alone does not help: below
    roughly twenty the undistilled solve degrades visibly, which trades the
    measured metric for speed rather than moving the speed-quality frontier.

    Distillation moves the frontier instead. The teacher walks a segment of the
    trajectory in several guided steps; the student is asked to cover the same
    segment in one. Because the teacher's output is already guided, the student
    learns the guided field directly and needs no guidance at sampling time, so
    a step becomes one forward rather than two.

    The scale is baked in. A student distilled at a fixed ``cfg_scale`` produces
    that scale and no other, which is the price of folding guidance into the
    weights; a deployment that needs the scale to stay adjustable keeps the
    teacher.
    """

    def __init__(self, teacher_substeps: int = 8, method: str = "euler"):
        super().__init__()
        if teacher_substeps < 1:
            raise ValueError(f"teacher_substeps must be positive, got {teacher_substeps}")
        self.teacher_substeps = teacher_substeps
        self.sampler = ODESampler(steps=teacher_substeps, method=method)

    def segments(self, student_steps: int, device=None) -> torch.Tensor:
        """The trajectory boundaries the student must learn to jump between."""
        if student_steps < 1:
            raise ValueError(f"student_steps must be positive, got {student_steps}")
        return torch.linspace(0.0, 1.0, student_steps + 1, device=device)

    @torch.no_grad()
    def teacher_endpoint(self, velocity_fn, x, t_start, t_end, context=None,
                         null_context=None, cfg_scale=None) -> torch.Tensor:
        """Where the guided teacher lands after walking the segment properly."""
        substeps = self.teacher_substeps
        timeline = torch.linspace(float(t_start), float(t_end), substeps + 1, device=x.device)

        for i in range(substeps):
            t_now, t_next = timeline[i], timeline[i + 1]
            dt = t_next - t_now
            velocity = self.sampler._guided(
                velocity_fn, x, t_now.expand(x.shape[0]), context, null_context, cfg_scale
            )
            x = x + dt * velocity
        return x

    def student_endpoint(self, student_velocity, x, t_start, t_end) -> torch.Tensor:
        """Where the student lands covering the same segment in one step."""
        return x + (float(t_end) - float(t_start)) * student_velocity

    def loss(self, student_velocity, x, t_start, t_end, teacher_endpoint) -> torch.Tensor:
        """How far the student's single step misses the teacher's walk."""
        return F.mse_loss(self.student_endpoint(student_velocity, x, t_start, t_end),
                          teacher_endpoint)


def sampling_cost(steps: int, method: str = "euler", guidance: bool = True,
                  guidance_distilled: bool = False) -> int:
    """Denoiser forwards one sample costs, which is what the time is spent on.

    Counting this rather than timing it keeps the number comparable across
    machines, and makes the two savings visible separately: fewer steps, and
    one forward per step instead of two.
    """
    per_step = 2 if method == "heun" else 1
    if guidance and not guidance_distilled:
        per_step *= 2
    return steps * per_step
