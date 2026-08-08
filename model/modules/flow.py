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

    def __init__(self, steps: int = 50, method: str = "euler"):
        super().__init__()
        if method not in ("euler", "heun"):
            raise ValueError(f"Unknown ODE method '{method}'. Expected 'euler' or 'heun'")
        self.steps = steps
        self.method = method

    @staticmethod
    def _guided(velocity_fn, x, t, context, null_context, cfg_scale):
        """One velocity evaluation, with classifier-free guidance when asked.

        The conditional and unconditional fields are evaluated in a single
        batch-doubled forward, then extrapolated apart. ``cfg_scale`` of 1.0 is
        the unguided field, so the doubling is skipped entirely.
        """
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
    ) -> torch.Tensor:
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

        return x
