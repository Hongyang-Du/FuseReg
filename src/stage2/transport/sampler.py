"""Euler sampling on the checkpoint's shifted noise-to-data time grid."""

import torch as th


class Sampler:
    def __init__(self, transport):
        self.transport = transport
        self.drift = transport.get_drift()

    def sample_ode(self, *, num_steps=50):
        t_grid = th.linspace(1.0, 0.0, num_steps + 1)
        shift = self.transport.time_dist_shift
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        def sample_fn(x, model, **model_kwargs):
            t_steps = t_grid.to(x.device)
            for i in range(num_steps):
                h = t_steps[i] - t_steps[i + 1]
                t_batch = th.full((x.shape[0],), t_steps[i].item(), device=x.device)
                x = x - h * self.drift(x, t_batch, model, **model_kwargs)
            return x.unsqueeze(0)

        return sample_fn
