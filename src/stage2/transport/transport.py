"""Full-fusion x-prediction with the original oldnorm flow-matching loss."""

import torch as th

from stage2.utils import apply_cfg_dropout


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (x.ndim - 1)))


class Transport:
    def __init__(self, time_dist_shift=1.0, t_eps=0.05):
        self.time_dist_shift = time_dist_shift
        self.t_eps = t_eps

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t = th.randn(x1.shape[0]).sigmoid().to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def training_losses(
        self, model, x1, model_kwargs, model_kwargs_null, *, x1_target,
        base_model_coeff=1.0, cfg_dropout_prob=0.1,
    ):
        """Noise the subset latent and supervise both heads with the full fusion.

        Preserve the original velocity-space squared loss, including its implicit
        1 / max(t, t_eps)^2 weighting of the x-prediction error.
        """
        model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)
        t, x0, x1 = self.sample(x1)
        xt = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        vt = (xt - x1_target) / _expand_t(t, xt).clamp_min(self.t_eps)
        full, base = model(xt, t, **model_kwargs)
        loss = self.compute_loss(full, vt, xt, t)
        loss_base = self.compute_loss(base, vt, xt, t)
        return {"loss": loss + base_model_coeff * loss_base, "loss_base": loss_base}

    def convert_model_pred(self, output, xt, t):
        return (xt - output) / _expand_t(t, xt).clamp_min(self.t_eps)

    def compute_loss(self, output, vt, xt, t):
        return (self.convert_model_pred(output, xt, t) - vt) ** 2

    def get_drift(self):
        def body_fn(x, t, model, **model_kwargs):
            output = model(x, t, **model_kwargs)
            # Unguided calls return both heads; IG returns their guided prediction.
            if isinstance(output, tuple):
                output = output[0]
            return self.convert_model_pred(output, x, t)
        return body_fn
