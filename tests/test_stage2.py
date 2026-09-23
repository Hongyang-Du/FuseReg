"""Scientific constraints of full-target flow matching and internal guidance."""

from types import SimpleNamespace

import pytest
import torch

from configs.stage2 import GuidanceConfig, IGConfig
from stage2.models.DDT import DiTwDDTHeadIG
from stage2.transport.transport import Transport
from stage2.transport.sampler import Sampler
from utils.guidance_utils import get_model_forward_fn


def test_full_target_and_time_weighting_train_both_heads():
    transport = Transport(t_eps=0.05)
    subset = torch.tensor([2.0, 3.0]).view(2, 1, 1, 1)
    target = torch.tensor([7.0, 9.0]).view_as(subset)
    noise = torch.zeros_like(subset)
    t = torch.tensor([0.01, 0.5])
    transport.sample = lambda x: (t, noise, x)
    full = torch.tensor([4.0, 6.0]).view_as(subset).requires_grad_()
    base = torch.tensor([5.0, 8.0]).view_as(subset).requires_grad_()
    seen = {}

    def model(x, time, context):
        seen["x"] = x
        seen["labels"] = context
        return full, base

    labels = {"context": torch.tensor([1, 2])}
    null = {"context": torch.full((2,), 10)}
    losses = transport.training_losses(model, subset, labels, null, x1_target=target,
                                       base_model_coeff=0.4, cfg_dropout_prob=1.0)
    denominator = t.clamp_min(0.05).view_as(subset) ** 2
    expected_base = (base - target) ** 2 / denominator
    expected = ((full - target) ** 2 + 0.4 * (base - target) ** 2) / denominator
    torch.testing.assert_close(seen["x"], (1 - t.view_as(subset)) * subset)
    torch.testing.assert_close(seen["labels"], null["context"])
    torch.testing.assert_close(losses["loss_base"], expected_base)
    torch.testing.assert_close(losses["loss"], expected)
    losses["loss"].sum().backward()
    torch.testing.assert_close(full.grad, 2 * (full.detach() - target) / denominator)
    torch.testing.assert_close(base.grad, 0.8 * (base.detach() - target) / denominator)


def test_internal_guidance_preserves_batch_and_interval():
    class Heads:
        def forward(self, x, t, context):
            return x + 3, x + 1
        __call__ = forward

    model = Heads()
    x = torch.zeros(4, 2, 2, 2)
    times = torch.tensor([0.1, 0.2, 0.8, 0.9])
    config = GuidanceConfig(ig=IGConfig(scale=1.78, t_min=0.2, t_max=0.8))
    fn, kwargs = get_model_forward_fn(model, config)
    actual = fn(x, times, context=torch.arange(4), **kwargs)
    expected = torch.tensor([3.0, 4.56, 4.56, 3.0]).view(4, 1, 1, 1).expand_as(x)
    torch.testing.assert_close(actual, expected)
    plain, kwargs = get_model_forward_fn(model, GuidanceConfig())
    torch.testing.assert_close(plain(x, times, context=torch.arange(4), **kwargs)[0], x + 3)


def test_euler_uses_shifted_noise_to_data_grid_and_full_head():
    transport = Transport(time_dist_shift=8)
    sampler = Sampler(transport)
    times_seen = []
    def velocity_model(x, t, context):
        times_seen.append(t[0])
        # A constant velocity of 2, represented as an x prediction.
        full = x - 2 * t.view(-1, 1, 1, 1).clamp_min(transport.t_eps)
        return full, torch.full_like(full, 99)
    sample = sampler.sample_ode(num_steps=50)(torch.ones(2, 1, 2, 2), velocity_model,
                                              context=torch.tensor([0, 1]))
    grid = torch.linspace(1, 0, 51)
    shifted = 8 * grid / (1 + 7 * grid)
    torch.testing.assert_close(torch.stack(times_seen), shifted[:-1])
    torch.testing.assert_close(sample, torch.full_like(sample, -1))


def test_ddt_outputs_nonzero_predictions_from_both_heads():
    torch.manual_seed(2)
    model = DiTwDDTHeadIG(input_size=2, in_channels=4, hidden_size=(16, 32),
                          depth=(2, 1), num_heads=(2, 4), base_model_depth=1,
                          num_classes=3, cond_arch=SimpleNamespace(num_t_tokens=4, num_c_tokens=8))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "final_layer" in name or "adaln_modulation" in name:
                parameter.normal_(0, 0.1)
    x = torch.randn(2, 4, 2, 2)
    full, base = model(x, torch.tensor([0.2, 0.8]), context=torch.tensor([0, 3]))
    assert full.shape == base.shape == x.shape
    assert full.abs().sum() > 0
    assert base.abs().sum() > 0
    (full.square().sum() + base.square().sum()).backward()
    assert model.base_final_layer.linear.weight.grad.abs().sum() > 0
    assert model.final_layer.linear.weight.grad.abs().sum() > 0


def test_base_head_depth_must_exist():
    with pytest.raises(ValueError, match="base_model_depth"):
        DiTwDDTHeadIG(depth=(2, 1), base_model_depth=3)
