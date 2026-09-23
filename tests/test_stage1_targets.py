"""Full-target conditioning and the fixed surrogate used by both paper stages."""
import torch
from torch import nn

from stage1.combine import MLSCombine
from stage1.rae_variants import RAECombine


class FixedEncoder(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.tokens


def make_wrapper(p_drop):
    # Construct only the frozen fusion path; no weights or network are needed.
    wrapper = RAECombine.__new__(RAECombine)
    nn.Module.__init__(wrapper)
    wrapper.resolution = 2
    wrapper.eps = 1e-5
    wrapper.img_mean = torch.zeros(1, 3, 1, 1)
    wrapper.img_std = torch.ones(1, 3, 1, 1)
    wrapper.encoder = FixedEncoder([torch.full((8, 4, 2), float(i)) for i in (1, 3, 9)])
    wrapper.decoder = nn.Identity()
    wrapper.combine = MLSCombine([1, 11, 23], p_drop=p_drop)
    wrapper.drop = True
    wrapper.do_normalization = True
    wrapper.latent_mean = torch.tensor([1., 2.]).reshape(1, 2, 1, 1)
    wrapper.latent_var = torch.tensor([4., 9.]).reshape(1, 2, 1, 1)
    wrapper.train()
    return wrapper


def test_full_target_uses_one_encoder_pass_and_shared_normalization():
    model = make_wrapper(1.)
    torch.manual_seed(2)
    conditioned, target = model.encode_cond_target(torch.zeros(8, 3, 2, 2))
    assert model.encoder.calls == 1
    assert model.combine.training
    assert not model.encoder.training
    assert not model.decoder.training
    raw_target = target * (model.latent_var + model.eps).sqrt() + model.latent_mean
    # Full fusion 13/3 plus the fixed last-layer surrogate 9.
    torch.testing.assert_close(raw_target, torch.full_like(raw_target, 13 / 3 + 9))
    raw_conditioned = conditioned * (model.latent_var + model.eps).sqrt() + model.latent_mean
    values = raw_conditioned[:, 0, 0, 0]
    torch.testing.assert_close(values, values.round())
    assert set(values.round().tolist()) <= {10., 12., 18.}
    torch.testing.assert_close(raw_conditioned, values[:, None, None, None].expand_as(raw_conditioned))


def test_no_drop_input_and_full_target_are_identical():
    model = make_wrapper(0.)
    conditioned, target = model.encode_cond_target(torch.zeros(8, 3, 2, 2))
    torch.testing.assert_close(conditioned, target, rtol=0, atol=0)
