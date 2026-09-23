"""Full-target and gradient-accumulation behavior without GPU/data downloads."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import torch

from configs.stage2 import Stage2Config
from stage2.engine import train_one_epoch
from stage2.transport.transport import Transport


class TinyIG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, x, t, context):
        return x * self.weight, x * self.weight * .5


class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.accumulating = False
        self.forward_modes = []

    @contextmanager
    def no_sync(self):
        self.accumulating = True
        try:
            yield
        finally:
            self.accumulating = False

    def forward(self, *args, **kwargs):
        self.forward_modes.append(self.accumulating)
        return self.module(*args, **kwargs)


class FrozenRAE:
    def encode_cond_target(self, images):
        return images * 2, images * 3


class Batches(list):
    def set_epoch(self, epoch):
        self.epoch = epoch


class Progress:
    def __init__(self):
        self.updates = 0

    def update(self, n):
        self.updates += n


def test_engine_uses_full_target_and_accumulates_before_ema_update(tmp_path):
    config = Stage2Config()
    config.training.grad_accum_steps = 2
    config.training.ema_decay = .5
    config.training.clip_grad = None
    config.training.log_interval = 0
    config.training.sample_every = 0
    config.training.checkpoint_interval = 0
    model = TinyIG()
    reference = deepcopy(model)
    ema, expected_ema = deepcopy(model), deepcopy(model)
    ddp = LocalDDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    data = Batches((torch.full((2, 1, 2, 2), (i + 1) / 10), torch.tensor([0, 1])) for i in range(5))
    transport = Transport(time_dist_shift=8)
    progress = Progress()
    torch.manual_seed(456)
    actual_step = train_one_epoch(
        ddp_model=ddp, ema_model=ema, rae=FrozenRAE(), transport=transport,
        eval_sampler=None, dataloader=data, optimizer=optimizer, scheduler=None,
        autocast_kwargs={"enabled": False}, device=torch.device("cpu"), epoch=3,
        global_step=5, config=config, args=SimpleNamespace(wandb=False), rank=0,
        checkpoint_dir=str(tmp_path), experiment_dir=str(tmp_path), progress_bar=progress,
        viz_fixed={"zs": torch.zeros(2, 1, 2, 2), "context": None},
    )
    torch.manual_seed(456)
    for i, (images, labels) in enumerate(data[:4]):
        terms = transport.training_losses(
            reference, images * 2, {"context": labels},
            {"context": torch.full_like(labels, 1000)}, x1_target=images * 3,
            cfg_dropout_prob=.1,
        )
        (terms["loss"].mean() / 2).backward()
        if i % 2:
            ref_optimizer.step()
            ref_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                expected_ema.weight.mul_(.5).add_(reference.weight, alpha=.5)
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(ema.weight, expected_ema.weight, rtol=0, atol=0)
    assert actual_step == 7 and progress.updates == 2
    assert ddp.forward_modes == [True, False, True, False]
    assert data.epoch == 3
