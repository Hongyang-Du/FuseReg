from .sampler import Sampler
from .transport import Transport


def create_transport(config, time_dist_shift=1.0):
    return Transport(time_dist_shift=time_dist_shift, t_eps=config.t_eps)


def create_sampler(transport):
    return Sampler(transport)


__all__ = ["create_transport", "create_sampler", "Transport", "Sampler"]
