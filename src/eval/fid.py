"""FID and Inception Score with torch-fidelity 0.3.0 and explicit reference images."""
from contextlib import nullcontext

import torch
from torch_fidelity import calculate_metrics

from .utils import ImgArrDataset


def _calculate(arr1, arr2, bs, device, isc):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('torch-fidelity supports CPU or CUDA evaluation')
    context = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
    with context:
        return calculate_metrics(input1=ImgArrDataset(arr1), input2=ImgArrDataset(arr2),
                                 batch_size=bs, fid=True, isc=isc, cuda=device.type == 'cuda')


def calculate_rfid(arr1, arr2, bs=64, device='cuda'):
    return _calculate(arr1, arr2, bs, device, False)['frechet_inception_distance']


def calculate_fid_isc(arr1, arr2, bs=64, device='cuda'):
    metrics = _calculate(arr1, arr2, bs, device, True)
    return metrics['frechet_inception_distance'], metrics['inception_score_mean']
