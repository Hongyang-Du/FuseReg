from math import sqrt
import torch
from transformers import AutoConfig
from .decoders import GeneralDecoder

def _load_decoder(config_path, hidden_size, patch_size, num_patches, pretrained_path=None):
    config = AutoConfig.from_pretrained(config_path)
    config.hidden_size = hidden_size
    config.patch_size = patch_size
    config.image_size = int(patch_size * sqrt(num_patches))
    decoder = GeneralDecoder(config, num_patches=num_patches)
    if pretrained_path is not None:
        print(f"Loading pretrained decoder from {pretrained_path}")
        from utils.inference import load_decoder_checkpoint
        state_dict, _, _ = load_decoder_checkpoint(pretrained_path)
        decoder.load_state_dict(state_dict, strict=True)
    return decoder

def _load_normalization_stats(path):
    if path is None:
        return None, None, False
    stats = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(stats, dict) or not {'mean', 'var'} <= stats.keys():
        raise ValueError(f'Latent statistics {path} must contain mean and var tensors')
    for name in ('mean', 'var'):
        if not isinstance(stats[name], torch.Tensor) or not torch.isfinite(stats[name]).all():
            raise ValueError(f'Invalid {name} tensor in {path}')
    if (stats['var'] < 0).any():
        raise ValueError(f'Negative latent variance in {path}')
    print(f"Loaded normalization stats from {path}")
    return stats.get('mean', None), stats.get('var', None), True
