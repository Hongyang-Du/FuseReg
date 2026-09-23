"""Read released EMA safetensors and original training checkpoints."""
import json
from pathlib import Path

import torch


def read_checkpoint(path):
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            metadata = handle.metadata() or {}
        for key in ('layers', 'epoch', 'step'):
            if key in metadata:
                try:
                    metadata[key] = json.loads(metadata[key])
                except (ValueError, TypeError):
                    pass
        return load_file(str(path), device='cpu'), metadata
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Expected a state dictionary in {path}')
    return checkpoint, {}


def strip_prefixes(state):
    result = {}
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        result[key] = value
    return result


def load_decoder_checkpoint(path, use_ema=True):
    checkpoint, metadata = read_checkpoint(path)
    decoder_key = 'ema_dec' if use_ema else 'decoder'
    combine_key = 'ema_combine' if use_ema else 'combine'
    if decoder_key in checkpoint:
        metadata.update({key: checkpoint[key] for key in ('layers', 'epoch', 'step') if key in checkpoint})
        return strip_prefixes(checkpoint[decoder_key]), checkpoint.get(combine_key), metadata
    # Baseline RAE training snapshots store the complete encoder+decoder wrapper.
    wrapper_key = 'ema' if use_ema else 'model'
    wrapper = checkpoint.get(wrapper_key)
    if isinstance(wrapper, dict):
        wrapper = strip_prefixes(wrapper)
        if 'decoder.decoder_embed.weight' in wrapper:
            decoder = {key[len('decoder.'):]: value for key, value in wrapper.items() if key.startswith('decoder.')}
            metadata.update({key: checkpoint[key] for key in ('layers', 'epoch', 'step') if key in checkpoint})
            metadata['checkpoint_format'] = 'baseline_rae_wrapper'
            return decoder, None, metadata
    state = strip_prefixes(checkpoint)
    if 'decoder_embed.weight' not in state:
        raise ValueError(f'{path} has no {decoder_key!r} or bare decoder state dictionary')
    if not use_ema and str(path).endswith('.safetensors'):
        raise ValueError('Released safetensors contain EMA weights only; remove --raw.')
    return state, None, metadata


def load_combine_weights(combine, state):
    if state is not None:
        combine.load_state_dict(state, strict=True)
    elif combine.state_dict():
        raise ValueError('The paper fusion must be parameter-free.')


def load_dit_checkpoint(path, use_ema=True):
    checkpoint, metadata = read_checkpoint(path)
    key = 'ema' if use_ema else 'model'
    if key in checkpoint:
        metadata.update({k: checkpoint[k] for k in ('epoch', 'step') if k in checkpoint})
        return strip_prefixes(checkpoint[key]), metadata
    if not use_ema and str(path).endswith('.safetensors'):
        raise ValueError('Released safetensors contain EMA weights only; remove --raw.')
    state = strip_prefixes(checkpoint)
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f'{path} has no {key!r} or bare DiT state dictionary')
    return state, metadata


def file_sha256(path):
    """Digest an artifact without loading it into memory."""
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_provenance(config):
    import hashlib
    from omegaconf import OmegaConf
    resolved = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    canonical = json.dumps(resolved, sort_keys=True, separators=(',', ':'))
    return {'resolved_config': resolved, 'config_sha256': hashlib.sha256(canonical.encode()).hexdigest()}
