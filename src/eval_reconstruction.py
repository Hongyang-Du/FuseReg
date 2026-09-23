#!/usr/bin/env python3
"""Evaluate released decoders on a fixed image subset, with explicit layer feeds."""
import argparse
import json
import hashlib
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from utils.inference import load_combine_weights, load_decoder_checkpoint, file_sha256, config_provenance
from utils.image_archive import inspect_image_archive
from utils.model_utils import get_obj_from_str


def resolve_output_normalization(encoder_name, p_drop, explicit=None):
    """Require an explicit convention for the unresolved DINO no-drop lineage."""
    if explicit is not None:
        if explicit not in ('encoder', 'raw'):
            raise ValueError('Output normalization must be encoder or raw')
        return explicit
    if encoder_name.startswith('dinov3') and p_drop == 0:
        raise ValueError(
            'DINO p_drop=0 checkpoint output normalization is unresolved: the released '
            'baseline uses a different training lineage and its checkpoint has no '
            'normalization metadata. Verify the checkpoint recipe and pass '
            '--decoder-output-normalization raw or --decoder-output-normalization encoder explicitly.'
        )
    return 'encoder'


def selected_positions(trained_layers, indices=None, layers=None):
    if indices is not None:
        result = list(indices)
        if not result or any(i < 0 or i >= len(trained_layers) for i in result):
            raise ValueError('Subset indices must be nonempty, nonnegative positions in the configured layers')
    elif layers is not None:
        if not layers or any(layer not in trained_layers for layer in layers):
            raise ValueError('Requested layers must belong to the configured checkpoint layers')
        result = [trained_layers.index(layer) for layer in layers]
    else:
        return None
    if len(set(result)) != len(result):
        raise ValueError('Subset must not contain duplicate layers')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--val-npz', required=True, help='uint8 NHWC images (.npz or memory-mapped .npy)')
    parser.add_argument('--num-images', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--idx', help='Comma-separated positions within the checkpoint layer list')
    group.add_argument('--layers', help='Comma-separated actual encoder block IDs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--metrics', nargs='+', choices=['psnr', 'ssim', 'rfid'], default=['psnr', 'ssim', 'rfid'])
    parser.add_argument('--decoder-output-normalization', choices=['encoder', 'raw'], default=None,
                        help='encoder: oldnorm normalized pixels; raw: [0,1]. Required for the unresolved DINO no-drop checkpoint lineage')
    parser.add_argument('--save-recon-npz')
    parser.add_argument('--work-dir', help='Directory for temporary memory-mapped FID images (about 20 GB for 50k at 256px)')
    parser.add_argument('opts', nargs='*', help='OmegaConf dotlist config overrides')
    args = parser.parse_args()
    if args.num_images < 1 or args.batch < 1:
        parser.error('--num-images and --batch must be positive')
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.opts))
    params = OmegaConf.to_container(cfg.combine.params, resolve=True)
    encoder_prefix = str(cfg.get('encoder_name') or 'dinov3mls-vit-l16').split('[')[0]
    args.decoder_output_normalization = resolve_output_normalization(
        encoder_prefix, params.get('p_drop'), args.decoder_output_normalization)
    resolution = int(cfg.get('data', {}).get('image_size', 256))
    inspect_image_archive(args.val_npz, resolution=resolution, min_images=args.num_images)
    device = torch.device(args.device)
    decoder_state, combine_state, metadata = load_decoder_checkpoint(args.ckpt)
    configured_layers = list(params['layers'])
    layers = metadata.get('layers', configured_layers)
    if not isinstance(layers, list) or layers != configured_layers:
        raise ValueError(f'Checkpoint layers {layers} do not match configuration layers {configured_layers}')
    idx = selected_positions(layers,
                             [int(i) for i in args.idx.split(',')] if args.idx else None,
                             [int(i) for i in args.layers.split(',')] if args.layers else None)
    encoder = create_encoder(encoder_prefix + '[layers=' + '.'.join(map(str, layers)) + ']', device, resolution).eval()
    encoder.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    combine = get_obj_from_str(cfg.combine.target)(**params).to(device).eval()
    load_combine_weights(combine, combine_state)
    dcfg = cfg.get('decoder', {})
    decoder = _load_decoder(dcfg.get('config_path', 'configs/decoder/ViTXL'),
                            hidden_size=dcfg.get('latent_dim', 1024),
                            patch_size=dcfg.get('patch_size', 16),
                            num_patches=dcfg.get('num_patches', (resolution // 16) ** 2)).to(device).eval()
    decoder.load_state_dict(decoder_state, strict=True)
    del decoder_state
    archive = np.load(args.val_npz, mmap_mode='r', allow_pickle=False)
    images = archive[archive.files[0]] if isinstance(archive, np.lib.npyio.NpzFile) else archive
    if images.dtype != np.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != (resolution, resolution, 3):
        raise ValueError(f'Expected uint8 [N,{resolution},{resolution},3] images; got {images.shape} {images.dtype}')
    if len(images) < args.num_images:
        raise ValueError(f'Requested {args.num_images} images but reference has only {len(images)}')
    sample_indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(args.seed))[:args.num_images].tolist()
    psnr_sum, ssim_sum, count = 0., 0., 0
    workspace = None
    recon_array = reference_array = None
    if 'rfid' in args.metrics or args.save_recon_npz:
        if args.work_dir:
            Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix='fusereg-eval-', dir=args.work_dir)
        shape = (args.num_images, resolution, resolution, 3)
        recon_array = np.lib.format.open_memmap(Path(workspace.name) / 'recon.npy', mode='w+', dtype=np.uint8, shape=shape)
        reference_array = np.lib.format.open_memmap(Path(workspace.name) / 'reference.npy', mode='w+', dtype=np.uint8, shape=shape)
    if 'ssim' in args.metrics:
        from torchmetrics.functional.image import structural_similarity_index_measure
    with torch.inference_mode():
        for offset in range(0, len(sample_indices), args.batch):
            chosen = sample_indices[offset:offset + args.batch]
            source = np.asarray(images[chosen]).copy()
            x = torch.from_numpy(source).permute(0, 3, 1, 2).float().to(device) / 255
            autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with autocast:
                normalized = (x - mean) / std
                tokens = encoder(normalized)
                z = combine(tokens, idx=idx)
                prediction = decoder(z, drop_cls_token=False).logits
                reconstructed = decoder.unpatchify(prediction)
                if args.decoder_output_normalization == 'encoder':
                    reconstructed = reconstructed * std + mean
            reconstructed = reconstructed.clamp(0, 1).float()
            if 'psnr' in args.metrics:
                mse = (reconstructed - x).square().mean((1, 2, 3)).clamp_min(1e-10)
                psnr_sum += (-10 * mse.log10()).sum().item()
            if 'ssim' in args.metrics:
                ssim_sum += structural_similarity_index_measure(reconstructed, x, data_range=1.).item() * len(chosen)
            if 'rfid' in args.metrics or args.save_recon_npz:
                recon_array[count:count + len(chosen)] = reconstructed.mul(255).round().permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                reference_array[count:count + len(chosen)] = source
            count += len(chosen)
            if offset == 0 or count % (args.batch * 50) == 0:
                print(f'Evaluated {count}/{args.num_images}', flush=True)
    result = dict(ckpt=args.ckpt, epoch=metadata.get('epoch'), trained_layers=layers,
                  eval_layers=[layers[i] for i in idx] if idx is not None else layers,
                  num_images=count, seed=args.seed, weights='ema', reference=args.val_npz,
                  decoder_output_normalization=args.decoder_output_normalization,
                  cls_surrogate=True,
                  encoder=encoder_prefix, device=str(device), precision='bf16' if device.type == 'cuda' else 'fp32')
    result.update(config_provenance(cfg))
    result['checkpoint_sha256'] = file_sha256(args.ckpt)
    result['reference_sha256'] = file_sha256(args.val_npz)
    result['checkpoint_metadata'] = metadata
    result['sample_indices_sha256'] = hashlib.sha256(np.asarray(sample_indices, dtype='<i8').tobytes()).hexdigest()
    if 'psnr' in args.metrics:
        result['psnr'] = psnr_sum / count
    if 'ssim' in args.metrics:
        result['ssim'] = ssim_sum / count
    if 'rfid' in args.metrics:
        from eval.fid import calculate_rfid
        result['rfid_tf'] = float(calculate_rfid(recon_array, reference_array, bs=args.batch, device=device))
        result['rfid_backend'] = 'torch-fidelity-0.3.0, paired reference images'
    if args.save_recon_npz:
        Path(args.save_recon_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_recon_npz, recon=recon_array, idxs=np.array(sample_indices))
    if workspace is not None:
        del recon_array, reference_array
        workspace.cleanup()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
