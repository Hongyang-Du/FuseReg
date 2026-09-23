"""Distributed ImageNet loading for full-target DiT training."""
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from .imagenet_hf_dataset import ImageNetHFDataset


@dataclass
class DataloaderResult:
    loader: DataLoader
    sampler: DistributedSampler
    dataset_size: int

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def prepare_imagenet_dataloader(config, image_size, batch_size, num_workers,
                               rank=0, world_size=1, transform=None, shuffle=True):
    # The source HF training recipe uses this deterministic resize/center crop.
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    split = config.get("split", "train")
    data_dir = config["data_dir"]
    if config.get("type", "hf") == "hf":
        dataset = ImageNetHFDataset(data_dir, split=split, transform=transform)
    elif config["type"] == "imagefolder":
        dataset = datasets.ImageFolder(Path(data_dir) / split, transform=transform)
    else:
        raise ValueError("ImageNet dataset.type must be 'hf' or 'imagefolder'")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True, drop_last=shuffle,
                        persistent_workers=num_workers > 0,
                        multiprocessing_context="forkserver" if num_workers > 0 else None)
    return DataloaderResult(loader, sampler, len(dataset))
