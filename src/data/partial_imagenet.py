"""
Load available Arrow shards from a partial ImageNet download.
Works even when only a subset of shards has been downloaded.
"""
import glob
import os
import datasets
from torch.utils.data import Dataset


class PartialImageNetDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        arrow_dir = os.path.join(data_dir, "imagenet-latents-images")
        shards = sorted(glob.glob(os.path.join(arrow_dir, "data-*.arrow")))
        assert shards, f"No Arrow shards found in {arrow_dir}"

        self.ds = datasets.concatenate_datasets([
            datasets.Dataset.from_file(s) for s in shards
        ])
        self.transform = transform
        print(f"[PartialImageNet] Loaded {len(self.ds):,} images "
              f"from {len(shards)} shards  (split={split})", flush=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(sample["label"])
