"""Lightweight eval data utilities. LPIPS lives in disc/lpips.py."""

import torch
from torch.utils.data import Dataset


class ImgArrDataset(Dataset):
    """Wrapper for torch-fidelity FID calculation — expects [B, H, W, C] uint8 arrays."""

    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return torch.from_numpy(self.arr[idx]).permute(2, 0, 1)
