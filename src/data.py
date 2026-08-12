from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedNpyDataset(Dataset):
    """Paired grayscale NoisyLR -> GT dataset for x2 restoration."""

    def __init__(self, root, ids, patch_size_lr=64, augment=False):
        self.root = Path(root)
        self.lr_dir = self.root / "NoisyLR"
        self.gt_dir = self.root / "GT"
        self.ids = list(ids)
        self.patch = patch_size_lr
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _load_pair(self, name):
        lr = np.load(self.lr_dir / name).astype(np.float32)
        gt = np.load(self.gt_dir / name).astype(np.float32)
        if gt.shape[0] != lr.shape[0] * 2 or gt.shape[1] != lr.shape[1] * 2:
            raise ValueError(f"Expected x2 pair for {name}: LR={lr.shape}, GT={gt.shape}")
        return lr, gt

    def _crop(self, lr, gt):
        if self.patch is None or self.patch >= min(lr.shape):
            return lr, gt
        p = self.patch
        y = random.randint(0, lr.shape[0] - p)
        x = random.randint(0, lr.shape[1] - p)
        lr = lr[y:y+p, x:x+p]
        gt = gt[2*y:2*(y+p), 2*x:2*(x+p)]
        return lr, gt

    def _augment(self, lr, gt):
        if random.random() < 0.5:
            lr, gt = np.fliplr(lr), np.fliplr(gt)
        if random.random() < 0.5:
            lr, gt = np.flipud(lr), np.flipud(gt)
        k = random.randint(0, 3)
        if k:
            lr, gt = np.rot90(lr, k), np.rot90(gt, k)
        return np.ascontiguousarray(lr), np.ascontiguousarray(gt)

    def __getitem__(self, idx):
        name = self.ids[idx]
        lr, gt = self._load_pair(name)
        lr, gt = self._crop(lr, gt)
        if self.augment:
            lr, gt = self._augment(lr, gt)
        # IMPORTANT: do not clip the LR input. KLA intentionally allows values outside [0,1].
        lr = torch.from_numpy(lr).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        return lr, gt, name


def make_split(root, val_fraction=0.10, seed=42):
    root = Path(root)
    ids = sorted(p.name for p in (root / "GT").glob("*.npy"))
    lr_ids = {p.name for p in (root / "NoisyLR").glob("*.npy")}
    ids = [x for x in ids if x in lr_ids]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    return ids[n_val:], ids[:n_val]
