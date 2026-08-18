import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import SyntheticDegradation


class PairedDataset(Dataset):
    """Load paired float32 NoisyLR and GT arrays without clipping the input."""

    def __init__(
        self,
        split_file: str,
        noisy_dir: str,
        gt_dir: str,
        augment: bool = False,
        synthetic_degradation: dict | None = None,
    ):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.augment = augment
        self.synthetic_degradation = (
            SyntheticDegradation(**synthetic_degradation)
            if synthetic_degradation is not None
            else None
        )

        with open(split_file, 'r') as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        for fname in self.filenames:
            noisy_path = os.path.join(noisy_dir, fname)
            gt_path = os.path.join(gt_dir, fname)
            assert os.path.exists(noisy_path), f"Missing NoisyLR: {noisy_path}"
            assert os.path.exists(gt_path), f"Missing GT: {gt_path}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        noisy = np.load(os.path.join(self.noisy_dir, fname))
        gt = np.load(os.path.join(self.gt_dir, fname))

        assert noisy.shape == (128, 128), f"Bad noisy shape: {noisy.shape}"
        assert gt.shape == (256, 256), f"Bad GT shape: {gt.shape}"
        assert noisy.dtype == np.float32, f"Bad noisy dtype: {noisy.dtype}"
        assert gt.dtype == np.float32, f"Bad GT dtype: {gt.dtype}"
        assert np.isfinite(noisy).all(), f"Non-finite NoisyLR values: {fname}"
        assert np.isfinite(gt).all(), f"Non-finite GT values: {fname}"
        assert 0.0 <= float(gt.min()) and float(gt.max()) <= 1.0, f"GT outside [0,1]: {fname}"

        if self.synthetic_degradation is not None and self.synthetic_degradation.should_apply():
            noisy = self.synthetic_degradation(gt)

        if self.augment:
            noisy, gt = self._augment(noisy, gt)

        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0)

        return noisy_t, gt_t

    def _augment(self, noisy: np.ndarray, gt: np.ndarray):
        """Apply identical flips and 90-degree rotations to an image pair."""
        if np.random.random() > 0.5:
            noisy = np.flip(noisy, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()

        if np.random.random() > 0.5:
            noisy = np.flip(noisy, axis=0).copy()
            gt = np.flip(gt, axis=0).copy()

        k = np.random.randint(0, 4)
        if k > 0:
            noisy = np.rot90(noisy, k).copy()
            gt = np.rot90(gt, k).copy()

        return noisy, gt


class TestDataset(Dataset):
    """Load blind-test NoisyLR arrays with their filenames."""

    def __init__(self, noisy_dir: str):
        self.noisy_dir = noisy_dir
        self.filenames = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith('.npy')
        ])
        assert len(self.filenames) > 0, f"No .npy files found in {noisy_dir}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        noisy = np.load(os.path.join(self.noisy_dir, fname))
        assert noisy.shape == (128, 128), f"Bad shape: {noisy.shape}"
        assert noisy.dtype == np.float32, f"Bad dtype: {noisy.dtype}"
        assert np.isfinite(noisy).all(), f"Non-finite values: {fname}"
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        return noisy_t, fname
