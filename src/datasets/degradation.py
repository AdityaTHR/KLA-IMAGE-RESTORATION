"""Controlled synthetic degradations for robustness experiments."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


class SyntheticDegradation:
    """Create varied speckle/Gaussian/x2 degradations from clean HR targets."""

    SUPPORTED_INTERPOLATION = {"area", "bilinear", "bicubic"}

    def __init__(
        self,
        probability: float = 0.5,
        gaussian_sigma: Sequence[float] = (0.0, 0.08),
        speckle_sigma: Sequence[float] = (0.0, 0.18),
        interpolation_modes: Sequence[str] = ("area", "bilinear", "bicubic"),
        high_resolution_noise_probability: float = 0.5,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("synthetic degradation probability must lie in [0,1]")
        if not 0.0 <= high_resolution_noise_probability <= 1.0:
            raise ValueError("high_resolution_noise_probability must lie in [0,1]")
        if (
            len(gaussian_sigma) != 2
            or gaussian_sigma[0] < 0
            or gaussian_sigma[1] < gaussian_sigma[0]
        ):
            raise ValueError("gaussian_sigma must be a nonnegative [min,max] range")
        if len(speckle_sigma) != 2 or speckle_sigma[0] < 0 or speckle_sigma[1] < speckle_sigma[0]:
            raise ValueError("speckle_sigma must be a nonnegative [min,max] range")
        modes = tuple(str(mode).lower() for mode in interpolation_modes)
        if not modes or any(mode not in self.SUPPORTED_INTERPOLATION for mode in modes):
            raise ValueError(
                f"interpolation_modes must come from {sorted(self.SUPPORTED_INTERPOLATION)}"
            )

        self.probability = float(probability)
        self.gaussian_sigma = tuple(float(value) for value in gaussian_sigma)
        self.speckle_sigma = tuple(float(value) for value in speckle_sigma)
        self.interpolation_modes = modes
        self.high_resolution_noise_probability = float(high_resolution_noise_probability)

    def should_apply(self) -> bool:
        return bool(np.random.random() < self.probability)

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        gaussian_sigma = np.random.uniform(*self.gaussian_sigma)
        speckle_sigma = np.random.uniform(*self.speckle_sigma)
        operations = ["gaussian", "speckle"]
        np.random.shuffle(operations)

        degraded = image.astype(np.float32, copy=True)
        for operation in operations:
            if operation == "gaussian" and gaussian_sigma > 0:
                noise = np.random.normal(0.0, gaussian_sigma, degraded.shape).astype(np.float32)
                degraded = degraded + noise
            elif operation == "speckle" and speckle_sigma > 0:
                noise = np.random.normal(0.0, speckle_sigma, degraded.shape).astype(np.float32)
                degraded = degraded + degraded * noise
        return degraded.astype(np.float32, copy=False)

    def _downsample(self, image: np.ndarray) -> np.ndarray:
        mode = self.interpolation_modes[np.random.randint(0, len(self.interpolation_modes))]
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        if mode == "area":
            result = F.interpolate(tensor, size=(128, 128), mode=mode)
        else:
            result = F.interpolate(
                tensor,
                size=(128, 128),
                mode=mode,
                align_corners=False,
                antialias=True,
            )
        return result.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)

    def __call__(self, gt: np.ndarray) -> np.ndarray:
        if gt.shape != (256, 256) or gt.dtype != np.float32:
            raise ValueError(
                f"Expected float32 GT with shape (256,256), got {gt.shape}/{gt.dtype}"
            )

        if np.random.random() < self.high_resolution_noise_probability:
            degraded = self._downsample(self._add_noise(gt))
        else:
            degraded = self._add_noise(self._downsample(gt))

        if degraded.shape != (128, 128) or degraded.dtype != np.float32:
            raise AssertionError("Synthetic degradation violated the NoisyLR shape/dtype contract")
        if not np.isfinite(degraded).all():
            raise FloatingPointError("Synthetic degradation produced NaN or Inf")
        return degraded
