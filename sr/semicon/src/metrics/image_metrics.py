import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


def calculate_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute PSNR for two HxW images."""
    pred_clipped = np.clip(pred, 0.0, 1.0)
    target_clipped = np.clip(target, 0.0, 1.0)
    return float(peak_signal_noise_ratio(target_clipped, pred_clipped, data_range=data_range))

def calculate_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute SSIM for two HxW images."""
    pred_clipped = np.clip(pred, 0.0, 1.0)
    target_clipped = np.clip(target, 0.0, 1.0)
    return float(structural_similarity(target_clipped, pred_clipped, data_range=data_range))

class LPIPSMetric:
    """LPIPS wrapper with internal grayscale-to-RGB conversion."""

    def __init__(self, net: str = 'alex', device: str = 'cpu'):
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
        self.loss_fn.eval()

    def calculate(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute LPIPS after mapping grayscale [0,1] images to RGB [-1,1]."""
        pred_clipped = np.clip(pred, 0.0, 1.0)
        target_clipped = np.clip(target, 0.0, 1.0)

        pred_t = torch.from_numpy(pred_clipped).float().unsqueeze(0).unsqueeze(0)
        target_t = torch.from_numpy(target_clipped).float().unsqueeze(0).unsqueeze(0)

        pred_t = pred_t.repeat(1, 3, 1, 1)
        target_t = target_t.repeat(1, 3, 1, 1)

        pred_t = pred_t * 2.0 - 1.0
        target_t = target_t * 2.0 - 1.0

        pred_t = pred_t.to(self.device)
        target_t = target_t.to(self.device)

        with torch.no_grad():
            score = self.loss_fn(pred_t, target_t)
        return float(score.item())

class MetricsCalculator:
    """Compute the project's PSNR, SSIM, and LPIPS metrics."""

    def __init__(self, device='cpu', lpips_net: str = 'alex'):
        device_str = str(device) if not isinstance(device, str) else device
        self.lpips_metric = LPIPSMetric(net=lpips_net, device=device_str)

    def calculate_all(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """Compute all metrics for prediction and target arrays."""
        return {
            'psnr': calculate_psnr(pred, target),
            'ssim': calculate_ssim(pred, target),
            'lpips': self.lpips_metric.calculate(pred, target),
        }
