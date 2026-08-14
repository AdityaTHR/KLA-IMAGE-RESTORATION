import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Robust pixel loss: sqrt((prediction - target)^2 + epsilon^2)."""

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon_sq = epsilon ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon_sq)
        return loss.mean()

class L1ReconstructionLoss(nn.Module):
    """Mean absolute reconstruction error."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target)


class SSIMLoss(nn.Module):
    """Differentiable local structural-similarity loss for grayscale images."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("SSIM window_size must be an odd integer >= 3")
        self.window_size = window_size
        self.padding = window_size // 2
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.padding)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=self.padding)
        mu_pred_sq = mu_pred.square()
        mu_target_sq = mu_target.square()
        mu_cross = mu_pred * mu_target
        sigma_pred = F.avg_pool2d(pred.square(), self.window_size, 1, self.padding) - mu_pred_sq
        sigma_target = (
            F.avg_pool2d(target.square(), self.window_size, 1, self.padding) - mu_target_sq
        )
        sigma_cross = F.avg_pool2d(pred * target, self.window_size, 1, self.padding) - mu_cross
        numerator = (2.0 * mu_cross + self.c1) * (2.0 * sigma_cross + self.c2)
        denominator = (mu_pred_sq + mu_target_sq + self.c1) * (
            sigma_pred + sigma_target + self.c2
        )
        ssim = numerator / denominator.clamp_min(1e-12)
        return 1.0 - ssim.mean()


class EdgeLoss(nn.Module):
    """Charbonnier distance between horizontal and vertical image gradients."""

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon_sq = epsilon ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        loss_x = torch.sqrt((pred_dx - target_dx).square() + self.epsilon_sq).mean()
        loss_y = torch.sqrt((pred_dy - target_dy).square() + self.epsilon_sq).mean()
        return 0.5 * (loss_x + loss_y)


class FrequencyLoss(nn.Module):
    """L1 distance between orthonormal 2D Fourier coefficients."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_frequency = torch.fft.rfft2(pred.float(), norm='ortho')
        target_frequency = torch.fft.rfft2(target.float(), norm='ortho')
        return torch.abs(pred_frequency - target_frequency).mean()


class CompositeRestorationLoss(nn.Module):
    """Charbonnier plus one or more explicitly weighted structure terms."""

    def __init__(
        self,
        epsilon: float = 1e-3,
        pixel_weight: float = 1.0,
        ssim_weight: float = 0.0,
        edge_weight: float = 0.0,
        frequency_weight: float = 0.0,
        ssim_window_size: int = 11,
    ):
        super().__init__()
        weights = (pixel_weight, ssim_weight, edge_weight, frequency_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Composite loss weights must be nonnegative with a positive total")
        self.pixel_weight = float(pixel_weight)
        self.ssim_weight = float(ssim_weight)
        self.edge_weight = float(edge_weight)
        self.frequency_weight = float(frequency_weight)
        self.pixel = CharbonnierLoss(epsilon=epsilon)
        self.ssim = SSIMLoss(window_size=ssim_window_size)
        self.edge = EdgeLoss(epsilon=epsilon)
        self.frequency = FrequencyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pred.new_zeros(())
        if self.pixel_weight:
            loss = loss + self.pixel_weight * self.pixel(pred, target)
        if self.ssim_weight:
            loss = loss + self.ssim_weight * self.ssim(pred, target)
        if self.edge_weight:
            loss = loss + self.edge_weight * self.edge(pred, target)
        if self.frequency_weight:
            loss = loss + self.frequency_weight * self.frequency(pred, target)
        return loss

def get_loss(name: str, **kwargs) -> nn.Module:
    """Build a configured reconstruction loss."""
    losses = {
        'charbonnier': CharbonnierLoss,
        'l1': L1ReconstructionLoss,
        'composite': CompositeRestorationLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'name'}
    return losses[name](**filtered_kwargs)
