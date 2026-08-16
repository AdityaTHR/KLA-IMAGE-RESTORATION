"""Reusable lightweight NAFNet-style restoration blocks."""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over channels independently at every spatial position."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d expects BCHW input, got shape {tuple(x.shape)}")
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Split channels in half and multiply the two feature groups."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError(f"SimpleGate requires an even channel count, got {x.shape[1]}")
        first, second = x.chunk(2, dim=1)
        return first * second


class SimplifiedChannelAttention(nn.Module):
    """Low-cost pooled channel reweighting with a configurable bottleneck."""

    def __init__(self, channels: int, reduction_factor: int = 4):
        super().__init__()
        if reduction_factor < 1:
            raise ValueError("attention reduction factor must be at least 1")
        hidden_channels = max(channels // reduction_factor, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_map = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel_map(self.pool(x))


class NAFBlock(nn.Module):
    """Compact two-branch NAFNet-style residual restoration block."""

    def __init__(
        self,
        channels: int,
        expansion_factor: int = 2,
        attention_reduction_factor: int = 4,
        residual_scale: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if channels < 1 or expansion_factor < 1:
            raise ValueError("channels and expansion_factor must be positive")

        expanded_channels = channels * expansion_factor
        if expanded_channels % 2 != 0:
            raise ValueError("channels * expansion_factor must be even for SimpleGate")
        gated_channels = expanded_channels // 2

        self.norm1 = LayerNorm2d(channels, eps=layer_norm_eps)
        self.expand1 = nn.Conv2d(channels, expanded_channels, 1)
        self.depthwise = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            padding=1,
            groups=expanded_channels,
        )
        self.gate1 = SimpleGate()
        self.attention = SimplifiedChannelAttention(
            gated_channels, reduction_factor=attention_reduction_factor
        )
        self.project1 = nn.Conv2d(gated_channels, channels, 1)

        self.norm2 = LayerNorm2d(channels, eps=layer_norm_eps)
        self.expand2 = nn.Conv2d(channels, expanded_channels, 1)
        self.gate2 = SimpleGate()
        self.project2 = nn.Conv2d(gated_channels, channels, 1)

        scale_shape = (1, channels, 1, 1)
        self.beta = nn.Parameter(torch.full(scale_shape, float(residual_scale)))
        self.gamma = nn.Parameter(torch.full(scale_shape, float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.expand1(self.norm1(x))
        features = self.depthwise(features)
        features = self.gate1(features)
        features = self.attention(features)
        features = self.project1(features)
        first_residual = x + self.beta * features

        refinement = self.expand2(self.norm2(first_residual))
        refinement = self.gate2(refinement)
        refinement = self.project2(refinement)
        return first_residual + self.gamma * refinement
