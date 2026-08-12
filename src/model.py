import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + 0.1 * self.body(x)


class BaselineRestorer(nn.Module):
    """Small joint denoising + x2 super-resolution baseline.

    This is deliberately simple. It gives us a learned reference before NAFNet/PromptIR ideas.
    """
    def __init__(self, channels=48, blocks=6, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.fuse = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x):
        feat = self.head(x)
        feat = feat + self.fuse(self.body(feat))
        return self.up(feat)
