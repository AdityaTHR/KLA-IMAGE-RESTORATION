import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Standard residual block with two conv layers and ReLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class UpsampleBlock(nn.Module):
    """Upsampling via sub-pixel convolution (PixelShuffle)."""
    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))
