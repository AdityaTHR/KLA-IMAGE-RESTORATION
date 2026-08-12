import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        return (
            x * self.weight[:, None, None]
            + self.bias[:, None, None]
        )


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        hidden = channels * 2

        self.norm1 = LayerNorm2d(channels)

        self.conv1 = nn.Conv2d(channels, hidden, 1)

        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=1,
            groups=hidden
        )

        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

        self.conv2 = nn.Conv2d(channels, channels, 1)

        self.norm2 = LayerNorm2d(channels)

        self.ffn1 = nn.Conv2d(channels, hidden, 1)
        self.ffn2 = nn.Conv2d(channels, channels, 1)

        self.beta = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.gamma = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, x):

        y = self.norm1(x)

        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg(y)

        y = y * self.sca(y)
        y = self.conv2(y)

        x = x + self.beta * y

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg(y)
        y = self.ffn2(y)

        return x + self.gamma * y


class NAFRestorer(nn.Module):

    def __init__(self, width=48, blocks=8, scale=2):
        super().__init__()

        self.scale = scale

        self.intro = nn.Conv2d(
            1,
            width,
            3,
            padding=1
        )

        self.body = nn.Sequential(
            *[
                NAFBlock(width)
                for _ in range(blocks)
            ]
        )

        self.fuse = nn.Conv2d(
            width,
            width,
            3,
            padding=1
        )

        self.up = nn.Sequential(

            nn.Conv2d(
                width,
                width * scale * scale,
                3,
                padding=1
            ),

            nn.PixelShuffle(scale),

            nn.Conv2d(
                width,
                1,
                3,
                padding=1
            )
        )

    def forward(self, x):

        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        feat = self.intro(x)

        shortcut = feat

        feat = self.body(feat)

        feat = shortcut + self.fuse(feat)

        correction = self.up(feat)

        return base + correction