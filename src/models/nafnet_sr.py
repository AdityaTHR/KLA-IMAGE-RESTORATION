"""Lightweight NAFNet-inspired network for joint restoration and 2x SR."""

import torch
import torch.nn as nn

from .nafnet_blocks import NAFBlock


class PixelShuffleUpsampler(nn.Module):
    """Learned 2x reconstruction performed only after LR-space restoration."""

    def __init__(self, channels: int, scale_factor: int = 2):
        super().__init__()
        self.projection = nn.Conv2d(
            channels, channels * scale_factor * scale_factor, 3, padding=1
        )
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.projection(x))


class NAFNetSR(nn.Module):
    """Compact grayscale NAFNet-style restoration model with late 2x upsampling.

    The network receives the original degraded float32 values without clipping or
    input normalization. LayerNorm is applied only to learned feature maps inside
    each block, over channels at each spatial location. A final sigmoid constrains
    only the reconstructed clean prediction to [0, 1].
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_width: int = 32,
        num_blocks: int = 8,
        expansion_factor: int = 2,
        attention_reduction_factor: int = 4,
        residual_scale: float = 0.1,
        upsampling_method: str = "pixelshuffle",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if upsampling_method.lower() != "pixelshuffle":
            raise ValueError(
                f"Unsupported upsampling method: {upsampling_method}. Use pixelshuffle."
            )
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = 2
        self.feature_width = feature_width
        self.num_blocks = num_blocks

        self.shallow = nn.Conv2d(in_channels, feature_width, 3, padding=1)
        self.restoration = nn.Sequential(
            *[
                NAFBlock(
                    channels=feature_width,
                    expansion_factor=expansion_factor,
                    attention_reduction_factor=attention_reduction_factor,
                    residual_scale=residual_scale,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_blocks)
            ]
        )
        self.refinement = nn.Conv2d(feature_width, feature_width, 3, padding=1)
        self.upsample = PixelShuffleUpsampler(feature_width, scale_factor=2)
        self.reconstruction = nn.Conv2d(feature_width, out_channels, 3, padding=1)
        self.output_mapping = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channel(s), got {tuple(x.shape)}"
            )

        shallow = self.shallow(x)
        restored = self.restoration(shallow)
        restored = shallow + self.refinement(restored)
        upsampled = self.upsample(restored)
        prediction = self.reconstruction(upsampled)
        return self.output_mapping(prediction)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture_summary(self) -> dict:
        return {
            "input_channels": self.in_channels,
            "output_channels": self.out_channels,
            "feature_width": self.feature_width,
            "naf_blocks": self.num_blocks,
            "processing_space": "128x128 LR feature space",
            "upsampling": "learned pixel shuffle x2 near output",
            "output_mapping": "sigmoid",
        }
