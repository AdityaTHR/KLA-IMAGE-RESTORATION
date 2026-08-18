"""Degradation-aware NAFNet-style joint restoration and 2x SR model."""

import torch
import torch.nn as nn

from .nafnet_blocks import NAFBlock
from .nafnet_sr import PixelShuffleUpsampler


class DegradationEncoder(nn.Module):
    """Estimate a compact degradation descriptor from the unmodified LR input."""

    def __init__(self, in_channels: int, hidden_channels: int, descriptor_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, descriptor_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


class ConditionedNAFBlock(nn.Module):
    """NAF block followed by lightweight descriptor-conditioned affine modulation."""

    def __init__(
        self,
        channels: int,
        descriptor_channels: int,
        expansion_factor: int,
        attention_reduction_factor: int,
        residual_scale: float,
        layer_norm_eps: float,
        modulation_scale: float,
    ):
        super().__init__()
        self.block = NAFBlock(
            channels=channels,
            expansion_factor=expansion_factor,
            attention_reduction_factor=attention_reduction_factor,
            residual_scale=residual_scale,
            layer_norm_eps=layer_norm_eps,
        )
        self.modulation = nn.Linear(descriptor_channels, channels * 2)
        self.modulation_scale = float(modulation_scale)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        restored = self.block(x)
        scale, shift = self.modulation(descriptor).chunk(2, dim=1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1) * self.modulation_scale
        shift = torch.tanh(shift).unsqueeze(-1).unsqueeze(-1) * self.modulation_scale
        return restored * (1.0 + scale) + shift


class DegradationAwareNAFNetSR(nn.Module):
    """B2-derived model with learned conditioning from the raw degraded input."""

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
        degradation_hidden_channels: int = 8,
        degradation_descriptor_channels: int = 16,
        modulation_scale: float = 0.1,
    ):
        super().__init__()
        if upsampling_method.lower() != "pixelshuffle":
            raise ValueError("DegradationAwareNAFNetSR requires pixelshuffle upsampling")
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_width = feature_width
        self.num_blocks = num_blocks
        self.scale_factor = 2
        self.descriptor_channels = degradation_descriptor_channels

        self.degradation_encoder = DegradationEncoder(
            in_channels,
            degradation_hidden_channels,
            degradation_descriptor_channels,
        )
        self.shallow = nn.Conv2d(in_channels, feature_width, 3, padding=1)
        self.restoration = nn.ModuleList(
            [
                ConditionedNAFBlock(
                    channels=feature_width,
                    descriptor_channels=degradation_descriptor_channels,
                    expansion_factor=expansion_factor,
                    attention_reduction_factor=attention_reduction_factor,
                    residual_scale=residual_scale,
                    layer_norm_eps=layer_norm_eps,
                    modulation_scale=modulation_scale,
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

        descriptor = self.degradation_encoder(x)
        shallow = self.shallow(x)
        restored = shallow
        for block in self.restoration:
            restored = block(restored, descriptor)
        restored = shallow + self.refinement(restored)
        prediction = self.reconstruction(self.upsample(restored))
        return self.output_mapping(prediction)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture_summary(self) -> dict:
        return {
            "input_channels": self.in_channels,
            "output_channels": self.out_channels,
            "feature_width": self.feature_width,
            "naf_blocks": self.num_blocks,
            "degradation_conditioning": "learned global descriptor with per-block affine modulation",
            "degradation_descriptor_channels": self.descriptor_channels,
            "processing_space": "128x128 LR feature space",
            "upsampling": "learned pixel shuffle x2 near output",
            "output_mapping": "sigmoid",
        }
