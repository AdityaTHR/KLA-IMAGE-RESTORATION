"""Spatial + global degradation-aware NAFNet-style joint restoration and 2x SR."""

import torch
import torch.nn as nn

from .nafnet_blocks import NAFBlock
from .nafnet_sr import PixelShuffleUpsampler


class GlobalDegradationEncoder(nn.Module):
    """Estimate a compact global degradation descriptor from raw LR input."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        descriptor_channels: int,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                descriptor_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


class SpatialDegradationEncoder(nn.Module):
    """
    Produce a spatial degradation representation.

    Unlike the global 16-D descriptor, this preserves HxW location
    information so different parts of the image can receive different
    restoration behaviour.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        spatial_channels: int,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                spatial_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpatialConditionedNAFBlock(nn.Module):
    """
    NAF block with:

        1. global degradation-conditioned affine modulation
        2. spatially varying degradation-conditioned modulation

    The names `block` and `modulation` intentionally match the previous
    model so that later we can transplant W48x10 weights if desired.
    """

    def __init__(
        self,
        channels: int,
        descriptor_channels: int,
        spatial_channels: int,
        expansion_factor: int,
        attention_reduction_factor: int,
        residual_scale: float,
        layer_norm_eps: float,
        modulation_scale: float,
        spatial_modulation_scale: float,
    ):
        super().__init__()

        self.block = NAFBlock(
            channels=channels,
            expansion_factor=expansion_factor,
            attention_reduction_factor=attention_reduction_factor,
            residual_scale=residual_scale,
            layer_norm_eps=layer_norm_eps,
        )

        # Global affine modulation.
        # Same attribute name as old model for checkpoint compatibility.
        self.modulation = nn.Linear(
            descriptor_channels,
            channels * 2,
        )

        # Local/spatial affine modulation.
        self.spatial_modulation = nn.Conv2d(
            spatial_channels,
            channels * 2,
            kernel_size=1,
        )

        self.modulation_scale = float(
            modulation_scale
        )

        self.spatial_modulation_scale = float(
            spatial_modulation_scale
        )

        # Start both conditioning branches conservatively.
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        nn.init.zeros_(self.spatial_modulation.weight)
        nn.init.zeros_(self.spatial_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        descriptor: torch.Tensor,
        spatial_descriptor: torch.Tensor,
    ) -> torch.Tensor:

        restored = self.block(x)

        # -------------------------------------------------
        # GLOBAL CONDITIONING
        # -------------------------------------------------

        global_scale, global_shift = (
            self.modulation(descriptor)
            .chunk(2, dim=1)
        )

        global_scale = (
            torch.tanh(global_scale)
            .unsqueeze(-1)
            .unsqueeze(-1)
            * self.modulation_scale
        )

        global_shift = (
            torch.tanh(global_shift)
            .unsqueeze(-1)
            .unsqueeze(-1)
            * self.modulation_scale
        )

        restored = (
            restored * (1.0 + global_scale)
            + global_shift
        )

        # -------------------------------------------------
        # SPATIAL CONDITIONING
        # -------------------------------------------------

        spatial_scale, spatial_shift = (
            self.spatial_modulation(
                spatial_descriptor
            ).chunk(2, dim=1)
        )

        spatial_scale = (
            torch.tanh(spatial_scale)
            * self.spatial_modulation_scale
        )

        spatial_shift = (
            torch.tanh(spatial_shift)
            * self.spatial_modulation_scale
        )

        restored = (
            restored * (1.0 + spatial_scale)
            + spatial_shift
        )

        return restored


class SpatialDegradationAwareNAFNetSR(nn.Module):
    """
    W48/Wxx NAFNet SR model using both global and local degradation cues.

    Input:
        B x 1 x 128 x 128

    Output:
        B x 1 x 256 x 256
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_width: int = 48,
        num_blocks: int = 10,
        expansion_factor: int = 2,
        attention_reduction_factor: int = 4,
        residual_scale: float = 0.1,
        upsampling_method: str = "pixelshuffle",
        layer_norm_eps: float = 1e-6,
        degradation_hidden_channels: int = 8,
        degradation_descriptor_channels: int = 16,
        spatial_hidden_channels: int = 8,
        spatial_descriptor_channels: int = 8,
        modulation_scale: float = 0.1,
        spatial_modulation_scale: float = 0.05,
    ):
        super().__init__()

        if upsampling_method.lower() != "pixelshuffle":
            raise ValueError(
                "SpatialDegradationAwareNAFNetSR "
                "requires pixelshuffle upsampling"
            )

        if num_blocks < 1:
            raise ValueError(
                "num_blocks must be at least 1"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_width = feature_width
        self.num_blocks = num_blocks
        self.scale_factor = 2

        self.descriptor_channels = (
            degradation_descriptor_channels
        )

        self.spatial_descriptor_channels = (
            spatial_descriptor_channels
        )

        # -------------------------------------------------
        # GLOBAL DEGRADATION PATH
        # -------------------------------------------------

        self.degradation_encoder = (
            GlobalDegradationEncoder(
                in_channels=in_channels,
                hidden_channels=degradation_hidden_channels,
                descriptor_channels=(
                    degradation_descriptor_channels
                ),
            )
        )

        # -------------------------------------------------
        # LOCAL/SPATIAL DEGRADATION PATH
        # -------------------------------------------------

        self.spatial_degradation_encoder = (
            SpatialDegradationEncoder(
                in_channels=in_channels,
                hidden_channels=spatial_hidden_channels,
                spatial_channels=(
                    spatial_descriptor_channels
                ),
            )
        )

        # -------------------------------------------------
        # IMAGE RESTORATION BACKBONE
        # -------------------------------------------------

        self.shallow = nn.Conv2d(
            in_channels,
            feature_width,
            kernel_size=3,
            padding=1,
        )

        self.restoration = nn.ModuleList(
            [
                SpatialConditionedNAFBlock(
                    channels=feature_width,
                    descriptor_channels=(
                        degradation_descriptor_channels
                    ),
                    spatial_channels=(
                        spatial_descriptor_channels
                    ),
                    expansion_factor=expansion_factor,
                    attention_reduction_factor=(
                        attention_reduction_factor
                    ),
                    residual_scale=residual_scale,
                    layer_norm_eps=layer_norm_eps,
                    modulation_scale=modulation_scale,
                    spatial_modulation_scale=(
                        spatial_modulation_scale
                    ),
                )
                for _ in range(num_blocks)
            ]
        )

        self.refinement = nn.Conv2d(
            feature_width,
            feature_width,
            kernel_size=3,
            padding=1,
        )

        self.upsample = PixelShuffleUpsampler(
            feature_width,
            scale_factor=2,
        )

        self.reconstruction = nn.Conv2d(
            feature_width,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # Keep same output mapping as champion.
        self.output_mapping = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if (
            x.ndim != 4
            or x.shape[1] != self.in_channels
        ):
            raise ValueError(
                "Expected BCHW input with "
                f"{self.in_channels} channel(s), "
                f"got {tuple(x.shape)}"
            )

        # Raw input is intentionally NOT clipped.
        descriptor = self.degradation_encoder(x)

        spatial_descriptor = (
            self.spatial_degradation_encoder(x)
        )

        shallow = self.shallow(x)

        restored = shallow

        for block in self.restoration:
            restored = block(
                restored,
                descriptor,
                spatial_descriptor,
            )

        restored = (
            shallow
            + self.refinement(restored)
        )

        prediction = self.reconstruction(
            self.upsample(restored)
        )

        return self.output_mapping(prediction)

    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def architecture_summary(self) -> dict:
        return {
            "input_channels": self.in_channels,
            "output_channels": self.out_channels,
            "feature_width": self.feature_width,
            "naf_blocks": self.num_blocks,
            "degradation_conditioning": (
                "global descriptor + learned spatial "
                "degradation map with per-block affine modulation"
            ),
            "degradation_descriptor_channels": (
                self.descriptor_channels
            ),
            "spatial_degradation_channels": (
                self.spatial_descriptor_channels
            ),
            "processing_space": (
                "128x128 LR feature space"
            ),
            "upsampling": (
                "learned pixel shuffle x2 near output"
            ),
            "output_mapping": "sigmoid",
        }
