"""Model registry for configuration-selectable restoration networks."""

import torch.nn as nn

from .lightweight_sr import LightweightSRNet
from .nafnet_sr import NAFNetSR
from .degradation_aware_sr import DegradationAwareNAFNetSR
from .spatial_degradation_aware_sr import SpatialDegradationAwareNAFNetSR


def build_model(config: dict) -> nn.Module:
    """Build a restoration model from the YAML model section."""

    model_name = config.get(
        "name",
        "lightweight_sr",
    )

    if model_name == "lightweight_sr":
        return LightweightSRNet(
            in_channels=config.get(
                "in_channels",
                1,
            ),
            out_channels=config.get(
                "out_channels",
                1,
            ),
            num_features=config.get(
                "num_features",
                64,
            ),
            num_blocks=config.get(
                "num_blocks",
                8,
            ),
        )

    if model_name == "nafnet_sr":
        return NAFNetSR(
            in_channels=config.get(
                "in_channels",
                1,
            ),
            out_channels=config.get(
                "out_channels",
                1,
            ),
            feature_width=config.get(
                "feature_width",
                32,
            ),
            num_blocks=config.get(
                "num_blocks",
                8,
            ),
            expansion_factor=config.get(
                "expansion_factor",
                2,
            ),
            attention_reduction_factor=config.get(
                "attention_reduction_factor",
                4,
            ),
            residual_scale=config.get(
                "residual_scale",
                0.1,
            ),
            upsampling_method=config.get(
                "upsampling_method",
                "pixelshuffle",
            ),
            layer_norm_eps=config.get(
                "layer_norm_eps",
                1e-6,
            ),
        )

    if model_name == "degradation_aware_nafnet_sr":
        return DegradationAwareNAFNetSR(
            in_channels=config.get(
                "in_channels",
                1,
            ),
            out_channels=config.get(
                "out_channels",
                1,
            ),
            feature_width=config.get(
                "feature_width",
                32,
            ),
            num_blocks=config.get(
                "num_blocks",
                8,
            ),
            expansion_factor=config.get(
                "expansion_factor",
                2,
            ),
            attention_reduction_factor=config.get(
                "attention_reduction_factor",
                4,
            ),
            residual_scale=config.get(
                "residual_scale",
                0.1,
            ),
            upsampling_method=config.get(
                "upsampling_method",
                "pixelshuffle",
            ),
            layer_norm_eps=config.get(
                "layer_norm_eps",
                1e-6,
            ),
            degradation_hidden_channels=config.get(
                "degradation_hidden_channels",
                8,
            ),
            degradation_descriptor_channels=config.get(
                "degradation_descriptor_channels",
                16,
            ),
            modulation_scale=config.get(
                "modulation_scale",
                0.1,
            ),
        )

    if model_name == "spatial_degradation_aware_nafnet_sr":
        return SpatialDegradationAwareNAFNetSR(
            in_channels=config.get(
                "in_channels",
                1,
            ),
            out_channels=config.get(
                "out_channels",
                1,
            ),
            feature_width=config.get(
                "feature_width",
                48,
            ),
            num_blocks=config.get(
                "num_blocks",
                10,
            ),
            expansion_factor=config.get(
                "expansion_factor",
                2,
            ),
            attention_reduction_factor=config.get(
                "attention_reduction_factor",
                4,
            ),
            residual_scale=config.get(
                "residual_scale",
                0.1,
            ),
            upsampling_method=config.get(
                "upsampling_method",
                "pixelshuffle",
            ),
            layer_norm_eps=config.get(
                "layer_norm_eps",
                1e-6,
            ),
            degradation_hidden_channels=config.get(
                "degradation_hidden_channels",
                8,
            ),
            degradation_descriptor_channels=config.get(
                "degradation_descriptor_channels",
                16,
            ),
            spatial_hidden_channels=config.get(
                "spatial_hidden_channels",
                8,
            ),
            spatial_descriptor_channels=config.get(
                "spatial_descriptor_channels",
                8,
            ),
            modulation_scale=config.get(
                "modulation_scale",
                0.1,
            ),
            spatial_modulation_scale=config.get(
                "spatial_modulation_scale",
                0.05,
            ),
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


__all__ = [
    "LightweightSRNet",
    "NAFNetSR",
    "DegradationAwareNAFNetSR",
    "SpatialDegradationAwareNAFNetSR",
    "build_model",
]
