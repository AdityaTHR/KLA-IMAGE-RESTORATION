import torch
import torch.nn as nn
from .blocks import ResidualBlock, UpsampleBlock

class LightweightSRNet(nn.Module):
    """Lightweight residual CNN for joint denoising + 2x super-resolution.

    Architecture:
        Input (1, 128, 128)
        -> Shallow feature extraction (conv)
        -> N residual blocks
        -> Feature refinement (conv)
        -> Global residual connection
        -> 2x upsample (PixelShuffle)
        -> Output conv
        -> Sigmoid -> (1, 256, 256)

    The model uses global residual learning in the feature space.
    Does NOT clip inputs.
    Output is constrained to [0,1] via sigmoid.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 num_features: int = 64, num_blocks: int = 8):
        super().__init__()

        # Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)

        # Deep feature extraction: stack of residual blocks
        body = [ResidualBlock(num_features) for _ in range(num_blocks)]
        body.append(nn.Conv2d(num_features, num_features, 3, padding=1))
        self.body = nn.Sequential(*body)

        # Upsampling
        self.upsample = UpsampleBlock(num_features, scale_factor=2)

        # Output
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Input tensor of shape (B, 1, 128, 128). Values may be outside [0,1].
        Returns:
            Output tensor of shape (B, 1, 256, 256) in [0, 1].
        """
        # Shallow features
        shallow = self.head(x)

        # Deep features with global residual
        deep = self.body(shallow)
        features = shallow + deep  # global skip connection

        # Upsample 2x
        up = self.upsample(features)

        # Output
        out = self.tail(up)
        out = self.sigmoid(out)

        return out

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(config: dict) -> nn.Module:
    """Build model from config dict."""
    model_name = config.get('name', 'lightweight_sr')
    if model_name == 'lightweight_sr':
        return LightweightSRNet(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            num_features=config.get('num_features', 64),
            num_blocks=config.get('num_blocks', 8),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
