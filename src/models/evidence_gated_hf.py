import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceGatedHFNAF(nn.Module):
    """
    Frozen degradation-aware NAF backbone +
    lightweight evidence-gated high-frequency residual head.
    """

    def __init__(
        self,
        base_model,
        feature_width=48,
        descriptor_channels=16,
        detail_width=24,
        max_residual=0.15,
    ):
        super().__init__()

        self.base = base_model
        self.max_residual = float(max_residual)

        # Preserve champion.
        for p in self.base.parameters():
            p.requires_grad = False

        # HR backbone features + measured HF + |HF| + gradient
        input_channels = feature_width + 3

        self.detail_head = nn.Sequential(
            nn.Conv2d(input_channels, detail_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(detail_width, detail_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(detail_width, 1, 3, padding=1),
        )

        # Local evidence: where should detail be restored?
        self.local_gate = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        # Existing degradation descriptor controls global detail strength.
        self.global_gate = nn.Linear(
            descriptor_channels,
            1,
        )

        # Start EXACTLY at old champion behaviour.
        nn.init.zeros_(self.detail_head[-1].weight)
        nn.init.zeros_(self.detail_head[-1].bias)

        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)

        nn.init.zeros_(self.global_gate.weight)
        nn.init.constant_(self.global_gate.bias, 1.0)

    @staticmethod
    def measurement_features(x):
        # Keep RAW input — no clipping.
        bicubic = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )

        low = F.avg_pool2d(
            bicubic,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        hf = bicubic - low

        gx = F.pad(
            bicubic[:, :, :, 1:] - bicubic[:, :, :, :-1],
            (0, 1, 0, 0),
        )

        gy = F.pad(
            bicubic[:, :, 1:, :] - bicubic[:, :, :-1, :],
            (0, 0, 0, 1),
        )

        grad = torch.sqrt(
            gx * gx + gy * gy + 1e-6
        )

        return hf, grad

    def forward(self, x, return_parts=False):

        # ---------- original champion ----------
        descriptor = self.base.degradation_encoder(x)

        shallow = self.base.shallow(x)
        restored = shallow

        for block in self.base.restoration:
            restored = block(
                restored,
                descriptor,
            )

        restored = (
            shallow
            + self.base.refinement(restored)
        )

        hr_features = self.base.upsample(
            restored
        )

        base_logits = self.base.reconstruction(
            hr_features
        )

        base_prediction = self.base.output_mapping(
            base_logits
        )

        # ---------- measured detail evidence ----------
        hf, grad = self.measurement_features(x)

        features = torch.cat(
            [
                hr_features,
                hf,
                torch.abs(hf),
                grad,
            ],
            dim=1,
        )

        # Predict correction, not whole image.
        detail = (
            torch.tanh(
                self.detail_head(features)
            )
            * self.max_residual
        )

        local_gate = torch.sigmoid(
            self.local_gate(features)
        )

        global_gate = torch.sigmoid(
            self.global_gate(descriptor)
        ).view(-1, 1, 1, 1)

        gate = local_gate * global_gate

        final = (
            base_prediction
            + gate * detail
        )

        if return_parts:
            return (
                final,
                base_prediction,
                detail,
                gate,
            )

        return final
