"""Synthetic CUDA/AMP contract test for B2; writes no project artifacts."""

import argparse
import json
import os
import sys

import torch
import yaml
from torch.amp import GradScaler, autocast

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from src.losses.reconstruction import get_loss
from src.models import build_model
from src.utils.seed import set_seed


EXPECTED_PARAMETERS = 108_609
EXPECTED_INPUT_SHAPE = (2, 1, 128, 128)
EXPECTED_OUTPUT_SHAPE = (2, 1, 256, 256)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical B2 CUDA/AMP smoke test")
    parser.add_argument('--config', default='configs/baseline_nafnet.yaml')
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            "CUDA-enabled PyTorch is required for this smoke test; "
            "torch.cuda.is_available() is False."
        )

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPOSITORY_ROOT, config_path)
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)

    if config['model'].get('name') != 'nafnet_sr':
        raise ValueError("CUDA smoke test requires the canonical nafnet_sr configuration")

    set_seed(config['training'].get('seed', 42))
    device = torch.device('cuda')
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    model = build_model(config['model']).to(device).train()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise AssertionError(
            f"B2 parameter count changed: expected {EXPECTED_PARAMETERS}, got {parameter_count}"
        )

    loss_config = config['training']['loss'].copy()
    loss_name = loss_config.pop('name')
    criterion = get_loss(loss_name, **loss_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['optimizer']['lr'])
    scaler = GradScaler('cuda', enabled=True)

    degraded = torch.rand(EXPECTED_INPUT_SHAPE, device=device, dtype=torch.float32) * 1.8 - 0.08
    degraded[0, 0, 0, 0] = -0.08
    degraded[0, 0, 0, 1] = 1.72
    degraded_before = degraded.detach().clone()
    target = torch.rand(EXPECTED_OUTPUT_SHAPE, device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type='cuda', enabled=True):
        prediction = model(degraded)
        loss = criterion(prediction, target)

    if tuple(prediction.shape) != EXPECTED_OUTPUT_SHAPE:
        raise AssertionError(f"Unexpected output shape: {tuple(prediction.shape)}")
    if not torch.equal(degraded, degraded_before):
        raise AssertionError("B2 modified the raw degraded input tensor")
    if not torch.isfinite(prediction).all():
        raise AssertionError("B2 prediction contains NaN or Inf")
    if float(prediction.detach().min()) < 0.0 or float(prediction.detach().max()) > 1.0:
        raise AssertionError("B2 prediction lies outside [0,1]")
    if not torch.isfinite(loss):
        raise AssertionError("Charbonnier loss is NaN or Inf")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    trainable_tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    if len(gradients) != trainable_tensors:
        raise AssertionError("At least one trainable parameter did not receive a gradient")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("At least one B2 gradient contains NaN or Inf")

    result = {
        'status': 'PASS',
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device),
        'pytorch_version': torch.__version__,
        'pytorch_cuda_build': torch.version.cuda,
        'amp_enabled': scaler.is_enabled(),
        'trainable_parameters': parameter_count,
        'input_shape': list(degraded.shape),
        'input_min': float(degraded.min()),
        'input_max': float(degraded.max()),
        'output_shape': list(prediction.shape),
        'output_finite': True,
        'output_min': float(prediction.detach().min()),
        'output_max': float(prediction.detach().max()),
        'charbonnier_loss_finite': True,
        'backward_passed': True,
        'gradients_finite': True,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
