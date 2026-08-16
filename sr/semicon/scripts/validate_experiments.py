"""Validate every registered experiment without training or writing artifacts."""

import argparse
import json
import os
import sys

import torch
import yaml

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from src.datasets.degradation import SyntheticDegradation
from src.losses.reconstruction import get_loss
from src.models import build_model
from src.utils.experiment_registry import (
    load_experiment_registry,
    validate_experiment_registry,
)


EXPECTED_SPLITS = {"train": 2560, "val": 320, "dev_test": 320}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SemiCon experiment registry")
    parser.add_argument('--registry', default='configs/experiment_registry.yaml')
    args = parser.parse_args()

    registry_path = args.registry
    if not os.path.isabs(registry_path):
        registry_path = os.path.join(REPOSITORY_ROOT, registry_path)
    registry = load_experiment_registry(registry_path)
    errors = validate_experiment_registry(registry, REPOSITORY_ROOT)

    split_counts = {}
    split_filenames = {}
    for split_name, expected_count in EXPECTED_SPLITS.items():
        split_path = os.path.join(REPOSITORY_ROOT, 'data', 'splits', f'{split_name}.txt')
        with open(split_path, 'r') as handle:
            filenames = [line.strip() for line in handle if line.strip()]
        split_counts[split_name] = len(filenames)
        split_filenames[split_name] = set(filenames)
        if len(filenames) != expected_count or len(filenames) != len(set(filenames)):
            errors.append(f"Invalid canonical {split_name} split")
    if any(
        split_filenames[left] & split_filenames[right]
        for left, right in (("train", "val"), ("train", "dev_test"), ("val", "dev_test"))
    ):
        errors.append("Canonical train/val/dev_test splits overlap")

    model_results = {}
    for experiment in registry['experiments']:
        if experiment['parameters'] == 0:
            continue
        config_path = os.path.join(REPOSITORY_ROOT, experiment['config'])
        with open(config_path, 'r') as handle:
            config = yaml.safe_load(handle)
        if config.get('run_id') != experiment['run_id']:
            errors.append(f"run_id mismatch in {experiment['config']}")
            continue
        expected_checkpoint = os.path.join(
            config['checkpoint']['dir'], config['checkpoint']['best_filename']
        ).replace('\\', '/')
        if experiment['checkpoint'] != expected_checkpoint:
            errors.append(f"checkpoint-path mismatch for {experiment['run_id']}")
        expected_history = os.path.join(
            config['checkpoint']['dir'], f"{config['run_id']}_training_history.csv"
        ).replace('\\', '/')
        if experiment['history'] != expected_history:
            errors.append(f"history-path mismatch for {experiment['run_id']}")
        for key, split_name in (
            ('train_split', 'train'),
            ('val_split', 'val'),
            ('dev_test_split', 'dev_test'),
        ):
            expected_path = f"data/splits/{split_name}.txt"
            if config['data'].get(key) != expected_path:
                errors.append(f"{experiment['run_id']} uses noncanonical {key}")

        model = build_model(config['model'])
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if parameter_count != experiment['parameters']:
            errors.append(
                f"{experiment['run_id']} parameter mismatch: "
                f"registry={experiment['parameters']} actual={parameter_count}"
            )

        degraded = torch.rand(1, 1, 128, 128, dtype=torch.float32) * 1.8 - 0.08
        degraded[0, 0, 0, 0] = -0.08
        degraded[0, 0, 0, 1] = 1.72
        degraded_before = degraded.clone()
        target = torch.rand(1, 1, 256, 256, dtype=torch.float32)
        prediction = model(degraded)
        loss_config = config['training']['loss'].copy()
        loss_name = loss_config.pop('name')
        loss = get_loss(loss_name, **loss_config)(prediction, target)
        loss.backward()

        detached_prediction = prediction.detach()
        valid = (
            tuple(prediction.shape) == (1, 1, 256, 256)
            and torch.equal(degraded, degraded_before)
            and bool(torch.isfinite(prediction).all())
            and float(detached_prediction.min()) >= 0.0
            and float(detached_prediction.max()) <= 1.0
            and bool(torch.isfinite(loss))
            and all(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        if not valid:
            errors.append(f"{experiment['run_id']} failed its model/loss contract")
        model_results[experiment['run_id']] = {
            'parameters': parameter_count,
            'output_shape': list(prediction.shape),
            'output_min': float(prediction.detach().min()),
            'output_max': float(prediction.detach().max()),
            'contract_passed': valid,
        }

        degradation_config = config['training'].get('synthetic_degradation')
        if degradation_config:
            synthetic_config = degradation_config.copy()
            synthetic_config['probability'] = 1.0
            degradation = SyntheticDegradation(**synthetic_config)
            clean_target = target[0, 0].detach().numpy().astype('float32', copy=False)
            synthetic_input = degradation(clean_target)
            if (
                synthetic_input.shape != (128, 128)
                or synthetic_input.dtype.name != 'float32'
                or not bool(torch.isfinite(torch.from_numpy(synthetic_input)).all())
            ):
                errors.append(f"{experiment['run_id']} failed its synthetic-degradation contract")

    result = {
        'status': 'PASS' if not errors else 'FAIL',
        'registered_experiments': len(registry['experiments']),
        'split_counts': split_counts,
        'models': model_results,
        'errors': errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
