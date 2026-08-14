"""Evaluate a checkpoint on a configured validation or development-test split."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from src.datasets.paired_dataset import PairedDataset
from src.metrics.image_metrics import MetricsCalculator
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.image_io import tensor_to_numpy


def project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPOSITORY_ROOT, path)


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the configured val or dev-test split"
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', required=True, choices=['val', 'dev_test'])
    parser.add_argument('--results-file', default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    checkpoint_path = os.path.abspath(args.checkpoint)
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)

    split_key = {'val': 'val_split', 'dev_test': 'dev_test_split'}[args.split]
    if split_key not in config['data']:
        raise KeyError(f"Configuration has no data.{split_key} entry")
    split_file = project_path(config['data'][split_key])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    model = build_model(config['model']).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    dataset = PairedDataset(
        split_file=split_file,
        noisy_dir=project_path(config['data']['noisy_dir']),
        gt_dir=project_path(config['data']['gt_dir']),
        augment=False,
    )
    if len(dataset) != 320:
        raise ValueError(
            f"Expected 320 samples in canonical {args.split} split, found {len(dataset)}"
        )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    metrics_calc = MetricsCalculator(device=device)
    results = {'psnr': [], 'ssim': [], 'lpips': [], 'time_ms': []}

    warmup_noisy, _ = dataset[0]
    with torch.no_grad():
        model(warmup_noisy.unsqueeze(0).to(device))
    synchronize(device)

    with torch.no_grad():
        for index, (noisy, gt) in enumerate(tqdm(loader, desc=f"Evaluating {args.split}")):
            noisy = noisy.to(device)
            synchronize(device)
            start = time.perf_counter()
            prediction = model(noisy)
            synchronize(device)
            results['time_ms'].append((time.perf_counter() - start) * 1000.0)

            filename = dataset.filenames[index]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(
                    f"Non-finite prediction (NaN or Inf) produced for sample: {filename}"
                )

            prediction_np = np.clip(tensor_to_numpy(prediction), 0.0, 1.0)
            gt_np = tensor_to_numpy(gt)
            metrics = metrics_calc.calculate_all(prediction_np, gt_np)
            for name, value in metrics.items():
                results[name].append(value)

    summary = {
        'run_id': config.get('run_id', config['model']['name']),
        'model': config['model']['name'],
        'architecture': (
            model.architecture_summary()
            if hasattr(model, 'architecture_summary')
            else {'class': type(model).__name__}
        ),
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'split': args.split,
        'split_file': os.path.abspath(split_file),
        'num_images': len(dataset),
        'device': str(device),
        'trainable_parameters': sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        'metrics': {
            'psnr': float(np.mean(results['psnr'])),
            'ssim': float(np.mean(results['ssim'])),
            'lpips': float(np.mean(results['lpips'])),
        },
        'ms_per_image': float(np.mean(results['time_ms'])),
    }

    print(json.dumps(summary, indent=2))
    if args.results_file:
        results_path = os.path.abspath(args.results_file)
        results_parent = os.path.dirname(results_path)
        if results_parent:
            os.makedirs(results_parent, exist_ok=True)
        with open(results_path, 'w') as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved summary to {results_path}")


if __name__ == '__main__':
    main()
