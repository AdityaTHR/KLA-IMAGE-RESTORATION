"""Synchronized model-only inference benchmark for registered restoration models."""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import yaml

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from src.datasets.paired_dataset import PairedDataset
from src.models import build_model
from src.utils.checkpoint import load_checkpoint


def project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPOSITORY_ROOT, path)


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained restoration checkpoint")
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='val', choices=['val', 'dev_test'])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--results-file', default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")

    with open(project_path(args.config), 'r') as handle:
        config = yaml.safe_load(handle)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model']).to(device)
    load_checkpoint(project_path(args.checkpoint), model, device=device)
    model.eval()

    split_key = 'val_split' if args.split == 'val' else 'dev_test_split'
    dataset = PairedDataset(
        project_path(config['data'][split_key]),
        project_path(config['data']['noisy_dir']),
        project_path(config['data']['gt_dir']),
        augment=False,
    )
    sample, _ = dataset[0]
    sample = sample.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(sample)
        synchronize(device)
        timings = []
        for _ in range(args.iterations):
            synchronize(device)
            start = time.perf_counter()
            prediction = model(sample)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)

    if tuple(prediction.shape) != (1, 1, 256, 256) or not torch.isfinite(prediction).all():
        raise AssertionError("Benchmark prediction violated the output contract")
    result = {
        'run_id': config['run_id'],
        'checkpoint': os.path.abspath(project_path(args.checkpoint)),
        'split_sample_source': args.split,
        'benchmark_type': 'model-only batch-1 synchronized forward pass',
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'warmup_iterations': args.warmup,
        'timed_iterations': args.iterations,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'mean_ms_per_image': statistics.fmean(timings),
        'median_ms_per_image': statistics.median(timings),
        'min_ms_per_image': min(timings),
        'max_ms_per_image': max(timings),
    }
    print(json.dumps(result, indent=2))
    if args.results_file:
        result_path = project_path(args.results_file)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as handle:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
