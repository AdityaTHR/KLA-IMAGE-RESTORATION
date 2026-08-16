import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.image_metrics import MetricsCalculator

def load_split(split_file: str) -> list:
    """Load filenames from a split text file."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Bicubic Baseline Benchmark")
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'dev_test'])
    parser.add_argument('--results_file', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    split_file = os.path.join(base_dir, 'data', 'splits', f'{args.split}.txt')
    noisy_dir = os.path.join(base_dir, 'data', 'extracted', 'train', 'NoisyLR')
    gt_dir = os.path.join(base_dir, 'data', 'extracted', 'train', 'GT')
    out_dir = os.path.join(base_dir, 'outputs', 'bicubic', args.split)

    os.makedirs(out_dir, exist_ok=True)

    files = load_split(split_file)
    print(f"Benchmarking {len(files)} files from {args.split} split...")

    metrics_calc = MetricsCalculator(device='cpu')

    results = {'psnr': [], 'ssim': [], 'lpips': [], 'time': []}

    for i, f in enumerate(files):
        noisy_img = np.load(os.path.join(noisy_dir, f))
        gt_img = np.load(os.path.join(gt_dir, f))

        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0)
        start_time = time.perf_counter()
        pred_tensor = F.interpolate(
            noisy_tensor, scale_factor=2, mode='bicubic', align_corners=False
        )
        pred_img = pred_tensor.squeeze(0).squeeze(0).numpy()
        # Clip output to [0, 1]
        pred_img = np.clip(pred_img, 0.0, 1.0)
        end_time = time.perf_counter()

        metrics = metrics_calc.calculate_all(pred_img, gt_img)

        results['psnr'].append(metrics['psnr'])
        results['ssim'].append(metrics['ssim'])
        results['lpips'].append(metrics['lpips'])
        results['time'].append(end_time - start_time)

        # Save a representative subset
        if i < 20:
            np.save(os.path.join(out_dir, f), pred_img)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(files)}")

    avg_psnr = np.mean(results['psnr'])
    avg_ssim = np.mean(results['ssim'])
    avg_lpips = np.mean(results['lpips'])
    avg_time = np.mean(results['time'])

    print("\nBenchmark Results:")
    print(f"PSNR:  {avg_psnr:.4f}")
    print(f"SSIM:  {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    print(f"Time:  {avg_time*1000:.2f} ms/img")

    summary = {
        'run_id': 'B0_BICUBIC',
        'split': args.split,
        'num_images': len(files),
        'implementation': 'torch.nn.functional.interpolate(mode=bicubic, align_corners=False)',
        'metrics': {
            'psnr': float(avg_psnr),
            'ssim': float(avg_ssim),
            'lpips': float(avg_lpips),
        },
        'ms_per_image': float(avg_time * 1000),
    }
    results_file = args.results_file or os.path.join(
        base_dir, 'outputs', f'bicubic_{args.split}_metrics.json'
    )
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {results_file}")

if __name__ == "__main__":
    main()
