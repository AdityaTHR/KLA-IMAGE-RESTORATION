import os
import yaml
import time
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

from src.utils.checkpoint import load_checkpoint
from src.utils.image_io import load_npy_image, save_npy_image, numpy_to_tensor, tensor_to_numpy
from src.metrics.image_metrics import MetricsCalculator
from src.models import build_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate SR Model")
    parser.add_argument('--config', required=True, type=str, help='Path to config yaml')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint')
    parser.add_argument(
        '--input_dir', required=True, type=str, help='Directory of input .npy files'
    )
    parser.add_argument('--output_dir', required=True, type=str, help='Directory to save outputs')
    parser.add_argument(
        '--gt_dir',
        type=str,
        default=None,
        help='Directory of GT .npy files for metrics calculation',
    )
    parser.add_argument('--split_file', type=str, default=None,
                        help='Optional text file restricting evaluation to listed filenames')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Optional path for a machine-readable JSON summary')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_model(config['model']).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.split_file:
        with open(args.split_file, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npy')])
    missing_inputs = [f for f in files if not os.path.isfile(os.path.join(args.input_dir, f))]
    if missing_inputs:
        raise FileNotFoundError(
            f"Missing {len(missing_inputs)} input files; first: {missing_inputs[0]}"
        )
    if args.gt_dir:
        missing_gt = [f for f in files if not os.path.isfile(os.path.join(args.gt_dir, f))]
        if missing_gt:
            raise FileNotFoundError(f"Missing {len(missing_gt)} GT files; first: {missing_gt[0]}")
    print(f"Found {len(files)} files to process.")

    metrics_calc = MetricsCalculator(device=device) if args.gt_dir else None
    metrics_results = {'psnr': [], 'ssim': [], 'lpips': []}
    times = []

    if files:
        warmup_np = load_npy_image(os.path.join(args.input_dir, files[0]))
        with torch.no_grad():
            _ = model(numpy_to_tensor(warmup_np).unsqueeze(0).to(device))

    with torch.no_grad():
        for f in tqdm(files, desc="Evaluating"):
            input_path = os.path.join(args.input_dir, f)
            noisy_np = load_npy_image(input_path)

            noisy_tensor = numpy_to_tensor(noisy_np).unsqueeze(0).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_t = time.perf_counter()
            pred_tensor = model(noisy_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_t = time.perf_counter()
            times.append(end_t - start_t)

            if not torch.isfinite(pred_tensor).all():
                raise FloatingPointError(
                    f"Non-finite prediction (NaN or Inf) produced for sample: {f}"
                )

            expected_shape = (1, 1, noisy_tensor.shape[-2] * 2, noisy_tensor.shape[-1] * 2)
            if tuple(pred_tensor.shape) != expected_shape:
                raise ValueError(
                    f"Invalid output shape for {f}: expected {expected_shape}, "
                    f"got {tuple(pred_tensor.shape)}"
                )
            output_min = float(pred_tensor.min())
            output_max = float(pred_tensor.max())
            if output_min < 0.0 or output_max > 1.0:
                raise ValueError(
                    f"Output outside [0,1] for {f}: min={output_min}, max={output_max}"
                )

            pred_np = tensor_to_numpy(pred_tensor.squeeze(0))
            pred_np = np.clip(pred_np, 0.0, 1.0)

            save_npy_image(pred_np, os.path.join(args.output_dir, f))

            if args.gt_dir:
                gt_path = os.path.join(args.gt_dir, f)
                gt_np = load_npy_image(gt_path)
                m = metrics_calc.calculate_all(pred_np, gt_np)
                for k, v in m.items():
                    metrics_results[k].append(v)

    total_time = sum(times)
    avg_time = np.mean(times)
    print(f"\nEvaluation Completed in {total_time:.2f}s ({avg_time*1000:.2f} ms/img)")

    summary = {
        'run_id': config.get('run_id'),
        'model': config['model']['name'],
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'input_dir': os.path.abspath(args.input_dir),
        'output_dir': os.path.abspath(args.output_dir),
        'split_file': os.path.abspath(args.split_file) if args.split_file else None,
        'num_images': len(files),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'pytorch_version': torch.__version__,
        'pytorch_cuda_build': torch.version.cuda,
        'trainable_parameters': sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        'total_inference_seconds': float(total_time),
        'ms_per_image': float(avg_time * 1000),
    }

    if args.gt_dir and len(metrics_results['psnr']) > 0:
        print("\nMetrics:")
        print(f"Avg PSNR:  {np.mean(metrics_results['psnr']):.4f}")
        print(f"Avg SSIM:  {np.mean(metrics_results['ssim']):.4f}")
        print(f"Avg LPIPS: {np.mean(metrics_results['lpips']):.4f}")
        summary['metrics'] = {
            key: float(np.mean(values)) for key, values in metrics_results.items()
        }

    if args.results_file:
        results_parent = os.path.dirname(args.results_file)
        if results_parent:
            os.makedirs(results_parent, exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.results_file}")

if __name__ == "__main__":
    main()
