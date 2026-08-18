import os
import time
import yaml
import argparse
import csv
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.image_io import tensor_to_numpy
from src.losses.reconstruction import get_loss
from src.metrics.image_metrics import MetricsCalculator, calculate_psnr, calculate_ssim
from src.models import build_model
from src.datasets.paired_dataset import PairedDataset
from src.utils.training import build_optimizer, build_scheduler


def capture_rng_state() -> dict:
    """Capture all RNGs that affect shuffling, augmentation, and model execution."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: dict, device: torch.device) -> None:
    """Restore a checkpointed RNG state without requiring CUDA on CPU hosts."""
    random.setstate(rng_state['python'])
    np.random.set_state(rng_state['numpy'])
    torch.set_rng_state(rng_state['torch_cpu'].cpu())

    cuda_states = rng_state.get('torch_cuda')
    if device.type == 'cuda' and cuda_states:
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
        else:
            torch.cuda.set_rng_state(cuda_states[0].cpu(), device=device)
            print(
                "Warning: checkpoint CUDA device count differs from this host; "
                "restored the active device RNG state only."
            )


def main():
    parser = argparse.ArgumentParser(description="Train configured restoration model")
    parser.add_argument('--config', required=True, type=str, help='Path to config yaml')
    parser.add_argument(
        '--resume', type=str, default=None, help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--no-lpips',
        action='store_true',
        help='Skip LPIPS during validation (faster on CPU)',
    )
    args = parser.parse_args()
    skip_lpips = args.no_lpips

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")

    set_seed(config['training'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU VRAM: {properties.total_memory / (1024 ** 3):.2f} GiB")

    train_dataset = PairedDataset(
        split_file=config['data']['train_split'],
        noisy_dir=config['data']['noisy_dir'],
        gt_dir=config['data']['gt_dir'],
        augment=config['training'].get('augment', True),
        synthetic_degradation=config['training'].get('synthetic_degradation'),
    )
    val_dataset = PairedDataset(
        split_file=config['data']['val_split'],
        noisy_dir=config['data']['noisy_dir'],
        gt_dir=config['data']['gt_dir'],
        augment=False
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config['training'].get('seed', 42))
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training']['num_workers'],
                              generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = build_model(config['model']).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameter_count:,}")
    if hasattr(model, 'architecture_summary'):
        print(f"Architecture: {json.dumps(model.architecture_summary(), sort_keys=True)}")

    optimizer = build_optimizer(model.parameters(), config['training']['optimizer'])
    scheduler = build_scheduler(optimizer, config['training']['scheduler'])

    loss_config = config['training']['loss'].copy()
    loss_name = loss_config.pop('name')
    criterion = get_loss(loss_name, **loss_config).to(device)

    start_epoch = 0
    best_psnr = float('-inf')
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = state.get('epoch', 0)
        best_psnr = state.get('best_psnr', state.get('val_psnr', best_psnr))
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        elif start_epoch > 0:
            raise ValueError(
                "This legacy checkpoint lacks scheduler state and cannot be resumed exactly. "
                "Start a fresh run or resume from a checkpoint produced by the current train.py."
            )
        if 'train_generator_state' in state:
            train_generator.set_state(state['train_generator_state'].cpu())
        print(f"Resumed from epoch {start_epoch}")

    use_amp = config['training'].get('amp', False) and torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)
    if args.resume and 'state' in locals() and 'scaler_state_dict' in state:
        scaler.load_state_dict(state['scaler_state_dict'])
    metrics_calc = MetricsCalculator(device=device) if not skip_lpips else None
    if args.resume and 'rng_state' in state:
        restore_rng_state(state['rng_state'], device)
    elif args.resume:
        print("Warning: checkpoint has no complete RNG state; resume is not bit-exact.")

    checkpoint_dir = config['checkpoint']['dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(checkpoint_dir, f"{config['run_id']}_training_history.csv")
    config_snapshot_path = os.path.join(checkpoint_dir, f"{config['run_id']}_config.json")
    with open(config_snapshot_path, 'w') as f:
        json.dump(config, f, indent=2)

    epochs = config['training']['epochs']

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_index, (noisy, gt) in enumerate(pbar):
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                pred = model(noisy)
                loss = criterion(pred, gt)

            if tuple(pred.shape) != tuple(gt.shape):
                raise ValueError(
                    f"Training output/target shape mismatch at epoch {epoch + 1}, "
                    f"batch {batch_index}: {tuple(pred.shape)} vs {tuple(gt.shape)}"
                )
            if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training prediction/loss at epoch {epoch + 1}, "
                    f"batch {batch_index}"
                )

            scaler.scale(loss).backward()
            if config['training'].get('gradient_clip'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['training']['gradient_clip']
                )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_metrics = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}

        with torch.no_grad():
            for noisy, gt in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy)
                loss = criterion(pred, gt)
                if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation prediction/loss at epoch {epoch + 1}"
                    )
                val_loss += loss.item()

                pred_np = np.clip(tensor_to_numpy(pred), 0.0, 1.0)
                gt_np = tensor_to_numpy(gt)

                val_metrics['psnr'] += calculate_psnr(pred_np, gt_np)
                val_metrics['ssim'] += calculate_ssim(pred_np, gt_np)
                if metrics_calc is not None:
                    val_metrics['lpips'] += metrics_calc.lpips_metric.calculate(pred_np, gt_np)

        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)

        scheduler.step()
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        lpips_text = "skipped" if skip_lpips else f"{val_metrics['lpips']:.4f}"
        print(
            f"Val PSNR: {val_metrics['psnr']:.2f} | "
            f"Val SSIM: {val_metrics['ssim']:.4f} | Val LPIPS: {lpips_text}"
        )

        is_best = val_metrics['psnr'] > best_psnr
        if is_best:
            best_psnr = val_metrics['psnr']

        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_generator_state': train_generator.get_state(),
            'rng_state': capture_rng_state(),
            'val_psnr': val_metrics['psnr'],
            'best_psnr': best_psnr,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_ssim': val_metrics['ssim'],
            'val_lpips': None if skip_lpips else val_metrics['lpips'],
            'learning_rate': scheduler.get_last_lr()[0],
            'epoch_time_seconds': epoch_time,
            'config': config,
            'hardware': {
                'device': str(device),
                'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'pytorch_version': torch.__version__,
                'pytorch_cuda_build': torch.version.cuda,
            },
        }

        save_path = os.path.join(
            checkpoint_dir,
            f"{config['checkpoint']['filename']}_epoch_{epoch+1}.pth",
        )

        if (epoch + 1) % config['checkpoint'].get('save_every', 10) == 0:
            save_checkpoint(checkpoint_state, save_path)

        if is_best:
            best_filename = config['checkpoint'].get('best_filename', 'baseline_cnn_best.pth')
            save_checkpoint(checkpoint_state, os.path.join(checkpoint_dir, best_filename))

        history_row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'psnr': val_metrics['psnr'],
            'ssim': val_metrics['ssim'],
            'lpips': '' if skip_lpips else val_metrics['lpips'],
            'learning_rate': scheduler.get_last_lr()[0],
            'epoch_time_seconds': epoch_time,
        }
        write_header = not os.path.exists(history_path) or (start_epoch == 0 and epoch == 0)
        mode = 'w' if write_header else 'a'
        with open(history_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(history_row)

    print(f"Training completed. Best PSNR: {best_psnr:.2f}")

if __name__ == "__main__":
    main()
