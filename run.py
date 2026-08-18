#!/usr/bin/env python3

import argparse
import os
import sys
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.models import build_model
from src.utils.checkpoint import load_checkpoint


CONFIG_PATH = os.path.join(HERE, "models", "config.yaml")
CHECKPOINT_PATH = os.path.join(HERE, "models", "final_model.pth")

BETA = 0.04
BATCH_SIZE = 16


# ---------------------------------------------------------
# Gaussian low-pass used for measurement-derived HF detail
# sigma = 1, radius = 4  -> 9x9 kernel
# ---------------------------------------------------------
def gaussian_kernel(device, dtype, sigma=1.0, radius=4):
    x = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=dtype,
    )

    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel = g[:, None] * g[None, :]
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])

    return kernel


def measured_detail(x):
    """
    x: raw degraded LR tensor [B,1,H,W]

    Important:
    Do NOT clip the input.
    Out-of-range NoisyLR values are intentional.
    """

    bicubic = F.interpolate(
        x,
        scale_factor=2,
        mode="bicubic",
        align_corners=False,
    )

    kernel = gaussian_kernel(
        bicubic.device,
        bicubic.dtype,
        sigma=1.0,
        radius=4,
    )

    # Reflect padding approximates standard Gaussian filtering
    padded = F.pad(
        bicubic,
        (4, 4, 4, 4),
        mode="reflect",
    )

    low = F.conv2d(
        padded,
        kernel,
    )

    return bicubic - low


def load_model(device):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg["model"]).to(device)

    load_checkpoint(
        CHECKPOINT_PATH,
        model,
        device=device,
    )

    model.eval()

    return model


def read_npy(path):
    arr = np.load(path).astype(np.float32)

    # Accept HxW
    if arr.ndim == 2:
        pass

    # Accept HxWx1
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # Accept 1xHxW
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    else:
        raise ValueError(
            f"Unsupported input shape {arr.shape} for {path}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            f"Input contains NaN/Inf: {path}"
        )

    return arr


def process_batch(model, arrays, device):
    x = np.stack(arrays, axis=0)

    x = torch.from_numpy(x)
    x = x.unsqueeze(1)
    x = x.to(device, non_blocking=True)

    with torch.inference_mode():
        restored = model(x)

        # Measurement-supported HF residual.
        detail = measured_detail(x)

        restored = restored + BETA * detail

        # Required submission range.
        restored = torch.nan_to_num(
            restored,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        restored = restored.clamp(0.0, 1.0)

    restored = restored[:, 0]
    restored = restored.cpu().numpy().astype(np.float32)

    return restored


def main():
    parser = argparse.ArgumentParser(
        description="Noise? IC None - KLA Image Restoration"
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing degraded .npy files",
    )

    parser.add_argument(
        "output_dir",
        help="Directory where restored .npy files will be written",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(
            f"Input directory does not exist: {args.input_dir}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    files = sorted(
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".npy")
    )

    if not files:
        raise RuntimeError(
            f"No .npy files found in {args.input_dir}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(f"Found {len(files)} input files.")
    print(f"Detail fusion beta: {BETA}")

    model = load_model(device)

    # Warm-up
    first = read_npy(
        os.path.join(args.input_dir, files[0])
    )

    warm = torch.from_numpy(first)[None, None].to(device)

    with torch.inference_mode():
        _ = model(warm)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for start_idx in range(
        0,
        len(files),
        BATCH_SIZE,
    ):

        batch_files = files[
            start_idx:start_idx + BATCH_SIZE
        ]

        arrays = [
            read_npy(
                os.path.join(args.input_dir, name)
            )
            for name in batch_files
        ]

        predictions = process_batch(
            model,
            arrays,
            device,
        )

        for name, pred in zip(
            batch_files,
            predictions,
        ):
            np.save(
                os.path.join(
                    args.output_dir,
                    name,
                ),
                pred,
            )

        done = min(
            start_idx + BATCH_SIZE,
            len(files),
        )

        print(
            f"\rProcessed {done}/{len(files)}",
            end="",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print()
    print("Restoration complete.")
    print(
        f"Total runtime: {elapsed:.2f}s | "
        f"{1000 * elapsed / len(files):.2f} ms/image"
    )


if __name__ == "__main__":
    main()
