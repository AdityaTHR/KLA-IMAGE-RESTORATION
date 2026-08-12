#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Folder containing GT/ and NoisyLR/")
    args = ap.parse_args()
    root = Path(args.data_root)
    gt = {p.name: p for p in (root / "GT").glob("*.npy")}
    lr = {p.name: p for p in (root / "NoisyLR").glob("*.npy")}
    ids = sorted(set(gt) & set(lr))
    print(f"Paired samples: {len(ids)}")
    print(f"GT-only: {len(set(gt)-set(lr))} | LR-only: {len(set(lr)-set(gt))}")
    if not ids:
        raise SystemExit("No matched .npy pairs found.")

    gt_min, gt_max = float("inf"), float("-inf")
    lr_min, lr_max = float("inf"), float("-inf")
    outside, total = 0, 0
    shapes = set()
    for name in ids:
        g = np.load(gt[name])
        x = np.load(lr[name])
        shapes.add((x.shape, g.shape))
        gt_min, gt_max = min(gt_min, float(g.min())), max(gt_max, float(g.max()))
        lr_min, lr_max = min(lr_min, float(x.min())), max(lr_max, float(x.max()))
        outside += int(((x < 0) | (x > 1)).sum())
        total += x.size

    print("Shape pairs:", sorted(shapes, key=str))
    print(f"GT range: [{gt_min:.6f}, {gt_max:.6f}]")
    print(f"NoisyLR range: [{lr_min:.6f}, {lr_max:.6f}]")
    print(f"NoisyLR outside [0,1]: {100*outside/total:.3f}%")
    print("PASS: raw NoisyLR should NOT be blindly clipped before the model.")


if __name__ == "__main__":
    main()
