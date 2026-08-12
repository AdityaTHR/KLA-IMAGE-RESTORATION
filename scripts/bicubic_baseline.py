#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.metrics import psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    args = ap.parse_args()
    root = Path(args.data_root)
    ids = sorted(p.name for p in (root / "GT").glob("*.npy"))
    ids = [x for x in ids if (root / "NoisyLR" / x).exists()]
    if args.limit:
        ids = ids[:args.limit]
    P, S = [], []
    for name in tqdm(ids, desc="Bicubic baseline"):
        gt = np.load(root / "GT" / name).astype(np.float32)
        lr = np.load(root / "NoisyLR" / name).astype(np.float32)
        pred = cv2.resize(lr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        pred = np.clip(pred, 0.0, 1.0)
        P.append(psnr(gt, pred))
        S.append(ssim(gt, pred))
    print(f"Samples: {len(ids)}")
    print(f"PSNR: {np.mean(P):.4f} dB")
    print(f"SSIM: {np.mean(S):.4f}")


if __name__ == "__main__":
    main()
