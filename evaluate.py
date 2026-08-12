#!/usr/bin/env python3
"""Standalone inference script: input folder of .npy LR images -> output folder of restored .npy images."""
from pathlib import Path
import argparse
import numpy as np
import torch
from tqdm import tqdm
from src.model import BaselineRestorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory containing .npy test images")
    ap.add_argument("--output", required=True, help="Directory to write restored .npy images")
    ap.add_argument("--weights", default="checkpoints/baseline_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineRestorer().to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt)
    model.eval()

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(inp.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files found in {inp}")

    with torch.inference_mode():
        for f in tqdm(files, desc=f"Inference on {device}"):
            arr = np.load(f).astype(np.float32)
            # IMPORTANT: preserve raw input range; no pre-model clipping.
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x).clamp(0.0,1.0)[0,0].cpu().numpy().astype(np.float32)
            np.save(out / f.name, pred)
    print(f"Wrote {len(files)} restored images to {out}")


if __name__ == "__main__":
    main()
