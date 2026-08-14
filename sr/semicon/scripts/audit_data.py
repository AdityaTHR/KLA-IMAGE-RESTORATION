import os
import sys
import numpy as np
import random
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.datasets.paired_dataset import PairedDataset
from src.models.lightweight_sr import LightweightSRNet

def load_split(split_file: str) -> list:
    """Load filenames from a split text file."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    train_noisy_dir = os.path.join(base_dir, 'data', 'extracted', 'train', 'NoisyLR')
    train_gt_dir = os.path.join(base_dir, 'data', 'extracted', 'train', 'GT')

    if not os.path.exists(splits_dir):
        print(f"Splits dir {splits_dir} not found. Run prepare_data.py first.")
        sys.exit(1)

    train_files = load_split(os.path.join(splits_dir, 'train.txt'))
    val_files = load_split(os.path.join(splits_dir, 'val.txt'))
    test_files = load_split(os.path.join(splits_dir, 'dev_test.txt'))

    all_files = train_files + val_files + test_files

    failures = 0

    if len(train_files) != 2560 or len(val_files) != 320 or len(test_files) != 320:
        print("FAIL: Split sizes incorrect.")
        failures += 1
    else:
        print("PASS: Split sizes (2560, 320, 320).")

    if len(set(all_files)) != len(all_files):
        print("FAIL: Overlap found between splits.")
        failures += 1
    else:
        print("PASS: No overlap between splits.")

    expected = sorted([f for f in os.listdir(train_gt_dir) if f.endswith('.npy')])
    random.Random(42).shuffle(expected)
    expected_splits = (expected[:2560], expected[2560:2880], expected[2880:])
    if (train_files, val_files, test_files) != expected_splits:
        print("FAIL: Saved splits do not match deterministic seed-42 regeneration.")
        failures += 1
    else:
        print("PASS: Split is exactly reproducible with seed 42.")

    out_of_bounds_noisy = 0

    print("Auditing image contents... this may take a moment.")
    for f in all_files:
        noisy_path = os.path.join(train_noisy_dir, f)
        gt_path = os.path.join(train_gt_dir, f)

        if not os.path.exists(noisy_path) or not os.path.exists(gt_path):
            print(f"FAIL: Missing file pair for {f}")
            failures += 1
            continue

        noisy_img = np.load(noisy_path)
        gt_img = np.load(gt_path)

        if noisy_img.shape != (128, 128) or noisy_img.dtype != np.float32:
            print(f"FAIL: NoisyLR shape/dtype wrong for {f}")
            failures += 1

        if gt_img.shape != (256, 256) or gt_img.dtype != np.float32:
            print(f"FAIL: GT shape/dtype wrong for {f}")
            failures += 1

        if gt_img.min() < 0 or gt_img.max() > 1.0:
            print(f"FAIL: GT bounds wrong for {f}: min={gt_img.min()}, max={gt_img.max()}")
            failures += 1

        if noisy_img.min() < 0 or noisy_img.max() > 1.0:
            out_of_bounds_noisy += 1

        if np.isnan(noisy_img).any() or np.isinf(noisy_img).any():
            print(f"FAIL: NaN/Inf in NoisyLR {f}")
            failures += 1
        if np.isnan(gt_img).any() or np.isinf(gt_img).any():
            print(f"FAIL: NaN/Inf in GT {f}")
            failures += 1

    print(f"INFO: {out_of_bounds_noisy} NoisyLR images have values outside [0, 1]")

    dataset = PairedDataset(
        os.path.join(splits_dir, 'val.txt'), train_noisy_dir, train_gt_dir, augment=False
    )
    noisy_t, _ = dataset[0]
    raw_noisy = np.load(os.path.join(train_noisy_dir, val_files[0]))
    if not np.array_equal(noisy_t.squeeze(0).numpy(), raw_noisy):
        print("FAIL: Dataset loader changed/clipped NoisyLR values.")
        failures += 1
    else:
        print("PASS: Dataset loader preserves NoisyLR values exactly (no clipping).")

    model = LightweightSRNet(num_features=8, num_blocks=1).eval()
    with torch.no_grad():
        output = model(noisy_t.unsqueeze(0))
    if output.shape != (1, 1, 256, 256):
        print(f"FAIL: Model output shape is {tuple(output.shape)}.")
        failures += 1
    elif not torch.isfinite(output).all():
        print("FAIL: Model output contains NaN/Inf.")
        failures += 1
    elif float(output.min()) < 0.0 or float(output.max()) > 1.0:
        print("FAIL: Model output is outside [0,1].")
        failures += 1
    else:
        print("PASS: Model output is finite grayscale (1,1,256,256) in [0,1].")

    if failures == 0:
        print("All audit checks PASSED.")
        sys.exit(0)
    else:
        print(f"Audit checks FAILED with {failures} errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
