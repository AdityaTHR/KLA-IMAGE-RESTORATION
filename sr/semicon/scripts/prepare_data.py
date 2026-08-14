import os
import sys
import zipfile
import random
import argparse
import numpy as np
import shutil
import tempfile
from pathlib import Path

def extract_zip(zip_path: str, extract_to: str) -> None:
    """Safely extract a ZIP, rejecting members that escape the destination."""
    print(f"Extracting {zip_path} to {extract_to}...")
    destination = Path(extract_to).resolve()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for info in zip_ref.infolist():
            member = info.filename
            if "__MACOSX" in member or ".DS_Store" in member or info.is_dir():
                continue
            target = (destination / member).resolve()
            if destination not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(info) as source, open(target, 'wb') as output:
                shutil.copyfileobj(source, output)

def discover_by_parent(root: str, parent_name: str) -> list[Path]:
    """Find NPY files below a directory component, independent of archive layout."""
    parent_name = parent_name.lower()
    return sorted([
        path for path in Path(root).rglob('*.npy')
        if parent_name in {part.lower() for part in path.parent.parts}
    ])


def copy_unique(files: list[Path], destination: str) -> None:
    """Copy files by basename and reject ambiguous duplicate names."""
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate .npy basenames discovered for {destination}")
    os.makedirs(destination, exist_ok=True)
    for source in files:
        shutil.copy2(source, os.path.join(destination, source.name))

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for SEMICOM project")
    parser.add_argument('--seed', type=int, default=42, help="Seed for split generation")
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Canonical SemiCon split generation requires seed 42")

    random.seed(args.seed)

    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    raw_archives_dir = os.path.join(base_dir, 'data', 'raw_archives')

    train_zip = os.path.join(raw_archives_dir, 'train.zip')
    test_zip = os.path.join(raw_archives_dir, 'Test_NoisyLR.zip')

    extract_dir = os.path.join(base_dir, 'data', 'extracted')
    splits_dir = os.path.join(base_dir, 'data', 'splits')

    if not os.path.exists(train_zip) or not os.path.exists(test_zip):
        print(f"Error: Missing zip files in {raw_archives_dir}")
        sys.exit(1)

    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    train_gt_dir = os.path.join(extract_dir, 'train', 'GT')
    train_noisy_dir = os.path.join(extract_dir, 'train', 'NoisyLR')
    test_noisy_dir = os.path.join(extract_dir, 'test', 'NoisyLR')

    existing_counts = [
        len(list(Path(path).glob('*.npy'))) if os.path.isdir(path) else 0
        for path in (train_gt_dir, train_noisy_dir, test_noisy_dir)
    ]
    if existing_counts == [3200, 3200, 400]:
        print("Prepared data already exists; verifying it without re-extracting archives.")
    else:
        with tempfile.TemporaryDirectory(prefix='prepare_', dir=extract_dir) as staging:
            train_stage = os.path.join(staging, 'train_archive')
            test_stage = os.path.join(staging, 'test_archive')
            extract_zip(train_zip, train_stage)
            extract_zip(test_zip, test_stage)

            train_gt = discover_by_parent(train_stage, 'GT')
            train_noisy = discover_by_parent(train_stage, 'NoisyLR')
            test_noisy = discover_by_parent(test_stage, 'NoisyLR')
            if not test_noisy:
                test_noisy = sorted(Path(test_stage).rglob('*.npy'))

            copy_unique(train_gt, train_gt_dir)
            copy_unique(train_noisy, train_noisy_dir)
            copy_unique(test_noisy, test_noisy_dir)

    gt_files = sorted([f for f in os.listdir(train_gt_dir) if f.endswith('.npy')])
    noisy_files = sorted([f for f in os.listdir(train_noisy_dir) if f.endswith('.npy')])
    test_files = sorted([f for f in os.listdir(test_noisy_dir) if f.endswith('.npy')])

    assert len(gt_files) == 3200, f"Expected 3200 GT files, got {len(gt_files)}"
    assert len(noisy_files) == 3200, f"Expected 3200 NoisyLR files, got {len(noisy_files)}"
    assert gt_files == noisy_files, "GT and NoisyLR filenames do not match!"
    assert len(test_files) == 400, f"Expected 400 Test files, got {len(test_files)}"

    print("Files verified successfully.")

    sample_gt = np.load(os.path.join(train_gt_dir, gt_files[0]))
    sample_noisy = np.load(os.path.join(train_noisy_dir, noisy_files[0]))

    print(f"Sample GT shape: {sample_gt.shape}, dtype: {sample_gt.dtype}")
    print(f"Sample NoisyLR shape: {sample_noisy.shape}, dtype: {sample_noisy.dtype}")

    random.shuffle(gt_files)

    train_split = gt_files[:2560]
    val_split = gt_files[2560:2880]
    test_split = gt_files[2880:]

    expected_splits = {
        'train.txt': train_split,
        'val.txt': val_split,
        'dev_test.txt': test_split,
    }
    for split_name, expected_filenames in expected_splits.items():
        split_path = os.path.join(splits_dir, split_name)
        if os.path.exists(split_path):
            with open(split_path, 'r') as handle:
                current_filenames = [line.strip() for line in handle if line.strip()]
            if current_filenames != expected_filenames:
                raise ValueError(
                    f"Existing frozen split does not match seed-42 canonical content: {split_path}"
                )
            print(f"Preserved verified frozen split: {split_path}")
        else:
            with open(split_path, 'x') as handle:
                handle.write('\n'.join(expected_filenames))
            print(f"Created missing canonical split: {split_path}")

    print(
        f"Created splits: Train ({len(train_split)}), Val ({len(val_split)}), "
        f"Dev Test ({len(test_split)})"
    )

if __name__ == "__main__":
    main()
