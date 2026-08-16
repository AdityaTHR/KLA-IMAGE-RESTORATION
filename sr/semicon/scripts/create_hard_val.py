"""Create a deterministic hard-validation subset without changing canonical splits."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def degradation_score(noisy: np.ndarray, gt: np.ndarray) -> float:
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
    reference = F.interpolate(
        gt_tensor,
        size=(128, 128),
        mode='bicubic',
        align_corners=False,
        antialias=True,
    ).squeeze().numpy()
    residual_rms = float(np.sqrt(np.mean((noisy - reference) ** 2)))
    out_of_range_fraction = float(np.mean((noisy < 0.0) | (noisy > 1.0)))
    return residual_rms + 0.25 * out_of_range_fraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank canonical validation samples by a deterministic degradation-severity proxy"
        )
    )
    parser.add_argument('--count', type=int, default=80)
    parser.add_argument('--output', default='data/derived_splits/hard_val.txt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.count <= 320:
        raise ValueError("count must lie between 1 and 320")

    val_path = os.path.join(REPOSITORY_ROOT, 'data', 'splits', 'val.txt')
    noisy_dir = os.path.join(REPOSITORY_ROOT, 'data', 'extracted', 'train', 'NoisyLR')
    gt_dir = os.path.join(REPOSITORY_ROOT, 'data', 'extracted', 'train', 'GT')
    with open(val_path, 'r') as handle:
        filenames = [line.strip() for line in handle if line.strip()]
    if len(filenames) != 320:
        raise ValueError(
            f"Canonical validation split must contain 320 entries, found {len(filenames)}"
        )

    ranked = []
    for filename in filenames:
        noisy = np.load(os.path.join(noisy_dir, filename))
        gt = np.load(os.path.join(gt_dir, filename))
        ranked.append((degradation_score(noisy, gt), filename))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [filename for _, filename in ranked[:args.count]]

    summary = {
        'method': 'validation-only degradation-severity proxy; not a source-family OOD label',
        'canonical_val_count': len(filenames),
        'hard_val_count': len(selected),
        'top_score': ranked[0][0],
        'lowest_selected_score': ranked[args.count - 1][0],
        'filenames': selected,
    }
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(REPOSITORY_ROOT, output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, 'r') as handle:
                existing = [line.strip() for line in handle if line.strip()]
            if existing != selected:
                raise FileExistsError(
                    f"Refusing to overwrite a different hard-validation manifest: {output_path}"
                )
            print(f"Existing hard-validation manifest already matches: {output_path}")
        else:
            with open(output_path, 'x') as handle:
                handle.write('\n'.join(selected))
            print(f"Created derived hard-validation manifest: {output_path}")


if __name__ == '__main__':
    main()
