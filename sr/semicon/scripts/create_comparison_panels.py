"""Create deterministic qualitative panels from already-generated predictions."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPOSITORY_ROOT, path)


def parse_prediction(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must use LABEL=DIRECTORY syntax")
    label, directory = value.split("=", 1)
    if not label.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("prediction label and directory must be nonempty")
    return label.strip(), project_path(directory.strip())


def load_checked(path: str, expected_shape: tuple[int, int], role: str) -> np.ndarray:
    image = np.load(path)
    if image.shape != expected_shape or image.dtype != np.float32:
        raise ValueError(
            f"{role} contract failure at {path}: expected {expected_shape}/float32, "
            f"got {image.shape}/{image.dtype}"
        )
    if not np.isfinite(image).all():
        raise FloatingPointError(f"{role} contains NaN or Inf: {path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create side-by-side panels without recomputing or modifying predictions"
    )
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument(
        '--prediction',
        action='append',
        type=parse_prediction,
        required=True,
        help='Repeat LABEL=DIRECTORY for each prediction source',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--filenames', nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")

    split_file = project_path(args.split_file)
    input_dir = project_path(args.input_dir)
    gt_dir = project_path(args.gt_dir)
    output_dir = project_path(args.output_dir)
    with open(split_file, 'r') as handle:
        split_filenames = [line.strip() for line in handle if line.strip()]
    if len(split_filenames) != len(set(split_filenames)):
        raise ValueError("split file contains duplicate filenames")

    if args.filenames:
        unknown = sorted(set(args.filenames) - set(split_filenames))
        if unknown:
            raise ValueError(f"Requested filenames are not in the split: {unknown[:3]}")
        filenames = args.filenames
    else:
        filenames = split_filenames[:args.count]

    os.makedirs(output_dir, exist_ok=True)
    for filename in filenames:
        noisy = load_checked(os.path.join(input_dir, filename), (128, 128), 'NoisyLR')
        gt = load_checked(os.path.join(gt_dir, filename), (256, 256), 'GT')
        if float(gt.min()) < 0.0 or float(gt.max()) > 1.0:
            raise ValueError(f"GT outside [0,1]: {filename}")

        images = [("NoisyLR (display [0,1])", noisy), ("GT", gt)]
        for label, prediction_dir in args.prediction:
            prediction = load_checked(
                os.path.join(prediction_dir, filename), (256, 256), f'{label} prediction'
            )
            if float(prediction.min()) < 0.0 or float(prediction.max()) > 1.0:
                raise ValueError(f"{label} prediction outside [0,1]: {filename}")
            images.append((label, prediction))

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        if os.path.exists(output_path) and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite comparison panel: {output_path}; "
                "use --overwrite explicitly"
            )
        figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
        for axis, (label, image) in zip(axes[0], images):
            axis.imshow(image, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
            axis.set_title(label)
            axis.axis('off')
        figure.suptitle(
            f"{filename} | raw NoisyLR min={float(noisy.min()):.4f}, max={float(noisy.max()):.4f}"
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(figure)
    print(f"Created {len(filenames)} comparison panels in {output_dir}")


if __name__ == '__main__':
    main()
