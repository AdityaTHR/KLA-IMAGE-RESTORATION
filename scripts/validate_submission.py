"""Validate blind-test predictions against the required deterministic NPY contract."""

import argparse
import json
import os

import numpy as np

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def validate_predictions(prediction_dir: str, input_dir: str, expected_count: int = 400) -> dict:
    prediction_files = sorted(name for name in os.listdir(prediction_dir) if name.endswith('.npy'))
    input_files = sorted(name for name in os.listdir(input_dir) if name.endswith('.npy'))
    if len(input_files) != expected_count:
        raise ValueError(f"Expected {expected_count} blind inputs, found {len(input_files)}")
    if prediction_files != input_files:
        missing = sorted(set(input_files) - set(prediction_files))
        extra = sorted(set(prediction_files) - set(input_files))
        raise ValueError(
            f"Prediction filenames do not match blind inputs; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    global_min = float('inf')
    global_max = float('-inf')
    for filename in prediction_files:
        prediction = np.load(os.path.join(prediction_dir, filename))
        if prediction.shape != (256, 256):
            raise ValueError(f"Invalid prediction shape for {filename}: {prediction.shape}")
        if prediction.dtype != np.float32:
            raise ValueError(f"Invalid prediction dtype for {filename}: {prediction.dtype}")
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite prediction values in {filename}")
        sample_min = float(prediction.min())
        sample_max = float(prediction.max())
        if sample_min < 0.0 or sample_max > 1.0:
            raise ValueError(
                f"Prediction outside [0,1] for {filename}: min={sample_min}, max={sample_max}"
            )
        global_min = min(global_min, sample_min)
        global_max = max(global_max, sample_max)

    return {
        'status': 'PASS',
        'prediction_dir': os.path.abspath(prediction_dir),
        'num_predictions': len(prediction_files),
        'shape': [256, 256],
        'dtype': 'float32',
        'global_min': global_min,
        'global_max': global_max,
        'filenames_match_blind_inputs': True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final blind-test predictions")
    parser.add_argument('--prediction-dir', required=True)
    parser.add_argument('--input-dir', default='data/extracted/test/NoisyLR')
    parser.add_argument('--expected-count', type=int, default=400)
    parser.add_argument('--results-file', default=None)
    args = parser.parse_args()

    prediction_dir = args.prediction_dir
    input_dir = args.input_dir
    if not os.path.isabs(prediction_dir):
        prediction_dir = os.path.join(REPOSITORY_ROOT, prediction_dir)
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(REPOSITORY_ROOT, input_dir)
    result = validate_predictions(prediction_dir, input_dir, args.expected_count)
    print(json.dumps(result, indent=2))
    if args.results_file:
        result_path = args.results_file
        if not os.path.isabs(result_path):
            result_path = os.path.join(REPOSITORY_ROOT, result_path)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as handle:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
