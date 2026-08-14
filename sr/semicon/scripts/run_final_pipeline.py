"""Evaluate and package an explicitly selected final candidate."""

import argparse
import json
import os
import subprocess
import sys

import torch
import yaml

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from scripts.validate_submission import validate_predictions
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.experiment_registry import load_experiment_registry, registry_by_run_id


def project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPOSITORY_ROOT, path)


def run_evaluation(arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, os.path.join(REPOSITORY_ROOT, 'evaluate.py'), *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the explicitly selected final candidate")
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--registry', default='configs/experiment_registry.yaml')
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--confirm-selected', action='store_true')
    args = parser.parse_args()

    if not args.confirm_selected:
        raise ValueError(
            "Final pipeline requires --confirm-selected after human quality/efficiency review"
        )

    registry = load_experiment_registry(project_path(args.registry))
    experiments = registry_by_run_id(registry)
    if args.run_id not in experiments:
        raise KeyError(f"Unknown registered run_id: {args.run_id}")
    experiment = experiments[args.run_id]
    if experiment['parameters'] == 0:
        raise ValueError(
            "A non-learned experiment cannot be used as the final checkpoint pipeline"
        )

    for required_metrics in ('validation_metrics', 'dev_test_metrics'):
        metric_path = project_path(experiment[required_metrics])
        if not os.path.isfile(metric_path):
            raise FileNotFoundError(
                f"Selection is premature; required completed metric file is missing: {metric_path}"
            )

    config_path = project_path(experiment['config'])
    checkpoint_path = project_path(args.checkpoint or experiment['checkpoint'])
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint_path}")
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)
    if config['run_id'] != args.run_id:
        raise ValueError("Selected config run_id does not match --run-id")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model']).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != experiment['parameters']:
        raise ValueError(
            f"Selected model parameter mismatch: expected {experiment['parameters']}, "
            f"got {parameter_count}"
        )

    output_root = args.output_root or f"outputs/final/{experiment['output_slug']}"
    output_root = project_path(output_root)
    os.makedirs(output_root, exist_ok=True)
    validation_dir = os.path.join(output_root, 'val')
    dev_test_dir = os.path.join(output_root, 'dev_test')
    blind_dir = os.path.join(output_root, 'blind_test')
    validation_results = os.path.join(output_root, 'validation_metrics.json')
    dev_test_results = os.path.join(output_root, 'dev_test_metrics.json')
    blind_results = os.path.join(output_root, 'blind_inference.json')

    common = ['--config', config_path, '--checkpoint', checkpoint_path]
    run_evaluation(
        [
            *common,
            '--input_dir', project_path(config['data']['noisy_dir']),
            '--output_dir', validation_dir,
            '--gt_dir', project_path(config['data']['gt_dir']),
            '--split_file', project_path(config['data']['val_split']),
            '--results_file', validation_results,
        ]
    )
    run_evaluation(
        [
            *common,
            '--input_dir', project_path(config['data']['noisy_dir']),
            '--output_dir', dev_test_dir,
            '--gt_dir', project_path(config['data']['gt_dir']),
            '--split_file', project_path(config['data']['dev_test_split']),
            '--results_file', dev_test_results,
        ]
    )
    run_evaluation(
        [
            *common,
            '--input_dir', project_path(config['data']['test_dir']),
            '--output_dir', blind_dir,
            '--results_file', blind_results,
        ]
    )

    submission_validation = validate_predictions(
        blind_dir,
        project_path(config['data']['test_dir']),
        expected_count=400,
    )
    with open(validation_results, 'r') as handle:
        validation_summary = json.load(handle)
    with open(dev_test_results, 'r') as handle:
        dev_test_summary = json.load(handle)
    with open(blind_results, 'r') as handle:
        blind_summary = json.load(handle)

    report = {
        'run_id': args.run_id,
        'config': os.path.abspath(config_path),
        'checkpoint': os.path.abspath(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'parameters': parameter_count,
        'selection_confirmation': 'human-confirmed before pipeline execution',
        'validation': validation_summary,
        'dev_test': dev_test_summary,
        'blind_test': blind_summary,
        'submission_validation': submission_validation,
        'note': (
            'This pipeline records executed artifacts; it does not fabricate or optimize metrics.'
        ),
    }
    report_path = os.path.join(output_root, 'final_pipeline_report.json')
    with open(report_path, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Final pipeline report saved to {report_path}")


if __name__ == '__main__':
    main()
