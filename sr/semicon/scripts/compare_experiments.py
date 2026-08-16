"""Build a factual comparison table from pipeline-generated result files."""

import argparse
import csv
import json
import os
import sys

REPOSITORY_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from src.utils.experiment_registry import load_experiment_registry


def project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPOSITORY_ROOT, path)


def load_json_if_present(path: str) -> dict | None:
    resolved = project_path(path)
    if not os.path.isfile(resolved):
        return None
    with open(resolved, 'r') as handle:
        return json.load(handle)


def metric_values(summary: dict | None) -> tuple[object, object, object, object]:
    if not summary or 'metrics' not in summary:
        return None, None, None, None
    metrics = summary['metrics']
    return (
        metrics.get('psnr'),
        metrics.get('ssim'),
        metrics.get('lpips'),
        summary.get('ms_per_image'),
    )


def training_time_seconds(path: str | None) -> float | None:
    if not path:
        return None
    resolved = project_path(path)
    if not os.path.isfile(resolved):
        return None
    with open(resolved, 'r', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or 'epoch_time_seconds' not in rows[0]:
        return None
    return sum(float(row['epoch_time_seconds']) for row in rows if row['epoch_time_seconds'])


def prepare_output(path: str, overwrite: bool) -> str:
    resolved = project_path(path)
    if os.path.exists(resolved) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {resolved}; pass --overwrite explicitly")
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare registered experiment artifacts")
    parser.add_argument('--registry', default='configs/experiment_registry.yaml')
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--output-csv', default=None)
    parser.add_argument('--output-log', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    registry = load_experiment_registry(project_path(args.registry))
    rows = []
    for experiment in registry['experiments']:
        validation = load_json_if_present(experiment['validation_metrics'])
        dev_test = load_json_if_present(experiment['dev_test_metrics'])
        val_psnr, val_ssim, val_lpips, val_ms = metric_values(validation)
        dev_psnr, dev_ssim, dev_lpips, dev_ms = metric_values(dev_test)
        effective_status = experiment['status']
        if validation and dev_test and effective_status not in {'KEEP', 'DROP'}:
            effective_status = 'EVALUATED'
        rows.append(
            {
                'run_id': experiment['run_id'],
                'phase': experiment['phase'],
                'experiment': experiment['experiment'],
                'parent': experiment['parent'],
                'parameters': experiment['parameters'],
                'validation_psnr': val_psnr,
                'validation_ssim': val_ssim,
                'validation_lpips': val_lpips,
                'validation_ms_per_image': val_ms,
                'dev_test_psnr': dev_psnr,
                'dev_test_ssim': dev_ssim,
                'dev_test_lpips': dev_lpips,
                'dev_test_ms_per_image': dev_ms,
                'train_time_seconds': training_time_seconds(experiment['history']),
                'status': effective_status,
                'decision': experiment['decision'],
            }
        )

    report = {
        'decision_policy': registry['decision_policy'],
        'experiments': rows,
        'note': 'No automatic KEEP/DROP decision is made by this report.',
    }
    print(json.dumps(report, indent=2))

    if args.output_json:
        output_json = prepare_output(args.output_json, args.overwrite)
        with open(output_json, 'w') as handle:
            json.dump(report, handle, indent=2)
    if args.output_csv:
        output_csv = prepare_output(args.output_csv, args.overwrite)
        with open(output_csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if args.output_log:
        output_log = prepare_output(args.output_log, args.overwrite)
        log_fields = [
            'run_id', 'phase', 'experiment', 'parent', 'config', 'parameters', 'status',
            'validation_psnr', 'validation_ssim', 'validation_lpips', 'dev_test_psnr',
            'dev_test_ssim', 'dev_test_lpips', 'inference_ms_per_image',
            'train_time_seconds', 'decision', 'notes',
        ]
        by_run_id = {row['run_id']: row for row in rows}
        log_rows = []
        for experiment in registry['experiments']:
            if int(experiment['phase']) < 2:
                continue
            row = by_run_id[experiment['run_id']]
            log_rows.append(
                {
                    'run_id': experiment['run_id'],
                    'phase': experiment['phase'],
                    'experiment': experiment['experiment'],
                    'parent': experiment['parent'],
                    'config': experiment['config'],
                    'parameters': experiment['parameters'],
                    'status': row['status'],
                    'validation_psnr': row['validation_psnr'],
                    'validation_ssim': row['validation_ssim'],
                    'validation_lpips': row['validation_lpips'],
                    'dev_test_psnr': row['dev_test_psnr'],
                    'dev_test_ssim': row['dev_test_ssim'],
                    'dev_test_lpips': row['dev_test_lpips'],
                    'inference_ms_per_image': row['dev_test_ms_per_image'],
                    'train_time_seconds': row['train_time_seconds'],
                    'decision': experiment['decision'],
                    'notes': (
                        'Generated from registry, training history, and evaluation JSON artifacts.'
                    ),
                }
            )
        with open(output_log, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)


if __name__ == '__main__':
    main()
