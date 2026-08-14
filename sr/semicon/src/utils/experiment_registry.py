"""Experiment-registry loading and structural validation."""

from pathlib import Path

import yaml


REQUIRED_EXPERIMENT_FIELDS = {
    "run_id",
    "phase",
    "experiment",
    "parent",
    "hypothesis",
    "isolated_change",
    "architecture",
    "parameters",
    "config",
    "checkpoint",
    "history",
    "output_slug",
    "validation_metrics",
    "dev_test_metrics",
    "status",
    "decision",
}


def load_experiment_registry(path: str | Path) -> dict:
    with open(path, "r") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict) or "experiments" not in registry:
        raise ValueError("Experiment registry must contain an experiments list")
    return registry


def validate_experiment_registry(registry: dict, repository_root: str | Path) -> list[str]:
    root = Path(repository_root)
    errors: list[str] = []
    experiments = registry.get("experiments", [])
    allowed_statuses = set(registry.get("allowed_statuses", []))
    run_ids = [experiment.get("run_id") for experiment in experiments]

    if len(run_ids) != len(set(run_ids)):
        errors.append("run_id values are not unique")

    checkpoint_paths: list[str] = []
    for experiment in experiments:
        missing = REQUIRED_EXPERIMENT_FIELDS - set(experiment)
        if missing:
            errors.append(f"{experiment.get('run_id', '?')} missing fields: {sorted(missing)}")
            continue

        run_id = experiment["run_id"]
        if experiment["status"] not in allowed_statuses:
            errors.append(f"{run_id} has unsupported status {experiment['status']}")
        if experiment["parent"] is not None and experiment["parent"] not in run_ids:
            errors.append(f"{run_id} references unknown parent {experiment['parent']}")
        if not (root / experiment["config"]).is_file():
            errors.append(f"{run_id} config does not exist: {experiment['config']}")
        if experiment["checkpoint"]:
            checkpoint_paths.append(experiment["checkpoint"])
        if experiment["status"] in {"PENDING_GPU", "IMPLEMENTED"} and experiment["decision"]:
            errors.append(f"{run_id} has a decision before evaluation")

    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        errors.append("checkpoint paths are not unique")
    return errors


def registry_by_run_id(registry: dict) -> dict[str, dict]:
    return {experiment["run_id"]: experiment for experiment in registry["experiments"]}
