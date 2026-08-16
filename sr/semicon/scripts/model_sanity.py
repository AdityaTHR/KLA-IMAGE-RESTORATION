"""Architecture contract, gradient, size, and inference sanity checks."""

import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.losses.reconstruction import get_loss
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    model: torch.nn.Module,
    benchmark_input: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(benchmark_input)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(benchmark_input)
        synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run restoration model architecture checks")
    parser.add_argument("--config", required=True, help="Model/training YAML configuration")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed inference iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Timed inference iterations")
    parser.add_argument("--results-file", default=None, help="Optional JSON results path")
    parser.add_argument("--comparison-config", default=None, help="Optional comparison model YAML")
    parser.add_argument(
        "--comparison-checkpoint", default=None, help="Optional comparison checkpoint"
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    seed = config["training"].get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)

    loss_config = config["training"]["loss"].copy()
    loss_name = loss_config.pop("name")
    criterion = get_loss(loss_name, **loss_config).to(device)

    # Include the audited degraded-value range explicitly. Nothing may clamp it.
    degraded = torch.rand(2, 1, 128, 128, device=device, dtype=torch.float32) * 1.8 - 0.08
    degraded[0, 0, 0, 0] = -0.08
    degraded[0, 0, 0, 1] = 1.72
    degraded_before = degraded.detach().clone()
    target = torch.rand(2, 1, 256, 256, device=device, dtype=torch.float32)

    model.train()
    prediction = model(degraded)
    if prediction.shape != (2, 1, 256, 256):
        raise AssertionError(f"Unexpected output shape: {tuple(prediction.shape)}")
    if prediction.dtype != torch.float32:
        raise AssertionError(f"Unexpected output dtype: {prediction.dtype}")
    if not torch.isfinite(prediction).all():
        raise AssertionError("Prediction contains NaN or Inf")
    detached_prediction = prediction.detach()
    if float(detached_prediction.min()) < 0.0 or float(detached_prediction.max()) > 1.0:
        raise AssertionError("Prediction lies outside [0,1]")
    if not torch.equal(degraded, degraded_before):
        raise AssertionError("The model modified the degraded input tensor")

    loss = criterion(prediction, target)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    trainable_tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    if len(gradients) != trainable_tensors:
        raise AssertionError("At least one trainable parameter did not receive a gradient")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("A model gradient contains NaN or Inf")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    benchmark_input = degraded[:1]
    ms_per_image = benchmark_model(
        model, benchmark_input, device, args.warmup, args.iterations
    )

    hardware = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_vram_bytes": properties.total_memory,
            }
        )

    architecture = (
        model.architecture_summary()
        if hasattr(model, "architecture_summary")
        else {"class": type(model).__name__}
    )
    results = {
        "run_id": config.get("run_id"),
        "hardware": hardware,
        "architecture": architecture,
        "trainable_parameters": parameter_count,
        "parameter_memory_bytes": parameter_bytes,
        "parameter_memory_mib": parameter_bytes / (1024 ** 2),
        "contract": {
            "input_shape": list(degraded.shape),
            "input_dtype": str(degraded.dtype),
            "input_min": float(degraded.min()),
            "input_max": float(degraded.max()),
            "input_preserved": True,
            "output_shape": list(prediction.shape),
            "output_dtype": str(prediction.dtype),
            "output_min": float(detached_prediction.min()),
            "output_max": float(detached_prediction.max()),
            "output_finite": True,
            "gradients_finite": True,
            "synthetic_charbonnier_loss": float(loss.detach()),
        },
        "benchmark": {
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "batch_size": 1,
            "ms_per_image": ms_per_image,
        },
    }

    if args.comparison_config:
        with open(args.comparison_config, "r") as handle:
            comparison_config = yaml.safe_load(handle)
        comparison_model = build_model(comparison_config["model"]).to(device)
        if args.comparison_checkpoint:
            load_checkpoint(args.comparison_checkpoint, comparison_model, device=device)
        comparison_parameters = sum(
            parameter.numel()
            for parameter in comparison_model.parameters()
            if parameter.requires_grad
        )
        comparison_ms = benchmark_model(
            comparison_model, benchmark_input, device, args.warmup, args.iterations
        )
        results["comparison"] = {
            "run_id": comparison_config.get("run_id"),
            "checkpoint": args.comparison_checkpoint,
            "trainable_parameters": comparison_parameters,
            "ms_per_image": comparison_ms,
            "primary_to_comparison_latency_ratio": ms_per_image / comparison_ms,
        }
    serialized = json.dumps(results, indent=2)
    print(serialized)
    if args.results_file:
        parent = os.path.dirname(args.results_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.results_file, "w") as handle:
            handle.write(serialized + "\n")
        print(f"Saved results to {args.results_file}")


if __name__ == "__main__":
    main()
