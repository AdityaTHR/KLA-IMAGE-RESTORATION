#!/usr/bin/env bash
set -euo pipefail

required_files=(
  README.md
  requirements.txt
  train.py
  evaluate.py
  configs/baseline_nafnet.yaml
  configs/experiment_registry.yaml
  scripts/audit_data.py
  scripts/cuda_smoke_test.py
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $path is missing. Run this script from the semicon project root." >&2
    exit 1
  fi
done

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is not available in this Git Bash session." >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: no Python virtual environment is active." >&2
  echo "Create one with: python -m venv .venv" >&2
  echo "Activate it with: source .venv/Scripts/activate" >&2
  exit 1
fi

python - <<'PY'
import os
import sys

import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available() or torch.version.cuda is None:
    raise SystemExit("ERROR: CUDA-enabled PyTorch and a visible NVIDIA GPU are required.")
print("GPU:", torch.cuda.get_device_name(0))

required_dirs = [
    "data/extracted/train/NoisyLR",
    "data/extracted/train/GT",
    "data/extracted/test/NoisyLR",
    "data/splits",
]
missing = [path for path in required_dirs if not os.path.isdir(path)]
if missing:
    raise SystemExit("ERROR: missing dataset directories: " + ", ".join(missing))

expected_splits = {"train.txt": 2560, "val.txt": 320, "dev_test.txt": 320}
for name, expected in expected_splits.items():
    path = os.path.join("data", "splits", name)
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: missing split manifest: {path}")
    with open(path, "r") as handle:
        count = sum(1 for line in handle if line.strip())
    if count != expected:
        raise SystemExit(f"ERROR: {path} has {count} entries; expected {expected}")
print("Dataset structure and canonical split counts: PASS")
PY

echo "Setup verification: PASS"
echo "Next command: python scripts/audit_data.py"
echo "Then run: python scripts/cuda_smoke_test.py --config configs/baseline_nafnet.yaml"
