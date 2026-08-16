# KLA PS01 - AI-Based Restoration of Degraded Images

## Current Status

**CODE_COMPLETE_GPU_EXPERIMENTS_PENDING**

The reproducible bicubic and lightweight-CNN baselines are complete, verified, and permanently frozen. The planned B2, degradation-aware, synthetic-degradation, and isolated loss-ablation implementations are code-complete. Their training, evaluation, comparison, and final selection remain pending on an NVIDIA/CUDA machine; no GPU result is claimed here.

## 1. Problem

Restore a degraded 128x128 float32 grayscale semiconductor-inspection image to a clean 256x256 float32 grayscale image. The system must jointly denoise and perform 2x super-resolution while preserving small structures and avoiding unsupported detail.

## 2. Dataset

| Split | NoisyLR | GT | Count |
|---|---|---|---:|
| Train | 128x128 float32 | 256x256 float32 | 2,560 |
| Validation | 128x128 float32 | 256x256 float32 | 320 |
| Development test | 128x128 float32 | 256x256 float32 | 320 |
| Blind test | 128x128 float32 | unavailable | 400 |

The seed-42 split is deterministic and has no overlap. Blind-test images are never used for training or model selection. Original archives are preserved under `data/raw_archives/`; extracted arrays are under `data/extracted/`.

The archives and extracted arrays are too large for ordinary Git. Use shared storage, DVC, or Git LFS as agreed by the team. `.gitignore` prevents accidental normal-Git commits of these data while keeping split manifests versionable.

## 3. Repository Structure

```text
semicon/
|-- train.py
|-- evaluate.py
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- baseline_bicubic.yaml
|   |-- baseline_cnn.yaml
|   |-- baseline_nafnet.yaml
|   |-- phase3_p1_degradation_aware.yaml
|   |-- phase4_p2_synthetic_degradation.yaml
|   |-- phase5_p3_edge_loss.yaml
|   |-- phase5_p3_ssim_loss.yaml
|   |-- phase5_p3_frequency_loss.yaml
|   `-- experiment_registry.yaml
|-- src/
|   |-- datasets/
|   |-- losses/
|   |-- metrics/
|   |-- models/
|   `-- utils/
|-- scripts/
|   |-- prepare_data.py
|   |-- audit_data.py
|   |-- benchmark.py
|   |-- evaluate_checkpoint.py
|   |-- model_sanity.py
|   |-- cuda_smoke_test.py
|   |-- validate_experiments.py
|   |-- compare_experiments.py
|   |-- create_comparison_panels.py
|   |-- create_hard_val.py
|   |-- benchmark_model.py
|   |-- validate_submission.py
|   |-- run_final_pipeline.py
|   `-- setup_nvidia_gitbash.sh
|-- data/
|   |-- raw_archives/
|   |-- extracted/
|   `-- splits/
|-- checkpoints/
|   |-- baseline_cnn_best.pth
|   |-- B1_LIGHTWEIGHT_CNN_config.json
|   |-- B1_LIGHTWEIGHT_CNN_training_history.csv
|   `-- archive/
|-- outputs/
|   |-- bicubic/
|   |-- baseline_cnn/
|   `-- *_metrics.json
`-- docs/
    |-- dataset_audit.md
    |-- experiment_log.csv
    |-- phase2_plus_experiment_log.csv
    `-- gpu_handoff.md
```

## 4. Phase 1 Baselines

- **B0 Bicubic:** PyTorch bicubic interpolation, scale factor 2, `align_corners=False`, no learning.
- **B1 Lightweight CNN:** 120,833 parameters, one grayscale input/output channel, 32 feature channels, four residual blocks, pixel-shuffle upsampling, and sigmoid output. Training used Charbonnier loss, Adam, flip/90-degree rotation augmentation, seed 42, and three CPU epochs.

## 5. Phase 1 Results

| Baseline | Split | PSNR (higher) | SSIM (higher) | LPIPS (lower) | CPU ms/image (lower) |
|---|---|---:|---:|---:|---:|
| Bicubic | Validation | 23.1308 | 0.5413 | 0.4400 | 2.88 |
| Lightweight CNN | Validation | 27.1655 | 0.7305 | 0.3280 | 48.52 |
| Bicubic | Development test | 22.7117 | 0.5285 | 0.4419 | 2.83 |
| Lightweight CNN | Development test | 26.7578 | 0.7060 | 0.3467 | 38.47 |

Timings cover only the transform/model forward pass at batch size 1 after one warm-up on the local CPU. They exclude file I/O and metric calculation.

PSNR and SSIM use predictions and GT clipped to `[0,1]` with `data_range=1.0`. LPIPS-Alex replicates grayscale to three channels only inside the metric and scales it to `[-1,1]`. The actual model remains one-channel.

## 6. Environment and Data Verification

Run commands from this repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py --seed 42
python scripts/audit_data.py
```

Phase 1 is frozen and must not be retrained. The commands below document evaluation of its preserved artifacts.

Full frozen evaluation commands:

```powershell
python scripts/benchmark.py --split val
python scripts/benchmark.py --split dev_test

python evaluate.py --config configs/baseline_cnn.yaml --checkpoint checkpoints/baseline_cnn_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/baseline_cnn/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/baseline_cnn_val_metrics.json

python evaluate.py --config configs/baseline_cnn.yaml --checkpoint checkpoints/baseline_cnn_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/baseline_cnn/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/baseline_cnn_dev_test_metrics.json

python evaluate.py --config configs/baseline_cnn.yaml --checkpoint checkpoints/baseline_cnn_best.pth --input_dir data/extracted/test/NoisyLR --output_dir outputs/baseline_cnn/blind_test --results_file outputs/baseline_cnn_blind_test_inference.json
```

The evaluation commands are recorded for reproducibility; repository cleanup does not require rerunning them.

## 7. Current Model

The canonical Phase 1 model is `checkpoints/baseline_cnn_best.pth`, selected by validation PSNR at epoch 3. Its SHA-256 is:

```text
0C3735AF661D0D082E43577B5B746E99E574000E35FB914715C20E109ED85B33
```

The exact configuration and training history are stored alongside it in `checkpoints/`. Older checkpoints are retained under `checkpoints/archive/` and must not be used as the canonical baseline.

## 8. Data Integrity Rules

- Never clip or normalize the original degraded NoisyLR arrays before the model.
- Keep inputs and targets as float32 grayscale arrays.
- Require input shape 128x128 and target/output shape 256x256.
- Keep GT and evaluated predictions in the valid `[0,1]` range.
- Preserve filename pairing and the exact seed-42 split manifests.
- Never use validation, development-test, or blind-test data for training.
- Never use blind-test data for model selection.
- Preserve the original archives without recompression or modification.

## 9. Experiment Policy

Baseline first, one clear change per experiment, and measure PSNR, SSIM, LPIPS, and inference time consistently. Record configuration, seed, split, loss, optimizer, learning rate, batch size, epochs, parameter count, checkpoint, and decision. Do not overwrite frozen Phase 1 artifacts; new experiments must receive distinct run IDs and output paths.

## Phase 2

### B2 NAFNet-style baseline

B2 was introduced to test one controlled change: replace B1's conventional residual blocks with a compact NAFNet-style restoration architecture while preserving the dataset, seed-42 splits, grayscale representation, augmentation, Charbonnier loss, optimizer settings, and metric definitions.

Phase 1 remains permanently frozen. B2 uses the unchanged seed-42 split manifests: 2,560 training pairs, 320 validation pairs, and 320 development-test pairs.

The model performs nearly all work in the 128x128 LR feature space. It uses shallow feature extraction, eight NAFNet-style blocks, feature refinement, late learned pixel-shuffle reconstruction, and a final sigmoid. Each block contains channel-wise LayerNorm on learned features, 1x1 expansion, depthwise spatial convolution, SimpleGate feature gating, bottleneck channel attention, projection, and two learnably scaled residual branches. Raw degraded pixels are not clipped or normalized.

| Property | B2 setting |
|---|---|
| Feature width | 32 |
| NAF blocks | 8 |
| Expansion factor | 2 |
| Attention reduction | 4 |
| Initial residual scale | 0.1 |
| Upsampling | Pixel shuffle x2 near output |
| Parameters | 108,609 |
| Input contract | `[N,1,128,128]` float32 NoisyLR, raw range preserved |
| Output contract | `[N,1,256,256]` prediction in `[0,1]` |
| Float32 parameter memory | 0.414 MiB |
| Loss | Charbonnier, epsilon 0.001 |
| Optimizer | Adam, LR 0.001 |
| Scheduler | Cosine, 3 configured epochs |
| Batch size / seed | 16 / 42 |

Architecture checks passed for an audited-range `[2,1,128,128]` float32 input and `[2,1,256,256]` float32 output. The input tensor remained unchanged, the output was finite and within `[0,1]`, `loss.backward()` succeeded, and every trainable gradient was finite.

CUDA is unavailable on the current machine, so the required hardware stop condition was applied. No B2 training, checkpoint, validation/dev-test evaluation, or visual comparison was produced. In a same-process architecture microbenchmark using batch size 1, 10 warm-ups, and 50 timed forwards, untrained B2 took 43.83 ms/image and checkpoint-loaded B1 took 22.64 ms/image. B2 was about 1.94x slower in this microbenchmark. These are not official quality/speed results and must not be mixed with the frozen Phase 1 timing or used for KEEP/DROP selection.

| Model | Validation PSNR / SSIM / LPIPS | Dev-test PSNR / SSIM / LPIPS | Status |
|---|---|---|---|
| B1 Lightweight CNN | 27.1655 / 0.7305 / 0.3280 | 26.7578 / 0.7060 / 0.3467 | Frozen KEEP |
| B2 NAFNet-style | Not run - CUDA required | Not run - CUDA required | PENDING GPU training |

B2 is neither kept nor dropped yet. The planned checkpoint is `checkpoints/baseline_nafnet_best.pth`; it does not exist until a proper training run selects a best validation epoch. B1 remains the canonical trained model.

Reproduce the hardware-safe architecture check:

```powershell
python scripts/model_sanity.py --config configs/baseline_nafnet.yaml --warmup 10 --iterations 50 --comparison-config configs/baseline_cnn.yaml --comparison-checkpoint checkpoints/baseline_cnn_best.pth --results-file outputs/nafnet_sr_architecture_sanity.json
```

NVIDIA/CUDA is recommended for B2 training. On the receiving machine, install a CUDA-enabled PyTorch 2.3 or newer build appropriate for its operating system and CUDA environment, then install the remaining requirements. Verify that the installed build can see the GPU before running B2; do not hardcode a GPU model:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python scripts/cuda_smoke_test.py --config configs/baseline_nafnet.yaml
```

The smoke test uses synthetic tensors only, exercises the canonical B2 CUDA/AMP forward and backward paths, and does not train the dataset or write checkpoints. It must pass before full training.

After the smoke test, run the controlled B2 experiment and then dedicated evaluation:

```powershell
python train.py --config configs/baseline_nafnet.yaml --no-lpips

python evaluate.py --config configs/baseline_nafnet.yaml --checkpoint checkpoints/baseline_nafnet_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/nafnet_sr/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/nafnet_sr_val_metrics.json

python evaluate.py --config configs/baseline_nafnet.yaml --checkpoint checkpoints/baseline_nafnet_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/nafnet_sr/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/nafnet_sr_dev_test_metrics.json
```

`--no-lpips` skips LPIPS during training to reduce overhead. Final validation and development-test evaluation through `evaluate.py` still calculates PSNR, SSIM, and LPIPS. B2 remains `PENDING_GPU`; no B2 quality metric may be claimed until the NVIDIA machine has completed actual training and evaluation.

Do not generate B2 blind-test predictions or comparison panels until B2 has been trained and selected without using blind data.

## 10. Implemented Experimental Sequence

The implementation follows the project master plan while keeping B2 unchanged:

| Run | Controlled change | Parameters | Status |
|---|---|---:|---|
| `B2_NAFNET_BASELINE` | Canonical compact NAFNet-style baseline | 108,609 | PENDING_GPU |
| `P1_DEGRADATION_AWARE` | Raw-LR degradation descriptor conditions each NAF block | 118,561 | PENDING_GPU |
| `P2_SYNTHETIC_DEGRADATION` | P1 plus varied, unclipped synthetic degradation on training samples only | 118,561 | PENDING_GPU |
| `P3_EDGE_LOSS` | P2 plus edge loss | 118,561 | PENDING_GPU |
| `P3_SSIM_LOSS` | P2 plus SSIM loss | 118,561 | PENDING_GPU |
| `P3_FREQUENCY_LOSS` | P2 plus frequency loss | 118,561 | PENDING_GPU |

Each P3 configuration is an isolated ablation; the three losses are not bundled. Every run uses the exact seed-42 manifests. Synthetic degradation is enabled only in the training dataset and never affects validation, development-test, or blind-test inputs. The experiment registry is `configs/experiment_registry.yaml`, and future factual results belong in `docs/phase2_plus_experiment_log.csv`; the frozen Phase 1 log is not reused or edited.

Run the complete non-training validation suite before launching experiments:

```bash
python scripts/validate_experiments.py
python scripts/create_hard_val.py --dry-run
python scripts/cuda_smoke_test.py --config configs/baseline_nafnet.yaml
```

Training proceeds in order, with a review after each stage. A representative command is:

```bash
python train.py --config configs/phase3_p1_degradation_aware.yaml --no-lpips
```

Do not auto-select a winner. Populate real metric JSON files by evaluating validation and development-test independently, benchmark inference under one hardware protocol, inspect structures visually, and make a human KEEP/DROP decision. The final pipeline requires an explicit selected run ID and confirmation flag, and refuses to proceed unless that run's validation and development-test metric files exist.

The complete Git Bash NVIDIA setup, every exact experiment/evaluation command, resume instructions, troubleshooting, backup procedure, and result checklist are in [`docs/gpu_handoff.md`](docs/gpu_handoff.md). Until those experiments are run, the project status remains `CODE_COMPLETE_GPU_EXPERIMENTS_PENDING`.
