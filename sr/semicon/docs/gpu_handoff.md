# NVIDIA GPU Experiment Handoff — Windows Git Bash

This guide executes the implemented SemiCon research plan. Phase 1 is immutable. B2 and all later experiments are implemented but remain `PENDING_GPU` until this workflow produces real checkpoints and metrics.

Never manually edit metrics, experiment logs, checkpoints, predictions, or training histories. If a run fails, report it as `FAILED` or `BLOCKED` with the real error.

## 1. First-time project setup

If the repository is hosted on Git, replace the placeholder with the real URL:

```bash
git clone <REPOSITORY_URL>
cd semicon
```

If there is no remote, copy the complete `semicon` folder—including `data/splits/` and the dataset—to the NVIDIA machine, open Git Bash, and enter it:

```bash
cd /path/to/semicon
```

Verify the project root:

```bash
pwd
ls
git status
```

The root must contain `README.md`, `requirements.txt`, `train.py`, `evaluate.py`, `configs/`, `src/`, `scripts/`, `data/`, `checkpoints/`, `outputs/`, and `docs/`. If the folder was copied without Git metadata, `git status` may say it is not a repository; that does not affect execution.

## 2. Python environment

Use a supported 64-bit Python release, preferably Python 3.10–3.12:

```bash
python -m venv .venv
source .venv/Scripts/activate
python --version
python -m pip install --upgrade pip
```

## 3. NVIDIA, CUDA, and PyTorch

First inspect the NVIDIA driver:

```bash
nvidia-smi
```

Install a CUDA-enabled PyTorch build compatible with that driver using the official PyTorch installation selector. Do not install an arbitrary CUDA wheel from this guide. Once the correct PyTorch is installed, verify it:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
```

Do not start training unless `CUDA available: True`, a CUDA build is printed, and an NVIDIA GPU name is printed.

Install project dependencies. A compatible installed PyTorch already satisfying `torch>=2.3` should not be reinstalled:

```bash
python -m pip install -r requirements.txt
python -c "import torch, numpy, yaml, skimage, lpips; print('Core imports: PASS')"
```

Run the non-mutating setup verifier:

```bash
bash scripts/setup_nvidia_gitbash.sh
```

## 4. Dataset preparation and verification

Expected structure:

```text
data/
├── raw_archives/
│   ├── train.zip
│   └── Test_NoisyLR.zip
├── extracted/
│   ├── train/
│   │   ├── NoisyLR/
│   │   └── GT/
│   └── test/
│       └── NoisyLR/
└── splits/
    ├── train.txt
    ├── val.txt
    └── dev_test.txt
```

If extracted data is absent, use the repository script. It verifies and preserves existing frozen seed-42 manifests rather than overwriting them:

```bash
python scripts/prepare_data.py
```

Audit everything:

```bash
python scripts/audit_data.py
python scripts/validate_experiments.py
```

Required counts are 3,200 paired NoisyLR images, 3,200 GT images, 400 blind-test NoisyLR images, and split counts 2,560/320/320. NoisyLR values outside `[0,1]` are expected. Never clip or normalize NoisyLR.

The master plan also defines a validation-only hard-degradation proxy. It never changes canonical splits or training data:

```bash
python scripts/create_hard_val.py --count 80 --output data/derived_splits/hard_val.txt
```

## 5. Required CUDA smoke test

Run this before expensive training:

```bash
python scripts/cuda_smoke_test.py --config configs/baseline_nafnet.yaml
```

It must confirm CUDA, GPU identity, CUDA AMP, 108,609 B2 parameters, `[2,1,128,128]` input, `[2,1,256,256]` output, preserved raw out-of-range input, finite `[0,1]` output, Charbonnier loss, backward propagation, and finite gradients.

If it fails, stop. Do not start training.

## 6. Phase 2 — B2 NAFNet-style baseline

- Hypothesis: NAFNet-style blocks improve restoration over frozen B1.
- Parent: `B1_LIGHTWEIGHT_CNN`
- Config: `configs/baseline_nafnet.yaml`
- Parameters: 108,609
- LPIPS during training: skipped; computed during final evaluation.
- Best checkpoint: `checkpoints/baseline_nafnet_best.pth`
- History: `checkpoints/B2_NAFNET_SR_BASELINE_training_history.csv`

```bash
python train.py --config configs/baseline_nafnet.yaml --no-lpips
```

Resume from an existing completed epoch checkpoint; replace `1` with the actual saved epoch:

```bash
python train.py --config configs/baseline_nafnet.yaml --resume checkpoints/baseline_nafnet_epoch_1.pth --no-lpips
```

Validation and dev-test:

```bash
python evaluate.py --config configs/baseline_nafnet.yaml --checkpoint checkpoints/baseline_nafnet_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/nafnet_sr/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/nafnet_sr_val_metrics.json
python evaluate.py --config configs/baseline_nafnet.yaml --checkpoint checkpoints/baseline_nafnet_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/nafnet_sr/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/nafnet_sr_dev_test_metrics.json
```

## 7. Phase 3 — P1 degradation-aware conditioning

- Hypothesis: learned conditioning improves mixed-degradation handling.
- Parent: `B2_NAFNET_SR_BASELINE`
- Isolated change: degradation encoder plus per-block affine modulation.
- Config: `configs/phase3_p1_degradation_aware.yaml`
- Parameters: 118,561
- Best checkpoint: `checkpoints/p1_degradation_aware_best.pth`
- History: `checkpoints/P1_DEGRADATION_AWARE_training_history.csv`

```bash
python train.py --config configs/phase3_p1_degradation_aware.yaml --no-lpips
python train.py --config configs/phase3_p1_degradation_aware.yaml --resume checkpoints/p1_degradation_aware_epoch_1.pth --no-lpips
python evaluate.py --config configs/phase3_p1_degradation_aware.yaml --checkpoint checkpoints/p1_degradation_aware_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p1_degradation_aware/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/p1_degradation_aware_val_metrics.json
python evaluate.py --config configs/phase3_p1_degradation_aware.yaml --checkpoint checkpoints/p1_degradation_aware_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p1_degradation_aware/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/p1_degradation_aware_dev_test_metrics.json
```

## 8. Phase 4 — P2 synthetic-degradation robustness

- Hypothesis: varied signal-dependent degradation improves robustness.
- Parent: `P1_DEGRADATION_AWARE`
- Isolated change: seeded synthetic speckle, Gaussian, interpolation, order, and noise-domain variation on 50% of training samples.
- Config: `configs/phase4_p2_synthetic_degradation.yaml`
- Parameters: 118,561
- Best checkpoint: `checkpoints/p2_synthetic_degradation_best.pth`
- History: `checkpoints/P2_SYNTHETIC_DEGRADATION_training_history.csv`

```bash
python train.py --config configs/phase4_p2_synthetic_degradation.yaml --no-lpips
python train.py --config configs/phase4_p2_synthetic_degradation.yaml --resume checkpoints/p2_synthetic_degradation_epoch_1.pth --no-lpips
python evaluate.py --config configs/phase4_p2_synthetic_degradation.yaml --checkpoint checkpoints/p2_synthetic_degradation_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p2_synthetic_degradation/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/p2_synthetic_degradation_val_metrics.json
python evaluate.py --config configs/phase4_p2_synthetic_degradation.yaml --checkpoint checkpoints/p2_synthetic_degradation_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p2_synthetic_degradation/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/p2_synthetic_degradation_dev_test_metrics.json
```

## 9. Phase 5 — P3 structure-loss ablations

All three runs use the P2 model/data pipeline and change exactly one loss term. Train and evaluate all three before any decision.

### P3-E edge loss

```bash
python train.py --config configs/phase5_p3_edge_loss.yaml --no-lpips
python train.py --config configs/phase5_p3_edge_loss.yaml --resume checkpoints/p3_edge_loss_epoch_1.pth --no-lpips
python evaluate.py --config configs/phase5_p3_edge_loss.yaml --checkpoint checkpoints/p3_edge_loss_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p3_edge_loss/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/p3_edge_loss_val_metrics.json
python evaluate.py --config configs/phase5_p3_edge_loss.yaml --checkpoint checkpoints/p3_edge_loss_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p3_edge_loss/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/p3_edge_loss_dev_test_metrics.json
```

Expected best checkpoint/history: `checkpoints/p3_edge_loss_best.pth` and `checkpoints/P3_EDGE_LOSS_training_history.csv`.

### P3-S SSIM loss

```bash
python train.py --config configs/phase5_p3_ssim_loss.yaml --no-lpips
python train.py --config configs/phase5_p3_ssim_loss.yaml --resume checkpoints/p3_ssim_loss_epoch_1.pth --no-lpips
python evaluate.py --config configs/phase5_p3_ssim_loss.yaml --checkpoint checkpoints/p3_ssim_loss_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p3_ssim_loss/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/p3_ssim_loss_val_metrics.json
python evaluate.py --config configs/phase5_p3_ssim_loss.yaml --checkpoint checkpoints/p3_ssim_loss_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p3_ssim_loss/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/p3_ssim_loss_dev_test_metrics.json
```

Expected best checkpoint/history: `checkpoints/p3_ssim_loss_best.pth` and `checkpoints/P3_SSIM_LOSS_training_history.csv`.

### P3-F frequency loss

```bash
python train.py --config configs/phase5_p3_frequency_loss.yaml --no-lpips
python train.py --config configs/phase5_p3_frequency_loss.yaml --resume checkpoints/p3_frequency_loss_epoch_1.pth --no-lpips
python evaluate.py --config configs/phase5_p3_frequency_loss.yaml --checkpoint checkpoints/p3_frequency_loss_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p3_frequency_loss/val --gt_dir data/extracted/train/GT --split_file data/splits/val.txt --results_file outputs/p3_frequency_loss_val_metrics.json
python evaluate.py --config configs/phase5_p3_frequency_loss.yaml --checkpoint checkpoints/p3_frequency_loss_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/p3_frequency_loss/dev_test --gt_dir data/extracted/train/GT --split_file data/splits/dev_test.txt --results_file outputs/p3_frequency_loss_dev_test_metrics.json
```

Expected best checkpoint/history: `checkpoints/p3_frequency_loss_best.pth` and `checkpoints/P3_FREQUENCY_LOSS_training_history.csv`.

## 10. Comparison, hard validation, and speed

Build factual comparison files after all validation/dev-test JSON files exist:

```bash
python scripts/compare_experiments.py --output-json outputs/experiment_comparison.json --output-csv outputs/experiment_comparison.csv
```

Use a balanced PSNR/SSIM/LPIPS/speed/parameter review. Do not select from one metric alone. The comparison script never assigns KEEP/DROP.

After checking the generated comparison, materialize the Phase 2+ experiment log directly from registry, history, and metric artifacts. This updates only the non-frozen log; it never edits `docs/experiment_log.csv`:

```bash
python scripts/compare_experiments.py --output-log docs/phase2_plus_experiment_log.csv --overwrite
```

The `--overwrite` flag is explicit because the repository contains the initial `PENDING_GPU` log template. Metric values are copied from pipeline output, never entered by hand. KEEP/DROP remains a separate human decision and is not inferred by this command.

For a trained run, measure synchronized model-only speed; substitute that run's real config/checkpoint:

```bash
python scripts/benchmark_model.py --config configs/baseline_nafnet.yaml --checkpoint checkpoints/baseline_nafnet_best.pth --split val --warmup 10 --iterations 100 --results-file outputs/nafnet_sr_benchmark.json
```

Optional hard-validation evaluation uses only the derived validation subset, never dev-test or blind test:

```bash
python evaluate.py --config configs/baseline_nafnet.yaml --checkpoint checkpoints/baseline_nafnet_best.pth --input_dir data/extracted/train/NoisyLR --output_dir outputs/nafnet_sr/hard_val --gt_dir data/extracted/train/GT --split_file data/derived_splits/hard_val.txt --results_file outputs/nafnet_sr_hard_val_metrics.json
```

Repeat with another run's actual config/checkpoint/output slug when comparing robustness.

Create deterministic qualitative panels from already-generated predictions. The first column displays raw NoisyLR with a fixed `[0,1]` display range but does not modify the stored array or any model input:

```bash
python scripts/create_comparison_panels.py --split-file data/splits/dev_test.txt --input-dir data/extracted/train/NoisyLR --gt-dir data/extracted/train/GT --prediction B1=outputs/baseline_cnn/dev_test --prediction B2=outputs/nafnet_sr/dev_test --output-dir outputs/comparison_panels/b1_vs_b2 --count 8
```

Add more `--prediction LABEL=DIRECTORY` arguments for later runs. To reproduce an exact review set, pass explicit split filenames after `--filenames`. Existing PNG panels are protected unless `--overwrite` is supplied explicitly.

## 11. Final blind-test inference

Blind test contains 400 samples and no GT. It must never influence model selection. Only after all comparisons and a documented human selection, run:

```bash
python scripts/run_final_pipeline.py --run-id <SELECTED_RUN_ID> --checkpoint <SELECTED_CHECKPOINT> --confirm-selected
```

`<SELECTED_RUN_ID>` and `<SELECTED_CHECKPOINT>` cannot be filled honestly until GPU results exist. They must match an actual registry entry and checkpoint. The script refuses to run unless that experiment's canonical validation and dev-test metric JSON files already exist. It reruns validation/dev-test, performs blind inference, validates all 400 deterministic filenames, shape `(256,256)`, float32 dtype, finiteness, and `[0,1]` range, then saves under `outputs/final/<selected-output-slug>/`.

The underlying direct blind inference interface is:

```bash
python evaluate.py --config <SELECTED_CONFIG> --checkpoint <SELECTED_CHECKPOINT> --input_dir data/extracted/test/NoisyLR --output_dir outputs/final/blind_test --results_file outputs/final/blind_inference.json
python scripts/validate_submission.py --prediction-dir outputs/final/blind_test --input-dir data/extracted/test/NoisyLR --expected-count 400 --results-file outputs/final/submission_validation.json
```

Do not invent the selected values or a leaderboard score.

## 12. Complete execution order

- [ ] Project root verified
- [ ] Virtual environment active
- [ ] NVIDIA driver visible through `nvidia-smi`
- [ ] CUDA-enabled PyTorch verified
- [ ] Dependencies imported
- [ ] Dataset and canonical splits audited
- [ ] Experiment registry validated
- [ ] CUDA smoke test passed
- [ ] B2 trained, validation evaluated, dev-test evaluated, and analyzed
- [ ] P1 trained, validation evaluated, dev-test evaluated, and analyzed
- [ ] P2 trained, validation evaluated, dev-test evaluated, and analyzed
- [ ] P3-E trained, validation evaluated, dev-test evaluated, and analyzed
- [ ] P3-S trained, validation evaluated, dev-test evaluated, and analyzed
- [ ] P3-F trained, validation evaluated, dev-test evaluated, and analyzed
- [ ] Hard-validation comparisons completed where useful
- [ ] Synchronized speed benchmarks completed
- [ ] Comparison report generated
- [ ] Qualitative comparison panels reviewed
- [ ] Best model selected by balanced human review
- [ ] Final pipeline executed
- [ ] All 400 blind predictions validated
- [ ] Final artifacts backed up

## 13. Checkpoint and result backup

Back up after each completed experiment:

```bash
mkdir -p backup/checkpoints backup/results
cp checkpoints/baseline_nafnet_best.pth checkpoints/B2_NAFNET_SR_BASELINE_training_history.csv backup/checkpoints/
cp checkpoints/p1_degradation_aware_best.pth checkpoints/P1_DEGRADATION_AWARE_training_history.csv backup/checkpoints/
cp checkpoints/p2_synthetic_degradation_best.pth checkpoints/P2_SYNTHETIC_DEGRADATION_training_history.csv backup/checkpoints/
cp checkpoints/p3_edge_loss_best.pth checkpoints/P3_EDGE_LOSS_training_history.csv backup/checkpoints/
cp checkpoints/p3_ssim_loss_best.pth checkpoints/P3_SSIM_LOSS_training_history.csv backup/checkpoints/
cp checkpoints/p3_frequency_loss_best.pth checkpoints/P3_FREQUENCY_LOSS_training_history.csv backup/checkpoints/
cp outputs/*_val_metrics.json outputs/*_dev_test_metrics.json backup/results/
```

Never overwrite another experiment's checkpoint. Preserve every best checkpoint, history CSV, metric JSON, benchmark JSON, environment details, and error log.

## 14. Safe Git workflow

Before work:

```bash
git status
git pull --ff-only
```

For intentional code/documentation changes:

```bash
git add README.md configs src scripts docs
git commit -m "Document NVIDIA experiment execution"
```

After returning an experiment's small metadata files, an example is:

```bash
git add outputs/experiment_comparison.json outputs/experiment_comparison.csv
git commit -m "Record GPU experiment comparison artifacts"
```

Do not commit dataset archives or extracted arrays. Do not invent Git LFS setup. Coordinate checkpoint handling with the repository owner before committing large weights.

## 15. Training interruption and resume

The trainer saves model, optimizer, scheduler, AMP scaler, DataLoader generator, Python, NumPy, CPU Torch, and CUDA RNG states after every epoch. List available checkpoints:

```bash
ls checkpoints/*_epoch_*.pth
```

Use the matching config and an existing checkpoint from the same experiment. The exact per-experiment resume commands are shown above. If a checkpoint is missing, do not invent one. If loading reports corruption or an architecture mismatch, stop and report it.

## 16. Troubleshooting

| Problem | Symptom | Cause | Safe fix | Stop when |
|---|---|---|---|---|
| CUDA unavailable | `CUDA available: False` | CPU PyTorch or driver/GPU unavailable | Reinstall the correct official CUDA-enabled PyTorch build after checking `nvidia-smi` | It remains false |
| CPU-only PyTorch | `torch.version.cuda` is `None` | CPU wheel installed | Replace it with a driver-compatible official CUDA build | CUDA build stays `None` |
| GPU not detected | `nvidia-smi` fails | Driver/device problem | Repair the NVIDIA driver outside the project | `nvidia-smi` still fails |
| CUDA mismatch | Import/runtime CUDA errors | Incompatible driver and PyTorch build | Select a compatible official PyTorch build | Compatibility is uncertain |
| CUDA OOM | `CUDA out of memory` | Insufficient free VRAM | Close GPU applications and retry the identical config | It still fails; do not change research hyperparameters silently |
| General OOM | Process killed or host memory error | Insufficient RAM/VRAM | Close applications, verify no duplicate runs, report hardware | Reproducible failure persists |
| LPIPS network error | Weight download failure | No network/cache | Restore network access or pre-cache official LPIPS dependencies | Weights cannot be obtained reliably |
| Missing dataset | File/directory assertion | Dataset not copied/extracted | Run `python scripts/prepare_data.py` with actual archives | Archives are absent |
| Wrong dataset path | Missing NoisyLR/GT | Not in project root or structure differs | Return to root and match documented layout | Moving data would risk overwriting it |
| Missing splits | `train.txt` not found | Incomplete repository copy | Restore the frozen manifests from the repository | Hash/source cannot be verified |
| Missing checkpoint | `FileNotFoundError` | Training not completed or wrong filename | List `checkpoints/` and use the real matching file | No matching file exists |
| Invalid resume | Clear resume-path failure | Typo or nonexistent epoch file | Use an existing same-run epoch checkpoint | Provenance is uncertain |
| Git Bash paths | Path not found | Windows path pasted in non-Bash form | `cd` from VS Code or use `/c/...` Git Bash syntax | Project root remains unclear |
| Import error | Missing Python package | Wrong venv or incomplete install | Activate `.venv`, rerun requirements, verify imports | Installing would alter CUDA PyTorch unexpectedly |
| AMP error | Failure inside autocast/scaler | Unsupported/incorrect PyTorch CUDA build | Run CUDA smoke test and repair environment | Smoke test still fails |
| Interrupted training | Process stopped | Power/session interruption | Resume from latest completed matching epoch checkpoint | Checkpoint is missing/corrupt |

Never redesign the architecture or alter canonical hyperparameters merely to bypass an environment problem.

## 17. Result-return checklist

After every experiment, return machine-readable artifacts and this metadata:

1. Experiment/run ID
2. Config used
3. Best checkpoint
4. Training history CSV
5. Validation metrics JSON
6. Dev-test metrics JSON
7. Inference timing or benchmark JSON
8. GPU model
9. GPU VRAM
10. PyTorch version
11. CUDA build/version
12. Training duration
13. Best/final epoch
14. Best validation PSNR
15. Validation SSIM
16. Validation LPIPS
17. Dev-test PSNR
18. Dev-test SSIM
19. Dev-test LPIPS
20. Every warning/error

Prefer files over screenshots. Never alter generated values to make results look better. Failed experiments remain failed or blocked.

## 18. Quick start

After entering the project folder:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
nvidia-smi
# Install the driver-compatible CUDA-enabled PyTorch command from the official selector here.
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/audit_data.py
python scripts/validate_experiments.py
bash scripts/setup_nvidia_gitbash.sh
python scripts/cuda_smoke_test.py --config configs/baseline_nafnet.yaml
python train.py --config configs/baseline_nafnet.yaml --no-lpips
```

Stop immediately if CUDA verification, dataset audit, registry validation, or the CUDA smoke test fails.
