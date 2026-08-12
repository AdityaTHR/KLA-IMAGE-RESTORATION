# KLA PS01 - Image Restoration Starter

A clean first baseline for the SemiCon AI Hackathon KLA image-restoration problem.

## 1. Ubuntu setup

```bash
sudo apt update
sudo apt install -y python3-venv unzip git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch using the official Linux/Pip selector for your machine: https://pytorch.org/get-started/locally/
Then:

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Put the data here

Expected extracted layout:

```text
data/train/
  GT/000000.npy
  NoisyLR/000000.npy
```

If you have `train.zip`:

```bash
unzip train.zip -d data/
# This normally creates data/train/GT and data/train/NoisyLR
```

## 3. Verify data before training

```bash
python scripts/audit_data.py --data-root data/train
```

Expected key facts for the provided dataset: 3200 pairs, LR 128x128, GT 256x256, float32, and some LR pixels outside [0,1].

## 4. Run non-learning baseline

```bash
python scripts/bicubic_baseline.py --data-root data/train
```

Reference result measured on all 3200 provided pairs (bicubic + output clipping): about 22.853 dB PSNR and 0.5361 SSIM.

## 5. Smoke-test the neural baseline

```bash
python scripts/smoke_test.py
```

## 6. Train first learned baseline

Quick test:

```bash
python train_baseline.py --data-root data/train --epochs 1 --batch-size 4 --workers 2
```

Normal first run on a GPU:

```bash
python train_baseline.py --data-root data/train --epochs 30 --batch-size 16 --workers 4
```

Best checkpoint is saved to `checkpoints/baseline_best.pt`.

## 7. Run inference

Extract test data so `.npy` files are in a folder, for example `data/test/NoisyLR/`, then:

```bash
python evaluate.py \
  --input data/test/NoisyLR \
  --output outputs/restored \
  --weights checkpoints/baseline_best.pt
```

## 8. Why this baseline exists

Do not try to win with this small model. It is a controlled learned baseline. Once it is proven end-to-end, compare stronger NAFNet/SwinIR-style and degradation-aware variants against it without breaking the evaluation pipeline.
