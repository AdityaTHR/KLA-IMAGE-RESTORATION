# NOISE_IC_NONE
## KLA Semiconductor Image Restoration

AI-based restoration of degraded grayscale images affected by Gaussian noise,
speckle noise and 2x downsampling.

## Method

The submitted system uses a lightweight degradation-aware NAF-based
super-resolution model followed by conservative measurement-derived
high-frequency detail fusion.

The detail fusion coefficient is fixed at beta = 0.04.

No external API, internet access, user interaction, or model download
is required during inference.

## Input

A directory containing `.npy` grayscale degraded images.

Expected standard task resolution:

    128 x 128

Input values are intentionally NOT clipped before inference.

## Output

For every input `.npy` file, the program creates one restored `.npy`
file with the identical filename.

Expected output resolution:

    256 x 256

Output arrays are:

- grayscale
- float32
- finite
- clipped to [0,1]

## Installation

Python 3 with CUDA-capable PyTorch is recommended.

Install dependencies:

    pip install -r requirements.txt

## Execution

From this directory:

    python run.py <input-dir> <output-dir>

Example:

    python run.py /path/to/NoisyLR /path/to/restored

The output directory is created automatically.

## GPU

If CUDA is available, the solution automatically uses the NVIDIA GPU.

No manual source-code modification is required.

## Files

    run.py
    requirements.txt
    README.md
    models/
    src/

`models/` contains the trained model weights and configuration required
for offline inference.

## Reproducibility

Inference uses a fixed trained checkpoint and deterministic fixed
post-processing coefficient. No stochastic test-time augmentation is used.
