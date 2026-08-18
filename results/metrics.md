# Final Results — Noise? IC None

## Final system
W48x10 degradation-aware NAF + averaged checkpoint + measurement-derived HF fusion beta=0.04.

## Validation
PSNR: 28.3994 dB
SSIM: 0.77685
LPIPS: 0.24805

## Dev-test
PSNR: 27.8400 dB
SSIM: 0.75019
LPIPS: 0.27887

## Base W48x10 model
Validation PSNR: 28.4238
Validation SSIM: 0.77724
Validation LPIPS: 0.27074

## Baseline
Bicubic PSNR: 22.8530 dB
Bicubic SSIM: 0.5361

## Final local inference validation
GPU: NVIDIA RTX 4060 Laptop GPU
Final processing measurement: 13.15 ms/image
Blind inputs processed: 400/400
Output: 256x256 float32
Output range: [0,1]
NaN/Inf: none

beta=0.04 was selected as a conservative balance point across PSNR, SSIM,
LPIPS and visual fidelity. It is not claimed to be a mathematical optimum.
