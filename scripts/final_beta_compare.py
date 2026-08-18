import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from src.metrics.image_metrics import MetricsCalculator

BETAS = [0.02, 0.03, 0.04, 0.05, 0.06]

NOISY = "data/extracted/train/NoisyLR"
GT = "data/extracted/train/GT"

SPLITS = {
    "val": {
        "split": "data/splits/val.txt",
        "base": "outputs/NOISE_IC_NONE_final/visual_val",
    },
    "dev_test": {
        "split": "data/splits/dev_test.txt",
        "base": "outputs/NOISE_IC_NONE_final/visual_dev",
    },
}

OUT = "outputs/FINAL_BETA_COMPARISON"
os.makedirs(OUT, exist_ok=True)


def gaussian_detail(noisy):
    x = torch.from_numpy(noisy.astype(np.float32))[None, None]

    bicubic = F.interpolate(
        x,
        scale_factor=2,
        mode="bicubic",
        align_corners=False,
    )

    radius = 4
    sigma = 1.0

    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    kernel = (g[:, None] * g[None, :])[None, None]

    low = F.conv2d(
        F.pad(
            bicubic,
            (radius, radius, radius, radius),
            mode="reflect",
        ),
        kernel,
    )

    detail = bicubic - low

    return detail[0, 0].numpy()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calc = MetricsCalculator(device=device)

all_results = {}

for split_name, info in SPLITS.items():

    with open(info["split"]) as f:
        files = [x.strip() for x in f if x.strip()]

    results = {
        b: {"psnr": [], "ssim": [], "lpips": []}
        for b in BETAS
    }

    # Used to select visually useful examples
    hard_scores = []

    print(f"\n===== {split_name.upper()} =====")

    for i, name in enumerate(files, 1):

        noisy = np.load(os.path.join(NOISY, name)).astype(np.float32)
        gt = np.load(os.path.join(GT, name)).astype(np.float32)
        base = np.load(os.path.join(info["base"], name)).astype(np.float32)

        detail = gaussian_detail(noisy)

        hard_scores.append(
            (float(np.mean((base - gt) ** 2)), name)
        )

        for beta in BETAS:

            pred = np.clip(
                base + beta * detail,
                0.0,
                1.0,
            ).astype(np.float32)

            m = calc.calculate_all(pred, gt)

            results[beta]["psnr"].append(m["psnr"])
            results[beta]["ssim"].append(m["ssim"])
            results[beta]["lpips"].append(m["lpips"])

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)}")

    print("\nBETA RESULTS")

    for beta in BETAS:

        p = np.mean(results[beta]["psnr"])
        s = np.mean(results[beta]["ssim"])
        l = np.mean(results[beta]["lpips"])

        print(
            f"beta={beta:.2f} | "
            f"PSNR={p:.4f} | "
            f"SSIM={s:.5f} | "
            f"LPIPS={l:.5f}"
        )

    all_results[split_name] = results

    # --------------------------------------------------
    # Visual comparison: 6 hard + 2 middle difficulty
    # --------------------------------------------------
    hard_scores.sort(reverse=True)

    selected = [x[1] for x in hard_scores[:6]]

    middle = len(hard_scores) // 2
    selected += [
        hard_scores[middle][1],
        hard_scores[min(middle + 20, len(hard_scores)-1)][1],
    ]

    vis_dir = os.path.join(OUT, split_name)
    os.makedirs(vis_dir, exist_ok=True)

    for idx, name in enumerate(selected, 1):

        noisy = np.load(os.path.join(NOISY, name)).astype(np.float32)
        gt = np.load(os.path.join(GT, name)).astype(np.float32)
        base = np.load(os.path.join(info["base"], name)).astype(np.float32)

        detail = gaussian_detail(noisy)

        images = [noisy, base]
        titles = ["Degraded", "Base NAF"]

        for beta in BETAS:
            pred = np.clip(base + beta * detail, 0, 1)
            images.append(pred)
            titles.append(f"β={beta:.2f}")

        images.append(gt)
        titles.append("Ground Truth")

        fig, axes = plt.subplots(
            1,
            len(images),
            figsize=(24, 4),
        )

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(name)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                vis_dir,
                f"beta_compare_{idx:02d}_{name.replace('.npy','')}.png"
            ),
            dpi=220,
            bbox_inches="tight",
        )

        plt.close()

print("\nDONE")
print("Images saved in:", OUT)
