import time
import torch
import lpips
import numpy as np
from torch.utils.data import DataLoader

from src.data import PairedNpyDataset, make_split
from src.model import BaselineRestorer
from src.metrics import psnr, ssim

DATA = "data/train/train"
WEIGHTS = "checkpoints/baseline_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_ids = make_split(DATA, 0.10, 42)
dataset = PairedNpyDataset(DATA, val_ids, None, False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = BaselineRestorer().to(device)
ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

lpips_fn = lpips.LPIPS(net="alex").to(device)

psnrs, ssims, lpips_scores, times = [], [], [], []

with torch.inference_mode():
    for x, gt, _ in loader:
        x = x.to(device)
        gt = gt.to(device)

        torch.cuda.synchronize()
        start = time.time()

        pred = model(x).clamp(0, 1)

        torch.cuda.synchronize()
        times.append(time.time() - start)

        g = gt[0, 0].cpu().numpy()
        p = pred[0, 0].cpu().numpy()

        psnrs.append(psnr(g, p))
        ssims.append(ssim(g, p))

        gt3 = gt.repeat(1, 3, 1, 1) * 2 - 1
        pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
        
        lpips_scores.append(lpips_fn(pred3, gt3).item())

print(f"Samples: {len(dataset)}")
print(f"PSNR : {np.mean(psnrs):.4f}")
print(f"SSIM : {np.mean(ssims):.4f}")
print(f"LPIPS: {np.mean(lpips_scores):.4f}")
print(f"Time/image: {np.mean(times)*1000:.2f} ms")