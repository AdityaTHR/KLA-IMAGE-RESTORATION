#!/usr/bin/env python3
from pathlib import Path
import argparse, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import PairedNpyDataset, make_split
from src.nafnet import NAFRestorer


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def batch_psnr(gt, pred):
    mse = F.mse_loss(pred, gt, reduction="none").flatten(1).mean(1).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/nafnet_best.pt")
    args = ap.parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ids, val_ids = make_split(args.data_root, 0.10, args.seed)
    tr = PairedNpyDataset(args.data_root, train_ids, args.patch, augment=True)
    va = PairedNpyDataset(args.data_root, val_ids, None, augment=False)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=(device.type=="cuda"), persistent_workers=args.workers>0)
    val_loader = DataLoader(va, batch_size=1, shuffle=False, num_workers=max(0,args.workers//2))

    model = NAFRestorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    best = -1.0
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train(); losses=[]; t0=time.time()
        for x,y,_ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                pred=model(x)
                loss=F.l1_loss(pred,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())

        model.eval(); ps=[]
        with torch.inference_mode():
            for x,y,_ in val_loader:
                x,y=x.to(device),y.to(device)
                pred=model(x).clamp(0,1)
                ps.append(batch_psnr(y,pred))
        val_psnr=float(np.mean(ps))
        print(f"epoch={epoch:03d} L1={np.mean(losses):.5f} val_PSNR={val_psnr:.3f} time={time.time()-t0:.1f}s")
        if val_psnr > best:
            best = val_psnr
            torch.save({"model":model.state_dict(),"val_psnr":best,"args":vars(args)},out)
            print(f"  saved best -> {out}")

    print(f"Best validation PSNR: {best:.3f} dB")


if __name__ == "__main__":
    main()
