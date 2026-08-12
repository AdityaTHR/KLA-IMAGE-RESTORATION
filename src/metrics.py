import numpy as np
from skimage.metrics import structural_similarity


def psnr(gt, pred, data_range=1.0):
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(gt, pred, data_range=1.0):
    return float(structural_similarity(gt, pred, data_range=data_range))
