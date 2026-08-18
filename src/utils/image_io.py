import numpy as np
import torch
import os


def load_npy_image(path: str) -> np.ndarray:
    """Load a float32 HxW array without changing its value range."""
    img = np.load(path)
    assert img.ndim == 2, f"Expected 2D array, got {img.ndim}D"
    assert img.dtype == np.float32, f"Expected float32, got {img.dtype}"
    return img

def save_npy_image(image: np.ndarray, path: str) -> None:
    """Save image as .npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.save(path, image.astype(np.float32))

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a single-image tensor to an HxW float32 array."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    return tensor.detach().cpu().numpy().astype(np.float32)

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert (H,W) numpy array to (1,H,W) tensor."""
    assert array.ndim == 2
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
