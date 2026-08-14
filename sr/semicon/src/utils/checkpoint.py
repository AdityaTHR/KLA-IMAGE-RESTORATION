import os
import tempfile

import torch


def save_checkpoint(state: dict, filepath: str) -> None:
    """Atomically save exactly the requested checkpoint path."""
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix='.checkpoint-', suffix='.tmp', dir=directory
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_checkpoint(filepath: str, model, optimizer=None, device='cpu'):
    """Load model and optional optimizer state from a checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
