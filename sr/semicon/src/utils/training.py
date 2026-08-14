"""Configuration-driven optimizer and scheduler factories."""

import torch


def build_optimizer(parameters, config: dict) -> torch.optim.Optimizer:
    name = str(config.get('name', 'adam')).lower()
    common = {
        'lr': config['lr'],
        'weight_decay': config.get('weight_decay', 0.0),
        'betas': tuple(config.get('betas', [0.9, 0.999])),
    }
    if name == 'adam':
        return torch.optim.Adam(parameters, **common)
    if name == 'adamw':
        return torch.optim.AdamW(parameters, **common)
    raise ValueError("Unsupported optimizer. Available: adam, adamw")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    name = str(config.get('name', 'cosine')).lower()
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['T_max'],
            eta_min=config.get('eta_min', 0.0),
        )
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config.get('gamma', 0.1),
        )
    raise ValueError("Unsupported scheduler. Available: cosine, step")
