"""Reproducibility helpers. Always seed before building data/model for a run."""

from __future__ import annotations


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import lightning as L
    import torch

    L.seed_everything(seed, workers=True)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
