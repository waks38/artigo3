"""Instance encoders φ: map a raw instance to an embedding of size ``dim``.

The encoder is held FIXED across the four aggregators so the aggregator is the
only thing that varies in the comparison. One encoder per modality:

- :class:`MLPEncoder` for tabular bags (MUSK, Elephant/Fox/Tiger features).
- :class:`CNNEncoder` for image bags (MNIST-bags), LeNet-style as in Ilse et al.

Every encoder maps ``(n_instances, ...)`` -> ``(n_instances, dim)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, dim: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (n, in_dim) -> (n, dim)
        return self.net(x)


class CNNEncoder(nn.Module):
    """LeNet-style encoder for small image instances.

    Defaults to 1x28x28 (MNIST-bags); set ``in_channels=3, image_size=32`` for
    CIFAR patch-bags. The flattened size is computed from a dummy pass so the
    same class works for any channel count / input size.
    """

    def __init__(self, dim: int = 128, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.dim = dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, image_size, image_size)).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (n, C, H, W) -> (n, dim)
        return self.fc(self.features(x))
