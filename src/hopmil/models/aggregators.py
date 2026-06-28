"""Instance aggregators — the swappable core of every experiment.

All aggregators share one contract: map a single bag's instance embeddings
``(n_instances, dim)`` to a fixed bag embedding ``(dim,)`` plus optional
per-instance attention weights ``(n_instances,)`` for interpretability.

The scientific question of this article lives here: replace the learned
attention pooling of Ilse et al. (2018) with a Modern Hopfield layer
(Ramsauer et al., 2020), which generalizes attention via an iterative,
energy-based associative retrieval, and measure the trade-offs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module, ABC):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (n_instances, dim) -> (bag_embedding (dim,), weights (n,) | None)."""


class MeanPool(Aggregator):
    def forward(self, x):
        return x.mean(dim=0), None


class MaxPool(Aggregator):
    def forward(self, x):
        return x.max(dim=0).values, None


class AttentionMIL(Aggregator):
    """Gated attention pooling (Ilse et al., 2018) — the baseline to beat."""

    def __init__(self, dim: int, hidden: int = 128, gated: bool = True) -> None:
        super().__init__(dim)
        self.gated = gated
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden) if gated else None
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):
        a = torch.tanh(self.V(x))
        if self.gated:
            a = a * torch.sigmoid(self.U(x))
        scores = self.w(a).squeeze(-1)  # (n,)
        weights = F.softmax(scores, dim=0)
        return weights @ x, weights


class HopfieldMIL(Aggregator):
    """Modern Hopfield pooling aggregator.

    Wraps ``hflayers.HopfieldPooling`` (ml-jku/hopfield-layers): a learnable
    state-pattern query retrieves a weighted combination of the bag instances
    through (iterated) softmax attention with inverse-temperature ``beta``.
    The retrieved pattern is the bag embedding; the final attention head's
    weights are exposed for interpretability comparisons against AttentionMIL.
    """

    def __init__(self, dim: int, beta: float | None = None, num_heads: int = 1) -> None:
        super().__init__(dim)
        try:
            from hflayers import HopfieldPooling  # lazy; optional heavy git dep
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HopfieldMIL needs the 'hopfield' extra: `uv sync --extra hopfield`"
            ) from e

        self.pool = HopfieldPooling(
            input_size=dim,
            output_size=dim,
            num_heads=num_heads,
            scaling=beta,  # None -> 1/sqrt(d), the standard attention temperature
        )

    def forward(self, x):
        # HopfieldPooling expects (batch, seq, dim); a bag is one sequence.
        out = self.pool(x.unsqueeze(0)).squeeze(0)  # (dim,)
        weights = self.pool.get_association_matrix(x.unsqueeze(0))
        return out, weights.detach().mean(dim=(0, 1)).squeeze() if weights is not None else None


def build_aggregator(name: str, dim: int, **kwargs) -> Aggregator:
    table = {
        "mean": MeanPool,
        "max": MaxPool,
        "attention": AttentionMIL,
        "hopfield": HopfieldMIL,
    }
    if name not in table:
        raise ValueError(f"unknown aggregator {name!r}; choose from {list(table)}")
    return table[name](dim=dim, **kwargs)
