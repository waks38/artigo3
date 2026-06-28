"""End-to-end MIL classifier: instance encoder -> aggregator -> bag classifier.

Keeping the encoder, aggregator, and head as separate components is what makes
the ablation clean: only the aggregator changes between baselines and the
Hopfield variant; encoder and head stay fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hopmil.models.aggregators import Aggregator


class MILClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        aggregator: Aggregator,
        num_classes: int = 1,
        dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.head = nn.Linear(dim or aggregator.dim, num_classes)

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """bags: list of (n_instances, in_dim). Returns (logits (B, C), attn weights)."""
        logits, attns = [], []
        for x in bags:
            h = self.encoder(x)                 # (n, dim)
            z, w = self.aggregator(h)           # (dim,), (n,) | None
            logits.append(self.head(z))
            attns.append(w)
        return torch.stack(logits), attns
