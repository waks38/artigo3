"""Core MIL data abstractions.

A *bag* is a variable-length set of instances with a single bag-level label.
The standard MIL assumption: a bag is positive iff at least one instance is
positive (instance labels are usually unknown). All dataset loaders in this
package yield ``Bag`` objects so models are dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class Bag:
    """A single MIL bag.

    Attributes:
        instances: float tensor ``(n_instances, feat_dim)`` (or raw inputs an
            encoder will embed, e.g. ``(n_instances, C, H, W)`` for image bags).
        label: scalar bag-level label.
        instance_labels: optional ``(n_instances,)`` ground-truth instance
            labels — only available for synthetic data; used to evaluate
            interpretability (does the Hopfield attention land on key instances?).
    """

    instances: torch.Tensor
    label: torch.Tensor
    instance_labels: torch.Tensor | None = None


class MILDataset(Dataset):
    """Base class: subclasses populate ``self.bags: list[Bag]``."""

    bags: list[Bag]

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> Bag:
        return self.bags[idx]


def mil_collate(batch: list[Bag]) -> dict:
    """Collate variable-length bags.

    Bags are NOT padded into a dense batch by default. Most aggregators here
    (mean/max/attention/Hopfield) operate per-bag, so we keep a list of
    instance tensors plus a stacked label vector. Switch to a padded+masked
    representation only if a model needs true batched matmuls.
    """
    return {
        "instances": [b.instances for b in batch],
        "labels": torch.stack([torch.as_tensor(b.label) for b in batch]),
        "instance_labels": [b.instance_labels for b in batch],
    }
