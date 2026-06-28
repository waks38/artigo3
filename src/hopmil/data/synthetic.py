"""Controlled assembly of synthetic image bags (MNIST / Fashion-MNIST / CIFAR).

The three synthetic loaders differ only in how they load + normalize the base
images; the bag-construction logic is identical and lives here.

Design (fixed condition, no OFAT sweeps):
- **Fixed bag size** (default 15 instances).
- **Exactly ``num_witnesses`` target instances in a positive bag** (default 1);
  negative bags have none. This makes the witness genuinely sparse (1/15) — the
  regime where attention/Hopfield should beat mean/max — and is fully controlled
  rather than left to chance.
- **Balanced** by ``positive_ratio`` (default 0.5).

Everything is seeded, so the same ``seed`` reproduces the same bags. Sampling of
base instances is with replacement (the base datasets are large relative to the
bag count, and this keeps construction simple and seed-stable).
"""

from __future__ import annotations

import torch

from hopmil.data.mil_dataset import Bag


def make_witness_bags(
    instances: torch.Tensor,
    labels: torch.Tensor,
    target: int,
    *,
    num_bags: int,
    bag_size: int = 15,
    num_witnesses: int = 1,
    positive_ratio: float = 0.5,
    seed: int = 0,
) -> list[Bag]:
    """Assemble MIL bags with a controlled number of witnesses.

    Args:
        instances: base instances already in per-instance shape, e.g.
            ``(N, 1, 28, 28)`` for MNIST or ``(N, 3, 32, 32)`` for CIFAR.
        labels: ``(N,)`` class label per base instance.
        target: the positive/target class.
        num_bags: how many bags to build.
        bag_size: fixed number of instances per bag.
        num_witnesses: number of target instances placed in each positive bag.
        positive_ratio: fraction of bags that are positive.
        seed: RNG seed (reproducible bags).
    """
    if not 0 < num_witnesses <= bag_size:
        raise ValueError(f"num_witnesses must be in [1, bag_size]; got {num_witnesses}")

    g = torch.Generator().manual_seed(seed)
    tgt_pool = (labels == target).nonzero(as_tuple=True)[0]
    non_pool = (labels != target).nonzero(as_tuple=True)[0]
    if len(tgt_pool) == 0 or len(non_pool) == 0:
        raise ValueError(f"target {target!r} leaves an empty target/non-target pool")

    n_pos = round(num_bags * positive_ratio)
    flags = torch.zeros(num_bags, dtype=torch.bool)
    flags[:n_pos] = True
    flags = flags[torch.randperm(num_bags, generator=g)]  # shuffle pos/neg order

    def _take(pool: torch.Tensor, k: int) -> torch.Tensor:
        return pool[torch.randint(len(pool), (k,), generator=g)]

    bags: list[Bag] = []
    for is_pos in flags.tolist():
        k = num_witnesses if is_pos else 0
        idx = _take(non_pool, bag_size - k)
        inst_labels = torch.zeros(bag_size, dtype=torch.long)
        if k > 0:
            idx = torch.cat([idx, _take(tgt_pool, k)])
            inst_labels[bag_size - k :] = 1
        order = torch.randperm(bag_size, generator=g)  # don't leave witnesses last
        idx, inst_labels = idx[order], inst_labels[order]
        bags.append(
            Bag(
                instances=instances[idx],
                label=torch.tensor(int(is_pos)),
                instance_labels=inst_labels,
            )
        )
    return bags
