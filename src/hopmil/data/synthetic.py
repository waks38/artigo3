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


def make_bags(
    instances: torch.Tensor,
    labels: torch.Tensor,
    target: int,
    *,
    bag_mode: str = "ilse",
    num_bags: int,
    bag_size: int = 15,
    num_witnesses: int = 1,
    mean_bag_size: float = 10.0,
    var_bag_size: float = 2.0,
    positive_ratio: float = 0.5,
    seed: int = 0,
) -> list[Bag]:
    """Dispatch to the selected synthetic bag construction.

    - ``"ilse"`` (default): variable-size, multi-witness bags (Ilse et al. 2018 /
      ml-jku demo) via :func:`make_ilse_bags`.
    - ``"witness"``: fixed-size bags with exactly ``num_witnesses`` witnesses via
      :func:`make_witness_bags` (the original tightly-controlled hard condition).
    """
    if bag_mode == "ilse":
        return make_ilse_bags(
            instances, labels, target,
            num_bags=num_bags, mean_bag_size=mean_bag_size, var_bag_size=var_bag_size,
            positive_ratio=positive_ratio, seed=seed,
        )
    if bag_mode == "witness":
        return make_witness_bags(
            instances, labels, target,
            num_bags=num_bags, bag_size=bag_size, num_witnesses=num_witnesses,
            positive_ratio=positive_ratio, seed=seed,
        )
    raise ValueError(f"unknown bag_mode {bag_mode!r}; choose 'ilse' or 'witness'")


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


def make_ilse_bags(
    instances: torch.Tensor,
    labels: torch.Tensor,
    target: int,
    *,
    num_bags: int,
    mean_bag_size: float = 10.0,
    var_bag_size: float = 2.0,
    positive_ratio: float = 0.5,
    seed: int = 0,
) -> list[Bag]:
    """Assemble MIL bags in the style of Ilse et al. (2018) / the ml-jku demo.

    Unlike :func:`make_witness_bags` (one fixed witness in fixed-size bags), here:

    - **bag size is variable**, drawn from ``round(Normal(mean_bag_size,
      var_bag_size))`` and clamped to ``>= 1`` (``var_bag_size`` is the *std*,
      matching Ilse's ``var_bag_length``);
    - a bag is **positive iff it contains >= 1 target instance**, and a positive
      bag holds a *natural* (>= 1) number of witnesses rather than exactly one —
      so the witness signal is usually redundant, a much easier MIL task.

    Classes are kept **balanced** via ``positive_ratio`` (negatives are sampled
    purely from non-target instances; positives are sampled from the full pool and
    nudged to contain at least one witness). ``instance_labels`` marks the targets.
    """
    g = torch.Generator().manual_seed(seed)
    tgt_pool = (labels == target).nonzero(as_tuple=True)[0]
    non_pool = (labels != target).nonzero(as_tuple=True)[0]
    full_pool = torch.arange(len(labels))
    if len(tgt_pool) == 0 or len(non_pool) == 0:
        raise ValueError(f"target {target!r} leaves an empty target/non-target pool")

    is_target = torch.zeros(len(labels), dtype=torch.bool)
    is_target[tgt_pool] = True

    n_pos = round(num_bags * positive_ratio)
    flags = torch.zeros(num_bags, dtype=torch.bool)
    flags[:n_pos] = True
    flags = flags[torch.randperm(num_bags, generator=g)]

    def _take(pool: torch.Tensor, k: int) -> torch.Tensor:
        return pool[torch.randint(len(pool), (k,), generator=g)]

    def _bag_size() -> int:
        s = torch.normal(float(mean_bag_size), float(var_bag_size), (1,), generator=g).item()
        return max(1, int(round(s)))

    bags: list[Bag] = []
    for is_pos in flags.tolist():
        bag_size = _bag_size()
        if is_pos:
            idx = _take(full_pool, bag_size)
            if not is_target[idx].any():  # guarantee at least one witness
                slot = torch.randint(bag_size, (1,), generator=g).item()
                idx[slot] = _take(tgt_pool, 1)[0]
            inst_labels = is_target[idx].long()
        else:
            idx = _take(non_pool, bag_size)
            inst_labels = torch.zeros(bag_size, dtype=torch.long)
        bags.append(
            Bag(
                instances=instances[idx],
                label=torch.tensor(int(is_pos)),
                instance_labels=inst_labels,
            )
        )
    return bags
