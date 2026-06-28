"""Synthetic MNIST-bags benchmark (Ilse et al., 2018).

A bag is a set of MNIST digits; the bag is positive iff it contains at least
one target digit (default: '9'). Useful for sanity checks and for evaluating
whether attention/Hopfield weights concentrate on the witness instances
(``instance_labels`` records which digits are the target).
"""

from __future__ import annotations

import torch
from torchvision import datasets, transforms

from hopmil.data.mil_dataset import Bag, MILDataset

_NORMALIZE = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class MNISTBags(MILDataset):
    def __init__(
        self,
        root: str = "data/raw",
        train: bool = True,
        target_digit: int = 9,
        mean_bag_size: float = 10.0,
        var_bag_size: float = 2.0,
        num_bags: int = 250,
        seed: int = 0,
        download: bool = True,
    ) -> None:
        self.target_digit = target_digit
        mnist = datasets.MNIST(root=root, train=train, download=download)
        images = mnist.data.float().div(255.0)          # (N, 28, 28)
        images = (images - 0.1307) / 0.3081
        digits = mnist.targets                            # (N,)

        g = torch.Generator().manual_seed(seed)
        sizes = torch.normal(mean_bag_size, var_bag_size, (num_bags,), generator=g)
        sizes = sizes.round().clamp(min=1).int()

        self.bags = []
        for n in sizes.tolist():
            idx = torch.randint(0, len(digits), (n,), generator=g)
            instances = images[idx].unsqueeze(1)          # (n, 1, 28, 28)
            inst_digits = digits[idx]
            instance_labels = (inst_digits == target_digit).long()  # (n,)
            label = instance_labels.any().long()          # MIL assumption
            self.bags.append(
                Bag(instances=instances, label=label, instance_labels=instance_labels)
            )
