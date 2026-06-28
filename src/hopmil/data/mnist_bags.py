"""Synthetic MNIST-bags benchmark (after Ilse et al., 2018).

A bag is a set of MNIST digits; positive iff it contains the target digit.
Fixed condition (see ``data/synthetic.py``): fixed bag size, exactly one witness
in positive bags, balanced classes. ``instance_labels`` records which digits are
the target, for the interpretability (localization) eval.
"""

from __future__ import annotations

from torchvision import datasets

from hopmil.data.mil_dataset import MILDataset
from hopmil.data.synthetic import make_bags


class MNISTBags(MILDataset):
    def __init__(
        self,
        root: str = "data/raw",
        train: bool = True,
        target_digit: int = 9,
        bag_mode: str = "ilse",
        bag_size: int = 15,
        num_witnesses: int = 1,
        mean_bag_size: float = 10.0,
        var_bag_size: float = 2.0,
        positive_ratio: float = 0.5,
        num_bags: int = 250,
        seed: int = 0,
        download: bool = True,
    ) -> None:
        self.target_digit = target_digit
        mnist = datasets.MNIST(root=root, train=train, download=download)
        images = mnist.data.float().div(255.0)  # (N, 28, 28)
        images = (images - 0.1307) / 0.3081
        instances = images.unsqueeze(1)  # (N, 1, 28, 28)

        self.bags = make_bags(
            instances,
            mnist.targets,
            target_digit,
            bag_mode=bag_mode,
            num_bags=num_bags,
            bag_size=bag_size,
            num_witnesses=num_witnesses,
            mean_bag_size=mean_bag_size,
            var_bag_size=var_bag_size,
            positive_ratio=positive_ratio,
            seed=seed,
        )
