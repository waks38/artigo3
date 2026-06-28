"""Synthetic Fashion-MNIST bags — same construction as MNIST-bags, harder texture.

A bag is a set of Fashion-MNIST images; positive iff it contains the target
class. Fixed condition (see ``data/synthetic.py``): fixed bag size, exactly one
witness in positive bags, balanced classes. ``instance_labels`` mark the
witnesses, so the localization metric applies.
"""

from __future__ import annotations

from torchvision import datasets

from hopmil.data.mil_dataset import MILDataset
from hopmil.data.synthetic import make_witness_bags

FASHION_CLASSES = [
    "tshirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]

# Fashion-MNIST per-channel normalization.
_MEAN, _STD = 0.2860, 0.3530


class FashionMNISTBags(MILDataset):
    def __init__(
        self,
        root: str = "data/raw",
        train: bool = True,
        target_class: int = 0,
        bag_size: int = 15,
        num_witnesses: int = 1,
        positive_ratio: float = 0.5,
        num_bags: int = 250,
        seed: int = 0,
        download: bool = True,
    ) -> None:
        self.target_class = target_class
        self.target_name = FASHION_CLASSES[target_class]
        ds = datasets.FashionMNIST(root=root, train=train, download=download)
        images = ds.data.float().div(255.0)  # (N, 28, 28)
        images = (images - _MEAN) / _STD
        instances = images.unsqueeze(1)  # (N, 1, 28, 28)

        self.bags = make_witness_bags(
            instances,
            ds.targets,
            target_class,
            num_bags=num_bags,
            bag_size=bag_size,
            num_witnesses=num_witnesses,
            positive_ratio=positive_ratio,
            seed=seed,
        )
