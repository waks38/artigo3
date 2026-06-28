"""CIFAR patch-bags — image MIL with VISIBLE instances and known witnesses.

Unlike Elephant/Fox/Tiger (feature vectors only), here each instance IS a real
RGB image you can look at. A bag is a set of CIFAR-10 images; the bag is
positive iff it contains at least one image of the target class. Because we
assemble the bags, we know exactly which instances are the witnesses
(``instance_labels``), so the instance-localization metric applies.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from hopmil.data.mil_dataset import Bag, MILDataset

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# fast.ai mirror (PNG image folders) — used as a fallback when the canonical
# torchvision source (cs.toronto.edu) is unreachable.
_FASTAI_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"


def _load_cifar10(root: str, train: bool, download: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images uint8 (N,32,32,3), labels (N,)) from any available source."""
    cache = Path(root) / f"cifar10_{'train' if train else 'test'}.pt"
    if cache.exists():
        d = torch.load(cache, weights_only=True)
        return d["x"], d["y"]
    try:
        ds = datasets.CIFAR10(root=root, train=train, download=download)
        x, y = torch.from_numpy(ds.data), torch.tensor(ds.targets)
    except Exception:
        x, y = _load_cifar10_fastai(root, train)
    torch.save({"x": x, "y": y}, cache)
    return x, y


def _load_cifar10_fastai(root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from PIL import Image

    root = Path(root)
    base = root / "cifar10"
    if not base.exists():
        blob = urllib.request.urlopen(
            urllib.request.Request(_FASTAI_URL, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=300,
        ).read()
        tarfile.open(fileobj=io.BytesIO(blob)).extractall(root)
    split = base / ("train" if train else "test")
    classes = sorted(p.name for p in split.iterdir() if p.is_dir())  # alphabetical == CIFAR order
    imgs, lbls = [], []
    for ci, c in enumerate(classes):
        for f in sorted((split / c).glob("*.png")):
            imgs.append(np.asarray(Image.open(f).convert("RGB")))
            lbls.append(ci)
    return torch.from_numpy(np.stack(imgs)), torch.tensor(lbls)

# CIFAR-10 per-channel normalization
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


class CIFARBags(MILDataset):
    def __init__(
        self,
        root: str = "data/raw",
        train: bool = True,
        target_class: int = 0,          # default: airplane
        mean_bag_size: float = 10.0,
        var_bag_size: float = 2.0,
        num_bags: int = 250,
        seed: int = 0,
        download: bool = True,
    ) -> None:
        self.target_class = target_class
        self.target_name = CIFAR10_CLASSES[target_class]
        data, targets = _load_cifar10(root, train, download)            # (N,32,32,3) uint8
        images = data.permute(0, 3, 1, 2).float().div(255.0)            # (N,3,32,32)
        images = (images - _MEAN) / _STD

        g = torch.Generator().manual_seed(seed)
        sizes = torch.normal(mean_bag_size, var_bag_size, (num_bags,), generator=g)
        sizes = sizes.round().clamp(min=1).int()

        self.bags = []
        for n in sizes.tolist():
            idx = torch.randint(0, len(targets), (n,), generator=g)
            instances = images[idx]                                  # (n,3,32,32)
            instance_labels = (targets[idx] == target_class).long()  # (n,)
            label = instance_labels.any().long()
            self.bags.append(
                Bag(instances=instances, label=label, instance_labels=instance_labels)
            )

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        """Undo normalization for display: (..,3,32,32) -> [0,1] image."""
        return (x * _STD + _MEAN).clamp(0, 1)
